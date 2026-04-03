"""
encode_tinyclip.py

Ce script encode toutes les frames extraites du dataset UCF-Crime avec le modèle TinyCLIP de Microsoft.

On sauvegarde trois fichiers à la fin :
    tinyclip_embeddings.npy     les vecteurs de toutes les frames
    tinyclip_metadata.csv       les infos associées à chaque frame (classe, vidéo, timestamp)
    tinyclip_metrics.json       les performances du modèle (temps, vitesse...)

Le script supporte la reprise après interruption grâce à un système de checkpoint.
Un checkpoint est sauvegardé toutes les 10 minutes dans results/tinyclip/tinyclip_checkpoint.npz.
Si le script est relancé et qu'un checkpoint existe, il reprend depuis où il s'était arrêté.

"""

import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import logging

# Affiche des logs avec timestamp pour suivre l'avancement et les performances
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# Chemins d'entrée et de sortie
FRAMES_DIR = Path("frames")
OUTPUT_DIR = Path("results/tinyclip")

# Variante TinyCLIP choisie
MODEL_NAME = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"

BATCH_SIZE = 32

# Intervalle de sauvegarde du checkpoint en secondes (10 minutes)
CHECKPOINT_INTERVAL = 10 * 60

# Classe qui permet de charger les frames depuis le disque de façon organisée pour PyTorch
class FrameDataset(Dataset):

    def __init__(self, frame_paths):
        self.frame_paths = frame_paths

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            return image, str(path), True
        except Exception:
            # Si une image est corrompue on retourne None
            return None, str(path), False

# Fonction qui crée une fonction de collate personnalisée pour le DataLoader afin de gérer les images corrompues
def make_collate_fn(processor):

    def collate_fn(batch):
        images, paths, valids = zip(*batch)

        valid_images  = [img for img, v in zip(images, valids) if v]
        valid_paths   = [p   for p,   v in zip(paths,  valids) if v]
        invalid_paths = [p   for p,   v in zip(paths,  valids) if not v]

        if valid_images:
            # Le processor retourne un dictionnaire de tensors directement utilisables par le modèle
            inputs = processor(images=valid_images, return_tensors="pt", padding=True)
        else:
            inputs = None

        return inputs, valid_paths, invalid_paths

    return collate_fn


def save_checkpoint(output_dir, all_embeddings, all_metadata, errors, nb_frames_processed):
    """Sauvegarde un checkpoint pour pouvoir reprendre plus tard."""
    checkpoint_path = output_dir / 'tinyclip_checkpoint.npz'
    tmp_path = output_dir / 'tinyclip_checkpoint_tmp.npz'

    if all_embeddings:
        emb_matrix = np.concatenate(all_embeddings, axis=0)
    else:
        emb_matrix = np.empty((0, 512), dtype=np.float32)

    np.savez(
        tmp_path,
        embeddings=emb_matrix,
        nb_frames_processed=np.array(nb_frames_processed),
        errors=np.array(errors)
    )
    meta_tmp = output_dir / 'tinyclip_checkpoint_meta.csv'
    pd.DataFrame(all_metadata).to_csv(meta_tmp, index=True)

    tmp_path.replace(checkpoint_path)
    logger.info(f"Checkpoint sauvegardé : {len(all_metadata)} frames traitées")


def load_checkpoint(output_dir):
    """Charge un checkpoint existant. Retourne None s'il n'y en a pas."""
    checkpoint_path = output_dir / 'tinyclip_checkpoint.npz'
    meta_path = output_dir / 'tinyclip_checkpoint_meta.csv'

    if not checkpoint_path.exists() or not meta_path.exists():
        return None

    data = np.load(checkpoint_path)
    embeddings = data['embeddings']
    nb_frames_processed = int(data['nb_frames_processed'])
    errors = int(data['errors'])

    meta_df = pd.read_csv(meta_path, index_col=0)
    metadata = meta_df.to_dict('records')

    return {
        'embeddings': embeddings,
        'metadata': metadata,
        'nb_frames_processed': nb_frames_processed,
        'errors': errors
    }


def cleanup_checkpoint(output_dir):
    """Supprime les fichiers de checkpoint une fois l'encodage terminé."""
    for name in ['tinyclip_checkpoint.npz', 'tinyclip_checkpoint_meta.csv', 'tinyclip_checkpoint_tmp.npz']:
        p = output_dir / name
        if p.exists():
            p.unlink()
    logger.info("Fichiers de checkpoint nettoyés")


# Fonction principale qui encode les frames et sauvegarde les résultats
def encode_frames(frames_dir, output_dir, model_name, batch_size):

    output_dir.mkdir(parents=True, exist_ok=True)

    # On vérifie si un GPU est disponible, sinon on tourne sur CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Appareil utilisé : {device}")
    if device == 'cuda':
        logger.info(f"GPU détecté : {torch.cuda.get_device_name(0)}")

    # HuggingFace télécharge les poids automatiquement au premier lancement et les met en cache dans ~/.cache/huggingface/
    logger.info(f"Chargement du modèle TinyCLIP ({model_name})...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model     = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()  # on passe en mode évaluation pour désactiver le dropout et autres mécanismes d'entraînement
    logger.info("Modèle TinyCLIP chargé")

    # On récupère la liste de toutes les frames du dataset
    logger.info(f"Recherche des frames dans {frames_dir}...")
    frame_paths = sorted(frames_dir.rglob('*_t*.jpg'))

    if not frame_paths:
        raise FileNotFoundError(f"Aucune frame trouvée dans {frames_dir}")

    logger.info(f"{len(frame_paths):,} frames trouvées")

    # Vérification d'un checkpoint existant pour reprendre l'encodage
    resume_checkpoint = load_checkpoint(output_dir)
    skip_frames = 0

    all_embeddings = []
    all_metadata   = []
    errors         = 0

    if resume_checkpoint is not None:
        skip_frames = resume_checkpoint['nb_frames_processed']
        all_embeddings.append(resume_checkpoint['embeddings'])
        all_metadata = resume_checkpoint['metadata']
        errors = resume_checkpoint['errors']
        logger.info(f"Checkpoint trouvé : {skip_frames:,} frames déjà traitées, reprise en cours...")

        frame_paths = frame_paths[skip_frames:]
        logger.info(f"{len(frame_paths):,} frames restantes à encoder")

        if len(frame_paths) == 0:
            logger.info("Toutes les frames ont déjà été encodées !")
            embeddings_matrix = resume_checkpoint['embeddings']
            encoding_time = 0
            nb_encoded = len(embeddings_matrix)
            _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                                encoding_time, nb_encoded, model_name, batch_size, device)
            cleanup_checkpoint(output_dir)
            return
    else:
        logger.info("Aucun checkpoint trouvé, démarrage depuis le début")

    # On utilise num_workers=0 ici car les objets PIL ne sont pas sérialisables sur Windows avec plusieurs workers
    dataset    = FrameDataset(frame_paths)
    collate_fn = make_collate_fn(processor)
    loader     = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Compteur de frames traitées dans cette session
    frames_this_session = 0

    logger.info("Démarrage de l'encodage...")
    start_time = time.time()
    last_checkpoint_time = start_time

    with torch.no_grad():
        for inputs, valid_paths, invalid_paths in tqdm(loader, desc="TinyCLIP"):

            errors += len(invalid_paths)
            frames_this_session += len(valid_paths) + len(invalid_paths)

            if inputs is None or len(valid_paths) == 0:
                continue

            pixel_values = inputs['pixel_values'].to(device)

            # get_image_features est l'équivalent HuggingFace de encode_image utilisé par CLIP et MobileCLIP
            embeddings = model.get_image_features(pixel_values=pixel_values)
            if hasattr(embeddings, 'image_embeds'):
                embeddings = embeddings.image_embeds
            elif hasattr(embeddings, 'pooler_output'):
                embeddings = embeddings.pooler_output
            else:
                embeddings = embeddings.last_hidden_state[:, 0, :]

            # On normalise les vecteurs pour que leur norme soit égale à 1
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().float().numpy())

            # On extrait les métadonnées depuis le nom de fichier pour pouvoir retrouver à quelle vidéo et à quel moment appartient chaque frame
            for path_str in valid_paths:
                p          = Path(path_str)
                class_name = p.parts[-3]
                video_name = p.parts[-2]
                filename   = p.name
                timestamp  = int(filename.rsplit('_t', 1)[1].replace('s.jpg', ''))

                all_metadata.append({
                    'filepath'  : path_str,
                    'class'     : class_name,
                    'video'     : video_name,
                    'timestamp' : timestamp,
                    'filename'  : filename
                })

            # Sauvegarde périodique du checkpoint toutes les CHECKPOINT_INTERVAL secondes
            now = time.time()
            if now - last_checkpoint_time >= CHECKPOINT_INTERVAL:
                total_processed = skip_frames + frames_this_session
                save_checkpoint(output_dir, all_embeddings, all_metadata, errors, total_processed)
                last_checkpoint_time = now

    encoding_time = time.time() - start_time
    nb_encoded    = sum(len(b) for b in all_embeddings)

    logger.info(f"Encodage terminé en {encoding_time:.1f}s ({encoding_time/60:.1f} min)")
    logger.info(f"{nb_encoded:,} frames encodées, {errors} erreurs")
    if encoding_time > 0:
        logger.info(f"Vitesse moyenne : {nb_encoded/encoding_time:.1f} frames/sec")

    # On concatène tous les batchs en une seule matrice
    embeddings_matrix = np.concatenate(all_embeddings, axis=0)

    _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                        encoding_time, nb_encoded, model_name, batch_size, device)

    cleanup_checkpoint(output_dir)


def _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                        encoding_time, nb_encoded, model_name, batch_size, device):
    """Sauvegarde les fichiers finaux (embeddings, métadonnées, métriques)."""

    embeddings_path = output_dir / 'tinyclip_embeddings.npy'
    np.save(embeddings_path, embeddings_matrix)
    logger.info(f"Embeddings sauvegardés dans {embeddings_path}, shape : {embeddings_matrix.shape}")

    metadata_path = output_dir / 'tinyclip_metadata.csv'
    pd.DataFrame(all_metadata).to_csv(metadata_path, index=True)
    logger.info(f"Métadonnées sauvegardées dans {metadata_path}")

    metrics = {
        'model'             : f'TinyCLIP ({model_name.split("/")[1]})',
        'nb_frames_encoded' : int(nb_encoded),
        'nb_errors'         : int(errors),
        'encoding_time_sec' : round(encoding_time, 2),
        'encoding_time_min' : round(encoding_time / 60, 2),
        'frames_per_second' : round(nb_encoded / max(encoding_time, 0.01), 2),
        'embedding_dim'     : int(embeddings_matrix.shape[1]),
        'batch_size'        : batch_size,
        'device'            : device
    }

    metrics_path = output_dir / 'tinyclip_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métriques sauvegardées dans {metrics_path}")


if __name__ == "__main__":
    encode_frames(
        frames_dir = FRAMES_DIR,
        output_dir = OUTPUT_DIR,
        model_name = MODEL_NAME,
        batch_size = BATCH_SIZE
    )
