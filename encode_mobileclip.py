"""

Ce script encode toutes les frames extraites du dataset UCF-Crime avec le modèle MobileCLIP-S2 d'Apple.

On sauvegarde trois fichiers à la fin :
    mobileclip_s2_embeddings.npy   les vecteurs de toutes les frames
    mobileclip_s2_metadata.csv     les infos associées à chaque frame (classe, vidéo, timestamp)
    mobileclip_s2_metrics.json     les performances du modèle (temps, vitesse...)

Le script supporte la reprise après interruption grâce à un système de checkpoint.
Un checkpoint est sauvegardé toutes les 10 minutes dans results/mobileclip/mobileclip_checkpoint.npz.
Si le script est relancé et qu'un checkpoint existe, il reprend depuis où il s'était arrêté.

"""

import json
import time
import numpy as np
import pandas as pd
import torch
import mobileclip
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
FRAMES_DIR      = Path("frames")
OUTPUT_DIR      = Path("results/mobileclip")
CHECKPOINT_PATH = Path("checkpoints/mobileclip_s2.pt")
BATCH_SIZE      = 32

# Intervalle de sauvegarde du checkpoint en secondes (10 minutes)
CHECKPOINT_INTERVAL = 10 * 60

# Classe qui permet de charger les frames depuis le disque de façon organisée pour PyTorch

class FrameDataset(Dataset):

    def __init__(self, frame_paths, preprocess):
        self.frame_paths = frame_paths
        self.preprocess  = preprocess

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            # On applique le préprocesseur du modèle qui redimensionne et normalise l'image selon ce qu'attend MobileCLIP
            return self.preprocess(image), str(path), True
        except Exception:
            # Si une image est corrompue ou illisible on retourne un tensor vide plutôt que de faire planter tout le script
            dummy = torch.zeros(3, 256, 256)
            return dummy, str(path), False


# Fonction pour charger le modèle MobileCLIP-S2 avec les poids pré-entraînés
def load_model(checkpoint_path, device):

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fichier de poids introuvable : {checkpoint_path}")

    logger.info("Chargement du modèle MobileCLIP-S2...")
    model, _, preprocess = mobileclip.create_model_and_transforms(
        'mobileclip_s2',
        pretrained=str(checkpoint_path)
    )

    model = model.to(device)
    model.eval()  # on passe en mode évaluation pour désactiver le dropout et autres mécanismes d'entraînement

    logger.info(f"Modèle chargé sur {device}")
    return model, preprocess

def save_checkpoint(output_dir, all_embeddings, all_metadata, errors, nb_frames_processed):
    """Sauvegarde un checkpoint pour pouvoir reprendre plus tard."""
    checkpoint_path = output_dir / 'mobileclip_checkpoint.npz'
    tmp_path = output_dir / 'mobileclip_checkpoint_tmp.npz'

    # On concatène les embeddings accumulés jusqu'ici
    if all_embeddings:
        emb_matrix = np.concatenate(all_embeddings, axis=0)
    else:
        emb_matrix = np.empty((0, 512), dtype=np.float32)

    # On sauvegarde d'abord dans un fichier temporaire puis on renomme pour éviter la corruption si le script est coupé pendant l'écriture
    np.savez(
        tmp_path,
        embeddings=emb_matrix,
        nb_frames_processed=np.array(nb_frames_processed),
        errors=np.array(errors)
    )
    # Sauvegarde des métadonnées du checkpoint
    meta_tmp = output_dir / 'mobileclip_checkpoint_meta.csv'
    pd.DataFrame(all_metadata).to_csv(meta_tmp, index=True)

    # Renommage atomique du checkpoint
    tmp_path.replace(checkpoint_path)
    logger.info(f"Checkpoint sauvegardé : {len(all_metadata)} frames traitées")


def load_checkpoint(output_dir):
    """Charge un checkpoint existant. Retourne None s'il n'y en a pas."""
    checkpoint_path = output_dir / 'mobileclip_checkpoint.npz'
    meta_path = output_dir / 'mobileclip_checkpoint_meta.csv'

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
    for name in ['mobileclip_checkpoint.npz', 'mobileclip_checkpoint_meta.csv', 'mobileclip_checkpoint_tmp.npz']:
        p = output_dir / name
        if p.exists():
            p.unlink()
    logger.info("Fichiers de checkpoint nettoyés")


# Fonction principale qui encode les frames et sauvegarde les résultats
def encode_frames(frames_dir, output_dir, checkpoint_path, batch_size):

    output_dir.mkdir(parents=True, exist_ok=True)

    # On vérifie si un GPU est disponible, sinon on tourne sur CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Appareil utilisé : {device}")
    if device == 'cuda':
        logger.info(f"GPU détecté : {torch.cuda.get_device_name(0)}")

    model, preprocess = load_model(checkpoint_path, device)

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

        # On ne garde que les frames restantes
        frame_paths = frame_paths[skip_frames:]
        logger.info(f"{len(frame_paths):,} frames restantes à encoder")

        if len(frame_paths) == 0:
            logger.info("Toutes les frames ont déjà été encodées !")
            embeddings_matrix = resume_checkpoint['embeddings']
            encoding_time = 0
            nb_encoded = len(embeddings_matrix)
            _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                                encoding_time, nb_encoded, batch_size, device)
            cleanup_checkpoint(output_dir)
            return
    else:
        logger.info("Aucun checkpoint trouvé, démarrage depuis le début")

    # Le DataLoader va charger les images par batchs et en parallèle ce qui est plus rapide que de les charger une par une
    dataset = FrameDataset(frame_paths, preprocess)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )

    # Compteur de frames traitées dans cette session (pour le checkpoint on ajoute skip_frames)
    frames_this_session = 0

    logger.info("Démarrage de l'encodage...")
    # On démarre le chrono ici pour mesurer uniquement le temps d'encodage pur sans compter le chargement du modèle
    start_time = time.time()
    last_checkpoint_time = start_time

    with torch.no_grad():  # on désactive le calcul des gradients car on n'entraîne pas le modèle
        for images, paths, valid_flags in tqdm(loader, desc="MobileCLIP"):

            valid_mask   = valid_flags.bool()
            valid_images = images[valid_mask].to(device)
            valid_paths  = [p for p, v in zip(paths, valid_flags) if v]

            if len(valid_images) == 0:
                errors += len(paths)
                frames_this_session += len(paths)
                continue

            # On passe les images dans le modèle pour obtenir les embeddings
            embeddings = model.encode_image(valid_images)

            # On normalise les vecteurs pour que leur norme soit égale à 1
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

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

            errors += (~valid_mask).sum().item()
            frames_this_session += len(paths)

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
                        encoding_time, nb_encoded, batch_size, device)

    # On supprime le checkpoint car l'encodage est terminé avec succès
    cleanup_checkpoint(output_dir)


def _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                        encoding_time, nb_encoded, batch_size, device):
    """Sauvegarde les fichiers finaux (embeddings, métadonnées, métriques)."""

    # Sauvegarde des embeddings
    embeddings_path = output_dir / 'mobileclip_s2_embeddings.npy'
    np.save(embeddings_path, embeddings_matrix)
    logger.info(f"Embeddings sauvegardés dans {embeddings_path}, shape : {embeddings_matrix.shape}")

    # Sauvegarde des métadonnées, l'index de chaque ligne correspond à la position du vecteur dans la matrice d'embeddings
    metadata_path = output_dir / 'mobileclip_s2_metadata.csv'
    pd.DataFrame(all_metadata).to_csv(metadata_path, index=True)
    logger.info(f"Métadonnées sauvegardées dans {metadata_path}")

    # Sauvegarde des métriques de performance
    metrics = {
        'model'             : 'MobileCLIP-S2',
        'nb_frames_encoded' : int(nb_encoded),
        'nb_errors'         : int(errors),
        'encoding_time_sec' : round(encoding_time, 2),
        'encoding_time_min' : round(encoding_time / 60, 2),
        'frames_per_second' : round(nb_encoded / max(encoding_time, 0.01), 2),
        'embedding_dim'     : int(embeddings_matrix.shape[1]),
        'batch_size'        : batch_size,
        'device'            : device
    }

    metrics_path = output_dir / 'mobileclip_s2_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métriques sauvegardées dans {metrics_path}")


if __name__ == "__main__":
    encode_frames(
        frames_dir      = FRAMES_DIR,
        output_dir      = OUTPUT_DIR,
        checkpoint_path = CHECKPOINT_PATH,
        batch_size      = BATCH_SIZE
    )
