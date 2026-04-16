"""
Ce script encode toutes les frames extraites du dataset UCF-Crime avec le modèle SigLIP de Google.

On sauvegarde trois fichiers à la fin :
    sigclip_embeddings.npy     les vecteurs de toutes les frames
    sigclip_metadata.csv       les infos associées à chaque frame (classe, vidéo, timestamp)
    sigclip_metrics.json       les performances du modèle (temps, vitesse...)

Le script supporte la reprise après interruption grâce à un système de checkpoint.
Un checkpoint est sauvegardé toutes les 10 minutes dans results/sigclip/sigclip_checkpoint.npz.
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
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


FRAMES_DIR  = Path("frames")
OUTPUT_DIR  = Path("results/siglip")

# Variante SigLIP — on peut aussi utiliser "google/siglip-large-patch16-384"
MODEL_NAME  = "google/siglip-base-patch16-224"

BATCH_SIZE  = 32

CHECKPOINT_INTERVAL = 10 * 60


class FrameDataset(Dataset):

    def __init__(self, frame_paths, processor):
        self.frame_paths = frame_paths
        self.processor   = processor

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            # Le processor HuggingFace gère le resize et la normalisation propres à SigLIP
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            return pixel_values, str(path), True
        except Exception:
            # Image corrompue : on retourne un tensor vide pour ne pas interrompre le script
            dummy_size   = self.processor.image_processor.size.get("height", 224)
            dummy        = torch.zeros(3, dummy_size, dummy_size)
            return dummy, str(path), False


def save_checkpoint(output_dir, all_embeddings, all_metadata, errors, nb_frames_processed):
    checkpoint_path = output_dir / 'siglip_checkpoint.npz'
    tmp_path        = output_dir / 'siglip_checkpoint_tmp.npz'

    emb_matrix = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 768), dtype=np.float32)

    np.savez(
        tmp_path,
        embeddings=emb_matrix,
        nb_frames_processed=np.array(nb_frames_processed),
        errors=np.array(errors)
    )
    meta_tmp = output_dir / 'siglip_checkpoint_meta.csv'
    pd.DataFrame(all_metadata).to_csv(meta_tmp, index=True)

    tmp_path.replace(checkpoint_path)
    logger.info(f"Checkpoint sauvegardé : {len(all_metadata)} frames traitées")


def load_checkpoint(output_dir):
    checkpoint_path = output_dir / 'siglip_checkpoint.npz'
    meta_path       = output_dir / 'siglip_checkpoint_meta.csv'

    if not checkpoint_path.exists() or not meta_path.exists():
        return None

    data               = np.load(checkpoint_path)
    embeddings         = data['embeddings']
    nb_frames_processed = int(data['nb_frames_processed'])
    errors             = int(data['errors'])

    meta_df  = pd.read_csv(meta_path, index_col=0)
    metadata = meta_df.to_dict('records')

    return {
        'embeddings'         : embeddings,
        'metadata'           : metadata,
        'nb_frames_processed': nb_frames_processed,
        'errors'             : errors
    }


def cleanup_checkpoint(output_dir):
    for name in ['siglip_checkpoint.npz', 'siglip_checkpoint_meta.csv', 'siglip_checkpoint_tmp.npz']:
        p = output_dir / name
        if p.exists():
            p.unlink()
    logger.info("Fichiers de checkpoint nettoyés")


def encode_frames(frames_dir, output_dir, model_name, batch_size):

    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Appareil utilisé : {device}")
    if device == 'cuda':
        logger.info(f"GPU détecté : {torch.cuda.get_device_name(0)}")

    logger.info(f"Chargement du modèle SigLIP {model_name}...")
    # AutoProcessor gère la normalisation spécifique à SigLIP (différente de CLIP classique)
    processor = AutoProcessor.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Dimension des embeddings selon la variante choisie (base → 768, large → 1024)
    embedding_dim = model.config.vision_config.hidden_size
    logger.info(f"Modèle SigLIP chargé — embedding dim : {embedding_dim}")

    logger.info(f"Recherche des frames dans {frames_dir}...")
    frame_paths = sorted(frames_dir.rglob('*_t*.jpg'))

    if not frame_paths:
        raise FileNotFoundError(f"Aucune frame trouvée dans {frames_dir}")

    logger.info(f"{len(frame_paths):,} frames trouvées")

    checkpoint   = load_checkpoint(output_dir)
    skip_frames  = 0
    all_embeddings = []
    all_metadata   = []
    errors         = 0

    if checkpoint is not None:
        skip_frames = checkpoint['nb_frames_processed']
        all_embeddings.append(checkpoint['embeddings'])
        all_metadata = checkpoint['metadata']
        errors       = checkpoint['errors']
        logger.info(f"Checkpoint trouvé : {skip_frames:,} frames déjà traitées, reprise en cours...")

        frame_paths = frame_paths[skip_frames:]
        logger.info(f"{len(frame_paths):,} frames restantes à encoder")

        if len(frame_paths) == 0:
            logger.info("Toutes les frames ont déjà été encodées !")
            embeddings_matrix = checkpoint['embeddings']
            _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                                0, len(embeddings_matrix), model_name, batch_size, device)
            cleanup_checkpoint(output_dir)
            return
    else:
        logger.info("Aucun checkpoint trouvé, démarrage depuis le début")

    dataset = FrameDataset(frame_paths, processor)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )

    frames_this_session = 0

    logger.info("Démarrage de l'encodage...")
    start_time          = time.time()
    last_checkpoint_time = start_time

    with torch.no_grad():
        for images, paths, valid_flags in tqdm(loader, desc="SigLIP"):

            valid_mask   = valid_flags.bool()
            valid_images = images[valid_mask].to(device)
            valid_paths  = [p for p, v in zip(paths, valid_flags) if v]

            if len(valid_images) == 0:
                errors              += len(paths)
                frames_this_session += len(paths)
                continue

            # SigLIP expose get_image_features() exactement comme CLIP, on l'utilise directement
            embeddings = model.vision_model(pixel_values=valid_images).pooler_output

            # Normalisation L2 — nécessaire pour que la similarité cosinus soit correcte
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().float().numpy())

            for path_str in valid_paths:
                p          = Path(path_str)
                class_name = p.parts[-3]
                video_name = p.parts[-2]
                filename   = p.name
                timestamp  = int(filename.rsplit('_t', 1)[1].replace('s.jpg', ''))

                all_metadata.append({
                    'filepath' : path_str,
                    'class'    : class_name,
                    'video'    : video_name,
                    'timestamp': timestamp,
                    'filename' : filename
                })

            errors              += (~valid_mask).sum().item()
            frames_this_session += len(paths)

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

    embeddings_matrix = np.concatenate(all_embeddings, axis=0)

    _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                        encoding_time, nb_encoded, model_name, batch_size, device)

    cleanup_checkpoint(output_dir)


def _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                        encoding_time, nb_encoded, model_name, batch_size, device):

    embeddings_path = output_dir / 'siglip_embeddings.npy'
    np.save(embeddings_path, embeddings_matrix)
    logger.info(f"Embeddings sauvegardés dans {embeddings_path}, shape : {embeddings_matrix.shape}")

    metadata_path = output_dir / 'siglip_metadata.csv'
    pd.DataFrame(all_metadata).to_csv(metadata_path, index=True)
    logger.info(f"Métadonnées sauvegardées dans {metadata_path}")

    metrics = {
        'model'             : f'SigLIP {model_name}',
        'nb_frames_encoded' : int(nb_encoded),
        'nb_errors'         : int(errors),
        'encoding_time_sec' : round(encoding_time, 2),
        'encoding_time_min' : round(encoding_time / 60, 2),
        'frames_per_second' : round(nb_encoded / max(encoding_time, 0.01), 2),
        'embedding_dim'     : int(embeddings_matrix.shape[1]),
        'batch_size'        : batch_size,
        'device'            : device
    }

    metrics_path = output_dir / 'siglip_metrics.json'
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