"""
test_encodage.py

Script de test pour vérifier que les trois modèles fonctionnent correctement
avant de lancer l'encodage complet sur tout le dataset.

On prend juste quelques frames au hasard et on fait tourner les trois modèles
dessus. A la fin on obtient exactement les mêmes fichiers que le vrai encodage
(.npy, .csv, .json) mais dans un dossier séparé "results_test" pour ne pas
écraser les vrais résultats.

Aucune modification des scripts principaux n'est nécessaire, tout est isolé ici.
"""

import json
import time
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Nombre de frames à tester par modèle, suffisant pour vérifier que tout fonctionne
NB_FRAMES_TEST = 20

FRAMES_DIR      = Path("frames")
OUTPUT_DIR_TEST = Path("results_test")  # dossier séparé pour ne pas toucher aux vrais résultats
CHECKPOINT_PATH = Path("checkpoints/mobileclip_s2.pt")
BATCH_SIZE      = 4


def charger_frames_test(frames_dir, nb_frames):
    """
    Récupère quelques frames au hasard dans le dataset pour le test.
    On prend des frames de classes différentes pour que ce soit représentatif.
    """
    toutes_les_frames = sorted(frames_dir.rglob('*_t*.jpg'))

    if not toutes_les_frames:
        raise FileNotFoundError(f"Aucune frame trouvée dans {frames_dir}")

    print(f"{len(toutes_les_frames):,} frames disponibles dans le dataset")

    # On tire au hasard pour avoir un échantillon varié
    frames_selectionnees = random.sample(toutes_les_frames, min(nb_frames, len(toutes_les_frames)))
    print(f"{len(frames_selectionnees)} frames sélectionnées pour le test\n")

    return frames_selectionnees


def extraire_metadata(path_str):
    """Extrait la classe, le nom de vidéo et le timestamp depuis le chemin d'une frame."""
    p          = Path(path_str)
    class_name = p.parts[-3]
    video_name = p.parts[-2]
    filename   = p.name
    timestamp  = int(filename.rsplit('_t', 1)[1].replace('s.jpg', ''))
    return {
        'filepath'  : path_str,
        'class'     : class_name,
        'video'     : video_name,
        'timestamp' : timestamp,
        'filename'  : filename
    }


def sauvegarder_resultats(embeddings_list, metadata_list, metrics, output_dir, prefix):
    """
    Sauvegarde les résultats du test dans le même format que le vrai encodage.
    Les trois fichiers produits sont identiques en structure à ceux du vrai encodage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_matrix = np.concatenate(embeddings_list, axis=0)

    embeddings_path = output_dir / f'{prefix}_embeddings.npy'
    np.save(embeddings_path, embeddings_matrix)

    metadata_path = output_dir / f'{prefix}_metadata.csv'
    pd.DataFrame(metadata_list).to_csv(metadata_path, index=True)

    metrics_path = output_dir / f'{prefix}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Résultats sauvegardés dans {output_dir}")
    print(f"  Shape des embeddings : {embeddings_matrix.shape}")
    print(f"  Temps d'encodage     : {metrics['encoding_time_sec']}s")
    print(f"  Vitesse              : {metrics['frames_per_second']} frames/sec\n")


# ─────────────────────────────────────────────────────────────────────────────
# TEST MOBILECLIP
# ─────────────────────────────────────────────────────────────────────────────

def test_mobileclip(frame_paths, output_dir, checkpoint_path, batch_size):
    print("=" * 50)
    print("TEST MOBILECLIP")
    print("=" * 50)

    import mobileclip

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Appareil utilisé : {device}")

    if not checkpoint_path.exists():
        print(f"Fichier de poids introuvable : {checkpoint_path}, test MobileCLIP ignoré")
        return

    print("Chargement du modèle...")
    model, _, preprocess = mobileclip.create_model_and_transforms(
        'mobileclip_s2', pretrained=str(checkpoint_path)
    )
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_metadata   = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(preprocess(img))
                except Exception:
                    continue

            if not images:
                continue

            batch_tensor = torch.stack(images).to(device)
            embeddings   = model.encode_image(batch_tensor)
            embeddings   = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())
            all_metadata.extend([extraire_metadata(str(p)) for p in batch_paths])

            print(f"  Batch {i // batch_size + 1} traité ({len(images)} images)")

    encoding_time = time.time() - start_time
    nb_encoded    = sum(len(b) for b in all_embeddings)

    metrics = {
        'model'             : 'MobileCLIP-S2',
        'nb_frames_encoded' : nb_encoded,
        'nb_errors'         : 0,
        'encoding_time_sec' : round(encoding_time, 2),
        'encoding_time_min' : round(encoding_time / 60, 2),
        'frames_per_second' : round(nb_encoded / encoding_time, 2),
        'embedding_dim'     : all_embeddings[0].shape[1] if all_embeddings else 0,
        'batch_size'        : batch_size,
        'device'            : device,
        'test'              : True  # marqueur pour distinguer les résultats de test
    }

    sauvegarder_resultats(all_embeddings, all_metadata, metrics, output_dir / 'mobileclip', 'mobileclip_s2')


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLIP
# ─────────────────────────────────────────────────────────────────────────────

def test_clip(frame_paths, output_dir, batch_size):
    print("=" * 50)
    print("TEST CLIP")
    print("=" * 50)

    import clip

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Appareil utilisé : {device}")

    print("Chargement du modèle (téléchargement automatique si premier lancement)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    all_embeddings = []
    all_metadata   = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(preprocess(img))
                except Exception:
                    continue

            if not images:
                continue

            batch_tensor = torch.stack(images).to(device)
            embeddings   = model.encode_image(batch_tensor)
            embeddings   = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().float().numpy())
            all_metadata.extend([extraire_metadata(str(p)) for p in batch_paths])

            print(f"  Batch {i // batch_size + 1} traité ({len(images)} images)")

    encoding_time = time.time() - start_time
    nb_encoded    = sum(len(b) for b in all_embeddings)

    metrics = {
        'model'             : 'CLIP ViT-B/32',
        'nb_frames_encoded' : nb_encoded,
        'nb_errors'         : 0,
        'encoding_time_sec' : round(encoding_time, 2),
        'encoding_time_min' : round(encoding_time / 60, 2),
        'frames_per_second' : round(nb_encoded / encoding_time, 2),
        'embedding_dim'     : all_embeddings[0].shape[1] if all_embeddings else 0,
        'batch_size'        : batch_size,
        'device'            : device,
        'test'              : True
    }

    sauvegarder_resultats(all_embeddings, all_metadata, metrics, output_dir / 'clip', 'clip')


# ─────────────────────────────────────────────────────────────────────────────
# TEST TINYCLIP
# ─────────────────────────────────────────────────────────────────────────────

def test_tinyclip(frame_paths, output_dir, batch_size):
    print("=" * 50)
    print("TEST TINYCLIP")
    print("=" * 50)

    from transformers import CLIPModel, CLIPProcessor

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
    print(f"Appareil utilisé : {device}")

    print("Chargement du modèle (téléchargement automatique si premier lancement)...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model     = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    all_metadata   = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                except Exception:
                    continue

            if not images:
                continue

            # TinyCLIP utilise le processor HuggingFace plutôt qu'un préprocesseur classique
            inputs       = processor(images=images, return_tensors="pt", padding=True)
            pixel_values = inputs['pixel_values'].to(device)
            embeddings = model.get_image_features(pixel_values=pixel_values)
            if hasattr(embeddings, 'image_embeds'):
                embeddings = embeddings.image_embeds
            elif hasattr(embeddings, 'pooler_output'):
                embeddings = embeddings.pooler_output
            else:
                embeddings = embeddings.last_hidden_state[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().float().numpy())
            all_metadata.extend([extraire_metadata(str(p)) for p in batch_paths])

            print(f"  Batch {i // batch_size + 1} traité ({len(images)} images)")

    encoding_time = time.time() - start_time
    nb_encoded    = sum(len(b) for b in all_embeddings)

    metrics = {
        'model'             : f'TinyCLIP ({model_name.split("/")[1]})',
        'nb_frames_encoded' : nb_encoded,
        'nb_errors'         : 0,
        'encoding_time_sec' : round(encoding_time, 2),
        'encoding_time_min' : round(encoding_time / 60, 2),
        'frames_per_second' : round(nb_encoded / encoding_time, 2),
        'embedding_dim'     : all_embeddings[0].shape[1] if all_embeddings else 0,
        'batch_size'        : batch_size,
        'device'            : device,
        'test'              : True
    }

    sauvegarder_resultats(all_embeddings, all_metadata, metrics, output_dir / 'tinyclip', 'tinyclip')


# ─────────────────────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    frame_paths = charger_frames_test(FRAMES_DIR, NB_FRAMES_TEST)

    test_mobileclip(frame_paths, OUTPUT_DIR_TEST, CHECKPOINT_PATH, BATCH_SIZE)
    test_clip(frame_paths, OUTPUT_DIR_TEST, BATCH_SIZE)
    test_tinyclip(frame_paths, OUTPUT_DIR_TEST, BATCH_SIZE)

    print("=" * 50)
    print("Tous les tests sont terminés.")
    print(f"Les résultats sont dans le dossier {OUTPUT_DIR_TEST}")
    print("Le dossier results/ n'a pas été touché.")
    print("=" * 50)
