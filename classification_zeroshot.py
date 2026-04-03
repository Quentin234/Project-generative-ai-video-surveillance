"""
Ce script évalue la capacité des trois modèles à classifier des frames
sans avoir été entraînés spécifiquement sur UCF-Crime.

Le principe du zero-shot est le suivant : on encode les 13 labels texte du dataset
(fighting, shooting, etc.) avec le même modèle qu'on a utilisé pour encoder
les images. Ensuite pour chaque frame on regarde quel label textuel lui ressemble
le plus. La classe avec le score de similarité le plus élevé devient la prédiction.

On compare ensuite les prédictions avec les vraies classes pour calculer l'accuracy,
c'est à dire le pourcentage de frames correctement classifiées.

Les résultats sont sauvegardés dans results/<modele>/zeroshot/ :
    predictions.csv     la prédiction et le vrai label pour chaque frame
    accuracy.json       le score final et les détails par classe
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
import mobileclip
import clip
from transformers import CLIPModel, CLIPProcessor

# Affiche des logs pour suivre l'avancement du script et détecter d'éventuels problèmes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


RESULTS_DIR = Path("results")

# 13 classes formulées comme des phrases car les modèles ont été entraînés sur des descriptions textuelles
LABELS = [
    "a person fighting",
    "a robbery",
    "a car accident",
    "vandalism",
    "a person being arrested",
    "a burglary",
    "a person shooting",
    "an abuse",
    "an arson",
    "an assault",
    "a road accident",
    "shoplifting",
    "a person stealing"
]

# Correspondance entre les noms de dossiers du dataset et les index de labels ci-dessus
CLASSES_DATASET = [
    "Fighting",
    "Robbery",
    "Accident",
    "Vandalism",
    "Arrest",
    "Burglary",
    "Shooting",
    "Abuse",
    "Arson",
    "Assault",
    "RoadAccidents",
    "Shoplifting",
    "Stealing"
]

MODELES = [
    {
        "nom"    : "MobileCLIP-S2",
        "dossier": RESULTS_DIR / "mobileclip",
        "prefixe": "mobileclip_s2",
        "type"   : "mobileclip"
    },
    {
        "nom"    : "CLIP ViT-B/32",
        "dossier": RESULTS_DIR / "clip",
        "prefixe": "clip",
        "type"   : "clip"
    },
    {
        "nom"    : "TinyCLIP",
        "dossier": RESULTS_DIR / "tinyclip",
        "prefixe": "tinyclip",
        "type"   : "tinyclip"
    }
]

# Encode les labels texte avec MobileCLIP.
def encoder_labels_mobileclip(labels, checkpoint_path, device):

    model, _, _ = mobileclip.create_model_and_transforms(
        "mobileclip_s2", pretrained=str(checkpoint_path)
    )
    model = model.to(device)
    model.eval()

    tokenizer = mobileclip.get_tokenizer("mobileclip_s2")
    tokens    = tokenizer(labels).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    return text_embeddings.cpu().numpy()

# Encode les labels texte avec CLIP.
def encoder_labels_clip(labels, device):

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    tokens = clip.tokenize(labels).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    return text_embeddings.cpu().float().numpy()

# Encode les labels texte avec TinyCLIP
def encoder_labels_tinyclip(labels, device):

    model_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
    processor  = CLIPProcessor.from_pretrained(model_name)
    model      = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    inputs = processor(text=labels, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)
        if not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings.pooler_output
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    return text_embeddings.cpu().float().numpy()

# Effectue la classification zero-shot pour un modèle donné
def classifier_modele(config):

    nom    = config["nom"]
    dossier = config["dossier"]
    prefixe = config["prefixe"]
    type_m  = config["type"]

    logger.info(f"Classification zero-shot pour {nom}...")

    embeddings_path = dossier / f"{prefixe}_embeddings.npy"
    metadata_path   = dossier / f"{prefixe}_metadata.csv"

    if not embeddings_path.exists() or not metadata_path.exists():
        logger.warning(f"Fichiers manquants pour {nom}, modèle ignoré")
        return

    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata   = pd.read_csv(metadata_path, index_col=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Encodage des labels texte...")
    if type_m == "mobileclip":
        checkpoint = Path("checkpoints/mobileclip_s2.pt")
        text_embeddings = encoder_labels_mobileclip(LABELS, checkpoint, device)
    elif type_m == "clip":
        text_embeddings = encoder_labels_clip(LABELS, device)
    else:
        text_embeddings = encoder_labels_tinyclip(LABELS, device)

    logger.info(f"Labels encodés, shape : {text_embeddings.shape}")

    # Calcul de la similarité entre chaque frame et chaque label par multiplication matricielle
    logger.info("Calcul des similarités...")
    scores = embeddings @ text_embeddings.T

    # Pour chaque frame, on prend l'index du label avec le score le plus élevé
    predictions_idx = np.argmax(scores, axis=1)
    predictions     = [CLASSES_DATASET[i] for i in predictions_idx]

    # Calcul de l'accuracy en comparant les prédictions avec les vraies classes
    vraies_classes = metadata["class"].tolist()
    correct        = sum(p == v for p, v in zip(predictions, vraies_classes))
    accuracy       = correct / len(vraies_classes)

    logger.info(f"Accuracy globale : {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Calcul de l'accuracy par classe pour voir où le modèle se trompe le plus
    accuracy_par_classe = {}
    for classe in CLASSES_DATASET:
        mask      = [v == classe for v in vraies_classes]
        nb_frames = sum(mask)
        if nb_frames == 0:
            continue
        nb_correct = sum(
            p == v for p, v, m in zip(predictions, vraies_classes, mask) if m
        )
        accuracy_par_classe[classe] = round(nb_correct / nb_frames, 4)

    # Sauvegarde des prédictions frame par frame pour pouvoir analyser les erreurs
    output_dir = dossier / "zeroshot"
    output_dir.mkdir(parents=True, exist_ok=True)

    resultats              = metadata.copy()
    resultats["prediction"] = predictions
    resultats["correct"]    = [p == v for p, v in zip(predictions, vraies_classes)]
    resultats["score_max"]  = scores.max(axis=1)

    predictions_path = output_dir / "predictions.csv"
    resultats.to_csv(predictions_path, index=True)
    logger.info(f"Prédictions sauvegardées dans {predictions_path}")

    # Sauvegarde du résumé des performances pour les graphiques de comparaison
    accuracy_data = {
        "modele"             : nom,
        "accuracy_globale"   : round(accuracy, 4),
        "accuracy_pourcent"  : round(accuracy * 100, 2),
        "nb_frames_total"    : len(vraies_classes),
        "nb_correct"         : correct,
        "accuracy_par_classe": accuracy_par_classe
    }

    accuracy_path = output_dir / "accuracy.json"
    with open(accuracy_path, "w") as f:
        json.dump(accuracy_data, f, indent=2)
    logger.info(f"Accuracy sauvegardée dans {accuracy_path}\n")


if __name__ == "__main__":
    logger.info("Démarrage de la classification zero-shot pour les trois modèles\n")

    for config in MODELES:
        classifier_modele(config)

    logger.info("Classification terminée pour tous les modèles")
