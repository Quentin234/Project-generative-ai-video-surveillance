"""
Ce script compare quatre façons différentes de représenter les classes textuelles
pour la classification zero-shot, sur tous les modèles du projet.

Le but est de trouver la meilleure stratégie de description avant de passer
aux évaluations suivantes (top-k, agrégation par vidéo).

Les quatre méthodes testées :

    desc_1_single
        Une seule description par classe, exactement comme dans classification_zeroshot.py.
        C'est notre baseline, on a obtenu environ 20% avec cette méthode.

    desc_2_best_of_5
        Cinq descriptions par classe. Pour chaque frame on calcule la similarité
        avec les 5 descriptions de chaque classe, et on garde le meilleur score
        comme score de la classe. L'idée est qu'au moins une des 5 descriptions
        devrait bien matcher l'image.

    desc_3_mean_score
        Cinq descriptions par classe. Pour chaque frame on calcule la similarité
        avec les 5 descriptions de chaque classe, et on prend la moyenne des 5 scores
        comme score de la classe. Ça donne une représentation plus stable.

    desc_4_mean_embed
        Cinq descriptions par classe. On encode les 5 descriptions et on moyenne
        leurs vecteurs pour obtenir un seul embedding représentant la classe.
        C'est la méthode recommandée par le papier original CLIP (prompt ensembling).
        On ne fait ensuite qu'un seul produit scalaire par classe, comme avec 1 description.

Pour chaque combinaison (méthode x modèle) on sauvegarde :
    <modele>_accuracy.json      accuracy globale et par classe
    <modele>_confusion.png      matrice de confusion

Tous les résultats vont dans results/evaluations/descriptions/<methode>/

Structure attendue pour les embeddings :
    results/<modele>/<prefixe>_embeddings.npy
    results/<modele>/<prefixe>_metadata.csv
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


RESULTS_DIR = Path("results")
EVAL_DIR    = RESULTS_DIR / "evaluations" / "descriptions"

# Les 13 classes du dataset avec leur correspondance dans les dossiers
CLASSES_DATASET = [
    "Fighting", "Robbery", "Accident", "Vandalism", "Arrest",
    "Burglary", "Shooting", "Abuse", "Arson", "Assault",
    "RoadAccidents", "Shoplifting", "Stealing"
]

# Une seule description par classe — c'est notre baseline (desc_1_single)
DESCRIPTIONS_SINGLE = {
    "Fighting"     : "a person fighting",
    "Robbery"      : "a robbery",
    "Accident"     : "a car accident",
    "Vandalism"    : "vandalism",
    "Arrest"       : "a person being arrested",
    "Burglary"     : "a burglary",
    "Shooting"     : "a person shooting",
    "Abuse"        : "an abuse",
    "Arson"        : "an arson",
    "Assault"      : "an assault",
    "RoadAccidents": "a road accident",
    "Shoplifting"  : "shoplifting",
    "Stealing"     : "a person stealing"
}

# Cinq descriptions par classe — utilisées pour les méthodes 2, 3 et 4
# Chaque formulation aborde la classe sous un angle légèrement différent pour
# couvrir la diversité visuelle des frames du dataset
DESCRIPTIONS_MULTI = {
    "Fighting": [
        "a person fighting",
        "people punching and kicking each other",
        "a violent fight between people in a surveillance video",
        "two people hitting each other",
        "a physical altercation caught on camera"
    ],
    "Robbery": [
        "a robbery",
        "someone robbing a person at gunpoint or by force",
        "a thief stealing from someone aggressively",
        "a mugging caught on surveillance camera",
        "someone threatening a person to steal their belongings"
    ],
    "Accident": [
        "a car accident",
        "a vehicle collision",
        "cars crashing into each other",
        "a traffic accident with damaged vehicles",
        "an accident caught on surveillance camera"
    ],
    "Vandalism": [
        "vandalism",
        "someone destroying public property",
        "a person spray painting graffiti",
        "someone breaking windows or damaging property",
        "vandalism caught on surveillance camera"
    ],
    "Arrest": [
        "a person being arrested by police",
        "police officers arresting someone",
        "law enforcement detaining a suspect",
        "a person being handcuffed by police",
        "police making an arrest"
    ],
    "Burglary": [
        "a burglary",
        "someone breaking into a building",
        "a person entering a building illegally",
        "a burglar breaking and entering",
        "someone sneaking into a building to steal"
    ],
    "Shooting": [
        "a person shooting a gun",
        "gunfire in a surveillance video",
        "someone firing a weapon",
        "a shooting incident caught on camera",
        "a person with a firearm shooting at someone"
    ],
    "Abuse": [
        "a person being abused",
        "someone hitting a vulnerable person",
        "physical abuse caught on camera",
        "a person violently attacking another person",
        "abuse of a child or elderly person on camera"
    ],
    "Arson": [
        "arson, someone setting fire to a building",
        "a person starting a fire intentionally",
        "fire and smoke from an intentional fire",
        "someone burning a car or building",
        "arson caught on surveillance camera"
    ],
    "Assault": [
        "an assault",
        "someone violently attacking a person",
        "a person being beaten up",
        "a violent attack on a person caught on camera",
        "someone assaulting another person"
    ],
    "RoadAccidents": [
        "a road accident",
        "a car crash on the road",
        "vehicles colliding at an intersection",
        "a traffic accident with pedestrians",
        "a road accident caught on a traffic camera"
    ],
    "Shoplifting": [
        "shoplifting in a store",
        "a person stealing merchandise from a shop",
        "someone hiding items to steal from a store",
        "shoplifting caught on store surveillance camera",
        "a person leaving a store without paying"
    ],
    "Stealing": [
        "a person stealing",
        "someone taking objects that do not belong to them",
        "theft caught on surveillance camera",
        "a person picking up and stealing an object",
        "someone stealing from a car or public place"
    ]
}

# Configuration des 7 modèles du projet
# Chaque modèle a son propre type d'encodeur texte qu'on utilisera pour encoder les descriptions
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
    },
    {
        "nom"    : "SigLIP",
        "dossier": RESULTS_DIR / "siglip",
        "prefixe": "siglip",
        "type"   : "siglip"
    },
    {
        "nom"             : "EVA-CLIP",
        "dossier"         : RESULTS_DIR / "evaclip",
        "prefixe"         : "evaclip",
        "type"            : "openclip",
        "model_name"      : "EVA02-B-16",
        "model_pretrained": "merged2b_s8b_b131k"
    },
    {
        "nom"             : "OpenCLIP",
        "dossier"         : RESULTS_DIR / "openclip",
        "prefixe"         : "openclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-32",
        "model_pretrained": "laion2b_s34b_b79k"
    },
    {
        "nom"             : "MetaCLIP",
        "dossier"         : RESULTS_DIR / "metaclip",
        "prefixe"         : "metaclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-32-quickgelu",
        "model_pretrained": "metaclip_400m"
    },
    {
        "nom"             : "DFN-CLIP",
        "dossier"         : RESULTS_DIR / "dfnclip",
        "prefixe"         : "dfnclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-16",
        "model_pretrained": "dfn2b"
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# ENCODEURS TEXTE
# Une fonction par famille de modèle, même logique que dans classification_zeroshot.py
# ─────────────────────────────────────────────────────────────────────────────

def encoder_textes_mobileclip(textes, device):
    """Encode une liste de textes avec MobileCLIP et retourne les embeddings normalisés."""
    import mobileclip

    model, _, _ = mobileclip.create_model_and_transforms(
        "mobileclip_s2", pretrained=str(Path("checkpoints/mobileclip_s2.pt"))
    )
    model = model.to(device)
    model.eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s2")
    tokens    = tokenizer(textes).to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().float().numpy()


def encoder_textes_clip(textes, device):
    """Encode une liste de textes avec CLIP et retourne les embeddings normalisés."""
    import clip

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    tokens = clip.tokenize(textes).to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().float().numpy()


def encoder_textes_tinyclip(textes, device):
    """Encode une liste de textes avec TinyCLIP et retourne les embeddings normalisés."""
    from transformers import CLIPModel, CLIPProcessor

    model_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
    processor  = CLIPProcessor.from_pretrained(model_name)
    model      = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    inputs = processor(text=textes, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().float().numpy()


def encoder_textes_siglip(textes, device):
    """Encode une liste de textes avec SigLIP et retourne les embeddings normalisés."""
    from transformers import AutoModel, AutoProcessor

    model_name = "google/siglip-base-patch16-224"
    processor  = AutoProcessor.from_pretrained(model_name)
    model      = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    inputs = processor(text=textes, return_tensors="pt", padding="max_length",
                       max_length=64, truncation=True).to(device)

    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().float().numpy()


def encoder_textes_openclip(textes, model_name, model_pretrained, device):
    """Encode une liste de textes avec un modèle open_clip et retourne les embeddings normalisés."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=model_pretrained
    )
    model     = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    tokens    = tokenizer(textes).to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().float().numpy()


def encoder_textes(textes, config, device):
    """
    Appelle le bon encodeur selon le type du modèle.
    Fonction centrale qui centralise le dispatch vers les encodeurs spécifiques.
    """
    type_m = config["type"]

    if type_m == "mobileclip":
        return encoder_textes_mobileclip(textes, device)
    elif type_m == "clip":
        return encoder_textes_clip(textes, device)
    elif type_m == "tinyclip":
        return encoder_textes_tinyclip(textes, device)
    elif type_m == "siglip":
        return encoder_textes_siglip(textes, device)
    elif type_m == "openclip":
        return encoder_textes_openclip(
            textes, config["model_name"], config["model_pretrained"], device
        )
    else:
        raise ValueError(f"Type de modèle inconnu : {type_m}")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DES MATRICES DE CLASSE
# Chaque méthode produit une matrice (nb_classes x dim_embedding)
# qu'on multiplie ensuite avec les embeddings image pour obtenir les scores
# ─────────────────────────────────────────────────────────────────────────────

def construire_matrice_single(config, device):
    """
    Méthode 1 : une seule description par classe.
    On encode les 13 descriptions et on obtient une matrice (13 x dim).
    """
    textes = [DESCRIPTIONS_SINGLE[c] for c in CLASSES_DATASET]
    return encoder_textes(textes, config, device)


def construire_matrices_multi(config, device):
    """
    Encode les 5 descriptions de chaque classe et retourne un dictionnaire
    classe -> matrice (5 x dim). Utilisé par les méthodes 2 et 3.
    """
    embeddings_par_classe = {}
    for classe in CLASSES_DATASET:
        emb = encoder_textes(DESCRIPTIONS_MULTI[classe], config, device)
        embeddings_par_classe[classe] = emb
    return embeddings_par_classe


def construire_matrice_mean_embed(config, device):
    """
    Méthode 4 : on encode les 5 descriptions de chaque classe et on moyenne
    leurs embeddings pour obtenir un seul vecteur représentant la classe.
    C'est la méthode recommandée dans le papier CLIP original (prompt ensembling).
    La moyenne se fait dans l'espace des embeddings avant normalisation finale.
    """
    vecteurs = []
    for classe in CLASSES_DATASET:
        emb   = encoder_textes(DESCRIPTIONS_MULTI[classe], config, device)
        # On fait la moyenne des 5 embeddings puis on renormalise
        moyen = emb.mean(axis=0)
        moyen = moyen / np.linalg.norm(moyen)
        vecteurs.append(moyen)

    return np.stack(vecteurs).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES SCORES POUR CHAQUE MÉTHODE
# ─────────────────────────────────────────────────────────────────────────────

def calculer_scores_single(embeddings_image, config, device):
    """
    Méthode 1 : produit scalaire direct entre images et descriptions uniques.
    Retourne une matrice (nb_frames x 13) de scores.
    """
    matrice_texte = construire_matrice_single(config, device)
    return embeddings_image @ matrice_texte.T


def calculer_scores_best_of_5(embeddings_image, config, device):
    """
    Méthode 2 : pour chaque classe on prend le meilleur score parmi les 5 descriptions.
    Pour chaque frame et chaque classe, on garde le maximum des similarités
    avec les 5 descriptions. Ça favorise les descriptions très spécifiques
    qui matchent bien certains types d'images.
    """
    embeddings_par_classe = construire_matrices_multi(config, device)
    nb_frames  = len(embeddings_image)
    nb_classes = len(CLASSES_DATASET)
    scores     = np.zeros((nb_frames, nb_classes), dtype=np.float32)

    for i, classe in enumerate(CLASSES_DATASET):
        emb_classe = embeddings_par_classe[classe]
        # Produit scalaire : (nb_frames x dim) @ (dim x 5) → (nb_frames x 5)
        scores_5   = embeddings_image @ emb_classe.T
        # On garde le meilleur score parmi les 5 descriptions
        scores[:, i] = scores_5.max(axis=1)

    return scores


def calculer_scores_mean_score(embeddings_image, config, device):
    """
    Méthode 3 : pour chaque classe on moyenne les scores des 5 descriptions.
    Ça donne une représentation plus lisse et moins sensible aux descriptions
    qui matchent accidentellement des mauvaises frames.
    """
    embeddings_par_classe = construire_matrices_multi(config, device)
    nb_frames  = len(embeddings_image)
    nb_classes = len(CLASSES_DATASET)
    scores     = np.zeros((nb_frames, nb_classes), dtype=np.float32)

    for i, classe in enumerate(CLASSES_DATASET):
        emb_classe = embeddings_par_classe[classe]
        scores_5   = embeddings_image @ emb_classe.T
        # On moyenne les 5 scores
        scores[:, i] = scores_5.mean(axis=1)

    return scores


def calculer_scores_mean_embed(embeddings_image, config, device):
    """
    Méthode 4 : on utilise l'embedding moyen des 5 descriptions comme représentant de la classe.
    Un seul produit scalaire par classe, comme avec 1 description,
    mais le vecteur de classe est plus riche car il résume 5 formulations.
    """
    matrice_texte = construire_matrice_mean_embed(config, device)
    return embeddings_image @ matrice_texte.T


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION ET SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

def evaluer_et_sauvegarder(scores, metadata, nom_modele, nom_methode, output_dir):
    """
    Prend une matrice de scores (nb_frames x 13), calcule l'accuracy top-1,
    l'accuracy par classe, et génère la matrice de confusion.
    Sauvegarde les résultats dans output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    vraies_classes  = metadata["class"].tolist()
    predictions_idx = np.argmax(scores, axis=1)
    predictions     = [CLASSES_DATASET[i] for i in predictions_idx]

    correct  = sum(p == v for p, v in zip(predictions, vraies_classes))
    accuracy = correct / len(vraies_classes)

    # Accuracy par classe pour voir où le modèle se trompe le plus
    accuracy_par_classe = {}
    for classe in CLASSES_DATASET:
        mask       = [v == classe for v in vraies_classes]
        nb_frames  = sum(mask)
        if nb_frames == 0:
            continue
        nb_correct = sum(p == v for p, v, m in zip(predictions, vraies_classes, mask) if m)
        accuracy_par_classe[classe] = round(nb_correct / nb_frames, 4)

    logger.info(f"  {nom_modele} / {nom_methode} : {accuracy*100:.2f}%")

    # Sauvegarde de l'accuracy
    nom_fichier = nom_modele.lower().replace(" ", "_").replace("/", "_")
    accuracy_data = {
        "modele"             : nom_modele,
        "methode_description": nom_methode,
        "accuracy_globale"   : round(accuracy, 4),
        "accuracy_pourcent"  : round(accuracy * 100, 2),
        "nb_frames_total"    : len(vraies_classes),
        "nb_correct"         : correct,
        "accuracy_par_classe": accuracy_par_classe
    }

    with open(output_dir / f"{nom_fichier}_accuracy.json", "w") as f:
        json.dump(accuracy_data, f, indent=2)

    # Génération de la matrice de confusion
    cm   = confusion_matrix(vraies_classes, predictions, labels=CLASSES_DATASET)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_DATASET)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(
        f"Matrice de confusion — {nom_modele}\nMéthode : {nom_methode} "
        f"(accuracy : {accuracy*100:.2f}%)",
        fontsize=11, pad=15
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{nom_fichier}_confusion.png", dpi=130, bbox_inches="tight")
    plt.close()

    return accuracy


def evaluer_modele(config):
    """
    Lance les 4 méthodes de description sur un modèle donné.
    Pour chaque méthode on calcule les scores, l'accuracy et la matrice de confusion.
    """
    nom     = config["nom"]
    dossier = config["dossier"]
    prefixe = config["prefixe"]

    embeddings_path = dossier / f"{prefixe}_embeddings.npy"
    metadata_path   = dossier / f"{prefixe}_metadata.csv"

    if not embeddings_path.exists() or not metadata_path.exists():
        logger.warning(f"Fichiers d'embeddings introuvables pour {nom}, modèle ignoré")
        return {}

    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata   = pd.read_csv(metadata_path, index_col=0)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"\n{nom} — {len(embeddings):,} frames chargées sur {device}")

    resultats = {}

    # Méthode 1 : une seule description
    logger.info("  Méthode 1 : single description...")
    scores   = calculer_scores_single(embeddings, config, device)
    output   = EVAL_DIR / "desc_1_single"
    accuracy = evaluer_et_sauvegarder(scores, metadata, nom, "1 description", output)
    resultats["desc_1_single"] = accuracy

    # Méthode 2 : 5 descriptions, meilleur score
    logger.info("  Méthode 2 : best of 5 descriptions...")
    scores   = calculer_scores_best_of_5(embeddings, config, device)
    output   = EVAL_DIR / "desc_2_best_of_5"
    accuracy = evaluer_et_sauvegarder(scores, metadata, nom, "5 desc. meilleur score", output)
    resultats["desc_2_best_of_5"] = accuracy

    # Méthode 3 : 5 descriptions, moyenne des scores
    logger.info("  Méthode 3 : mean of 5 scores...")
    scores   = calculer_scores_mean_score(embeddings, config, device)
    output   = EVAL_DIR / "desc_3_mean_score"
    accuracy = evaluer_et_sauvegarder(scores, metadata, nom, "5 desc. score moyen", output)
    resultats["desc_3_mean_score"] = accuracy

    # Méthode 4 : embedding moyen des 5 descriptions
    logger.info("  Méthode 4 : mean embedding of 5 descriptions...")
    scores   = calculer_scores_mean_embed(embeddings, config, device)
    output   = EVAL_DIR / "desc_4_mean_embed"
    accuracy = evaluer_et_sauvegarder(scores, metadata, nom, "5 desc. embedding moyen", output)
    resultats["desc_4_mean_embed"] = accuracy

    return resultats


def generer_tableau_comparatif(tous_resultats):
    """
    Génère un tableau récapitulatif de toutes les combinaisons modèle x méthode
    et le sauvegarde en CSV et en graphique pour faciliter la comparaison.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    methodes = ["desc_1_single", "desc_2_best_of_5", "desc_3_mean_score", "desc_4_mean_embed"]
    labels_methodes = {
        "desc_1_single"    : "1 description",
        "desc_2_best_of_5" : "5 desc. meilleur score",
        "desc_3_mean_score": "5 desc. score moyen",
        "desc_4_mean_embed": "5 desc. embedding moyen"
    }

    lignes = []
    for nom_modele, resultats in tous_resultats.items():
        ligne = {"modele": nom_modele}
        for m in methodes:
            ligne[labels_methodes[m]] = round(resultats.get(m, 0) * 100, 2)
        lignes.append(ligne)

    df = pd.DataFrame(lignes).set_index("modele")
    df.to_csv(EVAL_DIR / "comparaison_descriptions.csv")
    logger.info(f"\nTableau comparatif sauvegardé dans {EVAL_DIR / 'comparaison_descriptions.csv'}")

    # Graphique de comparaison des 4 méthodes pour chaque modèle
    fig, ax = plt.subplots(figsize=(14, 7))

    x          = np.arange(len(df.index))
    nb_methodes = len(df.columns)
    largeur    = 0.8 / nb_methodes
    couleurs   = ["#4A6FD4", "#E07B39", "#3DAA6E", "#C45BAA"]

    for i, (col, couleur) in enumerate(zip(df.columns, couleurs)):
        offset = (i - nb_methodes / 2 + 0.5) * largeur
        barres = ax.bar(x + offset, df[col], width=largeur, label=col,
                        color=couleur, alpha=0.85, edgecolor="white", linewidth=0.8)
        # On affiche la valeur au-dessus de chaque barre
        for barre in barres:
            h = barre.get_height()
            if h > 0:
                ax.text(barre.get_x() + barre.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Comparaison des méthodes de description — classification zero-shot (top-1, frame par frame)",
                 fontsize=11, pad=12)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_ylim(0, max(df.values.max() * 1.15, 30))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(EVAL_DIR / "comparaison_descriptions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé dans {EVAL_DIR / 'comparaison_descriptions.png'}")

    # Affichage du résumé dans les logs
    logger.info("\n── Résumé des accuracy (%) ──")
    logger.info(df.to_string())


if __name__ == "__main__":
    logger.info("Évaluation des méthodes de description pour tous les modèles\n")
    logger.info("Résultats dans : results/evaluations/descriptions/\n")

    tous_resultats = {}

    for config in MODELES:
        resultats = evaluer_modele(config)
        if resultats:
            tous_resultats[config["nom"]] = resultats

    if tous_resultats:
        generer_tableau_comparatif(tous_resultats)

    logger.info("\nÉvaluation terminée")
