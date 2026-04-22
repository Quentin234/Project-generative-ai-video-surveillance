"""
classification_zeroshot_topk.py

Ce script évalue la classification zero-shot avec 12 variantes, obtenues en
combinant 4 modes d'agrégation (granularité des données) et 3 niveaux de top-k.

Les 4 modes d'agrégation :
────────────────────────────────────────────────────────────────────────────────
  frame       Évaluation frame par frame. Chaque frame est classifiée
              indépendamment. C'est la baseline.

  video_mean  Évaluation par vidéo via moyenne des scores. On moyenne les
              matrices de scores (nb_frames x 13) de toutes les frames d'une
              vidéo, et la classe avec le meilleur score moyen est la prédiction
              de la vidéo.

  video_best  Évaluation par vidéo via la frame la plus confiante. Pour chaque
              vidéo, on sélectionne la frame dont le score max (toutes classes)
              est le plus élevé, c'est-à-dire la frame sur laquelle le modèle
              est le plus certain. Sa prédiction top-1 représente la vidéo.

  video_vote  Évaluation par vidéo via vote majoritaire. Chaque frame vote pour
              sa classe top-1 (ou top-k pour les variantes top-3 et top-5). La
              classe qui obtient le plus de votes parmi toutes les frames de la
              vidéo est retenue comme prédiction de la vidéo.

Les 3 niveaux de top-k :
────────────────────────────────────────────────────────────────────────────────
  top1        La prédiction est correcte uniquement si la classe prédite en
              premier est la bonne classe.

  top3        La prédiction est correcte si la bonne classe figure parmi les
              3 classes ayant les scores les plus élevés.

  top5        La prédiction est correcte si la bonne classe figure parmi les
              5 classes ayant les scores les plus élevés.

  Note sur top-k et vote : pour video_vote en top-3 ou top-5, chaque frame
  distribue un vote à chacune de ses k meilleures classes (vote pondéré par
  le rang : 3 points pour le 1er, 2 pour le 2ème, 1 pour le 3ème). La classe
  avec le total le plus élevé gagne.

On utilise la méthode d'embedding moyen (desc_4_mean_embed) qui a été choisie
comme meilleure stratégie de description texte lors de l'évaluation précédente.

Résultats sauvegardés dans results/evaluations/topk/<mode>_top<k>/ :
    <modele>_accuracy.json      accuracy globale, par classe (et par vidéo si applicable)
    <modele>_confusion.png      matrice de confusion (au niveau où l'évaluation se fait)

Un tableau comparatif global est sauvegardé dans :
    results/evaluations/topk/comparaison_topk.csv
    results/evaluations/topk/comparaison_topk.png

Structure attendue :
    results/<modele>/<prefixe>_embeddings.npy
    results/<modele>/<prefixe>_metadata.csv
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
EVAL_DIR    = RESULTS_DIR / "evaluations" / "topk"

CLASSES_DATASET = [
    "Fighting", "Robbery", "Vandalism", "Arrest",
    "Burglary", "Shooting", "Abuse", "Arson", "Assault",
    "RoadAccidents", "Shoplifting", "Stealing", "Explosion"
]

# Cinq descriptions par classe — méthode mean_embed choisie comme meilleure
DESCRIPTIONS_MULTI = {
    "Fighting": [
        "a photo of a person fighting",
        "two or more people engaging in a mutual physical fight",
        "opponents punching and kicking each other simultaneously",
        "a brawl where multiple parties are actively hitting one another",
        "a reciprocal physical altercation between individuals"
    ],
    "Robbery": [
        "a photo of a robbery",
        "someone robbing a person by using immediate force or a weapon",
        "a victim being threatened with a weapon to hand over valuables",
        "an aggressive mugging where a person is held up",
        "stealing from a person through direct confrontation and intimidation"
    ],
    "Vandalism": [
        "a photo of vandalism",
        "someone intentionally destroying or defacing public property",
        "a person spray painting graffiti on a wall or building",
        "breaking windows, mirrors or damaging structures",
        "deliberate destruction of property caught on camera"
    ],
    "Arrest": [
        "a photo of a person being arrested",
        "police officers in uniform apprehending a suspect",
        "law enforcement officers placing handcuffs on an individual",
        "a suspect being detained and led away by police",
        "official police intervention to take a person into custody"
    ],
    "Burglary": [
        "a photo of a burglary",
        "someone breaking into a closed building or house to steal",
        "a person forcing entry through a door or window of a property",
        "illegal entry into a home or business, usually while unoccupied",
        "a burglar sneaking into a private premises to commit theft"
    ],
    "Shooting": [
        "a photo of a person shooting",
        "someone discharging a firearm or handgun",
        "a person aiming and firing a gun at a target",
        "active gunfire and muzzle flashes from a weapon",
        "an individual using a firearm in a violent incident"
    ],
    "Abuse": [
        "a photo of an abuse",
        "a person physically mistreating a child, elderly, or defenseless person",
        "someone inflicting repetitive harm on a vulnerable individual",
        "physical violence against a victim who is not fighting back",
        "domestic or interpersonal violence caught on camera"
    ],
    "Arson": [
        "a photo of an arson",
        "a person intentionally starting a fire with fuel or a lighter",
        "someone deliberately setting fire to a building or vehicle",
        "the act of lighting a structure on fire to cause destruction",
        "intentional ignition of a fire caught on surveillance"
    ],
    "Assault": [
        "a photo of an assault",
        "one person suddenly attacking or beating a victim",
        "a violent physical attack where the victim is being overpowered",
        "an aggressor punching or hitting a person who is retreating",
        "a person being physically assaulted by an attacker"
    ],
    "RoadAccidents": [
        "a photo of a road accident",
        "a car crash occurring in the middle of a road or intersection",
        "vehicles colliding during traffic flow",
        "a traffic mishap involving moving cars or pedestrians on a street",
        "motor vehicle accident caught on a roadway camera"
    ],
    "Shoplifting": [
        "a photo of shoplifting",
        "a customer stealing merchandise inside a retail store",
        "someone concealing store items in their clothes or bag",
        "taking products from shop shelves without paying at the counter",
        "theft committed during business hours inside a commercial shop"
    ],
    "Stealing": [
        "a photo of a person stealing",
        "someone taking an unattended object that does not belong to them",
        "theft of a bag, bicycle, or item from a public space",
        "a person picking up and walking away with someone else's property",
        "non-violent theft of belongings in an open environment"
    ],
    "Explosion": [
        "a photo of an explosion",
        "a sudden release of energy causing damage",
        "a violent rupture or burst of an object or structure",
        "the force of an explosion caught on camera",
        "a dramatic and destructive event involving fire or debris"
    ]
}

MODELES = [
    {
        "nom"    : "SigLIP",
        "dossier": RESULTS_DIR / "siglip",
        "prefixe": "siglip",
        "type"   : "siglip"
    },
    {
        "nom"    : "TinyCLIP",
        "dossier": RESULTS_DIR / "tinyclip",
        "prefixe": "tinyclip",
        "type"   : "tinyclip"
    },
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

# Couleurs par modèle pour les graphiques
COULEURS_MODELES = {
    "MobileCLIP-S2" : "#E07B39",
    "CLIP ViT-B/32" : "#4A6FD4",
    "SigLIP"        : "#3DAA6E",
    "EVA-CLIP"      : "#D4A017",
    "OpenCLIP"      : "#7B68EE",
    "MetaCLIP"      : "#E05C5C",
    "DFN-CLIP"      : "#2ABBE8",
}


# ─────────────────────────────────────────────────────────────────────────────
# ENCODEURS TEXTE
# Même logique que dans eval_descriptions.py
# ─────────────────────────────────────────────────────────────────────────────

def encoder_textes_mobileclip(textes, device):
    import mobileclip
    model, _, _ = mobileclip.create_model_and_transforms(
        "mobileclip_s2", pretrained=str(Path("checkpoints/mobileclip_s2.pt"))
    )
    model = model.to(device).eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s2")
    tokens    = tokenizer(textes).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def encoder_textes_clip(textes, device):
    import clip
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    tokens = clip.tokenize(textes).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def encoder_textes_tinyclip(textes, device):
    from transformers import CLIPModel, CLIPProcessor
    model_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
    processor  = CLIPProcessor.from_pretrained(model_name)
    model      = CLIPModel.from_pretrained(model_name).to(device).eval()
    inputs     = processor(text=textes, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def encoder_textes_siglip(textes, device):
    from transformers import AutoModel, AutoProcessor
    model_name = "google/siglip-base-patch16-224"
    processor  = AutoProcessor.from_pretrained(model_name)
    model      = AutoModel.from_pretrained(model_name).to(device).eval()
    inputs     = processor(text=textes, return_tensors="pt", padding="max_length",
                           max_length=64, truncation=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def encoder_textes_openclip(textes, model_name, model_pretrained, device):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrained)
    model     = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    tokens    = tokenizer(textes).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def encoder_textes(textes, config, device):
    """Dispatch vers le bon encodeur texte selon le type du modèle."""
    t = config["type"]
    if t == "mobileclip":
        return encoder_textes_mobileclip(textes, device)
    elif t == "clip":
        return encoder_textes_clip(textes, device)
    elif t == "tinyclip":
        return encoder_textes_tinyclip(textes, device)
    elif t == "siglip":
        return encoder_textes_siglip(textes, device)
    elif t == "openclip":
        return encoder_textes_openclip(textes, config["model_name"], config["model_pretrained"], device)
    else:
        raise ValueError(f"Type de modèle inconnu : {t}")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DE LA MATRICE TEXTE (mean_embed)
# ─────────────────────────────────────────────────────────────────────────────

def construire_matrice_mean_embed(config, device):
    """
    Encode les 5 descriptions de chaque classe et moyenne leurs embeddings.
    On renormalise le vecteur moyen. C'est la méthode desc_4_mean_embed,
    choisie comme meilleure stratégie lors de l'évaluation précédente.
    Retourne une matrice (13 x dim).
    """
    vecteurs = []
    for classe in CLASSES_DATASET:
        emb   = encoder_textes(DESCRIPTIONS_MULTI[classe], config, device)
        moyen = emb.mean(axis=0)
        moyen = moyen / np.linalg.norm(moyen)
        vecteurs.append(moyen)
    return np.stack(vecteurs).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DE L'ACCURACY TOP-K
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_topk(scores_matrix, vraies_classes, k):
    """
    Calcule l'accuracy top-k globale et par classe.

    Pour chaque entrée (frame ou vidéo), la prédiction est considérée correcte
    si la vraie classe figure parmi les k classes ayant les scores les plus élevés.

    Args:
        scores_matrix : np.ndarray (N x 13) — scores bruts
        vraies_classes : list de str (len N) — vraies classes
        k : int — 1, 3 ou 5

    Returns:
        accuracy_globale : float
        accuracy_par_classe : dict str -> float
        predictions_top1 : list de str (prédictions top-1, pour la matrice de confusion)
    """
    # Indices triés par score décroissant pour chaque entrée
    topk_idx = np.argsort(scores_matrix, axis=1)[:, ::-1][:, :k]  # (N x k)

    predictions_top1 = [CLASSES_DATASET[topk_idx[i, 0]] for i in range(len(vraies_classes))]

    correct = 0
    for i, vraie in enumerate(vraies_classes):
        vraie_idx = CLASSES_DATASET.index(vraie)
        if vraie_idx in topk_idx[i]:
            correct += 1

    accuracy_globale = correct / len(vraies_classes)

    # Accuracy par classe
    accuracy_par_classe = {}
    for classe in CLASSES_DATASET:
        indices = [i for i, v in enumerate(vraies_classes) if v == classe]
        if not indices:
            continue
        classe_idx = CLASSES_DATASET.index(classe)
        nb_correct = sum(1 for i in indices if classe_idx in topk_idx[i])
        accuracy_par_classe[classe] = round(nb_correct / len(indices), 4)

    return accuracy_globale, accuracy_par_classe, predictions_top1


# ─────────────────────────────────────────────────────────────────────────────
# 4 MODES D'AGRÉGATION
# Chacun produit une matrice de scores et une liste de vraies classes
# au niveau d'évaluation correspondant (frame ou vidéo).
# ─────────────────────────────────────────────────────────────────────────────

def aggreger_frame(scores_frame, metadata):
    """
    Mode frame : on évalue chaque frame individuellement.
    Retourne les scores tels quels et les vraies classes frame par frame.
    """
    vraies_classes = metadata["class"].tolist()
    return scores_frame, vraies_classes


def aggreger_video_mean(scores_frame, metadata):
    """
    Mode video_mean : pour chaque vidéo on moyenne les scores de toutes ses frames.
    La matrice retournée est (nb_videos x 13), une ligne par vidéo.

    Avantage : toutes les frames contribuent de façon égale à la décision finale.
    """
    metadata = metadata.reset_index(drop=True)
    groupes  = metadata.groupby("video")

    scores_video    = []
    vraies_classes  = []

    for video_nom, groupe in groupes:
        indices = groupe.index.tolist()
        # Moyenne des scores sur toutes les frames de la vidéo
        score_moyen = scores_frame[indices].mean(axis=0)
        scores_video.append(score_moyen)
        vraies_classes.append(groupe["class"].iloc[0])

    return np.array(scores_video, dtype=np.float32), vraies_classes


def aggreger_video_best(scores_frame, metadata):
    """
    Mode video_best : pour chaque vidéo, on sélectionne la frame dont le modèle
    est le plus certain (score max le plus élevé parmi toutes les classes).
    Cette frame représente à elle seule la vidéo.

    Avantage : évite le bruit des frames ambiguës en se concentrant sur la frame
    la plus informative selon le modèle.
    """
    metadata = metadata.reset_index(drop=True)
    groupes  = metadata.groupby("video")

    scores_video   = []
    vraies_classes = []

    for video_nom, groupe in groupes:
        indices = groupe.index.tolist()
        scores_frames_video = scores_frame[indices]
        # Confiance = score max toutes classes pour chaque frame
        confiances = scores_frames_video.max(axis=1)
        meilleure  = np.argmax(confiances)
        scores_video.append(scores_frames_video[meilleure])
        vraies_classes.append(groupe["class"].iloc[0])

    return np.array(scores_video, dtype=np.float32), vraies_classes


def aggreger_video_vote(scores_frame, metadata, k):
    """
    Mode video_vote : chaque frame vote pour ses k meilleures classes avec
    un système de points pondérés par le rang (k points pour le 1er, k-1 pour
    le 2ème, ..., 1 pour le k-ème). La classe avec le total le plus élevé gagne.

    En top-1 c'est un vote simple (chaque frame vote pour sa meilleure classe).
    En top-3 / top-5 chaque frame distribue plusieurs votes pondérés.

    Retourne une matrice (nb_videos x 13) où chaque colonne est le total des
    points reçus par la classe correspondante, ainsi que les vraies classes.
    """
    metadata = metadata.reset_index(drop=True)
    groupes  = metadata.groupby("video")

    scores_video   = []
    vraies_classes = []

    for video_nom, groupe in groupes:
        indices = groupe.index.tolist()
        scores_frames_video = scores_frame[indices]

        # Rang des classes par score décroissant pour chaque frame
        topk_idx = np.argsort(scores_frames_video, axis=1)[:, ::-1][:, :k]

        # Accumulation des points pondérés par le rang
        # Le 1er reçoit k points, le 2ème k-1, ..., le k-ème 1 point
        votes = np.zeros(len(CLASSES_DATASET), dtype=np.float32)
        for frame_topk in topk_idx:
            for rang, classe_idx in enumerate(frame_topk):
                votes[classe_idx] += k - rang

        scores_video.append(votes)
        vraies_classes.append(groupe["class"].iloc[0])

    return np.array(scores_video, dtype=np.float32), vraies_classes


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE DES RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────

def sauvegarder_resultats(scores_eval, vraies_classes, nom_modele, mode, k, output_dir):
    """
    Calcule l'accuracy top-k, génère la matrice de confusion et sauvegarde tout.

    Args:
        scores_eval   : matrice (N x 13) au niveau d'évaluation (frame ou vidéo)
        vraies_classes: liste de vraies classes (len N)
        nom_modele    : str
        mode          : str parmi {frame, video_mean, video_best, video_vote}
        k             : int parmi {1, 3, 5}
        output_dir    : Path — dossier de sortie

    Returns:
        accuracy_globale : float
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_globale, accuracy_par_classe, predictions_top1 = accuracy_topk(
        scores_eval, vraies_classes, k
    )

    logger.info(f"  {nom_modele:15s} | {mode:12s} | top-{k} : {accuracy_globale * 100:.2f}%")

    # JSON avec les métriques
    nom_fichier = nom_modele.lower().replace(" ", "_").replace("/", "_")
    resultats   = {
        "modele"             : nom_modele,
        "mode_agregation"    : mode,
        "top_k"              : k,
        "accuracy_globale"   : round(accuracy_globale, 4),
        "accuracy_pourcent"  : round(accuracy_globale * 100, 2),
        "nb_evaluations"     : len(vraies_classes),
        "accuracy_par_classe": accuracy_par_classe
    }

    with open(output_dir / f"{nom_fichier}_accuracy.json", "w") as f:
        json.dump(resultats, f, indent=2)

    # Matrice de confusion (toujours basée sur les prédictions top-1)
    cm   = confusion_matrix(vraies_classes, predictions_top1, labels=CLASSES_DATASET)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_DATASET)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)

    niveau = "frame" if mode == "frame" else "vidéo"
    ax.set_title(
        f"Matrice de confusion — {nom_modele}\n"
        f"Mode : {mode} | Top-{k} | Accuracy top-1 : {accuracy_globale * 100:.2f}% "
        f"(évalué au niveau {niveau})",
        fontsize=10, pad=15
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{nom_fichier}_confusion.png", dpi=130, bbox_inches="tight")
    plt.close()

    return accuracy_globale


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION D'UN MODÈLE — 12 COMBINAISONS
# ─────────────────────────────────────────────────────────────────────────────

def evaluer_modele(config):
    """
    Lance les 12 évaluations (4 modes x 3 top-k) pour un modèle.
    Retourne un dictionnaire {(mode, k): accuracy}.
    """
    nom     = config["nom"]
    dossier = config["dossier"]
    prefixe = config["prefixe"]

    embeddings_path = dossier / f"{prefixe}_embeddings.npy"
    metadata_path   = dossier / f"{prefixe}_metadata.csv"

    if not embeddings_path.exists() or not metadata_path.exists():
        logger.warning(f"Fichiers introuvables pour {nom}, modèle ignoré")
        return {}

    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata   = pd.read_csv(metadata_path, index_col=0)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"\n{'─'*60}")
    logger.info(f"{nom} — {len(embeddings):,} frames | device : {device}")
    logger.info(f"{'─'*60}")

    # Encodage des labels texte (une seule fois par modèle)
    logger.info("  Encodage des embeddings texte (mean_embed)...")
    matrice_texte = construire_matrice_mean_embed(config, device)
    logger.info(f"  Matrice texte shape : {matrice_texte.shape}")

    # Scores bruts au niveau frame : (nb_frames x 13)
    scores_frame = embeddings @ matrice_texte.T
    logger.info(f"  Scores frame shape : {scores_frame.shape}")

    resultats = {}

    for k in [1, 3, 5]:
        for mode in ["frame", "video_mean", "video_best", "video_vote"]:

            # Agrégation selon le mode
            if mode == "frame":
                scores_eval, vraies = aggreger_frame(scores_frame, metadata)
            elif mode == "video_mean":
                scores_eval, vraies = aggreger_video_mean(scores_frame, metadata)
            elif mode == "video_best":
                scores_eval, vraies = aggreger_video_best(scores_frame, metadata)
            elif mode == "video_vote":
                # Pour vote, k influe sur combien de classes votent par frame
                scores_eval, vraies = aggreger_video_vote(scores_frame, metadata, k)

            # Dossier de sortie : results/evaluations/topk/<mode>_top<k>/
            output_dir = EVAL_DIR / f"{mode}_top{k}"

            acc = sauvegarder_resultats(scores_eval, vraies, nom, mode, k, output_dir)
            resultats[(mode, k)] = acc

    return resultats


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES COMPARATIFS GLOBAUX
# ─────────────────────────────────────────────────────────────────────────────

def generer_tableau_comparatif(tous_resultats):
    """
    Génère un CSV récapitulatif et plusieurs graphiques de comparaison.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    modes = ["frame", "video_mean", "video_best", "video_vote"]
    ks    = [1, 3, 5]

    # ── 1. CSV récapitulatif ──────────────────────────────────────────────────
    colonnes = [(mode, k) for k in ks for mode in modes]
    lignes   = []
    for nom_modele, resultats in tous_resultats.items():
        ligne = {"modele": nom_modele}
        for (mode, k) in colonnes:
            cle         = f"{mode}_top{k}"
            ligne[cle]  = round(resultats.get((mode, k), 0) * 100, 2)
        lignes.append(ligne)

    df = pd.DataFrame(lignes).set_index("modele")
    df.to_csv(EVAL_DIR / "comparaison_topk.csv")
    logger.info(f"\nTableau comparatif sauvegardé dans {EVAL_DIR / 'comparaison_topk.csv'}")

    # ── 2. Graphique 1 : accuracy par mode, organisé par top-k ───────────────
    # Une figure avec 3 colonnes (top-1, top-3, top-5), une barre par modèle
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
    fig.suptitle(
        "Comparaison des 12 méthodes de classification zero-shot — UCF-Crime\n"
        "(méthode de description : embedding moyen de 5 descriptions)",
        fontsize=13, fontweight="bold", y=1.01
    )

    labels_modes = {
        "frame"      : "Frame par frame",
        "video_mean" : "Vidéo — moyenne",
        "video_best" : "Vidéo — meilleure frame",
        "video_vote" : "Vidéo — vote pondéré"
    }
    motifs = ["///", "\\\\\\", "xxx", "..."]

    for col_idx, k in enumerate(ks):
        ax = axes[col_idx]

        modeles_noms = list(tous_resultats.keys())
        x = np.arange(len(modeles_noms))
        n_modes  = len(modes)
        largeur  = 0.7 / n_modes

        for mode_idx, mode in enumerate(modes):
            offsets = (mode_idx - n_modes / 2 + 0.5) * largeur
            vals    = [tous_resultats[nom].get((mode, k), 0) * 100 for nom in modeles_noms]
            couleurs_barres = [COULEURS_MODELES.get(nom, "#999999") for nom in modeles_noms]

            barres = ax.bar(
                x + offsets, vals,
                width=largeur,
                label=labels_modes[mode],
                color=couleurs_barres,
                alpha=0.75 if mode_idx > 0 else 0.95,
                edgecolor="white",
                linewidth=0.8,
                hatch=motifs[mode_idx] if mode_idx > 0 else None
            )

            for barre, val in zip(barres, vals):
                if val > 0:
                    ax.text(
                        barre.get_x() + barre.get_width() / 2,
                        barre.get_height() + 0.3,
                        f"{val:.1f}",
                        ha="center", va="bottom", fontsize=5.5, color="#333333"
                    )

        ax.set_title(f"Top-{k}", fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(modeles_noms, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy (%)" if col_idx == 0 else "")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_facecolor("#F8F9FA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Légende des modes en bas de la figure
    patches = [
        mpatches.Patch(facecolor="#888888", alpha=0.95, label=labels_modes["frame"]),
        mpatches.Patch(facecolor="#888888", alpha=0.75, hatch="///", label=labels_modes["video_mean"]),
        mpatches.Patch(facecolor="#888888", alpha=0.75, hatch="\\\\\\", label=labels_modes["video_best"]),
        mpatches.Patch(facecolor="#888888", alpha=0.75, hatch="xxx", label=labels_modes["video_vote"])
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=4,
        fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04)
    )

    plt.tight_layout()
    plt.savefig(EVAL_DIR / "comparaison_topk_par_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique par top-k sauvegardé dans {EVAL_DIR / 'comparaison_topk_par_k.png'}")

    # ── 3. Graphique 2 : accuracy par modèle, progressions top-1/3/5 ─────────
    # Une courbe par mode pour chaque modèle, x = top-k
    n_modeles = len(tous_resultats)
    n_cols    = 4
    n_rows    = (n_modeles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharey=False)
    axes_flat = axes.flatten() if n_modeles > 1 else [axes]
    fig.suptitle(
        "Progression top-1 → top-5 par modèle et par mode d'agrégation",
        fontsize=13, fontweight="bold", y=1.01
    )

    couleurs_modes = {
        "frame"      : "#4A6FD4",
        "video_mean" : "#E07B39",
        "video_best" : "#3DAA6E",
        "video_vote" : "#C45BAA"
    }
    marqueurs_modes = {
        "frame"      : "o",
        "video_mean" : "s",
        "video_best" : "^",
        "video_vote" : "D"
    }

    for ax_idx, (nom_modele, resultats) in enumerate(tous_resultats.items()):
        ax = axes_flat[ax_idx]

        for mode in modes:
            vals = [resultats.get((mode, k), 0) * 100 for k in ks]
            ax.plot(
                ks, vals,
                color=couleurs_modes[mode],
                marker=marqueurs_modes[mode],
                linewidth=2,
                markersize=6,
                label=labels_modes[mode]
            )
            # Annoter le point top-5
            ax.annotate(
                f"{vals[-1]:.1f}%",
                (5, vals[-1]),
                xytext=(5.1, vals[-1]),
                fontsize=7,
                color=couleurs_modes[mode]
            )

        ax.set_title(nom_modele, fontsize=10, fontweight="bold",
                     color=COULEURS_MODELES.get(nom_modele, "#333333"))
        ax.set_xticks([1, 3, 5])
        ax.set_xticklabels(["Top-1", "Top-3", "Top-5"], fontsize=8)
        ax.set_ylabel("Accuracy (%)", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_facecolor("#F8F9FA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_idx == 0:
            ax.legend(fontsize=7, framealpha=0.9, loc="upper left")

    # Masquer les axes vides si le nombre de modèles n'est pas multiple de n_cols
    for ax_idx in range(len(tous_resultats), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(EVAL_DIR / "comparaison_topk_progression.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique progression sauvegardé dans {EVAL_DIR / 'comparaison_topk_progression.png'}")

    # ── 4. Graphique 3 : heatmap accuracy (modèle x méthode) ─────────────────
    methodes_labels = [f"{mode}\ntop-{k}" for k in ks for mode in modes]
    methodes_cles   = [(mode, k) for k in ks for mode in modes]

    data_heatmap = np.array([
        [tous_resultats[nom].get(cle, 0) * 100 for cle in methodes_cles]
        for nom in tous_resultats
    ])

    fig, ax = plt.subplots(figsize=(18, max(5, n_modeles * 0.7 + 2)))
    im = ax.imshow(data_heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)

    ax.set_xticks(range(len(methodes_cles)))
    ax.set_xticklabels(methodes_labels, fontsize=8, ha="center")
    ax.set_yticks(range(len(tous_resultats)))
    ax.set_yticklabels(list(tous_resultats.keys()), fontsize=9)

    # Séparateurs visuels entre les top-k
    for sep in [3.5, 7.5]:
        ax.axvline(sep, color="white", linewidth=2.5)

    # Texte dans chaque cellule
    for i in range(len(tous_resultats)):
        for j in range(len(methodes_cles)):
            val = data_heatmap[i, j]
            couleur_txt = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color=couleur_txt, fontweight="bold")

    # Étiquettes des groupes top-k en haut
    for g_idx, (label_g, x_centre) in enumerate(
        [("Top-1", 1.5), ("Top-3", 5.5), ("Top-5", 9.5)]
    ):
        ax.text(x_centre, -0.9, label_g, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333",
                transform=ax.get_xaxis_transform())

    ax.set_title(
        "Heatmap accuracy (%) — 12 méthodes × 8 modèles\n"
        "Séparateurs verticaux = changement de top-k",
        fontsize=11, pad=25
    )

    plt.tight_layout()
    plt.savefig(EVAL_DIR / "comparaison_topk_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap sauvegardée dans {EVAL_DIR / 'comparaison_topk_heatmap.png'}")

    # ── 5. Résumé console ────────────────────────────────────────────────────
    logger.info("\n── Résumé global (accuracy %) ──")
    df_print = df.copy()
    logger.info(df_print.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Évaluation zero-shot — 12 méthodes (4 modes × 3 top-k)")
    logger.info("Méthode de description : embedding moyen de 5 descriptions (mean_embed)")
    logger.info(f"Résultats dans : {EVAL_DIR}\n")

    tous_resultats = {}

    for config in MODELES:
        resultats = evaluer_modele(config)
        if resultats:
            tous_resultats[config["nom"]] = resultats

    if tous_resultats:
        logger.info("\n\nGénération des graphiques comparatifs...")
        generer_tableau_comparatif(tous_resultats)

    logger.info("\nÉvaluation terminée")
