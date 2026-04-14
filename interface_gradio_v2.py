"""
interface_gradio_v2.py

Interface Gradio en 4 onglets pour explorer le dataset UCF-Crime avec des modèles CLIP.

Tab 1 – Recherche par frames
    L'utilisateur saisit une requête texte, le système retourne les N frames
    individuelles les plus similaires via l'index FAISS du modèle choisi.
    (comportement de l'interface originale)

Tab 2 – Recherche par vidéos
    Pour chaque vidéo du dataset, on agrège les scores FAISS de toutes ses frames
    (moyenne des top-K scores par vidéo). On retourne les M vidéos les plus
    pertinentes avec leur frame la plus représentative.

Tab 3 – Recherche par séquences
    On parcourt chaque vidéo avec une fenêtre glissante sur les embeddings de ses
    frames. Le score d'une fenêtre est la moyenne des scores de ses frames.
    On retourne les segments continus les plus similaires à la requête.

Tab 4 – Graphique de similarité d'une vidéo
    L'utilisateur choisit une vidéo du dataset. Pour chaque frame, on calcule
    la similarité avec les 13 classes via l'embedding moyen de 5 descriptions
    (prompt ensembling, méthode recommandée par le papier CLIP original).
    On trace les 13 courbes de similarité temporelle avec un marqueur de la
    vraie classe de la vidéo.
"""

import numpy as np
import pandas as pd
import faiss
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import gradio as gr
import logging
import mobileclip
import clip
import open_clip
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR     = Path("results")
CHECKPOINT_PATH = Path("checkpoints/mobileclip_s2.pt")

MODELES_DISPO = {
    "MobileCLIP-S2": {
        "dossier": RESULTS_DIR / "mobileclip",
        "prefixe": "mobileclip_s2",
        "type"   : "mobileclip"
    },
    "CLIP ViT-B/32": {
        "dossier": RESULTS_DIR / "clip",
        "prefixe": "clip",
        "type"   : "clip"
    },
    "TinyCLIP": {
        "dossier": RESULTS_DIR / "tinyclip",
        "prefixe": "tinyclip",
        "type"   : "tinyclip"
    },
    "SigLIP": {
        "dossier": RESULTS_DIR / "siglip",
        "prefixe": "siglip",
        "type"   : "siglip"
    },
    "EVA-CLIP": {
        "dossier"         : RESULTS_DIR / "evaclip",
        "prefixe"         : "evaclip",
        "type"            : "openclip",
        "model_name"      : "EVA02-B-16",
        "model_pretrained": "merged2b_s8b_b131k"
    },
    "OpenCLIP": {
        "dossier"         : RESULTS_DIR / "openclip",
        "prefixe"         : "openclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-32",
        "model_pretrained": "laion2b_s34b_b79k"
    },
    "MetaCLIP": {
        "dossier"         : RESULTS_DIR / "metaclip",
        "prefixe"         : "metaclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-32-quickgelu",
        "model_pretrained": "metaclip_400m"
    },
    "DFN-CLIP": {
        "dossier"         : RESULTS_DIR / "dfnclip",
        "prefixe"         : "dfnclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-16",
        "model_pretrained": "dfn2b"
    }
}

# 13 classes avec leurs 5 descriptions (prompt ensembling, méthode desc_4_mean_embed)
CLASSES_DATASET = [
    "Fighting", "Robbery", "Accident", "Vandalism", "Arrest",
    "Burglary", "Shooting", "Abuse", "Arson", "Assault",
    "RoadAccidents", "Shoplifting", "Stealing"
]

DESCRIPTIONS_MULTI = {
    "Fighting": [
        "a person fighting",
        "two or more people engaging in a mutual physical fight",
        "opponents punching and kicking each other simultaneously",
        "a brawl where multiple parties are actively hitting one another",
        "a reciprocal physical altercation between individuals"
    ],
    "Robbery": [
        "a robbery",
        "someone robbing a person by using immediate force or a weapon",
        "a victim being threatened with a weapon to hand over valuables",
        "an aggressive mugging where a person is held up",
        "stealing from a person through direct confrontation and intimidation"
    ],
    "Accident": [
        "a car accident",
        "a vehicle collision involving cars, trucks or motorcycles",
        "cars crashing into stationary objects or other vehicles",
        "a sudden traffic collision on a street or highway",
        "impact between motor vehicles resulting in damage"
    ],
    "Vandalism": [
        "vandalism",
        "someone intentionally destroying or defacing public property",
        "a person spray painting graffiti on a wall or building",
        "breaking windows, mirrors or damaging structures",
        "deliberate destruction of property caught on camera"
    ],
    "Arrest": [
        "a person being arrested",
        "police officers in uniform apprehending a suspect",
        "law enforcement officers placing handcuffs on an individual",
        "a suspect being detained and led away by police",
        "official police intervention to take a person into custody"
    ],
    "Burglary": [
        "a burglary",
        "someone breaking into a closed building or house to steal",
        "a person forcing entry through a door or window of a property",
        "illegal entry into a home or business, usually while unoccupied",
        "a burglar sneaking into a private premises to commit theft"
    ],
    "Shooting": [
        "a person shooting",
        "someone discharging a firearm or handgun",
        "a person aiming and firing a gun at a target",
        "active gunfire and muzzle flashes from a weapon",
        "an individual using a firearm in a violent incident"
    ],
    "Abuse": [
        "an abuse",
        "a person physically mistreating a child, elderly, or defenseless person",
        "someone inflicting repetitive harm on a vulnerable individual",
        "physical violence against a victim who is not fighting back",
        "domestic or interpersonal violence caught on camera"
    ],
    "Arson": [
        "an arson",
        "a person intentionally starting a fire with fuel or a lighter",
        "someone deliberately setting fire to a building or vehicle",
        "the act of lighting a structure on fire to cause destruction",
        "intentional ignition of a fire caught on surveillance"
    ],
    "Assault": [
        "an assault",
        "one person suddenly attacking or beating a victim",
        "a violent physical attack where the victim is being overpowered",
        "an aggressor punching or hitting a person who is retreating",
        "a person being physically assaulted by an attacker"
    ],
    "RoadAccidents": [
        "a road accident",
        "a car crash occurring in the middle of a road or intersection",
        "vehicles colliding during traffic flow",
        "a traffic mishap involving moving cars or pedestrians on a street",
        "motor vehicle accident caught on a roadway camera"
    ],
    "Shoplifting": [
        "shoplifting",
        "a customer stealing merchandise inside a retail store",
        "someone concealing store items in their clothes or bag",
        "taking products from shop shelves without paying at the counter",
        "theft committed during business hours inside a commercial shop"
    ],
    "Stealing": [
        "a person stealing",
        "someone taking an unattended object that does not belong to them",
        "theft of a bag, bicycle, or item from a public space",
        "a person picking up and walking away with someone else's property",
        "non-violent theft of belongings in an open environment"
    ]
}

# Couleurs pour les 13 classes dans le graphique de similarité (tab 4)
COULEURS_CLASSES = [
    "#E05C5C", "#4A6FD4", "#3DAA6E", "#E07B39", "#C45BAA",
    "#2ABBE8", "#D4A017", "#7B68EE", "#5FAD8E", "#D45B8A",
    "#8BC34A", "#FF7043", "#607D8B"
]


# ─────────────────────────────────────────────────────────────────────────────
# CACHE (modèles, index FAISS, embeddings bruts pour les tabs 3 et 4)
# ─────────────────────────────────────────────────────────────────────────────

_cache_encodeurs   = {}   # nom_modele -> tuple (type, model, extra, device)
_cache_index       = {}   # nom_modele -> (faiss_index, metadata_df)
_cache_embeddings  = {}   # nom_modele -> (embeddings np.array, metadata_df)
_cache_class_embed = {}   # nom_modele -> matrice (13 x dim) des embeddings de classe


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES ET INDEX
# ─────────────────────────────────────────────────────────────────────────────

def charger_encodeur(nom_modele: str):
    """Charge et met en cache l'encodeur texte du modèle demandé."""
    if nom_modele in _cache_encodeurs:
        return _cache_encodeurs[nom_modele]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg    = MODELES_DISPO[nom_modele]
    type_m = cfg["type"]
    logger.info(f"Chargement de l'encodeur texte pour {nom_modele}...")

    if type_m == "mobileclip":
        model, _, _ = mobileclip.create_model_and_transforms(
            "mobileclip_s2", pretrained=str(CHECKPOINT_PATH)
        )
        model = model.to(device).eval()
        extra = mobileclip.get_tokenizer("mobileclip_s2")
        _cache_encodeurs[nom_modele] = ("mobileclip", model, extra, device)

    elif type_m == "clip":
        model, _ = clip.load("ViT-B/32", device=device)
        model.eval()
        _cache_encodeurs[nom_modele] = ("clip", model, None, device)

    elif type_m == "siglip":
        hf_name   = "google/siglip-base-patch16-224"
        processor = AutoProcessor.from_pretrained(hf_name)
        model     = AutoModel.from_pretrained(hf_name).to(device).eval()
        _cache_encodeurs[nom_modele] = ("siglip", model, processor, device)

    elif type_m == "openclip":
        model, _, _ = open_clip.create_model_and_transforms(
            cfg["model_name"], pretrained=cfg["model_pretrained"]
        )
        model     = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(cfg["model_name"])
        _cache_encodeurs[nom_modele] = ("openclip", model, tokenizer, device)

    else:  # tinyclip
        hf_name   = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
        processor = CLIPProcessor.from_pretrained(hf_name)
        model     = CLIPModel.from_pretrained(hf_name).to(device).eval()
        _cache_encodeurs[nom_modele] = ("tinyclip", model, processor, device)

    logger.info(f"Encodeur chargé pour {nom_modele}")
    return _cache_encodeurs[nom_modele]


def charger_index(nom_modele: str):
    """Charge et met en cache l'index FAISS + métadonnées du modèle."""
    if nom_modele in _cache_index:
        return _cache_index[nom_modele]

    cfg     = MODELES_DISPO[nom_modele]
    dossier = cfg["dossier"]
    idx_p   = dossier / "index" / "index.faiss"
    meta_p  = dossier / "index" / "metadata.csv"

    if not idx_p.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable pour {nom_modele} ({idx_p}).\n"
            "Lance d'abord indexation_faiss.py."
        )

    logger.info(f"Chargement de l'index FAISS pour {nom_modele}...")
    index    = faiss.read_index(str(idx_p))
    metadata = pd.read_csv(meta_p, index_col=0)
    _cache_index[nom_modele] = (index, metadata)
    logger.info(f"Index chargé : {index.ntotal:,} vecteurs")
    return _cache_index[nom_modele]


def charger_embeddings_bruts(nom_modele: str):
    """
    Charge et met en cache la matrice d'embeddings bruts et les métadonnées.
    Utilisé par les tabs 3 et 4 qui ont besoin d'accéder aux embeddings par vidéo.
    """
    if nom_modele in _cache_embeddings:
        return _cache_embeddings[nom_modele]

    cfg     = MODELES_DISPO[nom_modele]
    emb_p   = cfg["dossier"] / f"{cfg['prefixe']}_embeddings.npy"
    meta_p  = cfg["dossier"] / f"{cfg['prefixe']}_metadata.csv"

    if not emb_p.exists():
        raise FileNotFoundError(
            f"Embeddings introuvables pour {nom_modele} ({emb_p}).\n"
            "Lance d'abord le script d'encodage correspondant."
        )

    logger.info(f"Chargement des embeddings bruts pour {nom_modele}...")
    embeddings = np.load(emb_p).astype(np.float32)
    metadata   = pd.read_csv(meta_p, index_col=0)
    _cache_embeddings[nom_modele] = (embeddings, metadata)
    logger.info(f"Embeddings chargés : {embeddings.shape}")
    return _cache_embeddings[nom_modele]


# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE TEXTE
# ─────────────────────────────────────────────────────────────────────────────

def encoder_texte(texte_ou_liste, encodeur):
    """
    Encode un texte ou une liste de textes avec l'encodeur donné.
    Retourne un tableau numpy (N x dim) normalisé.
    """
    type_m, model, extra, device = encodeur
    textes = [texte_ou_liste] if isinstance(texte_ou_liste, str) else texte_ou_liste

    with torch.no_grad():
        if type_m == "mobileclip":
            tokens    = extra(textes).to(device)
            emb       = model.encode_text(tokens)

        elif type_m == "clip":
            tokens = clip.tokenize(textes).to(device)
            emb    = model.encode_text(tokens)

        elif type_m == "siglip":
            inputs = extra(
                text=textes, return_tensors="pt",
                padding="max_length", max_length=64, truncation=True
            ).to(device)
            emb = model.get_text_features(**inputs)

        elif type_m == "openclip":
            tokens = extra(textes).to(device)
            emb    = model.encode_text(tokens)

        else:  # tinyclip
            inputs = extra(text=textes, return_tensors="pt", padding=True).to(device)
            emb    = model.get_text_features(**inputs)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output

        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().float().numpy()


def encoder_requete_unique(texte: str, nom_modele: str) -> np.ndarray:
    """Encode une requête utilisateur en un vecteur (1 x dim)."""
    encodeur = charger_encodeur(nom_modele)
    return encoder_texte(texte, encodeur)


def obtenir_embeddings_classes(nom_modele: str) -> np.ndarray:
    """
    Construit et met en cache la matrice de classe (13 x dim) par prompt ensembling :
    pour chaque classe, on encode ses 5 descriptions et on moyenne leurs embeddings
    avant de renormaliser — méthode desc_4_mean_embed recommandée par le papier CLIP.
    """
    if nom_modele in _cache_class_embed:
        return _cache_class_embed[nom_modele]

    logger.info(f"Construction des embeddings de classe (mean_embed) pour {nom_modele}...")
    encodeur = charger_encodeur(nom_modele)
    vecteurs = []

    for classe in CLASSES_DATASET:
        emb   = encoder_texte(DESCRIPTIONS_MULTI[classe], encodeur)  # (5 x dim)
        moyen = emb.mean(axis=0)
        moyen = moyen / np.linalg.norm(moyen)
        vecteurs.append(moyen)

    matrice = np.stack(vecteurs).astype(np.float32)
    _cache_class_embed[nom_modele] = matrice
    logger.info(f"Matrice de classe construite : {matrice.shape}")
    return matrice


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RECHERCHE PAR FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def recherche_frames(requete: str, nom_modele: str, nb_resultats: int):
    """
    Recherche FAISS directe : retourne les N frames les plus proches du vecteur requête.
    Identique à l'interface d'origine.
    """
    if not requete.strip():
        return [], "Veuillez saisir une requête texte."

    try:
        vecteur          = encoder_requete_unique(requete, nom_modele)
        index, metadata  = charger_index(nom_modele)
        scores, ids      = index.search(vecteur, k=int(nb_resultats))

        images, lignes = [], []
        for rang, (score, id_frame) in enumerate(zip(scores[0], ids[0])):
            if id_frame < 0 or id_frame >= len(metadata):
                continue
            info   = metadata.iloc[id_frame]
            chemin = Path(info["filepath"])
            images.append(
                Image.open(chemin).convert("RGB") if chemin.exists()
                else Image.new("RGB", (224, 224), (80, 80, 80))
            )
            lignes.append(
                f"#{rang+1}  sim={score:.3f}  "
                f"classe={info['class']}  vidéo={info['video']}  t={info['timestamp']}s"
            )

        resume = (
            f"{len(images)} frames trouvées avec {nom_modele} "
            f"pour « {requete} »"
        )
        return images, resume + "\n\n" + "\n".join(lignes)

    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        logger.error(f"Erreur tab 1 : {e}")
        return [], f"Erreur : {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RECHERCHE PAR VIDÉOS
# ─────────────────────────────────────────────────────────────────────────────

def recherche_videos(requete: str, nom_modele: str, nb_resultats: int, top_frames_par_video: int):
    """
    Pour chaque vidéo du dataset, calcule un score d'agrégation :
        score_video = moyenne des top-K scores FAISS de ses frames.
    Retourne les M vidéos les plus pertinentes avec leur frame la plus représentative.

    Stratégie :
      1. On encode la requête et on lance une recherche FAISS large (10 000 résultats)
         pour couvrir toutes les vidéos potentiellement pertinentes.
      2. On regroupe les scores par vidéo et on calcule la moyenne des top-K.
      3. On trie et retourne les M meilleures vidéos.
    """
    if not requete.strip():
        return [], "Veuillez saisir une requête texte."

    try:
        vecteur         = encoder_requete_unique(requete, nom_modele)
        index, metadata = charger_index(nom_modele)

        # On demande suffisamment de résultats pour couvrir toutes les vidéos
        k_large = min(index.ntotal, 50_000)
        scores_flat, ids_flat = index.search(vecteur, k=k_large)

        # Regroupement par vidéo : liste des (score, id_frame) pour chaque vidéo
        scores_par_video  = defaultdict(list)
        meilleure_frame   = {}   # video_key -> (score, id_frame) de la frame la plus haute

        for score, id_frame in zip(scores_flat[0], ids_flat[0]):
            if id_frame < 0 or id_frame >= len(metadata):
                continue
            info      = metadata.iloc[id_frame]
            video_key = (info["class"], info["video"])
            scores_par_video[video_key].append((float(score), int(id_frame)))

            # On garde la frame ayant le meilleur score pour la miniature
            if video_key not in meilleure_frame or score > meilleure_frame[video_key][0]:
                meilleure_frame[video_key] = (float(score), int(id_frame))

        # Calcul du score de chaque vidéo : moyenne des top-K frames
        scores_videos = {}
        for video_key, liste in scores_par_video.items():
            liste.sort(key=lambda x: x[0], reverse=True)
            top_k   = liste[:top_frames_par_video]
            scores_videos[video_key] = sum(s for s, _ in top_k) / len(top_k)

        # Tri des vidéos par score décroissant
        videos_triees = sorted(scores_videos.items(), key=lambda x: x[1], reverse=True)
        videos_triees = videos_triees[:int(nb_resultats)]

        images, lignes = [], []
        for rang, ((classe, video), score_video) in enumerate(videos_triees):
            # Frame représentative = frame avec le score individuel le plus élevé
            _, id_best = meilleure_frame[(classe, video)]
            info_best  = metadata.iloc[id_best]
            chemin     = Path(info_best["filepath"])

            images.append(
                Image.open(chemin).convert("RGB") if chemin.exists()
                else Image.new("RGB", (224, 224), (80, 80, 80))
            )
            nb_frames_video = len(scores_par_video[(classe, video)])
            lignes.append(
                f"#{rang+1}  score_moyen={score_video:.3f}  "
                f"classe={classe}  vidéo={video}  "
                f"({nb_frames_video} frames dans le dataset)"
            )

        resume = (
            f"{len(images)} vidéos trouvées avec {nom_modele} "
            f"pour « {requete} » (agrégation top-{top_frames_par_video} frames)"
        )
        return images, resume + "\n\n" + "\n".join(lignes)

    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        logger.error(f"Erreur tab 2 : {e}")
        return [], f"Erreur : {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RECHERCHE PAR SÉQUENCES (fenêtre glissante)
# ─────────────────────────────────────────────────────────────────────────────

def recherche_sequences(
    requete       : str,
    nom_modele    : str,
    nb_resultats  : int,
    taille_fenetre: int
):
    """
    Applique une fenêtre glissante sur les embeddings de chaque vidéo.
    Le score d'une fenêtre = similarité cosinus moyenne entre le vecteur requête
    et les embeddings de ses frames.

    Retourne les N meilleurs segments avec leur frame centrale comme miniature,
    et l'intervalle de timestamps [t_debut, t_fin].
    """
    if not requete.strip():
        return [], "Veuillez saisir une requête texte."

    try:
        vecteur                   = encoder_requete_unique(requete, nom_modele)   # (1 x dim)
        vecteur_1d                = vecteur[0]  # (dim,)
        embeddings, metadata      = charger_embeddings_bruts(nom_modele)

        # On calcule la similarité cosinus de chaque frame avec la requête en une seule opération
        # Forme : (nb_frames_total,) — on a accès direct à tous les embeddings
        sims_toutes = embeddings @ vecteur_1d  # dot product, équivalent cosinus car normalisé

        # Regroupement des frames par vidéo, dans l'ordre temporel (par timestamp)
        videos = defaultdict(list)  # video_key -> liste de (timestamp, idx_global)
        for idx, row in metadata.iterrows():
            video_key = (row["class"], row["video"])
            # idx est l'index pandas qui correspond à la position dans la matrice embeddings
            videos[video_key].append((row["timestamp"], int(idx)))

        # Tri par timestamp pour respecter l'ordre temporel au sein de chaque vidéo
        for key in videos:
            videos[key].sort(key=lambda x: x[0])

        # Fenêtre glissante sur chaque vidéo
        meilleurs_segments = []  # liste de (score, classe, video, debut_ts, fin_ts, idx_frame_centrale)

        for (classe, video), frames_triees in videos.items():
            nb = len(frames_triees)
            if nb < taille_fenetre:
                # Vidéo trop courte : on prend toutes les frames comme unique fenêtre
                indices_globaux = [idx for _, idx in frames_triees]
                score   = float(sims_toutes[indices_globaux].mean())
                debut   = frames_triees[0][0]
                fin     = frames_triees[-1][0]
                idx_mid = indices_globaux[len(indices_globaux) // 2]
                meilleurs_segments.append((score, classe, video, debut, fin, idx_mid))
                continue

            for i in range(nb - taille_fenetre + 1):
                fenetre         = frames_triees[i : i + taille_fenetre]
                indices_globaux = [idx for _, idx in fenetre]
                score           = float(sims_toutes[indices_globaux].mean())
                debut           = fenetre[0][0]
                fin             = fenetre[-1][0]
                # Frame centrale pour la miniature
                idx_mid         = indices_globaux[taille_fenetre // 2]
                meilleurs_segments.append((score, classe, video, debut, fin, idx_mid))

        # Tri par score décroissant, puis déduplication par vidéo + position
        # (on garde le meilleur segment par vidéo pour éviter les doublons trop proches)
        meilleurs_segments.sort(key=lambda x: x[0], reverse=True)

        # Déduplication : on ne garde qu'un segment par vidéo dans les résultats finaux
        # (commentez ce bloc si vous voulez plusieurs segments par vidéo)
        vus     = set()
        filtres = []
        for seg in meilleurs_segments:
            _, classe, video, debut, fin, _ = seg
            cle = (classe, video)
            if cle not in vus:
                vus.add(cle)
                filtres.append(seg)
            if len(filtres) >= int(nb_resultats):
                break

        images, lignes = [], []
        for rang, (score, classe, video, debut, fin, idx_mid) in enumerate(filtres):
            info   = metadata.iloc[idx_mid]
            chemin = Path(info["filepath"])
            images.append(
                Image.open(chemin).convert("RGB") if chemin.exists()
                else Image.new("RGB", (224, 224), (80, 80, 80))
            )
            lignes.append(
                f"#{rang+1}  score={score:.3f}  "
                f"classe={classe}  vidéo={video}  "
                f"t={debut}s → t={fin}s  (fenêtre {taille_fenetre} frames)"
            )

        resume = (
            f"{len(images)} séquences trouvées avec {nom_modele} "
            f"pour « {requete} » (fenêtre glissante de {taille_fenetre} frames)"
        )
        return images, resume + "\n\n" + "\n".join(lignes)

    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        logger.error(f"Erreur tab 3 : {e}")
        return [], f"Erreur : {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — GRAPHIQUE DE SIMILARITÉ D'UNE VIDÉO
# ─────────────────────────────────────────────────────────────────────────────

def lister_videos(nom_modele: str):
    """
    Retourne la liste de toutes les vidéos disponibles pour le modèle choisi,
    sous la forme "Classe / NomVidéo" triées alphabétiquement.
    Utilisée pour peupler le menu déroulant du tab 4.
    """
    try:
        _, metadata = charger_embeddings_bruts(nom_modele)
        paires      = metadata[["class", "video"]].drop_duplicates()
        return sorted(f"{r['class']} / {r['video']}" for _, r in paires.iterrows())
    except Exception:
        return []


def graphique_similarite_video(
    video_selection: str,
    nom_modele     : str,
    classes_visibles: list
):
    """
    Pour la vidéo sélectionnée, trace la similarité cosinus de chaque frame
    avec les 13 classes (via embedding moyen des 5 descriptions).

    Retourne une figure matplotlib avec :
      - Une courbe par classe sélectionnée (tracé temporel frame par frame)
      - Un fond coloré indiquant la vraie classe de la vidéo
      - Une ligne verticale pour marquer le début de l'action si disponible
    """
    if not video_selection:
        return None, "Sélectionnez une vidéo dans la liste."

    try:
        embeddings, metadata = charger_embeddings_bruts(nom_modele)
        matrice_classes      = obtenir_embeddings_classes(nom_modele)  # (13 x dim)

        # Extraction de classe et nom de vidéo depuis la sélection "Classe / NomVidéo"
        parties = video_selection.split(" / ", maxsplit=1)
        if len(parties) != 2:
            return None, "Format de sélection invalide."
        classe_video, nom_video = parties[0].strip(), parties[1].strip()

        # Filtrage des frames de cette vidéo
        masque     = (metadata["class"] == classe_video) & (metadata["video"] == nom_video)
        idx_frames = metadata.index[masque].tolist()

        if not idx_frames:
            return None, f"Aucune frame trouvée pour {video_selection}."

        # Extraction et tri par timestamp
        frames_meta = metadata.loc[idx_frames].copy()
        frames_meta = frames_meta.sort_values("timestamp")
        idx_tries   = frames_meta.index.tolist()

        emb_video  = embeddings[idx_tries]        # (nb_frames x dim)
        timestamps = frames_meta["timestamp"].tolist()

        # Calcul des similarités : (nb_frames x 13)
        sims = emb_video @ matrice_classes.T

        # Construction du graphique
        fig, ax = plt.subplots(figsize=(14, 6))

        idx_vraie_classe = CLASSES_DATASET.index(classe_video) if classe_video in CLASSES_DATASET else -1

        # Fond coloré pour indiquer la vraie classe
        ax.axhspan(
            sims.min() - 0.02, sims.max() + 0.02,
            alpha=0.04,
            color=COULEURS_CLASSES[idx_vraie_classe] if idx_vraie_classe >= 0 else "#CCCCCC",
            label=f"_fond"
        )

        # Tracé de chaque classe sélectionnée
        classes_a_tracer = classes_visibles if classes_visibles else CLASSES_DATASET
        for i, classe in enumerate(CLASSES_DATASET):
            if classe not in classes_a_tracer:
                continue
            lw         = 2.5 if classe == classe_video else 1.0
            ls         = "-"  if classe == classe_video else "--"
            alpha      = 0.95 if classe == classe_video else 0.55
            label_txt  = f"★ {classe} (vraie classe)" if classe == classe_video else classe
            ax.plot(
                timestamps, sims[:, i],
                color=COULEURS_CLASSES[i],
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                label=label_txt,
                zorder=3 if classe == classe_video else 2
            )

        # Ligne horizontale à 0 pour référence
        ax.axhline(0, color="#BBBBBB", linewidth=0.7, linestyle=":")

        ax.set_xlabel("Timestamp (secondes)", fontsize=11)
        ax.set_ylabel("Similarité cosinus", fontsize=11)
        ax.set_title(
            f"Similarité par classe — {nom_video}  [{classe_video}]  |  {nom_modele}\n"
            f"({len(timestamps)} frames, embeddings de classe : moyenne de 5 descriptions)",
            fontsize=11, pad=12
        )
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.grid(True, linestyle="--", alpha=0.35, color="gray")
        ax.set_facecolor("#F8F9FA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Légende en deux colonnes pour ne pas surcharger
        ax.legend(
            loc="upper right",
            fontsize=8,
            ncol=2,
            framealpha=0.92,
            edgecolor="#DDDDDD",
            fancybox=True
        )

        plt.tight_layout()

        # Résumé textuel
        sim_vraie_classe = sims[:, idx_vraie_classe] if idx_vraie_classe >= 0 else np.zeros(len(timestamps))
        resume = (
            f"Vidéo : {nom_video}  |  Vraie classe : {classe_video}  |  "
            f"{len(timestamps)} frames  |  Modèle : {nom_modele}\n"
            f"Sim. moy. vraie classe : {sim_vraie_classe.mean():.3f}  |  "
            f"Sim. max. vraie classe : {sim_vraie_classe.max():.3f}"
        )
        return fig, resume

    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        logger.error(f"Erreur tab 4 : {e}")
        return None, f"Erreur : {e}"


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS DE MISE À JOUR DES LISTES DÉROULANTES
# ─────────────────────────────────────────────────────────────────────────────

def maj_liste_videos(nom_modele: str):
    """
    Appelé quand l'utilisateur change de modèle dans le tab 4.
    Recharge la liste des vidéos disponibles pour ce modèle.
    """
    videos = lister_videos(nom_modele)
    if videos:
        return gr.update(choices=videos, value=videos[0])
    return gr.update(choices=[], value=None)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DE L'INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def construire_interface():
    noms_modeles = list(MODELES_DISPO.keys())

    with gr.Blocks(title="UCF-Crime — Recherche CLIP", theme=gr.themes.Soft()) as interface:

        gr.Markdown(
            "# 🎥 Recherche vidéo UCF-Crime — Modèles CLIP\n"
            "Explorez le dataset UCF-Crime à l'aide de requêtes textuelles "
            "et comparez les résultats de 8 variantes de modèles CLIP."
        )

        # ── TAB 1 : FRAMES ─────────────────────────────────────────────────
        with gr.Tab("🖼️ Frames"):
            gr.Markdown(
                "**Recherche par frames individuelles** — "
                "Retourne les N frames du dataset les plus similaires à votre requête."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    t1_requete  = gr.Textbox(
                        label="Requête texte",
                        placeholder="Ex : person fighting, car accident, robbery...",
                        lines=1
                    )
                with gr.Column(scale=1):
                    t1_modele   = gr.Dropdown(
                        choices=noms_modeles, value="CLIP ViT-B/32", label="Modèle"
                    )
            with gr.Row():
                t1_nb       = gr.Slider(1, 24, value=6, step=1, label="Nombre de frames")
                t1_btn      = gr.Button("Rechercher", variant="primary")

            t1_resume   = gr.Textbox(label="Résumé", lines=4, interactive=False)
            t1_galerie  = gr.Gallery(label="Frames", columns=4, height=520)

            t1_btn.click(
                fn=recherche_frames,
                inputs=[t1_requete, t1_modele, t1_nb],
                outputs=[t1_galerie, t1_resume]
            )
            t1_requete.submit(
                fn=recherche_frames,
                inputs=[t1_requete, t1_modele, t1_nb],
                outputs=[t1_galerie, t1_resume]
            )

            gr.Markdown(
                "**Exemples :** `person fighting` | `car crash on road` | "
                "`robbery in store` | `person with gun` | `fire and smoke`"
            )

        # ── TAB 2 : VIDÉOS ─────────────────────────────────────────────────
        with gr.Tab("🎞️ Vidéos"):
            gr.Markdown(
                "**Recherche par vidéos** — "
                "Agrège les scores de toutes les frames de chaque vidéo "
                "(moyenne des top-K meilleures frames). "
                "La miniature affichée est la frame la plus similaire de la vidéo."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    t2_requete  = gr.Textbox(
                        label="Requête texte",
                        placeholder="Ex : person fighting, car accident...",
                        lines=1
                    )
                with gr.Column(scale=1):
                    t2_modele   = gr.Dropdown(
                        choices=noms_modeles, value="CLIP ViT-B/32", label="Modèle"
                    )
            with gr.Row():
                t2_nb       = gr.Slider(1, 20, value=6, step=1, label="Nombre de vidéos")
                t2_topk     = gr.Slider(1, 30, value=5, step=1,
                                        label="Top-K frames par vidéo pour l'agrégation")
                t2_btn      = gr.Button("Rechercher", variant="primary")

            t2_resume   = gr.Textbox(label="Résumé", lines=5, interactive=False)
            t2_galerie  = gr.Gallery(
                label="Vidéos (frame la plus représentative)", columns=3, height=520
            )

            t2_btn.click(
                fn=recherche_videos,
                inputs=[t2_requete, t2_modele, t2_nb, t2_topk],
                outputs=[t2_galerie, t2_resume]
            )
            t2_requete.submit(
                fn=recherche_videos,
                inputs=[t2_requete, t2_modele, t2_nb, t2_topk],
                outputs=[t2_galerie, t2_resume]
            )

        # ── TAB 3 : SÉQUENCES ──────────────────────────────────────────────
        with gr.Tab("🎬 Séquences"):
            gr.Markdown(
                "**Recherche par séquences** — "
                "Fenêtre glissante sur les frames de chaque vidéo. "
                "Retourne les segments continus les plus similaires à la requête "
                "avec leur intervalle de timestamps `[t_début → t_fin]`."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    t3_requete  = gr.Textbox(
                        label="Requête texte",
                        placeholder="Ex : person fighting, fire and smoke...",
                        lines=1
                    )
                with gr.Column(scale=1):
                    t3_modele   = gr.Dropdown(
                        choices=noms_modeles, value="CLIP ViT-B/32", label="Modèle"
                    )
            with gr.Row():
                t3_nb       = gr.Slider(1, 20, value=6, step=1, label="Nombre de séquences")
                t3_fenetre  = gr.Slider(2, 30, value=5, step=1,
                                        label="Taille de la fenêtre (frames)")
                t3_btn      = gr.Button("Rechercher", variant="primary")

            t3_resume   = gr.Textbox(label="Résumé", lines=6, interactive=False)
            t3_galerie  = gr.Gallery(
                label="Séquences (frame centrale de la fenêtre)", columns=3, height=520
            )

            t3_btn.click(
                fn=recherche_sequences,
                inputs=[t3_requete, t3_modele, t3_nb, t3_fenetre],
                outputs=[t3_galerie, t3_resume]
            )
            t3_requete.submit(
                fn=recherche_sequences,
                inputs=[t3_requete, t3_modele, t3_nb, t3_fenetre],
                outputs=[t3_galerie, t3_resume]
            )

        # ── TAB 4 : GRAPHIQUE DE SIMILARITÉ ────────────────────────────────
        with gr.Tab("📊 Similarité par classe"):
            gr.Markdown(
                "**Graphique de similarité temporelle** — "
                "Sélectionnez une vidéo du dataset. Le graphique trace, frame par frame, "
                "la similarité cosinus avec chacune des 13 classes. "
                "L'embedding de classe est calculé par **prompt ensembling** : "
                "moyenne des vecteurs de 5 descriptions par classe (méthode `mean_embed` "
                "recommandée par le papier CLIP original). "
                "La vraie classe de la vidéo est mise en avant (trait plein, ★)."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    t4_modele   = gr.Dropdown(
                        choices=noms_modeles, value="CLIP ViT-B/32", label="Modèle"
                    )
                with gr.Column(scale=3):
                    # La liste des vidéos est initialisée au premier chargement
                    videos_init = lister_videos("CLIP ViT-B/32")
                    t4_video    = gr.Dropdown(
                        choices=videos_init,
                        value=videos_init[0] if videos_init else None,
                        label="Vidéo (Classe / NomVidéo)"
                    )

            # Filtre sur les classes à afficher (toutes par défaut)
            t4_classes  = gr.CheckboxGroup(
                choices=CLASSES_DATASET,
                value=CLASSES_DATASET,
                label="Classes à afficher sur le graphique"
            )

            t4_btn      = gr.Button("Tracer le graphique", variant="primary")
            t4_resume   = gr.Textbox(label="Résumé", lines=2, interactive=False)
            t4_graphe   = gr.Plot(label="Similarité temporelle par classe")

            # Mise à jour de la liste des vidéos quand on change de modèle
            t4_modele.change(
                fn=maj_liste_videos,
                inputs=[t4_modele],
                outputs=[t4_video]
            )

            t4_btn.click(
                fn=graphique_similarite_video,
                inputs=[t4_video, t4_modele, t4_classes],
                outputs=[t4_graphe, t4_resume]
            )

    return interface


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Démarrage de l'interface Gradio v2 (4 onglets)...")
    interface = construire_interface()
    interface.launch(share=False)
