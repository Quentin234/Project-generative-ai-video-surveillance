"""
indexation_faiss.py

Ce script construit les index FAISS pour les trois modèles du projet.

Pour chaque modèle on va charger les embeddings produits par les scripts d'encodage,
construire un index FAISS, et le sauvegarder sur le disque avec ses métadonnées.

Fichiers produits dans results/<modele>/index/ :
    index.faiss     l'index FAISS qui contient tous les vecteurs
    metadata.csv    copie des métadonnées alignée avec l'index (même ordre)
    index_info.json informations sur l'index (dimension, nb vecteurs, type...)

"""

import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
import logging


# Affiche des messages d'information pour suivre la progression du script et détecter d'éventuels problèmes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# Dossier racine contenant les résultats des trois encodages
RESULTS_DIR = Path("results")

# Configuration des trois modèles : nom, dossier, préfixe des fichiers
MODELES = [
    {
        "nom"    : "MobileCLIP-S2",
        "dossier": RESULTS_DIR / "mobileclip",
        "prefixe": "mobileclip_s2"
    },
    {
        "nom"    : "CLIP ViT-B/32",
        "dossier": RESULTS_DIR / "clip",
        "prefixe": "clip"
    },
    {
        "nom"    : "TinyCLIP",
        "dossier": RESULTS_DIR / "tinyclip",
        "prefixe": "tinyclip"
    },
    {
        "nom"    : "SigLIP",
        "dossier": RESULTS_DIR / "siglip",
        "prefixe": "siglip"
    },
    {
        "nom"    : "EVA-CLIP",
        "dossier": RESULTS_DIR / "evaclip",
        "prefixe": "evaclip"
    },
    {
        "nom"    : "OpenCLIP",
        "dossier": RESULTS_DIR / "openclip",
        "prefixe": "openclip"
    },
    {
        "nom"    : "MetaCLIP",
        "dossier": RESULTS_DIR / "metaclip",
        "prefixe": "metaclip"
    },
    {
        "nom"    : "DFN-CLIP",
        "dossier": RESULTS_DIR / "dfnclip",
        "prefixe": "dfnclip"
    }
]


# Construit un index FAISS à partir d'une matrice d'embeddings
def construire_index(embeddings: np.ndarray) -> faiss.Index:

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # On enveloppe l'index dans un IndexIDMap pour pouvoir associer chaque vecteur à un identifiant numérique qui correspond à sa ligne dans le CSV de métadonnées.
    index_avec_ids = faiss.IndexIDMap(index)

    # Les IDs vont de 0 à N-1, un par frame, dans le même ordre que le CSV
    ids = np.arange(len(embeddings), dtype=np.int64)

    # On s'assure que les embeddings sont bien en float32
    embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

    index_avec_ids.add_with_ids(embeddings_f32, ids)

    return index_avec_ids

# Fonction principale qui traite un modèle : chargement des embeddings et métadonnées, construction de l'index, sauvegarde sur le disque
def indexer_modele(config: dict):
    nom     = config["nom"]
    dossier = config["dossier"]
    prefixe = config["prefixe"]

    logger.info(f"Traitement de {nom}...")

    # Vérification que les fichiers nécessaires existent bien
    embeddings_path = dossier / f"{prefixe}_embeddings.npy"
    metadata_path   = dossier / f"{prefixe}_metadata.csv"

    if not embeddings_path.exists():
        logger.warning(f"Fichier embeddings introuvable pour {nom} : {embeddings_path}, modèle ignoré")
        return

    if not metadata_path.exists():
        logger.warning(f"Fichier métadonnées introuvable pour {nom} : {metadata_path}, modèle ignoré")
        return

    # Chargement des embeddings produits par les scripts d'encodage
    logger.info(f"Chargement des embeddings depuis {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    logger.info(f"{len(embeddings):,} vecteurs chargés, dimension {embeddings.shape[1]}")

    # Chargement des métadonnées qui contiennent classe, vidéo et timestamp par frame
    metadata = pd.read_csv(metadata_path, index_col=0)

    # Vérification que le nombre de vecteurs correspond au nombre de lignes dans le CSV
    # Si ce n'est pas le cas il y a eu un problème lors de l'encodage
    if len(embeddings) != len(metadata):
        logger.error(
            f"Incohérence pour {nom} : {len(embeddings)} embeddings "
            f"mais {len(metadata)} lignes dans les métadonnées, modèle ignoré"
        )
        return

    # Construction de l'index FAISS
    logger.info("Construction de l'index FAISS...")
    index = construire_index(embeddings)
    logger.info(f"Index construit avec {index.ntotal} vecteurs")

    # Sauvegarde dans un sous-dossier index/ pour garder les choses organisées
    output_dir = dossier / "index"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde de l'index FAISS sur le disque
    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info(f"Index sauvegardé dans {index_path}")

    # Sauvegarde des métadonnées à côté de l'index pour qu'elles restent alignées
    metadata_output_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_output_path, index=True)
    logger.info(f"Métadonnées sauvegardées dans {metadata_output_path}")

    # Sauvegarde d'un fichier d'informations sur l'index
    index_info = {
        "modele"         : nom,
        "nb_vecteurs"    : int(index.ntotal),
        "dimension"      : int(embeddings.shape[1]),
        "type_index"     : "IndexIDMap(IndexFlatIP)",
        "normalisation"  : "L2",
        "embeddings_src" : str(embeddings_path),
        "metadata_src"   : str(metadata_path)
    }

    info_path = output_dir / "index_info.json"
    with open(info_path, "w") as f:
        json.dump(index_info, f, indent=2)
    logger.info(f"Informations de l'index sauvegardées dans {info_path}")

    logger.info(f"Indexation de {nom} terminée\n")

# Fonction de test pour vérifier que l'index fonctionne correctement en effectuant une recherche de proximité
def tester_index(config: dict):

    nom     = config["nom"]
    dossier = config["dossier"]
    prefixe = config["prefixe"]

    index_path    = dossier / "index" / "index.faiss"
    metadata_path = dossier / "index" / "metadata.csv"

    if not index_path.exists():
        logger.warning(f"Index introuvable pour {nom}, test ignoré")
        return

    logger.info(f"Test de l'index {nom}...")

    index    = faiss.read_index(str(index_path))
    metadata = pd.read_csv(metadata_path, index_col=0)

    # On charge le premier embedding comme vecteur de requête
    embeddings = np.load(dossier / f"{prefixe}_embeddings.npy")
    requete    = np.ascontiguousarray(embeddings[0:1], dtype=np.float32)

    # Recherche des 5 frames les plus similaires
    scores, ids = index.search(requete, k=5)

    logger.info(f"Résultat du test pour {nom} :")
    for rang, (score, id_frame) in enumerate(zip(scores[0], ids[0])):
        frame_info = metadata.iloc[id_frame]
        logger.info(
            f"  rang {rang + 1} : score {score:.4f} "
            f"{frame_info['class']} "
            f"{frame_info['video']} "
            f"t={frame_info['timestamp']}s"
        )

    logger.info("")


if __name__ == "__main__":

    logger.info("Démarrage de l'indexation FAISS pour les trois modèles\n")

    # Indexation des trois modèles
    for config in MODELES:
        indexer_modele(config)

    # Test rapide de chaque index pour vérifier que tout fonctionne
    logger.info("Tests de vérification des index...")
    for config in MODELES:
        tester_index(config)

    logger.info("Indexation terminée pour tous les modèles")
