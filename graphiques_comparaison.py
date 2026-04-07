"""
Ce script génère les graphiques de comparaison entre les trois modèles

On génère deux graphiques :
    comparaison_vitesse_accuracy.png    vitesse vs accuracy zero-shot
    comparaison_vitesse_encodage.png    vitesse vs frames encodées par seconde

Les fichiers sont sauvegardés dans results/graphiques/.

"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
import logging

# Affiche des logs pour suivre l'avancement du script et détecter d'éventuels problèmes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


RESULTS_DIR    = Path("results")
GRAPHIQUES_DIR = RESULTS_DIR / "graphiques"

# Chemins vers les fichiers produits par les scripts précédents
FICHIERS_METRIQUES = [
    RESULTS_DIR / "mobileclip" / "mobileclip_s2_metrics.json",
    RESULTS_DIR / "clip"       / "clip_metrics.json",
    RESULTS_DIR / "tinyclip"   / "tinyclip_metrics.json",
    RESULTS_DIR / "siglip"     / "siglip_metrics.json",
    RESULTS_DIR / "evaclip"    / "evaclip_metrics.json",
    RESULTS_DIR / "openclip"   / "openclip_metrics.json",
    RESULTS_DIR / "metaclip"   / "metaclip_metrics.json",
    RESULTS_DIR / "dfnclip"    / "dfnclip_metrics.json"
]

FICHIERS_ACCURACY = [
    RESULTS_DIR / "mobileclip" / "zeroshot" / "accuracy.json",
    RESULTS_DIR / "clip"       / "zeroshot" / "accuracy.json",
    RESULTS_DIR / "tinyclip"   / "zeroshot" / "accuracy.json",
    RESULTS_DIR / "siglip"     / "zeroshot" / "accuracy.json",
    RESULTS_DIR / "evaclip"    / "zeroshot" / "accuracy.json",
    RESULTS_DIR / "openclip"   / "zeroshot" / "accuracy.json",
    RESULTS_DIR / "metaclip"   / "zeroshot" / "accuracy.json",
    RESULTS_DIR / "dfnclip"    / "zeroshot" / "accuracy.json"
]

# Couleurs et nombre de paramètres de chaque modèle
# Les paramètres indiqués correspondent aux encodeurs image+texte combinés
INFOS_MODELES = {
    "MobileCLIP-S2" : {"couleur": "#E07B39", "params_M": 35,  "label_court": "MobileCLIP-S2"},
    "CLIP ViT-B/32" : {"couleur": "#4A6FD4", "params_M": 151, "label_court": "CLIP B/32"},
    "TinyCLIP"      : {"couleur": "#C45BAA", "params_M": 39,  "label_court": "TinyCLIP-39M"},
    "SigLIP"        : {"couleur": "#3DAA6E", "params_M": 86,  "label_court": "SigLIP-B/16"},
    "EVA-CLIP"      : {"couleur": "#D4A017", "params_M": 149, "label_court": "EVA-CLIP B/16"},
    "OpenCLIP"      : {"couleur": "#7B68EE", "params_M": 151, "label_court": "OpenCLIP B/32"},
    "MetaCLIP"      : {"couleur": "#E05C5C", "params_M": 151, "label_court": "MetaCLIP B/32"},
    "DFN-CLIP"      : {"couleur": "#2ABBE8", "params_M": 149, "label_court": "DFN-CLIP B/16"},
}


TAILLE_BASE = 400

def charger_json(chemin):
    p = Path(chemin)
    if not p.exists():
        logger.warning(f"Fichier introuvable : {chemin}")
        return None
    with open(p) as f:
        return json.load(f)

# Calcule la taille d'un point en fonction du nombre de paramètres du modèle.
def calculer_taille_point(params_M, max_params):
    return TAILLE_BASE * (params_M / max_params) ** 0.6

# Applique un style soigné au graphique pour le rendre plus lisible et esthétique
def style(ax, titre, xlabel, ylabel):
    ax.set_title(titre, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, color="gray", linewidth=0.7)
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=9)

# Ajoute une légende qui explique la taille des points en haut du graphique
def ajouter_legende_taille(ax, infos_modeles, max_params):
    legendes = []
    for nom, infos in infos_modeles.items():
        taille = calculer_taille_point(infos["params_M"], max_params)
        point  = ax.scatter(
            [], [],
            s=taille,
            color=infos["couleur"],
            alpha=0.85,
            edgecolors="white",
            linewidths=1.5,
            label=f"{infos['label_court']} ({infos['params_M']}M params)"
        )
        legendes.append(point)

    ax.legend(
        handles=legendes,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.9,
        edgecolor="#DDDDDD",
        fancybox=True
    )

# Graphique principal : vitesse d'encodage (axe X) vs accuracy zero-shot (axe Y).
def graphique_vitesse_accuracy():
    donnees = []

    for f_metriques, f_accuracy in zip(FICHIERS_METRIQUES, FICHIERS_ACCURACY):
        m = charger_json(f_metriques)
        a = charger_json(f_accuracy)
        if m is None or a is None:
            continue

        nom = m["model"]
        infos = None
        for cle, val in INFOS_MODELES.items():
            if cle in nom or nom in cle:
                infos = val
                break
        if infos is None:
            infos = {"couleur": "#999999", "params_M": 50, "label_court": nom}

        donnees.append({
            "nom"      : nom,
            "vitesse"  : m["frames_per_second"],
            "accuracy" : a["accuracy_pourcent"],
            "infos"    : infos
        })

    if not donnees:
        logger.warning("Pas de données suffisantes pour générer le graphique vitesse vs accuracy")
        return

    max_params = max(d["infos"]["params_M"] for d in donnees)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    style(
        ax,
        titre  = "Vitesse d'encodage vs Accuracy zero-shot — UCF-Crime",
        xlabel = "Vitesse d'encodage (frames / seconde)",
        ylabel = "Accuracy zero-shot (%)"
    )

    # On trie par vitesse pour relier les points avec une ligne dans l'ordre croissant
    donnees_tries = sorted(donnees, key=lambda d: d["vitesse"])

    xs = [d["vitesse"]  for d in donnees_tries]
    ys = [d["accuracy"] for d in donnees_tries]

    ax.plot(xs, ys, color="#AAAAAA", linewidth=1.2, zorder=1, linestyle="-", alpha=0.6)

    # On dessine chaque point individuellement pour avoir des couleurs différentes
    for d in donnees:
        taille = calculer_taille_point(d["infos"]["params_M"], max_params)
        ax.scatter(
            d["vitesse"], d["accuracy"],
            s=taille,
            color=d["infos"]["couleur"],
            alpha=0.88,
            edgecolors="white",
            linewidths=1.8,
            zorder=3
        )

        # On place le label légèrement décalé pour qu'il ne chevauche pas le point
        offset_x = d["vitesse"] * 0.02
        offset_y = 0.4
        ax.annotate(
            d["infos"]["label_court"],
            (d["vitesse"], d["accuracy"]),
            xytext=(d["vitesse"] + offset_x, d["accuracy"] + offset_y),
            fontsize=9,
            color=d["infos"]["couleur"],
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

    ajouter_legende_taille(ax, INFOS_MODELES, max_params)

    plt.tight_layout()
    chemin = GRAPHIQUES_DIR / "comparaison_vitesse_accuracy.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé dans {chemin}")


# Graphique secondaire : nombre de paramètres (axe X) vs accuracy (axe Y).
def graphique_params_accuracy():
    donnees = []

    for f_metriques, f_accuracy in zip(FICHIERS_METRIQUES, FICHIERS_ACCURACY):
        m = charger_json(f_metriques)
        a = charger_json(f_accuracy)
        if m is None or a is None:
            continue

        nom   = m["model"]
        infos = None
        for cle, val in INFOS_MODELES.items():
            if cle in nom or nom in cle:
                infos = val
                break
        if infos is None:
            infos = {"couleur": "#999999", "params_M": 50, "label_court": nom}

        donnees.append({
            "nom"      : nom,
            "params"   : infos["params_M"],
            "accuracy" : a["accuracy_pourcent"],
            "vitesse"  : m["frames_per_second"],
            "infos"    : infos
        })

    if not donnees:
        logger.warning("Pas de données suffisantes pour le graphique params vs accuracy")
        return

    max_vitesse = max(d["vitesse"] for d in donnees)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    style(
        ax,
        titre  = "Taille du modèle vs Accuracy zero-shot — UCF-Crime",
        xlabel = "Nombre de paramètres (millions)",
        ylabel = "Accuracy zero-shot (%)"
    )

    donnees_tries = sorted(donnees, key=lambda d: d["params"])
    xs = [d["params"]   for d in donnees_tries]
    ys = [d["accuracy"] for d in donnees_tries]
    ax.plot(xs, ys, color="#AAAAAA", linewidth=1.2, zorder=1, linestyle="-", alpha=0.6)

    for d in donnees:
        # La taille du point représente la vitesse d'encodage
        taille = TAILLE_BASE * (d["vitesse"] / max_vitesse) ** 0.6
        ax.scatter(
            d["params"], d["accuracy"],
            s=taille,
            color=d["infos"]["couleur"],
            alpha=0.88,
            edgecolors="white",
            linewidths=1.8,
            zorder=3
        )
        ax.annotate(
            d["infos"]["label_court"],
            (d["params"], d["accuracy"]),
            xytext=(d["params"] + max(xs) * 0.015, d["accuracy"] + 0.4),
            fontsize=9,
            color=d["infos"]["couleur"],
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

    ax.text(
        0.02, 0.02,
        "Taille des points proportionnelle à la vitesse d'encodage",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#999999",
        style="italic"
    )

    plt.tight_layout()
    chemin = GRAPHIQUES_DIR / "comparaison_params_accuracy.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé dans {chemin}")


if __name__ == "__main__":
    GRAPHIQUES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Génération des graphiques de comparaison...\n")

    graphique_vitesse_accuracy()
    graphique_params_accuracy()

    logger.info(f"\nGraphiques disponibles dans {GRAPHIQUES_DIR}")
