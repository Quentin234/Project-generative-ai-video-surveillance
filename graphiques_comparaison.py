import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
GRAPHIQUES_DIR = RESULTS_DIR / "graphiques"

MODELS = ["mobileclip", "clip", "tinyclip", "siglip", "evaclip", "openclip", "metaclip", "dfnclip"]
FICHIERS_METRIQUES = [RESULTS_DIR / m / f"{m}_metrics.json" for m in MODELS]
FICHIERS_ACCURACY = [RESULTS_DIR / m / "zeroshot" / "accuracy.json" for m in MODELS]

# Utilisation des paramètres de l'ENCODEUR IMAGE uniquement
INFOS_MODELES = {
    "mobileclip": {"couleur": "#E07B39", "label": "MobileCLIP", "params": 26},
    "tinyclip":   {"couleur": "#C45BAA", "label": "TinyCLIP",   "params": 39},
    "metaclip":   {"couleur": "#E05C5C", "label": "MetaCLIP",   "params": 88},
    "siglip":     {"couleur": "#3DAA6E", "label": "SigLIP",     "params": 86},
    "eva":        {"couleur": "#D4A017", "label": "EVA-CLIP",   "params": 86},
    "dfn":        {"couleur": "#2ABBE8", "label": "DFN-CLIP",   "params": 86},
    "laion2b":    {"couleur": "#7B68EE", "label": "OpenCLIP",   "params": 88},
    "clip":       {"couleur": "#4A6FD4", "label": "CLIP B/32",  "params": 88},
}

def charger_json(chemin):
    if not chemin.exists(): return None
    with open(chemin) as f: return json.load(f)

def style_graph(ax, titre, xlabel, ylabel):
    ax.set_title(titre, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("#F8F9FA")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

def generer_graphiques():
    donnees = []
    for folder, f_met, f_acc in zip(MODELS, FICHIERS_METRIQUES, FICHIERS_ACCURACY):
        m, a = charger_json(f_met), charger_json(f_acc)
        if not m or not a: continue
        
        nom_brut = m["model"].lower()
        conf = next((v for k, v in INFOS_MODELES.items() if k in nom_brut), None)
        if not conf or (conf['label'] == "CLIP B/32" and folder != "clip"):
            conf = INFOS_MODELES.get(folder)

        donnees.append({
            "vitesse": m["frames_per_second"],
            "accuracy": a["accuracy_pourcent"],
            "params": conf["params"],
            "couleur": conf["couleur"],
            "label": conf["label"]
        })

    # --- Graph 1 : Vitesse vs Accuracy ---
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    style_graph(ax1, "Vitesse d'encodage vs Accuracy (Zero-shot)", "Frames / seconde", "Accuracy (%)")
    d_v = sorted(donnees, key=lambda x: x["vitesse"])
    ax1.plot([x["vitesse"] for x in d_v], [x["accuracy"] for x in d_v], color="#CCC", zorder=1)
    for d in donnees:
        ax1.scatter(d["vitesse"], d["accuracy"], s=250, color=d["couleur"], edgecolors="white", zorder=3, label=d["label"])
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8)
    plt.savefig(GRAPHIQUES_DIR / "comparaison_vitesse_accuracy.png", dpi=150, bbox_inches="tight")

    # --- Graph 2 : Paramètres Image vs Accuracy ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    style_graph(ax2, "Taille Image Encoder vs Accuracy (Zero-shot)", "Paramètres Image (Millions)", "Accuracy (%)")
    d_p = sorted(donnees, key=lambda x: x["params"])
    ax2.plot([x["params"] for x in d_p], [x["accuracy"] for x in d_p], color="#CCC", zorder=1)
    
    max_v = max(x["vitesse"] for x in donnees)
    for d in donnees:
        # La taille du point montre la vitesse relative
        s_v = 50 + (d["vitesse"] / max_v) * 500
        ax2.scatter(d["params"], d["accuracy"], s=s_v, color=d["couleur"], edgecolors="white", zorder=3, label=d["label"])
    
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8)
    plt.savefig(GRAPHIQUES_DIR / "comparaison_params_accuracy.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    GRAPHIQUES_DIR.mkdir(parents=True, exist_ok=True)
    generer_graphiques()
    print("Graphiques mis à jour avec les paramètres de l'encodeur image uniquement.")