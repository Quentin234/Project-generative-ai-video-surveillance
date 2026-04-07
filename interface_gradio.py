"""
L'utilisateur écrit une description en texte (par exemple "person fighting")
et le système retrouve les frames du dataset qui correspondent le mieux
à cette description en cherchant dans l'index FAISS du modèle choisi.

"""

import numpy as np
import pandas as pd
import faiss
import torch
from pathlib import Path
from PIL import Image
import gradio as gr
import logging
import mobileclip
import clip
import open_clip
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor


# Affiche des logs pour suivre l'avancement du script et détecter d'éventuels problèmes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


RESULTS_DIR     = Path("results")
CHECKPOINT_PATH = Path("checkpoints/mobileclip_s2.pt")

# Configuration des trois modèles
MODELES_DISPO = {
    "MobileCLIP-S2": {
        "dossier": RESULTS_DIR / "mobileclip",
        "type"   : "mobileclip"
    },
    "CLIP ViT-B/32": {
        "dossier": RESULTS_DIR / "clip",
        "type"   : "clip"
    },
    "TinyCLIP": {
        "dossier": RESULTS_DIR / "tinyclip",
        "type"   : "tinyclip"
    },
    "SigLIP": {
        "dossier"         : RESULTS_DIR / "siglip",
        "type"            : "siglip"
    },
    "EVA-CLIP": {
        "dossier"         : RESULTS_DIR / "evaclip",
        "type"            : "openclip",
        "model_name"      : "EVA02-B-16",
        "model_pretrained": "merged2b_s8b_b131k"
    },
    "OpenCLIP": {
        "dossier"         : RESULTS_DIR / "openclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-32",
        "model_pretrained": "laion2b_s34b_b79k"
    },
    "MetaCLIP": {
        "dossier"         : RESULTS_DIR / "metaclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-32-quickgelu",
        "model_pretrained": "metaclip_400m"
    },
    "DFN-CLIP": {
        "dossier"         : RESULTS_DIR / "dfnclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-16",
        "model_pretrained": "dfn2b"
    }
}

# Cache pour éviter de recharger les modèles et index à chaque recherche
# On garde en mémoire ce qui a déjà été chargé
cache_modeles = {}
cache_index   = {}

# Charge et met en cache l'encodeur texte du modèle demandé.
def charger_encodeur_texte(nom_modele):

    if nom_modele in cache_modeles:
        return cache_modeles[nom_modele]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    type_m = MODELES_DISPO[nom_modele]["type"]

    logger.info(f"Chargement de l'encodeur texte pour {nom_modele}...")

    if type_m == "mobileclip":
        model, _, _ = mobileclip.create_model_and_transforms(
            "mobileclip_s2", pretrained=str(CHECKPOINT_PATH)
        )
        model = model.to(device)
        model.eval()
        tokenizer = mobileclip.get_tokenizer("mobileclip_s2")
        cache_modeles[nom_modele] = ("mobileclip", model, tokenizer, device)

    elif type_m == "clip":
        model, _ = clip.load("ViT-B/32", device=device)
        model.eval()
        cache_modeles[nom_modele] = ("clip", model, None, device)

    elif type_m == "siglip":
        # SigLIP utilise AutoProcessor et AutoModel de HuggingFace
        model_hf  = "google/siglip-base-patch16-224"
        processor = AutoProcessor.from_pretrained(model_hf)
        model     = AutoModel.from_pretrained(model_hf).to(device)
        model.eval()
        cache_modeles[nom_modele] = ("siglip", model, processor, device)

    elif type_m == "openclip":
        # EVA-CLIP, OpenCLIP, MetaCLIP et DFN-CLIP partagent tous la même API open_clip
        # On récupère model_name et model_pretrained depuis la config du modèle
        config    = MODELES_DISPO[nom_modele]
        model, _, _ = open_clip.create_model_and_transforms(
            config["model_name"], pretrained=config["model_pretrained"]
        )
        model     = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(config["model_name"])
        cache_modeles[nom_modele] = ("openclip", model, tokenizer, device)

    else:
        model_hf   = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
        processor  = CLIPProcessor.from_pretrained(model_hf)
        model      = CLIPModel.from_pretrained(model_hf).to(device)
        model.eval()
        cache_modeles[nom_modele] = ("tinyclip", model, processor, device)

    logger.info(f"Encodeur chargé pour {nom_modele}")
    return cache_modeles[nom_modele]

# Charge et met en cache l'index FAISS
def charger_index(nom_modele):

    if nom_modele in cache_index:
        return cache_index[nom_modele]

    dossier = MODELES_DISPO[nom_modele]["dossier"]
    index_path    = dossier / "index" / "index.faiss"
    metadata_path = dossier / "index" / "metadata.csv"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable pour {nom_modele}.\n"
            f"Lance d'abord indexation_faiss.py pour construire les index."
        )

    logger.info(f"Chargement de l'index FAISS pour {nom_modele}...")
    index    = faiss.read_index(str(index_path))
    metadata = pd.read_csv(metadata_path, index_col=0)

    cache_index[nom_modele] = (index, metadata)
    logger.info(f"Index chargé : {index.ntotal:,} vecteurs")
    return cache_index[nom_modele]

# Encode une requête texte en vecteur avec le bon modèle.
def encoder_requete(texte, encodeur):
    type_m, model, extra, device = encodeur

    with torch.no_grad():
        if type_m == "mobileclip":
            tokens     = extra([texte]).to(device)
            embedding  = model.encode_text(tokens)

        elif type_m == "clip":
            import clip
            tokens    = clip.tokenize([texte]).to(device)
            embedding = model.encode_text(tokens)

        elif type_m == "siglip":
            # SigLIP a des contraintes de longueur différentes, on fixe max_length à 64
            inputs    = extra(text=[texte], return_tensors="pt", padding="max_length",
                              max_length=64, truncation=True).to(device)
            embedding = model.get_text_features(**inputs)

        elif type_m == "openclip":
            # EVA-CLIP, OpenCLIP, MetaCLIP et DFN-CLIP utilisent tous encode_text via open_clip
            tokens    = extra([texte]).to(device)
            embedding = model.encode_text(tokens)

        else:
            inputs    = extra(text=[texte], return_tensors="pt", padding=True).to(device)
            embedding = model.get_text_features(**inputs)
            if hasattr(embedding, "pooler_output"):
                embedding = embedding.pooler_output

        # Normalisation pour que la recherche par produit scalaire soit correcte
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().float().numpy()

# Fonction principale appelée par l'interface Gradio à chaque recherche.
def rechercher(requete_texte, nom_modele, nb_resultats):
    if not requete_texte.strip():
        return [], "Veuillez saisir une requête texte pour lancer la recherche."

    try:
        encodeur         = charger_encodeur_texte(nom_modele)
        index, metadata  = charger_index(nom_modele)

        # On encode la requête texte pour obtenir son vecteur
        vecteur_requete = encoder_requete(requete_texte, encodeur)

        # FAISS cherche les nb_resultats vecteurs les plus proches du vecteur requête
        scores, ids = index.search(vecteur_requete, k=int(nb_resultats))

        resultats_images = []
        infos_texte      = []

        for rang, (score, id_frame) in enumerate(zip(scores[0], ids[0])):
            if id_frame < 0 or id_frame >= len(metadata):
                continue

            frame_info = metadata.iloc[id_frame]
            chemin     = Path(frame_info["filepath"])

            # Chargement de l'image si le fichier existe
            if chemin.exists():
                image = Image.open(chemin).convert("RGB")
            else:
                # Si le fichier n'est pas trouvé on génère une image grise de remplacement
                image = Image.new("RGB", (224, 224), color=(80, 80, 80))

            resultats_images.append(image)

            infos_texte.append(
                f"Résultat {rang + 1}  |  "
                f"Similarité : {score:.3f}  |  "
                f"Classe : {frame_info['class']}  |  "
                f"Vidéo : {frame_info['video']}  |  "
                f"Timestamp : {frame_info['timestamp']}s"
            )

        resume = (
            f"{len(resultats_images)} résultats trouvés avec {nom_modele} "
            f"pour la requête \"{requete_texte}\""
        )

        return resultats_images, resume + "\n\n" + "\n".join(infos_texte)

    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        logger.error(f"Erreur lors de la recherche : {e}")
        return [], f"Une erreur s'est produite : {e}"

# Construit et configure l'interface Gradio.
def construire_interface():
    with gr.Blocks(title="Recherche vidéo UCF-Crime") as interface:

        gr.Markdown("## Recherche de frames par description textuelle")
        gr.Markdown(
            "Veuillez entrer une description textuelle d'une scène (par exemple `person fighting`) "
            "Le système va alors retrouver les frames du dataset qui correspondent le plus."
        )

        with gr.Row():
            with gr.Column(scale=3):
                champ_requete = gr.Textbox(
                    label="Requête texte",
                    placeholder="Exemple : person fighting, car accident, robbery...",
                    lines=1
                )
            with gr.Column(scale=1):
                menu_modele = gr.Dropdown(
                    choices=list(MODELES_DISPO.keys()),
                    value="CLIP ViT-B/32",
                    label="Modèle"
                )

        with gr.Row():
            slider_nb = gr.Slider(
                minimum=1, maximum=20, value=6, step=1,
                label="Nombre de résultats"
            )
            bouton_recherche = gr.Button("Rechercher", variant="primary")

        texte_resume = gr.Textbox(label="Résultats", lines=4, interactive=False)
        galerie       = gr.Gallery(label="Frames retrouvées", columns=3, height=500)

        # Bouton de recherche et validation du champ texte déclenchent la même fonction de recherche
        bouton_recherche.click(
            fn=rechercher,
            inputs=[champ_requete, menu_modele, slider_nb],
            outputs=[galerie, texte_resume]
        )
        champ_requete.submit(
            fn=rechercher,
            inputs=[champ_requete, menu_modele, slider_nb],
            outputs=[galerie, texte_resume]
        )

        gr.Markdown(
            "**Exemples de requêtes :** `person fighting` | `car crash on road` | "
            "`robbery in store` | `person with gun` | `fire and smoke`"
        )

    return interface


if __name__ == "__main__":
    logger.info("Démarrage de l'interface Gradio...")
    interface = construire_interface()
    interface.launch(share=False)
