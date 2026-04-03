
import os
import subprocess
from pathlib import Path
import logging


# Chemin vers le dossier Dataset (contenant les 13 sous-dossiers de classes)
DATASET_DIR = Path("Dataset")

# Dossier de sortie où seront stockées toutes les frames extraites
OUTPUT_DIR = Path("frames")

# Nombre de frames extraites par seconde de vidéo
FPS = 1

# Extensions vidéo reconnues par le script
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


# Affiche les messages dans le terminal avec horodatage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

#Extrait les frames d'une vidéo avec ffmpeg
def extract_frames_from_video(video_path: Path, output_folder: Path, fps: int = 1) -> int:


    # Crée le dossier de sortie s'il n'existe pas encore
    output_folder.mkdir(parents=True, exist_ok=True)

    # Modèle de nom pour les frames, exemple: Shooting004_x264_t0001s.jpg
    video_name = video_path.stem
    output_pattern = output_folder / f"{video_name}_t%04ds.jpg"

    # Construction de la commande FFmpeg
    command = [
        "ffmpeg",
        "-i", str(video_path),          # vidéo source
        "-vf", f"fps={fps}",            # filtre : N frames par seconde
        "-frame_pts", "1",              # utilise le timestamp réel pour numéroter
        "-q:v", "2",                    # qualité des JPEG générés
        "-loglevel", "error",           # masque les logs verbose de FFmpeg
        str(output_pattern)             # chemin de sortie avec pattern horodaté
    ]

    try:
        # Lance la commande FFmpeg dans un sous-processus
        subprocess.run(command, check=True)

        # Compte le nombre de frames réellement créées dans le dossier
        nb_frames = len(list(output_folder.glob("*_t*.jpg")))
        return nb_frames

    except subprocess.CalledProcessError as e:
        # FFmpeg a rencontré une erreur sur cette vidéo
        logger.error(f"Erreur FFmpeg sur {video_path.name} : {e}")
        return 0

    except FileNotFoundError:
        # FFmpeg n'est pas installé ou pas dans le PATH
        logger.critical("FFmpeg introuvable. Installe-le avec : conda install -c conda-forge ffmpeg")
        raise

# Parcourt tout le dataset et extrait les frames de chaque vidéo.
def process_dataset(dataset_dir: Path, output_dir: Path, fps: int = 1):


    # Vérifie que le dossier dataset existe bien
    if not dataset_dir.exists():
        logger.error(f"Le dossier dataset '{dataset_dir}' est introuvable.")
        return

    # Récupère la liste des 13 classes (sous-dossiers du dataset)
    class_folders = sorted([f for f in dataset_dir.iterdir() if f.is_dir()])

    if not class_folders:
        logger.warning("Aucun sous-dossier de classe trouvé dans le dataset.")
        return

    logger.info(f"Dataset trouvé : {len(class_folders)} classes détectées")
    logger.info(f"Frames de sortie → {output_dir.resolve()}")
    logger.info(f"FPS d'extraction : {fps} frame(s)/seconde\n")

    # Compteurs globaux pour le résumé final
    total_videos = 0
    total_frames = 0
    total_errors = 0

    # Boucle sur chaque classe 
    for class_folder in class_folders:
        class_name = class_folder.name 

        # Liste toutes les vidéos dans ce dossier de classe
        video_files = sorted([
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ])

        if not video_files:
            logger.warning(f"[{class_name}] Aucune vidéo trouvée, classe ignorée.")
            continue

        logger.info(f"[{class_name}] {len(video_files)} vidéo(s) trouvée(s)")

        # Boucle sur chaque vidéo de la classe
        for video_path in video_files:

            # Nomme le dossier de sortie pour les frames de cette vidéo, ex: frames/Shooting/Shooting004_x264/
            video_stem = video_path.stem
            output_folder = output_dir / class_name / video_stem

            # Si le dossier existe déjà et contient des frames alors on saute
            if output_folder.exists() and any(output_folder.glob("*_t*.jpg")):
                existing = len(list(output_folder.glob("*_t*.jpg")))
                logger.info(f"[{video_stem}] déjà extrait ({existing} frames), ignoré.")
                total_videos += 1
                total_frames += existing
                continue

            logger.info(f"[{video_stem}] extraction en cours...")

            # Appel de la fonction d'extraction
            nb_frames = extract_frames_from_video(video_path, output_folder, fps)

            if nb_frames > 0:
                logger.info(f"[{video_stem}] {nb_frames} frames extraites")
                total_frames += nb_frames
            else:
                logger.warning(f"[{video_stem}] échec ou vidéo vide")
                total_errors += 1

            total_videos += 1

    # Résumé final
    logger.info("EXTRACTION TERMINÉE")
    logger.info(f"  Vidéos traitées : {total_videos}")
    logger.info(f"  Frames extraites : {total_frames:,}")
    logger.info(f"  Erreurs          : {total_errors}")
    logger.info(f"  Sortie           : {output_dir.resolve()}")


# POINT D'ENTRÉE
if __name__ == "__main__":
    process_dataset(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        fps=FPS
    )
