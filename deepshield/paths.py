"""Single source of truth for project locations.

Every location can be overridden with an environment variable (or a line in
the project-root `.env`); real environment variables take precedence.

    DEEPSHIELD_DATA_DIR   derived data (frames, face crops, features)  default <repo>/data
    DEEPSHIELD_FFPP_DIR   FaceForensics++ download (official layout)   default <repo>/data/FaceForensics++
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # optional: the same .env the backend and Streamlit app read
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


def _env_path(name, default):
    return Path(os.environ.get(name) or default).expanduser().resolve()


DATA_ROOT = _env_path("DEEPSHIELD_DATA_DIR", PROJECT_ROOT / "data")
FFPP_ROOT = _env_path("DEEPSHIELD_FFPP_DIR", DATA_ROOT / "FaceForensics++")
RESULTS_ROOT = PROJECT_ROOT / "results"
MANIFEST_DIR = PROJECT_ROOT / "manifests"  # tracked in git (small, provenance)
ASSETS_DIR = PROJECT_ROOT / "assets"

FFPP_MANIFEST = MANIFEST_DIR / "ffpp_c23.csv"
FFPP_OFFICIAL_SPLITS = MANIFEST_DIR / "ffpp_official_splits"
FACE_DETECTOR_MODEL = ASSETS_DIR / "face_detector" / "face_detection_yunet_2023mar.onnx"

# Preprocessed FF++ c23 frames (written by scripts/extract_faces.py)
FFPP_FACES_DIR = DATA_ROOT / "ffpp_c23" / "faces"          # tracked 224x224 face crops
FFPP_FULLFRAME_DIR = DATA_ROOT / "ffpp_c23" / "fullframe"  # same frames, whole frame resized to 224x224
FFPP_FRAME_INDEX = DATA_ROOT / "ffpp_c23" / "frame_index.csv"

# Class contract used everywhere (matches backend CLASS_NAMES): index 0 = fake, 1 = real
CLASS_NAMES = ["fake", "real"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
