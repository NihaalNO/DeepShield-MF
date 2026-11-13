import os
import json
import hashlib
import shutil
import tempfile
import urllib.request

from dotenv import load_dotenv

import cv2
import numpy as np
import torch
import timm
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Configuration is read from environment variables; a `.env` file at the
# project root (see .env.example) is loaded if present. Real environment
# variables take precedence over values in .env.
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(DOTENV_PATH, override=False)

app = Flask(__name__)

def get_env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"WARNING: Invalid integer for {name}={value!r}, using default={default}")
        return default


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = f"{dest_path}.download"
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp_path, dest_path)


cors_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if cors_allowed_origins:
    origins = [origin.strip() for origin in cors_allowed_origins.split(",") if origin.strip()]
    CORS(app, resources={r"/*": {"origins": origins}})
else:
    CORS(app)

# Max upload size in MB (default 100 for free-tier stability)
max_upload_mb = max(1, get_env_int("MAX_CONTENT_LENGTH_MB", 100))
app.config['MAX_CONTENT_LENGTH'] = max_upload_mb * 1024 * 1024

# Local default is loopback only; Docker/HF images set HOST=0.0.0.0 explicitly.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = get_env_int("PORT", 5000)

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the model architecture using timm
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
model = model.to(device)


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict


# Deployed checkpoint: ViT-B/16 frame classifier (state_dict, optionally with the
# "module." prefix written by nn.DataParallel). Default: backend/models/best_vit_model.pth
MODEL_ARCH = "vit_base_patch16_224"
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BACKEND_DIR, "models"))
MODEL_PATH = os.path.abspath(os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "best_vit_model.pth")))
MODEL_DIR = os.path.dirname(MODEL_PATH)


def parse_class_names(raw):
    names = [name.strip().lower() for name in raw.split(",") if name.strip()]
    if len(names) != 2 or "real" not in names:
        raise ValueError(
            f"CLASS_NAMES must list exactly two classes including 'real' "
            f"(e.g. 'fake,real'), got {raw!r}"
        )
    return names


# Output index -> class. The shipped checkpoint was trained with
# torchvision.datasets.ImageFolder on folders ['manipulated', 'real']
# (alphabetical), i.e. index 0 = fake/manipulated, index 1 = real
# (see notebooks/DD_ViT_Final.ipynb). Override only for a checkpoint
# trained with a different class order.
CLASS_NAMES = parse_class_names(os.getenv("CLASS_NAMES", "fake,real"))
REAL_INDEX = CLASS_NAMES.index("real")
MODEL_URL = os.getenv("MODEL_URL", "").strip()
MODEL_SHA256 = os.getenv("MODEL_SHA256", "").strip()


def ensure_model_file():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and MODEL_SHA256:
        current = sha256_file(MODEL_PATH)
        if current != MODEL_SHA256:
            print("WARNING: Model checksum mismatch, removing cached file.")
            try:
                os.remove(MODEL_PATH)
            except OSError:
                pass

    if os.path.exists(MODEL_PATH):
        return True

    if not MODEL_URL:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Place the trained checkpoint there, or set MODEL_PATH / MODEL_URL.")
        return False

    print(f"Downloading model from {MODEL_URL} ...")
    try:
        download_file(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        print(f"ERROR: Failed to download model: {exc}")
        return False

    if MODEL_SHA256:
        downloaded = sha256_file(MODEL_PATH)
        if downloaded != MODEL_SHA256:
            print("ERROR: Downloaded model checksum mismatch.")
            try:
                os.remove(MODEL_PATH)
            except OSError:
                pass
            return False

    return True


MODEL_LOADED = False
MODEL_LOAD_ERROR = None

if ensure_model_file():
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)
        MODEL_LOADED = True
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as exc:  # wrong architecture, corrupt file, ...
        MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: Could not load checkpoint {MODEL_PATH}: {MODEL_LOAD_ERROR}")
else:
    MODEL_LOAD_ERROR = f"Checkpoint not found at {MODEL_PATH}"

if not MODEL_LOADED:
    print("WARNING: No trained weights loaded - /predict will return HTTP 503.")
print(f"Class mapping (output index -> label): {dict(enumerate(CLASS_NAMES))}")

model.eval()

# ImageNet normalization values (must match training transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_frame(frame):
    """Preprocess a single BGR frame for the ViT model."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to 224x224
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    # Normalize to [0, 1] then apply ImageNet normalization
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_normalized = (frame_normalized - IMAGENET_MEAN) / IMAGENET_STD
    # Convert to tensor: (1, 3, 224, 224)
    tensor = torch.tensor(frame_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    return tensor


def is_image_upload(filename, mimetype):
    if mimetype and mimetype.startswith("image/"):
        return True
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in IMAGE_EXTS


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model weights are not loaded", "detail": MODEL_LOAD_ERROR}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    video_file = request.files["file"]
    filename = video_file.filename or ""
    is_image = is_image_upload(filename, video_file.mimetype)
    suffix = os.path.splitext(filename)[1].lower() if filename else (".png" if is_image else ".mp4")

    # Uploads are written to a temporary file (system temp dir) only for the
    # duration of the request and deleted afterwards; nothing is persisted.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        video_file.save(temp.name)
        video_path = temp.name

    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }

    if is_image:
        frame = cv2.imread(video_path)
        try:
            if frame is None:
                return jsonify({"error": "Unable to read image"}), 400

            input_tensor = preprocess_frame(frame)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)

            real_count = 1 if predicted.item() == REAL_INDEX else 0
            fake_count = 1 - real_count
            total_frames = 1

            data = {
                "total_frames": total_frames,
                "real_count": real_count,
                "fake_count": fake_count,
                "real_percentage": (real_count / total_frames) * 100,
                "fake_percentage": (fake_count / total_frames) * 100,
            }
            return Response(f"data: {json.dumps(data)}\n\n", mimetype="text/event-stream", headers=response_headers)
        finally:
            try:
                os.remove(video_path)
            except OSError:
                pass

    def generate():
        cap = cv2.VideoCapture(video_path)
        real_count = 0
        fake_count = 0
        total_frames = 0

        # Get total frame count and FPS for frame-skipping
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process ~2 frames per second to avoid being excessively slow
        skip = max(1, int(fps / 2))

        try:
            if not cap.isOpened():
                yield f"data: {json.dumps({'error': 'Unable to read video'})}\n\n"
                return

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                # Skip frames to speed up processing
                if frame_idx % skip != 0:
                    continue

                total_frames += 1
                input_tensor = preprocess_frame(frame)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    # Label mapping comes from CLASS_NAMES (default: 0 = fake, 1 = real)
                    if predicted.item() == REAL_INDEX:
                        real_count += 1
                    else:
                        fake_count += 1

                data = {
                    "total_frames": total_frames,
                    "real_count": real_count,
                    "fake_count": fake_count,
                    "real_percentage": (real_count / total_frames) * 100,
                    "fake_percentage": (fake_count / total_frames) * 100,
                }
                yield f"data: {json.dumps(data)}\n\n"

            if total_frames == 0:
                yield f"data: {json.dumps({'error': 'No frames could be decoded from the video'})}\n\n"
        finally:
            cap.release()
            # Always clean up temp file
            try:
                os.remove(video_path)
            except OSError:
                pass

    return Response(generate(), mimetype="text/event-stream", headers=response_headers)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": MODEL_LOADED,
        "model_error": MODEL_LOAD_ERROR,
        "model_arch": MODEL_ARCH,
        "model_path": MODEL_PATH,
        "class_names": CLASS_NAMES,
        "max_upload_mb": max_upload_mb,
        "model_url_set": bool(MODEL_URL),
    })


if __name__ == "__main__":
    print(f"Starting DeepShield backend on http://{HOST}:{PORT}  (health: /health, inference: POST /predict)")
    app.run(host=HOST, port=PORT, debug=False)
