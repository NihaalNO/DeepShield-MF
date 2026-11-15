# DeepShield-MF

## Deepfake Video Detection using CNN, Vision Transformer, FFT and BiLSTM

DeepShield-MF is a deepfake detection system that analyzes videos using spatial, frequency-domain and temporal features to classify content as **Real** or **Fake**.

The repository contains two model tracks:

| Track | What it is | Where it lives | Served by the app? |
| --- | --- | --- | --- |
| **ViT baseline** | Frame-level ViT-B/16 classifier | `notebooks/DD_ViT_Final.ipynb` → `backend/models/best_vit_model.pth` | **Yes**: the Flask backend serves this model |
| **Hybrid research model** | Video-level ResNet50 + ViT-B/16 + FFT + BiLSTM | `notebooks/frac_df_cnnvit_fft_temporal.ipynb` | No. It is trained and evaluated in the notebook only |

> The deployed app (Flask + Streamlit) runs the **ViT baseline**. The hybrid model is the research model; its checkpoint is written to `results/hybrid_cnn_vit_fft_bilstm/` and is **not** loaded by the backend (see [Limitations](#limitations)).

---

## Contents

1. [Project structure](#project-structure)
2. [Local setup (one virtual environment)](#local-setup-one-virtual-environment)
3. [Dataset](#dataset)
4. [Training and evaluation (notebooks)](#training-and-evaluation-notebooks)
5. [Where results are saved](#where-results-are-saved)
6. [Model weights](#model-weights)
7. [Run the app locally (backend + Streamlit)](#run-the-app-locally-backend--streamlit)
8. [Configuration (.env)](#configuration-env)
9. [Architecture and method](#architecture-and-method)
10. [Limitations](#limitations)

---

## Project structure

```text
DeepShield-MF/
├── backend/
│   ├── app.py                  # Flask API: GET /health, POST /predict (ViT baseline)
│   ├── requirements.txt        # -> points to ../requirements.txt
│   └── models/
│       └── best_vit_model.pth  # deployed checkpoint (NOT in git, ~343 MB)
├── streamlit_app.py            # Streamlit frontend (talks to the Flask backend)
├── notebooks/
│   ├── frac_df_cnnvit_fft_temporal.ipynb  # hybrid research model (CNN+ViT+FFT+BiLSTM)
│   ├── DD_ViT_Final.ipynb                 # ViT baseline (source of the deployed checkpoint)
│   ├── deepfake_cnn_vit.ipynb             # frame extraction + CNN/ViT fusion experiment
│   ├── frac_df_cnnvit.ipynb               # CNN/ViT fusion on a data fraction
│   └── deepfake_detection_vit_best.ipynb  # archived Colab experiment (not runnable as-is)
├── data/                       # datasets (NOT in git, you create it, see "Dataset")
├── results/                    # all training/evaluation outputs (NOT in git, auto-created)
├── requirements.txt            # the single dependency list for everything
├── .env.example                # configuration template, copy to .env
├── Dockerfile                  # backend container (gunicorn, port 7860)
└── space/                      # Hugging Face Space deployment bundle (backend only)
```

`package.json` / `package-lock.json` are leftovers from an earlier React frontend. There is no React source in this repository and they are not used. The frontend is `streamlit_app.py`.

---

## Local setup (one virtual environment)

The whole project (backend, frontend, notebooks) uses **one** virtual environment, `.venv/`, at the **repository root**, and **one** dependency file, `requirements.txt`. Do not create a venv inside `backend/`, and never create one while another venv is active.

Requirements: Python **3.10 to 3.13** (tested locally with 3.13.1 on Windows), Git.

### 1. Clone

```bash
git clone https://github.com/Kshubham0315/DeepShield-MF.git
cd DeepShield-MF
```

### 2. Create the virtual environment (repository root)

Windows (PowerShell or cmd):

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell refuses to run `Activate.ps1`, skip activation and call the interpreter directly, e.g. `.venv\Scripts\python -m pip ...`, `.venv\Scripts\python backend\app.py`, `.venv\Scripts\python -m streamlit run streamlit_app.py`.

### 3. Install PyTorch, then everything else

Pick **one** PyTorch build:

```bash
# CPU only (works everywhere, smallest download)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# or NVIDIA GPU: take the exact command for your CUDA version from
# https://pytorch.org/get-started/locally/  (e.g. --index-url .../whl/cu126)
```

Then:

```bash
python -m pip install -r requirements.txt
```

`pip` keeps the PyTorch build you installed in the previous step. (If you skip that step, `pip` installs the default PyPI build, which is CPU-only on Windows.)

Check the install:

```bash
python -m pip check
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

> Existing clones: older instructions created `backend/.venv`. That environment is no longer used. Delete it after the root `.venv` works.

---

## Dataset

Datasets are **not** included and are **not** downloaded automatically. The canonical location is `data/` at the repository root (git-ignored). All notebooks resolve it as `<repo>/data`, or as the folder in the environment variable `DEEPSHIELD_DATA_DIR` if you set it.

```text
data/
├── FF++/                       # (optional) raw videos, input to frame extraction
│   ├── fake/   *.mp4
│   └── real/   *.mp4
├── FF_frames/                  # extracted frames, input to ALL training notebooks
│   ├── fake/
│   │   ├── <video_id>_0.jpg
│   │   ├── <video_id>_10.jpg
│   │   └── ...
│   └── real/
│       └── <video_id>_<frame_index>.jpg
└── samples/                    # (optional) test videos, e.g. sample.mp4 for DD_ViT_Final
```

Rules the code relies on:

* **Class folder names** must be `fake` and `real` (lower case). Classes are indexed alphabetically, so **index 0 = fake, index 1 = real**. This matches the deployed checkpoint, which was trained on folders `manipulated`/`real` (same order).
* **Frame file names** must be `<video_id>_<frame_number>.<jpg|jpeg|png>`. The hybrid notebook groups frames by `<video_id>`, orders them by `<frame_number>`, and skips videos with fewer than `SEQ_LEN` (8) frames.

### Preparing frames from raw videos

Put the videos into `data/FF++/fake/` and `data/FF++/real/`. Then run, in `notebooks/deepfake_cnn_vit.ipynb`, the *Project paths* cell and the *frame extraction* cell (the one that writes to `OUTPUT_PATH`). It saves **every 10th frame** of each video to `data/FF_frames/<class>/<video>_<frame_index>.jpg`. You only need to run that part of the notebook. It does not train anything.

---

## FaceForensics++ research pipeline (scripts)

The evaluation work on FaceForensics++ (source-disjoint splits, face crops, ablations) lives in `deepshield/` and `scripts/`, not in the notebooks. All locations are defined in `deepshield/paths.py`.

**Dataset location.** Keep the official FaceForensics++ layout and point `DEEPSHIELD_FFPP_DIR` at it in `.env` (default `<repo>/data/FaceForensics++`). Nothing is copied.

```text
<DEEPSHIELD_FFPP_DIR>/
├── original_sequences/youtube/c23/videos/NNN.mp4
└── manipulated_sequences/{Deepfakes,Face2Face,FaceShifter,FaceSwap,NeuralTextures}/c23/videos/TTT_SSS.mp4
```

**Step 1: manifest and splits** (about a minute):

```bash
python scripts/build_manifest.py
```

Writes `manifests/ffpp_c23.csv` (one row per video: label, method, target and source ID, source family, split, frame count, size) and `manifests/ffpp_c23_summary.json`. A fake `TTT_SSS` and its two originals form one source family. Families follow the official FaceForensics++ split in `manifests/ffpp_official_splits/` (360/70/70 pairs), so no family is shared between train, validation and test. The script exits with an error if one is. Both files are tracked in git.

**Step 2: face crops** (roughly 1.5 hours for 6,000 videos with 8 workers, resumable):

```bash
python scripts/extract_faces.py --workers 8
```

Samples 32 frames per video (every 8th frame, centred), tracks one face with OpenCV's YuNet detector (`assets/face_detector/face_detection_yunet_2023mar.onnx`, from the OpenCV model zoo) and writes:

```text
data/ffpp_c23/
├── faces/<method>/<video>/<frame>.jpg       # 224x224 tracked face crop
├── faces/<method>/<video>/meta.json         # frame indices, crop boxes, detection rate
├── fullframe/<method>/<video>/<frame>.jpg   # same frames, whole frame resized to 224x224
└── frame_index.csv                          # one row per video: status, frames saved, detection rate
```

Videos where a face is found in fewer than half of the frames are marked `failed` in `frame_index.csv` and get no crops. Use `--limit N` to test on the first N videos per method.

**Step 3: cache backbone features** (about 20 minutes per variant on an RTX 3050):

```bash
python scripts/cache_features.py --variant faces
```

```bash
python scripts/cache_features.py --variant fullframe
```

ResNet50 and ViT-B/16 are frozen in the hybrid model, so their per-frame outputs are computed once and reused by every experiment. The grayscale frames are cached too, because the trainable FFT branch needs pixels.

```text
data/ffpp_c23/features/<variant>/
├── resnet50.npy               # float16 [videos, 32, 2048]
├── vit_base_patch16_224.npy   # float16 [videos, 32, 768]
├── gray.npy                   # uint8   [videos, 32, 224, 224]  (about 9.6 GB)
├── index.csv                  # row -> video_id
└── meta.json                  # backbones, normalisation, shapes
```

**Step 4: train and validate one experiment:**

```bash
python scripts/train.py --exp F --seed 0
```

`--exp` selects a row of the ablation matrix (`deepshield/models.py`): A ResNet50, B ViT, C ResNet50+ViT (all mean-pooled), D ResNet50+BiLSTM, E ResNet50+ViT+BiLSTM, F the full hybrid with FFT. Other options: `--variant fullframe`, `--seq-len 4|8|16`, `--train-methods` (leave-one-method-out), `--limit-families N` (quick test).

How a video is scored: its 32 cached frames are cut into overlapping windows (8 frames, 50 % overlap, so 7 windows). The model gives each window a fake probability, and the video score is their mean. The video is called fake if that score is at or above a threshold chosen on the validation split to maximise balanced accuracy. The best epoch is chosen by validation video-level ROC-AUC.

Each run writes to `results/ffpp_c23/<run>/seed<S>/`:

```text
config.json                   # full configuration, data counts, hardware
checkpoints/best.pth          # weights + model config + class_names + threshold
logs/history.csv              # per-epoch losses, validation AUC, learning rate, seconds
metrics/val_metrics.json      # video-level metrics, per-method breakdown, 95 % CIs
predictions/val_videos.csv    # one row per validation video
```

Checkpoints carry their own `class_names`, model configuration and threshold. `deepshield.engine.load_checkpoint` refuses a checkpoint whose class order differs from the project's (`fake`, `real`). Confidence intervals resample source families, not individual videos.

The **test split must stay untouched** until the final evaluation: tune and select models on train and validation only. `FeatureStore.split_rows("test")` raises an error unless called with `allow_test=True`, and `scripts/train.py` never does that.

---

## Training and evaluation (notebooks)

Open the notebooks with the project `.venv` as kernel:

* **VS Code**: open the repository folder, open a notebook, then *Select Kernel → Python Environments → `.venv`*.
* **Jupyter in the browser** (optional, not in `requirements.txt`): `python -m pip install jupyterlab`, then run `python -m jupyter lab` **from the repository root** and open `notebooks/…`.

Each notebook starts with a **Project paths** cell. It finds the repository root by walking up from the working directory, so it works whether the kernel starts in `notebooks/` (the default) or in the repo root. It prints the dataset folder and the output folder, and the data-loading cell stops with a clear error if `data/FF_frames` is missing.

| Notebook | Purpose | Run order |
| --- | --- | --- |
| `frac_df_cnnvit_fft_temporal.ipynb` | **Hybrid research model**: video-level split → 8-frame sequences → ResNet50 + ViT + FFT → BiLSTM | run all cells top to bottom (cell 12 "overfit single batch" is an optional debug check) |
| `DD_ViT_Final.ipynb` | **ViT baseline** (frame-level, `ImageFolder`, random 80/20 frame split) | run all. The last cell runs on a local video: set `DEEPSHIELD_SAMPLE_VIDEO` or put one at `data/samples/sample.mp4` |
| `deepfake_cnn_vit.ipynb` | frame extraction + ResNet50/ViT feature fusion (full fine-tuning) | run all, or only the extraction cell |
| `frac_df_cnnvit.ipynb` | ResNet50/ViT fusion with frozen backbones on 30 % of frames (CPU) | run all |
| `deepfake_detection_vit_best.ipynb` | archived Google Colab experiment | not runnable as-is (see note in its first cell) |

`DD_ViT_Final.ipynb` and `deepfake_cnn_vit.ipynb` keep their original `DataLoader(num_workers=4)`. On Windows, if the training cell hangs or reports worker errors, set `num_workers=0` in those `DataLoader` calls. The hybrid and fraction notebooks already use `0`.

Hybrid model defaults (edit the *Config* cell): `SEQ_LEN=8`, `SEQ_STRIDE=2`, `BATCH_SIZE=2`, `EPOCHS=8`, `LR=3e-4`, `FREEZE_BACKBONES=True`, `NUM_WORKERS=0` (Windows-safe). The ResNet50 and ViT ImageNet weights are downloaded by `timm` from the Hugging Face Hub on first use and cached in `~/.cache/huggingface/`.

---

## Where results are saved

Every notebook writes **only** under `results/<experiment>/` (git-ignored, created automatically) and prints each path it saves. Re-running a notebook overwrites the files of that experiment.

```text
results/
├── hybrid_cnn_vit_fft_bilstm/            # frac_df_cnnvit_fft_temporal.ipynb
│   ├── checkpoints/best_hybrid_cnn_vit_fft_bilstm.pth   # {model_state_dict, class_names, config}
│   ├── checkpoints/best_checkpoint_info.json            # best epoch, val acc/loss
│   ├── logs/history.csv, logs/history.json              # per-epoch loss/acc/lr (written every epoch)
│   ├── metrics/final_metrics.json                       # acc, precision, recall, F1, ROC-AUC, confusion matrix
│   ├── metrics/classification_report.txt
│   ├── plots/training_curves.png, confusion_matrix.png, roc_curve.png
│   └── predictions/val_predictions.csv                  # one row per validation sequence
├── vit_baseline/                         # DD_ViT_Final.ipynb
│   ├── checkpoints/best_vit_model.pth
│   ├── logs/history.csv, logs/history.json
│   ├── metrics/final_metrics.json
│   └── plots/confusion_matrix.png
├── cnn_vit_fusion/                       # deepfake_cnn_vit.ipynb
│   ├── checkpoints/best_cnn_vit_model.pth, checkpoint_epoch_<N>.pth
│   ├── logs/history.csv, logs/history.json
│   ├── metrics/final_metrics.json
│   └── plots/confusion_matrix.png
├── cnn_vit_fusion_fraction/              # frac_df_cnnvit.ipynb
│   ├── checkpoints/small_cnn_vit_model.pth
│   ├── logs/history.csv, logs/history.json
│   ├── metrics/final_metrics.json, classification_report.txt
│   └── plots/confusion_matrix.png, sample_predictions.png, roc_curve.png
└── cnn_vit_resnet34_colab/checkpoints/   # deepfake_detection_vit_best.ipynb (when run locally)
```

The backend saves **nothing**. An uploaded file is written to the OS temp folder only while the request runs and is deleted afterwards. Predictions are returned to the caller and not stored.

To deploy a newly trained ViT baseline, copy `results/vit_baseline/checkpoints/best_vit_model.pth` to `backend/models/best_vit_model.pth` (or point `MODEL_PATH` at it).

---

## Model weights

| File | Model | Used by |
| --- | --- | --- |
| `backend/models/best_vit_model.pth` | `timm` `vit_base_patch16_224`, 2 classes, raw `state_dict` saved from `nn.DataParallel` (`module.` prefix, stripped on load) | Flask backend |
| `results/hybrid_cnn_vit_fft_bilstm/checkpoints/best_hybrid_cnn_vit_fft_bilstm.pth` | `TemporalHybridModel` | notebook only |

Model weights are git-ignored. The original README states that the ViT checkpoint is also published at https://huggingface.co/RohRoos84/deepshield-model. To have the backend download it on startup when `backend/models/best_vit_model.pth` is missing, set `MODEL_URL` (direct file URL) and optionally `MODEL_SHA256` in `.env`.

**Class mapping of the deployed checkpoint: output index 0 = fake/manipulated, 1 = real.** This comes from training with `ImageFolder` on folders `['manipulated', 'real']` (see the `print(...classes)` output in `DD_ViT_Final.ipynb`). The backend uses `CLASS_NAMES=fake,real` by default. Change it only for a checkpoint trained with another class order.

---

## Run the app locally (backend + Streamlit)

Prerequisites: setup done, and `backend/models/best_vit_model.pth` present. Without the checkpoint the backend still starts, but `/health` reports `model_loaded: false` and `/predict` returns HTTP 503 instead of guessing with untrained weights.

Optional: `copy .env.example .env` (Windows) or `cp .env.example .env` and edit. The defaults work without a `.env`.

**Terminal 1: backend** (repository root, venv active):

```bash
python backend/app.py
```

Expected log lines:

```text
Using device: cpu
Model loaded from ...\backend\models\best_vit_model.pth
Class mapping (output index -> label): {0: 'fake', 1: 'real'}
Starting DeepShield backend on http://127.0.0.1:5000  (health: /health, inference: POST /predict)
```

Check it: open http://127.0.0.1:5000/health, or

```bash
curl http://127.0.0.1:5000/health
curl -N -F "file=@path/to/video.mp4" http://127.0.0.1:5000/predict
```

**Terminal 2: Streamlit frontend** (repository root, venv active):

```bash
python -m streamlit run streamlit_app.py
```

Open http://localhost:8501. The sidebar shows *Backend Online*, the device, *Model Loaded: True* and the checkpoint path. Upload an image or video and click **Analyze Media**.

The frontend finds the backend through `DEEPSHIELD_BACKEND_URL` (default `http://127.0.0.1:5000`). If you change the backend `HOST`/`PORT`, update that variable too.

### API

* `GET /health` → `{status, device, model_loaded, model_error, model_arch, model_path, class_names, max_upload_mb, model_url_set}`
* `POST /predict` (multipart field `file`) → Server-Sent Events. Each `data:` line is
  `{total_frames, real_count, fake_count, real_percentage, fake_percentage}`. The last event holds the final counts. Videos are sampled at ~2 frames per second; an image is one frame. Errors arrive as `{"error": ...}`.

### Docker (optional)

`Dockerfile` builds the backend only and serves it with gunicorn on port 7860. It expects the checkpoint inside `backend/models/` at build time, or `MODEL_URL` at runtime.

---

## Configuration (.env)

All settings are optional. Both `backend/app.py` and `streamlit_app.py` read `<repo>/.env`; real environment variables take precedence. See `.env.example`.

| Variable | Default | Used by |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | backend |
| `MODEL_PATH` | `backend/models/best_vit_model.pth` | backend |
| `MODEL_URL`, `MODEL_SHA256` | unset | backend (download if `MODEL_PATH` missing) |
| `CLASS_NAMES` | `fake,real` | backend (output index → label) |
| `MAX_CONTENT_LENGTH_MB` | `100` | backend upload limit |
| `CORS_ALLOWED_ORIGINS` | all | backend |
| `DEEPSHIELD_BACKEND_URL` | `http://127.0.0.1:5000` | Streamlit |
| `DEEPSHIELD_DATA_DIR` | `<repo>/data` | notebooks |
| `DEEPSHIELD_SAMPLE_VIDEO` | `data/samples/sample.mp4` | `DD_ViT_Final.ipynb` last cell |

---

## Architecture and method

### Overview

Deepfake videos can contain artifacts that are not always visible in a single frame. DeepShield-MF analyzes a video from three perspectives:

1. **Spatial domain**: visual features from individual frames (ResNet50, ViT-B/16).
2. **Frequency domain**: FFT magnitude spectrum of each frame.
3. **Temporal domain**: frame-to-frame relationships (BiLSTM).

### Hybrid research model

```text
Input Video
     |
Frame Extraction (every 10th frame)
     |
Sequence Builder (8 frames, stride 2, 224x224)
     |
     +----------------+----------------+
     |                |                |
     v                v                v
   FFT             ResNet50          ViT-B/16
 Branch          Spatial Encoder   Spatial Encoder
  (128)              (2048)            (768)
     |                |                |
     +----------------+----------------+
                      |
          Feature Fusion (concat + LayerNorm)
                      |
          BiLSTM Temporal Modeling (hidden 192)
                      |
          Temporal Pooling (mean)
                      |
                  Classifier
                      |
                 Real / Fake
```

* **ResNet50** (`timm`, ImageNet weights) captures local texture, blending and fine-grained artifacts.
* **ViT-B/16** (`vit_base_patch16_224`, ImageNet weights) models global relationships between image patches.
* **FFT branch**: grayscale → `fft2` → log-magnitude → small CNN. Generation methods often leave frequency-domain traces.
* **BiLSTM** learns temporal inconsistencies across the fused per-frame features.

Training details (from the notebook): video-level train/validation split (no frames from one video in both splits), class-weighted cross-entropy with label smoothing 0.05, AdamW, `ReduceLROnPlateau` on validation accuracy, gradient clipping 1.0, early stopping (patience 3), best-validation checkpoint, frozen backbones by default. Metrics: accuracy, weighted precision/recall/F1, ROC-AUC, confusion matrix, classification report.

### ViT baseline (deployed)

`timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)`, fine-tuned on single frames (224×224, ImageNet normalization). At inference the backend samples ~2 frames per second, classifies each frame, and reports the share of real and fake frames. The frontend labels the video by the majority.

---

## Results

### ViT baseline

Recorded in `notebooks/DD_ViT_Final.ipynb` (Kaggle run, random 80/20 split at **frame** level):

| Metric | Result |
| --- | ---: |
| Best validation accuracy | 88.13 % |
| Final-epoch training accuracy | 90.03 % |
| Validation precision / recall / F1 (weighted) | 0.883 / 0.883 / 0.883 |

Because the split is per frame, frames of the same video can appear in both training and validation. These numbers are therefore optimistic.

### Hybrid model

Experimental. No validated result is recorded in the repository. Run the notebook and read `results/hybrid_cnn_vit_fft_bilstm/metrics/final_metrics.json`.

---

## Limitations

* **The deployed app uses the ViT baseline, not the hybrid model.** The backend has no sequence-level inference path for `TemporalHybridModel`. Serving it would need a new endpoint and frame-sequence preprocessing.
* **Domain shift:** on 44 real and 44 fake DFDC videos (not part of training), the deployed ViT separates the classes only weakly (mean P(real) 0.67 on real vs 0.58 on fake videos). Treat predictions as an AI-assisted signal, not proof.
* The ViT baseline numbers come from a frame-level split and may be inflated by leakage between frames of the same video.
* No dataset, and no trained hybrid checkpoint, is shipped. The training notebooks cannot run until `data/FF_frames` exists.
* `deepfake_detection_vit_best.ipynb` is an archived Colab experiment and does not run as-is.
* The Flask development server (`python backend/app.py`) is for local use. Use gunicorn/Docker (Linux) for anything public.
* Model performance also varies with resolution, compression, lighting, face quality and unseen manipulation methods.

---

## License

This project is intended for educational, research and experimental purposes.

Shubham Kumar
