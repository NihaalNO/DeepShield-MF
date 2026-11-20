# DeepShield-MF

## Deepfake Video Detection using CNN, Vision Transformer, FFT and BiLSTM

DeepShield-MF is a deepfake detection system that analyzes videos using spatial, frequency-domain and temporal features to classify content as **Real** or **Fake**.

The repository contains these model tracks:

| Track | What it is | Where it lives | Served by the app? |
| --- | --- | --- | --- |
| **Fine-tuned ResNet50 + ViT** | Video-level model on face crops: ResNet50 and ViT-B/16 with their last layers fine-tuned, scored over 8-frame windows | `scripts/finetune_e2e.py` → `results/ffpp_c23/e2e_finetune_aug-frame_ffpp+celebdf/` (trained on FF++ + Celeb-DF v2; the FF++-only `e2e_finetune_aug-frame_final/` is kept) | **Yes, by default** (`SERVE_MODEL=finetuned`) |
| **ViT baseline** | Frame-level ViT-B/16 classifier | `notebooks/DD_ViT_Final.ipynb` → `backend/models/best_vit_model.pth` | With `SERVE_MODEL=vit`, and in the Docker image |
| **Hybrid research model** | Video-level ResNet50 + ViT-B/16 + FFT + BiLSTM | `notebooks/frac_df_cnnvit_fft_temporal.ipynb`, `scripts/` (row F) | No; evaluated by the research scripts only |

> The local app (Flask + Streamlit) serves the **fine-tuned ResNet50 + ViT trained on FF++ + Celeb-DF v2** (0.874 AUC on the FF++ test split, 0.956 on the Celeb-DF v2 test list; see [Results](#results)). The ViT baseline remains available with `SERVE_MODEL=vit`.

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
│   ├── app.py                  # Flask API: GET /health, POST /predict (fine-tuned model or ViT baseline)
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

**Step 5: ablation matrix** (resumable; finished runs are skipped):

```bash
python scripts/run_ablations.py
```

Trains rows A–F with 3 seeds, picks the best BiLSTM row by mean validation AUC, then runs the frame-order shuffle test, sequence lengths 4 and 16 and the full-frame comparison for it. `scripts/summarize.py` writes `results/ffpp_c23/summary/ablation_table.md` and the CSVs behind it.

**Step 6: analysis of trained runs** (validation split only; nothing is retrained except in the first script):

| Script | What it does | Output in `results/ffpp_c23/` |
| --- | --- | --- |
| `scripts/run_lomo.py` | Leave-one-method-out: trains on four manipulation methods and scores the fifth. Model selection and threshold use only the four training methods. | `<exp>_faces_T8_lomo-<Method>/`, `summary/lomo_table.md` |
| `scripts/calibration.py` | ECE, Brier score and NLL of the video-level P(fake), raw and after Platt scaling cross-fitted over families. | `summary/calibration_table.md`, `summary/plots/reliability_<run>.png` |
| `scripts/robustness.py` | Re-scores the frozen checkpoints on degraded face crops: JPEG, blur, downscaling, noise, brightness. | `summary/robustness_table.md`, `summary/plots/robustness_auc.png` |
| `scripts/explain_occlusion.py --run-dir <run>/seed0` | Occlusion-sensitivity maps (7x7 grid) and a deletion test. | `<run>/seed0/explanations/` |
| `scripts/profile_compute.py` | Time per stage (decode, face crop, each backbone, head) on GPU and CPU, parameters, peak GPU memory. Run it with the GPU otherwise idle. | `summary/compute_profile.md` |

**Step 7: optional model changes** (validation split only):

```bash
python scripts/cache_extra_backbone.py --backbone efficientnet_b0
```

```bash
python scripts/run_improvements.py
```

`run_improvements.py` trains one-change variants of the ablation rows on the cached features with 3 seeds and writes `summary/improvements_table.md`. The rows are defined in `deepshield/models.py`: `I` branch projections before fusion, `J` learned softmax gate over the branches (`J0`: fixed equal weights, the gate's ablation), `K` one-directional LSTM, `L` EfficientNet-B0 instead of ResNet50 (needs the first command), `M1`–`M3` frequency-band variants of the FFT branch.

```bash
python scripts/finetune_e2e.py --mode finetune --aug sequence
```

`finetune_e2e.py` trains on face-crop pixels instead of cached features: `--mode frozen|finetune` (fine-tuning unfreezes the last ResNet50 stage and the last two ViT blocks), `--aug none|sequence|frame` (the same augmentation for all frames of a window, or a different one per frame). It uses 120 training families by default (about 45 minutes on a 6 GB laptop GPU; `--train-families 360` uses the whole training split, about 2 hours), and it also scores the result on degraded validation crops. Runs go to `results/ffpp_c23/e2e_<mode>_aug-<aug>_fam<N>/` or `--name`.

**Step 8: final evaluation** (run once, after all choices are made on validation):

```bash
python scripts/finetune_e2e.py --mode finetune --aug frame --train-families 360 --name e2e_finetune_aug-frame_final
```

```bash
python scripts/final_eval.py
```

`final_eval.py` scores the models fixed in `docs/final_evaluation_protocol.md` on the FF++ test split and the Celeb-DF v1/v2 test lists with their frozen validation thresholds. It writes `results/final/<dataset>/<model>.json` and `results/final/final_table.md`, and never overwrites an existing result.

**Step 9: second-domain fine-tuning (FF++ + Celeb-DF v2).** The FF++-only model called many real Celeb-DF videos fake: its only real training videos are 720 FF++ news clips. This step adds Celeb-DF v2 videos that are in neither official test list:

```bash
python scripts/build_manifest_celebdf_train.py
```

```bash
python scripts/extract_faces.py --dataset celebdf_v2_train
```

```bash
python scripts/finetune_e2e.py --mode finetune --aug frame --train-families 360 --extra-dataset celebdf_v2_train --name e2e_finetune_aug-frame_ffpp+celebdf
```

```bash
python scripts/final_eval.py --models final_cdf
```

**Unseen-dataset test (DFD).** Download FaceForensics++'s `DeepFakeDetection` and `DeepFakeDetection_original` (c23) with the FaceForensics++ script into `DEEPSHIELD_FFPP_DIR`, then:

```bash
python scripts/build_manifest_dfd.py
```

```bash
python scripts/extract_faces.py --dataset dfd_c23
```

```bash
python scripts/cache_features.py --dataset dfd_c23 --variant faces
```

```bash
python scripts/final_eval.py --datasets dfd
```

DFD is never used for training, model selection or thresholds. Videos without a usable face track (43) or with fewer than 32 face crops (13 short clips) are left out, leaving 3,375 videos.

**Step 10: calibration of the served model.**

```bash
python scripts/calibrate_served.py
```

Fits Platt scaling, p = sigmoid(a x logit(score) + b), on the served model's validation scores (FF++ val + Celeb-DF v2 val), weighted so that both datasets and both classes count equally. It writes `calibration.json` next to the checkpoint's run directory; `deepshield/serving.py` reads it and adds `p_calibrated` to each result. As recorded in `docs/final_evaluation_protocol.md` before fitting, the decision rule "calibrated p >= 0.5" would replace the threshold only if it raised mean per-dataset balanced accuracy on validation. It did not (0.838 vs 0.840), so the threshold stays at 0.816.

The manifest (`manifests/celebdf_v2_train.csv`) excludes every video of the v1 and v2 test lists (v1 is a byte-identical subset of v2), keeps all reals, subsamples fakes (1,500 train, 300 validation), and puts 6 celebrities in validation only. Model selection and the threshold use FF++ val + Celeb-DF val. The rule for serving it was written in `docs/final_evaluation_protocol.md` before training. After this step the Celeb-DF test lists are same-dataset tests: unseen videos, but 54 of 59 test celebrities also appear in training.

**Cross-dataset data (Celeb-DF v1 and v2).** Point `DEEPSHIELD_CELEBDF_V1_DIR` and `DEEPSHIELD_CELEBDF_V2_DIR` at the downloads (official layout with `List_of_testing_videos.txt`). The official test lists are used only for the final evaluation; the FF++-only models never trained, selected or set thresholds on Celeb-DF. Step 9 trains on other Celeb-DF v2 videos.

```bash
python scripts/build_manifest_celebdf.py --version v2
```

```bash
python scripts/extract_faces.py --dataset celebdf_v2
```

```bash
python scripts/cache_features.py --dataset celebdf_v2 --variant faces
```

The same three commands with `v1` / `celebdf_v1` prepare Celeb-DF v1. Manifests go to `manifests/celebdf_<version>.csv` (tracked), derived data to `data/celebdf_<version>/` with the same layout as `data/ffpp_c23/`.

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

Prerequisites: setup done, and the checkpoint of the served model present:

* `SERVE_MODEL=finetuned` (default): `results/ffpp_c23/e2e_finetune_aug-frame_ffpp+celebdf/seed0/checkpoints/best_trainable.pth`, written by README Step 9 (`scripts/finetune_e2e.py`); set `FINETUNED_CHECKPOINT` to `results/ffpp_c23/e2e_finetune_aug-frame_final/seed0/checkpoints/best_trainable.pth` for the FF++-only model of Step 8. It holds only the fine-tuned layers; the rest of ResNet50 and ViT-B/16 comes from the `timm` pretrained weights (downloaded once, then cached). It also needs `deepshield/`, `scripts/` and the face detector in `assets/face_detector/`.
* `SERVE_MODEL=vit`: `backend/models/best_vit_model.pth`.

Without the checkpoint the backend still starts, but `/health` reports `model_loaded: false` and `/predict` returns HTTP 503 instead of guessing with untrained weights.

**What the fine-tuned model does with an upload.** Exactly what it was evaluated with: 32 frames at a constant stride (every 8th frame, fewer for short clips) from the middle of the video, a YuNet face track, square crops of 1.3x the face, 7 windows of 8 frames, and the mean window P(fake) as the video score. The video is fake if the score reaches the threshold chosen on validation (0.816 for the served model, chosen on FF++ val + Celeb-DF val). A video with a face in fewer than half of those frames is refused, and so is an image without a face. Only about 10 seconds from the middle of a video are read. A still image is scored as one window of the same face repeated, which the model was never evaluated on. On the GPU a video takes about 1 second.

Optional: `copy .env.example .env` (Windows) or `cp .env.example .env` and edit. The defaults work without a `.env`.

**Terminal 1: backend** (repository root, venv active):

```bash
python backend/app.py
```

Expected log lines:

```text
Using device: cuda
Serving model: finetuned
Fine-tuned model loaded from ...\results\ffpp_c23\e2e_finetune_aug-frame_ffpp+celebdf\seed0\checkpoints\best_trainable.pth (threshold 0.8161)
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

Open http://localhost:8501. The top bar shows *Backend online*, *Model loaded*, the device and the served model; the sidebar shows the checkpoint path and threshold. Upload an image or video and click **Analyze media**.

The frontend finds the backend through `DEEPSHIELD_BACKEND_URL` (default `http://127.0.0.1:5000`). If you change the backend `HOST`/`PORT`, update that variable too.

### API

* `GET /health` → `{status, serve_model, threshold, device, model_loaded, model_error, model_arch, model_path, class_names, max_upload_mb, model_url_set}`
* `POST /predict` (multipart field `file`) → Server-Sent Events, one JSON object per `data:` line. Errors arrive as `{"error": ...}` (with `"kind": "no_face"` when no face was found).
  * `SERVE_MODEL=finetuned`: progress events `{"stage": "decoding"}`, `{"stage": "faces", "done", "of"}`, `{"stage": "scoring"}`, then `{"stage": "done", "model_kind": "finetuned", "score", "threshold", "label", "window_probs", "windows_fake", "mode", "evaluated", "frames_analyzed", "face_rate", "segment_seconds", "video_seconds", "thumbs"}`. `score` is the mean window P(fake), not a calibrated probability; `p_calibrated` (present when a `calibration.json` exists) is P(fake) for equally likely real and fake videos, typically off by about 10 points; `thumbs` are small JPEG data URLs of the face crops that were scored.
  * `SERVE_MODEL=vit`: `{total_frames, real_count, fake_count, real_percentage, fake_percentage}` after every sampled frame (~2 per second); the last event holds the final counts.

### Docker (optional)

`Dockerfile` builds the backend only and serves it with gunicorn on port 7860. It copies only `backend/`, so it sets `SERVE_MODEL=vit` and serves the ViT baseline; it expects that checkpoint inside `backend/models/` at build time, or `MODEL_URL` at runtime. `space/` (the Hugging Face Space copy) is unchanged and serves the ViT baseline as before.

---

## Configuration (.env)

All settings are optional. Both `backend/app.py` and `streamlit_app.py` read `<repo>/.env`; real environment variables take precedence. See `.env.example`.

| Variable | Default | Used by |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | backend |
| `SERVE_MODEL` | `finetuned` (`vit` in the Docker image) | backend: which model `/predict` serves |
| `FINETUNED_CHECKPOINT` | `results/ffpp_c23/e2e_finetune_aug-frame_ffpp+celebdf/seed0/checkpoints/best_trainable.pth` | backend (`SERVE_MODEL=finetuned`) |
| `MODEL_PATH` | `backend/models/best_vit_model.pth` | backend (`SERVE_MODEL=vit`) |
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

### Research pipeline (FaceForensics++ c23, Chapter 16)

Measured with the scripts above. Video level; the positive class is fake. Splits are the official
source-family-disjoint FF++ splits. Models and thresholds were chosen on the validation split only. The test
split and Celeb-DF were each scored once, under the protocol in
[`docs/final_evaluation_protocol.md`](docs/final_evaluation_protocol.md), written before any test score
existed.

**Second-domain model** (README Step 9, served by the app; scored once like the others):

| Data | FF++-only model | FF++ + Celeb-DF v2 model |
| --- | ---: | ---: |
| FF++ test AUC | 0.853 | **0.874** (0.851–0.899) |
| Celeb-DF v1 test AUC | 0.837 (unseen dataset) | **0.980** (0.949–0.998), same dataset |
| Celeb-DF v2 test AUC | 0.781 (unseen dataset) | **0.956** (0.938–0.973), same dataset |
| Real Celeb-real test videos called fake | 50% | 21% |
| Real FF++ test videos called fake | 19% | 31% |
| **DFD test AUC (unseen dataset, 3,375 videos)** | 0.759 (0.729–0.786) | **0.814** (0.779–0.847) |
| DFD: fakes caught / real videos called fake | 59% / 20% | 65% / 18% |

**Calibration of the served model** (`scripts/calibrate_served.py`, fitted on validation, applied once to the stored test scores; class-balanced expected calibration error, lower is better):

| | Validation | FF++ test | Celeb-DF v1 | Celeb-DF v2 | DFD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw score | 0.092 | 0.159 | 0.106 | 0.109 | 0.127 |
| Platt-calibrated | 0.044 (cross-fitted) | 0.138 | 0.141 | 0.126 | 0.097 |

Calibration helps on validation, FF++ and DFD but not on Celeb-DF, so the shown probability is approximate. It does not fix the FF++ trade-off below: one threshold cannot serve both datasets better than 0.816 does on validation (the calibrated rule would lower FF++ validation specificity from 0.764 to 0.707).

The second model's threshold (0.816) was chosen on FF++ val + Celeb-DF val. It ranks better everywhere, but at that threshold it trades FF++ specificity (0.807 to 0.693) for Celeb-DF specificity (0.554 to 0.825). Its Celeb-DF numbers are same-dataset results. Its unseen-dataset score comes from FaceForensics++'s DeepFakeDetection set (DFD: 28 paid actors in scripted scenes, never used for training, selection or thresholds): 0.814 AUC, against 0.648 ± 0.004 for C and 0.636 ± 0.021 for the full hybrid F. At its threshold it misses 35% of DFD's fakes, so on new footage a "real" verdict is weak evidence.

**Final evaluation, FF++-only models** (`scripts/final_eval.py`; thresholds frozen from FF++ validation):

| Data | Model | AUC | Balanced acc | Fake recall | Specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| FF++ test (839 videos) | Fine-tuned ResNet50+ViT, per-frame aug. | **0.853** (95 % CI 0.823–0.886) | 0.795 | 0.783 | 0.807 |
| | C: ResNet50+ViT, frozen features | 0.752 ± 0.004 | 0.672 | 0.729 | 0.614 |
| | F: full hybrid, frozen features | 0.745 ± 0.009 | 0.676 | 0.732 | 0.619 |
| Celeb-DF v1 (100) | Fine-tuned ResNet50+ViT, per-frame aug. | **0.837** (0.740–0.922) | 0.741 | 0.903 | 0.579 |
| | C | 0.704 ± 0.027 | 0.572 | 0.968 | 0.175 |
| | F | 0.658 ± 0.007 | 0.532 | 0.914 | 0.149 |
| Celeb-DF v2 (516) | Fine-tuned ResNet50+ViT, per-frame aug. | **0.781** (0.738–0.821) | 0.718 | 0.882 | 0.554 |
| | C | 0.627 ± 0.013 | 0.578 | 0.881 | 0.275 |
| | F | 0.609 ± 0.015 | 0.565 | 0.861 | 0.269 |

C and F: mean (± std for AUC) over 3 seeds. The final model is one training run (seed 0) on all 360 training
families; its CI comes from resampling source families (FF++) or identities (Celeb-DF). FF++ test AUC per
method for the final model: Deepfakes 0.929, Face2Face 0.880, FaceShifter 0.859, FaceSwap 0.847,
NeuralTextures 0.750.

**What the experiments showed** (validation split unless stated; details in `results/ffpp_c23/summary/`):

* **Fine-tuning is the change that matters.** Unfreezing ResNet50 `layer4` and the last two ViT blocks raised
  validation AUC from 0.670 to 0.774–0.796 on the same 120-family subset, and to 0.867 on all training data.
  It holds on the test split (0.853 vs 0.752) and across datasets (Celeb-DF v2 0.781 vs 0.627).
* **The original hybrid's extra parts add nothing on frozen features.** Mean-pooled ResNet50+ViT (row C, 0.73M
  trainable parameters) matches or beats the full hybrid with FFT branch and BiLSTM (row F, 4.96M) on
  validation (0.754 vs 0.745), on the test split and on Celeb-DF.
* **The BiLSTM does not use frame order.** Shuffling frames at evaluation changes AUC by less than 0.001.
  Window length 4, 8 or 16 makes no difference.
* **Face cropping is worth about 0.17 AUC** (0.750 vs 0.577 on full frames).
* **Unseen manipulation methods lose 0.07–0.14 AUC** (leave-one-method-out, frozen features).
* **Raw probabilities are miscalibrated** (ECE 0.21–0.28); Platt scaling cross-fitted on validation brings ECE
  to 0.01–0.03.
* **Blur and downscaling break the frozen-feature models' threshold**: specificity drops from 0.69 to 0.17 at
  mild blur. The fine-tuned model degrades less (validation AUC 0.867 clean, 0.812 blurred, 0.810 at half
  resolution).
* **Section 16.4 head changes did not help**: branch projections, a learned softmax gate (no better than its
  fixed-weight ablation, and unstable across seeds), a causal LSTM (−0.006 AUC, half the parameters), and FFT
  band variants were all within noise of their reference rows. EfficientNet-B0 beat ResNet50 as a single
  encoder (+0.050 AUC) but not ResNet50+ViT.
* **Sequence-consistent augmentation gave no advantage** over per-frame augmentation (validation AUC 0.781
  vs 0.796, one seed each).
* **Cross-dataset thresholds transfer badly.** On Celeb-DF the frozen-feature models call most real videos
  fake (specificity 0.15–0.28); the fine-tuned model keeps 0.55–0.58. The FF++ threshold is not a reliable
  operating point for new data.

Compute (GPU, 32 frames): about 1 s per video end to end, dominated by video decoding and face detection;
CPU about 6 s, dominated by the ViT.

The hybrid notebook (`notebooks/`) is unchanged and still writes to
`results/hybrid_cnn_vit_fft_bilstm/`; no result from it is recorded.

---

## Limitations

* **The local app serves the fine-tuned ResNet50 + ViT trained on FF++ + Celeb-DF v2, not the full hybrid.** The full hybrid (FFT branch + BiLSTM) scored no better than the simpler model on every test set, so it is not served. The served model reads only the middle ~10 seconds of a video and needs a visible face. On held-out test videos it still calls 21% of real Celeb-real interviews and 31% of real FF++ videos fake (the FF++-only model: 50% and 19%). On DFD, a dataset it never saw, it reaches 0.814 AUC but misses 35% of the fakes and calls 18% of the real videos fake; other sources may differ again. The Docker image and `space/` still serve the ViT baseline.
* **Domain shift:** on 44 real and 44 fake DFDC videos (not part of training), the deployed ViT separates the classes only weakly (mean P(real) 0.67 on real vs 0.58 on fake videos). Treat predictions as an AI-assisted signal, not proof.
* The ViT baseline numbers come from a frame-level split and may be inflated by leakage between frames of the same video.
* No dataset, and no trained hybrid checkpoint, is shipped. The training notebooks cannot run until `data/FF_frames` exists.
* `deepfake_detection_vit_best.ipynb` is an archived Colab experiment and does not run as-is.
* The Flask development server (`python backend/app.py`) is for local use. Use gunicorn/Docker (Linux) for anything public.
* Model performance also varies with resolution, compression, lighting, face quality and unseen manipulation methods.
* Research pipeline: each fine-tuned model is a single training run (one seed); its test numbers carry the bootstrap CIs shown, not seed variance. The checkpoints under `results/ffpp_c23/` are local only (git-ignored). Celeb-DF v1 has only 100 test videos, so its CI is wide.

---

## License

This project is intended for educational, research and experimental purposes.

Shubham Kumar
