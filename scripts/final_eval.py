"""Final evaluation (Phase 8): score the pre-registered models ONCE on held-out data.

The models, data and protocol are fixed in docs/final_evaluation_protocol.md before this script
is run. Nothing here trains, selects or tunes: every model keeps the decision threshold it chose
on the FF++ validation split, and the scores are reported as they come out.

Datasets
  ffpp_test   FaceForensics++ c23 official test split (same source domain as training)
  celebdf_v1  Celeb-DF v1 official test list (cross-dataset, never seen in any form)
  celebdf_v2  Celeb-DF v2 official test list (cross-dataset, never seen in any form)

Models (see MODELS)
  final       fine-tuned ResNet50 + ViT with per-frame augmentation, all 360 train families
  C           ResNet50 + ViT, mean over time, frozen features (simplest reliable baseline), 3 seeds
  F           original full hybrid (ResNet50 + ViT + FFT + BiLSTM), frozen features, 3 seeds

Writes  results/final/<dataset>/<model>[_seed<S>].json  (refuses to overwrite: single evaluation)
        results/final/final_table.md

usage:  python scripts/final_eval.py [--datasets ffpp_test celebdf_v1 celebdf_v2] [--models final C F]
"""
import argparse
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_e2e import FRAMES, T, Net  # noqa: E402
from deepshield import engine, metrics, paths  # noqa: E402
from deepshield.data import FeatureStore, WindowBatches, window_starts  # noqa: E402

RESULTS = paths.RESULTS_ROOT / "ffpp_c23"
OUT = paths.RESULTS_ROOT / "final"
DATASETS = {"ffpp_test": "ffpp_c23", "celebdf_v1": "celebdf_v1", "celebdf_v2": "celebdf_v2"}
MODELS = {
    "final": [RESULTS / "e2e_finetune_aug-frame_final" / "seed0"],
    # amendment (docs/final_evaluation_protocol.md, "Second-domain model"): FF++ + Celeb-DF v2 train
    "final_cdf": [RESULTS / "e2e_finetune_aug-frame_ffpp+celebdf" / "seed0"],
    "C": [RESULTS / "C_faces_T8" / f"seed{s}" for s in range(3)],
    "F": [RESULTS / "F_faces_T8" / f"seed{s}" for s in range(3)],
}
LABELS = {"final": "Fine-tuned ResNet50+ViT, per-frame aug. (final)",
          "final_cdf": "Same, trained on FF++ + Celeb-DF v2 train", "C": "C: ResNet50+ViT, mean (frozen)",
          "F": "F: full hybrid (frozen)"}
KEYS = ("roc_auc", "balanced_accuracy", "fake_recall", "specificity")


def test_rows(dataset, in_memory=True):
    store = FeatureStore("faces", dataset=DATASETS[dataset], in_memory=in_memory)
    return store, store.split_rows("test", allow_test=True)


@torch.no_grad()
def score_head(run_dir, store, rows, device):
    model, ckpt = engine.load_checkpoint(run_dir / "checkpoints" / "best.pth", device)
    dc = ckpt["data_config"]
    batches = WindowBatches(store, rows, seq_len=dc["seq_len"], stride=dc["stride"], batch_size=32,
                            use_gray=ckpt["model_config"]["use_fft"], device=device)
    p, _ = engine.predict(model, batches, None, ckpt["fake_index"])
    return p, ckpt["threshold"]


@torch.no_grad()
def score_e2e(run_dir, dataset, rows, device):
    ckpt = torch.load(run_dir / "checkpoints" / "best_trainable.pth", map_location=device, weights_only=True)
    if ckpt["class_names"] != paths.CLASS_NAMES:
        raise ValueError(f"{run_dir}: class_names {ckpt['class_names']} != {paths.CLASS_NAMES}")
    model = Net(ckpt["mode"]).to(device)
    missing = set(ckpt["trainable_state_dict"]) - set(model.state_dict())
    if missing:
        raise ValueError(f"unexpected keys in checkpoint: {sorted(missing)[:5]}")
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model.eval()
    faces = paths.DATASETS[DATASETS[dataset]]["data_dir"] / "faces"
    fake_index, starts, pool = ckpt["fake_index"], window_starts(FRAMES, T), ThreadPoolExecutor(12)

    def load_video(i):
        files = sorted((faces / rows[i]["video_id"]).glob("*.jpg"))
        if len(files) != FRAMES:
            raise RuntimeError(f"{rows[i]['video_id']}: {len(files)} frames, expected {FRAMES}")
        return np.stack([cv2.imread(str(f))[:, :, ::-1] for f in files])

    p_video = np.zeros(len(rows))
    for b in range(0, len(rows), 4):
        ids = list(range(b, min(b + 4, len(rows))))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(list(pool.map(load_video, ids))))).to(device)
        flat = x.reshape(-1, 224, 224, 3).permute(0, 3, 1, 2).float().div(255)
        flat = (flat - model.mean) / model.std
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            c, v = model.cnn(flat).float(), model.vit(flat).float()
        c, v = c.reshape(len(ids), FRAMES, -1), v.reshape(len(ids), FRAMES, -1)
        probs = [torch.softmax(model.head({"cnn": c[:, s:s + T], "vit": v[:, s:s + T]}), dim=1)[:, fake_index]
                 for s in starts]
        p_video[ids] = torch.stack(probs, dim=1).mean(dim=1).cpu().numpy()
    return p_video, ckpt["threshold"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, run_dirs in MODELS.items():
        if name not in args.models:
            continue
        for d in run_dirs:
            if not d.is_dir():
                sys.exit(f"Missing pre-registered run {d}")

    for ds in args.datasets:
        store, rows = test_rows(ds)
        y, methods, families = engine.rows_to_arrays(rows, paths.CLASS_NAMES.index("fake"))
        print(f"\n== {ds}: {len(rows)} videos ({int(y.sum())} fake / {int((1 - y).sum())} real)", flush=True)
        for name in args.models:
            for run_dir in MODELS[name]:
                tag = name if len(MODELS[name]) == 1 else f"{name}_{run_dir.name}"
                out = OUT / ds / f"{tag}.json"
                if out.exists():
                    print(f"{tag}: already evaluated ({out}), not re-run")
                    continue
                if name.startswith("final"):
                    p, thr = score_e2e(run_dir, ds, rows, device)
                else:
                    p, thr = score_head(run_dir, store, rows, device)
                rep = metrics.video_report(y, p, methods, thr)
                rep.update({"dataset": ds, "model": name, "run_dir": str(run_dir),
                            "threshold_source": "FF++ c23 validation split (frozen)",
                            "mean_p_fake": float(p.mean()),
                            "ci95_family_bootstrap": metrics.bootstrap_ci(y, p, families, thr)})
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rep, indent=2) + "\n")
                np.save(out.with_suffix(".npy"), p)
                print(f"{tag}: AUC {rep['roc_auc']:.4f} bal acc {rep['balanced_accuracy']:.3f} "
                      f"fake recall {rep['fake_recall']:.3f} specificity {rep['specificity']:.3f}", flush=True)
        del store
    write_table()


def write_table():
    lines = ["Final evaluation, video level. Thresholds were chosen on each model's validation split(s) and frozen",
             "(FF++ val; final_cdf: FF++ val + Celeb-DF v2 val). After final_cdf's training, Celeb-DF is a same-dataset test.",
             "One evaluation per model and dataset. C and F: mean ± std over 3 seeds; final: one model with a",
             "95% family-bootstrap CI for AUC.", "",
             "| Dataset | Model | AUC | Balanced acc | Fake recall | Specificity | Mean P(fake) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for ds in DATASETS:
        for name, run_dirs in MODELS.items():
            files = [OUT / ds / (f"{name}.json" if len(run_dirs) == 1 else f"{name}_{d.name}.json") for d in run_dirs]
            reps = [json.loads(f.read_text()) for f in files if f.exists()]
            if not reps:
                continue
            if len(reps) == 1:
                r, ci = reps[0], reps[0]["ci95_family_bootstrap"]["roc_auc"]
                auc = f"{r['roc_auc']:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"
                cells = [f"{r[k]:.3f}" for k in KEYS[1:]] + [f"{r['mean_p_fake']:.3f}"]
            else:
                def ms(k):
                    v = [r[k] for r in reps]
                    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
                auc = ms("roc_auc") + (f" (n={len(reps)})" if len(reps) < len(run_dirs) else "")
                cells = [ms(k) for k in KEYS[1:]] + [ms("mean_p_fake")]
            lines.append(f"| {ds} | {LABELS[name]} | {auc} | " + " | ".join(cells) + " |")
    lines += ["", "Per-method AUC on the FF++ test split (each method vs. all test reals):", "",
              "| Model | " + " | ".join(["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]) + " |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, run_dirs in MODELS.items():
        files = [OUT / "ffpp_test" / (f"{name}.json" if len(run_dirs) == 1 else f"{name}_{d.name}.json")
                 for d in run_dirs]
        reps = [json.loads(f.read_text()) for f in files if f.exists()]
        if reps:
            cells = [f"{st.mean(r['per_method'][m]['roc_auc'] for r in reps):.3f}"
                     for m in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]]
            lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_table.md").write_text(md, encoding="utf-8")
    print("\nSaved ->", OUT / "final_table.md")
    print(md.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
