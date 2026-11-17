"""Re-evaluate a trained run on the validation split with its frozen threshold.

Used for diagnostics that must not retrain or retune anything:
  --shuffle-frames   shuffle the frame order inside every window (ablation G);
                     repeated --repeats times with different shuffles
Also measures head inference time per video (backbone feature extraction excluded).

Writes  <run_dir>/metrics/val_eval[_shuffled].json

usage:  python scripts/evaluate.py --run-dir results/ffpp_c23/F_faces_T8/seed0 --shuffle-frames
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import engine, metrics  # noqa: E402
from deepshield.data import FeatureStore, WindowBatches  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--shuffle-frames", action="store_true")
    ap.add_argument("--repeats", type=int, default=5, help="number of different shuffles")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = engine.load_checkpoint(args.run_dir / "checkpoints" / "best.pth", device)
    dc, threshold = ckpt["data_config"], ckpt["threshold"]
    store = FeatureStore(dc["variant"], cnn_backbone=ckpt["model_config"].get("cnn_backbone", "resnet50"))
    rows = store.split_rows("val")
    y, methods, _ = engine.rows_to_arrays(rows, ckpt["fake_index"])

    def run(shuffle, seed):
        batches = WindowBatches(store, rows, seq_len=dc["seq_len"], stride=dc["stride"], batch_size=args.batch_size,
                                use_gray=ckpt["model_config"]["use_fft"], shuffle_frames=shuffle, seed=seed, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        p, _ = engine.predict(model, batches, None, ckpt["fake_index"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        return p, time.time() - t0

    p_ordered, seconds = run(False, 0)
    ordered = metrics.video_report(y, p_ordered, methods, threshold)
    out = {"run_dir": str(args.run_dir), "split": "val", "threshold_from_checkpoint": threshold,
           "ordered": {k: ordered[k] for k in ("roc_auc", "balanced_accuracy", "macro_f1", "fake_recall", "specificity")},
           "head_inference_ms_per_video": round(1000 * seconds / len(rows), 3),
           "head_inference_note": "includes reading cached features; excludes ResNet50/ViT feature extraction",
           "device": str(device)}
    name = "val_eval.json"
    if args.shuffle_frames:
        reps = []
        for s in range(args.repeats):
            p, _ = run(True, 1000 + s)
            r = metrics.video_report(y, p, methods, threshold)
            reps.append({k: r[k] for k in ("roc_auc", "balanced_accuracy", "macro_f1", "fake_recall", "specificity")})
        mean = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
        out["shuffled_mean"] = mean
        out["shuffled_repeats"] = reps
        out["delta_shuffled_minus_ordered"] = {k: mean[k] - out["ordered"][k] for k in mean}
        name = "val_eval_shuffled.json"
    path = args.run_dir / "metrics" / name
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("Saved ->", path)
    print("ordered AUC %.4f | %.2f ms/video" % (out["ordered"]["roc_auc"], out["head_inference_ms_per_video"]),
          ("| shuffled AUC %.4f (delta %+.4f)" % (out["shuffled_mean"]["roc_auc"],
                                                   out["delta_shuffled_minus_ordered"]["roc_auc"])
           if args.shuffle_frames else ""))


if __name__ == "__main__":
    main()
