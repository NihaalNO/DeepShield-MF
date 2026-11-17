"""Train one head on cached FF++ features and evaluate it per video on validation.

Uses the train split for fitting and the validation split for model selection,
the decision threshold and the reported metrics. The test split is never read.

Writes  results/ffpp_c23/<run_name>/seed<S>/
            config.json                       full configuration + environment
            checkpoints/best.pth              model + class_names + threshold (see deepshield/engine.py)
            logs/history.csv                  per-epoch train loss, val loss, val AUC, lr, seconds
            metrics/val_metrics.json          video-level metrics, per-method breakdown, bootstrap CIs
            predictions/val_videos.csv        one row per validation video

usage:  python scripts/train.py --exp F --seed 0
        python scripts/train.py --exp E --seq-len 16 --variant fullframe --name E_fullframe_T16
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import engine, metrics, paths  # noqa: E402
from deepshield.data import FeatureStore, WindowBatches  # noqa: E402
from deepshield.models import EXPERIMENTS, HybridHead, count_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS), help="experiment row: A-F ablations, I-M3 improvements (deepshield/models.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=["faces", "fullframe"], default="faces")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=0, help="window stride in cached frames (default seq_len/2)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32, help="windows per batch (32 keeps the FFT branch within 6 GB VRAM)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=4, help="early stopping on validation video AUC")
    ap.add_argument("--train-methods", nargs="*", default=None,
                    help="restrict training fakes to these methods (leave-one-method-out)")
    ap.add_argument("--limit-families", type=int, default=0, help="use only N families per split (smoke test)")
    ap.add_argument("--name", default="", help="run name (default: <exp>_<variant>_T<seq_len>)")
    ap.add_argument("--out-root", type=Path, default=paths.RESULTS_ROOT / "ffpp_c23")
    args = ap.parse_args()

    run_name = args.name or f"{args.exp}_{args.variant}_T{args.seq_len}"
    run_dir = args.out_root / run_name / f"seed{args.seed}"
    dirs = {k: run_dir / k for k in ("checkpoints", "logs", "metrics", "predictions")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine.set_seed(args.seed)

    model_config = dict(EXPERIMENTS[args.exp])
    store = FeatureStore(args.variant, cnn_backbone=model_config.get("cnn_backbone", "resnet50"))
    train_rows = store.split_rows("train", limit_families=args.limit_families, methods=args.train_methods)
    val_rows = store.split_rows("val", limit_families=args.limit_families)
    fake_index = paths.CLASS_NAMES.index("fake")
    stride = args.stride or None
    common = dict(seq_len=args.seq_len, stride=stride, batch_size=args.batch_size,
                  use_gray=model_config["use_fft"], device=device)
    train_batches = WindowBatches(store, train_rows, shuffle=True, seed=args.seed, **common)
    val_batches = WindowBatches(store, val_rows, **common)

    model = HybridHead(**model_config, num_classes=len(paths.CLASS_NAMES)).to(device)
    counts = np.bincount(train_batches.labels, minlength=len(paths.CLASS_NAMES))
    class_weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6)

    data_config = {"dataset": "FaceForensics++ c23", "variant": args.variant, "seq_len": args.seq_len,
                   "stride": stride or max(1, args.seq_len // 2), "windows_per_video": val_batches.windows_per_video,
                   "frames_per_video": store.n_frames, "backbones": store.meta["backbones"],
                   "normalization": store.meta["normalization"], "train_methods": args.train_methods}
    config = {"run_name": run_name, "experiment": args.exp, "args": {k: str(v) if isinstance(v, Path) else v
                                                                      for k, v in vars(args).items()},
              "model_config": model_config, "data_config": data_config, "class_names": paths.CLASS_NAMES,
              "trainable_params": count_params(model),
              "train": {"videos": len(train_rows), "windows": len(train_batches.samples),
                        "window_class_counts": dict(zip(paths.CLASS_NAMES, counts.tolist())),
                        "class_weights": class_weights.tolist()},
              "val": {"videos": len(val_rows)},
              "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "torch": torch.__version__}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Run        : {run_name} seed {args.seed} | {model_config}")
    print(f"Outputs -> : {run_dir}")
    print(f"Train {len(train_rows)} videos / {len(train_batches.samples)} windows | val {len(val_rows)} videos | "
          f"trainable params {config['trainable_params']:,} | device {device}", flush=True)

    y_val, methods_val, families_val = engine.rows_to_arrays(val_rows, fake_index)
    # Leave-one-method-out: model selection and the threshold may only see validation videos of the
    # training methods (plus reals), otherwise the held-out method would leak into the selection.
    sel = np.ones(len(val_rows), bool)
    if args.train_methods is not None:
        sel = np.isin(methods_val, ["original", *args.train_methods])
    ckpt_path = dirs["checkpoints"] / "best.pth"
    history, best_auc, best_epoch, bad = [], -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = engine.train_epoch(model, train_batches, criterion, optimizer)
        p_val, val_loss = engine.predict(model, val_batches, criterion, fake_index)
        val_auc = metrics.safe_auc(y_val[sel], p_val[sel])
        scheduler.step(val_auc)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
                        "val_video_auc": round(val_auc, 5), "lr": optimizer.param_groups[0]["lr"],
                        "seconds": round(time.time() - t0, 1)})
        with open(dirs["logs"] / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(history)
        improved = val_auc > best_auc + 1e-4
        print(f"epoch {epoch:2d} train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_video_auc {val_auc:.4f} "
              f"{history[-1]['seconds']:.0f}s{' *' if improved else ''}", flush=True)
        if improved:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            threshold = metrics.pick_threshold(y_val[sel], p_val[sel])
            engine.save_checkpoint(ckpt_path, model, model_config, data_config, paths.CLASS_NAMES, threshold,
                                   extra={"run_name": run_name, "seed": args.seed, "epoch": epoch, "val_video_auc": val_auc})
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break
    train_minutes = (time.time() - t_start) / 60

    # ---- final validation report from the saved best checkpoint ----------------
    model, ckpt = engine.load_checkpoint(ckpt_path, device)
    p_val, val_loss = engine.predict(model, val_batches, criterion, ckpt["fake_index"])
    threshold = ckpt["threshold"]
    report = metrics.video_report(y_val, p_val, methods_val, threshold)
    report.update({"split": "val", "note": "threshold and model selection used this split; not a held-out estimate",
                   "best_epoch": best_epoch, "epochs_run": len(history), "val_loss": val_loss,
                   "train_minutes": round(train_minutes, 2), "trainable_params": config["trainable_params"],
                   "selection_roc_auc": metrics.safe_auc(y_val[sel], p_val[sel]),
                   "selection_videos": int(sel.sum()), "train_methods": args.train_methods,
                   "ci95_family_bootstrap": metrics.bootstrap_ci(y_val, p_val, families_val, threshold),
                   "checkpoint": str(ckpt_path)})
    (dirs["metrics"] / "val_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(dirs["predictions"] / "val_videos.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video_id", "method", "family_id", "label", "p_fake", "pred"])
        for r, p in zip(val_rows, p_val):
            w.writerow([r["video_id"], r["method"], r["family_id"], paths.CLASS_NAMES[r["label_idx"]],
                        f"{p:.6f}", "fake" if p >= threshold else "real"])
    print("Saved ->", ckpt_path)
    print("Saved ->", dirs["logs"] / "history.csv")
    print("Saved ->", dirs["metrics"] / "val_metrics.json")
    print("Saved ->", dirs["predictions"] / "val_videos.csv")
    print(f"VAL (video level, best epoch {best_epoch}): AUC {report['roc_auc']:.4f} | balanced acc "
          f"{report['balanced_accuracy']:.4f} | macro F1 {report['macro_f1']:.4f} | fake recall "
          f"{report['fake_recall']:.4f} | specificity {report['specificity']:.4f} | threshold {threshold:.3f}")


if __name__ == "__main__":
    main()
