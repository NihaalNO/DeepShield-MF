"""End-to-end fine-tuning on a subset of FF++ c23, with optional augmentation (Chapter 16.4).

Augmentation changes pixels, so it cannot use the cached features. This script trains from the
saved face crops instead: ResNet50 + ViT-B/16 -> row-C head (concatenate, mean over frames).

  --mode frozen     backbones frozen, only the head trains (same model as row C, pixel pipeline)
  --mode finetune   additionally trains ResNet50 layer4 and the last two ViT blocks (BatchNorm
                    statistics stay frozen)
  --aug none        no augmentation
  --aug sequence    sequence-consistent: one random draw per window, applied to all its frames
  --aug frame       the same augmentations drawn independently for every frame
Augmentations: horizontal flip, random crop (80-100 % of the side), brightness and contrast
(+-20 %), JPEG re-compression (quality 40-100, p 0.5), Gaussian blur (sigma up to 1.5, p 0.3).

Training uses the first --train-families source families of the train split (2 random windows
per video and epoch); validation is the full validation split with the usual protocol (7 windows
of 8 frames, mean P(fake), threshold chosen on validation). After training, the best epoch is
also scored on degraded validation crops (see scripts/robustness.py) with its frozen threshold.

Writes  results/ffpp_c23/<name>/seed<S>/{config.json, checkpoints/best_trainable.pth,
        logs/history.csv, metrics/val_metrics.json, metrics/val_robustness.json}

usage:  python scripts/finetune_e2e.py --mode finetune --aug sequence [--train-families 120] [--epochs 8]
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness import degrade  # noqa: E402
from deepshield import backbones, engine, metrics, paths  # noqa: E402
from deepshield.data import FeatureStore, window_starts  # noqa: E402
from deepshield.models import EXPERIMENTS, HybridHead  # noqa: E402

T, FRAMES = 8, 32
ROBUST = [("clean", 0), ("jpeg", 40), ("blur", 1), ("downscale", 2), ("noise", 10), ("brightness", 0.6)]


def draw_params(rng):
    scale = rng.uniform(0.8, 1.0)
    side = int(round(224 * scale))
    return {"flip": rng.random() < 0.5, "side": side, "x0": int(rng.integers(0, 224 - side + 1)),
            "y0": int(rng.integers(0, 224 - side + 1)), "brightness": rng.uniform(0.8, 1.2),
            "contrast": rng.uniform(0.8, 1.2), "jpeg": int(rng.integers(40, 101)) if rng.random() < 0.5 else 0,
            "blur": rng.uniform(0.3, 1.5) if rng.random() < 0.3 else 0.0}


def apply_aug(bgr, p):
    img = bgr[p["y0"]:p["y0"] + p["side"], p["x0"]:p["x0"] + p["side"]]
    if p["side"] != 224:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    if p["flip"]:
        img = img[:, ::-1]
    f = img.astype(np.float32)
    f = (f - f.mean()) * p["contrast"] + f.mean() * p["brightness"]
    img = np.clip(f, 0, 255).astype(np.uint8)
    if p["blur"]:
        img = cv2.GaussianBlur(img, (0, 0), p["blur"])
    if p["jpeg"]:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])[1], cv2.IMREAD_COLOR)
    return img


class Net(nn.Module):
    def __init__(self, mode):
        super().__init__()
        nets = backbones.load("cpu")
        self.cnn, self.vit = nets["resnet50"], nets["vit_base_patch16_224"]
        self.head = HybridHead(**EXPERIMENTS["C"])
        for p in list(self.cnn.parameters()) + list(self.vit.parameters()):
            p.requires_grad = False
        self.tuned = []
        if mode == "finetune":
            self.tuned = [self.cnn.layer4, self.vit.blocks[-2], self.vit.blocks[-1], self.vit.norm]
            for m in self.tuned:
                for p in m.parameters():
                    p.requires_grad = True
        self.register_buffer("mean", torch.tensor(backbones.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(backbones.STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.cnn.eval()  # BatchNorm statistics and drop-path of the backbones stay in inference mode
        self.vit.eval()
        return self

    def forward(self, rgb):  # uint8 [B, T, 224, 224, 3]
        b, t = rgb.shape[:2]
        x = rgb.reshape(b * t, 224, 224, 3).permute(0, 3, 1, 2).float().div(255)
        x = (x - self.mean) / self.std
        with torch.autocast(x.device.type, enabled=x.device.type == "cuda"):
            c, v = self.cnn(x), self.vit(x)
        return self.head({"cnn": c.float().reshape(b, t, -1), "vit": v.float().reshape(b, t, -1)})

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in keep}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["frozen", "finetune"], default="finetune")
    ap.add_argument("--aug", choices=["none", "sequence", "frame"], default="none")
    ap.add_argument("--train-families", type=int, default=120)
    ap.add_argument("--val-videos", type=int, default=0, help="only N validation videos (testing)")
    ap.add_argument("--windows-per-video", type=int, default=2, help="random training windows per video and epoch")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4, help="windows per batch (x8 frames)")
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    name = args.name or f"e2e_{args.mode}_aug-{args.aug}_fam{args.train_families}"
    run_dir = paths.RESULTS_ROOT / "ffpp_c23" / name / f"seed{args.seed}"
    for d in ("checkpoints", "logs", "metrics"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    store = FeatureStore("faces", in_memory=False)  # only for the manifest rows and splits
    train_rows = store.split_rows("train", limit_families=args.train_families)
    val_rows = store.split_rows("val")
    if args.val_videos:
        val_rows = val_rows[:: max(1, len(val_rows) // args.val_videos)][: args.val_videos]
    fake_index = paths.CLASS_NAMES.index("fake")
    y_val, methods_val, families_val = engine.rows_to_arrays(val_rows, fake_index)
    files = {r["video_id"]: sorted((paths.FFPP_FACES_DIR / r["video_id"]).glob("*.jpg")) for r in train_rows + val_rows}
    pool = ThreadPoolExecutor(12)

    def load_window(job):
        video_id, start, params = job  # params: None | one dict | list of T dicts
        out = []
        for k, f in enumerate(files[video_id][start:start + T]):
            img = cv2.imread(str(f))
            if params is not None:
                img = apply_aug(img, params[k] if isinstance(params, list) else params)
            out.append(img[:, :, ::-1])
        return np.stack(out)

    def aug_params():
        if args.aug == "none":
            return None
        return draw_params(rng) if args.aug == "sequence" else [draw_params(rng) for _ in range(T)]

    model = Net(args.mode).to(device)
    labels = np.array([r["label_idx"] for r in train_rows])
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    groups = [{"params": model.head.parameters(), "lr": args.lr_head}]
    if model.tuned:
        groups.append({"params": [p for m in model.tuned for p in m.parameters()], "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = {"run_name": name, "args": vars(args), "head": EXPERIMENTS["C"], "class_names": paths.CLASS_NAMES,
              "tuned_modules": ["resnet50.layer4", "vit.blocks[-2:]", "vit.norm"] if model.tuned else [],
              "trainable_params": n_train, "train": {"videos": len(train_rows), "families": args.train_families,
                                                     "class_counts": dict(zip(paths.CLASS_NAMES, counts.tolist()))},
              "val": {"videos": len(val_rows)}, "device": str(device), "torch": torch.__version__}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Run {name} seed {args.seed} | train {len(train_rows)} videos | val {len(val_rows)} videos | "
          f"trainable params {n_train:,} | device {device}", flush=True)

    @torch.no_grad()
    def predict_val(kind="clean", level=0):
        model.eval()
        starts = window_starts(FRAMES, T)
        p_video = np.zeros(len(val_rows))

        def load_video(i):
            noise = np.random.default_rng(i)
            imgs = [degrade(cv2.imread(str(f)), kind, level, noise)[:, :, ::-1] for f in files[val_rows[i]["video_id"]]]
            return np.stack(imgs)

        for b in range(0, len(val_rows), 4):
            ids = list(range(b, min(b + 4, len(val_rows))))
            vids = np.stack(list(pool.map(load_video, ids)))  # [V, 32, 224, 224, 3]
            x = torch.from_numpy(np.ascontiguousarray(vids)).to(device)
            # mean pooling over frames: score each window from the 32 frames' features
            flat = x.reshape(-1, 224, 224, 3).permute(0, 3, 1, 2).float().div(255)
            flat = (flat - model.mean) / model.std
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                c, v = model.cnn(flat).float(), model.vit(flat).float()
            c, v = c.reshape(len(ids), FRAMES, -1), v.reshape(len(ids), FRAMES, -1)
            probs = [torch.softmax(model.head({"cnn": c[:, s:s + T], "vit": v[:, s:s + T]}), dim=1)[:, fake_index]
                     for s in starts]
            p_video[ids] = torch.stack(probs, dim=1).mean(dim=1).cpu().numpy()
        return p_video

    ckpt_path = run_dir / "checkpoints" / "best_trainable.pth"
    history, best_auc, best_epoch, bad, threshold = [], -1.0, 0, 0, 0.5
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        jobs = [(r["video_id"], int(rng.integers(0, FRAMES - T + 1)), aug_params(), r["label_idx"])
                for r in train_rows for _ in range(args.windows_per_video)]
        order = rng.permutation(len(jobs))
        total, n = 0.0, 0
        for b in range(0, len(order), args.batch_size):
            part = [jobs[i] for i in order[b:b + args.batch_size]]
            x = torch.from_numpy(np.stack(list(pool.map(load_window, [j[:3] for j in part])))).to(device)
            y = torch.tensor([j[3] for j in part], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(part)
            n += len(part)
        t_train = time.time() - t0
        p_val = predict_val()
        val_auc = metrics.safe_auc(y_val, p_val)
        improved = val_auc > best_auc + 1e-4
        history.append({"epoch": epoch, "train_loss": round(total / n, 5), "val_video_auc": round(val_auc, 5),
                        "train_seconds": round(t_train, 1), "seconds": round(time.time() - t0, 1)})
        with open(run_dir / "logs" / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(history)
        print(f"epoch {epoch:2d} train_loss {total / n:.4f} val_video_auc {val_auc:.4f} "
              f"{history[-1]['seconds']:.0f}s{' *' if improved else ''}", flush=True)
        if improved:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            threshold = metrics.pick_threshold(y_val, p_val)
            torch.save({"format": "deepshield-e2e-trainable/1", "trainable_state_dict": model.trainable_state(),
                        "mode": args.mode, "head_config": EXPERIMENTS["C"], "class_names": paths.CLASS_NAMES,
                        "fake_index": fake_index, "threshold": threshold, "epoch": epoch}, ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break
    train_minutes = (time.time() - t_start) / 60

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    threshold = ckpt["threshold"]
    robust = {}
    for kind, level in ROBUST:
        p = predict_val(kind, level)
        rep = metrics.video_report(y_val, p, methods_val, threshold)
        robust[f"{kind}_{level}" if kind != "clean" else "clean"] = {
            k: rep[k] for k in ("roc_auc", "balanced_accuracy", "fake_recall", "specificity")}
        if kind == "clean":
            rep.update({"split": "val", "note": "threshold and model selection used this split; single seed",
                        "best_epoch": best_epoch, "epochs_run": len(history), "train_minutes": round(train_minutes, 2),
                        "trainable_params": n_train,
                        "ci95_family_bootstrap": metrics.bootstrap_ci(y_val, p, families_val, threshold)})
            (run_dir / "metrics" / "val_metrics.json").write_text(json.dumps(rep, indent=2) + "\n")
        print(f"{kind:10s} {level:>4} AUC {rep['roc_auc']:.4f} fake recall {rep['fake_recall']:.3f} "
              f"specificity {rep['specificity']:.3f}", flush=True)
    (run_dir / "metrics" / "val_robustness.json").write_text(json.dumps(robust, indent=2) + "\n")
    print("Saved ->", run_dir)


if __name__ == "__main__":
    main()
