"""Cache the features of one more frozen backbone next to an existing feature cache.

Used for the lightweight-encoder experiment (row L: EfficientNet-B0 instead of ResNet50). The
rows follow the existing index.csv, so the new file lines up with resnet50.npy and gray.npy.

Reads   <DATA_ROOT>/<dataset>/<variant>/<video>/*.jpg and features/<variant>/index.csv
Writes  <DATA_ROOT>/<dataset>/features/<variant>/<backbone>.npy   float16 [N, 32, dim]
        and an "extra_backbones" entry in meta.json

usage:  python scripts/cache_extra_backbone.py --backbone efficientnet_b0 [--dataset ffpp_c23] [--variant faces]
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
import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import backbones, paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="efficientnet_b0", help="timm model name")
    ap.add_argument("--dataset", choices=sorted(paths.DATASETS), default="ffpp_c23")
    ap.add_argument("--variant", choices=["faces", "fullframe"], default="faces")
    ap.add_argument("--batch-videos", type=int, default=8)
    args = ap.parse_args()

    base = paths.DATASETS[args.dataset]["data_dir"]
    out_dir = base / "features" / args.variant
    meta_path = out_dir / "meta.json"
    if not meta_path.is_file():
        sys.exit(f"No feature cache at {out_dir}. Run scripts/cache_features.py first.")
    meta = json.loads(meta_path.read_text())
    video_ids = [r["video_id"] for r in csv.DictReader(open(out_dir / "index.csv", encoding="utf-8"))]
    n, frames = len(video_ids), meta["frames_per_video"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).eval().to(device)
    dim = model.num_features
    cfg = model.pretrained_cfg
    if tuple(cfg["mean"]) != backbones.MEAN or tuple(cfg["std"]) != backbones.STD or cfg["input_size"][-1] != 224:
        sys.exit(f"{args.backbone} expects other preprocessing than the cached pipeline: {cfg}")
    out_path = out_dir / f"{args.backbone}.npy"
    done_path = out_dir / f"{args.backbone}.done.npy"
    resume = out_path.exists() and done_path.exists()
    feats = np.lib.format.open_memmap(out_path, mode="r+" if resume else "w+", dtype=np.float16,
                                      shape=None if resume else (n, frames, dim))
    done = np.load(done_path) if resume else np.zeros(n, bool)
    mean = torch.tensor(backbones.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(backbones.STD, device=device).view(1, 3, 1, 1)
    print(f"{args.backbone} ({dim}-d, {sum(p.numel() for p in model.parameters()):,} params) -> {out_path}")
    print(f"Videos: {n} | already cached: {int(done.sum())} | device: {device}", flush=True)

    def load_video(i):
        files = sorted((base / args.variant / video_ids[i]).glob("*.jpg"))
        if len(files) != frames:
            raise RuntimeError(f"{video_ids[i]}: expected {frames} frames, found {len(files)}")
        return np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files])

    todo = [i for i in range(n) if not done[i]]
    t0, pool = time.time(), ThreadPoolExecutor(8)
    for b in range(0, len(todo), args.batch_videos):
        rows = todo[b:b + args.batch_videos]
        rgb = np.concatenate(list(pool.map(load_video, rows)))
        x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div_(255)
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model((x - mean) / std).float().cpu().numpy()
        feats[rows] = out.reshape(len(rows), frames, dim).astype(np.float16)
        done[rows] = True
        k = b + len(rows)
        if (b // args.batch_videos) % 50 == 0 or k == len(todo):
            np.save(done_path, done)
            rate = k / (time.time() - t0)
            print(f"[{k}/{len(todo)}] {rate:.1f} videos/s eta={(len(todo) - k) / rate / 60:.0f} min", flush=True)
    feats.flush()
    np.save(done_path, done)
    meta.setdefault("extra_backbones", {})[args.backbone] = {
        "dim": dim, "file": out_path.name, "params": sum(p.numel() for p in model.parameters()),
        "pretrained": cfg.get("hf_hub_id") or cfg.get("tag", "")}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print("Saved ->", out_path)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Complete: {bool(done.all())}")


if __name__ == "__main__":
    main()
