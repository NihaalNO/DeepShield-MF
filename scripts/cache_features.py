"""Cache frozen-backbone features for the FF++ frames (run once per input variant).

The hybrid model keeps ResNet50 and ViT-B/16 frozen, so their per-frame outputs
never change. Computing them once makes every later experiment train only its
small head. The FFT branch is trainable and works on pixels, so its input (the
grayscale frame) is cached as well.

Reads   <DATA_ROOT>/ffpp_c23/<variant>/<method>/<video>/*.jpg   (variant: faces | fullframe)
        <DATA_ROOT>/ffpp_c23/frame_index.csv
Writes  <DATA_ROOT>/ffpp_c23/features/<variant>/
            resnet50.npy               float16 [N, 32, 2048]
            vit_base_patch16_224.npy   float16 [N, 32, 768]
            gray.npy                   uint8   [N, 32, 224, 224]  (mean of RGB; skipped with --no-gray)
            index.csv                  row -> video_id
            meta.json                  backbones, normalisation, shapes

Backbones and preprocessing are the ones in notebooks/frac_df_cnnvit_fft_temporal.ipynb:
timm resnet50 / vit_base_patch16_224 (pretrained, num_classes=0), ImageNet mean/std.

usage:  python scripts/cache_features.py [--dataset ffpp_c23|celebdf_v1|celebdf_v2] --variant faces [--limit 20] [--batch-videos 4]
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
from deepshield import paths  # noqa: E402

BACKBONES = {"resnet50": 2048, "vit_base_patch16_224": 768}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FRAMES = 32


def load_video(frame_dir):
    files = sorted(frame_dir.glob("*.jpg"))
    if len(files) != FRAMES:
        raise RuntimeError(f"{frame_dir}: expected {FRAMES} frames, found {len(files)}")
    return np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files])  # [T, 224, 224, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(paths.DATASETS), default="ffpp_c23")
    ap.add_argument("--variant", choices=["faces", "fullframe"], default="faces")
    ap.add_argument("--batch-videos", type=int, default=4, help="videos per GPU batch (x32 frames)")
    ap.add_argument("--no-gray", action="store_true", help="do not cache grayscale frames (FFT branch input)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (testing)")
    ap.add_argument("--out-root", type=Path, default=None, help="default <DATA_ROOT>/<dataset>/features")
    ap.add_argument("--io-threads", type=int, default=8)
    args = ap.parse_args()

    base = paths.DATASETS[args.dataset]["data_dir"]
    frames_root = base / args.variant
    index_rows = [r for r in csv.DictReader(open(base / "frame_index.csv", encoding="utf-8")) if r["status"] == "ok"]
    # every model reads exactly FRAMES crops per video; clips too short to give them are left out
    short = [r["video_id"] for r in index_rows if int(r["n_saved"]) != FRAMES]
    if short:
        print(f"Skipping {len(short)} videos with fewer than {FRAMES} face crops (too short), e.g. {short[:3]}")
    video_ids = sorted(r["video_id"] for r in index_rows if int(r["n_saved"]) == FRAMES)
    if args.limit:
        video_ids = video_ids[:: max(1, len(video_ids) // args.limit)][: args.limit]
    n = len(video_ids)
    out_dir = (args.out_root or base / "features") / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Frames from :", frames_root)
    print("Features -> :", out_dir)
    print(f"Videos: {n} | device: {device} | gray: {not args.no_gray}", flush=True)

    # Resume only if the existing cache was made for exactly this list of videos.
    index_path, done_path = out_dir / "index.csv", out_dir / "done.npy"
    resume = False
    if index_path.exists() and done_path.exists():
        old = [r["video_id"] for r in csv.DictReader(open(index_path, encoding="utf-8"))]
        resume = old == video_ids
    mode = "r+" if resume else "w+"
    feats = {name: np.lib.format.open_memmap(out_dir / f"{name}.npy", mode=mode, dtype=np.float16,
                                             shape=None if resume else (n, FRAMES, dim))
             for name, dim in BACKBONES.items()}
    gray = None
    if not args.no_gray:
        gray = np.lib.format.open_memmap(out_dir / "gray.npy", mode=mode, dtype=np.uint8,
                                         shape=None if resume else (n, FRAMES, 224, 224))
    done = np.load(done_path) if resume else np.zeros(n, bool)
    if not resume:
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["row", "video_id"])
            w.writerows(enumerate(video_ids))
    print(f"Resuming: {int(done.sum())} videos already cached" if resume else "Starting a new cache", flush=True)

    models = {name: timm.create_model(name, pretrained=True, num_classes=0).eval().to(device) for name in BACKBONES}
    pretrained_tags = {name: m.pretrained_cfg.get("hf_hub_id") or m.pretrained_cfg.get("tag", "")
                       for name, m in models.items()}
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    todo = [i for i in range(n) if not done[i]]
    t0, pool = time.time(), ThreadPoolExecutor(args.io_threads)
    for b in range(0, len(todo), args.batch_videos):
        rows = todo[b:b + args.batch_videos]
        vids = list(pool.map(lambda i: load_video(frames_root / video_ids[i]), rows))
        rgb = np.concatenate(vids)  # [B*T, 224, 224, 3] uint8
        x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            out = {name: m(x).float().cpu().numpy() for name, m in models.items()}
        for name, arr in out.items():
            feats[name][rows] = arr.reshape(len(rows), FRAMES, -1).astype(np.float16)
        if gray is not None:
            g = np.rint(rgb.astype(np.float32).mean(axis=3)).astype(np.uint8)
            gray[rows] = g.reshape(len(rows), FRAMES, 224, 224)
        done[rows] = True
        k = b + len(rows)
        if (b // args.batch_videos) % 50 == 0 or k == len(todo):
            np.save(done_path, done)
            rate = k / (time.time() - t0)
            print(f"[{k}/{len(todo)}] {rate:.1f} videos/s eta={(len(todo) - k) / rate / 60:.0f} min", flush=True)

    for arr in list(feats.values()) + ([gray] if gray is not None else []):
        arr.flush()
    np.save(done_path, done)
    meta = {"variant": args.variant, "n_videos": n, "frames_per_video": FRAMES,
            "backbones": {name: {"dim": dim, "pretrained": pretrained_tags[name], "file": f"{name}.npy"}
                          for name, dim in BACKBONES.items()},
            "normalization": {"mean": MEAN, "std": STD}, "feature_dtype": "float16",
            "gray": None if gray is None else {"file": "gray.npy", "dtype": "uint8", "definition": "round(mean(R,G,B))"},
            "torch": torch.__version__, "timm": timm.__version__, "autocast": device.type == "cuda"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Saved ->", out_dir / "meta.json")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Complete: {bool(done.all())}")


if __name__ == "__main__":
    main()
