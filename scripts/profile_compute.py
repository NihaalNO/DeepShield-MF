"""Compute profile of the full video pipeline: decode -> face crops -> backbones -> head.

Times every stage per video on a sample of validation videos, on the GPU and (fewer videos) on
the CPU, and records parameter counts and peak GPU memory. The stages are the ones used to build
the dataset (scripts/extract_faces.py, scripts/cache_features.py) run on one video at a time:
32 frames per video, 224x224 face crops, 7 windows of 8 frames.

Run it while nothing else uses the GPU, otherwise the timings are inflated.

Writes  results/ffpp_c23/summary/compute_profile.json, compute_profile.md

usage:  python scripts/profile_compute.py [--videos 20] [--cpu-videos 4] [--runs C_faces_T8 E_faces_T8 F_faces_T8]
"""
import argparse
import csv
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_faces as ef  # noqa: E402
from deepshield import backbones, engine, paths  # noqa: E402
from deepshield.data import window_starts  # noqa: E402

ROOT = paths.RESULTS_ROOT / "ffpp_c23"
FRAMES, MAX_STRIDE, MARGIN = 32, 8, 1.3


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def crops_of(row):
    """Decode and crop one video exactly like scripts/extract_faces.py -> (rgb [T,224,224,3], seconds per stage)."""
    t0 = time.perf_counter()
    indices, _ = ef.choose_frame_indices(int(row["n_frames"]), FRAMES, MAX_STRIDE)
    frames = ef.read_frames(paths.FFPP_ROOT / row["rel_path"], indices)
    indices = [i for i in indices if i in frames]
    t1 = time.perf_counter()
    smooth = ef.fill_and_smooth(ef.track([ef.detect(frames[i]) for i in indices]))
    rgb = np.stack([ef.crop_square(frames[i], cx, cy, float(size) * MARGIN)[:, :, ::-1]
                    for i, (cx, cy, size) in zip(indices, smooth)])
    t2 = time.perf_counter()
    return np.ascontiguousarray(rgb), {"decode": t1 - t0, "face_detect_crop": t2 - t1}


def profile(device, rows, heads):
    nets = backbones.load(device)
    models = {name: engine.load_checkpoint(path, device) for name, path in heads.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for n, row in enumerate(rows):
        rgb, t = crops_of(row)
        if len(rgb) != FRAMES:
            continue
        feats = {}
        for name in nets:
            sync(device)
            t0 = time.perf_counter()
            feats[name] = _embed_one(nets[name], rgb, device)
            sync(device)
            t[f"backbone_{name}"] = time.perf_counter() - t0
        starts = window_starts(FRAMES, 8)
        idx = np.array(starts)[:, None] + np.arange(8)[None, :]
        batch = {"cnn": torch.from_numpy(feats["resnet50"][idx]).to(device),
                 "vit": torch.from_numpy(feats["vit_base_patch16_224"][idx]).to(device),
                 "gray": torch.from_numpy(backbones.to_gray(rgb)[idx]).to(device)}
        for name, (model, ckpt) in models.items():
            sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                p = torch.softmax(model(batch).float(), dim=1)[:, ckpt["fake_index"]].mean().item()
            sync(device)
            t[f"head_{name}"] = time.perf_counter() - t0
        if n > 0:  # the first video warms up cuDNN / the allocator and is not counted
            times.append(t)
    stages = {k: {"mean_ms": round(1000 * st.mean(x[k] for x in times), 2),
                  "median_ms": round(1000 * st.median(x[k] for x in times), 2)} for k in times[0]}
    out = {"device": str(device), "videos_timed": len(times), "stages": stages}
    if device.type == "cuda":
        out["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        out["gpu"] = torch.cuda.get_device_name(0)
    shared = sum(v["mean_ms"] for k, v in stages.items() if not k.startswith("head_"))
    out["end_to_end_ms_per_video"] = {name: round(shared + stages[f"head_{name}"]["mean_ms"], 1) for name in models}
    return out


@torch.no_grad()
def _embed_one(net, rgb, device):
    """Features of one backbone as float32 (via float16, like the cache)."""
    mean = torch.tensor(backbones.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(backbones.STD, device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div_(255)
    x = (x - mean) / std
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        return net(x).float().cpu().numpy().astype(np.float16).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=int, default=20, help="validation videos timed on the GPU (+1 warm-up)")
    ap.add_argument("--cpu-videos", type=int, default=4, help="videos timed on the CPU (+1 warm-up); 0 to skip")
    ap.add_argument("--runs", nargs="+", default=["C_faces_T8", "E_faces_T8", "F_faces_T8"])
    args = ap.parse_args()

    ok = {r["video_id"] for r in csv.DictReader(open(paths.FFPP_FRAME_INDEX, encoding="utf-8")) if r["status"] == "ok"}
    rows = [r for r in csv.DictReader(open(paths.FFPP_MANIFEST, encoding="utf-8"))
            if r["split"] == "val" and r["video_id"] in ok]
    rows = rows[:: max(1, len(rows) // (args.videos + 1))][: args.videos + 1]
    heads = {run: ROOT / run / "seed0" / "checkpoints" / "best.pth" for run in args.runs}

    nets = backbones.load(torch.device("cpu"))
    params = {"backbones_frozen": {k: sum(p.numel() for p in m.parameters()) for k, m in nets.items()},
              "heads_trainable": {run: sum(p.numel() for p in engine.load_checkpoint(path)[0].parameters())
                                  for run, path in heads.items()}}
    del nets
    result = {"split": "val", "frames_per_video": FRAMES, "windows_per_video": len(window_starts(FRAMES, 8)),
              "video_resolutions": sorted({f"{r['width']}x{r['height']}" for r in rows}),
              "parameters": params, "torch": torch.__version__, "cpu": platform.processor(),
              "cpu_threads": torch.get_num_threads(), "profiles": []}
    devices = ([torch.device("cuda")] if torch.cuda.is_available() else []) + \
        ([torch.device("cpu")] if args.cpu_videos else [])
    for device in devices:
        sample = rows if device.type == "cuda" else rows[: args.cpu_videos + 1]
        print(f"Profiling on {device}: {len(sample) - 1} videos (+1 warm-up)", flush=True)
        result["profiles"].append(profile(device, sample, heads))

    out_dir = ROOT / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compute_profile.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"Mean milliseconds per video ({FRAMES} frames, {result['windows_per_video']} windows), validation videos, "
             "first video excluded as warm-up.", "",
             "| Stage | " + " | ".join(f"{p['device']} ({p['videos_timed']} videos)" for p in result["profiles"]) + " |",
             "| --- | " + " | ".join("---:" for _ in result["profiles"]) + " |"]
    for stage in result["profiles"][0]["stages"]:
        lines.append(f"| {stage} | " + " | ".join(f"{p['stages'][stage]['mean_ms']:.1f}" for p in result["profiles"]) + " |")
    for run in args.runs:
        lines.append(f"| **end to end, {run}** | " +
                     " | ".join(f"**{p['end_to_end_ms_per_video'][run]:.0f}**" for p in result["profiles"]) + " |")
    lines += ["", "| Component | Parameters |", "| --- | ---: |"]
    lines += [f"| {k} (frozen) | {v:,} |" for k, v in params["backbones_frozen"].items()]
    lines += [f"| head {k} (trainable) | {v:,} |" for k, v in params["heads_trainable"].items()]
    for p in result["profiles"]:
        if "peak_gpu_memory_mb" in p:
            lines += ["", f"Peak GPU memory allocated by PyTorch: {p['peak_gpu_memory_mb']:.0f} MB ({p['gpu']})."]
    md = "\n".join(lines) + "\n"
    (out_dir / "compute_profile.md").write_text(md, encoding="utf-8")
    print("Saved ->", out_dir / "compute_profile.json")
    print("Saved ->", out_dir / "compute_profile.md")
    print(md)


if __name__ == "__main__":
    main()
