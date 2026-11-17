"""Occlusion-sensitivity explanations for a trained run (validation videos only).

The heads work on frozen backbone features, so gradients never reach the pixels of the cached
pipeline. Occlusion needs no gradients: a 32x32 patch (7x7 grid) is replaced by the frame's mean
colour in all frames of one window, the backbones and the head are re-run, and the change of
P(fake) is the importance of that patch:

    importance = P(fake | original) - P(fake | patch hidden)      (> 0: the patch supported "fake")

One window per video is used: the middle `seq_len` cached frames.

Faithfulness check (deletion test): hiding the K most important patches at once should lower
P(fake) more than hiding K random patches. Reported for fake videos.

Writes  <run_dir>/explanations/
            occlusion_maps.npz          importance [N, 7, 7], p_base, video ids, labels, methods
            occlusion_summary.json      per-method statistics and the deletion test
            occlusion_mean_maps.png     mean importance map per manipulation method and for reals
            occlusion_examples.png      one detected fake per method: frame, heat map overlay

usage:  python scripts/explain_occlusion.py --run-dir results/ffpp_c23/C_faces_T8/seed0 [--per-method 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import backbones, engine, paths  # noqa: E402
from deepshield.data import FeatureStore  # noqa: E402

GRID, PATCH = 7, 32


def occlude(rgb, cells):
    """rgb [T, 224, 224, 3]; hide the given grid cells (row, col) with each frame's mean colour."""
    out = rgb.copy()
    fill = rgb.reshape(len(rgb), -1, 3).mean(axis=1).astype(np.uint8)  # [T, 3]
    for r, c in cells:
        out[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = fill[:, None, None, :]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=paths.RESULTS_ROOT / "ffpp_c23" / "C_faces_T8" / "seed0")
    ap.add_argument("--per-method", type=int, default=20, help="validation videos per method (and reals)")
    ap.add_argument("--top-k", type=int, default=5, help="patches hidden in the deletion test")
    ap.add_argument("--random-draws", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = engine.load_checkpoint(args.run_dir / "checkpoints" / "best.pth", device)
    dc, fake_index, use_fft = ckpt["data_config"], ckpt["fake_index"], ckpt["model_config"]["use_fft"]
    if dc["variant"] != "faces":
        sys.exit("Occlusion maps are implemented for the face-crop variant.")
    nets = backbones.load(device)
    store = FeatureStore("faces", in_memory=False)
    rows = store.split_rows("val")
    picked = []
    for m in sorted({r["method"] for r in rows}):
        group = sorted((r for r in rows if r["method"] == m), key=lambda r: r["family_id"])
        picked += group[:: max(1, len(group) // args.per_method)][: args.per_method]
    T = dc["seq_len"]
    first = (store.n_frames - T) // 2
    cells = [(r, c) for r in range(GRID) for c in range(GRID)]
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def p_fake(variants):
        """variants [V, T, 224, 224, 3] uint8 -> P(fake) [V]"""
        flat = variants.reshape(-1, 224, 224, 3)
        cnn, vit = backbones.embed(nets, flat, device)
        batch = {"cnn": torch.from_numpy(cnn.astype(np.float32)).view(len(variants), T, -1).to(device),
                 "vit": torch.from_numpy(vit.astype(np.float32)).view(len(variants), T, -1).to(device)}
        out = []
        for b in range(0, len(variants), 16):  # small chunks: the FFT branch is memory hungry
            part = {k: v[b:b + 16] for k, v in batch.items()}
            if use_fft:
                part["gray"] = torch.from_numpy(backbones.to_gray(variants[b:b + 16])).to(device)
            out.append(torch.softmax(model(part).float(), dim=1)[:, fake_index].cpu().numpy())
        return np.concatenate(out)

    print(f"Run: {args.run_dir} | videos: {len(picked)} | window: frames {first}..{first + T - 1} | device: {device}",
          flush=True)
    maps, p_base, top_drop, rand_drop, frames_mid = [], [], [], [], []
    t0 = time.time()
    for n, r in enumerate(picked, 1):
        files = sorted((paths.FFPP_FACES_DIR / r["video_id"]).glob("*.jpg"))[first:first + T]
        rgb = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files])
        probs = p_fake(np.stack([rgb] + [occlude(rgb, [c]) for c in cells]))
        imp = (probs[0] - probs[1:]).reshape(GRID, GRID)
        order = np.argsort(imp.ravel())[::-1][: args.top_k]
        deletion = [occlude(rgb, [cells[i] for i in order])]
        deletion += [occlude(rgb, [cells[i] for i in rng.choice(len(cells), args.top_k, replace=False)])
                     for _ in range(args.random_draws)]
        d = p_fake(np.stack(deletion))
        maps.append(imp)
        p_base.append(probs[0])
        top_drop.append(probs[0] - d[0])
        rand_drop.append(probs[0] - d[1:].mean())
        frames_mid.append(rgb[T // 2])
        if n % 20 == 0 or n == len(picked):
            print(f"[{n}/{len(picked)}] {(time.time() - t0) / n:.1f} s/video", flush=True)

    maps, p_base = np.array(maps), np.array(p_base)
    top_drop, rand_drop = np.array(top_drop), np.array(rand_drop)
    methods = np.array([r["method"] for r in picked])
    is_fake = np.array([r["label_idx"] == fake_index for r in picked])
    out_dir = args.run_dir / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "occlusion_maps.npz", importance=maps.astype(np.float32), p_base=p_base,
                        video_id=np.array([r["video_id"] for r in picked]), method=methods, is_fake=is_fake)

    centre = np.zeros((GRID, GRID), bool)
    centre[2:5, 2:5] = True
    per_method = {}
    for m in sorted(set(methods)):
        sel = methods == m
        mm = maps[sel]
        per_method[m] = {"n_videos": int(sel.sum()), "mean_p_fake": float(p_base[sel].mean()),
                         "mean_importance_centre_3x3": float(mm[:, centre].mean()),
                         "mean_importance_border": float(mm[:, ~centre].mean()),
                         "mean_abs_importance": float(np.abs(mm).mean()),
                         "max_single_patch_drop_mean": float(mm.reshape(len(mm), -1).max(axis=1).mean())}
    summary = {"run_dir": str(args.run_dir), "split": "val", "grid": GRID, "patch_px": PATCH, "window_frames": T,
               "importance": "P(fake | original) - P(fake | patch hidden), one window per video",
               "per_method": per_method,
               "deletion_test_fake_videos": {
                   "n_videos": int(is_fake.sum()), "patches_hidden": args.top_k, "random_draws": args.random_draws,
                   "mean_p_fake_original": float(p_base[is_fake].mean()),
                   "mean_drop_top_k": float(top_drop[is_fake].mean()),
                   "mean_drop_random_k": float(rand_drop[is_fake].mean()),
                   "fraction_videos_top_k_drop_larger": float((top_drop[is_fake] > rand_drop[is_fake]).mean()),
                   "note": "the top-k patches are chosen from the same model's single-patch scores, so a larger "
                           "drop than random is expected; it shows the map is consistent, not that it is correct"}}
    (out_dir / "occlusion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    names = sorted(set(methods))
    lim = max(abs(maps[methods == m].mean(axis=0)).max() for m in names)
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 3))
    for ax, m in zip(axes, names):
        im = ax.imshow(maps[methods == m].mean(axis=0), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{m}\n(n={int((methods == m).sum())})", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean drop of P(fake) when hidden")
    fig.savefig(out_dir / "occlusion_mean_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fakes = [m for m in names if m != "original"]
    fig, axes = plt.subplots(2, len(fakes), figsize=(2.6 * len(fakes), 5.4))
    for col, m in enumerate(fakes):
        idx = np.flatnonzero(methods == m)
        i = idx[np.argmax(p_base[idx])]  # the most confidently detected fake of this method
        heat = cv2.resize(maps[i], (224, 224), interpolation=cv2.INTER_CUBIC)
        axes[0, col].imshow(frames_mid[i])
        axes[0, col].set_title(f"{m}\nP(fake)={p_base[i]:.2f}", fontsize=9)
        axes[1, col].imshow(frames_mid[i])
        v = np.abs(maps[i]).max() + 1e-9
        axes[1, col].imshow(heat, cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.6)
        for ax in axes[:, col]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "occlusion_examples.png", dpi=130)
    plt.close(fig)

    for name in ("occlusion_maps.npz", "occlusion_summary.json", "occlusion_mean_maps.png", "occlusion_examples.png"):
        print("Saved ->", out_dir / name)
    print(json.dumps(summary["deletion_test_fake_videos"], indent=2))
    for m, s in per_method.items():
        print(f"{m:15s} centre {s['mean_importance_centre_3x3']:+.4f} border {s['mean_importance_border']:+.4f} "
              f"max-patch {s['max_single_patch_drop_mean']:+.4f}")


if __name__ == "__main__":
    main()
