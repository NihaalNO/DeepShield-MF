"""Robustness sweep: degrade the validation face crops and re-score trained runs.

Each condition is applied to the saved 224x224 face crops of every validation video, the frozen
backbones are re-run on the degraded pixels, and every seed of the given runs is scored with its
own frozen checkpoint and validation threshold. Nothing is retrained or retuned.

Conditions (level):  jpeg (quality)  blur (Gaussian sigma, px)  downscale (factor, then back up)
                     noise (Gaussian sigma on the 0-255 scale)  brightness (multiplier)
"clean" recomputes the features from the unmodified crops and must reproduce the cached result.

The degradation is applied to the crop, not to the source video before face detection, so it
measures the classifier's sensitivity and leaves the face detector out.

Finished conditions are skipped (delete robustness.csv to start again).

Writes  results/ffpp_c23/summary/robustness.csv        one row per run, seed and condition
        results/ffpp_c23/summary/robustness_table.md   AUC, mean ± std over seeds
        results/ffpp_c23/summary/plots/robustness_auc.png

usage:  python scripts/robustness.py [--runs C_faces_T8 E_faces_T8 F_faces_T8] [--limit-videos 40]
"""
import argparse
import csv
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import backbones, engine, metrics, paths  # noqa: E402
from deepshield.data import FeatureStore, WindowBatches  # noqa: E402

ROOT = paths.RESULTS_ROOT / "ffpp_c23"
CONDITIONS = [("clean", 0)] + [("jpeg", q) for q in (70, 40, 20)] + [("blur", s) for s in (1, 2, 3)] + \
    [("downscale", f) for f in (2, 4)] + [("noise", s) for s in (5, 10, 20)] + [("brightness", m) for m in (0.6, 1.4)]
KEYS = ("roc_auc", "balanced_accuracy", "macro_f1", "fake_recall", "specificity")


def degrade(bgr, kind, level, rng):
    if kind == "clean":
        return bgr
    if kind == "jpeg":
        return cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(level)])[1], cv2.IMREAD_COLOR)
    if kind == "blur":
        return cv2.GaussianBlur(bgr, (0, 0), float(level))
    if kind == "downscale":
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (w // level, h // level), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "noise":
        return np.clip(bgr.astype(np.float32) + rng.normal(0, level, bgr.shape), 0, 255).astype(np.uint8)
    if kind == "brightness":
        return np.clip(bgr.astype(np.float32) * level, 0, 255).astype(np.uint8)
    raise ValueError(kind)


def load_video(frame_dir, kind, level, seed):
    rng = np.random.default_rng(seed)
    files = sorted(frame_dir.glob("*.jpg"))
    return np.stack([cv2.cvtColor(degrade(cv2.imread(str(f)), kind, level, rng), cv2.COLOR_BGR2RGB) for f in files])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["C_faces_T8", "E_faces_T8", "F_faces_T8"])
    ap.add_argument("--limit-videos", type=int, default=0, help="only every k-th validation video up to N (testing)")
    ap.add_argument("--out-name", default="robustness", help="basename of the outputs")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "summary"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_name}.csv"

    store = FeatureStore("faces", in_memory=False)
    rows = store.split_rows("val")
    if args.limit_videos:
        rows = rows[:: max(1, len(rows) // args.limit_videos)][: args.limit_videos]
    rows = [{**r, "row": i} for i, r in enumerate(rows)]  # "row" now indexes the in-memory arrays
    y, methods, _ = engine.rows_to_arrays(rows)
    faces_root = paths.FFPP_FACES_DIR

    ckpts = []
    for run in args.runs:
        for path in sorted((ROOT / run).glob("seed*/checkpoints/best.pth")):
            model, ckpt = engine.load_checkpoint(path, device)
            ckpts.append((run, int(path.parent.parent.name[4:]), model, ckpt))
    if not ckpts:
        sys.exit("No checkpoints found for " + ", ".join(args.runs))
    need_gray = any(c["model_config"]["use_fft"] for *_, c in ckpts)
    nets = backbones.load(device)

    results = list(csv.DictReader(open(csv_path, encoding="utf-8"))) if csv_path.exists() else []
    done = {(r["condition"], float(r["level"])) for r in results}
    print(f"Validation videos: {len(rows)} | checkpoints: {len(ckpts)} | device: {device}", flush=True)
    pool = ThreadPoolExecutor(8)
    for kind, level in CONDITIONS:
        if (kind, float(level)) in done:
            print(f"skip {kind} {level} (done)", flush=True)
            continue
        t0 = time.time()
        cnn = np.empty((len(rows), store.n_frames, 2048), np.float16)
        vit = np.empty((len(rows), store.n_frames, 768), np.float16)
        gray = np.empty((len(rows), store.n_frames, 224, 224), np.uint8) if need_gray else None
        for b in range(0, len(rows), 4):
            part = rows[b:b + 4]
            vids = list(pool.map(lambda r: load_video(faces_root / r["video_id"], kind, level, r["row"]), part))
            rgb = np.concatenate(vids)
            c, v = backbones.embed(nets, rgb, device)
            cnn[b:b + len(part)] = c.reshape(len(part), store.n_frames, -1)
            vit[b:b + len(part)] = v.reshape(len(part), store.n_frames, -1)
            if need_gray:
                gray[b:b + len(part)] = backbones.to_gray(rgb).reshape(len(part), store.n_frames, 224, 224)
        mem = SimpleNamespace(cnn=cnn, vit=vit, gray=gray, gray_in_memory=True, n_frames=store.n_frames, variant="faces")
        for run, seed, model, ckpt in ckpts:
            dc = ckpt["data_config"]
            batches = WindowBatches(mem, rows, seq_len=dc["seq_len"], stride=dc["stride"], batch_size=32,
                                    use_gray=ckpt["model_config"]["use_fft"], device=device)
            p, _ = engine.predict(model, batches, None, ckpt["fake_index"])
            rep = metrics.video_report(y, p, methods, ckpt["threshold"])
            results.append({"run": run, "seed": seed, "condition": kind, "level": level, "n_videos": len(rows),
                            "mean_p_fake": float(p.mean()), **{k: rep[k] for k in KEYS}})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(results)
        aucs = {run: st.mean(float(r["roc_auc"]) for r in results
                             if r["run"] == run and (r["condition"], float(r["level"])) == (kind, float(level)))
                for run in args.runs}
        print(f"{kind:10s} {level:>4} | " + " | ".join(f"{k} AUC {v:.4f}" for k, v in aucs.items()) +
              f" | {(time.time() - t0) / 60:.1f} min", flush=True)

    def agg(run, kind, level, key):
        vals = [float(r[key]) for r in results
                if r["run"] == run and (r["condition"], float(r["level"])) == (kind, float(level))]
        return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

    lines = ["Validation split, video level, mean ± std over seeds; thresholds frozen from the clean validation data.",
             "Degradations are applied to the saved face crops (the face detector is not part of this test).", "",
             "| Condition | Level | " + " | ".join(f"{r} AUC | {r} fake recall | {r} specificity" for r in args.runs) + " |",
             "| --- | ---: | " + " | ".join(["---: | ---: | ---:"] * len(args.runs)) + " |"]
    for kind, level in CONDITIONS:
        cells = []
        for run in args.runs:
            m, s = agg(run, kind, level, "roc_auc")
            cells.append(f"{m:.3f} ± {s:.3f} | {agg(run, kind, level, 'fake_recall')[0]:.3f} | "
                         f"{agg(run, kind, level, 'specificity')[0]:.3f}")
        lines.append(f"| {kind} | {level if kind != 'clean' else '-'} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (out_dir / f"{args.out_name}_table.md").write_text(md, encoding="utf-8")

    kinds = [k for k in dict.fromkeys(k for k, _ in CONDITIONS) if k != "clean"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.4 * len(kinds), 3.6), sharey=True)
    for ax, kind in zip(axes, kinds):
        levels = [lv for k, lv in CONDITIONS if k == kind]
        for run in args.runs:
            pts = [agg(run, kind, lv, "roc_auc") for lv in levels]
            ax.errorbar(range(len(levels)), [m for m, _ in pts], yerr=[s for _, s in pts], marker="o", capsize=3, label=run)
            ax.axhline(agg(run, "clean", 0, "roc_auc")[0], color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(range(len(levels)), [str(lv) for lv in levels])
        ax.set_title(kind)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Validation video AUC")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"{args.out_name}_auc.png", dpi=130)
    print("Saved ->", csv_path)
    print("Saved ->", out_dir / f"{args.out_name}_table.md")
    print("Saved ->", out_dir / "plots" / f"{args.out_name}_auc.png")
    print(md.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
