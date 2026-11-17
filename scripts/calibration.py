"""Calibration of the video-level fake probability on the validation split.

Reads the saved predictions (predictions/val_videos.csv) of every seed of the given runs and
reports, for P(fake):
    ECE (10 equal-width bins), Brier score and negative log-likelihood
      raw     as produced by the model (mean window probability)
      platt   after Platt scaling  sigmoid(a * logit(p) + b)
Platt scaling is cross-fitted: the validation families are cut into 5 folds, (a, b) is fitted
on 4 folds and applied to the 5th, so no video is scored by parameters fitted on its family.
Ranking metrics (AUC) are unchanged by this monotone map.

Writes  results/ffpp_c23/summary/calibration.csv        one row per run and seed
        results/ffpp_c23/summary/calibration_table.md   mean ± std over seeds
        results/ffpp_c23/summary/plots/reliability_<run>.png

usage:  python scripts/calibration.py [--runs C_faces_T8 E_faces_T8 F_faces_T8]
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

ROOT = paths.RESULTS_ROOT / "ffpp_c23"
EPS = 1e-6
BINS = 10


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def bin_stats(y, p):
    """Per-bin (count, mean confidence, observed fake rate) over equal-width bins of P(fake)."""
    idx = np.minimum((p * BINS).astype(int), BINS - 1)
    return [(int((idx == b).sum()), float(p[idx == b].mean()), float(y[idx == b].mean()))
            for b in range(BINS) if (idx == b).any()]


def scores(y, p):
    stats = bin_stats(y, p)
    ece = sum(n * abs(conf - rate) for n, conf, rate in stats) / len(y)
    pc = np.clip(p, EPS, 1 - EPS)
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))}


def platt_crossfit(y, p, families, folds=5, seed=0):
    uniq = np.unique(families)
    fold_of = dict(zip(np.random.default_rng(seed).permutation(uniq), np.arange(len(uniq)) % folds))
    fold = np.array([fold_of[f] for f in families])
    x, out, params = logit(p)[:, None], np.empty_like(p), []
    for k in range(folds):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(x[fold != k], y[fold != k])
        out[fold == k] = lr.predict_proba(x[fold == k])[:, 1]
        params.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
    return out, params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[f"{e}_faces_T8" for e in "ABCDEF"])
    args = ap.parse_args()
    out_dir = ROOT / "summary"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    rows = []
    for run in args.runs:
        files = sorted((ROOT / run).glob("seed*/predictions/val_videos.csv"))
        if not files:
            sys.exit(f"No predictions for run {run} under {ROOT}")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
        for f in files:
            recs = list(csv.DictReader(open(f, encoding="utf-8")))
            y = np.array([r["label"] == "fake" for r in recs], dtype=float)
            p = np.array([float(r["p_fake"]) for r in recs])
            fam = np.array([r["family_id"] for r in recs])
            p_cal, params = platt_crossfit(y, p, fam)
            seed = int(f.parent.parent.name[4:])
            row = {"run": run, "seed": seed, "n_videos": len(y), "fake_rate": float(y.mean()),
                   "mean_p_fake_raw": float(p.mean()),
                   "platt_a_mean": st.mean(a for a, _ in params), "platt_b_mean": st.mean(b for _, b in params)}
            for tag, probs in (("raw", p), ("platt", p_cal)):
                row.update({f"{k}_{tag}": v for k, v in scores(y, probs).items()})
            rows.append(row)
            for ax, probs in zip(axes, (p, p_cal)):
                stats = bin_stats(y, probs)
                ax.plot([c for _, c, _ in stats], [r for _, _, r in stats], marker="o", label=f"seed {seed}")
        for ax, title in zip(axes, ("Raw", "After Platt scaling (cross-fitted)")):
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"{run}: {title}")
            ax.set_xlabel("Predicted P(fake), bin mean")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Observed fraction of fakes")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"reliability_{run}.png", dpi=130)
        plt.close(fig)

    with open(out_dir / "calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    def cell(group, k):
        vals = [r[k] for r in group]
        return f"{st.mean(vals):.3f} ± {(st.stdev(vals) if len(vals) > 1 else 0):.3f}"

    lines = ["Validation split, video level, P(fake); mean ± std over seeds. Lower is better.",
             f"Fakes are {rows[0]['fake_rate']:.1%} of the validation videos. Platt scaling is cross-fitted over families.",
             "", "| Run | Seeds | Mean P(fake) raw | ECE raw | ECE Platt | Brier raw | Brier Platt | NLL raw | NLL Platt |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in args.runs:
        g = [r for r in rows if r["run"] == run]
        lines.append(f"| {run} | {len(g)} | {cell(g, 'mean_p_fake_raw')} | {cell(g, 'ece_raw')} | {cell(g, 'ece_platt')} | "
                     f"{cell(g, 'brier_raw')} | {cell(g, 'brier_platt')} | {cell(g, 'nll_raw')} | {cell(g, 'nll_platt')} |")
    md = "\n".join(lines) + "\n"
    (out_dir / "calibration_table.md").write_text(md, encoding="utf-8")
    print("Saved ->", out_dir / "calibration.csv")
    print("Saved ->", out_dir / "calibration_table.md")
    print("Saved ->", out_dir / "plots")
    print(md.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
