"""Leave-one-method-out generalisation on FF++ c23 (substitute for a cross-dataset test).

For every manipulation method M the head is trained on the other four methods (plus reals)
and scored on validation videos of M against the validation reals. Model selection and the
threshold use only validation videos of the four training methods, so M is unseen in every
sense. The same row trained on all five methods is the "seen" reference.

Finished runs are skipped. Requires the reference runs <exp>_faces_T8 (scripts/run_ablations.py).

Writes  results/ffpp_c23/<exp>_faces_T8_lomo-<Method>/seed<S>/...   (same layout as train.py)
        results/ffpp_c23/summary/lomo.csv, lomo_table.md

usage:  python scripts/run_lomo.py [--exps C E] [--seeds 3]
"""
import argparse
import csv
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

ROOT = paths.RESULTS_ROOT / "ffpp_c23"
METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]


def metrics_of(name, seed):
    return json.loads((ROOT / name / f"seed{seed}" / "metrics" / "val_metrics.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="+", default=["C", "E"])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = range(args.seeds)

    for exp in args.exps:
        for held in METHODS:
            name = f"{exp}_faces_T8_lomo-{held}"
            for s in seeds:
                if (ROOT / name / f"seed{s}" / "metrics" / "val_metrics.json").exists():
                    print(f"skip {name} seed{s} (done)", flush=True)
                    continue
                cmd = [sys.executable, str(Path(__file__).parent / "train.py"), "--exp", exp, "--seed", str(s),
                       "--name", name, "--train-methods", *[m for m in METHODS if m != held]]
                print("\n>>>", " ".join(cmd[1:]), flush=True)
                subprocess.run(cmd, check=True)

    rows = []
    for exp in args.exps:
        for held in METHODS:
            unseen = [metrics_of(f"{exp}_faces_T8_lomo-{held}", s)["per_method"][held] for s in seeds]
            seen = [metrics_of(f"{exp}_faces_T8", s)["per_method"][held] for s in seeds]
            row = {"experiment": exp, "held_out_method": held, "seeds": len(unseen)}
            for tag, group in (("seen", seen), ("unseen", unseen)):
                for k in ("roc_auc", "fake_recall"):
                    vals = [g[k] for g in group]
                    row[f"{k}_{tag}_mean"] = round(st.mean(vals), 4)
                    row[f"{k}_{tag}_std"] = round(st.stdev(vals), 4) if len(vals) > 1 else 0.0
            row["roc_auc_drop"] = round(row["roc_auc_seen_mean"] - row["roc_auc_unseen_mean"], 4)
            rows.append(row)

    out = ROOT / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "lomo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    lines = ["Validation split, video level, each method vs. all validation reals; mean ± std over seeds.",
             "'Seen': trained on all five methods. 'Unseen': the method was left out of training, model",
             "selection and threshold choice. Fake recall uses each run's own validation threshold.", "",
             "| Row | Held-out method | AUC seen | AUC unseen | AUC drop | Fake recall seen | Fake recall unseen |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['held_out_method']} | "
                     f"{r['roc_auc_seen_mean']:.3f} ± {r['roc_auc_seen_std']:.3f} | "
                     f"{r['roc_auc_unseen_mean']:.3f} ± {r['roc_auc_unseen_std']:.3f} | {r['roc_auc_drop']:+.3f} | "
                     f"{r['fake_recall_seen_mean']:.3f} ± {r['fake_recall_seen_std']:.3f} | "
                     f"{r['fake_recall_unseen_mean']:.3f} ± {r['fake_recall_unseen_std']:.3f} |")
    md = "\n".join(lines) + "\n"
    (out / "lomo_table.md").write_text(md, encoding="utf-8")
    print("Saved ->", out / "lomo.csv")
    print("Saved ->", out / "lomo_table.md")
    print(md.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
