"""Run the head-level model improvements of Chapter 16.4 on FF++ c23 (resumable).

Every row changes one thing relative to a reference row of the ablation matrix and is trained
with the same recipe, data and seeds (face crops, 8-frame windows):

    I   branch projections before fusion           reference C
    J   projections + learned softmax branch gate  reference C   (J0: fixed 1/K weights, gate ablation)
    K   one-directional (causal) LSTM              reference E
    L   EfficientNet-B0 instead of ResNet50        reference A   (needs scripts/cache_extra_backbone.py)
    M1  FFT branch: radial spectrum profile        reference F   (and E, which has no FFT branch)
    M2  FFT branch: high-pass spectrum             reference F
    M3  FFT branch: low/mid/high bands             reference F

Writes  results/ffpp_c23/<row>_faces_T8/seed<S>/...   and   results/ffpp_c23/summary/improvements_table.md

usage:  python scripts/run_improvements.py [--rows I J J0 K L M1 M2 M3] [--seeds 3]
"""
import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

ROOT = paths.RESULTS_ROOT / "ffpp_c23"
REFERENCE = {"I": "C", "J": "C", "J0": "C", "K": "E", "L": "A", "M1": "F", "M2": "F", "M3": "F"}
WHAT = {"I": "branch projections", "J": "learned branch gate", "J0": "projections, fixed 1/K weights",
        "K": "causal LSTM", "L": "EfficientNet-B0 encoder", "M1": "FFT radial profile",
        "M2": "FFT high-pass", "M3": "FFT low/mid/high bands"}
KEYS = ("roc_auc", "balanced_accuracy", "fake_recall", "specificity")


def load(row, seed):
    f = ROOT / f"{row}_faces_T8" / f"seed{seed}" / "metrics" / "val_metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", default=list(REFERENCE))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = range(args.seeds)

    for row in args.rows:
        for s in seeds:
            if load(row, s):
                print(f"skip {row} seed{s} (done)", flush=True)
                continue
            cmd = [sys.executable, str(Path(__file__).parent / "train.py"), "--exp", row, "--seed", str(s)]
            print("\n>>>", " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=True)

    def stats(row):
        runs = [m for m in (load(row, s) for s in seeds) if m]
        if not runs:
            return None
        out = {"n": len(runs), "params": runs[0]["trainable_params"],
               "minutes": st.mean(m["train_minutes"] for m in runs)}
        for k in KEYS:
            vals = [m[k] for m in runs]
            out[k] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
        return out

    lines = ["Validation split, video level, mean ± std over seeds. 'Delta' is the AUC difference to the reference row.",
             "", "| Row | Change | Reference | Seeds | Trainable params | AUC | Delta AUC | Balanced acc | Train min |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ref in dict.fromkeys(REFERENCE.values()):
        r = stats(ref)
        if r:
            lines.append(f"| {ref} | (reference) | - | {r['n']} | {r['params']:,} | {r['roc_auc'][0]:.3f} ± "
                         f"{r['roc_auc'][1]:.3f} | - | {r['balanced_accuracy'][0]:.3f} | {r['minutes']:.1f} |")
        for row in [k for k, v in REFERENCE.items() if v == ref]:
            m = stats(row)
            if m and r:
                lines.append(f"| {row} | {WHAT[row]} | {ref} | {m['n']} | {m['params']:,} | {m['roc_auc'][0]:.3f} ± "
                             f"{m['roc_auc'][1]:.3f} | {m['roc_auc'][0] - r['roc_auc'][0]:+.3f} | "
                             f"{m['balanced_accuracy'][0]:.3f} | {m['minutes']:.1f} |")
    md = "\n".join(lines) + "\n"
    out = ROOT / "summary" / "improvements_table.md"
    out.write_text(md, encoding="utf-8")
    print("Saved ->", out)
    print(md.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
