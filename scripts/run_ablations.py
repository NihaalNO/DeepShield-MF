"""Run the ablation matrix (Chapter 16.2) on FF++ c23. Finished runs are skipped, so it is resumable.

Stage 1  rows A-F on face crops, seq_len 8, seeds 0..N-1
Stage 2  best BiLSTM row among D/E/F (mean validation AUC over seeds), chosen automatically:
           G  frame-order shuffle at evaluation (also run for F if F is not the best)
           H  sequence lengths 4 and 16 (8 comes from stage 1)
           full-frame input instead of face crops
Then scripts/summarize.py builds the tables.

usage:  python scripts/run_ablations.py [--seeds 3] [--stage 1|2|all]
"""
import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

ROOT = paths.RESULTS_ROOT / "ffpp_c23"
SCRIPTS = Path(__file__).resolve().parent


def run(script, *cli):
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, cli)]
    print("\n>>>", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"<<< {(time.time() - t0) / 60:.1f} min", flush=True)


def train(exp, seed, variant="faces", seq_len=8):
    name = f"{exp}_{variant}_T{seq_len}"
    if (ROOT / name / f"seed{seed}" / "metrics" / "val_metrics.json").exists():
        print(f"skip {name} seed{seed} (done)", flush=True)
    else:
        run("train.py", "--exp", exp, "--seed", seed, "--variant", variant, "--seq-len", seq_len)
    return name


def mean_auc(name, seeds):
    return st.mean(json.loads((ROOT / name / f"seed{s}" / "metrics" / "val_metrics.json").read_text())["roc_auc"]
                   for s in seeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    args = ap.parse_args()
    seeds = range(args.seeds)

    if args.stage in ("1", "all"):
        for exp in "ABCDEF":  # cheap rows first
            for s in seeds:
                train(exp, s)

    if args.stage in ("2", "all"):
        aucs = {exp: mean_auc(f"{exp}_faces_T8", seeds) for exp in "DEF"}
        best = max(aucs, key=aucs.get)
        print(f"\nMean validation AUC of BiLSTM rows: {aucs} -> best candidate: {best}", flush=True)
        (ROOT / "summary").mkdir(parents=True, exist_ok=True)
        (ROOT / "summary" / "best_candidate.json").write_text(json.dumps(
            {"selected_on": "mean validation video-level ROC-AUC over seeds", "candidates": aucs, "best": best}, indent=2) + "\n")
        for exp in dict.fromkeys([best, "F"]):  # G
            for s in seeds:
                run_dir = ROOT / f"{exp}_faces_T8" / f"seed{s}"
                if not (run_dir / "metrics" / "val_eval_shuffled.json").exists():
                    run("evaluate.py", "--run-dir", run_dir, "--shuffle-frames")
        for seq_len in (4, 16):  # H
            for s in seeds:
                train(best, s, seq_len=seq_len)
        for s in seeds:  # full frame vs face crop
            train(best, s, variant="fullframe")

    for exp in "ABCDEF":  # head inference time for every stage-1 run
        for s in seeds:
            run_dir = ROOT / f"{exp}_faces_T8" / f"seed{s}"
            if (run_dir / "checkpoints" / "best.pth").exists() and not (run_dir / "metrics" / "val_eval.json").exists():
                run("evaluate.py", "--run-dir", run_dir)
    run("summarize.py")


if __name__ == "__main__":
    main()
