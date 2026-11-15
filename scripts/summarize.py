"""Collect all runs under results/ffpp_c23 into one table (mean and std over seeds).

Writes  results/ffpp_c23/summary/runs.csv            one row per run and seed
        results/ffpp_c23/summary/ablation_table.csv  one row per run name (mean, std, n seeds)
        results/ffpp_c23/summary/ablation_table.md   the same as a Markdown table

usage:  python scripts/summarize.py
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

METRICS = ["roc_auc", "balanced_accuracy", "macro_f1", "fake_recall", "specificity", "accuracy"]


def main():
    root = paths.RESULTS_ROOT / "ffpp_c23"
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for mfile in sorted(root.glob("*/seed*/metrics/val_metrics.json")):
        run_dir = mfile.parent.parent
        m = json.loads(mfile.read_text())
        cfg = json.loads((run_dir / "config.json").read_text())
        row = {"run": run_dir.parent.name, "seed": int(run_dir.name[4:]), "experiment": cfg["experiment"],
               "variant": cfg["data_config"]["variant"], "seq_len": cfg["data_config"]["seq_len"],
               "train_videos": cfg["train"]["videos"], "trainable_params": m["trainable_params"],
               "best_epoch": m["best_epoch"], "train_minutes": m["train_minutes"],
               **{k: m[k] for k in METRICS},
               **{f"auc_{name}": v["roc_auc"] for name, v in m["per_method"].items()}}
        for extra, key in (("val_eval.json", "head_ms_per_video"), ("val_eval_shuffled.json", "shuffled_auc_delta")):
            f = run_dir / "metrics" / extra
            if f.exists():
                e = json.loads(f.read_text())
                row[key] = (e["head_inference_ms_per_video"] if key == "head_ms_per_video"
                            else e["delta_shuffled_minus_ordered"]["roc_auc"])
        runs.append(row)
    if not runs:
        sys.exit(f"No finished runs under {root}")

    fields = sorted({k for r in runs for k in r}, key=lambda k: (k not in ("run", "seed"), k))
    with open(out_dir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(runs)

    table = []
    for name in sorted({r["run"] for r in runs}):
        group = [r for r in runs if r["run"] == name]
        agg = {"run": name, "seeds": len(group), "trainable_params": group[0]["trainable_params"],
               "train_videos": group[0]["train_videos"]}
        numeric = METRICS + ["train_minutes", "head_ms_per_video", "shuffled_auc_delta"] + \
            sorted(k for k in group[0] if k.startswith("auc_"))
        for k in numeric:
            vals = [r[k] for r in group if k in r]
            if vals:
                agg[f"{k}_mean"] = round(st.mean(vals), 4)
                agg[f"{k}_std"] = round(st.stdev(vals), 4) if len(vals) > 1 else 0.0
        table.append(agg)
    tfields = list(dict.fromkeys(k for a in table for k in a))
    with open(out_dir / "ablation_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tfields, lineterminator="\n")
        w.writeheader()
        w.writerows(table)

    def cell(a, k):
        return f"{a[k + '_mean']:.3f} ± {a[k + '_std']:.3f}" if k + "_mean" in a else "-"

    lines = ["| Run | Seeds | Trainable params | AUC | Balanced acc | Macro F1 | Fake recall | Specificity | Train min |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for a in table:
        lines.append(f"| {a['run']} | {a['seeds']} | {a['trainable_params']:,} | {cell(a, 'roc_auc')} | "
                     f"{cell(a, 'balanced_accuracy')} | {cell(a, 'macro_f1')} | {cell(a, 'fake_recall')} | "
                     f"{cell(a, 'specificity')} | {a.get('train_minutes_mean', 0):.1f} |")
    md = "Validation split, video level, mean ± std over seeds. Thresholds were chosen on this split.\n\n" + "\n".join(lines) + "\n"
    (out_dir / "ablation_table.md").write_text(md, encoding="utf-8")
    for name in ("runs.csv", "ablation_table.csv", "ablation_table.md"):
        print("Saved ->", out_dir / name)
    print(md.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
