"""Whole-video reading: score every segment of every video, choose the aggregation rule on validation,
report the test sets once (docs/final_evaluation_protocol.md, "Whole-video reading").

Sets: validation  FF++ val, Celeb-DF v2 val (manifests/celebdf_v2_train.csv)
      test        FF++ test, Celeb-DF v2 test list, DFD
Rules: R1 middle (the evaluated segment), R2 mean over scored segments, R3 max.

CPU work (decoding, face tracks, crops) runs in worker processes (deepshield/segments.py); the served
model scores the crops on the GPU in this process. Each set is saved when done, so a rerun resumes.

Writes  <run>/segments/<set>.json      every video's segment scores
        <run>/aggregation.json          the chosen rule and threshold (read by deepshield/serving.py)
        results/final/segments_report.json and segments_report.md

usage:  python scripts/eval_segments.py [--workers 6] [--sets ...]
"""
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepshield import metrics, paths, segments as sg  # noqa: E402

RUN = paths.RESULTS_ROOT / "ffpp_c23" / "e2e_finetune_aug-frame_ffpp+celebdf" / "seed0"
VAL_SETS = {"val_ffpp": ("ffpp_c23", "val"), "val_celebdf": ("celebdf_v2_train", "val")}
TEST_SETS = {"test_ffpp": ("ffpp_c23", "test"), "test_celebdf_v2": ("celebdf_v2", "test"), "test_dfd": ("dfd_c23", "test")}
STORED = {"test_ffpp": "ffpp_test", "test_celebdf_v2": "celebdf_v2", "test_dfd": "dfd"}


def rows_of(dataset, split):
    """Rows in the same order as the stored evaluations (feature cache index for test sets)."""
    ds = paths.DATASETS[dataset]
    man = {r["video_id"]: r for r in csv.DictReader(open(ds["manifest"], encoding="utf-8"))}
    if dataset == "celebdf_v2_train":
        from finetune_e2e import extra_rows
        tr, va = extra_rows(dataset)
        rows = va if split == "val" else tr
    else:
        from deepshield.data import FeatureStore
        rows = FeatureStore("faces", dataset=dataset, in_memory=False).split_rows(split, allow_test=split == "test")
    return [{"video_id": r["video_id"], "fake": int(r["label_idx"] == paths.CLASS_TO_IDX["fake"]),
             "family": r["family_id"], "path": str(ds["root"] / man[r["video_id"]]["rel_path"]),
             "n_frames": int(man[r["video_id"]]["n_frames"])} for r in rows]


def work(job):
    path, n = job
    try:
        return sg.video_crops(path, n)
    except Exception as exc:  # recorded, never silently dropped
        return {"error": f"{type(exc).__name__}: {exc}"}


def score_set(name, dataset, split, det, workers):
    out_path = RUN / "segments" / f"{name}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    rows = rows_of(dataset, split)
    partial = out_path.with_suffix(".partial.json")
    res = json.loads(partial.read_text()) if partial.exists() else []  # resume after an interruption
    print(f"== {name}: {len(rows)} videos ({len(res)} already scored)", flush=True)
    t0, done0 = time.time(), len(res)
    todo = rows[len(res):]
    with Pool(workers) as pool:
        for k, (r, v) in enumerate(zip(todo, pool.imap(work, [(r["path"], r["n_frames"]) for r in todo], chunksize=2)), len(res) + 1):
            if "error" in v:
                res.append({**{x: r[x] for x in ("video_id", "fake", "family")}, "error": v["error"], "segments": []})
                continue
            segs = []
            for s in v["segments"]:
                score = None if s is None or s["crops"] is None else float(np.mean(det._window_probs(list(s["crops"]))))
                segs.append({"start": None if s is None else s["start"], "end": None if s is None else s["end"],
                             "score": score, "face_rate": None if s is None else s["face_rate"]})
            res.append({**{x: r[x] for x in ("video_id", "fake", "family")}, "middle": v["middle"], "fps": v["fps"],
                        "n_frames": v["n_frames"], "segments": segs})
            if k % 200 == 0 or k == len(rows):
                rate = (k - done0) / (time.time() - t0)
                print(f"   [{k}/{len(rows)}] {rate:.2f} videos/s eta {(len(rows) - k) / rate / 60:.0f} min", flush=True)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                partial.write_text(json.dumps(res))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res))
    partial.unlink(missing_ok=True)
    return res


def rule_scores(videos, rule):
    """-> scores (np.nan where no segment has a face)."""
    out = []
    for v in videos:
        scored = [(i, s["score"]) for i, s in enumerate(v["segments"]) if s["score"] is not None]
        if not scored:
            out.append(np.nan)
        elif rule == "middle":
            out.append(min(scored, key=lambda t: abs(t[0] - v["middle"]))[1])
        elif rule == "mean":
            out.append(float(np.mean([s for _, s in scored])))
        else:
            out.append(float(max(s for _, s in scored)))
    return np.array(out)


def evaluate(videos, rule, thr):
    s = rule_scores(videos, rule)
    y = np.array([v["fake"] for v in videos])
    ok = ~np.isnan(s)
    m = metrics.threshold_metrics(y[ok], s[ok], thr)
    return {"n": int(ok.sum()), "n_without_face": int((~ok).sum()), "roc_auc": metrics.safe_auc(y[ok], s[ok]),
            **{k: m[k] for k in ("balanced_accuracy", "fake_recall", "specificity")}}


def best_threshold(sets, rule):
    """Threshold maximising the mean of the sets' balanced accuracies (validation only)."""
    cands = np.unique(np.concatenate([rule_scores(v, rule) for v in sets]))
    cands = cands[~np.isnan(cands)]
    best = (-1, 0.5)
    for t in cands:
        m = np.mean([evaluate(v, rule, t)["balanced_accuracy"] for v in sets])
        if m > best[0]:
            best = (m, float(t))
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    from deepshield.serving import FineTunedDetector
    det = FineTunedDetector(RUN / "checkpoints" / "best_trainable.pth")
    thr1 = det.threshold

    data = {}
    for name, (dataset, split) in {**VAL_SETS, **TEST_SETS}.items():
        data[name] = score_set(name, dataset, split, det, args.workers)

    # ---- choice on validation ----
    val = [data[n] for n in VAL_SETS]
    choice = {}
    for rule in ("middle", "mean", "max"):
        choice[rule] = {"mean_val_auc": float(np.mean([evaluate(v, rule, thr1)["roc_auc"] for v in val]))}
    best_alt = max(("mean", "max"), key=lambda r: choice[r]["mean_val_auc"])
    adopt = choice[best_alt]["mean_val_auc"] > choice["middle"]["mean_val_auc"]
    rule = best_alt if adopt else "middle"
    thr = best_threshold(val, rule) if adopt else thr1
    agg = {"format": "deepshield-aggregation/1", "rule": rule, "threshold": thr, "middle_threshold": thr1,
           "validation_mean_auc": choice, "max_segments": sg.MAX_SEGMENTS}
    (RUN / "aggregation.json").write_text(json.dumps(agg, indent=2) + "\n")

    # ---- report ----
    report = {"aggregation": agg, "validation": {}, "test": {}, "consistency": {}}
    for n in VAL_SETS:
        report["validation"][n] = {r: evaluate(data[n], r, thr1 if r == "middle" else thr) for r in ("middle", "mean", "max")}
    for n in TEST_SETS:
        report["test"][n] = {"middle": evaluate(data[n], "middle", thr1)}
        if rule != "middle":
            report["test"][n][rule] = evaluate(data[n], rule, thr)
        seg_counts = [sum(s["score"] is not None for s in v["segments"]) for v in data[n]]
        report["test"][n]["segments_scored_median"] = float(np.median(seg_counts))
        stored = np.load(paths.RESULTS_ROOT / "final" / STORED[n] / "final_cdf.npy")
        mids = np.array([v["segments"][v["middle"]]["score"] if v.get("segments") and v["segments"][v["middle"]]["score"] is not None
                         else np.nan for v in data[n]])
        ok = ~np.isnan(mids)
        report["consistency"][n] = {"compared": int(ok.sum()), "max_abs_diff": float(np.max(np.abs(mids[ok] - stored[ok])))}
    (paths.RESULTS_ROOT / "final" / "segments_report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [f"Whole-video reading of the served model. Rule chosen on validation: **{rule}** (threshold {thr:.3f}).", "",
             "Validation mean AUC (FF++ val, Celeb-DF val): " +
             ", ".join(f"{r} {choice[r]['mean_val_auc']:.3f}" for r in ("middle", "mean", "max")), "",
             "| Set | Rule | Videos | AUC | Balanced acc | Fake recall | Specificity |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for part in ("validation", "test"):
        for n, rs in report[part].items():
            for r, m in rs.items():
                if isinstance(m, dict):
                    lines.append(f"| {n} | {r} | {m['n']} | {m['roc_auc']:.3f} | {m['balanced_accuracy']:.3f} | "
                                 f"{m['fake_recall']:.3f} | {m['specificity']:.3f} |")
    lines += ["", "Check against the stored test scores (middle segment): " +
              ", ".join(f"{n} max |diff| {c['max_abs_diff']:.1e} over {c['compared']}" for n, c in report["consistency"].items())]
    md = "\n".join(lines) + "\n"
    (paths.RESULTS_ROOT / "final" / "segments_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
