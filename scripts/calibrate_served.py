"""Calibrate the served model's video score (Platt scaling) and choose its decision rule on validation.

Follows the amendment "calibration of the served model" in docs/final_evaluation_protocol.md:

  1. Score final_cdf's validation videos (FF++ val + Celeb-DF v2 val), cached in its run directory.
  2. Fit p = sigmoid(a * logit(score) + b) with weights that give each (dataset x real/fake) cell the
     same total weight, so p reads as P(fake) for equally likely real and fake videos.
  3. Estimate calibration on validation by 5-fold cross-fitting over families.
  4. Decision rule "fake if p >= 0.5" replaces the current threshold only if it raises the mean of the
     two datasets' validation balanced accuracies.
  5. Apply the fitted map once to the stored test scores (results/final/<dataset>/final_cdf.npy).

Writes  <run>/calibration.json           read by deepshield/serving.py
        results/final/calibration_report.json and calibration_report.md

usage:  python scripts/calibrate_served.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_eval as fe  # noqa: E402
from finetune_e2e import extra_rows  # noqa: E402
from deepshield import engine, metrics, paths  # noqa: E402
from deepshield.data import FeatureStore  # noqa: E402

RUN = fe.RESULTS / "e2e_finetune_aug-frame_ffpp+celebdf" / "seed0"
EPS, BINS = 1e-6, 10
TEST_SETS = ["ffpp_test", "celebdf_v1", "celebdf_v2", "dfd"]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cell_weights(y, dom):
    """Each (domain, class) cell gets the same total weight; weights sum to len(y)."""
    w = np.zeros(len(y))
    cells = {(d, c) for d, c in zip(dom, y)}
    for d, c in cells:
        m = (dom == d) & (y == c)
        w[m] = 1.0 / m.sum()
    return w * len(y) / w.sum()


def fit_platt(s, y, w):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s)[:, None], y, sample_weight=w)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calib_scores(y, p, w):
    """Weighted ECE (10 equal-width bins), Brier and NLL."""
    p = np.clip(p, EPS, 1 - EPS)
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, BINS - 1)
    ece = 0.0
    for b in range(BINS):
        m = idx == b
        if m.any():
            ece += w[m].sum() / w.sum() * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    brier = float(np.average((p - y) ** 2, weights=w))
    nll = float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w))
    return {"ece": float(ece), "brier": brier, "nll": nll}


def bal_acc(y, s, thr):
    return metrics.threshold_metrics(y, s, thr)["balanced_accuracy"]


def val_scores():
    cache = RUN / "predictions" / "val_scores.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["s"], z["y"], z["dom"], z["fam"]
    ff = FeatureStore("faces", in_memory=False).split_rows("val")
    _, cdf = extra_rows("celebdf_v2_train")
    fe.DATASETS["cdf_val"] = "celebdf_v2_train"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_ff, _ = fe.score_e2e(RUN, "ffpp_test", ff, dev)    # dataset key only selects the FF++ faces folder
    s_cdf, _ = fe.score_e2e(RUN, "cdf_val", cdf, dev)
    rows = ff + cdf
    s = np.concatenate([s_ff, s_cdf])
    y = np.array([int(r["label_idx"] == paths.CLASS_TO_IDX["fake"]) for r in rows])
    dom = np.array(["ffpp"] * len(ff) + ["celebdf"] * len(cdf))
    fam = np.array([f"{d}:{r['family_id']}" for d, r in zip(dom, rows)])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, s=s, y=y, dom=dom, fam=fam)
    return s, y, dom, fam


def main():
    ckpt = torch.load(RUN / "checkpoints" / "best_trainable.pth", map_location="cpu", weights_only=True)
    thr_old = float(ckpt["threshold"])
    s, y, dom, fam = val_scores()
    w = cell_weights(y, dom)

    # cross-fitted calibration estimate on validation
    p_cf = np.zeros(len(s))
    for tr, te in GroupKFold(5).split(s, y, fam):
        a_, b_ = fit_platt(s[tr], y[tr], w[tr])
        p_cf[te] = sigmoid(a_ * logit(s[te]) + b_)
    a, b = fit_platt(s, y, w)
    thr_new = float(sigmoid(-b / a))  # raw score at which the calibrated probability is 0.5

    doms = ["ffpp", "celebdf"]
    val = {"n": {d: int((dom == d).sum()) for d in doms},
           "calibration_raw": calib_scores(y, s, w), "calibration_platt_crossfit": calib_scores(y, p_cf, w),
           "balanced_accuracy": {}}
    for name, t in (("current_threshold", thr_old), ("calibrated_0.5", thr_new)):
        per = {d: bal_acc(y[dom == d], s[dom == d], t) for d in doms}
        val["balanced_accuracy"][name] = {**per, "mean": float(np.mean(list(per.values()))), "raw_threshold": t}
        val["balanced_accuracy"][name].update({f"specificity_{d}": metrics.threshold_metrics(y[dom == d], s[dom == d], t)["specificity"] for d in doms})
    adopt = val["balanced_accuracy"]["calibrated_0.5"]["mean"] > val["balanced_accuracy"]["current_threshold"]["mean"]
    thr_served = thr_new if adopt else thr_old

    calib = {"format": "deepshield-calibration/1", "method": "platt on logit(score), (dataset x class)-balanced weights",
             "a": a, "b": b, "fit_on": "final_cdf validation: FF++ val + Celeb-DF v2 val",
             "decision_rule": "calibrated >= 0.5" if adopt else "raw score >= checkpoint threshold",
             "threshold_raw": thr_served, "threshold_checkpoint": thr_old, "adopted_new_rule": bool(adopt),
             "validation": val}
    (RUN / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n")

    # ---- test report: the fitted map applied once to the stored test scores ----
    report = {"a": a, "b": b, "threshold_raw": thr_served, "threshold_checkpoint": thr_old, "adopted": bool(adopt), "test": {}}
    for ds in TEST_SETS:
        f = paths.RESULTS_ROOT / "final" / ds / "final_cdf.npy"
        if not f.exists():
            continue
        st = FeatureStore("faces", dataset=fe.DATASETS[ds], in_memory=False)
        rows = st.split_rows("test", allow_test=True)
        yt, _, _ = engine.rows_to_arrays(rows, paths.CLASS_TO_IDX["fake"])
        st_s = np.load(f)
        wt = cell_weights(yt, np.zeros(len(yt)))  # class-balanced within the dataset
        pt = sigmoid(a * logit(st_s) + b)
        report["test"][ds] = {
            "n": int(len(yt)), "calibration_raw": calib_scores(yt, st_s, wt), "calibration_platt": calib_scores(yt, pt, wt),
            "old_rule": {k: metrics.threshold_metrics(yt, st_s, thr_old)[k] for k in ("balanced_accuracy", "fake_recall", "specificity")},
            "served_rule": {k: metrics.threshold_metrics(yt, st_s, thr_served)[k] for k in ("balanced_accuracy", "fake_recall", "specificity")}}
    out = paths.RESULTS_ROOT / "final" / "calibration_report.json"
    out.write_text(json.dumps({"validation": val, **report}, indent=2) + "\n")

    lines = [f"Served model calibration: p = sigmoid({a:.3f} * logit(score) + {b:.3f}); decision rule "
             f"{'calibrated p >= 0.5 (raw score >= %.3f)' % thr_new if adopt else 'unchanged (raw >= %.3f)' % thr_old}.",
             "", "Validation (FF++ val + Celeb-DF v2 val, cell-balanced weights):", "",
             "| | ECE | Brier | NLL |", "| --- | ---: | ---: | ---: |",
             f"| raw score | {val['calibration_raw']['ece']:.3f} | {val['calibration_raw']['brier']:.3f} | {val['calibration_raw']['nll']:.3f} |",
             f"| Platt, cross-fitted | {val['calibration_platt_crossfit']['ece']:.3f} | {val['calibration_platt_crossfit']['brier']:.3f} | {val['calibration_platt_crossfit']['nll']:.3f} |",
             "", "| Validation rule | Raw threshold | Bal. acc FF++ | Bal. acc Celeb-DF | Mean | Spec. FF++ | Spec. Celeb-DF |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in val["balanced_accuracy"].items():
        lines.append(f"| {name} | {r['raw_threshold']:.3f} | {r['ffpp']:.3f} | {r['celebdf']:.3f} | {r['mean']:.3f} | "
                     f"{r['specificity_ffpp']:.3f} | {r['specificity_celebdf']:.3f} |")
    lines += ["", "Test (stored final_cdf scores, map applied once; class-balanced weights):", "",
              "| Test set | ECE raw | ECE Platt | Spec. old | Spec. served | Fake recall old | Fake recall served | Bal. acc old | Bal. acc served |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ds, r in report["test"].items():
        o, n = r["old_rule"], r["served_rule"]
        lines.append(f"| {ds} | {r['calibration_raw']['ece']:.3f} | {r['calibration_platt']['ece']:.3f} | {o['specificity']:.3f} | "
                     f"{n['specificity']:.3f} | {o['fake_recall']:.3f} | {n['fake_recall']:.3f} | {o['balanced_accuracy']:.3f} | "
                     f"{n['balanced_accuracy']:.3f} |")
    md = "\n".join(lines) + "\n"
    (paths.RESULTS_ROOT / "final" / "calibration_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
