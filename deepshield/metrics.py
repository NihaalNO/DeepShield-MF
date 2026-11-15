"""Video-level metrics. Positive class = fake throughout."""
import numpy as np
from sklearn.metrics import roc_auc_score


def aggregate_videos(p_fake_windows, video_index, n_videos):
    """Mean fake probability of each video's windows -> [n_videos]."""
    sums = np.bincount(video_index, weights=p_fake_windows, minlength=n_videos)
    counts = np.bincount(video_index, minlength=n_videos)
    return sums / np.maximum(counts, 1)


def confusion(y_fake, pred_fake):
    tp = int(((y_fake == 1) & (pred_fake == 1)).sum())
    tn = int(((y_fake == 0) & (pred_fake == 0)).sum())
    fp = int(((y_fake == 0) & (pred_fake == 1)).sum())
    fn = int(((y_fake == 1) & (pred_fake == 0)).sum())
    return tp, tn, fp, fn


def threshold_metrics(y_fake, p_fake, threshold):
    pred = (p_fake >= threshold).astype(int)
    tp, tn, fp, fn = confusion(y_fake, pred)
    recall_fake = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)  # recall of the real class
    prec_fake, prec_real = tp / max(tp + fp, 1), tn / max(tn + fn, 1)
    f1_fake = 2 * prec_fake * recall_fake / max(prec_fake + recall_fake, 1e-12)
    f1_real = 2 * prec_real * specificity / max(prec_real + specificity, 1e-12)
    return {
        "accuracy": (tp + tn) / max(len(y_fake), 1),
        "balanced_accuracy": (recall_fake + specificity) / 2,
        "macro_f1": (f1_fake + f1_real) / 2,
        "fake_recall": recall_fake,
        "specificity": specificity,
        "confusion_matrix": {"tp_fake": tp, "tn_real": tn, "fp_real_as_fake": fp, "fn_fake_as_real": fn},
    }


def safe_auc(y_fake, p_fake):
    return float(roc_auc_score(y_fake, p_fake)) if 0 < y_fake.sum() < len(y_fake) else float("nan")


def pick_threshold(y_fake, p_fake):
    """Threshold maximising balanced accuracy. Call on validation data only."""
    candidates = np.unique(np.concatenate([p_fake, [0.5]]))
    scores = [threshold_metrics(y_fake, p_fake, t)["balanced_accuracy"] for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def video_report(y_fake, p_fake, methods, threshold):
    """Overall metrics plus a per-manipulation-method breakdown (each method vs. all reals)."""
    y_fake, p_fake, methods = np.asarray(y_fake), np.asarray(p_fake), np.asarray(methods)
    report = {"n_videos": int(len(y_fake)), "n_fake": int(y_fake.sum()), "n_real": int((1 - y_fake).sum()),
              "threshold": float(threshold), "roc_auc": safe_auc(y_fake, p_fake),
              **threshold_metrics(y_fake, p_fake, threshold)}
    real = methods == "original"
    per_method = {}
    for m in sorted(set(methods[~real])):
        sel = real | (methods == m)
        per_method[m] = {"n_fake": int((methods == m).sum()), "roc_auc": safe_auc(y_fake[sel], p_fake[sel]),
                         "fake_recall": float((p_fake[methods == m] >= threshold).mean())}
    report["per_method"] = per_method
    return report


def bootstrap_ci(y_fake, p_fake, families, threshold, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CIs from resampling source families (videos of a family are not independent)."""
    y_fake, p_fake, families = np.asarray(y_fake), np.asarray(p_fake), np.asarray(families)
    uniq, inverse = np.unique(families, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    keys = ("roc_auc", "balanced_accuracy", "macro_f1", "fake_recall", "specificity", "accuracy")
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        y, p = y_fake[idx], p_fake[idx]
        m = threshold_metrics(y, p, threshold)
        m["roc_auc"] = safe_auc(y, p)
        for k in keys:
            samples[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))] for k, v in samples.items()}
