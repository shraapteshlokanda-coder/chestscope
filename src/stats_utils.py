"""Metrics, bootstrap CIs, McNemar, thresholds. numpy/sklearn/scipy only."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def pick_threshold(y, p, target_recall=0.95):
    """Highest threshold whose recall (sensitivity) on y,p is >= target_recall."""
    fpr, tpr, thr = roc_curve(y, p)
    idx = int(np.argmax(tpr >= target_recall))
    t = float(thr[idx])
    return 1.0 if not np.isfinite(t) else min(t, 1.0)


def all_metrics(y, p, thresh=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                recall=rec, specificity=spec, precision=prec, f1=f1,
                balanced_accuracy=(rec + spec) / 2, accuracy=float((pred == y).mean()),
                tp=tp, fp=fp, tn=tn, fn=fn)


METRICS = ["auc", "pr_auc", "recall", "specificity", "precision", "f1", "balanced_accuracy"]


def bootstrap_ci(y, p, thresh=0.5, n=1000, seed=0, alpha=0.05):
    """Returns ({metric: (point, lo, hi)}, point_metrics) with percentile bootstrap CIs."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    point = all_metrics(y, p, thresh)
    samples = {m: [] for m in METRICS}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        m = all_metrics(y[idx], p[idx], thresh)
        for k in METRICS:
            samples[k].append(m[k])
    out = {}
    for k in METRICS:
        lo, hi = np.percentile(samples[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = (point[k], float(lo), float(hi))
    return out, point


def paired_bootstrap_auc_diff(y, p_a, p_b, n=2000, seed=0):
    """AUC(a) - AUC(b) with a paired bootstrap 95% CI and approximate two-sided p-value."""
    y = np.asarray(y).astype(int); p_a = np.asarray(p_a, float); p_b = np.asarray(p_b, float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(diff=float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
                lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)),
                p_value=float(min(max(p, 1.0 / len(diffs)), 1.0)))


def mcnemar_exact(y, pred_a, pred_b):
    """Exact McNemar test on paired correctness. b = only A correct, c = only B correct."""
    y = np.asarray(y).astype(int)
    a_ok = np.asarray(pred_a).astype(int) == y
    b_ok = np.asarray(pred_b).astype(int) == y
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    pval = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return dict(only_a_correct=b, only_b_correct=c, p_value=pval)