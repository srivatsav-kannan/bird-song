"""Evaluation utilities: window->recording aggregation, bootstrap CIs, McNemar."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

SPECIES = ["BanasuraLaughingthrush", "BugunLiocichla", "ForestOwlet", "JerdonsCourser"]


def aggregate_recordings(window_probs: np.ndarray, recording_ids: list[str]):
    """Mean-probability aggregation of window predictions per recording."""
    recs = {}
    for p, rid in zip(window_probs, recording_ids):
        recs.setdefault(rid, []).append(p)
    ids = sorted(recs)
    probs = np.stack([np.mean(recs[r], axis=0) for r in ids])
    return ids, probs


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> dict:
    labels = list(range(n_classes))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            (SPECIES + ["Other"])[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                                       "f1": float(f1[i]), "support": int(support[i])}
            for i in labels
        },
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over evaluation units. Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_pred)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact binomial McNemar test on paired correct/incorrect outcomes."""
    from scipy.stats import binomtest
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n01 = int((~a_ok & b_ok).sum())   # A wrong, B right
    n10 = int((a_ok & ~b_ok).sum())   # A right, B wrong
    n = n01 + n10
    p = binomtest(min(n01, n10), n, 0.5).pvalue if n > 0 else 1.0
    return {"n_A_only_correct": n10, "n_B_only_correct": n01, "p_value": float(p)}
