"""Evaluation metrics suited to heavily imbalanced fraud data."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)


def evaluate(y, score, threshold: float) -> dict:
    y = np.asarray(y)
    score = np.asarray(score)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, score)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "flag_rate": float(pred.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def curves(y, score, n_points: int = 300) -> dict:
    """PR and ROC curves sampled on even recall / log-spaced FPR grids (small enough for JSON).

    Thinning by index would keep almost only the low-precision tail of a 1M-point curve.
    """
    p, r, _ = precision_recall_curve(y, score)          # recall decreasing
    fpr, tpr, _ = roc_curve(y, score)                   # fpr increasing
    pr_idx = np.unique(np.searchsorted(-r, -np.linspace(1, 0, n_points), side="left").clip(0, len(r) - 1))
    roc_grid = np.concatenate([[0], np.logspace(-5, 0, n_points)])
    roc_idx = np.unique(np.searchsorted(fpr, roc_grid, side="right").clip(1, len(fpr)) - 1)
    return {
        "pr": {"recall": r[pr_idx].round(5).tolist(), "precision": p[pr_idx].round(5).tolist()},
        "roc": {"fpr": fpr[roc_idx].round(7).tolist(), "tpr": tpr[roc_idx].round(5).tolist()},
    }
