"""Validation metrics: discrimination, calibration and clinical error rates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from .config import RANDOM_SEED

_EPS = 1e-6


def diagnostic_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Metrics for probabilistic diagnosis at a decision ``threshold``.

    Sensitivity (recall for disease) matters most clinically, since a missed
    diagnosis is costlier than a false alarm, so it is reported next to
    specificity, PPV and NPV rather than folded into accuracy.
    """
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    if y_true.shape != p.shape:
        raise ValueError("y_true and p must have the same shape")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    safe = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "ppv": safe(tp, tp + fp),
        "npv": safe(tn, tn + fn),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    p: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    mass: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a test-set metric."""
    y_true, p = np.asarray(y_true), np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():  # AUC undefined for one class
            continue
        values.append(metric(y_true[idx], p[idx]))
    lo, hi = np.quantile(values, [(1 - mass) / 2, 1 - (1 - mass) / 2])
    return float(lo), float(hi)


def metrics_table(y_true: np.ndarray, predictions: dict[str, np.ndarray], threshold: float = 0.5) -> pd.DataFrame:
    """One row per model, with a bootstrap 95% CI for ROC-AUC."""
    rows = []
    for name, p in predictions.items():
        row = {"model": name, **diagnostic_metrics(y_true, p, threshold)}
        row["auc_ci_low"], row["auc_ci_high"] = bootstrap_ci(y_true, p, roc_auc_score)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def calibration_frame(y_true: np.ndarray, p: np.ndarray, n_bins: int = 8) -> pd.DataFrame:
    """Observed event rate against mean predicted probability (quantile bins)."""
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 8) -> float:
    """Bin-weighted mean |observed - predicted| over quantile bins."""
    y_true, p = np.asarray(y_true), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        ece += mask.mean() * abs(y_true[mask].mean() - p[mask].mean())
    return float(ece)
