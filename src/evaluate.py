"""
Metrics + bootstrap statistical comparison (Задачи 9, 13, 14 in TASKA.md).

All metrics are computed in log10(SI) space, as required by TASKA.md §2/§6.

    python -m src.evaluate --config configs/baseline.json
(invoked by src/pipeline.py's "evaluate" command; see that module for the
end-to-end CLI described in Задача 12)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rho = float(spearmanr(y_true, y_pred).correlation)
    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": rho}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "rmse",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Non-parametric bootstrap CI for one metric, resampling molecules
    with replacement (Задача 14: 'statistical uncertainty via bootstrap').
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    metric_fn = {
        "rmse": lambda a, b: np.sqrt(mean_squared_error(a, b)),
        "mae": mean_absolute_error,
        "r2": r2_score,
        "spearman": lambda a, b: spearmanr(a, b).correlation,
    }[metric]
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.array(stats)
    alpha = 1 - ci
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point_estimate": float(metric_fn(y_true, y_pred)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def paired_bootstrap_difference(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str = "rmse",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for (metric(A) - metric(B)) on the SAME resampled
    molecules each iteration (paired), to judge whether the gap between
    two strategies (e.g. direct vs two_head, Задача 14) is larger than
    bootstrap noise. If the CI for the difference excludes 0, the
    difference is considered statistically meaningful at this level.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric_fn = {
        "rmse": lambda a, b: np.sqrt(mean_squared_error(a, b)),
        "mae": mean_absolute_error,
        "r2": r2_score,
        "spearman": lambda a, b: spearmanr(a, b).correlation,
    }[metric]
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx]))
    diffs = np.array(diffs)
    alpha = 1 - ci
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)
    return {
        "point_estimate_diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant_at_ci": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }


def repeat_split_stability(rmse_values: List[float]) -> Dict[str, float]:
    """Задача 14: summarize a metric across several independent
    scaffold-split repeats (different seeds)."""
    arr = np.array(rmse_values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_repeats": len(arr),
    }
