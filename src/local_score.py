"""Raw-SI MSE and paired scaffold-cluster uncertainty. No leaderboard fitting."""
import numpy as np


def score(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth):
        raise ValueError("Scoring requires nonempty aligned one-dimensional arrays")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all() or (truth < 0).any() or (predicted < 0).any():
        raise ValueError("Scoring requires finite nonnegative values; no invalid rows are silently dropped")
    error = predicted-truth
    positive = (truth > 0) & (predicted > 0)
    return {"n": len(truth), "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(abs(error))),
        "bias": float(error.mean()), "rmse_log10": float(np.sqrt(np.mean(
            (np.log10(predicted[positive])-np.log10(truth[positive]))**2))) if positive.any() else None}


def paired_interval(rows, candidate, baseline="variant1", draws=2000):
    # Average loss over repeats per ORIGINAL row; repeats are not independent samples.
    delta = (rows[baseline]-rows.SI_true)**2 - (rows[candidate]-rows.SI_true)**2
    work = rows[["original_row", "scaffold"]].copy()
    work["gain"] = delta
    per_row = work.groupby(["original_row", "scaffold"], sort=False).gain.mean().reset_index()
    groups = per_row.groupby("scaffold").gain.agg(["sum", "size"])
    rng = np.random.default_rng(20261106)
    indices = rng.integers(0, len(groups), (draws, len(groups)))
    samples = groups["sum"].to_numpy()[indices].sum(axis=1) / groups["size"].to_numpy()[indices].sum(axis=1)
    return {"mean_mse_gain": float(per_row.gain.mean()),
        "ci95": np.quantile(samples, [.025, .975]).tolist(),
        "scaffold_clusters": len(groups), "method": "paired scaffold-cluster bootstrap, repeats averaged by original row"}
