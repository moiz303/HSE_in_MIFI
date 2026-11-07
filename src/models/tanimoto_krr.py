"""Independent Tanimoto kernel-ridge heads for log10(IC50) and log10(CC50)."""

from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge


def tanimoto_kernel(left, right=None) -> np.ndarray:
    """Compute the binary Morgan-fingerprint Tanimoto kernel."""
    left = np.asarray(left)
    right = left if right is None else np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("fingerprints must be 2D matrices with matching feature counts")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("fingerprints must be finite")
    if ((left != 0) & (left != 1)).any() or ((right != 0) & (right != 1)).any():
        raise ValueError("Tanimoto KRR requires binary fingerprints")
    left, right = left.astype(bool), right.astype(bool)
    intersection = left.astype(np.float64) @ right.T.astype(np.float64)
    left_size = left.sum(axis=1, dtype=np.float64)[:, None]
    right_size = right.sum(axis=1, dtype=np.float64)[None, :]
    union = left_size + right_size - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class TanimotoKRR:
    """Two independently masked KRR regressors trained on log10 concentrations."""

    def __init__(self, params: dict | None = None, clip_si: float | None = None):
        params = dict(params or {})
        unknown = set(params) - {"alpha"}
        if unknown:
            raise ValueError(f"unknown Tanimoto KRR parameters: {sorted(unknown)}")
        alpha = params.get("alpha", 1.0)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        if clip_si is not None and (not np.isfinite(clip_si) or clip_si <= 0):
            raise ValueError("clip_si must be positive and finite")
        self.params = {"alpha": float(alpha)}
        self.clip_si = clip_si
        self.heads = {}
        self.n_train = {}

    def fit(self, fingerprints, ic50, cc50):
        fingerprints = np.asarray(fingerprints)
        if fingerprints.ndim != 2 or not len(fingerprints):
            raise ValueError("fingerprints must be a non-empty 2D matrix")
        trainable = tanimoto_kernel(fingerprints)
        targets = {"IC50": np.asarray(ic50, dtype=float), "CC50": np.asarray(cc50, dtype=float)}
        self.heads = {}
        for name, values in targets.items():
            if values.shape != (len(fingerprints),):
                raise ValueError(f"{name}: target length does not match fingerprints")
            mask = np.isfinite(values) & (values > 0)
            if not mask.any():
                raise ValueError(f"{name}: no exact positive measurements")
            model = KernelRidge(alpha=self.params["alpha"], kernel="precomputed")
            model.fit(trainable[np.ix_(mask, mask)], np.log10(values[mask]))
            self.heads[name] = {"model": model, "fingerprints": fingerprints[mask].copy()}
            self.n_train[name] = int(mask.sum())
            self.n_train[name + "_skipped"] = int((~mask).sum())
        self.n_features_in_ = fingerprints.shape[1]
        return self

    def predict_log_concentrations(self, fingerprints):
        fingerprints = np.asarray(fingerprints)
        if set(self.heads) != {"IC50", "CC50"}:
            raise ValueError("fit both concentration heads first")
        if fingerprints.ndim != 2 or fingerprints.shape[1] != self.n_features_in_:
            raise ValueError("fingerprint shape does not match training")
        result = {}
        for name, head in self.heads.items():
            kernel = tanimoto_kernel(fingerprints, head["fingerprints"])
            result[name] = head["model"].predict(kernel)
        if not all(np.isfinite(values).all() for values in result.values()):
            raise ValueError("non-finite concentration prediction")
        return result

    def predict_log_si(self, fingerprints):
        prediction = self.predict_log_concentrations(fingerprints)
        log_si = prediction["CC50"] - prediction["IC50"]
        if self.clip_si is not None:
            log_si = np.minimum(log_si, np.log10(self.clip_si))
        return log_si

    def predict_si(self, fingerprints):
        result = np.power(10.0, self.predict_log_si(fingerprints))
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("SI prediction is outside the finite positive range")
        return result
