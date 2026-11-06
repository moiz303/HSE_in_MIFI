"""Baseline: две стратегии предсказания SI.

Владелец: B (стадия 3–4).

Стратегия решается замером на одном разбиении, а не заранее:
  direct    — одна голова на log10(SI);
  two_head  — две головы на log10(IC50) и log10(CC50), SI считается из них.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

DEFAULT_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.06,
    "max_depth": None,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
    "random_state": 20261106,
}


def _model(params: dict | None = None) -> HistGradientBoostingRegressor:
    merged = dict(DEFAULT_PARAMS)
    merged.update(params or {})
    return HistGradientBoostingRegressor(**merged)


def to_log(values: np.ndarray) -> np.ndarray:
    """Концентрации и SI живут на порядках, поэтому учимся в логах."""
    array = np.asarray(values, dtype=float)
    return np.log10(np.clip(array, 1e-9, None))


def from_log(values: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(values, dtype=float))


class DirectSI:
    """Одна голова: признаки → log10(SI)."""

    def __init__(self, params: dict | None = None) -> None:
        self.model = _model(params)

    def fit(self, features: np.ndarray, si: np.ndarray) -> "DirectSI":
        self.model.fit(features, to_log(si))
        return self

    def predict_log_si(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


class TwoHeadSI:
    """Две головы: признаки → log10(IC50) и log10(CC50), затем разность.

    log10(SI) = log10(CC50) - log10(IC50), поэтому разность в логах и есть
    предсказание индекса селективности: отдельного деления не требуется.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.ic50 = _model(params)
        self.cc50 = _model(params)

    def fit(self, features: np.ndarray, ic50: np.ndarray, cc50: np.ndarray) -> "TwoHeadSI":
        self.ic50.fit(features, to_log(ic50))
        self.cc50.fit(features, to_log(cc50))
        return self

    def predict_log_si(self, features: np.ndarray) -> np.ndarray:
        return self.cc50.predict(features) - self.ic50.predict(features)


def build(strategy: str, params: dict | None = None):
    if strategy == "direct":
        return DirectSI(params)
    if strategy == "two_head":
        return TwoHeadSI(params)
    raise ValueError(f"неизвестная стратегия: {strategy!r}, ожидалось direct или two_head")
