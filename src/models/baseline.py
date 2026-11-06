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


def _usable(values: np.ndarray) -> np.ndarray:
    """Маска строк, где величина измерена и положительна.

    Пропуски не заполняются: подставить медиану в IC50 значит выдумать
    измерение, которого не было, и дальше модель учится на фантазии.
    """
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array > 0)


def _fit_head(model, features: np.ndarray, values: np.ndarray, name: str):
    """Обучить голову только на строках, где её величина есть."""
    mask = _usable(values)
    if not mask.any():
        raise ValueError(f"нет ни одной строки с измеренным {name}")
    model.fit(features[mask], to_log(values[mask]))
    return int(mask.sum()), int((~mask).sum())


class DirectSI:
    """Одна голова: признаки → log10(SI)."""

    def __init__(self, params: dict | None = None) -> None:
        self.model = _model(params)
        self.n_train: dict[str, int] = {}

    def fit(self, features: np.ndarray, si: np.ndarray) -> "DirectSI":
        used, skipped = _fit_head(self.model, features, si, "SI")
        self.n_train = {"SI": used, "SI_skipped": skipped}
        return self

    def predict_log_si(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


class TwoHeadSI:
    """Две головы: признаки → log10(IC50) и log10(CC50), затем разность.

    log10(SI) = log10(CC50) - log10(IC50), поэтому разность в логах и есть
    предсказание индекса селективности: отдельного деления не требуется.

    Каждая голова обучается на своих строках. Это не мелочь реализации, а
    смысл конструкции: цитотоксичности не нужна парная активность, поэтому
    голова CC50 видит строки, которых модель прямого SI не видит вовсе. В
    нашей выгрузке ChEMBL таких непарных записей 806 для CC50 и 671 для IC50.

    Обратная сторона: если у строки нет IC50, а цель задана меткой (SI = 10000
    у организаторов), эта конструкция метку выучить не может — она собирает SI
    из двух величин, одной из которых нет. Такую разметку умеет только
    стратегия direct.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.ic50 = _model(params)
        self.cc50 = _model(params)
        self.n_train: dict[str, int] = {}

    def fit(self, features: np.ndarray, ic50: np.ndarray, cc50: np.ndarray) -> "TwoHeadSI":
        used_ic50, skipped_ic50 = _fit_head(self.ic50, features, ic50, "IC50")
        used_cc50, skipped_cc50 = _fit_head(self.cc50, features, cc50, "CC50")
        self.n_train = {
            "IC50": used_ic50,
            "IC50_skipped": skipped_ic50,
            "CC50": used_cc50,
            "CC50_skipped": skipped_cc50,
        }
        return self

    def predict_log_si(self, features: np.ndarray) -> np.ndarray:
        return self.cc50.predict(features) - self.ic50.predict(features)


def build(strategy: str, params: dict | None = None):
    if strategy == "direct":
        return DirectSI(params)
    if strategy == "two_head":
        return TwoHeadSI(params)
    raise ValueError(f"неизвестная стратегия: {strategy!r}, ожидалось direct или two_head")
