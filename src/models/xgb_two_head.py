"""Два независимых XGBoost на log10 концентраций в µM; SI из прогнозов."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "reg_alpha": 0.1,
    "random_state": 20261106,
    "n_jobs": 4,
}


def exact_target(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Не выдавать границы цензуры за точные измерения. Маска у каждой головы своя.

    После агрегации смешанные exact/censored группы исключаются консервативно:
    агрегированная концентрация уже может содержать порог вместо измерения.
    Метка SI=10000 сама по себе не отменяет имеющуюся точную концентрацию.
    """
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float).copy()
    qualifier = name + "_qualifier"
    if qualifier not in frame:
        raise ValueError(f"нет {qualifier}: нельзя проверить цензуру {name}")
    marked = frame[qualifier].astype("string").fillna("").str.strip().ne("").to_numpy(dtype=bool)
    values[marked | ~np.isfinite(values) | (values <= 0)] = np.nan
    return values


class XGBTwoHead:
    def __init__(self, params: dict | None = None, clip_si: float | None = None):
        unknown = set(params or {}) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"неизвестные параметры XGBoost: {sorted(unknown)}")
        if clip_si is not None and (not np.isfinite(clip_si) or clip_si <= 0):
            raise ValueError("clip_si должен быть положительным конечным числом")
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.clip_si = clip_si
        self.heads = {}
        self.n_train = {}

    def fit(self, features, ic50, cc50):
        features = np.asarray(features)
        targets = {"IC50": np.asarray(ic50, dtype=float), "CC50": np.asarray(cc50, dtype=float)}
        masks = {}
        for name, values in targets.items():
            if values.shape != (len(features),):
                raise ValueError(f"{name}: длина цели не совпадает с признаками")
            masks[name] = np.isfinite(values) & (values > 0)
            if not masks[name].any():
                raise ValueError(f"{name}: нет точных положительных измерений")
        self.heads = {}
        self.n_features_in_ = features.shape[1]
        for name, values in targets.items():
            mask = masks[name]
            model = XGBRegressor(
                **self.params, objective="reg:squarederror", tree_method="hist", device="cpu"
            )
            model.fit(features[mask], np.log10(values[mask]))
            self.heads[name] = model
            self.n_train[name] = int(mask.sum())
            self.n_train[name + "_skipped"] = int((~mask).sum())
        return self

    def predict_log_concentrations(self, features):
        if set(self.heads) != {"IC50", "CC50"}:
            raise ValueError("сначала обучите обе головы")
        if np.asarray(features).shape[1] != self.n_features_in_:
            raise ValueError("набор признаков не совпадает с обучением")
        predicted = {name: model.predict(features).astype(float) for name, model in self.heads.items()}
        if not all(np.isfinite(values).all() for values in predicted.values()):
            raise ValueError("нечисловой прогноз концентрации")
        return predicted

    def predict_log_si(self, features):
        values = self.predict_log_concentrations(features)
        result = values["CC50"] - values["IC50"]
        if self.clip_si is not None:
            result = np.minimum(result, np.log10(self.clip_si))
        return result

    def predict_si(self, features):
        result = np.power(10.0, self.predict_log_si(features))
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("SI вне численно допустимого диапазона")
        return result
