"""Одна модель с переключателями: масштаб цели, функция потерь, веса, обучатель.

Зачем отдельный класс. Метрика соревнования — MSE по сырому SI, и к ней ведут
разные рычаги, которые надо перебрать, а не выбрать верой:

* **масштаб цели.** Обучение в логарифмах минимизирует ошибку логарифма и даёт
  примерно условную медиану; MSE в линейном масштабе требует условного
  среднего. У правоскошенного SI среднее сильно выше медианы. `sqrt` — середина
  между двумя крайностями.
* **функция потерь.** `squared_error` подгоняет среднее, но выбросы в обучении
  тянут его на себя. `absolute_error` устойчив, но целится в медиану. `gamma` и
  `poisson` созданы для положительных правоскошенных величин и моделируют
  именно среднее. `quantile` с высоким уровнем — грубое приближение среднего
  сверху.
* **веса.** Вклад строки в метрику растёт как SI², поэтому имеет смысл дать
  большим SI больший вес и при обучении.
* **обучатель.** HistGradientBoosting, XGBoost и случайный лес. Лес —
  классический сильный baseline в QSAR на фингерпринтах, и у него есть своя
  причина быть лучше именно здесь: он даёт лучше калиброванные вероятности, а
  смесь для класса «неактивно» взвешена по вероятности и оптимальна только при
  честной вероятности. Обратная сторона: лес усредняет по деревьям, стягивает
  прогноз к общему среднему и хуже достаёт до крайних значений — а метрику
  определяет как раз хвост.

Комбинации перебирает `sweep.py`; здесь только механика.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.06,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
    "random_state": 20261106,
}

# Параметры леса и XGBoost, которых HistGradientBoostingRegressor не принимает.
FOREST_ONLY_PARAMS = ("n_estimators", "max_features", "criterion", "n_jobs")

# Потери, которым нужна строго положительная цель — значит только сырой масштаб.
POSITIVE_ONLY_LOSSES = ("poisson", "gamma")

TRANSFORMS = ("raw", "log", "sqrt")
LOSSES = ("squared_error", "absolute_error", "poisson", "gamma", "quantile")
LEARNERS = ("hgb", "xgb", "rf", "extra")

# Лес предсказывает среднее по листу, поэтому квантиль и гамма ему не выразить:
# у него настраивается только критерий расщепления.
RF_CRITERIA = {
    "squared_error": "squared_error",
    "absolute_error": "absolute_error",
    "poisson": "poisson",
}


def _forward(values: np.ndarray, transform: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if transform == "raw":
        return array
    if transform == "log":
        return np.log10(np.clip(array, 1e-9, None))
    if transform == "sqrt":
        return np.sqrt(np.clip(array, 0.0, None))
    raise ValueError(f"неизвестный масштаб цели: {transform!r}")


def _inverse(values: np.ndarray, transform: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if transform == "raw":
        return np.clip(array, 0.0, None)
    if transform == "log":
        return np.power(10.0, array)
    if transform == "sqrt":
        return np.clip(array, 0.0, None) ** 2
    raise ValueError(f"неизвестный масштаб цели: {transform!r}")


class FlexibleSI:
    """Регрессия SI с настраиваемыми масштабом, потерями, весами и обучателем."""

    def __init__(
        self,
        transform: str = "raw",
        loss: str = "squared_error",
        clip_si: float | None = None,
        weight_power: float = 0.0,
        quantile: float = 0.8,
        learner: str = "hgb",
        params: dict | None = None,
    ) -> None:
        if transform not in TRANSFORMS:
            raise ValueError(f"масштаб {transform!r}; доступны {TRANSFORMS}")
        if loss not in LOSSES:
            raise ValueError(f"потери {loss!r}; доступны {LOSSES}")
        if learner not in LEARNERS:
            raise ValueError(f"обучатель {learner!r}; доступны {LEARNERS}")
        if loss in POSITIVE_ONLY_LOSSES and transform != "raw":
            raise ValueError(
                f"потери {loss} требуют положительной цели, поэтому только transform='raw'"
            )
        if learner in ("rf", "extra") and loss not in RF_CRITERIA:
            raise ValueError(
                f"лес предсказывает среднее по листу, потери {loss} ему недоступны; "
                f"доступны {tuple(RF_CRITERIA)}"
            )

        self.transform = transform
        self.loss = loss
        self.clip_si = clip_si
        self.weight_power = float(weight_power)
        self.quantile = float(quantile)
        self.learner = learner
        self.params = dict(params or {})
        self.model = None
        self.n_train: dict[str, int] = {}

    # ------------------------------------------------------------------ сборка
    def _build(self):
        merged = dict(DEFAULT_PARAMS)
        merged.update(self.params)

        if self.learner == "xgb":
            from xgboost import XGBRegressor

            objective = {
                "squared_error": "reg:squarederror",
                "absolute_error": "reg:absoluteerror",
                "poisson": "count:poisson",
                "gamma": "reg:gamma",
                "quantile": "reg:quantileerror",
            }[self.loss]
            kwargs = dict(
                n_estimators=merged["max_iter"],
                learning_rate=merged["learning_rate"],
                reg_lambda=merged["l2_regularization"],
                min_child_weight=merged["min_samples_leaf"],
                random_state=merged["random_state"],
                objective=objective,
                tree_method="hist",
            )
            if self.loss == "quantile":
                kwargs["quantile_alpha"] = self.quantile
            return XGBRegressor(**kwargs)

        if self.learner in ("rf", "extra"):
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

            forest = ExtraTreesRegressor if self.learner == "extra" else RandomForestRegressor
            return forest(
                n_estimators=self.params.get("n_estimators", 500),
                criterion=RF_CRITERIA[self.loss],
                min_samples_leaf=self.params.get("min_samples_leaf", 1),
                max_features=self.params.get("max_features", 0.3),
                random_state=merged["random_state"],
                n_jobs=-1,
            )

        from sklearn.ensemble import HistGradientBoostingRegressor

        # Параметры, которых у HistGradientBoosting нет: он падает на незнакомом
        # ключе, а sweep.py передаёт один общий набор для скрининга всем
        # обучателям. Без этого фильтра вся ветка hgb молча «пропускалась».
        kwargs = dict(
            {key: value for key, value in merged.items() if key not in FOREST_ONLY_PARAMS},
            loss=self.loss,
        )
        if self.loss == "quantile":
            kwargs["quantile"] = self.quantile
        return HistGradientBoostingRegressor(**kwargs)

    # ----------------------------------------------------------------- обучение
    def fit(self, features: np.ndarray, si: np.ndarray) -> "FlexibleSI":
        target = np.asarray(si, dtype=float)
        usable = np.isfinite(target) & (target > 0)
        if not usable.any():
            raise ValueError("нет ни одной строки с измеренным SI")

        capped = target.copy()
        if self.clip_si is not None:
            capped = np.minimum(capped, self.clip_si)

        weights = None
        if self.weight_power:
            # Вклад строки в MSE растёт как SI^2; weight_power задаёт, насколько
            # сильно это учитывать при обучении.
            weights = capped[usable] ** self.weight_power
            weights = weights / weights.mean()

        self.model = self._build()
        self.model.fit(
            features[usable], _forward(capped[usable], self.transform), sample_weight=weights
        )
        self.n_train = {"SI": int(usable.sum()), "SI_skipped": int((~usable).sum())}
        return self

    # --------------------------------------------------------------- применение
    def predict_si(self, features: np.ndarray) -> np.ndarray:
        predicted = _inverse(self.model.predict(features), self.transform)
        predicted = np.clip(predicted, 0.0, None)
        clip = getattr(self, "clip_si", None)
        if clip is not None:
            predicted = np.minimum(predicted, clip)
        return predicted

    def predict_log_si(self, features: np.ndarray) -> np.ndarray:
        return np.log10(np.clip(self.predict_si(features), 1e-9, None))

    def describe(self) -> str:
        parts = [self.learner, self.transform, self.loss]
        if self.loss == "quantile":
            parts.append(f"q={self.quantile:g}")
        if self.weight_power:
            parts.append(f"w=SI^{self.weight_power:g}")
        if self.clip_si is not None:
            parts.append(f"clip={self.clip_si:g}")
        return " · ".join(parts)
