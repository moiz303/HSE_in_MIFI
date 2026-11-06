"""Baseline: две стратегии предсказания SI.

Владелец: B (стадия 3–4).

Стратегия решается замером на одном разбиении, а не заранее:
  direct    — одна голова на log10(SI);
  two_head  — две головы на log10(IC50) и log10(CC50), SI считается из них.
"""

from __future__ import annotations

import math

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


def _clip_of(model) -> float | None:
    """Порог обрезки у модели, загруженной из старого pickle, отсутствует.

    Модели коммитятся в репозиторий, и судьи будут их загружать: падать на
    AttributeError из-за того, что объект сохранён предыдущей версией кода,
    недопустимо.
    """
    return getattr(model, "clip_si", None)


def _clip_log(values: np.ndarray, clip_si: float | None) -> np.ndarray:
    """Ограничить прогноз сверху правдоподобным максимумом SI.

    Метрика соревнования — MSE по сырому SI, и она определяется хвостом: три
    молекулы из 1184 дают 85% суммы квадратов. Прогноз выше того, что вообще
    встречается в данных организаторов, не может быть правильным, зато может
    стоить сотни миллионов в метрике.
    """
    if clip_si is None:
        return values
    return np.minimum(values, math.log10(clip_si))


class DirectSI:
    """Одна голова: признаки → log10(SI)."""

    def __init__(self, params: dict | None = None, clip_si: float | None = None) -> None:
        self.model = _model(params)
        self.clip_si = clip_si
        self.n_train: dict[str, int] = {}

    def fit(self, features: np.ndarray, si: np.ndarray) -> "DirectSI":
        target = np.asarray(si, dtype=float)
        if self.clip_si is not None:
            # Обрезаем цель: три молекулы из ChEMBL с SI до 2.2e5 лежат вне
            # диапазона соревнования (максимум 3.6e4) и перетягивают обучение
            # на себя. Обрезка цели — не подгонка под тест: порог взят из
            # train.csv организаторов, в тест мы не смотрим.
            target = np.minimum(target, self.clip_si)
        used, skipped = _fit_head(self.model, features, target, "SI")
        self.n_train = {"SI": used, "SI_skipped": skipped}
        return self

    def predict_log_si(self, features: np.ndarray) -> np.ndarray:
        return _clip_log(self.model.predict(features), _clip_of(self))


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

    def __init__(self, params: dict | None = None, clip_si: float | None = None) -> None:
        self.ic50 = _model(params)
        self.cc50 = _model(params)
        # Цель здесь не одна, а две физические величины, поэтому обрезать
        # «цель» нечего: ограничение применяется только к итоговому прогнозу.
        self.clip_si = clip_si
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
        return _clip_log(
            self.cc50.predict(features) - self.ic50.predict(features), _clip_of(self)
        )


class RawSI:
    """Регрессия прямо по сырому SI, без логарифма.

    Зачем. Обучение в логарифмах минимизирует ошибку логарифма, то есть даёт
    примерно условную медиану. Метрика соревнования — MSE в линейном масштабе,
    а ей нужно условное среднее. У правоскошенного распределения среднее
    заметно выше медианы, поэтому лог-модель систематически занижает прогноз
    именно там, где ошибка стоит дороже всего: на больших SI.

    Эта стратегия убирает рассогласование: squared_error считается по той же
    величине, которую оценивают.
    """

    def __init__(self, params: dict | None = None, clip_si: float | None = None) -> None:
        self.model = _model(params)
        self.clip_si = clip_si
        self.n_train: dict[str, int] = {}

    def fit(self, features: np.ndarray, si: np.ndarray) -> "RawSI":
        target = np.asarray(si, dtype=float)
        mask = np.isfinite(target) & (target > 0)
        if not mask.any():
            raise ValueError("нет ни одной строки с измеренным SI")
        if self.clip_si is not None:
            target = np.minimum(target, self.clip_si)
        self.model.fit(features[mask], target[mask])
        self.n_train = {"SI": int(mask.sum()), "SI_skipped": int((~mask).sum())}
        return self

    def predict_si(self, features: np.ndarray) -> np.ndarray:
        predicted = np.clip(self.model.predict(features), 1e-9, None)
        clip = _clip_of(self)
        if clip is not None:
            predicted = np.minimum(predicted, clip)
        return predicted

    def predict_log_si(self, features: np.ndarray) -> np.ndarray:
        return to_log(self.predict_si(features))


class SentinelSI:
    """Классификатор неактивности плюс регрессия SI на активных молекулах.

    Зачем. Метрика соревнования — MSE по сырому SI. В train.csv организаторов
    25% строк размечены меткой SI = 10000: это не измерение, а «активности не
    обнаружено». Если модель даёт на такой молекуле свою обычную медиану около
    5, одна строка стоит (10000-5)^2 ≈ 1e8. На 150 таких строках из 597 это
    около 2.5e7 метрики — больше, чем всё остальное вместе.

    Как. Вместо порога берём ожидаемое значение: для квадратичной ошибки
    оптимальный прогноз это E[y|x], то есть

        SI = p * 10000 + (1 - p) * SI_регрессии,

    где p — вероятность, что молекула размечена меткой. Порог терял бы
    информацию: при p = 0.5 и жёстком решении ошибка максимальна, а смесь
    даёт 5000 и режет квадрат ошибки вчетверо.

    Регрессия учится только на строках с измеренным SI: подмешивать туда
    метку 10000 значит учить её предсказывать не концентрацию, а разметку.
    """

    def __init__(self, params: dict | None = None, clip_si: float | None = None,
                 label_value: float = 10000.0, regressor: str = "log") -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        merged = dict(DEFAULT_PARAMS)
        merged.update(params or {})
        self.classifier = HistGradientBoostingClassifier(**merged)
        # "raw" согласует регрессию с метрикой, "log" устойчивее на хвосте.
        # Что лучше — замеряется, а не выбирается верой.
        self.regressor = (
            RawSI(params, clip_si) if regressor == "raw" else DirectSI(params, clip_si)
        )
        self.regressor_kind = regressor
        self.label_value = label_value
        self.clip_si = clip_si
        self.n_train: dict[str, int] = {}
        self._single_class: bool | None = None

    def fit(self, features: np.ndarray, si: np.ndarray, is_label: np.ndarray) -> "SentinelSI":
        labels = np.asarray(is_label, dtype=bool)
        if labels.all() or not labels.any():
            # Один класс: классификатор не нужен и обучаться не может.
            self._single_class = bool(labels.all())
        else:
            self._single_class = None
            self.classifier.fit(features, labels.astype(int))

        self.regressor.fit(features[~labels], np.asarray(si, dtype=float)[~labels])
        self.n_train = {
            "регрессия": int((~labels).sum()),
            "метки": int(labels.sum()),
            "доля меток": round(float(labels.mean()), 3),
        }
        return self

    def predict_proba_label(self, features: np.ndarray) -> np.ndarray:
        if self._single_class is not None:
            return np.full(features.shape[0], 1.0 if self._single_class else 0.0)
        return self.classifier.predict_proba(features)[:, 1]

    def predict_si(self, features: np.ndarray) -> np.ndarray:
        """Ожидаемое значение SI в линейном масштабе — так оценивает метрика."""
        probability = self.predict_proba_label(features)
        regression = (
            self.regressor.predict_si(features)
            if hasattr(self.regressor, "predict_si")
            else from_log(self.regressor.predict_log_si(features))
        )
        blended = probability * self.label_value + (1.0 - probability) * regression
        clip = _clip_of(self)
        if clip is not None:
            blended = np.minimum(blended, max(clip, self.label_value))
        return blended

    def predict_log_si(self, features: np.ndarray) -> np.ndarray:
        return to_log(self.predict_si(features))


def build(strategy: str, params: dict | None = None, clip_si: float | None = None):
    """Собрать модель. clip_si — правдоподобный максимум SI или None."""
    if strategy == "direct":
        return DirectSI(params, clip_si)
    if strategy == "two_head":
        return TwoHeadSI(params, clip_si)
    if strategy == "raw":
        return RawSI(params, clip_si)
    if strategy == "sentinel":
        return SentinelSI(params, clip_si, regressor="log")
    if strategy == "sentinel_raw":
        return SentinelSI(params, clip_si, regressor="raw")
    raise ValueError(
        f"неизвестная стратегия: {strategy!r}; доступны: "
        "direct, raw, two_head, sentinel, sentinel_raw"
    )
