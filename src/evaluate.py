"""Метрики и доверительные интервалы.

Владелец: B и D (стадии 4 и 8).

Все метрики считаются в лог-пространстве: в линейном масштабе одна молекула
с SI = 5000 определяет и RMSE, и знак улучшения.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def metrics(log_true: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(log_true, dtype=float)
    pred = np.asarray(log_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("размеры истинных и предсказанных значений не совпадают")
    if true.size == 0:
        raise ValueError("пустая выборка")

    residual = pred - true
    denominator = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / denominator if denominator > 0 else float("nan")

    spearman = float("nan")
    if true.size > 2:
        spearman = float(stats.spearmanr(true, pred).statistic)

    return {
        "rmse_log10": float(np.sqrt((residual**2).mean())),
        "mae_log10": float(np.abs(residual).mean()),
        "r2_log10": r2,
        "spearman": spearman,
        "n": int(true.size),
    }


def competition_mse(log_true: np.ndarray, log_pred: np.ndarray) -> float:
    """MSE по сырому SI — метрика соревнования.

    Лог-пространство удобно для обучения, но оценивают нас не в нём. Ошибка
    0.8 log10 — это множитель 6.3: на молекуле с SI = 10000 она даёт промах в
    десятки тысяч, а в квадрате — сотни миллионов. Поэтому метрику
    соревнования считаем отдельно и смотрим на неё, а не только на RMSE.
    """
    true = np.power(10.0, np.asarray(log_true, dtype=float))
    predicted = np.power(10.0, np.asarray(log_pred, dtype=float))
    return float(((predicted - true) ** 2).mean())


def bootstrap_ci(
    log_true: np.ndarray,
    log_pred: np.ndarray,
    metric: str = "rmse_log10",
    n_resamples: int = 1000,
    seed: int = 20261106,
) -> tuple[float, float]:
    """95% интервал бутстрапом. Без него разница метрик на 200 молекулах
    неотличима от шума, а на хакатоне это главный источник ложных выводов."""
    true = np.asarray(log_true, dtype=float)
    pred = np.asarray(log_pred, dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        index = rng.integers(0, true.size, true.size)
        try:
            values.append(metrics(true[index], pred[index])[metric])
        except ValueError:
            continue
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def applicability_warning(
    train_fingerprints: np.ndarray,
    query_fingerprints: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Максимальное сходство Танимото с обучающей выборкой.

    Владелец D (стадия 8). Молекула, у которой сходство ниже порога, лежит вне
    области применимости: предсказание для неё выдаётся с пометкой, а не как
    равноценное остальным.
    """
    train = np.asarray(train_fingerprints, dtype=bool)
    query = np.asarray(query_fingerprints, dtype=bool)
    best = np.zeros(query.shape[0], dtype=float)
    for index, row in enumerate(query):
        intersection = np.logical_and(train, row).sum(axis=1)
        union = np.logical_or(train, row).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tanimoto = np.where(union > 0, intersection / union, 0.0)
        best[index] = float(tanimoto.max()) if tanimoto.size else 0.0
    return best
