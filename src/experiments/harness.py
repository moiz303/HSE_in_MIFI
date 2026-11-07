"""Общий каркас для опытов: данные, фолды, out-of-fold прогноз, сравнение.

Зачем модуль. Проверка гипотезы — это всегда одно и то же: взять выборку,
нарезать scaffold-фолды, получить out-of-fold прогноз для каждого варианта,
сравнить варианты парно и записать отчёт. Шесть скриптов опытов повторяли эту
механику каждый у себя, и новый человек, чтобы проверить свою идею, должен был
копировать её из любого из них.

Теперь новая гипотеза — это один короткий файл: описать варианты и вызвать
`compare`. Всё остальное каркас берёт на себя, одинаково для всех, поэтому
результаты разных опытов сравнимы между собой.

**Что здесь важно знать, прежде чем писать свой опыт.**

*Метрика считается на строках в распределении соревнования.* Наша выборка
смешанная: 310 строк из `train.csv` организаторов и 874 из ChEMBL с хвостом SI
до 2.2e5 при максимуме соревнования 3.6e4. На полной выборке метрика давала 7e7
при реальных 1.2e7 на лидерборде, то есть не предсказывала результат.

*Сравнение парное, построчно.* Все варианты предсказывают одни и те же
молекулы на одних и тех же фолдах, поэтому сравнивать надо разницу по строке, а
не две независимые средние. Разница убирает дисперсию от того, какие именно
выбросы попали в отложенную часть, — а она у нас и съедала измерения:
абсолютная MSE гуляла на треть, хотя варианты отличались на проценты.

*База для сравнения — то, что уже работает.* Выигрыш над искусственно
ослабленным вариантом ничего не значит; на этом я один раз уже ошибся, объявив
улучшение 28% над регрессией, которую сам же и ослабил.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src import splits
from src.features import fingerprints

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"
REPORTS = ROOT / "reports"
FEATURE_CACHE = ROOT / "data" / "interim" / "features_cache.npz"

# Конфигурация, победившая в переборе. Опоры для сравнения берём отсюда, чтобы
# разные опыты мерили одно и то же.
BASE_SETTINGS = {"transform": "raw", "loss": "squared_error", "clip_si": 20000.0}

# Урезанные параметры для скрининга: нужен порядок величин, не финальное
# качество. Победителя доучивают полными.
SCREEN_PARAMS = {"max_iter": 150, "n_estimators": 200}

LABEL_VALUE = 10000.0

_STARTED = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    """Строка лога с отметкой времени от старта, сразу в поток."""
    seconds = time.monotonic() - _STARTED
    print(f"  {int(seconds // 60)}:{seconds % 60:04.1f}  {' ' * indent}{message}", flush=True)


# ----------------------------------------------------------------------- данные
def load_pool() -> tuple[pd.DataFrame, np.ndarray]:
    """train и validation одной таблицей плюс маска строк соревнования.

    Holdout сюда не входит и не должен: он трогается один раз, на выпуске.
    """
    for path in (TRAIN_CSV, VALIDATION_CSV):
        if not path.exists():
            raise SystemExit(
                f"нет {path}. На машине с данными: python -m src.pipeline export"
            )
    frame = pd.concat(
        [pd.read_csv(TRAIN_CSV), pd.read_csv(VALIDATION_CSV)], ignore_index=True
    )
    in_distribution = (
        frame["source"].fillna("").str.contains("kaggle_train").to_numpy()
        if "source" in frame.columns
        else np.ones(len(frame), dtype=bool)
    )
    return frame, in_distribution


def smiles_of(frame: pd.DataFrame) -> list[str]:
    column = "standard_smiles" if "standard_smiles" in frame.columns else "smiles"
    return frame[column].astype(str).tolist()


def features_of(frame: pd.DataFrame, name: str = "morgan2048+desc13",
                preset: dict | None = None) -> np.ndarray:
    """Признаки с кэшем на диске: RDKit считает их десятками секунд."""
    smiles = smiles_of(frame)
    cache: dict[str, np.ndarray] = {}
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE) as stored:
            cache = {key: stored[key] for key in stored.files}
    key = f"{name}|{len(smiles)}"
    if key in cache:
        return cache[key]
    matrix = fingerprints.featurize(smiles, **(preset or {}))
    cache[key] = matrix
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, **cache)
    return matrix


# ------------------------------------------------------------------------ фолды
def fold_indices(smiles: list[str], folds: int, seed: int) -> list[np.ndarray]:
    """Scaffold-фолды по всем строкам сразу.

    Разбиение строится по всей выборке, поэтому строка ChEMBL с каркасом из
    отложенного fold тоже оказывается отложенной — утечки каркасов нет.
    """
    fractions = {f"f{number}": 1.0 / folds for number in range(folds)}
    split = splits.scaffold_split(smiles, fractions, seed=seed)
    splits.assert_no_leaks(smiles, split)
    return [np.array(positions, dtype=int) for positions in split.values()]


def fold_sets(smiles: list[str], folds: int, repeats: int,
              seed: int | None = None) -> list[list[np.ndarray]]:
    base = splits.DEFAULT_SEED if seed is None else seed
    return [fold_indices(smiles, folds, base + repeat) for repeat in range(repeats)]


# ------------------------------------------------------------- прогноз и метрика
def oof_predictions(
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_rows: int,
    fold_list: list[np.ndarray],
) -> np.ndarray:
    """Out-of-fold прогноз. fit_predict получает индексы обучения и отложенные.

    Такая сигнатура нужна, чтобы вариант мог делать что угодно: обучать одну
    модель, две, классификатор со смесью или вообще считать константу.
    """
    prediction = np.full(n_rows, np.nan)
    for held in fold_list:
        # Фолд может оказаться пустым: если почти все молекулы делят каркас,
        # групповое разбиение сложит их в одну часть.
        if held.size == 0:
            continue
        rest = np.setdiff1d(np.arange(n_rows), held)
        if rest.size < 30:
            continue
        prediction[held] = fit_predict(rest, held)
    return prediction


def model_variant(settings: dict, features: np.ndarray, si: np.ndarray,
                  screen: bool = True) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Вариант «обучить FlexibleSI с такими настройками» для oof_predictions."""
    from src.models.flexible import FlexibleSI

    active = dict(settings)
    if screen:
        params = dict(active.get("params") or {})
        params.update(SCREEN_PARAMS)
        active["params"] = params

    def fit_predict(rest: np.ndarray, held: np.ndarray) -> np.ndarray:
        return FlexibleSI(**active).fit(features[rest], si[rest]).predict_si(features[held])

    return fit_predict


def paired_interval(differences: np.ndarray, draws: int = 10000,
                    seed: int = 7) -> tuple[float, float]:
    """95% интервал для среднего парной разницы — ресэмплинг по строкам."""
    usable = differences[np.isfinite(differences)]
    if usable.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(usable), size=(draws, len(usable)))
    means = usable[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mse_of(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    selected = mask & np.isfinite(prediction)
    if selected.sum() == 0:
        return float("nan")
    return float(((prediction[selected] - truth[selected]) ** 2).mean())


# --------------------------------------------------------------------- сравнение
def compare(
    variants: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    si: np.ndarray,
    in_distribution: np.ndarray,
    fold_sets_list: list[list[np.ndarray]],
    base: str | None = None,
) -> pd.DataFrame:
    """Парное сравнение вариантов на одних и тех же фолдах.

    Возвращает таблицу: MSE каждого варианта, разница с базой, интервал и
    вердикт. База по умолчанию — первый вариант в словаре.
    """
    base = base or next(iter(variants))
    if base not in variants:
        raise ValueError(f"базы {base!r} нет среди вариантов")

    squared: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    for repeat, fold_list in enumerate(fold_sets_list, start=1):
        for name, fit_predict in variants.items():
            started = time.monotonic()
            prediction = oof_predictions(fit_predict, len(si), fold_list)
            errors = np.full(len(si), np.nan)
            done = np.isfinite(prediction)
            errors[done] = (prediction[done] - si[done]) ** 2
            squared[name].append(errors)
            log(f"повтор {repeat}  {name:<28} MSE "
                f"{mse_of(prediction, si, in_distribution):.4g}  "
                f"({time.monotonic() - started:.0f} с)", indent=2)

    def mean_errors(name: str) -> np.ndarray:
        return np.nanmean(np.vstack(squared[name]), axis=0)[in_distribution]

    base_errors = mean_errors(base)
    rows: list[dict] = []
    for name in variants:
        errors = mean_errors(name)
        row: dict[str, object] = {
            "вариант": name,
            "mse": float(np.nanmean(errors)),
            "повторов": len(fold_sets_list),
            "это_база": name == base,
            "разница_с_базой": 0.0,
            "интервал_низ": 0.0,
            "интервал_верх": 0.0,
            "вердикт": "база",
        }
        if name != base:
            differences = errors - base_errors
            low, high = paired_interval(differences)
            mean_difference = float(np.nanmean(differences))
            row.update({
                "разница_с_базой": mean_difference,
                "интервал_низ": low,
                "интервал_верх": high,
                "вердикт": ("лучше" if high < 0 else
                            "хуже" if low > 0 else "в пределах шума"),
            })
        rows.append(row)

    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def print_table(table: pd.DataFrame) -> None:
    log("")
    log("итог (парное сравнение с базой, строки соревнования):")
    for _, row in table.iterrows():
        tail = ""
        if not row["это_база"]:
            tail = (f"  разница {row['разница_с_базой']:+.4g} "
                    f"[{row['интервал_низ']:+.4g}; {row['интервал_верх']:+.4g}]"
                    f"  {row['вердикт']}")
        else:
            tail = "  база"
        log(f"{row['вариант']:<28} MSE {row['mse']:.4g}{tail}", indent=2)


def save(name: str, table: pd.DataFrame, extra: dict | None = None) -> dict:
    """Записать отчёт: csv с таблицей и json со сводкой и вердиктом."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / f"{name}.csv", index=False)

    winners = table[(~table["это_база"]) & (table["вердикт"] == "лучше")]
    payload = {
        "опыт": name,
        "база": str(table.loc[table["это_база"], "вариант"].iloc[0]),
        "лучший": str(table.iloc[0]["вариант"]),
        "вердикт": "принять" if not winners.empty else "улучшения нет",
        "варианты": table.to_dict(orient="records"),
        **(extra or {}),
    }
    (REPORTS / f"{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("")
    log(f"вердикт: {payload['вердикт']}"
        + (f" ({payload['лучший']})" if not winners.empty else ""))
    log(f"отчёт → reports/{name}.json и .csv")
    return payload
