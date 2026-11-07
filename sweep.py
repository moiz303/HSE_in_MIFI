#!/usr/bin/env python
"""Перебор методов по метрике соревнования. Запускать на маке.

    python sweep.py                 # основная сетка, 3 разбиения
    python sweep.py --repeats 5     # надёжнее, дольше
    python sweep.py --grid full     # ещё и CatBoost, и веса
    python sweep.py --top 5         # сколько лучших доучить и сохранить

Что меряем. Ранжируем по MSE на строках из train.csv соревнования: наша
валидация содержит хвост ChEMBL с SI до 2.2e5, которого в тесте нет, и на ней
метрика была 7e7 при реальных 1.2e7 на лидерборде. Сопоставима с лидербордом
только подвыборка в распределении соревнования.

Каждая комбинация оценивается на одних и тех же scaffold-разбиениях, так что
сравнение честное. Результат — таблица в reports/sweep.csv, лучшие модели
доучиваются на train.csv и сохраняются в artifacts/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import splits
from src.experiments.harness import (
    BASE_SETTINGS,
    SCREEN_PARAMS,
    fold_indices,
    load_pool,
    model_variant,
    oof_predictions as oof_generic,
    paired_interval,
)
from src.features import fingerprints
from src.models.flexible import FlexibleSI

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

CLIP_FROM_TRAIN = 36066.7
FEATURE_CACHE = Path("data/interim/features_cache.npz")

_STARTED = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    seconds = time.monotonic() - _STARTED
    print(f"  {int(seconds // 60)}:{seconds % 60:04.1f}  {' ' * indent}{message}", flush=True)


def candidates(grid: str) -> list[dict]:
    """Комбинации для перебора.

    `base` — живые гипотезы, быстрый прогон.

    `full` — включая направления, ранее отвергнутые. Это не повтор зря: они
    отвергались старым измерителем, где в распределении соревнования было 64
    строки и std доходил до трети среднего. На такой точности «в пределах
    шума» значило «не знаем», а не «хуже». Кросс-валидация на 310 строках
    сужает погрешность примерно в 2.2 раза, и часть отказов может
    перевернуться. Исключение — масштаб цели: там разрыв был втрое больше
    погрешности (raw 2.54e7 против log 3.69e7), и перепроверять нечего, но
    опорная точка в сетке остаётся.
    """
    items: list[dict] = []

    # Опора: лучшее на сейчас.
    items.append({"transform": "raw", "loss": "squared_error", "clip_si": 20000.0})

    # Лес и ExtraTrees: отвергнуты с разрывом 0.1e7 при std 0.5-0.7e7.
    for learner in ("rf", "extra"):
        items.append({"transform": "raw", "loss": "squared_error", "learner": learner,
                      "clip_si": 20000.0})

    # Вес больших SI: вклад строки в метрику растёт как SI^2.
    for power in (0.25, 0.5):
        items.append({"transform": "raw", "loss": "squared_error",
                      "weight_power": power, "clip_si": 20000.0})

    if grid == "base":
        return items

    # Потери: отвергнуты на шумном измерителе.
    for loss in ("absolute_error", "poisson", "gamma"):
        items.append({"transform": "raw", "loss": loss, "clip_si": 20000.0})
    for quantile in (0.7, 0.8, 0.9):
        items.append({"transform": "raw", "loss": "quantile", "quantile": quantile,
                      "clip_si": 20000.0})

    # Порог обрезки: сетка шире прежней, оптимум мог оказаться между точками.
    for clip in (5000.0, 10000.0, 15000.0, 30000.0, CLIP_FROM_TRAIN, None):
        items.append({"transform": "raw", "loss": "squared_error", "clip_si": clip})

    # Масштаб цели: опорные точки, разрыв был велик и вряд ли перевернётся.
    for transform in ("sqrt", "log"):
        items.append({"transform": transform, "loss": "squared_error",
                      "clip_si": 20000.0})

    # Лес с другими критериями и обрезкой, CatBoost, сильный вес.
    items.append({"transform": "raw", "loss": "absolute_error", "learner": "rf",
                  "clip_si": 20000.0})
    items.append({"transform": "raw", "loss": "poisson", "learner": "rf",
                  "clip_si": 20000.0})
    items.append({"transform": "raw", "loss": "squared_error", "learner": "catboost",
                  "clip_si": 20000.0})
    items.append({"transform": "raw", "loss": "squared_error", "weight_power": 1.0,
                  "clip_si": 20000.0})

    unique: list[dict] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def featurize_cached(smiles: list[str], name: str, preset: dict) -> np.ndarray:
    """Признаки с кэшем на диске: RDKit пересчитывает их десятками секунд."""
    cache: dict[str, np.ndarray] = {}
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE) as stored:
            cache = {key: stored[key] for key in stored.files}
    key = f"{name}|{len(smiles)}"
    if key in cache:
        return cache[key]
    matrix = fingerprints.featurize(smiles, **preset)
    cache[key] = matrix
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, **cache)
    return matrix


def oof_predictions(
    settings: dict,
    features: np.ndarray,
    si: np.ndarray,
    fold_list: list[np.ndarray],
    screen: bool = True,
) -> np.ndarray:
    """Out-of-fold прогноз для набора настроек — обёртка над каркасом."""
    return oof_generic(
        model_variant(settings, features, si, screen=screen), len(si), fold_list
    )


def race(
    grid: list[dict],
    frame: pd.DataFrame,
    features: np.ndarray,
    in_distribution: np.ndarray,
    smiles: list[str],
    folds: int,
    min_repeats: int,
    max_repeats: int,
    deadline: float,
    shrink: float = 0.1,
) -> tuple[list[dict], dict]:
    """Отбор кандидатов по метрике, а не по времени.

    Схема: все выжившие кандидаты считаются на одном и том же наборе фолдов,
    затем каждый сравнивается с действующим лидером построчно. Кандидат
    выбывает, когда интервал парной разницы целиком выше нуля, то есть он
    достоверно хуже. Лидер меняется, когда интервал целиком ниже нуля.

    Повторы добавляются, пока есть неопределённые кандидаты. Прогон кончается
    по одной из трёх причин, и все они про метрику, а не про время:

    * остался один кандидат;
    * все остальные достоверно хуже лидера;
    * интервалы неопределённых перестали сужаться — очередной повтор уменьшил
      ширину меньше чем на `shrink` долю. Значит разницы попросту нет, и
      добивать до верхней границы повторов бессмысленно.
    """
    si = frame["SI"].to_numpy(float)
    squared: dict[int, list[np.ndarray]] = {index: [] for index in range(len(grid))}
    alive = list(range(len(grid)))
    leader = 0
    history: list[dict] = []
    stopped = "исчерпаны повторы"
    previous_width: float | None = None

    for repeat in range(max_repeats):
        if time.monotonic() > deadline:
            stopped = "бюджет по времени"
            break

        fold_list = fold_indices(smiles, folds, splits.DEFAULT_SEED + repeat)
        log(f"повтор {repeat + 1}: кандидатов в гонке {len(alive)}")

        for index in alive:
            prediction = oof_predictions(grid[index], features, si, fold_list)
            done = np.isfinite(prediction)
            errors = np.full(len(si), np.nan)
            errors[done] = (prediction[done] - si[done]) ** 2
            squared[index].append(errors)

        def mean_errors(index: int) -> np.ndarray:
            return np.nanmean(np.vstack(squared[index]), axis=0)

        scores = {index: float(np.nanmean(mean_errors(index)[in_distribution]))
                  for index in alive}
        # min по словарю с NaN выбирает NaN: в Python сравнения с ним всегда
        # ложны, и лидером стал бы кандидат, у которого вообще нет прогнозов.
        # Дальше все сравнения шли бы против мусора, молча.
        finite = {index: value for index, value in scores.items() if np.isfinite(value)}
        if not finite:
            log("ни у одного кандидата нет годных прогнозов — гонка прервана", indent=2)
            stopped = "нет годных прогнозов"
            break
        for index in [i for i in alive if i not in finite]:
            log(f"выбывает: {FlexibleSI(**grid[index]).describe():<46} нет годных прогнозов",
                indent=2)
        alive = list(finite)
        leader = min(finite, key=finite.get)
        log(f"лидер: {grid[leader]['описание'] if 'описание' in grid[leader] else FlexibleSI(**grid[leader]).describe()}"
            f"  MSE {scores[leader]:.4g}", indent=2)

        if repeat + 1 < min_repeats:
            continue

        leader_errors = mean_errors(leader)[in_distribution]
        survivors: list[int] = [leader]
        undecided = 0
        width = 0.0
        for index in alive:
            if index == leader:
                continue
            differences = mean_errors(index)[in_distribution] - leader_errors
            usable = np.isfinite(differences)
            low, high = paired_interval(differences[usable])
            if low > 0:
                log(f"выбывает: {FlexibleSI(**grid[index]).describe():<46} "
                    f"хуже на {differences[usable].mean():.4g} [{low:.4g}; {high:.4g}]",
                    indent=2)
                continue
            survivors.append(index)
            if high >= 0:
                undecided += 1
                width = max(width, high - low)

        alive = survivors
        history.append({"повтор": repeat + 1, "в_гонке": len(alive),
                        "лидер": FlexibleSI(**grid[leader]).describe(),
                        "mse_лидера": scores[leader], "неопределённых": undecided,
                        "ширина_интервала": width})

        if len(alive) == 1:
            stopped = "остался один кандидат"
            break
        if undecided == 0:
            stopped = "все остальные достоверно хуже лидера"
            break

        # Сходимость: если ширина интервала почти не уменьшилась, разницы нет,
        # и новые повторы только жгут время.
        if previous_width is not None and width > previous_width * (1.0 - shrink):
            stopped = (
                f"интервалы перестали сужаться ({previous_width:.4g} -> {width:.4g}), "
                f"разницы между выжившими нет"
            )
            break
        previous_width = width

    log("")
    log(f"гонка остановлена: {stopped}")

    results: list[dict] = []
    for index in range(len(grid)):
        if not squared[index]:
            continue
        errors = np.nanmean(np.vstack(squared[index]), axis=0)
        inside = errors[in_distribution]
        row = {
            **grid[index],
            "описание": FlexibleSI(**grid[index]).describe(),
            "mse_в_распределении": float(np.nanmean(inside)),
            "mse_вся_валидация": float(np.nanmean(errors)),
            "повторов": len(squared[index]),
        }
        if index != leader:
            differences = inside - np.nanmean(np.vstack(squared[leader]), axis=0)[in_distribution]
            usable = np.isfinite(differences)
            low, high = paired_interval(differences[usable])
            row["разница_с_лидером"] = float(differences[usable].mean())
            row["разница_интервал_95"] = [low, high]
            row["достоверно_хуже"] = bool(low > 0)
        results.append(row)

    return results, {"остановка": stopped, "история": history,
                     "лидер": FlexibleSI(**grid[leader]).describe()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Перебор методов по метрике соревнования")
    parser.add_argument("--min-repeats", type=int, default=2,
                        help="сколько повторов пройти до первого отсева")
    parser.add_argument("--max-repeats", type=int, default=8,
                        help="верхняя граница; гонка обычно кончается раньше")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid", choices=["base", "full"], default="base")
    parser.add_argument("--top", type=int, default=3, help="сколько лучших сохранить")
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="жёсткий бюджет в минутах; 0 — без ограничения")
    parser.add_argument("--features", action="store_true",
                        help="второй этап по наборам признаков (закрытое направление)")
    args = parser.parse_args(argv)

    frame, in_distribution = load_pool()
    features = featurize_cached(
        frame["standard_smiles"].astype(str).tolist(), "morgan2048+desc13", {}
    )
    smiles = frame["standard_smiles"].astype(str).tolist()

    log(f"оценка out-of-fold на {int(in_distribution.sum())} строках соревнования; "
        f"сравнение парное, построчно")

    grid = candidates(args.grid)
    log(f"комбинаций: {len(grid)}, фолдов {args.folds}, "
        f"повторов от {args.min_repeats} до {args.max_repeats}")
    log("остановка по метрике: кандидат выбывает, когда интервал парной разницы")
    log("целиком выше нуля; гонка кончается, когда неопределённых не осталось", indent=2)
    log("")

    deadline = (
        time.monotonic() + args.max_minutes * 60 if args.max_minutes > 0 else float("inf")
    )

    results, verdict = race(
        grid, frame, features, in_distribution, smiles,
        args.folds, args.min_repeats, args.max_repeats, deadline,
    )
    if not results:
        raise SystemExit("ни одна комбинация не отработала")

    table = pd.DataFrame(results).sort_values("mse_в_распределении").reset_index(drop=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "sweep.csv", index=False)

    (REPORTS / "sweep_race.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("")
    log("лучшие по метрике соревнования:")
    for position, row in table.head(10).iterrows():
        log(f"{position + 1:>2}. {row['описание']:<46} "
            f"{row['mse_в_распределении']:.4g} ± {row['mse_в_распределении_std']:.3g}", indent=2)

    # Доучиваем лучшие на train.csv и сохраняем: отправлять нужно модель.
    si = pd.read_csv(TRAIN_CSV)["SI"].to_numpy(float)
    train_features = featurize_cached(
        pd.read_csv(TRAIN_CSV)["standard_smiles"].astype(str).tolist(),
        "morgan2048+desc13_train", {}
    )
    import joblib

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    keys = [k for k in ("transform", "loss", "clip_si", "weight_power", "quantile", "learner")]
    saved: list[dict] = []
    log("")
    for position in range(min(args.top, len(table))):
        row = table.iloc[position]
        settings = {k: row[k] for k in keys if k in row and pd.notna(row[k])}
        model = FlexibleSI(**settings).fit(train_features, si)  # полные параметры
        name = f"model_sweep{position + 1}.joblib"
        joblib.dump({"model": model, "settings": settings, "описание": row["описание"]},
                    ARTIFACTS / name)
        if position == 0:
            joblib.dump({"model": model, "settings": settings, "strategy": "sweep"},
                        ARTIFACTS / "model.joblib")
        saved.append({"файл": name, "описание": row["описание"],
                      "mse_в_распределении": row["mse_в_распределении"]})
        log(f"сохранено: {name} — {row['описание']}")

    (REPORTS / "sweep_best.json").write_text(
        json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # ------------------------------------------------------------------------
    # Этап 2: лучшие настройки модели фиксируем и перебираем наборы признаков.
    # Полный перекрёст это 9 x 17 обучений; жадный проход даёт 9 и ловит
    # основное: 41% ошибки метрики приходится на класс "неактивно", а это
    # задача классификации, где состав признаков обычно важнее вида потерь.
    # ------------------------------------------------------------------------
    best_settings = {k: table.iloc[0][k] for k in keys
                     if k in table.columns and pd.notna(table.iloc[0][k])}

    feature_rows: list[dict] = []
    if not args.features:
        log("")
        log("этап признаков пропущен: направление закрыто (--features включает)")
    else:
        log("")
        log(f"этап 2: признаки при фиксированной модели ({table.iloc[0]['описание']})")

    for name, preset in (fingerprints.PRESETS.items() if args.features else []):
        if time.monotonic() > deadline:
            log("бюджет исчерпан, остальные наборы признаков не проверены", indent=2)
            break
        started = time.monotonic()
        try:
            matrix = featurize_cached(smiles, name, preset)
            fold_list = fold_indices(smiles, args.folds, splits.DEFAULT_SEED)
            si_all = frame["SI"].to_numpy(float)
            prediction = oof_predictions(best_settings, matrix, si_all, fold_list)
            done = np.isfinite(prediction) & in_distribution
            result = {
                "mse_в_распределении": float(((prediction[done] - si_all[done]) ** 2).mean()),
                "mse_в_распределении_std": 0.0,
            }
        except Exception as error:
            log(f"{name} — пропущено: {error}", indent=2)
            continue
        result["признаки"] = name
        result["колонок"] = int(matrix.shape[1])
        feature_rows.append(result)
        log(f"{name:<26} {result['колонок']:>5} колонок  "
            f"MSE {result['mse_в_распределении']:.4g} ± "
            f"{result['mse_в_распределении_std']:.3g}  ({time.monotonic() - started:.0f} с)",
            indent=2)

    if feature_rows:
        feature_table = pd.DataFrame(feature_rows).sort_values("mse_в_распределении")
        feature_table.to_csv(REPORTS / "sweep_features.csv", index=False)
        best_features = feature_table.iloc[0]
        log("")
        log(f"лучший набор признаков: {best_features['признаки']} "
            f"({best_features['mse_в_распределении']:.4g} против "
            f"{table.iloc[0]['mse_в_распределении']:.4g} у базового)")
        (REPORTS / "sweep_best.json").write_text(
            json.dumps({
                "модель": saved,
                "признаки": {
                    "лучший": str(best_features["признаки"]),
                    "mse_в_распределении": float(best_features["mse_в_распределении"]),
                    "пресет": fingerprints.PRESETS[str(best_features["признаки"])],
                },
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    log("")
    log("таблицы → reports/sweep.csv и reports/sweep_features.csv")
    log("лучшая модель → artifacts/model.joblib")
    log("закоммить artifacts/ и reports/ — отправку соберёт тот, у кого данные соревнования")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
