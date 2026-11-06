#!/usr/bin/env python
"""Перебор методов по метрике соревнования. Запускать на маке.

    python sweep.py                 # основная сетка, 3 разбиения
    python sweep.py --repeats 5     # надёжнее, дольше
    python sweep.py --grid full     # ещё и XGBoost, и веса
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

from src import evaluate, splits
from src.features import fingerprints
from src.models.flexible import FlexibleSI

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

CLIP_FROM_TRAIN = 36066.7

_STARTED = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    seconds = time.monotonic() - _STARTED
    print(f"  {int(seconds // 60)}:{seconds % 60:04.1f}  {' ' * indent}{message}", flush=True)


def candidates(grid: str) -> list[dict]:
    """Комбинации для перебора.

    Базовая сетка — то, что по теории должно двигать именно эту метрику:
    масштаб цели, функция потерь для положительных правоскошенных величин,
    порог обрезки. Полная добавляет веса и XGBoost.
    """
    items: list[dict] = []

    # Масштаб цели против функции потерь, на одном пороге обрезки.
    for transform, loss in [
        ("raw", "squared_error"),
        ("raw", "absolute_error"),
        ("raw", "poisson"),
        ("raw", "gamma"),
        ("sqrt", "squared_error"),
        ("sqrt", "absolute_error"),
        ("log", "squared_error"),
        ("log", "absolute_error"),
    ]:
        items.append({"transform": transform, "loss": loss, "clip_si": CLIP_FROM_TRAIN})

    # Высокие квантили: приближение среднего сверху на скошенном распределении.
    for quantile in (0.7, 0.8, 0.9):
        items.append({"transform": "raw", "loss": "quantile", "quantile": quantile,
                      "clip_si": CLIP_FROM_TRAIN})

    # Порог обрезки — такой же параметр, как остальные.
    for clip in (3000.0, 10000.0, 20000.0, CLIP_FROM_TRAIN, None):
        items.append({"transform": "raw", "loss": "squared_error", "clip_si": clip})

    # Лес: классический baseline в QSAR на фингерпринтах, которого у нас не было.
    for learner in ("rf", "extra"):
        for loss in ("squared_error", "absolute_error"):
            items.append({"transform": "raw", "loss": loss, "learner": learner,
                          "clip_si": CLIP_FROM_TRAIN})
    items.append({"transform": "raw", "loss": "squared_error", "learner": "rf",
                  "clip_si": 20000.0})

    if grid == "full":
        for power, loss in itertools.product((0.25, 0.5, 1.0), ("squared_error", "absolute_error")):
            items.append({"transform": "raw", "loss": loss, "weight_power": power,
                          "clip_si": CLIP_FROM_TRAIN})
        for transform, loss in [("raw", "squared_error"), ("raw", "gamma"),
                                ("sqrt", "squared_error")]:
            items.append({"transform": transform, "loss": loss, "learner": "xgb",
                          "clip_si": CLIP_FROM_TRAIN})

    # Убираем дубликаты, сохраняя порядок.
    unique: list[dict] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def load_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Объединённые train и validation плюс маска строк в распределении теста."""
    for path in (TRAIN_CSV, VALIDATION_CSV):
        if not path.exists():
            raise SystemExit(f"нет {path}. На машине с данными: python -m src.pipeline export")
    frame = pd.concat(
        [pd.read_csv(TRAIN_CSV), pd.read_csv(VALIDATION_CSV)], ignore_index=True
    )
    in_distribution = (
        frame["source"].fillna("").str.contains("kaggle_train").to_numpy()
        if "source" in frame.columns
        else np.ones(len(frame), dtype=bool)
    )
    return frame, in_distribution


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    for path in (TRAIN_CSV, VALIDATION_CSV):
        if not path.exists():
            raise SystemExit(f"нет {path}. На машине с данными: python -m src.pipeline export")
    frame = pd.concat(
        [pd.read_csv(TRAIN_CSV), pd.read_csv(VALIDATION_CSV)], ignore_index=True
    )
    log(f"объединено {len(frame)} молекул, считаю признаки…")
    features = fingerprints.featurize(frame["standard_smiles"].astype(str).tolist())
    in_distribution = (
        frame["source"].fillna("").str.contains("kaggle_train").to_numpy()
        if "source" in frame.columns
        else np.ones(len(frame), dtype=bool)
    )
    log(f"матрица {features.shape[0]}×{features.shape[1]}, "
        f"в распределении соревнования: {int(in_distribution.sum())}")
    return frame, features, in_distribution


def evaluate_candidate(
    settings: dict,
    frame: pd.DataFrame,
    features: np.ndarray,
    in_distribution: np.ndarray,
    split_list: list[dict[str, list[int]]],
) -> dict:
    """Один набор настроек на всех разбиениях."""
    si = frame["SI"].to_numpy(float)
    scores: list[float] = []
    overall: list[float] = []

    for split in split_list:
        train_index = np.array(split["train"])
        validation_index = np.array(split["validation"])
        model = FlexibleSI(**settings).fit(features[train_index], si[train_index])
        predicted = model.predict_si(features[validation_index])
        true = si[validation_index]

        overall.append(float(((predicted - true) ** 2).mean()))
        mask = in_distribution[validation_index]
        if mask.sum() >= 10:
            scores.append(float(((predicted[mask] - true[mask]) ** 2).mean()))

    return {
        **settings,
        "описание": FlexibleSI(**settings).describe(),
        "mse_в_распределении": float(np.mean(scores)) if scores else float("nan"),
        "mse_в_распределении_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "mse_вся_валидация": float(np.mean(overall)),
        "разбиений": len(split_list),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Перебор методов по метрике соревнования")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grid", choices=["base", "full"], default="base")
    parser.add_argument("--top", type=int, default=3, help="сколько лучших сохранить")
    args = parser.parse_args(argv)

    frame, features, in_distribution = load_data()
    smiles = frame["standard_smiles"].astype(str).tolist()

    log(f"готовлю {args.repeats} scaffold-разбиений…")
    split_list = []
    for index in range(args.repeats):
        split = splits.scaffold_split(smiles, seed=splits.DEFAULT_SEED + index)
        splits.assert_no_leaks(smiles, split)
        split_list.append(split)

    grid = candidates(args.grid)
    log(f"комбинаций: {len(grid)}, обучений: {len(grid) * args.repeats}\n")

    results: list[dict] = []
    for number, settings in enumerate(grid, start=1):
        started = time.monotonic()
        try:
            result = evaluate_candidate(settings, frame, features, in_distribution, split_list)
        except Exception as error:  # комбинация может быть недопустимой или падать
            log(f"[{number}/{len(grid)}] {settings} — пропущено: {error}")
            continue
        results.append(result)
        log(f"[{number}/{len(grid)}] {result['описание']:<46} "
            f"MSE {result['mse_в_распределении']:.4g} ± {result['mse_в_распределении_std']:.3g}  "
            f"({time.monotonic() - started:.0f} с)")

    if not results:
        raise SystemExit("ни одна комбинация не отработала")

    table = pd.DataFrame(results).sort_values("mse_в_распределении").reset_index(drop=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "sweep.csv", index=False)

    log("")
    log("лучшие по метрике соревнования:")
    for position, row in table.head(10).iterrows():
        log(f"{position + 1:>2}. {row['описание']:<46} "
            f"{row['mse_в_распределении']:.4g} ± {row['mse_в_распределении_std']:.3g}", indent=2)

    # Доучиваем лучшие на train.csv и сохраняем: отправлять нужно модель.
    si = pd.read_csv(TRAIN_CSV)["SI"].to_numpy(float)
    train_features = fingerprints.featurize(
        pd.read_csv(TRAIN_CSV)["standard_smiles"].astype(str).tolist()
    )
    import joblib

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    keys = [k for k in ("transform", "loss", "clip_si", "weight_power", "quantile", "learner")]
    saved: list[dict] = []
    log("")
    for position in range(min(args.top, len(table))):
        row = table.iloc[position]
        settings = {k: row[k] for k in keys if k in row and pd.notna(row[k])}
        model = FlexibleSI(**settings).fit(train_features, si)
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
    log("")
    log(f"этап 2: признаки при фиксированной модели ({table.iloc[0]['описание']})")

    feature_rows: list[dict] = []
    for name, preset in fingerprints.PRESETS.items():
        started = time.monotonic()
        try:
            matrix = fingerprints.featurize(smiles, **preset)
            result = evaluate_candidate(
                best_settings, frame, matrix, in_distribution, split_list
            )
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
