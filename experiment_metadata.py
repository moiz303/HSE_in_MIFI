#!/usr/bin/env python
"""Опыт: добавляют ли метаданные анализа сигнал поверх структуры молекулы.

    python experiment_metadata.py                   # все варианты, 4 повтора
    python experiment_metadata.py --repeats 6
    python experiment_metadata.py --only структура+линии

**Повод.** Модель до сих пор видела только SMILES. При этом `SI = CC50 / IC50`,
а цитотоксичность измеряется на конкретной клеточной линии и сильно от неё
зависит: одно соединение на Vero и на MT-4 даёт разные CC50. Колонки `strain` и
`Culture` заполнены в тесте для всех 597 строк и не запрещены — запрет касается
только `CC50_uM` и `IC50_uM_VACV`. Это было крупное упущение.

**Что сравниваем.** Пять наборов признаков на одних и тех же scaffold-фолдах,
сравнение парное построчно относительно структуры. Отдельно считается вариант
«только метаданные» — это диагностика: если он сам по себе предсказывает почти
так же, значит модель опирается на протокол, а не на химию, и так и надо
сказать в отчёте.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import splits
from src.features import fingerprints, metadata
from src.models.flexible import FlexibleSI
from sweep import SCREEN_PARAMS, fold_indices, oof_predictions, paired_interval

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"
REPORTS = ROOT / "reports"
FEATURE_CACHE = ROOT / "data" / "interim" / "features_cache.npz"

BASE_SETTINGS = {"transform": "raw", "loss": "squared_error", "clip_si": 20000.0}

# Варианты признаков. Первый — база, с ней сравниваются остальные.
VARIANTS: dict[str, dict] = {
    "структура": {"structure": True},
    "структура+линии": {"structure": True, "cells": True},
    "структура+штамм": {"structure": True, "strain": True},
    "структура+метаданные": {"structure": True, "cells": True, "strain": True,
                             "flags": True},
    "только метаданные": {"structure": False, "cells": True, "strain": True,
                          "flags": True},
}

_STARTED = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    seconds = time.monotonic() - _STARTED
    print(f"  {int(seconds // 60)}:{seconds % 60:04.1f}  {' ' * indent}{message}", flush=True)


def load_frame() -> tuple[pd.DataFrame, np.ndarray]:
    for path in (TRAIN_CSV, VALIDATION_CSV):
        if not path.exists():
            raise SystemExit(f"нет {path}. На машине с данными: python -m src.pipeline export")
    frame = pd.concat(
        [pd.read_csv(TRAIN_CSV), pd.read_csv(VALIDATION_CSV)], ignore_index=True
    )
    in_distribution = frame["source"].fillna("").str.contains("kaggle_train").to_numpy()
    return frame, in_distribution


def structure_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Фингерпринты с кэшем: RDKit считает их десятками секунд."""
    smiles = frame["standard_smiles"].astype(str).tolist()
    cache: dict[str, np.ndarray] = {}
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE) as stored:
            cache = {key: stored[key] for key in stored.files}
    key = f"morgan2048+desc13|{len(smiles)}"
    if key in cache:
        return cache[key]
    matrix = fingerprints.featurize(smiles)
    cache[key] = matrix
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, **cache)
    return matrix


def build_features(frame: pd.DataFrame, structure: np.ndarray, variant: dict) -> np.ndarray:
    blocks: list[np.ndarray] = []
    if variant.get("structure"):
        blocks.append(structure.astype(np.float32))
    if variant.get("cells") or variant.get("strain") or variant.get("flags"):
        blocks.append(metadata.metadata_matrix(
            frame,
            use_cells=bool(variant.get("cells")),
            use_strain=bool(variant.get("strain")),
            use_flags=bool(variant.get("flags")),
        ))
    if not blocks:
        raise ValueError("пустой набор признаков")
    return np.hstack(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only", default=None, help="проверить один вариант против базы")
    args = parser.parse_args(argv)

    frame, in_distribution = load_frame()
    smiles = frame["standard_smiles"].astype(str).tolist()
    si = frame["SI"].to_numpy(float)
    structure = structure_matrix(frame)
    log(f"{len(frame)} строк, в распределении соревнования {int(in_distribution.sum())}")

    found = metadata.describe(frame)
    log(f"распознано линий: {int((found['cell_line'] != metadata.UNKNOWN).sum())}, "
        f"штаммов: {int((found['strain'] != metadata.UNKNOWN).sum())}")

    chosen = dict(VARIANTS)
    if args.only:
        if args.only not in VARIANTS:
            raise SystemExit(f"нет варианта {args.only!r}; есть: {', '.join(VARIANTS)}")
        chosen = {"структура": VARIANTS["структура"], args.only: VARIANTS[args.only]}

    matrices = {name: build_features(frame, structure, variant)
                for name, variant in chosen.items()}
    for name, matrix in matrices.items():
        log(f"{name:<24} {matrix.shape[1]:>5} колонок", indent=2)

    # Накопление построчных квадратов ошибки: сравнение парное, поэтому нужны
    # ошибки на одних и тех же молекулах и одних и тех же фолдах.
    squared: dict[str, list[np.ndarray]] = {name: [] for name in chosen}
    log("")
    for repeat in range(args.repeats):
        fold_list = fold_indices(smiles, args.folds, splits.DEFAULT_SEED + repeat)
        for name, matrix in matrices.items():
            started = time.monotonic()
            prediction = oof_predictions(BASE_SETTINGS, matrix, si, fold_list)
            done = np.isfinite(prediction)
            errors = np.full(len(si), np.nan)
            errors[done] = (prediction[done] - si[done]) ** 2
            squared[name].append(errors)
            inside = done & in_distribution
            log(f"повтор {repeat + 1}  {name:<24} MSE "
                f"{float(((prediction[inside] - si[inside]) ** 2).mean()):.4g}  "
                f"({time.monotonic() - started:.0f} с)", indent=2)

    def mean_errors(name: str) -> np.ndarray:
        return np.nanmean(np.vstack(squared[name]), axis=0)[in_distribution]

    base_errors = mean_errors("структура")
    rows: list[dict] = []
    for name in chosen:
        errors = mean_errors(name)
        row = {
            "вариант": name,
            "колонок": int(matrices[name].shape[1]),
            "mse": float(np.nanmean(errors)),
            "повторов": args.repeats,
        }
        if name != "структура":
            differences = errors - base_errors
            usable = np.isfinite(differences)
            low, high = paired_interval(differences[usable])
            row["разница_с_базой"] = float(differences[usable].mean())
            row["интервал_95"] = [low, high]
            row["достоверно_лучше"] = bool(high < 0)
            row["достоверно_хуже"] = bool(low > 0)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "experiment_metadata.csv", index=False)

    log("")
    log("итог (парное сравнение со структурой, по строкам соревнования):")
    for _, row in table.iterrows():
        verdict = ""
        if row["вариант"] != "структура":
            if row.get("достоверно_лучше"):
                verdict = "  <- достоверно лучше"
            elif row.get("достоверно_хуже"):
                verdict = "  достоверно хуже"
            else:
                verdict = "  в пределах шума"
        log(f"{row['вариант']:<24} {row['колонок']:>5} кол.  MSE {row['mse']:.4g}"
            + (f"  разница {row['разница_с_базой']:+.4g} "
               f"[{row['интервал_95'][0]:+.4g}; {row['интервал_95'][1]:+.4g}]"
               if row["вариант"] != "структура" else "")
            + verdict, indent=2)

    best = table.iloc[0]
    winner = bool(best["вариант"] != "структура" and best.get("достоверно_лучше"))
    (REPORTS / "experiment_metadata.json").write_text(
        json.dumps({
            "конфигурация": FlexibleSI(**BASE_SETTINGS).describe(),
            "повторов": args.repeats,
            "folds": args.folds,
            "варианты": rows,
            "лучший": str(best["вариант"]),
            "вердикт": "принять" if winner else "в пределах шума",
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log("")
    log(f"вердикт: {'принять ' + str(best['вариант']) if winner else 'улучшения нет'}")
    log("отчёт → reports/experiment_metadata.json и .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
