#!/usr/bin/env python
"""Из чего состоит метрика соревнования и добавляет ли модель хоть что-то.

Опыт с калибровкой (experiment_calibration.py) дал неожиданный побочный
результат: оптимальная по MSE линейная поправка имеет b около 0.26 и сдвиг
a около 2000 — то есть МНК просит заменить три четверти прогноза константой,
подозрительно близкой к среднему SI по обучающим строкам. Отсюда два вопроса,
на которые эта проверка отвечает прямо.

**1. Модель против константного пола.** Сравниваем на одних и тех же
out-of-fold прогнозах: модель против «всегда предсказывай среднее по обучающей
части». Разница меряется парно по строкам, с bootstrap-интервалом: средних
значений на 260 строках мало, нужен разброс.

**2. Из чего складывается MSE.** В распределении соревнования 64 строки из 260
(24.6%) имеют SI ровно 10000 — это не измерение, а метка «неактивно»
(CC50/IC50 упёрлись в границу диапазона). Строка с меткой даёт вклад в MSE
порядка (10000 - прогноз)², то есть десятки миллионов на строку, тогда как
медианный SI в этой выборке 17.4. Раскладываем метрику на вклад строк-меток и
всех остальных и считаем «оракула»: MSE при идеальном знании метки. Это
верхняя граница того, что вообще можно выиграть, разбираясь с метками, и она
говорит, куда имеет смысл тратить время.

Обе части считаются на scaffold-сгруппированных folds: каждая строка
предсказывается моделью, не видевшей её каркас. Holdout не трогается.

    python experiment_floor.py                  # 4 повтора по 5 folds
    python experiment_floor.py --repeats 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import splits
from src.features import fingerprints
from src.models.flexible import FlexibleSI

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"
REPORTS = ROOT / "reports"
FEATURE_CACHE = ROOT / "data" / "interim" / "features_cache.npz"

BASE_SETTINGS = {"transform": "raw", "loss": "squared_error", "clip_si": 20000.0}
SENTINEL_SI = 10000.0

_STARTED = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    seconds = time.monotonic() - _STARTED
    print(f"  {int(seconds // 60)}:{seconds % 60:04.1f}  {' ' * indent}{message}", flush=True)


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    for path in (TRAIN_CSV, VALIDATION_CSV):
        if not path.exists():
            raise SystemExit(f"нет {path}. На машине с данными: python -m src.pipeline export")
    frame = pd.concat(
        [pd.read_csv(TRAIN_CSV), pd.read_csv(VALIDATION_CSV)], ignore_index=True
    )
    smiles = frame["standard_smiles"].astype(str).tolist()
    cache: dict[str, np.ndarray] = {}
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE) as stored:
            cache = {key: stored[key] for key in stored.files}
    key = f"morgan2048+desc13|{len(smiles)}"
    if key in cache:
        features = cache[key]
    else:
        features = fingerprints.featurize(smiles)
        cache[key] = features
        FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(FEATURE_CACHE, **cache)
    in_distribution = (
        frame["source"].fillna("").str.contains("kaggle_train").to_numpy()
        if "source" in frame.columns
        else np.ones(len(frame), dtype=bool)
    )
    return frame, features, in_distribution


def out_of_fold(
    features: np.ndarray,
    si: np.ndarray,
    smiles: list[str],
    in_distribution: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """OOF прогнозы модели и константы на всех строках.

    Константа тоже считается out-of-fold — по обучающей части каждого fold, и
    только по строкам в распределении соревнования: средний SI по всем строкам
    тянут вниз малые значения ChEMBL, и как пол он нечестно слаб.
    """
    fractions = {f"f{number}": 1.0 / folds for number in range(folds)}
    split = splits.scaffold_split(smiles, fractions, seed=seed)
    splits.assert_no_leaks(smiles, split)

    model_prediction = np.full(len(si), np.nan)
    constant_prediction = np.full(len(si), np.nan)
    for positions in split.values():
        held = np.array(positions, dtype=int)
        if held.size == 0:
            continue
        rest = np.setdiff1d(np.arange(len(si)), held)
        model = FlexibleSI(**BASE_SETTINGS).fit(features[rest], si[rest])
        model_prediction[held] = model.predict_si(features[held])
        reference = si[rest][in_distribution[rest]]
        constant_prediction[held] = float(
            np.mean(np.minimum(reference, BASE_SETTINGS["clip_si"]))
        )
    return model_prediction, constant_prediction


def bootstrap_interval(
    differences: np.ndarray, draws: int = 10000, seed: int = 7
) -> tuple[float, float]:
    """95% интервал для среднего парной разницы — ресэмплинг по строкам."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    means = differences[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Пол метрики и её разложение")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)

    frame, features, in_distribution = load_data()
    smiles = frame["standard_smiles"].astype(str).tolist()
    si = frame["SI"].to_numpy(float)
    log(f"{len(frame)} молекул, в распределении соревнования {int(in_distribution.sum())}")

    per_repeat: list[dict] = []
    model_errors: list[np.ndarray] = []
    constant_errors: list[np.ndarray] = []

    for number in range(args.repeats):
        seed = splits.DEFAULT_SEED + number
        started = time.monotonic()
        model_prediction, constant_prediction = out_of_fold(
            features, si, smiles, in_distribution, args.folds, seed
        )
        scored = in_distribution & np.isfinite(model_prediction)
        true = si[scored]
        model_error = (model_prediction[scored] - true) ** 2
        constant_error = (constant_prediction[scored] - true) ** 2
        model_errors.append(model_error)
        constant_errors.append(constant_error)

        sentinel = true == SENTINEL_SI
        per_repeat.append({
            "повтор": number + 1,
            "seed": seed,
            "строк": int(scored.sum()),
            "mse_модель": float(model_error.mean()),
            "mse_константа": float(constant_error.mean()),
            "константа": float(constant_prediction[scored][0]),
            "доля_ошибки_на_метках": float(model_error[sentinel].sum() / model_error.sum()),
            "mse_оракул_метки": float(
                np.r_[np.zeros(int(sentinel.sum())), model_error[~sentinel]].mean()
            ),
        })
        row = per_repeat[-1]
        log(f"повтор {number + 1}/{args.repeats} (seed {seed}, {row['строк']} строк, "
            f"{time.monotonic() - started:.0f} с)")
        log(f"модель {row['mse_модель']:.4g}   константа {row['константа']:.0f} -> "
            f"{row['mse_константа']:.4g}   оракул по меткам {row['mse_оракул_метки']:.4g}",
            indent=2)
        log(f"на строках-метках (SI=10000, {int(sentinel.sum())} из {row['строк']}) "
            f"лежит {row['доля_ошибки_на_метках']:.0%} всей ошибки", indent=2)

    table = pd.DataFrame(per_repeat)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "experiment_floor.csv", index=False)

    # Парная разница по строкам, объединяя повторы: интервал по строкам, а не
    # по четырём средним, иначе интервал строится по четырём числам.
    model_all = np.concatenate(model_errors)
    constant_all = np.concatenate(constant_errors)
    difference = constant_all - model_all          # > 0 значит модель лучше пола
    low, high = bootstrap_interval(difference)

    log("")
    log("модель против константного пола (парно по строкам, все повторы):")
    log(f"MSE модели     {model_all.mean():.4g}", indent=2)
    log(f"MSE константы  {constant_all.mean():.4g}", indent=2)
    log(f"выигрыш модели {difference.mean():.4g}  "
        f"95% интервал [{low:.4g}, {high:.4g}]", indent=2)
    beats_floor = low > 0
    log(f"вердикт: модель {'бьёт' if beats_floor else 'НЕ бьёт'} константу "
        f"({'интервал не включает ноль' if beats_floor else 'интервал включает ноль'})",
        indent=2)

    log("")
    log("разложение метрики:")
    log(f"доля ошибки на строках-метках SI=10000: "
        f"{table['доля_ошибки_на_метках'].mean():.0%}", indent=2)
    log(f"MSE при идеальном знании метки: {table['mse_оракул_метки'].mean():.4g} "
        f"против {table['mse_модель'].mean():.4g} сейчас — "
        f"потолок выигрыша {100 * (1 - table['mse_оракул_метки'].mean() / table['mse_модель'].mean()):.0f}%",
        indent=2)

    (REPORTS / "experiment_floor.json").write_text(
        json.dumps({
            "конфигурация": FlexibleSI(**BASE_SETTINGS).describe(),
            "повторов": int(args.repeats),
            "folds": int(args.folds),
            "mse_модель": float(model_all.mean()),
            "mse_константа": float(constant_all.mean()),
            "выигрыш_модели": float(difference.mean()),
            "выигрыш_интервал_95": [low, high],
            "модель_бьёт_пол": bool(beats_floor),
            "доля_ошибки_на_метках": float(table["доля_ошибки_на_метках"].mean()),
            "mse_оракул_метки": float(table["mse_оракул_метки"].mean()),
            "по_повторам": per_repeat,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log("")
    log("таблица → reports/experiment_floor.csv, итог → reports/experiment_floor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
