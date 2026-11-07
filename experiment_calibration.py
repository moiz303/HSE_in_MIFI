#!/usr/bin/env python
"""Проверка гипотезы: линейная поправка масштаба прогнозов SI.

Гипотеза (idea 1.md). Модель может правильно упорядочивать молекулы, но
систематически занижать крупные значения SI: бустинг с регуляризацией стягивает
прогноз к среднему. Тогда саму модель менять не нужно — достаточно поправки

    SI_итог = a + b * SI_прогноз,

где a и b подбираются по MSE в сырых единицах (метрика соревнования), а не по
ошибке логарифмов.

Как проверяем, чтобы не обмануть себя. Поправка, подобранная на тех же
прогнозах, на которых её потом измеряют, запоминает ошибки и всегда «улучшает»
метрику. Поэтому схема вложенная:

    внешнее scaffold-разбиение -> train / validation
      на внешнем train: scaffold-OOF прогнозы (молекула не видит свой каркас)
      по OOF подбираем a и b
      модель переобучаем на всём внешнем train, применяем к validation
      сравниваем MSE до и после поправки

Так a и b ни разу не видят ту выборку, на которой считается выигрыш. Решение
принимается по числу внешних разбиений, где поправка выигрывает: одно
разбиение на 140 молекулах ничего не решает.

Отдельно важно, на каких строках подбирать поправку. Валидация содержит хвост
ChEMBL с SI до 2.2e5, которого в тесте соревнования нет; коэффициенты, выбранные
по этому хвосту, уедут от распределения теста. Поэтому считаем оба варианта:
подбор только по строкам kaggle_train ("в распределении") и по всем строкам.

Holdout здесь не участвует.

    python experiment_calibration.py                 # 5 внешних разбиений
    python experiment_calibration.py --repeats 3 --inner-folds 5
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

# Лучшая на сейчас конфигурация из reports/sweep.csv: её и калибруем.
BASE_SETTINGS = {"transform": "raw", "loss": "squared_error", "clip_si": 20000.0}

_STARTED = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    seconds = time.monotonic() - _STARTED
    print(f"  {int(seconds // 60)}:{seconds % 60:04.1f}  {' ' * indent}{message}", flush=True)


# --------------------------------------------------------------------- данные
def featurize_cached(smiles: list[str], name: str) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE) as stored:
            cache = {key: stored[key] for key in stored.files}
    key = f"{name}|{len(smiles)}"
    if key in cache:
        return cache[key]
    matrix = fingerprints.featurize(smiles)
    cache[key] = matrix
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, **cache)
    return matrix


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    for path in (TRAIN_CSV, VALIDATION_CSV):
        if not path.exists():
            raise SystemExit(f"нет {path}. На машине с данными: python -m src.pipeline export")
    frame = pd.concat(
        [pd.read_csv(TRAIN_CSV), pd.read_csv(VALIDATION_CSV)], ignore_index=True
    )
    smiles = frame["standard_smiles"].astype(str).tolist()
    log(f"объединено {len(frame)} молекул, считаю признаки…")
    features = featurize_cached(smiles, "morgan2048+desc13")
    in_distribution = (
        frame["source"].fillna("").str.contains("kaggle_train").to_numpy()
        if "source" in frame.columns
        else np.ones(len(frame), dtype=bool)
    )
    log(f"матрица {features.shape[0]}×{features.shape[1]}, "
        f"в распределении соревнования: {int(in_distribution.sum())}")
    return frame, features, in_distribution


# ------------------------------------------------------------------- поправка
def fit_affine(predicted: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """a и b по методу наименьших квадратов в сырых единицах."""
    design = np.column_stack([np.ones_like(predicted), predicted])
    coefficients, *_ = np.linalg.lstsq(design, true, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def fit_scale(predicted: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Только масштаб: b без сдвига. Одна степень свободы вместо двух."""
    denominator = float((predicted**2).sum())
    if denominator <= 0:
        return 0.0, 1.0
    return 0.0, float((predicted * true).sum() / denominator)


FITTERS = {"affine": fit_affine, "scale": fit_scale}


def apply_calibration(predicted: np.ndarray, a: float, b: float) -> np.ndarray:
    """Поправка плюс единственное ограничение: SI не бывает отрицательным.

    Верхнюю обрезку сознательно не возвращаем: смысл гипотезы в том, чтобы
    поднять крупные прогнозы, а повторный clip по clip_si отменил бы ровно это.
    """
    return np.clip(a + b * predicted, 0.0, None)


def oof_predictions(
    features: np.ndarray,
    si: np.ndarray,
    smiles: list[str],
    index: np.ndarray,
    settings: dict,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Прогнозы «как будто молекула новая»: scaffold-folds внутри внешнего train."""
    local_smiles = [smiles[i] for i in index]
    fractions = {f"f{number}": 1.0 / folds for number in range(folds)}
    inner = splits.scaffold_split(local_smiles, fractions, seed=seed)
    splits.assert_no_leaks(local_smiles, inner)

    out_of_fold = np.full(len(index), np.nan)
    for positions in inner.values():
        held = np.array(positions, dtype=int)
        if held.size == 0:
            continue
        rest = np.setdiff1d(np.arange(len(index)), held)
        model = FlexibleSI(**settings).fit(features[index[rest]], si[index[rest]])
        out_of_fold[held] = model.predict_si(features[index[held]])
    return out_of_fold


def mse(predicted: np.ndarray, true: np.ndarray) -> float:
    return float(((predicted - true) ** 2).mean())


# ----------------------------------------------------------------------- опыт
def run(repeats: int, folds: int, settings: dict) -> pd.DataFrame:
    frame, features, in_distribution = load_data()
    smiles = frame["standard_smiles"].astype(str).tolist()
    si = frame["SI"].to_numpy(float)

    rows: list[dict] = []
    for number in range(repeats):
        seed = splits.DEFAULT_SEED + number
        outer = splits.scaffold_split(smiles, seed=seed)
        splits.assert_no_leaks(smiles, outer)
        train_index = np.array(outer["train"], dtype=int)
        validation_index = np.array(outer["validation"], dtype=int)

        started = time.monotonic()
        out_of_fold = oof_predictions(
            features, si, smiles, train_index, settings, folds, seed
        )
        usable = np.isfinite(out_of_fold) & np.isfinite(si[train_index]) & (si[train_index] > 0)

        model = FlexibleSI(**settings).fit(features[train_index], si[train_index])
        raw_validation = model.predict_si(features[validation_index])
        true_validation = si[validation_index]
        scored = in_distribution[validation_index]

        base_in = mse(raw_validation[scored], true_validation[scored])
        base_all = mse(raw_validation, true_validation)

        log(f"разбиение {number + 1}/{repeats} (seed {seed}): "
            f"OOF по {int(usable.sum())} молекулам, "
            f"валидация {len(validation_index)} из них в распределении "
            f"{int(scored.sum())}  ({time.monotonic() - started:.0f} с)")
        log(f"без поправки: MSE в распределении {base_in:.4g}, вся валидация {base_all:.4g}",
            indent=2)

        # Пол, ниже которого модель бессмысленна: средний SI по обучающим
        # строкам. Нужен здесь, а не только в constant_baseline.py: подобранные
        # коэффициенты (b < 1, a ~ 2000) подозрительно похожи на «вернуться
        # к константе», и проверить это можно только на тех же разбиениях.
        train_si = si[train_index]
        train_in = train_si[in_distribution[train_index] & np.isfinite(train_si)]
        constants = {
            "константа_train": float(np.nanmean(np.minimum(train_si, settings["clip_si"]))),
            "константа_в_распределении": float(np.mean(np.minimum(train_in, settings["clip_si"]))),
        }
        for label, value in constants.items():
            rows.append({
                "разбиение": number + 1, "seed": seed, "поправка": label, "подбор": "—",
                "a": value, "b": 0.0, "n_подбора": int(len(train_in)),
                "mse_база_в_распределении": base_in,
                "mse_поправка_в_распределении": mse(
                    np.full(int(scored.sum()), value), true_validation[scored]
                ),
                "mse_база_вся": base_all,
                "mse_поправка_вся": mse(np.full(len(true_validation), value), true_validation),
            })
            rows[-1]["выигрыш_в_распределении"] = (
                base_in - rows[-1]["mse_поправка_в_распределении"]
            )
            sign = "лучше модели" if rows[-1]["выигрыш_в_распределении"] > 0 else "хуже модели"
            log(f"{label:<26} SI={value:>8.4g}       "
                f"MSE {rows[-1]['mse_поправка_в_распределении']:.4g}  ({sign})", indent=2)

        for form, fitter in FITTERS.items():
            for subset in ("indist", "all"):
                mask = usable.copy()
                if subset == "indist":
                    mask &= in_distribution[train_index]
                if mask.sum() < 20:
                    continue
                a, b = fitter(out_of_fold[mask], si[train_index][mask])
                calibrated = apply_calibration(raw_validation, a, b)
                row = {
                    "разбиение": number + 1,
                    "seed": seed,
                    "поправка": form,
                    "подбор": subset,
                    "a": a,
                    "b": b,
                    "n_подбора": int(mask.sum()),
                    "mse_база_в_распределении": base_in,
                    "mse_поправка_в_распределении": mse(
                        calibrated[scored], true_validation[scored]
                    ),
                    "mse_база_вся": base_all,
                    "mse_поправка_вся": mse(calibrated, true_validation),
                }
                row["выигрыш_в_распределении"] = (
                    row["mse_база_в_распределении"] - row["mse_поправка_в_распределении"]
                )
                rows.append(row)
                sign = "лучше" if row["выигрыш_в_распределении"] > 0 else "хуже"
                log(f"{form:<7} подбор={subset:<7} a={a:>10.4g} b={b:>6.3f}  "
                    f"MSE {row['mse_поправка_в_распределении']:.4g}  ({sign})", indent=2)

    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Сводка с парной ошибкой: среднего выигрыша мало, нужен его разброс.

    Разбиения различаются по сложности втрое (MSE от 1.3e7 до 3.6e7), поэтому
    сравнивать надо попарно — выигрыш внутри разбиения — и смотреть, перекрывает
    ли средний выигрыш собственную ошибку. Отношение `t` = выигрыш / ошибка
    среднего: меньше 2 означает «не показано», сколько бы ни было побед.
    """
    grouped = table.groupby(["поправка", "подбор"], as_index=False).agg(
        разбиений=("разбиение", "count"),
        a_медиана=("a", "median"),
        b_медиана=("b", "median"),
        b_мин=("b", "min"),
        b_макс=("b", "max"),
        mse_база=("mse_база_в_распределении", "mean"),
        mse_поправка=("mse_поправка_в_распределении", "mean"),
        побед=("выигрыш_в_распределении", lambda values: int((values > 0).sum())),
        выигрыш_среднее=("выигрыш_в_распределении", "mean"),
        выигрыш_sd=("выигрыш_в_распределении", lambda values: float(values.std(ddof=1))),
    )
    grouped["выигрыш_ошибка"] = grouped["выигрыш_sd"] / np.sqrt(grouped["разбиений"])
    grouped["t"] = np.where(
        grouped["выигрыш_ошибка"] > 0,
        grouped["выигрыш_среднее"] / grouped["выигрыш_ошибка"],
        0.0,
    )
    grouped["изменение_%"] = (
        100.0 * (grouped["mse_поправка"] - grouped["mse_база"]) / grouped["mse_база"]
    )
    return grouped.sort_values("mse_поправка").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Линейная поправка масштаба прогнозов SI")
    parser.add_argument("--repeats", type=int, default=5, help="внешних scaffold-разбиений")
    parser.add_argument("--inner-folds", type=int, default=5, help="folds для OOF внутри train")
    args = parser.parse_args(argv)

    log(f"конфигурация модели: {FlexibleSI(**BASE_SETTINGS).describe()}")
    table = run(args.repeats, args.inner_folds, BASE_SETTINGS)
    if table.empty:
        raise SystemExit("ни один вариант поправки не удалось подобрать")

    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "experiment_calibration.csv", index=False)
    summary = summarise(table)

    log("")
    log("итог (MSE в распределении соревнования, базовая модель "
        f"{summary['mse_база'].iloc[0]:.4g}):")
    for _, row in summary.iterrows():
        log(f"{row['поправка']:<26} подбор={row['подбор']:<7} "
            f"{row['mse_поправка']:.4g}  ({row['изменение_%']:+.1f}%)  "
            f"побед {row['побед']}/{row['разбиений']}  t={row['t']:+.2f}", indent=2)

    # Вердикт выносим только по вариантам поправки: константы здесь — опорная
    # точка, а не кандидат на применение.
    candidates = summary[summary["поправка"].isin(FITTERS)]
    best = candidates.iloc[0]
    # Критерий приёмки строгий намеренно. Разбиений пять, и «3 из 5 побед»
    # получается подбрасыванием монеты в треть случаев, поэтому одного
    # большинства мало: нужен выигрыш, перекрывающий свою же ошибку (t >= 2),
    # и устойчивые коэффициенты. Разъехавшийся b — это ровно то, чего боялся
    # автор гипотезы: подстройка под несколько редких молекул.
    stable = best["b_макс"] <= 2.0 * best["b_мин"] if best["b_мин"] > 0 else False
    verdict = (
        "принять"
        if best["t"] >= 2.0 and best["побед"] >= 4 and best["изменение_%"] < 0 and stable
        else "отклонить"
    )
    log("")
    log(f"вердикт: {verdict} — лучший вариант {best['поправка']}/{best['подбор']}: "
        f"{best['изменение_%']:+.1f}%, побед {best['побед']}/{best['разбиений']}, "
        f"t={best['t']:+.2f}, b от {best['b_мин']:.3f} до {best['b_макс']:.3f}")

    (REPORTS / "experiment_calibration.json").write_text(
        json.dumps({
            "конфигурация": FlexibleSI(**BASE_SETTINGS).describe(),
            "разбиений": int(args.repeats),
            "folds_внутри": int(args.inner_folds),
            "вердикт": verdict,
            "варианты": summary.to_dict(orient="records"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log("таблица → reports/experiment_calibration.csv, итог → "
        "reports/experiment_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
