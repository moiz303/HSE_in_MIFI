#!/usr/bin/env python
"""Два опыта: качество классификатора меток и разбавление данными ChEMBL.

    python experiment_sentinel.py                  # оба, бюджет 5 минут
    python experiment_sentinel.py --only classifier
    python experiment_sentinel.py --only dilution
    python experiment_sentinel.py --repeats 4 --max-minutes 8

**Зачем.** `experiment_floor.py` показал, что 57% MSE приходится на строки, где
цель задана меткой `SI = 10000`, и что с идеальным угадыванием метки ошибка
упала бы с 1.98e7 до 8.47e6 — то есть в 2.3 раза. Это самый большой известный
рычаг, и до сих пор мы мерили его только через итоговый MSE, где вклад
классификатора тонет.

**Опыт 1 — классификатор как классификатор.** Считаем AUC, точность, Brier и
ошибку калибровки (ECE) для четырёх семейств. Калибровка здесь не
академическая деталь: смесь `p·10000 + (1−p)·регрессия` оптимальна для
квадратичной ошибки только при честной вероятности. Переуверенный
классификатор хуже честного даже при той же точности, а бустинг известен
переуверенностью — поэтому в сравнении есть лес и логистическая регрессия.

**Опыт 2 — разбавление.** Из 1184 обучающих строк 874 взяты из ChEMBL: другие
клеточные линии, другие протоколы, хвост SI до 2.2e5 при максимуме
соревнования 3.6e4. Они дают объём, но могут тянуть модель от распределения
теста. Сравниваем обучение только на строках соревнования, смешанное и
смешанное с повышенным весом строк соревнования.

Оба опыта считаются на одних и тех же scaffold-фолдах, поэтому сравнимы между
собой. Утечки каркасов не возникает: разбиение строится по всем строкам сразу,
и строка ChEMBL с каркасом из отложенного fold тоже оказывается отложенной.
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

# Та же конфигурация, что победила в переборе: сравнивать надо при прочих равных.
BASE_SETTINGS = {"transform": "raw", "loss": "squared_error", "clip_si": 20000.0}
LABEL_VALUE = 10000.0
SCREEN_PARAMS = {"max_iter": 150, "n_estimators": 200}

_STARTED = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    seconds = time.monotonic() - _STARTED
    print(f"  {int(seconds // 60)}:{seconds % 60:04.1f}  {' ' * indent}{message}", flush=True)


# --------------------------------------------------------------------- данные
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

    in_distribution = frame["source"].fillna("").str.contains("kaggle_train").to_numpy()
    return frame, features, in_distribution


def fold_indices(smiles: list[str], folds: int, seed: int) -> list[np.ndarray]:
    fractions = {f"f{number}": 1.0 / folds for number in range(folds)}
    split = splits.scaffold_split(smiles, fractions, seed=seed)
    splits.assert_no_leaks(smiles, split)
    return [np.array(positions, dtype=int) for positions in split.values()]


# ------------------------------------------------------------------- метрики
def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC через ранги: доля правильно упорядоченных пар."""
    positive, negative = labels.astype(bool), ~labels.astype(bool)
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Средние ранги для связок, иначе AUC смещается на одинаковых вероятностях.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            mask = inverse == index
            ranks[mask] = ranks[mask].mean()
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """ECE: средний по корзинам разрыв между вероятностью и частотой.

    Именно эта величина говорит, можно ли доверять вероятности как весу в
    смеси. Точность может быть хорошей при плохой калибровке.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1.0 else probability <= 1.0)
        if not mask.any():
            continue
        total += mask.mean() * abs(probability[mask].mean() - labels[mask].mean())
    return float(total)


def build_classifier(kind: str):
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=SCREEN_PARAMS["max_iter"], learning_rate=0.06,
            min_samples_leaf=10, l2_regularization=1.0, random_state=20261106,
        )
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=SCREEN_PARAMS["n_estimators"], min_samples_leaf=1,
            max_features=0.3, random_state=20261106, n_jobs=-1,
        )
    if kind == "extra":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=SCREEN_PARAMS["n_estimators"], min_samples_leaf=1,
            max_features=0.3, random_state=20261106, n_jobs=-1,
        )
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Линейная модель калибрована по построению — опорная точка для ECE.
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, C=1.0),
        )
    raise ValueError(f"неизвестный классификатор: {kind!r}")


# ------------------------------------------------- опыт 1: классификатор
def run_classifier(
    frame: pd.DataFrame,
    features: np.ndarray,
    in_distribution: np.ndarray,
    folds_list: list[list[np.ndarray]],
    deadline: float,
) -> list[dict]:
    si = frame["SI"].to_numpy(float)
    labels = frame["target_is_label"].fillna(False).to_numpy(dtype=bool)
    log(f"меток в выборке: {int(labels.sum())} из {len(labels)} "
        f"(в распределении соревнования: {int((labels & in_distribution).sum())})")

    results: list[dict] = []
    for kind in ("hgb", "rf", "extra", "logreg"):
        if time.monotonic() > deadline:
            log(f"бюджет исчерпан, не проверено: {kind}")
            break
        started = time.monotonic()
        per_repeat: list[dict] = []

        for repeat, fold_list in enumerate(folds_list):
            probability = np.full(len(si), np.nan)
            regression = np.full(len(si), np.nan)
            baseline_prediction = np.full(len(si), np.nan)
            for held in fold_list:
                rest = np.setdiff1d(np.arange(len(si)), held)
                if labels[rest].sum() < 5 or (~labels[rest]).sum() < 5:
                    continue
                classifier = build_classifier(kind)
                classifier.fit(features[rest], labels[rest].astype(int))
                probability[held] = classifier.predict_proba(features[held])[:, 1]
                # Регрессия для смеси учится только на строках с измеренным SI:
                # подмешивать метку значит учить её предсказывать разметку.
                measured = rest[~labels[rest]]
                model = FlexibleSI(**BASE_SETTINGS, params=SCREEN_PARAMS)
                model.fit(features[measured], si[measured])
                regression[held] = model.predict_si(features[held])

                # База для сравнения — рабочая модель, обученная на ВСЕХ
                # строках, включая метки. Сравнивать смесь с регрессией без
                # меток нельзя: это искусственно ослабленная база, и выигрыш
                # над ней ничего не значит. Прямая модель уже выучивает класс
                # меток сама, и исключение этих строк стоит 45% метрики.
                full = FlexibleSI(**BASE_SETTINGS, params=SCREEN_PARAMS)
                full.fit(features[rest], si[rest])
                baseline_prediction[held] = full.predict_si(features[held])

            done = np.isfinite(probability) & in_distribution
            if done.sum() < 20:
                continue
            true_label = labels[done]
            probability_done = probability[done]
            blended = probability_done * LABEL_VALUE + (1 - probability_done) * regression[done]
            per_repeat.append({
                "auc": auc(true_label, probability_done),
                "accuracy": float(((probability_done >= 0.5) == true_label).mean()),
                "brier": float(((probability_done - true_label) ** 2).mean()),
                "ece": expected_calibration_error(true_label.astype(float), probability_done),
                "mse_смесь": float(((blended - si[done]) ** 2).mean()),
                "mse_база": float(((baseline_prediction[done] - si[done]) ** 2).mean()),
                "mse_регрессия_без_меток": float(((regression[done] - si[done]) ** 2).mean()),
            })

        if not per_repeat:
            continue
        summary = {"классификатор": kind, "повторов": len(per_repeat)}
        for key in per_repeat[0]:
            values = np.array([item[key] for item in per_repeat], dtype=float)
            summary[key] = float(values.mean())
            summary[f"{key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        results.append(summary)
        log(f"{kind:<7} AUC {summary['auc']:.3f}  точность {summary['accuracy']:.3f}  "
            f"Brier {summary['brier']:.4f}  ECE {summary['ece']:.4f}  "
            f"MSE смеси {summary['mse_смесь']:.4g} против {summary['mse_только_регрессия']:.4g} "
            f"без неё  ({time.monotonic() - started:.0f} с)")
    return results


# --------------------------------------------------- опыт 2: разбавление
def run_dilution(
    frame: pd.DataFrame,
    features: np.ndarray,
    in_distribution: np.ndarray,
    folds_list: list[list[np.ndarray]],
    deadline: float,
) -> list[dict]:
    si = frame["SI"].to_numpy(float)
    log(f"строк соревнования {int(in_distribution.sum())}, "
        f"только ChEMBL {int((~in_distribution).sum())}")

    variants = {
        "только соревнование": None,
        "смешанное (как сейчас)": 1.0,
        "смешанное, вес соревнования x3": 3.0,
        "смешанное, вес соревнования x10": 10.0,
    }
    results: list[dict] = []
    for name, weight in variants.items():
        if time.monotonic() > deadline:
            log(f"бюджет исчерпан, не проверено: {name}")
            break
        started = time.monotonic()
        per_repeat: list[float] = []

        for fold_list in folds_list:
            prediction = np.full(len(si), np.nan)
            for held in fold_list:
                rest = np.setdiff1d(np.arange(len(si)), held)
                if weight is None:
                    train = rest[in_distribution[rest]]
                    weights = None
                else:
                    train = rest
                    weights = np.where(in_distribution[train], weight, 1.0)
                if train.size < 30:
                    continue
                model = FlexibleSI(**BASE_SETTINGS, params=SCREEN_PARAMS)
                model.fit(features[train], si[train], sample_weight=weights)
                prediction[held] = model.predict_si(features[held])

            done = np.isfinite(prediction) & in_distribution
            if done.sum() >= 20:
                per_repeat.append(float(((prediction[done] - si[done]) ** 2).mean()))

        if not per_repeat:
            continue
        values = np.array(per_repeat)
        results.append({
            "вариант": name,
            "mse": float(values.mean()),
            "mse_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "повторов": len(values),
        })
        log(f"{name:<32} MSE {values.mean():.4g} ± "
            f"{values.std(ddof=1) if len(values) > 1 else 0:.3g}  "
            f"({time.monotonic() - started:.0f} с)")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="жёсткий бюджет в минутах; 0 — без ограничения")
    parser.add_argument("--only", choices=["classifier", "dilution"], default=None)
    args = parser.parse_args(argv)

    frame, features, in_distribution = load_data()
    smiles = frame["standard_smiles"].astype(str).tolist()
    log(f"{len(frame)} строк, матрица {features.shape[0]}×{features.shape[1]}")

    folds_list = [
        fold_indices(smiles, args.folds, splits.DEFAULT_SEED + repeat)
        for repeat in range(args.repeats)
    ]
    log(f"{args.repeats} повторов по {args.folds} фолдов, бюджет "
        f"{f'{args.max_minutes:g} мин' if args.max_minutes > 0 else 'без ограничения'}")

    deadline = (
        time.monotonic() + args.max_minutes * 60 if args.max_minutes > 0 else float("inf")
    )
    report: dict[str, object] = {"повторов": args.repeats, "folds": args.folds}

    if args.only in (None, "classifier"):
        log("")
        log("опыт 1: классификатор меток как классификатор")
        classifier_rows = run_classifier(frame, features, in_distribution, folds_list, deadline)
        report["классификатор"] = classifier_rows
        if classifier_rows:
            table = pd.DataFrame(classifier_rows).sort_values("mse_смесь")
            table.to_csv(REPORTS / "experiment_sentinel_classifier.csv", index=False)
            best = table.iloc[0]
            gain = best["mse_база"] - best["mse_смесь"]
            log("")
            log(f"лучший классификатор: {best['классификатор']}. Смесь "
                f"{'выигрывает' if gain > 0 else 'проигрывает'} базе {abs(gain):.4g} "
                f"при разбросе {best['mse_смесь_sd']:.3g}")
            report["вердикт_классификатор"] = (
                "принять" if gain > best["mse_смесь_sd"] else
                "отклонить" if gain < -best["mse_смесь_sd"] else "в пределах шума"
            )

    if args.only in (None, "dilution"):
        log("")
        log("опыт 2: разбавление данными ChEMBL")
        dilution_rows = run_dilution(frame, features, in_distribution, folds_list, deadline)
        report["разбавление"] = dilution_rows
        if dilution_rows:
            table = pd.DataFrame(dilution_rows).sort_values("mse")
            table.to_csv(REPORTS / "experiment_sentinel_dilution.csv", index=False)
            best, current = table.iloc[0], table[table["вариант"].str.contains("как сейчас")]
            log("")
            if not current.empty:
                gain = float(current.iloc[0]["mse"]) - float(best["mse"])
                log(f"лучший: {best['вариант']}, выигрыш над текущим {gain:.4g} "
                    f"при разбросе {best['mse_sd']:.3g}")
                report["вердикт_разбавление"] = (
                    "принять" if gain > best["mse_sd"] else "в пределах шума"
                )

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "experiment_sentinel.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("")
    log("отчёт → reports/experiment_sentinel.json и две csv-таблицы")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
