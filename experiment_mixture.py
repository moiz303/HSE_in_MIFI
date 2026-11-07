#!/usr/bin/env python
"""Смесь прогноза с вероятностью метки SI=10000: забирается ли потолок в MSE.

Рядом живёт `experiment_sentinel.py` — он про те же строки-метки, но мерит
качество самого классификатора (AUC, Brier, ошибку калибровки) и разбавление
данными ChEMBL. Здесь же вопрос ровно один: даёт ли смесь выигрыш в метрике
соревнования. Опыты дополняют друг друга и считаются на одних и тех же folds.

`experiment_floor.py` показал, где лежит метрика: 57% MSE в распределении
соревнования приходится на 64 строки из 260, у которых SI ровно 10000. Это не
измерение, а метка «неактивно»: CC50 и IC50 упёрлись в границу диапазона.
Идеальное знание метки снизило бы MSE с 1.98e7 до 8.47e6, то есть потолок
выигрыша 57% — больше, чем даёт любой перебор гиперпараметров.

Но потолок достижим только если метка предсказуема по структуре. Поэтому первое,
что здесь считается, — **AUC и average precision OOF-классификатора** `SI=10000`.
Если AUC около 0.5, направление закрывается числом, и дальше смотреть незачем.

Если сигнал есть, проверяем смесь, оптимальную по MSE:

    прогноз = p * 10000 + (1 - p) * регрессия_по_неметкам

Здесь `p` — вероятность метки, а регрессия учится только на строках без метки,
чтобы метки не тянули её вверх. При честной вероятности это условное среднее,
а именно его и требует MSE.

Отдельный вопрос — на каких строках учить классификатор. Все 64 метки лежат в
распределении соревнования; 747 строк ChEMBL меток почти не содержат, и
классификатор, обученный на всём, может уехать к «метки не бывает». Поэтому
меряем оба варианта: обучение на всех строках и только на строках
в распределении.

Сравнение парное по строкам с bootstrap-интервалом, scaffold-сгруппированные
folds, holdout не трогается.

    python experiment_mixture.py                # 3 повтора по 5 folds
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
CLASSIFIER_PARAMS = {"max_iter": 300, "learning_rate": 0.06, "min_samples_leaf": 10,
                     "l2_regularization": 1.0, "random_state": 20261106}

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


def sentinel_probability(
    features: np.ndarray, label: np.ndarray, train: np.ndarray, held: np.ndarray
) -> np.ndarray:
    """P(SI=10000). Если в обучающей части один класс — вероятность постоянна."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    if train.size == 0 or len(np.unique(label[train])) < 2:
        return np.full(len(held), float(label[train].mean()) if train.size else 0.0)
    classifier = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS)
    classifier.fit(features[train], label[train])
    return classifier.predict_proba(features[held])[:, 1]


SHRINK = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def build_variants(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Все сравниваемые прогнозы из уже посчитанных OOF-величин.

    Ни одного дополнительного обучения: классификатор и регрессия посчитаны по
    folds один раз, а варианты — арифметика над ними. Это даёт разложение
    проигрыша смеси на две независимые причины:

    * `регрессия_неметки` — это смесь при коэффициенте сжатия 0, то есть вклад
      метки выключен совсем. Если она уже хуже базы, то вредит не формула
      смеси, а само обучение регрессии без строк-меток.
    * коэффициент сжатия от 0 до 1 умножает вероятность: `сж * p * 10000 +
      (1 - сж * p) * регрессия`. При сж=1 это исходная смесь. Если минимум
      кривой лежит внутри интервала, вероятность переоценена, и лечится это
      сжатием, а не другим классификатором.
    """
    built = {"база": out["база"], "регрессия_неметки": out["регрессия_неметки"]}
    for name in ("все", "в_распределении"):
        for shrink in SHRINK:
            if shrink == 0.0:
                continue        # это и есть регрессия_неметки, не дублируем
            weight = shrink * out[f"p_{name}"]
            built[f"смесь_{name}_сж{shrink:g}"] = (
                weight * SENTINEL_SI + (1.0 - weight) * out["регрессия_неметки"]
            )
            # Та же смесь, но вторым слагаемым базовая модель, обученная на всех
            # строках, а не регрессия без меток. Кривая сжатия показала, что
            # выбрасывание 64 строк-меток из обучения стоит дороже всего
            # остального, так что второе слагаемое менять нельзя — а подмешать
            # к нему вероятность метки по-прежнему можно.
            built[f"сбазой_{name}_сж{shrink:g}"] = (
                weight * SENTINEL_SI + (1.0 - weight) * out["база"]
            )
    return built


def bootstrap_interval(
    differences: np.ndarray, draws: int = 10000, seed: int = 7
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    means = differences[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def one_repeat(
    features: np.ndarray,
    si: np.ndarray,
    smiles: list[str],
    in_distribution: np.ndarray,
    folds: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """OOF прогнозы всех вариантов плюс вероятность метки, на всех строках."""
    label = (si == SENTINEL_SI).astype(int)
    fractions = {f"f{number}": 1.0 / folds for number in range(folds)}
    split = splits.scaffold_split(smiles, fractions, seed=seed)
    splits.assert_no_leaks(smiles, split)

    out: dict[str, np.ndarray] = {
        name: np.full(len(si), np.nan)
        for name in ("база", "регрессия_неметки", "смесь_все",
                     "смесь_в_распределении", "p_все", "p_в_распределении")
    }

    for positions in split.values():
        held = np.array(positions, dtype=int)
        if held.size == 0:
            continue
        rest = np.setdiff1d(np.arange(len(si)), held)

        out["база"][held] = (
            FlexibleSI(**BASE_SETTINGS).fit(features[rest], si[rest]).predict_si(features[held])
        )

        # Регрессия только по строкам без метки: иначе метки тянут её вверх и
        # смесь считает вклад «неактивно» дважды.
        plain = rest[label[rest] == 0]
        regression = (
            FlexibleSI(**BASE_SETTINGS).fit(features[plain], si[plain]).predict_si(features[held])
        )
        out["регрессия_неметки"][held] = regression

        for name, subset in (("все", rest),
                             ("в_распределении", rest[in_distribution[rest]])):
            probability = sentinel_probability(features, label, subset, held)
            out[f"p_{name}"][held] = probability
            out[f"смесь_{name}"][held] = (
                probability * SENTINEL_SI + (1.0 - probability) * regression
            )

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Строки-метки SI=10000: есть ли сигнал")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)

    from sklearn.metrics import average_precision_score, roc_auc_score

    frame, features, in_distribution = load_data()
    smiles = frame["standard_smiles"].astype(str).tolist()
    si = frame["SI"].to_numpy(float)
    label = (si == SENTINEL_SI).astype(int)
    log(f"{len(frame)} молекул, в распределении {int(in_distribution.sum())}, "
        f"строк-меток {int(label.sum())}")

    errors: dict[str, list[np.ndarray]] = {}
    rows: list[dict] = []
    variants: list[str] = []

    for number in range(args.repeats):
        seed = splits.DEFAULT_SEED + number
        started = time.monotonic()
        out = one_repeat(features, si, smiles, in_distribution, args.folds, seed)
        scored = in_distribution & np.isfinite(out["база"])
        true = si[scored]

        row = {"повтор": number + 1, "seed": seed, "строк": int(scored.sum())}
        for name in ("все", "в_распределении"):
            probability = out[f"p_{name}"][scored]
            row[f"auc_{name}"] = float(roc_auc_score(label[scored], probability))
            row[f"ap_{name}"] = float(average_precision_score(label[scored], probability))
        predictions = build_variants(out)
        if not variants:
            variants = list(predictions)
            errors = {name: [] for name in variants}
        for name in variants:
            error = (predictions[name][scored] - true) ** 2
            errors[name].append(error)
            row[f"mse_{name}"] = float(error.mean())
        rows.append(row)

        log(f"повтор {number + 1}/{args.repeats} (seed {seed}, {time.monotonic() - started:.0f} с)")
        log(f"классификатор метки: AUC {row['auc_все']:.3f} (обучен на всех) / "
            f"{row['auc_в_распределении']:.3f} (в распределении), "
            f"AP {row['ap_все']:.3f} / {row['ap_в_распределении']:.3f}  "
            f"[доля меток {label[scored].mean():.3f}]", indent=2)
        # В логе повтора только опорные точки: полная кривая сжатия — в итоге.
        log("  ".join(
            f"{name} {row[f'mse_{name}']:.4g}" for name in variants
            if name in ("база", "регрессия_неметки") or name.endswith("сж1")
        ), indent=2)

    table = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "experiment_mixture.csv", index=False)

    base_all = np.concatenate(errors["база"])
    log("")
    log(f"сигнал метки: AUC {table['auc_все'].mean():.3f} / "
        f"{table['auc_в_распределении'].mean():.3f}, "
        f"AP {table['ap_все'].mean():.3f} / {table['ap_в_распределении'].mean():.3f} "
        f"против базовой доли {label[in_distribution].mean():.3f}")
    log("")
    log(f"MSE базовой модели {base_all.mean():.4g}; остальное парно по строкам "
        f"(выигрыш > 0 значит лучше базы):")

    summary: list[dict] = []
    for name in variants:
        if name == "база":
            continue
        current = np.concatenate(errors[name])
        difference = base_all - current          # > 0 значит вариант лучше базы
        low, high = bootstrap_interval(difference)
        better = low > 0
        summary.append({
            "вариант": name, "mse": float(current.mean()),
            "выигрыш": float(difference.mean()), "интервал_95": [low, high],
            "лучше_базы": bool(better),
            "изменение_%": float(-100.0 * difference.mean() / base_all.mean()),
        })
        log(f"{name:<28} {current.mean():.4g}  "
            f"выигрыш {difference.mean():+.4g}  95% [{low:+.4g}, {high:+.4g}]  "
            f"{'ЛУЧШЕ' if better else ('хуже' if high < 0 else 'не показано')}", indent=2)

    # Критерий приёмки: мало того, что интервал не включает ноль. Вариантов
    # здесь восемнадцать, и при 95% интервалах один ложный выход за ноль — это
    # ожидаемое число, а не находка (поправка на множественные сравнения). Плюс
    # выигрыш должен быть практически значимым: разброс между scaffold-
    # разбиениями в этой задаче порядка 25%, и улучшение на полпроцента в нём
    # не живёт. Поэтому требуем ещё и порог по величине.
    MIN_GAIN = 0.02
    best = max(summary, key=lambda item: item["выигрыш"])
    significant = [item for item in summary if item["лучше_базы"]]
    worthwhile = best["лучше_базы"] and -best["изменение_%"] / 100.0 >= MIN_GAIN
    verdict = "принять" if worthwhile else "отклонить"
    log("")
    log(f"вердикт: {verdict} — лучший из вариантов {best['вариант']}: "
        f"MSE {best['изменение_%']:+.1f}% к базовой "
        f"({'улучшение' if best['изменение_%'] < 0 else 'ухудшение'})")
    log(f"вариантов проверено {len(summary)}, интервал не включает ноль у "
        f"{len(significant)} — при 95% интервалах примерно столько и ожидается "
        f"случайно; порог практической значимости {MIN_GAIN:.0%}", indent=2)

    (REPORTS / "experiment_mixture.json").write_text(
        json.dumps({
            "конфигурация": FlexibleSI(**BASE_SETTINGS).describe(),
            "повторов": int(args.repeats), "folds": int(args.folds),
            "строк_меток": int(label.sum()),
            "auc_все": float(table["auc_все"].mean()),
            "auc_в_распределении": float(table["auc_в_распределении"].mean()),
            "ap_все": float(table["ap_все"].mean()),
            "ap_в_распределении": float(table["ap_в_распределении"].mean()),
            "mse_база": float(base_all.mean()),
            "варианты": summary,
            "вердикт": verdict,
            "по_повторам": rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log("таблица → reports/experiment_mixture.csv, итог → reports/experiment_mixture.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
