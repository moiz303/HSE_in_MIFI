#!/usr/bin/env python
"""Обучение модели SI. Запускать на сильной машине, не на ноутбуке сборщика данных.

Скрипт самодостаточен: берёт два готовых csv, обучает, печатает метрики с
доверительными интервалами и сохраняет модель. Сети не требует.

    python -m src.pipeline export        # это уже сделано, файлы в репозитории
    python train_model.py                # обе стратегии, сравнение
    python train_model.py --strategy direct --repeats 5
    python train_model.py --predict data/raw/test.csv --out reports/submission.csv

Что он решает. Открытый вопрос команды: предсказывать SI одной головой
(`direct`) или считать его из двух прогнозов, log IC50 и log CC50 (`two_head`).
На одном разбиении из 140 молекул разница внутри шума, поэтому здесь есть
`--repeats`: несколько scaffold-разбиений с разными seed и усреднение. Это и
есть замер, которого не хватало.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate, splits
from src.features import fingerprints
from src.models import baseline

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

SMILES_COLUMN = "standard_smiles"

_STARTED = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    """Строка лога с отметкой времени от старта, сразу в поток.

    Обучение идёт минутами, и запускающий должен видеть прогресс, а не пустой
    терминал. flush обязателен: вывод уходит в tee.
    """
    seconds = time.monotonic() - _STARTED
    stamp = f"{int(seconds // 60):d}:{seconds % 60:04.1f}"
    print(f"  {stamp:>7}  {' ' * indent}{message}", flush=True)


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"нет {path}. На машине со данными: python -m src.pipeline export"
        )
    frame = pd.read_csv(path)
    missing = [c for c in (SMILES_COLUMN, "IC50", "CC50", "SI") if c not in frame.columns]
    if missing:
        raise SystemExit(f"в {path.name} нет колонок: {', '.join(missing)}")
    return frame


def featurize(frame: pd.DataFrame) -> np.ndarray:
    return fingerprints.featurize(frame[SMILES_COLUMN].astype(str).tolist())


def fit(strategy: str, features: np.ndarray, frame: pd.DataFrame, params: dict | None = None):
    model = baseline.build(strategy, params)
    if strategy == "two_head":
        model.fit(features, frame["IC50"].to_numpy(), frame["CC50"].to_numpy())
    else:
        model.fit(features, frame["SI"].to_numpy())
    return model


def score(model, features: np.ndarray, frame: pd.DataFrame) -> dict:
    true = baseline.to_log(frame["SI"].to_numpy())
    predicted = model.predict_log_si(features)
    result = evaluate.metrics(true, predicted)
    low, high = evaluate.bootstrap_ci(true, predicted)
    result["rmse_log10_ci95"] = [low, high]
    return result


def run_fixed(strategies: list[str], params: dict | None) -> dict[str, dict]:
    """Обучить на train.csv, проверить на validation.csv — как есть."""
    train, validation = load(TRAIN_CSV), load(VALIDATION_CSV)
    log(f"train {len(train)} молекул, validation {len(validation)} молекул")

    log("считаю признаки: Morgan 2048 бит + RDKit 2D…")
    features_train, features_validation = featurize(train), featurize(validation)
    log(f"матрица признаков {features_train.shape[0]}×{features_train.shape[1]}")

    report: dict[str, dict] = {}
    for number, strategy in enumerate(strategies, start=1):
        log(f"[{number}/{len(strategies)}] обучаю {strategy}…")
        started = time.monotonic()
        model = fit(strategy, features_train, train, params)
        log(f"обучено за {time.monotonic() - started:.1f} с, считаю метрики с бутстрапом…", indent=2)

        result = score(model, features_validation, validation)
        report[strategy] = result
        low, high = result["rmse_log10_ci95"]
        log(f"{strategy}: RMSE log10 = {result['rmse_log10']:.3f} [{low:.3f}; {high:.3f}], "
            f"Spearman = {result['spearman']:.3f}", indent=2)

        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump({"model": model, "strategy": strategy}, ARTIFACTS / f"model_{strategy}.joblib")
        log(f"модель → artifacts/model_{strategy}.joblib", indent=2)

    if len(report) > 1:
        best = min(report, key=lambda key: report[key]["rmse_log10"])
        gap = abs(report[strategies[0]]["rmse_log10"] - report[strategies[-1]]["rmse_log10"])
        width = max(r["rmse_log10_ci95"][1] - r["rmse_log10_ci95"][0] for r in report.values())
        log("")
        if gap > width:
            log(f"итог: {best} лучше (разница {gap:.3f} шире интервала {width:.3f})")
        else:
            log(f"итог: не решено — разница {gap:.3f} меньше ширины интервала {width:.3f}; "
                f"нужен прогон с --repeats")
    return report


def run_repeats(strategies: list[str], repeats: int, params: dict | None) -> dict[str, dict]:
    """Несколько scaffold-разбиений объединённых данных.

    Holdout сюда не попадает: он не экспортируется. Объединяем train и
    validation и режем заново — так разница между стратегиями перестаёт
    зависеть от одного удачного разбиения.
    """
    combined = pd.concat([load(TRAIN_CSV), load(VALIDATION_CSV)], ignore_index=True)
    log(f"объединено {len(combined)} молекул, разбиений: {repeats}")
    log("считаю признаки один раз на всю выборку…")
    features = featurize(combined)
    smiles = combined[SMILES_COLUMN].astype(str).tolist()
    log(f"матрица признаков {features.shape[0]}×{features.shape[1]}")
    log(f"всего обучений: {repeats * len(strategies)}")

    collected: dict[str, list[dict]] = {strategy: [] for strategy in strategies}
    for index in range(repeats):
        seed = splits.DEFAULT_SEED + index
        log(f"[разбиение {index + 1}/{repeats}] seed {seed}, режу по каркасам Мурко…")
        split = splits.scaffold_split(smiles, seed=seed)
        splits.assert_no_leaks(smiles, split)
        train_index, validation_index = split["train"], split["validation"]
        log(f"train {len(train_index)}, validation {len(validation_index)}, утечек нет", indent=2)

        for strategy in strategies:
            started = time.monotonic()
            model = fit(strategy, features[train_index], combined.iloc[train_index], params)
            result = score(model, features[validation_index], combined.iloc[validation_index])
            collected[strategy].append(result)
            log(f"{strategy:<9} RMSE {result['rmse_log10']:.3f}  "
                f"Spearman {result['spearman']:.3f}  ({time.monotonic() - started:.1f} с)",
                indent=2)

    log("")
    log("среднее по разбиениям:")
    report: dict[str, dict] = {}
    for strategy, results in collected.items():
        rmse = np.array([r["rmse_log10"] for r in results])
        spearman = np.array([r["spearman"] for r in results])
        report[strategy] = {
            "rmse_log10_mean": float(rmse.mean()),
            "rmse_log10_std": float(rmse.std(ddof=1)) if len(rmse) > 1 else 0.0,
            "spearman_mean": float(spearman.mean()),
            "spearman_std": float(spearman.std(ddof=1)) if len(spearman) > 1 else 0.0,
            "n_splits": len(results),
        }
        log(f"{strategy:<9} RMSE {rmse.mean():.3f} ± {report[strategy]['rmse_log10_std']:.3f}   "
            f"Spearman {spearman.mean():.3f} ± {report[strategy]['spearman_std']:.3f}", indent=2)

    if len(strategies) > 1:
        best = min(report, key=lambda key: report[key]["rmse_log10_mean"])
        spread = max(report[s]["rmse_log10_std"] for s in report)
        gap = abs(
            report[strategies[0]]["rmse_log10_mean"] - report[strategies[-1]]["rmse_log10_mean"]
        )
        log("")
        if gap > spread:
            log(f"итог: {best} лучше — разница {gap:.3f} шире разброса между разбиениями {spread:.3f}")
        else:
            log(f"итог: не решено — разница {gap:.3f} меньше разброса между разбиениями {spread:.3f}")
    return report


def predict(path: Path, strategy: str, id_column: str, smiles_column: str,
            target_column: str, out: Path) -> None:
    """Предсказать для файла теста и записать csv для Kaggle."""
    import joblib

    model_path = ARTIFACTS / f"model_{strategy}.joblib"
    if not model_path.exists():
        raise SystemExit(f"нет {model_path.name}: сначала обучите модель")
    model = joblib.load(model_path)["model"]

    test = pd.read_csv(path)
    for column in (id_column, smiles_column):
        if column not in test.columns:
            raise SystemExit(
                f"в {path.name} нет колонки {column!r}. Есть: {', '.join(test.columns)}"
            )
    features = fingerprints.featurize(test[smiles_column].astype(str).tolist())
    predicted = baseline.from_log(model.predict_log_si(features))

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({id_column: test[id_column], target_column: predicted}).to_csv(out, index=False)
    log(f"предсказания: {len(test)} строк → {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strategy", choices=["two_head", "direct", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=0,
                        help="сколько scaffold-разбиений прогнать вместо одного фиксированного")
    parser.add_argument("--params", default=None,
                        help="json с гиперпараметрами для HistGradientBoostingRegressor")
    parser.add_argument("--metrics-out", default=str(REPORTS / "train_metrics.json"))
    parser.add_argument("--predict", default=None, help="csv теста для предсказания")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--target-column", default="SI")
    parser.add_argument("--out", default=str(REPORTS / "submission.csv"))
    args = parser.parse_args(argv)

    strategies = ["two_head", "direct"] if args.strategy == "both" else [args.strategy]
    params = json.loads(Path(args.params).read_text(encoding="utf-8")) if args.params else None

    report = (run_repeats(strategies, args.repeats, params) if args.repeats > 0
              else run_fixed(strategies, params))

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"метрики → {metrics_path}")

    if args.predict:
        predict(Path(args.predict), strategies[0], args.id_column,
                args.smiles_column, args.target_column, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
