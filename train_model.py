#!/usr/bin/env python
"""Обучение модели SI. Запускать на сильной машине, не на ноутбуке сборщика данных.

Скрипт самодостаточен: берёт два готовых csv, обучает, печатает метрики с
доверительными интервалами и сохраняет модель. Сети не требует.

    python -m src.pipeline export        # это уже сделано, файлы в репозитории
    python train_model.py                # обе стратегии, сравнение
    python train_model.py --strategy direct --repeats 5
    python train_model.py --predict data/raw/kaggle/test.csv --id-column index --smiles-column SMILES --target-column "SI VACV"

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
from src.pipeline import load_config, save_submission
from src.run_record import record_run

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


def labels_of(frame: pd.DataFrame) -> np.ndarray:
    """Строки, где цель задана меткой организаторов, а не измерена."""
    if "target_is_label" in frame.columns:
        return frame["target_is_label"].fillna(False).to_numpy(dtype=bool)
    return np.zeros(len(frame), dtype=bool)


def fit(strategy: str, features: np.ndarray, frame: pd.DataFrame,
        params: dict | None = None, clip_si: float | None = None):
    model = baseline.build(strategy, params, clip_si)
    if strategy == "two_head":
        model.fit(features, frame["IC50"].to_numpy(), frame["CC50"].to_numpy())
    elif strategy.startswith("sentinel"):
        model.fit(features, frame["SI"].to_numpy(), labels_of(frame))
    else:
        model.fit(features, frame["SI"].to_numpy())
    return model


def score(model, features: np.ndarray, frame: pd.DataFrame) -> dict:
    true = baseline.to_log(frame["SI"].to_numpy())
    predicted = model.predict_log_si(features)
    result = evaluate.metrics(true, predicted)
    low, high = evaluate.bootstrap_ci(true, predicted)
    result["rmse_log10_ci95"] = [low, high]
    # Метрика соревнования. На неё и смотрим: лидерборд считает MSE по сырому
    # SI, а не по логарифму.
    result["competition_mse"] = evaluate.competition_mse(true, predicted)

    # Наша валидация содержит хвост ChEMBL с SI до 2.2e5, которого в тесте нет:
    # на ней MSE была 7e7 при реальных 2.1e6 на лидерборде. Поэтому отдельно
    # считаем метрику только на строках из train.csv соревнования — они в том
    # же распределении, что тест, и только эта цифра сопоставима с лидербордом.
    if "source" in frame.columns:
        mask = frame["source"].fillna("").str.contains("kaggle_train").to_numpy()
        if mask.sum() >= 10:
            result["competition_mse_in_distribution"] = evaluate.competition_mse(
                true[mask], predicted[mask]
            )
            result["n_in_distribution"] = int(mask.sum())
    return result


def run_fixed(strategies: list[str], params: dict | None,
              clip_si: float | None = None) -> dict[str, dict]:
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
        model = fit(strategy, features_train, train, params, clip_si)
        counts = getattr(model, "n_train", {})
        if counts:
            log("строк на голову: " + ", ".join(
                f"{k} {v}" for k, v in counts.items() if not k.endswith("_skipped")
            ) + "".join(
                f" (пропущено {k.replace('_skipped','')}: {v})"
                for k, v in counts.items() if k.endswith("_skipped") and v
            ), indent=2)
        log(f"обучено за {time.monotonic() - started:.1f} с, считаю метрики с бутстрапом…", indent=2)

        result = score(model, features_validation, validation)
        report[strategy] = result
        low, high = result["rmse_log10_ci95"]
        log(f"{strategy}: RMSE log10 = {result['rmse_log10']:.3f} [{low:.3f}; {high:.3f}], "
            f"Spearman = {result['spearman']:.3f}", indent=2)
        log(f"метрика соревнования (MSE по сырому SI): {result['competition_mse']:.4g}", indent=2)
        if "competition_mse_in_distribution" in result:
            log(f"она же только на строках соревнования "
                f"(n={result['n_in_distribution']}): "
                f"{result['competition_mse_in_distribution']:.4g}  <- сопоставимо с лидербордом",
                indent=2)

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


def run_repeats(strategies: list[str], repeats: int, params: dict | None,
                clip_si: float | None = None) -> dict[str, dict]:
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
            model = fit(strategy, features[train_index], combined.iloc[train_index], params, clip_si)
            result = score(model, features[validation_index], combined.iloc[validation_index])
            collected[strategy].append(result)
            log(f"{strategy:<9} RMSE {result['rmse_log10']:.3f}  "
                f"Spearman {result['spearman']:.3f}  "
                f"MSE_соревнования {result['competition_mse']:.4g}  "
                f"({time.monotonic() - started:.1f} с)", indent=2)

    log("")
    log("среднее по разбиениям:")
    report: dict[str, dict] = {}
    for strategy, results in collected.items():
        rmse = np.array([r["rmse_log10"] for r in results])
        spearman = np.array([r["spearman"] for r in results])
        competition = np.array([r["competition_mse"] for r in results])
        in_dist = np.array([
            r["competition_mse_in_distribution"] for r in results
            if "competition_mse_in_distribution" in r
        ])
        report[strategy] = {
            "rmse_log10_mean": float(rmse.mean()),
            "rmse_log10_std": float(rmse.std(ddof=1)) if len(rmse) > 1 else 0.0,
            "spearman_mean": float(spearman.mean()),
            "spearman_std": float(spearman.std(ddof=1)) if len(spearman) > 1 else 0.0,
            "competition_mse_mean": float(competition.mean()),
            "competition_mse_in_distribution_mean": float(in_dist.mean()) if in_dist.size else None,
            "competition_mse_in_distribution_std": (
                float(in_dist.std(ddof=1)) if in_dist.size > 1 else 0.0
            ),
            "competition_mse_std": float(competition.std(ddof=1)) if len(competition) > 1 else 0.0,
            "n_splits": len(results),
        }
        log(f"{strategy:<9} RMSE {rmse.mean():.3f} ± {report[strategy]['rmse_log10_std']:.3f}   "
            f"Spearman {spearman.mean():.3f} ± {report[strategy]['spearman_std']:.3f}   "
            f"MSE_соревнования {competition.mean():.4g}" +
            (f"   в распределении {in_dist.mean():.4g}" if in_dist.size else ""), indent=2)

    if len(strategies) > 1:
        def ranking_key(name: str) -> float:
            value = report[name].get("competition_mse_in_distribution_mean")
            return value if value is not None else report[name]["rmse_log10_mean"]

        best = min(report, key=ranking_key)
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

    if not path.is_file():
        raise SystemExit(
            f"нет тестового файла {path}; скачайте его командой `python -m src.pipeline kaggle`"
        )
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
    save_submission(test, predicted, id_column, target_column, out)
    log(f"предсказания: {len(test)} строк → {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strategy",
                        choices=["sentinel", "sentinel_raw", "raw", "direct",
                                 "two_head", "both", "all"],
                        default="sentinel")
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
    parser.add_argument("--no-clip", action="store_true",
                        help="не ограничивать прогноз максимумом SI из данных организаторов")
    args = parser.parse_args(argv)

    if args.strategy == "both":
        strategies = ["two_head", "direct"]
    elif args.strategy == "all":
        strategies = ["sentinel_raw", "sentinel", "raw", "direct", "two_head"]
    else:
        strategies = [args.strategy]
    params_path = Path(args.params) if args.params else None
    params = load_config(params_path) if params_path else None
    config_path = ROOT / "configs" / "baseline.json"
    config = load_config(config_path)
    run_config = {**config, "run_options": vars(args), "params": params}
    inputs = [TRAIN_CSV, VALIDATION_CSV, config_path]
    inputs.extend(Path(value) for value in (args.params, args.predict) if value)
    command = ["python", "train_model.py", *(argv if argv is not None else sys.argv[1:])]
    with record_run(command, run_config, inputs):
        clip_si = None if args.no_clip else config["model"].get("clip_si")
        log(f"ограничение прогноза: {'нет' if clip_si is None else f'SI <= {clip_si:g}'}")

        report = (run_repeats(strategies, args.repeats, params, clip_si) if args.repeats > 0
                  else run_fixed(strategies, params, clip_si))

        # После сравнения по разбиениям моделей на диске не остаётся, а отправлять
        # нужно именно модель. Поэтому лучшую стратегию доучиваем на train.csv и
        # сохраняем — один прогон даёт и сравнение, и готовый артефакт.
        if args.repeats > 0:
            best = min(
                report,
                key=lambda name: (
                    report[name].get("competition_mse_in_distribution_mean")
                    if report[name].get("competition_mse_in_distribution_mean") is not None
                    else report[name]["rmse_log10_mean"]
                ),
            )
            log(f"доучиваю лучшую стратегию ({best}) на train.csv и сохраняю…")
            train = load(TRAIN_CSV)
            model = fit(best, featurize(train), train, params, clip_si)
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            import joblib

            joblib.dump({"model": model, "strategy": best}, ARTIFACTS / f"model_{best}.joblib")
            joblib.dump({"model": model, "strategy": best}, ARTIFACTS / "model.joblib")
            log(f"модель → artifacts/model_{best}.joblib (и model.joblib)", indent=2)
            report["best_strategy"] = best

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
