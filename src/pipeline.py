"""Единая точка входа. От закоммиченных данных до submission.csv.

Владелец: D.

Воспроизводимость проверяют именно здесь: клон репозитория, одна команда,
csv на выходе. Поэтому ни один шаг не ходит в сеть, кроме шага data, и ни
один не зависит от того, что лежит в чьей-то локальной папке.

    python -m src.pipeline data        # выгрузка и очистка          (A)
    python -m src.pipeline splits      # scaffold-разбиение          (B)
    python -m src.pipeline train       # обучение по конфигу         (B C)
    python -m src.pipeline evaluate    # метрики на validation       (B D)
    python -m src.pipeline submission  # предсказания для Kaggle     (D)
    python -m src.pipeline all         # всё, кроме data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import splits as splits_module
from src.features import fingerprints
from src.models import baseline

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DATASET = PROCESSED / "dataset.csv"
MANIFEST = PROCESSED / "splits.json"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
DEFAULT_CONFIG = ROOT / "configs" / "baseline.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        raise SystemExit(
            f"нет {DATASET.relative_to(ROOT)} — сначала `python -m src.pipeline data` (владелец A)"
        )
    return pd.read_csv(DATASET)


def step_data(config: dict) -> None:
    """Выгрузка из открытых баз, валидация, стандартизация, агрегация."""
    from src.data import download, schema, standardize

    download.install_cache()
    frames: list[pd.DataFrame] = []
    for name in config["data"]["sources"]:
        print(f"[data] источник: {name}")
        frames.append(download.SOURCES[name](config["data"]))

    raw = pd.concat(frames, ignore_index=True)
    accepted, rejected_schema = schema.validate_table(raw)
    accepted, rejected_parse = standardize.add_standard_smiles(accepted)

    # Цензурированные значения ("IC50 > 100") всегда попадают в сдаваемую базу,
    # но для обучения их можно исключить: из такого измерения следует лишь
    # граница SI. Переключатель замеряется на validation, а не выбирается верой.
    rejected_censored = pd.DataFrame()
    if config["data"].get("censored", "keep") == "drop":
        qualifiers = ("IC50_qualifier", "CC50_qualifier")
        present = [column for column in qualifiers if column in accepted.columns]
        if present:
            censored = accepted[present].fillna("").astype(str).ne("").any(axis=1)
            rejected_censored = accepted[censored].copy()
            rejected_censored["reason"] = "цензурированное значение, data.censored=drop"
            accepted = accepted[~censored].reset_index(drop=True)

    dataset = standardize.aggregate_duplicates(accepted)

    # Журналы отказов самих выгрузчиков: отказ по единицам или отсутствию пары
    # случается до валидации, и без этого шага он потерялся бы.
    journals = [rejected_schema, rejected_parse, rejected_censored]
    for path in sorted((ROOT / "data" / "interim").glob("*_rejected.csv")):
        journal = pd.read_csv(path)
        if not journal.empty:
            journals.append(journal)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET, index=False)
    rejected = pd.concat(journals, ignore_index=True)
    rejected.to_csv(PROCESSED / "rejected.csv", index=False)
    print(f"[data] принято {len(dataset)}, отбраковано {len(rejected)}")
    if "reason" in rejected.columns and not rejected.empty:
        for reason, count in rejected["reason"].value_counts().head(8).items():
            print(f"         {count:>6}  {reason}")


def step_splits(config: dict) -> None:
    dataset = load_dataset()
    column = "standard_smiles" if "standard_smiles" in dataset.columns else "smiles"
    smiles = dataset[column].astype(str).tolist()
    split = splits_module.scaffold_split(
        smiles,
        fractions=config["splits"]["fractions"],
        seed=config["splits"]["seed"],
    )
    splits_module.assert_no_leaks(smiles, split)
    splits_module.save_manifest(MANIFEST, split, config["splits"]["seed"], len(smiles))
    print("[splits] " + ", ".join(f"{name}={len(idx)}" for name, idx in split.items()))


def _features(dataset: pd.DataFrame, config: dict) -> np.ndarray:
    column = "standard_smiles" if "standard_smiles" in dataset.columns else "smiles"
    return fingerprints.featurize(
        dataset[column].astype(str).tolist(),
        use_morgan=config["features"]["morgan"],
        use_descriptors=config["features"]["descriptors"],
    )


def step_train(config: dict) -> None:
    import joblib

    dataset = load_dataset()
    split = splits_module.load_manifest(MANIFEST)
    features = _features(dataset, config)
    train_index = split["train"]

    strategy = config["model"]["strategy"]
    model = baseline.build(strategy, config["model"].get("params"))
    if strategy == "two_head":
        model.fit(
            features[train_index],
            dataset["IC50"].to_numpy()[train_index],
            dataset["CC50"].to_numpy()[train_index],
        )
    else:
        model.fit(features[train_index], dataset["SI"].to_numpy()[train_index])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "config": config}, ARTIFACTS / "model.joblib")
    print(f"[train] стратегия {strategy}, обучено на {len(train_index)} молекулах")


def step_evaluate(config: dict) -> None:
    import joblib

    from src import evaluate as evaluate_module

    dataset = load_dataset()
    split = splits_module.load_manifest(MANIFEST)
    features = _features(dataset, config)
    model = joblib.load(ARTIFACTS / "model.joblib")["model"]

    report: dict[str, dict] = {}
    for part in ("train", "validation"):
        index = split[part]
        true = baseline.to_log(dataset["SI"].to_numpy()[index])
        pred = model.predict_log_si(features[index])
        values = evaluate_module.metrics(true, pred)
        low, high = evaluate_module.bootstrap_ci(true, pred)
        values["rmse_log10_ci95"] = [low, high]
        report[part] = values
        print(f"[evaluate] {part}: rmse_log10={values['rmse_log10']:.3f} "
              f"[{low:.3f}; {high:.3f}], spearman={values['spearman']:.3f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def step_submission(config: dict) -> None:
    import joblib

    test_path = Path(config["submission"]["test_csv"])
    if not test_path.exists():
        raise SystemExit(f"нет файла теста Kaggle: {test_path}")

    test = pd.read_csv(test_path)
    model = joblib.load(ARTIFACTS / "model.joblib")["model"]
    features = fingerprints.featurize(
        test[config["submission"]["smiles_column"]].astype(str).tolist(),
        use_morgan=config["features"]["morgan"],
        use_descriptors=config["features"]["descriptors"],
    )
    predicted = baseline.from_log(model.predict_log_si(features))

    out = pd.DataFrame({
        config["submission"]["id_column"]: test[config["submission"]["id_column"]],
        config["submission"]["target_column"]: predicted,
    })
    target = ROOT / config["submission"]["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(f"[submission] {len(out)} строк → {target.relative_to(ROOT)}")


STEPS = {
    "data": step_data,
    "splits": step_splits,
    "train": step_train,
    "evaluate": step_evaluate,
    "submission": step_submission,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Пайплайн VACV SI")
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    order = ["splits", "train", "evaluate", "submission"] if args.step == "all" else [args.step]
    for name in order:
        STEPS[name](config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
