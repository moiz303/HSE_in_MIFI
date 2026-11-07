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

import pandas as pd

from src import splits as splits_module
from src import xgb_workflow

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DATASET = PROCESSED / "dataset.csv"
MANIFEST = PROCESSED / "splits.json"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
DEFAULT_CONFIG = ROOT / "configs" / "xgb_hybrid.json"


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
    # граница SI. Режим замеряется на validation, а не выбирается верой.
    #
    #   keep           — всё как есть (решение команды)
    #   drop_artifacts — только строки, где SI = 1 возник из-за того, что оба
    #                    значения цензурированы и равны: это артефакт протокола
    #   drop           — вся цензура
    mode = config["data"].get("censored", "keep")
    rejected_censored = pd.DataFrame()

    if mode == "drop_artifacts" and "si_artifact" in accepted.columns:
        artifact = accepted["si_artifact"].fillna(False).astype(bool)
        rejected_censored = accepted[artifact].copy()
        rejected_censored["reason"] = "артефакт SI=1, data.censored=drop_artifacts"
        accepted = accepted[~artifact].reset_index(drop=True)
    elif mode == "drop":
        present = [c for c in ("IC50_qualifier", "CC50_qualifier") if c in accepted.columns]
        if present:
            censored = accepted[present].fillna("").astype(str).ne("").any(axis=1)
            rejected_censored = accepted[censored].copy()
            rejected_censored["reason"] = "цензурированное значение, data.censored=drop"
            accepted = accepted[~censored].reset_index(drop=True)
    elif mode not in ("keep", "drop_artifacts", "drop"):
        raise SystemExit(
            f"data.censored: ожидалось keep | drop_artifacts | drop, получено {mode!r}"
        )

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


def step_export(config: dict) -> None:
    """Выложить обучающую и проверочную выборки отдельными файлами.

    Обучение идёт не на машине, где собираются данные: тяжёлые прогоны делает
    тот, у кого сильный мак. Этот шаг готовит ему два самодостаточных csv.

    Holdout намеренно не экспортируется. Он остаётся только в манифесте
    разбиения и трогается один раз, на стадии 9: файл, лежащий рядом с
    обучающим, рано или поздно попадёт в обучение.
    """
    dataset = load_dataset()
    split = splits_module.load_manifest(MANIFEST)

    for part in ("train", "validation"):
        if part not in split:
            raise SystemExit(f"в манифесте нет части {part}: запустите splits")
        frame = dataset.iloc[split[part]].reset_index(drop=True)
        target = PROCESSED / f"{part}.csv"
        frame.to_csv(target, index=False)
        print(f"[export] {part}: {len(frame)} строк → {target.relative_to(ROOT)}")

    print(f"[export] holdout ({len(split.get('holdout', []))} строк) не выгружается "
          f"намеренно — он только в {MANIFEST.relative_to(ROOT)}")


def step_train(config: dict) -> None:
    xgb_workflow.train(config)


def step_evaluate(config: dict) -> None:
    xgb_workflow.evaluate(config)


def step_submission(config: dict) -> None:
    xgb_workflow.submission(config)


def step_kaggle(config: dict) -> None:
    from src.data import kaggle_data
    kaggle_data.download(config["submission"].get("competition", kaggle_data.COMPETITION))


STEPS = {
    "kaggle": step_kaggle,
    "data": step_data,
    "splits": step_splits,
    "export": step_export,
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
    order = ["train", "evaluate", "submission"] if args.step == "all" else [args.step]
    for name in order:
        STEPS[name](config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
