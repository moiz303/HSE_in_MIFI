"""Единая точка входа. От закоммиченных данных до submission.csv.

Владелец: D.

Воспроизводимость проверяют именно здесь: клон репозитория, одна команда,
csv на выходе. В сеть ходят только шаги data и kaggle; all использует
закоммиченный dataset.csv и уже скачанный файл соревнования test.csv.

    python -m src.pipeline data        # выгрузка и очистка          (A)
    python -m src.pipeline splits      # scaffold-разбиение          (B)
    python -m src.pipeline train       # обучение по конфигу         (B C)
    python -m src.pipeline evaluate    # метрики на validation       (B D)
    python -m src.pipeline submission  # предсказания для Kaggle     (D)
    python -m src.pipeline all         # splits, train, evaluate, submission
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
from src.run_record import record_run

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DATASET = PROCESSED / "dataset.csv"
MANIFEST = PROCESSED / "splits.json"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
DEFAULT_CONFIG = ROOT / "configs" / "baseline.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"нет конфигурации {path} — проверьте --config")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"ошибка в конфигурации {path}: {error}") from error


def load_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        raise SystemExit(
            f"нет {DATASET.relative_to(ROOT)} — сначала `python -m src.pipeline data` (владелец A)"
        )
    return pd.read_csv(DATASET)


def require_file(path: Path, instruction: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"нет {path.relative_to(ROOT)} — {instruction}")
    return path


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame,
                        id_column: str, target_column: str) -> None:
    expected = [id_column, target_column]
    if list(submission.columns) != expected:
        raise SystemExit(f"ошибка сабмита: нужны колонки {expected}, получены {list(submission.columns)}")
    if len(submission) != len(test):
        raise SystemExit(f"ошибка сабмита: нужно {len(test)} строк, получено {len(submission)}")
    predictions = pd.to_numeric(submission[target_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise SystemExit("ошибка сабмита: есть пустые или нечисловые прогнозы")


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
    require_file(MANIFEST, "сначала запустите `python -m src.pipeline splits`")
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
    require_file(MANIFEST, "сначала запустите `python -m src.pipeline splits`")
    split = splits_module.load_manifest(MANIFEST)
    features = _features(dataset, config)
    train_index = split["train"]

    strategy = config["model"]["strategy"]
    model = baseline.build(
        strategy, config["model"].get("params"), config["model"].get("clip_si")
    )
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
    require_file(MANIFEST, "сначала запустите `python -m src.pipeline splits`")
    split = splits_module.load_manifest(MANIFEST)
    features = _features(dataset, config)
    require_file(ARTIFACTS / "model.joblib", "сначала запустите `python -m src.pipeline train`")
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


def step_kaggle(config: dict) -> None:
    """Скачать данные соревнования."""
    from src.data import kaggle_data

    directory = kaggle_data.download(
        config["submission"].get("competition", kaggle_data.COMPETITION)
    )
    train = kaggle_data.load("train")
    print(f"[kaggle] файлы в {directory}, train: {len(train)} строк")


def step_submission(config: dict) -> None:
    """Собрать файл для Kaggle: предсказание модели по SMILES.

    Организаторы запретили выводить целевую переменную из колонок тестового
    файла. В test.csv лежат CC50_uM и IC50_uM_VACV, то есть SI считался бы
    делением без всякой модели — этот путь закрыт и в коде отсутствует.

    Поэтому здесь используется только структура молекулы. Колонки CC50_uM и
    IC50_uM_VACV из теста не читаются вообще, включая соглашения организаторов
    о метках пропусков: они тоже опираются на эти колонки.

    SI = CC50 / IC50 остаётся законным при сборке ОБУЧАЮЩИХ данных из
    литературы и баз (см. schema.selectivity_index) — запрет про тестовый файл.
    """
    import joblib

    settings = config["submission"]
    test_path = ROOT / settings["test_csv"]
    if not test_path.exists():
        raise SystemExit(f"нет файла теста: {test_path}. Запустите: python -m src.pipeline kaggle")

    test = pd.read_csv(test_path)

    required = (settings["id_column"], settings["smiles_column"])
    missing = [column for column in required if column not in test.columns]
    if missing:
        raise SystemExit(
            f"в {test_path.relative_to(ROOT)} нет колонок: {', '.join(missing)}; "
            f"ожидались {', '.join(required)}"
        )

    strategy = config["model"]["strategy"]
    # submission.model_file задаёт файл напрямую: модели из sweep.py называются
    # по месту в рейтинге, а не по стратегии.
    explicit = settings.get("model_file")
    candidates = (
        [ROOT / explicit] if explicit
        else [ARTIFACTS / f"model_{strategy}.joblib", ARTIFACTS / "model.joblib"]
    )
    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        raise SystemExit(
            f"нет обученной модели ({', '.join(path.name for path in candidates)}). "
            "Сначала запустите `python -m src.pipeline train` или `python -m src.pipeline all`."
        )
    bundle = joblib.load(model_path)
    model = bundle["model"]
    description = bundle.get("описание") or bundle.get("strategy") or strategy
    print(f"[submission] модель: {model_path.name} — {description}")

    features = fingerprints.featurize(
        test[settings["smiles_column"]].astype(str).tolist(),
        use_morgan=config["features"]["morgan"],
        use_descriptors=config["features"]["descriptors"],
    )
    predicted = baseline.from_log(model.predict_log_si(features))
    if len(predicted) != len(test):
        raise SystemExit(
            f"ошибка сабмита: тест содержит {len(test)} строк, модель вернула {len(predicted)} прогнозов"
        )

    out = pd.DataFrame({
        settings["id_column"]: test[settings["id_column"]],
        settings["target_column"]: predicted,
    })
    validate_submission(test, out, settings["id_column"], settings["target_column"])
    target = ROOT / settings["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    validate_submission(test, pd.read_csv(target), settings["id_column"], settings["target_column"])
    print(f"[submission] {len(out)} строк предсказано по SMILES → {target.relative_to(ROOT)}")


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

    config_path = Path(args.config)
    config = load_config(config_path)
    order = ["splits", "train", "evaluate", "submission"] if args.step == "all" else [args.step]
    inputs = [config_path]
    if args.step in ("splits", "train", "evaluate", "all"):
        inputs.append(DATASET)
    if args.step in ("train", "evaluate"):
        inputs.append(MANIFEST)
    if args.step == "evaluate":
        inputs.append(ARTIFACTS / "model.joblib")
    if args.step in ("submission", "all"):
        inputs.extend([
            ROOT / config["submission"]["test_csv"],
            ARTIFACTS / f"model_{config['model']['strategy']}.joblib",
            ARTIFACTS / "model.joblib",
        ])
    command = ["python", "-m", "src.pipeline", *(argv if argv is not None else sys.argv[1:])]
    with record_run(command, config, inputs):
        for name in order:
            STEPS[name](config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
