"""Воспроизводимый запуск XGBoost без переборов и без обращения к holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import splits
from src.data.standardize import standardize_smiles
from src.features.fingerprints import featurize
from src.models.xgb_two_head import XGBTwoHead, exact_target
from src.models.catboost_two_head import CatBoostTwoHead
from src.models.xgb_hybrid import XGBHybrid, select_hybrid
from src.models.xgb_ensemble import XGBEnsemble

ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPES = {"xgb_two_head": XGBTwoHead, "catboost_two_head": CatBoostTwoHead}
MODEL_TYPES["xgb_hybrid"] = XGBHybrid
MODEL_TYPES["xgb_ensemble"] = XGBEnsemble


def is_final(config):
    return config.get("training", {}).get("mode", "train_only") == "train_validation"


def report_prefix(config):
    if config["model"]["strategy"] == "xgb_ensemble":
        return "xgb_ensemble"
    if is_final(config):
        return "xgb_final"
    if cv_protocol(config):
        return "xgb_cv"
    if config["model"]["strategy"] == "xgb_hybrid":
        return "xgb_hybrid"
    return "catboost" if config["model"]["strategy"] == "catboost_two_head" else "xgb"


def validate_config(config):
    mode = config.get("training", {}).get("mode", "train_only")
    if mode not in ("train_only", "train_validation"):
        raise ValueError("training.mode must be train_only or train_validation; holdout fitting is forbidden")
    if is_final(config) and (config["model"]["strategy"] not in ("xgb_two_head", "xgb_ensemble") or cv_protocol(config)):
        raise ValueError("Final mode supports fixed concentration XGBoost heads or their seed ensemble")
    if config["model"]["strategy"] not in MODEL_TYPES:
        raise ValueError(f"неизвестная стратегия; доступны {tuple(MODEL_TYPES)}")
    if config["features"] != {"morgan": True, "descriptors": True}:
        raise ValueError("признаки зафиксированы: Morgan 2048 + 13 дескрипторов, без метаданных")


def features(frame, column="standard_smiles"):
    smiles = [standardize_smiles(value) for value in frame[column]]
    invalid = [i for i, value in enumerate(smiles) if value is None]
    if invalid:
        raise ValueError(f"невалидные SMILES, индексы строк: {invalid[:10]}")
    return featurize(smiles)


def competition_rows(frame):
    return frame["source"].fillna("").str.split(";").apply(lambda v: "kaggle_train" in v).to_numpy()


def load_parts():
    directory = ROOT / "data/processed"
    dataset = pd.read_csv(directory / "dataset.csv")
    manifest = splits.load_manifest(directory / "splits.json")
    if set(manifest) != {"train", "validation", "holdout"}:
        raise ValueError("неполный манифест разбиения")
    indices = [i for part in manifest.values() for i in part]
    if sorted(indices) != list(range(len(dataset))):
        raise ValueError("манифест содержит повторы или не соответствует датасету")
    smiles = [standardize_smiles(value) for value in dataset.standard_smiles]
    if any(value is None for value in smiles):
        raise ValueError("невалидный SMILES в датасете")
    splits.assert_no_leaks(smiles, manifest)
    parts = {}
    for name in ("train", "validation"):
        parts[name] = pd.read_csv(directory / f"{name}.csv")
        expected = dataset.iloc[manifest[name]].reset_index(drop=True)
        pd.testing.assert_frame_equal(parts[name], expected, check_dtype=False)
    # Проверка индексов/структур holdout не вычисляет его метрики и не обучает на нём.
    return parts


def metric(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    mask = np.isfinite(truth) & (truth > 0)
    if not mask.any():
        return {"n": 0, "mse": None, "rmse_log10": None}
    actual, forecast = truth[mask], predicted[mask]
    if not np.isfinite(forecast).all() or (forecast <= 0).any():
        raise ValueError("некорректные прогнозы для метрик")
    return {"n": int(mask.sum()), "mse": float(np.mean((actual - forecast) ** 2)),
            "rmse_log10": float(np.sqrt(np.mean((np.log10(actual) - np.log10(forecast)) ** 2)))}


def data_hashes():
    names = ["train.csv", "validation.csv", "splits.json"]
    if (ROOT / "data/processed/kaggle_reference.csv").exists():
        names.append("kaggle_reference.csv")
    return {name: hashlib.sha256((ROOT / "data/processed" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in names}


def cv_protocol(config):
    return config["model"].get("selection_protocol") == "train_scaffold_cv"


def reference_parts(parts, config):
    if not cv_protocol(config):
        return parts, None
    from src.data.kaggle_reference import restore_targets
    reference = pd.read_csv(ROOT / "data/processed/kaggle_reference.csv")
    return {name: restore_targets(frame, reference) for name, frame in parts.items()}, reference


def original_mse(frame, prediction, reference):
    from src.data.kaggle_reference import original_pairs
    truth, predicted = original_pairs(frame, prediction, reference)
    return {"n": len(truth), "mse": float(np.mean((truth-predicted)**2)) if len(truth) else None}


def train(config):
    validate_config(config)
    parts, reference = reference_parts(load_parts(), config)
    training_parts = ["train", "validation"] if is_final(config) else ["train"]
    frame = pd.concat([parts[name] for name in training_parts], ignore_index=True)
    settings = config["model"]
    model = MODEL_TYPES[settings["strategy"]](settings.get("params"), settings.get("clip_si"))
    print(f"[train] {settings['strategy']}: fitting={'+'.join(training_parts)}; rows={len(frame)}", flush=True)
    if is_final(config):
        print("[FINAL] Validation is included in training. No independent validation metrics will be reported. Holdout excluded.", flush=True)
    matrix = features(frame)
    print(f"[train] features={matrix.shape}; holdout unused", flush=True)
    cv_result = None
    if cv_protocol(config):
        from src.hybrid_cv import select_on_train
        cv_result = select_on_train(frame, matrix, reference, settings.get("params"))
        print("[CV result] " + json.dumps(cv_result), flush=True)
    model.fit(matrix, exact_target(frame, "IC50"), exact_target(frame, "CC50"))
    if isinstance(model, XGBHybrid):
        labels = np.full(len(frame), np.nan)
        pure = competition_rows(frame) if cv_result is not None else frame.source.eq("kaggle_train").to_numpy()
        flags = frame.target_is_label.astype("string").fillna("unknown").str.lower()
        known = flags.isin(["true", "false"]).to_numpy()
        labels[pure & known] = flags.eq("true").to_numpy(dtype=bool)[pure & known]
        model.fit_classifier(matrix, labels)
        if cv_result is not None:
            model.mixture_weight = cv_result["weight"]
    mask = competition_rows(frame)
    si = pd.to_numeric(frame.SI, errors="coerce").to_numpy(float)
    usable = mask & np.isfinite(si) & (si > 0)
    if not usable.any():
        raise ValueError("нет обучающих SI соревнования для константного сравнения")
    constant_si = float(si[usable].mean())
    if reference is not None:
        from src.data.kaggle_reference import original_pairs
        truth, _ = original_pairs(frame, np.zeros(len(frame)), reference)
        constant_si = float(truth.mean())
    import xgboost
    import importlib.metadata
    from rdkit import rdBase
    bundle = {"model": model, "config": config, "data_sha256": data_hashes(),
              "constant_si": constant_si,
              "versions": {"xgboost": xgboost.__version__, "rdkit": rdBase.rdkitVersion}}
    if cv_result is not None:
        bundle["train_cv"] = cv_result
    if is_final(config):
        bundle["training"] = {"parts": training_parts, "rows": len(frame),
            "source_rows": frame.source.value_counts().to_dict(),
            "n_train": model.n_train, "parameters": model.params,
            "independent_evaluation": False, "holdout_used": False}
        if isinstance(model, XGBEnsemble):
            bundle["training"]["ensemble"] = {"seeds": model.seeds, "members": len(model.members),
                "aggregation": "arithmetic_mean_of_member_SI", "trees_per_head": model.params["n_estimators"]}
    if settings["strategy"] == "catboost_two_head":
        bundle["versions"]["catboost"] = importlib.metadata.version("catboost")
    path = ROOT / config["submission"]["model_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    if settings["strategy"] == "catboost_two_head":
        for name, head in model.heads.items():
            head.save_model(str(path.with_name(f"{path.stem}_{name}.cbm")))
    print(f"[train] {model.n_train} -> {path}", flush=True)
    if is_final(config):
        report_path = ROOT / f"reports/{report_prefix(config)}_training.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {"config": config, "training": bundle["training"],
                  "data_sha256": bundle["data_sha256"], "versions": bundle["versions"]}
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[FINAL] Training report -> {report_path}", flush=True)


def load_bundle(config):
    validate_config(config)
    path = ROOT / config["submission"]["model_file"]
    bundle = joblib.load(path)
    if type(bundle["model"]) is not MODEL_TYPES[config["model"]["strategy"]] or bundle["config"] != config:
        raise ValueError("артефакт не соответствует типу модели или конфигу; переобучите на Mac")
    return bundle


def evaluate(config):
    if is_final(config):
        raise ValueError("Final model trained on train+validation: independent validation evaluation is unavailable; holdout remains reserved")
    bundle = load_bundle(config)
    if bundle["data_sha256"] != data_hashes():
        raise ValueError("данные изменились после обучения")
    parts, reference = reference_parts(load_parts(), config)
    report = {"config": config, "data_sha256": bundle["data_sha256"],
              "versions": bundle["versions"], "n_train": bundle["model"].n_train,
              "note": "Fixed validation; no tuning, significance claim or holdout evaluation."}
    model = bundle["model"]
    if isinstance(model, XGBHybrid):
        validation = parts["validation"]
        matrix = features(validation)
        mask = competition_rows(validation)
        targets = validation.SI.to_numpy(float)
        base = metric(targets[mask], model.predict_baseline_si(matrix)[mask])["mse"]
        candidate = metric(targets[mask], model.predict_candidate_si(matrix)[mask])["mse"]
        if reference is not None:
            base = original_mse(validation, model.predict_baseline_si(matrix), reference)["mse"]
            candidate = original_mse(validation, model.predict_candidate_si(matrix), reference)["mse"]
        eligible = select_hybrid(base, candidate)
        model.use_classifier = eligible and config["model"].get("promote_candidate", False)
        bundle["selection"] = {"baseline_mse": base, "candidate_mse": candidate,
            "eligible_on_validation": bool(eligible),
            "promotion_enabled": bool(config["model"].get("promote_candidate", False)),
            "selected": "hybrid" if model.use_classifier else "baseline",
            "rule": "candidate MSE <= 0.98 * baseline MSE on fixed competition validation"}
        report["selection"] = bundle["selection"]
        if "train_cv" in bundle:
            report["train_cv"] = bundle["train_cv"]
            report["selection"]["weight"] = model.mixture_weight
            report["selection"]["targets"] = "original competition training rows, matched to fixed validation"
        report["note"] = "Validation used for two-candidate selection; no independent performance claim. Holdout unused."
        print("[selection] " + json.dumps(bundle["selection"]), flush=True)
    predictions = None
    for name, frame in parts.items():
        matrix = features(frame)
        logs = model.predict_log_concentrations(matrix)
        si = model.predict_si(matrix)
        mask = competition_rows(frame)
        targets = pd.to_numeric(frame.SI, errors="coerce").to_numpy(float)
        report[name] = {
            target: metric(exact_target(frame, target), 10.0 ** logs[target])
            for target in ("IC50", "CC50")
        }
        report[name]["SI_all"] = metric(targets, si)
        report[name]["SI_competition"] = metric(targets[mask], si[mask])
        report[name]["constant_competition"] = metric(
            targets[mask], np.full(mask.sum(), bundle["constant_si"]))
        if reference is not None:
            report[name]["SI_kaggle_original"] = original_mse(frame, si, reference)
            report[name]["constant_kaggle_original"] = original_mse(
                frame, np.full(len(frame), bundle["constant_si"]), reference)
        if name == "validation":
            predictions = pd.DataFrame({"standard_smiles": frame.standard_smiles,
                "IC50_pred_uM": 10.0 ** logs["IC50"], "CC50_pred_uM": 10.0 ** logs["CC50"],
                "SI_pred": si, "SI_true": targets, "competition": mask})
            if isinstance(model, XGBHybrid):
                predictions["SI_baseline"] = model.predict_baseline_si(matrix)
                predictions["SI_candidate"] = model.predict_candidate_si(matrix)
                predictions["label_probability"] = model.predict_label_probability(matrix)
    directory = ROOT / "reports"
    directory.mkdir(exist_ok=True)
    prefix = report_prefix(config)
    (directory / f"{prefix}_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    predictions.to_csv(directory / f"{prefix}_validation.csv", index=False)
    if isinstance(model, XGBHybrid):
        joblib.dump(bundle, ROOT / config["submission"]["model_file"])
    print(json.dumps(report["validation"], indent=2), flush=True)


def submission(config):
    settings = config["submission"]
    bundle = load_bundle(config)
    if isinstance(bundle["model"], XGBHybrid) and "selection" not in bundle:
        raise ValueError("Run evaluate before generating the hybrid submission")
    # Read only identifiers and SMILES: no concentrations or derived flags from test.
    frame = pd.read_csv(ROOT / settings["test_csv"], usecols=[settings["smiles_column"], settings["id_column"]])
    if frame[settings["id_column"]].isna().any() or frame[settings["id_column"]].duplicated().any():
        raise ValueError("пустые или повторяющиеся идентификаторы теста")
    matrix = features(frame, settings["smiles_column"])
    model = bundle["model"]
    members = model.predict_members_si(matrix) if isinstance(model, XGBEnsemble) else None
    predicted = members.mean(axis=1) if members is not None else model.predict_si(matrix)
    target = ROOT / settings["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({settings["id_column"]: frame[settings["id_column"]],
                  settings["target_column"]: predicted}).to_csv(target, index=False)
    if members is not None:
        diagnostic = pd.DataFrame(members, columns=[f"SI_seed_{seed}" for seed in model.seeds])
        diagnostic.insert(0, settings["id_column"], frame[settings["id_column"]].to_numpy())
        diagnostic["SI_mean"] = predicted
        diagnostic["SI_seed_std"] = members.std(axis=1)
        diagnostic.to_csv(target.with_name(target.stem + "_members.csv"), index=False)
        print(f"[ensemble] median SI={np.median(predicted):.6g}; "
              f"median seed spread={np.median(members.std(axis=1)):.6g}", flush=True)
    print(f"[submission] {len(frame)} rows -> {target}", flush=True)
