"""Four ordered architectural ablations; only original TRAIN targets enter OOF selection."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src import xgb_workflow as workflow, splits
from src.architecture_features import representations
from src.data.quality_weights import row_weights
from src.data.standardize import standardize_smiles
from src.features.fingerprints import morgan_matrix
from src.hypotheses import aggregate_oof
from src.local_compare import fold_indices
from src.local_score import score
from src.models.architecture_heads import FrozenHeads
from src.models.joint_gp import JointGP
from src.models.xgb_two_head import exact_target

ROOT = Path(__file__).resolve().parents[1]


def candidates_for(method):
    if method in ("chemeleon", "unimol"):
        candidates = {"ridge_10": {"kind": "ridge", "alpha": 10.}, "ridge_100": {"kind": "ridge", "alpha": 100.},
                      "mlp": {"kind": "mlp"}}
        if method == "chemeleon":
            candidates["random_control"] = {"kind": "mlp", "encoder": "random_chemeleon", "control_only": True}
        return candidates
    if method == "joint_gp":
        return {"median": {}, "mean": {}}
    return {"domain": {"kind": "domain"}, "pooled_control": {"kind": "mlp", "control_only": True},
            "kaggle_only_control": {"kind": "kaggle_only", "control_only": True}}


def choose_candidate(protocols, candidates):
    eligible = [key for key, settings in candidates.items() if not settings.get("control_only")]
    return min(eligible, key=lambda key: sum(protocols[p][key]["measured"]["mse"] /
        max(protocols[p]["baseline"]["measured"]["mse"], 1e-12) for p in ("molecule", "scaffold")))


def inputs():
    parts = workflow.load_parts()
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    reference = pd.read_csv(ROOT / "data/processed/kaggle_reference.csv")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    quality = pd.read_csv(ROOT / "data/processed/public_quality.csv")
    weights = row_weights(frame, quality)
    targets = np.column_stack([exact_target(frame, name) for name in ("IC50", "CC50")])
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv", usecols=["index", "SMILES"])
    if test["index"].isna().any() or test["index"].duplicated().any():
        raise ValueError("Invalid test IDs")
    smiles = frame.standard_smiles.tolist()+[standardize_smiles(value) for value in test.SMILES]
    if any(s is None for s in smiles):
        raise ValueError("Invalid molecular structure")
    return frame, reference, weights, targets, test, smiles


def matrix_for(method, smiles, config):
    if method == "joint_gp":
        return {"joint_gp": morgan_matrix(smiles)}, {"joint_gp": {"Morgan_radius": 2, "bits": 2048}}
    encoder = "chemeleon" if method == "domain_heads" else method
    names = [encoder]+(["random_chemeleon"] if method == "chemeleon" else [])
    matrices, metadata = {}, {}
    for name in names:
        matrices[name], metadata[name] = representations(smiles, name, config["encoder_batch_size"])
    return matrices, metadata


def make_model(method, settings, config):
    if method == "joint_gp":
        return JointGP(**config["gp"])
    return FrozenHeads(settings["kind"], settings.get("alpha", 10.), config["neural"], config["seed"])


def run(method, config):
    started = time.monotonic()
    directory = ROOT / "reports/architectures" / method
    directory.mkdir(parents=True, exist_ok=True)
    frame, reference, weights, targets, test, smiles = inputs()
    matrices, representation_metadata = matrix_for(method, smiles, config)
    default_encoder = "chemeleon" if method == "domain_heads" else method
    source = workflow.competition_rows(frame)
    baseline_oof = pd.read_csv(ROOT / config["baseline_oof"])
    candidates = candidates_for(method)
    output, manifests, audits = [], [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(config["repeats"]):
            fold_map = fold_indices(frame.standard_smiles.tolist(), protocol, config["folds"], config["seed"]+repeat)
            manifests.append({"protocol": protocol, "repeat": repeat, "indices": fold_map})
            for fold, indices in fold_map.items():
                test_idx = np.asarray(indices, dtype=int)
                train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
                rows = reference[reference.standard_smiles.isin(frame.iloc[test_idx].standard_smiles)].copy().rename(columns={"SI": "SI_true"})
                positions = rows.standard_smiles.map(pd.Series(np.arange(len(test_idx)), index=frame.iloc[test_idx].standard_smiles)).to_numpy(int)
                prior = baseline_oof[(baseline_oof.protocol == protocol) & baseline_oof.repeat.eq(repeat) & baseline_oof.fold.eq(int(fold))]
                if set(prior.original_row) != set(rows.original_row):
                    raise ValueError("Baseline fold differs from architecture fold")
                rows["baseline"] = rows.original_row.map(prior.set_index("original_row").weighted)
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, fold
                rows["scaffold"] = rows.standard_smiles.map(splits.murcko_scaffold)
                print(f"[architecture] {method}: {protocol} repeat={repeat+1}/{config['repeats']} fold={int(fold)+1}/{config['folds']}", flush=True)
                gp_parts = None
                for candidate, settings in candidates.items():
                    matrix = matrices[settings.get("encoder", default_encoder)]
                    fit_started = time.monotonic()
                    if method == "joint_gp" and gp_parts is not None:
                        prediction = gp_parts[candidate]
                    else:
                        model = make_model(method, settings, config).fit(matrix[train_idx], targets[train_idx], weights[train_idx], source[train_idx])
                        if method == "joint_gp":
                            gp_parts = model.predict_components(matrix[test_idx])
                            prediction = gp_parts[candidate]
                        else:
                            prediction = model.predict_si(matrix[test_idx])
                        audits.append({"protocol": protocol, "repeat": repeat, "fold": fold, "candidate": candidate,
                            "targets": model.n_train, "optimization": getattr(model, "optimization", None)})
                    rows[candidate] = prediction[positions]
                    measured = rows.target_is_label.astype("string").str.lower().eq("false").fillna(False)
                    print(f"[architecture] {candidate}: measured MSE={score(rows.loc[measured,'SI_true'], rows.loc[measured,candidate])['mse']:.4f}; "
                          f"all MSE={score(rows.SI_true,rows[candidate])['mse']:.4f}; {time.monotonic()-fit_started:.1f}s", flush=True)
                output.append(rows)
    oof = pd.concat(output, ignore_index=True)
    for _, rows in oof.groupby("protocol"):
        if not rows.groupby("original_row").size().eq(config["repeats"]).all() or len(rows) != len(reference)*config["repeats"]:
            raise AssertionError("OOF coverage invalid")
    protocols = aggregate_oof(oof, candidates)
    selected = choose_candidate(protocols, candidates)
    settings = candidates[selected]
    matrix = matrices[settings.get("encoder", default_encoder)]
    model = make_model(method, settings, config).fit(matrix[:len(frame)], targets, weights, source)
    if method == "joint_gp":
        model.prediction_kind = selected
    artifacts = ROOT / "artifacts/architectures"
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / f"{method}.joblib"
    bundle = {"model": model, "method": method, "encoder": settings.get("encoder", default_encoder),
        "selected": selected, "config": config, "representations": representation_metadata}
    joblib.dump(bundle, path, compress=3)
    model = joblib.load(path)["model"]
    prediction = model.predict_si(matrix[len(frame):])
    if len(prediction) != len(test) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid final prediction")
    submission = directory / "submission.csv"
    pd.DataFrame({"index": test["index"], "SI VACV": prediction}).to_csv(submission, index=False)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "rdkit", "torch", "chemprop", "unimol-tools")}
    report = {"method": method, "status": "complete", "config": config, "candidates": candidates,
        "selected": selected, "selection_metric": config["selection"], "selection_used_test_endpoints": False,
        "protocols": protocols, "fit_audits": audits, "final_targets": model.n_train,
        "optimization": getattr(model, "optimization", None), "development_rows": len(frame),
        "original_training_reference_rows": len(reference), "holdout_used": False,
        "data_sha256": workflow.data_hashes({"model": {"strategy": "xgb_weighted"}}),
        "versions": versions, "representations": representation_metadata,
        "submission": submission.relative_to(ROOT).as_posix(), "artifact": path.relative_to(ROOT).as_posix(),
        "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic()-started}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "splits.json").write_text(json.dumps({"development_smiles": frame.standard_smiles.tolist(), "splits": manifests}, indent=2), encoding="utf-8")
    oof.to_csv(directory / "oof.csv", index=False)
    print(f"[architecture done] {method}: selected={selected}; seconds={report['elapsed_seconds']:.1f}", flush=True)


def predict(artifact, test_csv, output):
    bundle = joblib.load(ROOT / artifact)
    test = pd.read_csv(ROOT / test_csv, usecols=["index", "SMILES"])
    if test["index"].isna().any() or test["index"].duplicated().any():
        raise ValueError("Invalid inference IDs")
    smiles = [standardize_smiles(s) for s in test.SMILES]
    if any(s is None for s in smiles):
        raise ValueError("Invalid inference SMILES")
    if bundle["method"] == "joint_gp":
        matrix = morgan_matrix(smiles)
    else:
        matrix, metadata = representations(smiles, bundle["encoder"], bundle["config"]["encoder_batch_size"])
        expected = bundle["representations"][bundle["encoder"]]
        def hashes(meta):
            checkpoints = [meta["checkpoint"]] if "checkpoint" in meta else meta["checkpoints"]
            return [record["sha256"] for record in checkpoints]
        if hashes(metadata) != hashes(expected):
            raise ValueError("Encoder weights differ from the saved model; restore the checkpoint URLs/revision in its report")
    prediction = bundle["model"].predict_si(matrix)
    if len(prediction) != len(test) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid inference predictions")
    path = ROOT / output
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": test["index"], "SI VACV": prediction}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/architectures.json")
    parser.add_argument("--method", choices=["chemeleon", "joint_gp", "unimol", "domain_heads", "all"], default="all")
    parser.add_argument("--predict-artifact")
    parser.add_argument("--test", default="data/raw/kaggle/test.csv")
    parser.add_argument("--out", default="reports/architecture_prediction.csv")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.predict_artifact:
            predict(args.predict_artifact, args.test, args.out)
            return
        config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
        for method in config["order"] if args.method == "all" else [args.method]:
            run(method, config)


if __name__ == "__main__":
    main()
