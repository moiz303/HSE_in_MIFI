"""Independent, reproducible development-OOF ablations. Never reads test endpoints."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from src import splits, xgb_workflow as workflow
from src.data.quality_weights import row_weights
from src.data.schema import parse_measurement
from src.data.standardize import standardize_smiles
from src.features.fingerprints import featurize
from src.local_compare import fold_indices
from src.local_score import score, paired_interval
from src.models.xgb_two_head import DEFAULT_PARAMS, exact_target

ROOT = Path(__file__).resolve().parents[1]
PARAMETER_PROFILES = {"default": {},
    "shallow": {"n_estimators": 800, "max_depth": 2, "min_child_weight": 10, "reg_lambda": 10.},
    "deeper": {"n_estimators": 800, "max_depth": 4, "min_child_weight": 10, "reg_lambda": 10.}}
HYPOTHESES = {
    "independent": {"expanded": {"expanded": True}},
    "features": {"counts": {"features": "counts"}, "radius3": {"features": "radius3"},
                 "feature_morgan": {"features": "feature_morgan"}},
    "censoring": {"aft_exact": {"aft": True}, "aft_censored": {"aft": True, "censored": True}},
    "direct_si": {f"mix_{alpha}": {"direct_alpha": alpha} for alpha in (.1, .25, .5, 1.)},
    "clean_pairs": {"strict": {"clean": True}},
    "head_params": {f"{ic}__{cc}": {"ic_params": ic, "cc_params": cc}
                    for ic, cc in itertools.product(PARAMETER_PROFILES, repeat=2)},
}


def feature_matrix(smiles, preset="default"):
    if preset == "counts":
        return featurize(smiles, counts=True)
    if preset == "radius3":
        return featurize(smiles, radius=3)
    result = featurize(smiles)
    if preset == "feature_morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
        result[:, :2048] = np.asarray([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles])
    elif preset != "default":
        raise ValueError(f"Unknown preset: {preset}")
    return result


def exclude_groups(extra, excluded_smiles, protocol):
    """Apply the same validation exclusion to every additional endpoint source."""
    forbidden = set(excluded_smiles)
    allowed = ~extra.standard_smiles.isin(forbidden)
    if protocol == "scaffold":
        scaffolds = {splits.murcko_scaffold(s) for s in forbidden}
        allowed &= ~extra.standard_smiles.map(splits.murcko_scaffold).isin(scaffolds)
    return extra.loc[allowed].copy()


def prepare_extra(dataset, holdout_smiles):
    raw = pd.read_csv(ROOT / "data/experiments/independent_endpoints.csv")
    # New structures only: avoid increasing the multiplicity of earlier paired exports.
    usable = raw.qualifier.fillna("").eq("") & ~raw.standard_smiles.isin(dataset.standard_smiles)
    usable &= (raw.endpoint.eq("IC50") & raw.measured_type.isin(["IC50", "EC50"])) | (
        raw.endpoint.eq("CC50") & raw.measured_type.isin(["CC50", "TC50"]))
    values = pd.to_numeric(raw.value_um, errors="coerce")
    usable &= np.isfinite(values) & values.gt(0)
    extra = exclude_groups(raw.loc[usable], holdout_smiles, "scaffold")
    extra["log_value"] = np.log10(extra.value_um)
    extra = extra.groupby(["standard_smiles", "endpoint"], sort=True).agg(
        log_value=("log_value", "median"), weight=("weight", "min"),
        records=("activity_id", "size"), document_id=("document_id", lambda v: ";".join(sorted(set(v))))).reset_index()
    extra["value_um"] = 10.**extra.pop("log_value")
    extra.to_csv(ROOT / "data/experiments/independent_training.csv", index=False)
    return extra


def strict_targets(frame, raw_pairs, reference):
    """Rebuild consistent public pairs; original competition endpoints take precedence."""
    raw = raw_pairs.copy()
    same = raw.cell_line_ic50.fillna("").ne("") & raw.cell_line_ic50.eq(raw.cell_line_cc50)
    same &= raw.cc50_cell_line_match.isin(["vaccinia_same_cell_line", "same_cell_line"])
    same &= raw.IC50_qualifier.fillna("").eq("") & raw.CC50_qualifier.fillna("").eq("")
    raw = raw.loc[same]
    targets = {}
    is_competition = workflow.competition_rows(frame)
    for name in ("IC50", "CC50"):
        public = raw.assign(log=np.log10(pd.to_numeric(raw[name], errors="coerce"))).groupby("standard_smiles").log.median()
        comp = reference.copy()
        # Reference concentrations are the original training values, not SI-imputed endpoints.
        values = pd.to_numeric(comp[name], errors="coerce")
        marks = comp[name+"_qualifier"].fillna("").eq("")
        comp = comp.loc[marks & np.isfinite(values) & values.gt(0)].copy()
        comp["log"] = np.log10(values.loc[comp.index])
        original = comp.groupby("standard_smiles").log.median()
        result = frame.standard_smiles.map(public).to_numpy(float).copy()
        result[is_competition] = frame.loc[is_competition, "standard_smiles"].map(original).to_numpy(float)
        targets[name] = 10.**result
    return targets


def endpoint_bounds(frame, name, include_censored):
    values = exact_target(frame, name)
    lower, upper = values.copy(), values.copy()
    if include_censored:
        qualifiers = frame[name+"_qualifier"].fillna("")
        numbers = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        # Aggregation can mix assays. Use bound labels only for unaggregated rows.
        single = pd.to_numeric(frame.n_measurements, errors="coerce").eq(1).to_numpy()
        valid = single & np.isfinite(numbers) & (numbers > 0)
        right = valid & qualifiers.isin([">", ">="]).to_numpy()
        left = valid & qualifiers.isin(["<", "<="]).to_numpy()
        lower[right], upper[right] = numbers[right], np.inf
        lower[left], upper[left] = 0., numbers[left]
    return lower, upper


def fit_booster(matrix, lower, upper, weights, params=None, aft=False):
    if aft:
        mask = ~np.isnan(lower) & ~np.isnan(upper) & (upper > 0) & (lower >= 0) & (lower <= upper)
    else:
        mask = np.isfinite(lower) & (lower > 0)
    if not mask.any():
        raise ValueError("No usable training endpoints")
    settings = {**DEFAULT_PARAMS, **(params or {})}
    rounds = settings.pop("n_estimators")
    settings["seed"] = settings.pop("random_state")
    settings["nthread"] = settings.pop("n_jobs")
    settings.update(tree_method="hist", device="cpu", objective="survival:aft" if aft else "reg:squarederror")
    if aft:
        settings.update(aft_loss_distribution="normal", aft_loss_distribution_scale=1.)
    w = np.asarray(weights, dtype=float)[mask]
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Invalid weights")
    train = xgb.DMatrix(matrix[mask], weight=w/w.mean())
    if aft:
        train.set_float_info("label_lower_bound", lower[mask])
        train.set_float_info("label_upper_bound", upper[mask])
    else:
        train.set_label(np.log10(lower[mask]))
    return xgb.train(settings, train, num_boost_round=rounds), {
        "n": int(mask.sum()), "censored": int((lower[mask] != upper[mask]).sum()), "parameters": settings, "rounds": rounds}


class HypothesisModel:
    """Portable native boosters; inference uses molecular structures only."""
    def __init__(self, settings):
        self.settings = settings.copy()
        self.heads, self.audit = {}, {}

    def __getstate__(self):
        return {**self.__dict__, "heads": {k: v.save_raw(raw_format="ubj") for k, v in self.heads.items()}}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.heads = {}
        for key, value in state["heads"].items():
            booster = xgb.Booster()
            booster.load_model(value)
            self.heads[key] = booster

    def fit(self, frame, matrix, reliability, reference, extra, extra_matrix, raw_pairs):
        clean = strict_targets(frame, raw_pairs, reference) if self.settings.get("clean") else None
        for name in ("IC50", "CC50"):
            if self.settings.get("aft"):
                lower, upper = endpoint_bounds(frame, name, self.settings.get("censored", False))
            else:
                lower = clean[name] if clean is not None else exact_target(frame, name)
                upper = lower.copy()
            x, weights = matrix, reliability
            added = 0
            if self.settings.get("expanded"):
                mask = extra.endpoint.eq(name).to_numpy()
                x = np.concatenate([matrix, extra_matrix[mask]])
                lower = np.concatenate([lower, extra.loc[mask, "value_um"].to_numpy(float)])
                upper = lower.copy()
                weights = np.concatenate([reliability, extra.loc[mask, "weight"].to_numpy(float)])
                added = int(mask.sum())
            profile = self.settings.get("ic_params" if name == "IC50" else "cc_params", "default")
            booster, audit = fit_booster(x, lower, upper, weights, PARAMETER_PROFILES[profile], self.settings.get("aft", False))
            audit["extra_rows"] = added
            self.heads[name], self.audit[name] = booster, audit
        if "direct_alpha" in self.settings:
            known = reference.target_is_label.astype("string").str.lower().eq("false").fillna(False)
            ref = reference.loc[known & reference.standard_smiles.isin(frame.standard_smiles)].copy()
            # Require actual paired exact concentrations, not just a numerical SI label.
            ic, cc = pd.to_numeric(ref.IC50, errors="coerce"), pd.to_numeric(ref.CC50, errors="coerce")
            valid = np.isfinite(ic) & ic.gt(0) & np.isfinite(cc) & cc.gt(0)
            valid &= ref.IC50_qualifier.fillna("").eq("") & ref.CC50_qualifier.fillna("").eq("")
            ref = ref.loc[valid]
            positions = ref.standard_smiles.map(pd.Series(np.arange(len(frame)), index=frame.standard_smiles)).to_numpy(int)
            xx, yy, ww = [matrix[positions]], [ref.SI.to_numpy(float)], [np.ones(len(ref))]
            public = ~workflow.competition_rows(frame)
            ic, cc = exact_target(frame, "IC50"), exact_target(frame, "CC50")
            paired = public & np.isfinite(ic) & np.isfinite(cc)
            xx.append(matrix[paired]); yy.append(cc[paired]/ic[paired]); ww.append(reliability[paired])
            y, w = np.concatenate(yy), np.concatenate(ww)
            if not len(y) or not np.isfinite(y).all() or (y < 0).any():
                raise ValueError("Invalid measured SI targets")
            params = {**DEFAULT_PARAMS, "objective": "reg:squarederror", "tree_method": "hist", "device": "cpu"}
            rounds = params.pop("n_estimators")
            params["seed"], params["nthread"] = params.pop("random_state"), params.pop("n_jobs")
            self.heads["SI"] = xgb.train(params, xgb.DMatrix(np.concatenate(xx), label=y, weight=w/w.mean()), num_boost_round=rounds)
            self.audit["SI"] = {"n": len(y), "measured_competition_rows": len(ref), "public_rows": int(paired.sum())}
        return self

    def predict_components(self, matrix):
        dmatrix = xgb.DMatrix(matrix)
        logs = {name: self.heads[name].predict(dmatrix, output_margin=True).astype(float) for name in ("IC50", "CC50")}
        if self.settings.get("aft"):
            logs = {name: value/np.log(10.) for name, value in logs.items()}
        base = 10.**(logs["CC50"]-logs["IC50"])
        direct = np.maximum(self.heads["SI"].predict(dmatrix).astype(float), 0) if "SI" in self.heads else base
        if not np.isfinite(base).all() or not np.isfinite(direct).all():
            raise ValueError("Nonfinite predictions")
        return logs, base, direct

    def predict_si(self, matrix):
        _, base, direct = self.predict_components(matrix)
        alpha = self.settings.get("direct_alpha", 0)
        return (1-alpha)*base + alpha*direct


def load_inputs(config):
    parts = workflow.load_parts()
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    reference = pd.read_csv(ROOT / "data/processed/kaggle_reference.csv")
    if "IC50" not in reference:  # Original reference normally contains all endpoint columns.
        raise ValueError("Original concentration reference required")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    for name in ("IC50", "CC50"):
        parsed = [parse_measurement(v) for v in reference[name]]
        reference[name] = [v for v, _ in parsed]
        reference[name+"_qualifier"] = [q for _, q in parsed]
    quality = pd.read_csv(ROOT / "data/processed/public_quality.csv")
    raw = pd.read_csv(ROOT / "data/raw/chembl.csv")
    raw["standard_smiles"] = raw.smiles.map(standardize_smiles)
    baseline = json.loads((ROOT / config["baseline_report"]).read_text(encoding="utf-8"))
    hashes = workflow.data_hashes({"model": {"strategy": "xgb_weighted"}})
    if baseline["data_sha256"] != hashes or baseline["folds"] != config["folds"] or baseline["repeats"] != config["repeats"] or baseline["seed"] != config["seed"]:
        raise ValueError("Cached baseline comparison is incompatible")
    if baseline["versions"] != {"xgboost": xgb.__version__, "rdkit": rdBase.rdkitVersion}:
        raise ValueError("Recompute baseline OOF for this library version")
    return frame, reference, row_weights(frame, quality), raw, hashes


def aggregate_oof(oof, candidates):
    protocols = {}
    for protocol, rows in oof.groupby("protocol", sort=False):
        variants = {}
        for candidate in ["baseline"] + list(candidates):
            values = score(rows.SI_true, rows[candidate])
            values["repeat_mse"] = [score(p.SI_true, p[candidate])["mse"] for _, p in rows.groupby("repeat")]
            for subset, mask in {"measured": rows.target_is_label.astype("string").str.lower().eq("false").fillna(False),
                                 "low_si": rows.SI_true.le(100)}.items():
                values[subset] = score(rows.loc[mask, "SI_true"], rows.loc[mask, candidate]) if mask.any() else None
            if candidate != "baseline":
                values["paired_gain"] = paired_interval(rows, candidate, baseline="baseline")
            variants[candidate] = values
        protocols[protocol] = variants
    return protocols


def choose_candidate(protocols, candidates):
    # Predeclared selection, before ANY endpoint diagnostic is run.
    selected = min(candidates, key=lambda c: sum(protocols[p][c]["mse"]/protocols[p]["baseline"]["mse"] for p in protocols))
    safe = True
    for p, variants in protocols.items():
        base, new = variants["baseline"], variants[selected]
        safe &= new["mse"] <= 1.02*base["mse"]
        for subset, tolerance in (("measured", 1.02), ("low_si", 1.20)):
            if base[subset] is not None:
                safe &= new[subset]["mse"] <= tolerance*base[subset]["mse"]
    return {"selected": selected, "oof_safe": bool(safe),
        "rule": "Minimize mean of molecule/scaffold MSE ratios; safety: <=2% overall/measured and <=20% SI<=100 deterioration",
        "note": "Exploratory candidate selection, no independent performance claim"}


def run_hypothesis(name, config):
    started = time.monotonic()
    frame, reference, reliability, raw_pairs, hashes = load_inputs(config)
    candidates = HYPOTHESES[name]
    directory = ROOT / "reports/hypotheses" / name
    directory.mkdir(parents=True, exist_ok=True)
    extra = pd.DataFrame(columns=["standard_smiles", "endpoint", "value_um", "weight"])
    if name == "independent":
        dataset = pd.read_csv(ROOT / "data/processed/dataset.csv")
        manifest = splits.load_manifest(ROOT / "data/processed/splits.json")
        extra = prepare_extra(dataset, dataset.iloc[manifest["holdout"]].standard_smiles.tolist())
        print(f"[suite] independent extra endpoints={len(extra)}: {extra.endpoint.value_counts().to_dict()}", flush=True)
    matrices = {preset: feature_matrix(frame.standard_smiles.tolist(), preset)
                for preset in {s.get("features", "default") for s in candidates.values()}}
    extra_features = feature_matrix(extra.standard_smiles.tolist()) if len(extra) else np.empty((0, 2061))
    baseline_oof = pd.read_csv(ROOT / config["baseline_oof"])
    output, manifests, fit_audits = [], [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(config["repeats"]):
            fold_map = fold_indices(frame.standard_smiles.tolist(), protocol, config["folds"], config["seed"]+repeat)
            manifests.append({"protocol": protocol, "repeat": repeat, "indices": fold_map})
            for fold, indices in fold_map.items():
                test = np.asarray(indices, dtype=int)
                train = np.setdiff1d(np.arange(len(frame)), test)
                rows = reference[reference.standard_smiles.isin(frame.iloc[test].standard_smiles)].copy().rename(columns={"SI": "SI_true"})
                positions = rows.standard_smiles.map(pd.Series(np.arange(len(test)), index=frame.iloc[test].standard_smiles)).to_numpy(int)
                prior = baseline_oof[(baseline_oof.protocol == protocol) & baseline_oof.repeat.eq(repeat) & baseline_oof.fold.eq(int(fold))]
                if set(prior.original_row) != set(rows.original_row):
                    raise ValueError("Cached baseline OOF fold does not match")
                rows["baseline"] = rows.original_row.map(prior.set_index("original_row").weighted)
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, fold
                rows["scaffold"] = rows.standard_smiles.map(splits.murcko_scaffold)
                extra_train = exclude_groups(extra, frame.iloc[test].standard_smiles, protocol)
                extra_matrix = extra_features[extra_train.index.to_numpy(int)]
                print(f"[suite] {name} {protocol} repeat={repeat+1}/{config['repeats']} fold={int(fold)+1}/{config['folds']}; extra={len(extra_train)}", flush=True)
                cached = {}
                for candidate, settings in candidates.items():
                    key_settings = settings.copy()
                    key_settings.pop("direct_alpha", None)
                    if name == "direct_si":
                        key_settings["direct_alpha"] = 1.
                    key = json.dumps(key_settings, sort_keys=True)
                    matrix = matrices[settings.get("features", "default")]
                    # Tune heads independently; reuse each profile's fitted pair for all combinations.
                    if name == "head_params":
                        for profile in (settings["ic_params"], settings["cc_params"]):
                            if profile not in cached:
                                model = HypothesisModel({"ic_params": profile, "cc_params": profile}).fit(
                                    frame.iloc[train], matrix[train], reliability[train], reference, extra_train, extra_matrix, raw_pairs)
                                cached[profile] = model.predict_components(matrix[test])[0]
                        prediction = 10.**(cached[settings["cc_params"]]["CC50"]-cached[settings["ic_params"]]["IC50"])
                    else:
                        if key not in cached:
                            model = HypothesisModel(key_settings).fit(frame.iloc[train], matrix[train], reliability[train], reference,
                                extra_train, extra_matrix, raw_pairs)
                            cached[key] = model.predict_components(matrix[test])
                            fit_audits.append({"protocol": protocol, "repeat": repeat, "fold": fold, "candidate": candidate, "heads": model.audit})
                        _, base, direct = cached[key]
                        alpha = settings.get("direct_alpha", 0)
                        prediction = (1-alpha)*base + alpha*direct
                    rows[candidate] = prediction[positions]
                    print(f"[suite] {candidate} MSE={score(rows.SI_true, rows[candidate])['mse']:.4f}", flush=True)
                output.append(rows)
    oof = pd.concat(output, ignore_index=True)
    for _, rows in oof.groupby("protocol"):
        if len(rows) != len(reference)*config["repeats"] or not rows.groupby("original_row").size().eq(config["repeats"]).all():
            raise AssertionError("Incomplete OOF coverage")
    protocols = aggregate_oof(oof, candidates)
    selection = choose_candidate(protocols, candidates)
    settings = candidates[selection["selected"]]
    # Selection is frozen before final refit and SMILES-only test inference.
    model = HypothesisModel(settings).fit(frame, matrices[settings.get("features", "default")], reliability,
        reference, extra, extra_features, raw_pairs)
    artifact_dir = ROOT / "artifacts/hypotheses"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"{name}.joblib"
    bundle = {"model": model, "selection": selection, "settings": settings, "config": config, "data_sha256": hashes}
    joblib.dump(bundle, artifact)
    # Exercise native model reload before generating the candidate file.
    model = joblib.load(artifact)["model"]
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv", usecols=["index", "SMILES"])
    if test["index"].isna().any() or test["index"].duplicated().any():
        raise ValueError("Invalid test identifiers")
    smiles = [standardize_smiles(s) for s in test.SMILES]
    if any(s is None for s in smiles):
        raise ValueError("Invalid test SMILES")
    prediction = model.predict_si(feature_matrix(smiles, settings.get("features", "default")))
    submission = directory / "submission.csv"
    pd.DataFrame({"index": test["index"], "SI VACV": prediction}).to_csv(submission, index=False)
    report = {"hypothesis": name, "config": config, "candidates": candidates, "selection": selection,
        "protocols": protocols, "fit_audits": fit_audits, "final_training": model.audit,
        "development_rows": len(frame), "original_reference_rows": len(reference), "holdout_used": False,
        "extra_endpoints": len(extra), "data_sha256": hashes,
        "extra_sha256": hashlib.sha256((ROOT / "data/experiments/independent_training.csv").read_bytes()).hexdigest() if name == "independent" else None,
        "versions": {"xgboost": xgb.__version__, "rdkit": rdBase.rdkitVersion},
        "artifact": artifact.relative_to(ROOT).as_posix(), "submission": submission.relative_to(ROOT).as_posix(),
        "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic()-started}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "splits.json").write_text(json.dumps({"development_smiles": frame.standard_smiles.tolist(), "splits": manifests}, indent=2), encoding="utf-8")
    oof.to_csv(directory / "oof.csv", index=False)
    print(f"[suite done] {name}: {selection}; elapsed={report['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hypotheses.json")
    parser.add_argument("--hypothesis", choices=list(HYPOTHESES)+["all"], default="all")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    for name in config["order"] if args.hypothesis == "all" else [args.hypothesis]:
        run_hypothesis(name, config)


if __name__ == "__main__":
    # Keep model pickle paths importable when this file is launched with -m.
    from src.hypotheses import main as run_cli
    run_cli()
