"""
Задачи 15-16 (TASKA.md §7): out-of-fold predictions for the ensemble.

Folds are built from TRAIN ONLY (never touching the held-out validation
or holdout sets) and are themselves scaffold-grouped, so no scaffold
spans two folds — consistent with the project's leakage requirements.
Each molecule's OOF prediction always comes from a model that did NOT
see that molecule during fitting (Задача 15's explicit warning: fitting
on the whole train set and calling that "OOF" is not allowed).

Output: reports/oof_baseline.csv with columns
    molecule_id, fold, y_true, y_pred_direct, y_pred_two_head
(TASKA.md's minimal field list), plus a sibling JSON with the config/
seed/feature description/metrics the team needs for ensembling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .evaluate import compute_metrics
from .features import FeatureBuilder
from .models.baseline import BaselineModel
from .targets import TargetConfig, apply_censoring_policy, compute_log_targets


def _scaffold_groups_for_fold_assignment(smiles_list: List[str]) -> Dict[int, str]:
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        scaffolds[i] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) if mol else f"__invalid_{i}"
    return scaffolds


def make_scaffold_group_folds(smiles_list: List[str], n_folds: int, seed: int) -> np.ndarray:
    """Assign each row to one of n_folds folds such that rows sharing a
    scaffold always land in the same fold (group k-fold on scaffolds)."""
    scaffold_by_idx = _scaffold_groups_for_fold_assignment(smiles_list)
    groups: Dict[str, List[int]] = {}
    for idx, scaffold in scaffold_by_idx.items():
        groups.setdefault(scaffold, []).append(idx)

    rng = np.random.default_rng(seed)
    scaffold_keys = sorted(groups.keys())
    order = rng.permutation(len(scaffold_keys))
    scaffold_keys = [scaffold_keys[i] for i in order]
    scaffold_keys.sort(key=lambda s: -len(groups[s]))  # largest groups placed first

    fold_counts = [0] * n_folds
    fold_of = np.full((len(smiles_list),), -1, dtype=int)
    for scaffold in scaffold_keys:
        idxs = groups[scaffold]
        target_fold = int(np.argmin(fold_counts))
        for idx in idxs:
            fold_of[idx] = target_fold
        fold_counts[target_fold] += len(idxs)
    assert (fold_of >= 0).all()
    return fold_of


def run_oof(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    train_df = pd.read_csv(data_cfg["train_path"])

    tcfg = TargetConfig(
        ic50_col=data_cfg["ic50_col"],
        cc50_col=data_cfg["cc50_col"],
        censor_ic50_col=data_cfg.get("censor_cols", {}).get("ic50"),
        censor_cc50_col=data_cfg.get("censor_cols", {}).get("cc50"),
        policy=cfg.get("censoring_policy", "exclude"),
    )
    usable, excluded = apply_censoring_policy(train_df, tcfg)
    if len(excluded):
        print(f"[oof] excluded {len(excluded)} censored rows per policy='{tcfg.policy}'")
    usable = compute_log_targets(usable, tcfg).reset_index(drop=True)

    smiles_col = data_cfg["smiles_col"]
    id_col = data_cfg["id_col"]
    oof_cfg = cfg["oof"]
    n_folds = oof_cfg["n_folds"]
    fold_seed = oof_cfg["fold_seed"]

    folds = make_scaffold_group_folds(usable[smiles_col].tolist(), n_folds, fold_seed)
    usable["fold"] = folds

    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    # OOF is produced for the chosen comparison feature variant (the one
    # used for the primary direct-vs-two_head comparison); default to
    # "morgan" since it needs no best-feature decision to already be final.
    variant = oof_cfg.get("feature_variant", "morgan")

    strategies = model_cfg["strategies_to_compare"]
    oof_pred: Dict[str, np.ndarray] = {s: np.full((len(usable),), np.nan) for s in strategies}

    for fold_id in range(n_folds):
        is_holdout_fold = usable["fold"] == fold_id
        fold_train = usable[~is_holdout_fold]
        fold_eval = usable[is_holdout_fold]
        if len(fold_eval) == 0:
            continue

        builder = FeatureBuilder(
            variant=variant,
            morgan_radius=feat_cfg["morgan"]["radius"],
            morgan_n_bits=feat_cfg["morgan"]["n_bits"],
            morgan_use_chirality=feat_cfg["morgan"].get("use_chirality", False),
        )
        fold_train_X = builder.fit_transform(fold_train[smiles_col].tolist())
        fold_eval_X = builder.transform(fold_eval[smiles_col].tolist())

        for strategy in strategies:
            model = BaselineModel(
                strategy=strategy,
                regressor_type=model_cfg["type"],
                regressor_params=model_cfg["params"][model_cfg["type"]],
                feature_variant=variant,
                random_state=cfg["seed"],
            )
            if strategy == "direct":
                model.fit(fold_train_X, y_log_si=fold_train["log10_si"].values)
            else:
                model.fit(fold_train_X, y_log_ic50=fold_train["log10_ic50"].values, y_log_cc50=fold_train["log10_cc50"].values)
            preds = model.predict_log_si(fold_eval_X)
            oof_pred[strategy][is_holdout_fold.values] = preds

    out_df = pd.DataFrame({
        "molecule_id": usable[id_col].values,
        "fold": usable["fold"].values,
        "y_true": usable["log10_si"].values,
    })
    for strategy in strategies:
        col_name = "y_pred_direct" if strategy == "direct" else "y_pred_two_head"
        out_df[col_name] = oof_pred[strategy]

    out_path = Path(oof_cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    metrics_by_strategy = {}
    for strategy in strategies:
        col_name = "y_pred_direct" if strategy == "direct" else "y_pred_two_head"
        valid = ~out_df[col_name].isna()
        metrics_by_strategy[strategy] = compute_metrics(
            out_df.loc[valid, "y_true"].values, out_df.loc[valid, col_name].values
        )

    sidecar = {
        "config": {
            "seed": cfg["seed"],
            "n_folds": n_folds,
            "fold_seed": fold_seed,
            "feature_variant": variant,
            "regressor_type": model_cfg["type"],
            "strategies": strategies,
        },
        "feature_description": (
            f"{variant} features (Morgan radius={feat_cfg['morgan']['radius']}, "
            f"n_bits={feat_cfg['morgan']['n_bits']})" if "morgan" in variant else variant
        ),
        "oof_metrics_by_strategy": metrics_by_strategy,
        "n_excluded_censored": len(excluded),
        "model_limitations": [
            "Classical ML baseline on hashed Morgan fingerprints / RDKit 2D descriptors — "
            "no learned molecular representation, no 3D information.",
            "Censored IC50/CC50 rows are excluded (policy='exclude') rather than imputed.",
            "OOF folds are scaffold-grouped but drawn only from TRAIN; holdout is untouched.",
        ],
        "output_path": str(out_path),
    }
    sidecar_path = out_path.with_suffix(".meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[oof] wrote {len(out_df)} OOF rows to {out_path}")
    print(f"[oof] metrics by strategy: {metrics_by_strategy}")
    return sidecar
