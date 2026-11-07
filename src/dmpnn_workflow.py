"""Nested scaffold OOF and train+validation refit for architecture 4."""

from __future__ import annotations

import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.local_compare import fold_indices
from src.local_score import score
from src.models.dmpnn_multitask import DMPNNMultiTask, batch_graphs, featurize_smiles, bounded_mse

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/dmpnn_arch4.json").read_text(encoding="utf-8"))
SEED = CONFIG["seed"]
HIDDEN, DEPTH, DROPOUT = (CONFIG["model"][key] for key in ("hidden", "depth", "dropout"))
AUX_CANDIDATES = tuple(CONFIG["training"]["aux_weight_candidates"])
BATCH_SIZE = CONFIG["training"]["batch_size"]
MAX_EPOCHS = CONFIG["training"]["max_epochs"]
PATIENCE = CONFIG["training"]["early_stopping_patience"]
DEVICE = CONFIG["training"]["device"]


def _targets(frame, reference):
    n = len(frame)
    targets = {"SI": np.full(n, np.nan), "SI_count": np.zeros(n, dtype=float), "IC50": np.full(n, np.nan),
               "CC50": np.full(n, np.nan)}
    qualifiers = {name: np.full(n, "", dtype=object) for name in ("IC50", "CC50")}
    positions = {smiles: index for index, smiles in enumerate(frame.standard_smiles)}
    # Direct SI supervision uses original Kaggle training targets only.
    ref = reference[reference.standard_smiles.isin(positions)].copy()
    if not ref.empty:
        si_groups = ref.groupby("standard_smiles").SI
        si_by_smiles = si_groups.mean()
        for smiles, value in si_by_smiles.items():
            targets["SI"][positions[smiles]] = float(value)
            targets["SI_count"][positions[smiles]] = int(si_groups.size().loc[smiles])
    for task in ("IC50", "CC50"):
        values = pd.to_numeric(frame[task], errors="coerce").to_numpy(dtype=float, copy=True)
        values[(~np.isfinite(values)) | (values <= 0)] = np.nan
        targets[task] = np.log10(values, where=np.isfinite(values), out=np.full(n, np.nan))
        q = frame[task + "_qualifier"].fillna("").astype(str).to_numpy(object)
        for i, value in enumerate(q):
            parts = [part.strip() for part in value.split(";") if part.strip()]
            signs = {part[:1] for part in parts}
            # Opposing censor bounds in an aggregate are ambiguous; omit that target.
            known = all(part.startswith((">", ">=", "<", "<=", "~")) for part in parts)
            if len(signs) > 1 or not known or (parts and signs == {"~"}):
                targets[task][i] = np.nan
                q[i] = ""
        qualifiers[task] = q
    targets["SI"] /= 10000.0
    return targets, qualifiers, ref


def _masked_loss(pred, target, qualifier, mask, sample_weight=None):
    if not mask.any():
        return None
    if qualifier is None:
        loss = (pred[mask] - torch.as_tensor(target[mask], device=pred.device,
                                             dtype=pred.dtype)).square()
    else:
        loss = bounded_mse(pred[mask], target[mask], qualifier[mask])
    if sample_weight is None:
        return loss.mean()
    weights = torch.as_tensor(sample_weight[mask], device=pred.device, dtype=pred.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _epoch(model, graphs, indices, targets, qualifiers, aux_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    order = np.asarray(indices, dtype=int).copy()
    if training:
        np.random.shuffle(order)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            pred = model(batch_graphs([graphs[i] for i in batch], DEVICE))
            total = None
            si = targets["SI"][batch]
            si_mask = np.isfinite(si)
            main = _masked_loss(pred[:, 0], si, None, si_mask, targets["SI_count"][batch])
            if main is not None:
                total = main
            aux = []
            for col, task in ((1, "IC50"), (2, "CC50")):
                y = targets[task][batch]
                mask = np.isfinite(y)
                loss = _masked_loss(pred[:, col], y, qualifiers[task][batch], mask)
                if loss is not None:
                    aux.append(loss)
            if aux:
                aux_mean = torch.stack(aux).mean()
                total = aux_weight * aux_mean if total is None else total + aux_weight * aux_mean
            if total is None:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            losses.append(float(total.detach()))
    return float(np.mean(losses)) if losses else float("nan")


def _predict(model, graphs, indices):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            batch = indices[start:start + BATCH_SIZE]
            output.extend((model(batch_graphs([graphs[i] for i in batch], DEVICE))[:, 0]
                           .cpu().numpy() * 10000.0).tolist())
    return np.asarray(output, dtype=float)


def _new_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return DMPNNMultiTask(HIDDEN, DEPTH, DROPOUT).to(DEVICE)


def _train_early_stop(graphs, train_idx, valid_idx, targets, qualifiers, reference, smiles, aux_weight, seed):
    model = _new_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"],
                                  weight_decay=CONFIG["training"]["weight_decay"])
    valid_reference = reference[reference.standard_smiles.isin(np.asarray(smiles)[valid_idx])]
    best_epoch, best_mse, best_state, stale = 1, float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        _epoch(model, graphs, train_idx, targets, qualifiers, aux_weight, optimizer)
        prediction = _predict(model, graphs, valid_idx)
        if valid_reference.empty:
            raise ValueError("inner scaffold fold has no competition SI targets")
        lookup = pd.Series(prediction, index=np.asarray(smiles)[valid_idx])
        rows = valid_reference.copy()
        rows["prediction"] = rows.standard_smiles.map(lookup)
        mse = _metrics(rows)["all"]["mse"]
        if mse < best_mse:
            best_mse, best_epoch, stale = mse, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def _fit_fixed(graphs, indices, targets, qualifiers, aux_weight, epochs, seed):
    model = _new_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"],
                                  weight_decay=CONFIG["training"]["weight_decay"])
    for _ in range(max(1, epochs)):
        _epoch(model, graphs, indices, targets, qualifiers, aux_weight, optimizer)
    return model


def _metrics(rows):
    y, p = rows.SI.to_numpy(float), rows.prediction.to_numpy(float)
    result = {"all": score(y, p)}
    labels = rows.target_is_label.astype("string").str.lower().eq("true").to_numpy()
    for name, mask in (("service_label", labels), ("measured", ~labels)):
        if mask.any():
            result[name] = score(y[mask], p[mask])
    return result


def run():
    started = time.time()
    torch.set_num_threads(1)
    from src.xgb_workflow import load_parts
    parts = load_parts()  # validates the committed manifest; only train/validation frames are returned
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    if frame.standard_smiles.duplicated().any():
        raise ValueError("development table must have unique standardized structures")
    reference = pd.read_csv(ROOT / "data/processed/kaggle_reference.csv")
    targets, qualifiers, reference = _targets(frame, reference)
    if reference.empty:
        raise ValueError("no original competition targets in development data")
    print(f"[D-MPNN] development structures={len(frame)}, original Kaggle targets={len(reference)}, holdout=0", flush=True)
    graphs = [featurize_smiles(value) for value in frame.standard_smiles]
    outer = fold_indices(frame.standard_smiles.tolist(), CONFIG["split"]["protocol"],
                         CONFIG["split"]["folds"], SEED)
    oof_parts, fold_reports, selected_weights, selected_epochs = [], [], [], []
    for fold, test_idx in outer.items():
        test_idx = np.asarray(test_idx, dtype=int)
        outer_train = np.setdiff1d(np.arange(len(frame)), test_idx)
        inner = fold_indices(frame.iloc[outer_train].standard_smiles.tolist(), "scaffold",
                             CONFIG["split"]["inner_folds"],
                             SEED + int(fold) + 1)
        inner_valid = outer_train[np.asarray(inner["0"], dtype=int)]
        inner_train = np.setdiff1d(outer_train, inner_valid)
        selection = []
        for candidate_index, weight in enumerate(AUX_CANDIDATES):
            model, epoch = _train_early_stop(graphs, inner_train, inner_valid, targets,
                                              qualifiers, reference, frame.standard_smiles.to_numpy(),
                                              weight, SEED + 100 * int(fold) + candidate_index)
            ref_inner = reference[reference.standard_smiles.isin(frame.iloc[inner_valid].standard_smiles)]
            inner_prediction = pd.Series(_predict(model, graphs, inner_valid),
                                          index=frame.iloc[inner_valid].standard_smiles)
            inner_rows = ref_inner.copy()
            inner_rows["prediction"] = inner_rows.standard_smiles.map(inner_prediction)
            selection.append({"aux_weight": weight, "best_epoch": epoch,
                              "inner_mse": _metrics(inner_rows)["all"]["mse"]})
        chosen = min(selection, key=lambda item: item["inner_mse"])
        selected_weights.append(chosen["aux_weight"])
        selected_epochs.append(chosen["best_epoch"])
        model = _fit_fixed(graphs, outer_train, targets, qualifiers, chosen["aux_weight"],
                           chosen["best_epoch"], SEED + 1000 + int(fold))
        test_prediction = pd.Series(_predict(model, graphs, test_idx),
                                     index=frame.iloc[test_idx].standard_smiles)
        rows = reference[reference.standard_smiles.isin(frame.iloc[test_idx].standard_smiles)].copy()
        rows["prediction"] = rows.standard_smiles.map(test_prediction)
        rows["fold"] = int(fold)
        oof_parts.append(rows)
        fold_reports.append({"fold": int(fold), "train_structures": len(outer_train),
                             "oof_structures": len(test_idx), "oof_targets": len(rows),
                             "inner_selection": selection, "selected": chosen,
                             "oof_metrics": _metrics(rows)})
        print(f"[D-MPNN] fold={fold}; selected aux={chosen['aux_weight']}, epoch={chosen['best_epoch']}; "
              f"OOF MSE={_metrics(rows)['all']['mse']:.3f}", flush=True)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values("original_row")
    if oof.original_row.duplicated().any() or len(oof) != len(reference):
        raise AssertionError("OOF must score each development Kaggle training target once")
    oof_path = ROOT / CONFIG["outputs"]["oof_predictions"]
    oof.to_csv(oof_path, index=False)

    # Refit the chosen architecture on train+validation with settings selected only
    # from nested train folds. The separately reserved holdout is never loaded.
    weight = float(np.median(selected_weights))
    epochs = max(1, int(np.median(selected_epochs)))
    model = _fit_fixed(graphs, np.arange(len(frame)), targets, qualifiers, weight, epochs, SEED + 9999)
    artifact = ROOT / CONFIG["outputs"]["artifact"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    versions = {name: version(name) for name in ("torch", "numpy", "pandas", "rdkit")}
    torch.save({"state_dict": model.state_dict(), "hidden": HIDDEN, "depth": DEPTH,
                "dropout": DROPOUT, "aux_weight": weight, "epochs": epochs,
                "training_structures": len(frame), "holdout_used": False,
                "seed": SEED, "versions": versions}, artifact)
    overall = _metrics(oof)
    report = {"model": "shared D-MPNN, direct SI + masked/bounded log10 IC50 and CC50",
              "metric": "raw SI MSE on original Kaggle training targets",
              "config": CONFIG,
              "protocol": "3-fold scaffold OOF, nested inner scaffold split for loss weight and early stopping",
              "development_structures": len(frame), "original_kaggle_targets": len(reference),
              "holdout_used": False, "test_endpoints_used": False,
              "settings": {"hidden": HIDDEN, "depth": DEPTH, "dropout": DROPOUT,
                           "candidate_aux_weights": AUX_CANDIDATES, "selected_aux_weight": weight,
                           "selected_epochs": epochs, "batch_size": BATCH_SIZE},
              "versions": {"python": platform.python_version(), **versions},
              "oof_metrics": overall, "folds": fold_reports,
              "artifact": str(artifact.relative_to(ROOT)), "oof_predictions": str(oof_path.relative_to(ROOT)),
              "elapsed_seconds": time.time() - started,
              "strata_counts": {"service_label": int(oof.target_is_label.astype(str).str.lower().eq("true").sum()),
                                "measured": int(oof.target_is_label.astype(str).str.lower().ne("true").sum())}}
    report_path = ROOT / CONFIG["outputs"]["report"]
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"oof_metrics": overall, "artifact": str(artifact),
                      "report": str(report_path.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    run()
