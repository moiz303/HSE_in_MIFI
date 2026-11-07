"""Nested OOF experiments; keeps the verified current workflow untouched."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.current.data import DIRECTORY, ROOT, sha
from src.current.models import SIModel
from src.current.results import paired_gain
from src.current.workflow import feature_inputs, matrix_for, select_fit_rows
from src.splits import murcko_scaffold


OUTPUT = ROOT / 'reports/current_robustness'
BASE_METHODS = ('chemeleon_domain_log1p', 'catboost_log1p')


def build_variants(settings):
    """Return a fixed, bounded search space for nested selection."""
    weights = settings['public_weights']
    variants = []
    for method in BASE_METHODS:
        for weight in weights:
            variants.append({
                'name': f'{method}_public_{weight:g}',
                'kind': 'regression',
                'method': method,
                'public_weight': weight,
                'seeds': [settings['seed']],
                'depth': settings['base_catboost_depth'],
                'iterations': settings['base_catboost_iterations'],
            })
    variants.append({
        'name': f'chemeleon_domain_log1p_ensemble_public_{settings["ensemble_public_weight"]:g}',
        'kind': 'regression',
        'method': 'chemeleon_domain_log1p',
        'public_weight': settings['ensemble_public_weight'],
        'seeds': settings['ensemble_seeds'],
        'depth': settings['base_catboost_depth'],
        'iterations': settings['base_catboost_iterations'],
    })
    for depth in settings['catboost_depths']:
        for iterations in settings['catboost_iterations']:
            variants.append({
                'name': f'catboost_log1p_tuned_d{depth}_i{iterations}_public_{settings["tuned_public_weight"]:g}',
                'kind': 'regression',
                'method': 'catboost_log1p',
                'public_weight': settings['tuned_public_weight'],
                'seeds': [settings['seed']],
                'depth': depth,
                'iterations': iterations,
            })
    for weight in weights:
        variants.append({
            'name': f'catboost_hurdle_public_{weight:g}',
            'kind': 'hurdle',
            'method': 'catboost_log1p',
            'public_weight': weight,
            'seeds': [settings['seed']],
            'depth': settings['base_catboost_depth'],
            'iterations': settings['base_catboost_iterations'],
        })
    return variants


def calibration_candidates(y, prediction, slopes):
    """Fit affine nonnegative calibrators on inner OOF predictions only."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    candidates = [
        {'name': 'identity', 'slope': 1.0, 'intercept': 0.0},
        {'name': 'constant_zero', 'slope': 0.0, 'intercept': 0.0},
        {'name': 'constant_mean', 'slope': 0.0, 'intercept': float(np.mean(y))},
    ]
    for slope in slopes:
        intercept = float(np.mean(y - slope * prediction))
        candidates.append({
            'name': f'affine_{slope:g}',
            'slope': float(slope),
            'intercept': intercept,
        })
    if np.var(prediction) > 0:
        slope = float(max(np.polyfit(prediction, y, 1)[0], 0.0))
        candidates.append({
            'name': 'affine_ols',
            'slope': slope,
            'intercept': float(np.mean(y - slope * prediction)),
        })
    return candidates


def apply_calibration(prediction, calibration):
    prediction = np.asarray(prediction, dtype=float)
    return np.maximum(calibration['slope'] * prediction + calibration['intercept'], 0.0)


def mse(y, prediction):
    return float(np.mean((np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)) ** 2))


def _weighted_frame(frame, public_weight):
    result = frame.copy()
    public = result.source.ne('kaggle')
    if public.any():
        counts = result.loc[public].groupby('standard_smiles').standard_smiles.transform('size')
        result.loc[public, 'weight'] = public_weight / counts.to_numpy(dtype=float)
    return result


def _model_config(config, variant, seed):
    result = copy.deepcopy(config)
    result['seed'] = int(seed)
    if variant['method'].startswith('catboost'):
        result['catboost']['depth'] = int(variant['depth'])
        result['catboost']['iterations'] = int(variant['iterations'])
        result['catboost']['random_seed'] = int(seed)
    return result


def _fit_predict(variant, frame, query, matrices, config):
    from catboost import CatBoostClassifier, CatBoostRegressor

    frame = _weighted_frame(frame, variant['public_weight'])
    matrix = matrix_for(variant['method'], matrices)
    x_train = matrix[frame.matrix_row.to_numpy(int)]
    x_query = matrix[query.matrix_row.to_numpy(int)]
    y = frame.SI.to_numpy(dtype=float)
    weights = frame.weight.to_numpy(dtype=float)
    source = frame.source.eq('kaggle').to_numpy()
    if variant['kind'] == 'hurdle':
        params = _model_config(config, variant, variant['seeds'][0])['catboost']
        classifier = CatBoostClassifier(**{**params, 'loss_function': 'Logloss'})
        classifier.fit(x_train, (y > 0).astype(int), sample_weight=weights)
        positive = y > 0
        if not positive.any():
            raise ValueError('Hurdle fit has no positive SI observations')
        regressor = CatBoostRegressor(**params)
        regressor.fit(
            x_train[positive], np.log1p(y[positive]), sample_weight=weights[positive])
        probability = classifier.predict_proba(x_query)[:, 1]
        conditional = np.expm1(np.clip(regressor.predict(x_query), -80, 80))
        return np.maximum(probability * conditional, 0.0)
    predictions = []
    for seed in variant['seeds']:
        model = SIModel(variant['method'], _model_config(config, variant, seed))
        model.fit(x_train, y, weights, source)
        predictions.append(model.predict(x_query))
    return np.mean(predictions, axis=0)


def _inner_splits(frame, protocol, folds, seed):
    groups = frame.standard_smiles.to_numpy() if protocol == 'molecule' else (
        frame.standard_smiles.map(murcko_scaffold).to_numpy())
    if protocol == 'molecule':
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(np.zeros(len(frame)), groups=groups))


def _outer_public(public, held, protocol):
    excluded = set(held.standard_smiles)
    result = public[~public.standard_smiles.isin(excluded)].copy()
    if protocol == 'scaffold':
        held_scaffolds = set(held.standard_smiles.map(murcko_scaffold))
        result = result[~result.standard_smiles.map(murcko_scaffold).isin(held_scaffolds)]
    return result


def _predict_variant_inner(variant, outer_train, public, protocol, matrices, config, settings, seed):
    prediction = np.full(len(outer_train), np.nan, dtype=float)
    splits = _inner_splits(outer_train, protocol, settings['inner_folds'], seed)
    for _, validation_indices in splits:
        inner_validation = outer_train.iloc[validation_indices].copy()
        regime = 'public_plus_kaggle' if variant['public_weight'] > 0 else 'kaggle_only'
        fit = select_fit_rows(
            outer_train, public, inner_validation.original_row.tolist(), protocol, regime)
        if not set(fit.original_row).isdisjoint(set(inner_validation.original_row)):
            raise ValueError('Competition rows leaked into inner fit')
        prediction[validation_indices] = _fit_predict(
            variant, fit, inner_validation, matrices, config)
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete inner OOF prediction coverage')
    return prediction


def _fit_selected(variant, outer_train, public, held, protocol, matrices, config):
    regime = 'public_plus_kaggle' if variant['public_weight'] > 0 else 'kaggle_only'
    fit = select_fit_rows(outer_train, public, held.original_row.tolist(), protocol, regime)
    return _fit_predict(variant, fit, held, matrices, config)


def run(config, settings, output=OUTPUT):
    kaggle, public, _, matrices, encoder, data_audit = feature_inputs(config)
    splits = json.loads((DIRECTORY / 'splits.json').read_text())['folds']
    variants = build_variants(settings)
    provenance = {
        'config': config,
        'settings': settings,
        'data': data_audit,
        'encoder': encoder['checkpoint'],
        'code_sha256': {
            path: sha(ROOT / path) for path in (
                'src/current/robustness.py',
                'src/current/models.py',
                'src/current/workflow.py',
                'src/current/data.py',
            )
        },
        'test_endpoints_read': False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'run_manifest.json'
    if output.exists():
        if not manifest_path.exists():
            raise ValueError(f'Existing output has no verified manifest: {output}')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('fingerprint') != fingerprint:
            raise ValueError(f'Existing output belongs to a different experiment: {output}')
    else:
        output.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            'fingerprint': fingerprint,
            'provenance': provenance,
            'resumable': True,
            'automatic_promotion': False,
        }, indent=2), encoding='utf-8')
    predictions = []
    choices = []
    for split in splits:
        protocol = split['protocol']
        stamp = f"{protocol}_{split['repeat']}_{split['fold']}"
        chunk_path = output / f'oof_{stamp}.csv'
        choice_path = output / f'choice_{stamp}.json'
        if chunk_path.exists() and choice_path.exists():
            saved_choice = json.loads(choice_path.read_text())
            if saved_choice.get('fingerprint') != fingerprint:
                raise ValueError(f'Checkpoint fingerprint mismatch: {stamp}')
            if saved_choice.get('chunk_sha256') != sha(chunk_path):
                raise ValueError(f'Checkpoint contents changed: {stamp}')
            predictions.append(pd.read_csv(chunk_path))
            choices.append(saved_choice['choice'])
            print(f'[nested resume] {stamp}', flush=True)
            continue
        seed = int(settings['seed'] + split['repeat'] * 100 + split['fold'])
        held = kaggle[kaggle.original_row.isin(split['held_original_rows'])].copy()
        outer_train = kaggle[~kaggle.original_row.isin(split['held_original_rows'])].copy()
        eligible_public = _outer_public(public, held, protocol)
        inner_results = []
        for variant in variants:
            inner_prediction = _predict_variant_inner(
                variant, outer_train, eligible_public, protocol, matrices, config, settings, seed)
            for calibration in calibration_candidates(
                    outer_train.SI.to_numpy(dtype=float), inner_prediction, settings['calibration_slopes']):
                calibrated = apply_calibration(inner_prediction, calibration)
                inner_results.append({
                    'variant': variant,
                    'calibration': calibration,
                    'inner_mse': mse(outer_train.SI.to_numpy(dtype=float), calibrated),
                })
        best = min(inner_results, key=lambda row: row['inner_mse'])
        outer_prediction = _fit_selected(
            best['variant'], outer_train, eligible_public, held, protocol, matrices, config)
        outer_prediction = apply_calibration(outer_prediction, best['calibration'])
        held_rows = held[['original_row', 'standard_smiles', 'SI']].copy()
        held_rows = held_rows.rename(columns={'SI': 'SI_true'})
        held_rows['protocol'] = protocol
        held_rows['repeat'] = split['repeat']
        held_rows['fold'] = split['fold']
        held_rows['prediction'] = outer_prediction
        held_rows['constant_zero'] = 0.0
        held_rows['constant_mean'] = float(outer_train.SI.mean())
        held_rows['selected_variant'] = best['variant']['name']
        held_rows['calibration'] = best['calibration']['name']
        predictions.append(held_rows)
        choice = {
            'protocol': protocol,
            'repeat': split['repeat'],
            'fold': split['fold'],
            'selected_variant': best['variant']['name'],
            'calibration': best['calibration'],
            'inner_mse': best['inner_mse'],
            'outer_mse': mse(held.SI.to_numpy(dtype=float), outer_prediction),
            'outer_train_rows': len(outer_train),
            'outer_public_rows': len(eligible_public),
        }
        choices.append(choice)
        held_rows.to_csv(chunk_path, index=False)
        choice_path.write_text(json.dumps({
            'fingerprint': fingerprint,
            'chunk_sha256': sha(chunk_path),
            'choice': choice,
        }, indent=2), encoding='utf-8')
        print(f"[nested OOF] {protocol}_{split['repeat']}_{split['fold']}: "
              f"{best['variant']['name']} / {best['calibration']['name']}; "
              f"MSE={choice['outer_mse']:.5f}", flush=True)

    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != len(kaggle) * config['repeats'] * 2:
        raise ValueError('Unexpected nested OOF row count')
    summary = {}
    for protocol, rows in oof.groupby('protocol'):
        if not rows.groupby('original_row').size().eq(config['repeats']).all():
            raise ValueError('Invalid nested OOF coverage')
        summary[protocol] = {
            'mse': mse(rows.SI_true, rows.prediction),
            'zero_mse': mse(rows.loc[rows.SI_true.eq(0), 'SI_true'],
                            rows.loc[rows.SI_true.eq(0), 'prediction']),
            'positive_mse': mse(rows.loc[rows.SI_true.gt(0), 'SI_true'],
                                rows.loc[rows.SI_true.gt(0), 'prediction']),
            'constant_zero_mse': mse(rows.SI_true, rows.constant_zero),
            'constant_mean_mse': mse(rows.SI_true, rows.constant_mean),
            'selected_vs_zero_cluster_bootstrap': paired_gain(
                rows, 'constant_zero', 'prediction'),
        }
    oof.to_csv(output / 'nested_oof.csv', index=False)
    (output / 'selection.json').write_text(json.dumps({
        'fingerprint': fingerprint,
        'provenance': provenance,
        'summary': summary,
        'fold_choices': choices,
        'variants': variants,
        'automatic_promotion': False,
        'official_score_known': False,
        'kaggle_submitted': False,
        'test_endpoints_used': False,
    }, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/current.json')
    parser.add_argument('--settings', default='configs/current_robustness.json')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    config = json.loads((ROOT / args.config).read_text())
    settings = json.loads((ROOT / args.settings).read_text())
    run(config, settings, args.output)


if __name__ == '__main__':
    main()