"""Matched readout fine-tuning/data-quality ablations, selected on development OOF only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.architecture_workflow import ROOT, inputs
from src.chemeleon_states import prepare
from src.hypotheses import aggregate_oof
from src.local_compare import fold_indices
from src.local_score import score
from src.models.architecture_heads import FrozenHeads
from src.models.chemeleon_tune import ReadoutTune
from src.oof_error_audit import original_priority_targets
from src import xgb_workflow as workflow, splits


def select(oof, candidates):
    """Predeclared measured-target OOF gate; no test diagnostics are accepted here."""
    measured = oof.target_is_label.astype('string').str.lower().eq('false').fillna(False)
    ratios, eligible = {}, []
    for candidate in candidates:
        ratios[candidate] = np.mean([score(r.SI_true, r[candidate])['mse']/score(r.SI_true, r.frozen)['mse']
            for _, r in oof.loc[measured].groupby('protocol')])
        safe = ratios[candidate] <= .99
        for protocol, rows in oof.groupby('protocol'):
            m = rows.target_is_label.astype('string').str.lower().eq('false').fillna(False)
            wins = sum(score(r.SI_true, r[candidate])['mse'] < score(r.SI_true, r.frozen)['mse']
                       for _, r in rows.loc[m].groupby('repeat'))
            safe &= wins >= 2
            if protocol == 'scaffold':
                safe &= score(rows.loc[m, 'SI_true'], rows.loc[m, candidate])['mse'] <= 1.02*score(rows.loc[m, 'SI_true'], rows.loc[m, 'frozen'])['mse']
            low = rows.SI_true.le(100)
            safe &= score(rows.loc[low, 'SI_true'], rows.loc[low, candidate])['mse'] <= 1.20*score(rows.loc[low, 'SI_true'], rows.loc[low, 'frozen'])['mse']
        if safe:
            eligible.append(candidate)
    return {'selected': min(eligible, key=ratios.get) if eligible else 'frozen',
            'eligible': eligible, 'measured_oof_mean_ratios': ratios, 'test_endpoints_used': False}


def fit_variants(cache, indices, targets, weights, source, reference, frame, cfg):
    start = time.monotonic()
    original, changes = original_priority_targets(frame.iloc[indices], reference[reference.standard_smiles.isin(frame.iloc[indices].standard_smiles)])
    models = {}
    for prefix, y in [('', targets[indices]), ('original_', original)]:
        warm = FrozenHeads('mlp', settings=cfg['warm_head'], seed=cfg['seed']).fit(
            cache.pooled[indices], y, weights[indices], source[indices])
        models[prefix+'warm' if prefix else 'frozen'] = warm
        for name, adapt in [('continue', False), ('lora', True)]:
            models[prefix+name] = ReadoutTune(warm, cfg['adaptation'], adapt).fit(cache, indices, y, weights[indices])
            print(f"[fit] {prefix+name}: delta_W={models[prefix+name].update_frobenius:.5f}; cumulative {time.monotonic()-start:.1f}s", flush=True)
    return models, changes


def logs_for(model, cache, indices):
    if isinstance(model, FrozenHeads):
        return model.predict_logs(cache.pooled[indices])
    return model.predict_logs(cache, indices)


def run(cfg, worker_repeat=None):
    started = time.monotonic()
    directory = ROOT/'reports/finetune'
    directory.mkdir(parents=True, exist_ok=True)
    frame, reference, weights, targets, test, smiles = inputs()
    cache = prepare(smiles, cfg['encoder_batch_size'])
    prior = pd.read_csv(ROOT/cfg['baseline_oof'])
    source = workflow.competition_rows(frame)
    output, audits, manifests = [], [], []
    for protocol in ('molecule', 'scaffold'):
        for repeat in range(cfg['repeats']):
            if worker_repeat is not None and repeat != worker_repeat:
                continue
            folds = fold_indices(frame.standard_smiles.tolist(), protocol, cfg['folds'], cfg['seed']+repeat)
            manifests.append({'protocol': protocol, 'repeat': repeat, 'indices': folds})
            for fold, held in folds.items():
                stamp = f'{protocol}_{repeat}_{fold}'
                print(f'[finetune] {stamp}', flush=True)
                chunk = directory/f'oof_{stamp}.csv'
                audit_path = directory/f'audit_{stamp}.json'
                # Resume only this exact, predeclared configuration and data fingerprint.
                code = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                    'src/chemeleon_finetune.py', 'src/models/chemeleon_tune.py', 'src/oof_error_audit.py', 'src/chemeleon_states.py')}
                fingerprint = hashlib.sha256(json.dumps({'config': cfg, 'code': code,
                    'data': workflow.data_hashes({'model': {'strategy': 'xgb_weighted'}})}, sort_keys=True).encode()).hexdigest()
                if chunk.exists() and audit_path.exists() and json.loads(audit_path.read_text())['fingerprint'] == fingerprint:
                    output.append(pd.read_csv(chunk)); audits.append(json.loads(audit_path.read_text()))
                    print('[resume] verified completed fold', flush=True)
                    continue
                held = np.asarray(held, dtype=int)
                fit = np.setdiff1d(np.arange(len(frame)), held)
                rows = reference[reference.standard_smiles.isin(frame.iloc[held].standard_smiles)].copy().rename(columns={'SI': 'SI_true'})
                positions = rows.standard_smiles.map(pd.Series(held, index=frame.iloc[held].standard_smiles)).to_numpy(int)
                rows['protocol'], rows['repeat'], rows['fold'] = protocol, repeat, fold
                rows['scaffold'] = rows.standard_smiles.map(splits.murcko_scaffold)
                models, changes = fit_variants(cache, fit, targets, weights, source, reference, frame, cfg)
                for candidate, model in models.items():
                    logs = logs_for(model, cache, positions)
                    rows[candidate] = np.exp(np.clip(logs[:, 1]-logs[:, 0], -80, 80))
                    rows[candidate+'_log_IC50'], rows[candidate+'_log_CC50'] = logs[:, 0], logs[:, 1]
                    m = rows.target_is_label.astype('string').str.lower().eq('false').fillna(False)
                    print(f"[OOF] {candidate}: measured MSE={score(rows.loc[m,'SI_true'],rows.loc[m,candidate])['mse']:.5f}", flush=True)
                old = prior[(prior.protocol == protocol) & prior.repeat.eq(repeat) & prior.fold.eq(int(fold))].set_index('original_row')
                np.testing.assert_allclose(rows.frozen, rows.original_row.map(old.mlp), rtol=2e-5, atol=.001)
                rows['baseline'] = rows.frozen
                audit = {'protocol': protocol, 'repeat': repeat, 'fold': fold, 'fingerprint': fingerprint,
                         'fit_rows': len(fit), 'original_policy_changes': changes,
                         'encoder_update_norms': {c: m.update_frobenius for c, m in models.items() if isinstance(m, ReadoutTune)}}
                rows.to_csv(chunk, index=False)
                audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
                output.append(rows); audits.append(audit)
    if worker_repeat is not None:
        print(f'[worker done] repeat={worker_repeat}; fold files saved; no selection/refit performed', flush=True)
        return
    oof = pd.concat(output, ignore_index=True)
    for _, rows in oof.groupby('protocol'):
        if len(rows) != len(reference)*cfg['repeats'] or not rows.groupby('original_row').size().eq(cfg['repeats']).all():
            raise ValueError('Invalid OOF coverage')
    oof.to_csv(directory/'oof.csv', index=False)
    selection = select(oof, cfg['candidates'])
    print('[selection] '+json.dumps(selection), flush=True)
    # Persist selection before final refit and before any test endpoint diagnostics.
    (directory/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    models, final_changes = fit_variants(cache, np.arange(len(frame)), targets, weights, source, reference, frame, cfg)
    artifacts = ROOT/'artifacts/finetune'
    artifacts.mkdir(parents=True, exist_ok=True)
    final = {}
    for candidate, model in models.items():
        path = artifacts/f'{candidate}.joblib'
        joblib.dump({'model': model, 'config': cfg, 'encoder': cache.metadata['checkpoint'], 'candidate': candidate}, path, compress=3)
        restored = joblib.load(path)['model']
        indices = np.arange(len(frame), len(smiles))
        logs = logs_for(restored, cache, indices)
        np.testing.assert_allclose(logs, logs_for(model, cache, indices), rtol=0, atol=0)
        prediction = np.exp(np.clip(logs[:, 1]-logs[:, 0], -80, 80))
        if not np.isfinite(prediction).all() or len(prediction) != len(test):
            raise ValueError('Invalid submission')
        dest = directory/f'submission_{candidate}.csv'
        pd.DataFrame({'index': test['index'], 'SI VACV': prediction}).to_csv(dest, index=False)
        final[candidate] = {'artifact': path.relative_to(ROOT).as_posix(), 'submission': dest.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'targets': model.n_train,
            'encoder_update_norm': getattr(model, 'update_frobenius', 0.)}
    # Baseline final reproduction must agree with the previously accepted model.
    old = joblib.load(ROOT/cfg['baseline_artifact'])['model']
    np.testing.assert_allclose(models['frozen'].predict_si(cache.pooled[len(frame):]), old.predict_si(cache.pooled[len(frame):]), rtol=1e-6, atol=1e-5)
    report = {'config': cfg, 'selection': selection, 'protocols': aggregate_oof(oof, cfg['candidates']),
              'fit_audits': audits, 'final_policy_changes': final_changes, 'final': final,
              'encoder_cache': cache.metadata, 'elapsed_seconds': time.monotonic()-started,
              'holdout_used': False, 'test_endpoints_used': False, 'kaggle_submitted': False}
    (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"[done] elapsed={report['elapsed_seconds']:.1f}s; selected={selection['selected']}", flush=True)


def predict(artifact, test_path, output):
    from src.data.standardize import standardize_smiles
    bundle = joblib.load(ROOT/artifact)
    test = pd.read_csv(ROOT/test_path, usecols=['index', 'SMILES'])
    if test['index'].isna().any() or test['index'].duplicated().any():
        raise ValueError('Invalid test IDs')
    smiles = [standardize_smiles(s) for s in test.SMILES]
    if any(s is None for s in smiles):
        raise ValueError('Invalid SMILES')
    cache = prepare(smiles, bundle['config']['encoder_batch_size'])
    if cache.metadata['checkpoint']['sha256'] != bundle['encoder']['sha256']:
        raise ValueError('Encoder checkpoint differs')
    logs = logs_for(bundle['model'], cache, np.arange(len(test)))
    values = np.exp(np.clip(logs[:, 1]-logs[:, 0], -80, 80))
    if not np.isfinite(values).all():
        raise ValueError('Invalid predictions')
    path = ROOT/output
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'index': test['index'], 'SI VACV': values}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/chemeleon_finetune.json')
    parser.add_argument('--predict-artifact')
    parser.add_argument('--worker-repeat', type=int, choices=[0, 1, 2], help='Fit disjoint repeat folds only; normal run assembles verified caches and refits')
    parser.add_argument('--test', default='data/raw/kaggle/test.csv')
    parser.add_argument('--out', default='reports/finetune_prediction.csv')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.predict_artifact:
            predict(args.predict_artifact, args.test, args.out)
        else:
            run(json.loads((ROOT/args.config).read_text()), args.worker_repeat)


if __name__ == '__main__':
    main()
