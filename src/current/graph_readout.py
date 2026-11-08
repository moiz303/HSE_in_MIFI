"""Strict SI-only low-rank graph readout adaptation with matched continuation.

The frozen message-passing prefix uses structures only. Legacy concentration
heads and their target loaders are never imported. W_i/W_h stay frozen; this
is partial W_o adaptation, not full message-passing fine-tuning.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.current.data import ROOT, DIRECTORY, sha
from src.current.models import SIModel, make_network
from src.current.workflow import feature_inputs, select_fit_rows, metrics
from src.chemeleon_states import prepare
from src.models.chemeleon_tune import pooled_delta
from src.data.standardize import standardize_smiles

REPORTS = ROOT / 'reports/current_graph_readout'
ARTIFACTS = ROOT / 'artifacts/current_graph_readout'
REGIMES = ['kaggle_only', 'public_plus_kaggle']
SETTINGS = dict(epochs=40, batch_size=32, rank=8, alpha=8., head_lr=.0001,
                encoder_lr=.0003, weight_decay=.01, feature_anchor=.01)


class SIGraphReadout:
    def __init__(self, warm, adapt, settings=None):
        if warm.method != 'chemeleon_domain_log1p':
            raise ValueError('Strict single SI domain head required')
        self.warm = copy.deepcopy(warm)
        self.adapt = bool(adapt)
        self.settings = dict(SETTINGS if settings is None else settings)

    def _head(self):
        import torch
        cfg = self.warm.config['neural']
        head = make_network(self.warm.dim, cfg['hidden'], cfg['dropout'], True)
        head.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in self.warm.state.items()})
        return head

    def fit(self, cache, indices, si, weights, source):
        import torch
        indices = np.asarray(indices, int)
        si, weights = np.asarray(si, float), np.asarray(weights, float)
        source = np.asarray(source, bool)
        if not (len(indices) == len(si) == len(weights) == len(source)):
            raise ValueError('Training row mismatch')
        if not np.isfinite(si).all() or (si < 0).any() or not source.any():
            raise ValueError('Nonnegative finite SI and competition rows required')
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Positive finite weights required')
        torch.set_num_threads(4)
        torch.manual_seed(self.warm.config['seed'])
        cfg = self.settings
        a = torch.nn.Parameter(torch.randn(cfg['rank'], cache.inputs.shape[1]) * .01,
                               requires_grad=self.adapt)
        b = torch.nn.Parameter(torch.zeros(self.warm.dim, cfg['rank']), requires_grad=self.adapt)
        # Initialize adapters in both arms to keep random draws/dropout matched.
        head = self._head()
        groups = [{'params': head.parameters(), 'lr': cfg['head_lr']}]
        if self.adapt:
            groups.append({'params': [a, b], 'lr': cfg['encoder_lr']})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        yy = torch.tensor((np.log1p(si)-self.warm.y_mean)/self.warm.y_scale, dtype=torch.float32)
        ww = torch.tensor(weights, dtype=torch.float32)
        ss = torch.tensor(source.astype(np.int64))
        base = torch.tensor((np.asarray(cache.pooled[indices], float)-self.warm.x_mean)/self.warm.x_scale,
                            dtype=torch.float32)
        scale = torch.tensor(self.warm.x_scale, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.warm.config['seed'])
        head.train()
        for epoch in range(cfg['epochs']):
            for batch in torch.randperm(len(indices), generator=generator).split(cfg['batch_size']):
                delta = torch.zeros_like(base[batch])
                if self.adapt:
                    z, p, g, c = cache.batch(indices[batch.numpy()])
                    delta = pooled_delta(torch.from_numpy(z), torch.from_numpy(p), torch.from_numpy(g),
                                         torch.from_numpy(c), a, b, cfg['alpha']/cfg['rank'])/scale
                features = (base[batch]+delta).clamp(-10, 10)
                prediction = head(features, ss[batch])
                anchor = (features-base[batch].clamp(-10, 10)).square().mean()
                loss = ((prediction-yy[batch]).square()*ww[batch]).sum()/ww[batch].sum()
                loss = loss+cfg['feature_anchor']*anchor
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(head.parameters())+([a, b] if self.adapt else []), 5.)
                optimizer.step()
        self.a, self.b = a.detach().numpy(), b.detach().numpy()
        self.warm.state = {k: v.detach().numpy() for k, v in head.state_dict().items()}
        self.update_frobenius = float(np.sqrt(max(0., np.sum((self.a@self.a.T)*(self.b.T@self.b))))
                                      *cfg['alpha']/cfg['rank'])
        self.fit_rows, self.competition_rows = len(si), int(source.sum())
        self.public_rows, self.zero_targets = int((~source).sum()), int((si == 0).sum())
        return self

    def predict(self, cache, indices):
        import torch
        head = self._head().eval()
        scale = torch.tensor(self.warm.x_scale, dtype=torch.float32)
        output = []
        with torch.no_grad():
            for begin in range(0, len(indices), self.settings['batch_size']):
                ii = np.asarray(indices, int)[begin:begin+self.settings['batch_size']]
                base = torch.tensor((np.asarray(cache.pooled[ii], float)-self.warm.x_mean)/self.warm.x_scale,
                                    dtype=torch.float32)
                if self.adapt:
                    z, p, g, c = cache.batch(ii)
                    base += pooled_delta(torch.from_numpy(z), torch.from_numpy(p), torch.from_numpy(g),
                                         torch.from_numpy(c), torch.from_numpy(self.a), torch.from_numpy(self.b),
                                         self.settings['alpha']/self.settings['rank'])/scale
                output.append(head(base.clamp(-10, 10), torch.ones(len(ii), dtype=torch.long)).numpy())
        logs = np.concatenate(output).astype(float)*self.warm.y_scale+self.warm.y_mean
        values = np.maximum(np.expm1(np.clip(logs, -80, 80)), 0.)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite SI prediction')
        return values


def run(pilot=False):
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'configs/current.json').read_text())
    kaggle, public, test, matrices, encoder, audit = feature_inputs(cfg)
    joined = pd.concat([kaggle, public], ignore_index=True)
    smiles = [standardize_smiles(s) for s in joined.SMILES.tolist()+test.SMILES.tolist()]
    cache = prepare(smiles)
    np.testing.assert_array_equal(cache.pooled, matrices['chemeleon'])
    provenance = {'config': cfg, 'settings': SETTINGS, 'data': audit, 'encoder': encoder,
                  'graph_cache': cache.metadata, 'code_sha256': {name: sha(ROOT/name) for name in
                  ['src/current/graph_readout.py', 'src/current/models.py', 'src/current/workflow.py',
                   'src/chemeleon_states.py', 'src/models/chemeleon_tune.py']},
                  'changed_factor': 'W_o rank8 adaptation on log1p(SI), compared with equal head-only continuation',
                  'legacy_failure': 'Concentration LoRA did not beat matched continuation; never reuse its targets/heads',
                  'gate': 'ALL MSE below frozen CheM reference; zero/bulk <=1.2x; >=2/3 scaffold-repeat wins. LoRA must also beat matched continuation.',
                  'official_training_columns': ['SMILES', 'SI VACV'], 'test_columns': ['index', 'SMILES'],
                  'holdout_used': False, 'automatic_submission': False}
    (REPORTS/'plan.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    previous = pd.read_csv(ROOT/'reports/current/oof.csv')
    splits = json.loads((DIRECTORY/'splits.json').read_text())['folds']
    chunks = []
    for ordinal, split in enumerate(splits):
        if pilot and ordinal:
            break
        stamp = f"{split['protocol']}_{split['repeat']}_{split['fold']}"
        path, marker = REPORTS/('fold_'+stamp+'.csv'), REPORTS/('fold_'+stamp+'.json')
        if path.exists() and marker.exists() and json.loads(marker.read_text())['fingerprint'] == fingerprint:
            chunks.append(pd.read_csv(path))
            print('[graph resume] '+stamp, flush=True)
            continue
        started = time.monotonic()
        held = kaggle[kaggle.original_row.isin(split['held_original_rows'])]
        hi = held.matrix_row.to_numpy(int)
        rows = held[['original_row', 'standard_smiles', 'SI']].rename(columns={'SI': 'SI_true'}).copy()
        for key in ['protocol', 'repeat', 'fold']:
            rows[key] = split[key]
        old = previous[previous.protocol.eq(split['protocol']) & previous.repeat.eq(split['repeat'])
                       & previous.fold.eq(split['fold'])].set_index('original_row')
        fits = []
        for regime in REGIMES:
            fit = select_fit_rows(kaggle, public, split['held_original_rows'], split['protocol'], regime)
            fi = fit.matrix_row.to_numpy(int)
            source = fit.source.eq('kaggle').to_numpy()
            warm = SIModel('chemeleon_domain_log1p', cfg).fit(cache.pooled[fi], fit.SI, fit.weight, source)
            reference = 'chemeleon_domain_log1p__'+regime
            np.testing.assert_allclose(warm.predict(cache.pooled[hi]), held.original_row.map(old[reference]),
                                       rtol=1e-5, atol=1e-5)
            for adapt in [False, True]:
                before = time.monotonic()
                model = SIGraphReadout(warm, adapt).fit(cache, fi, fit.SI, fit.weight, source)
                name = ('lora' if adapt else 'continue')+'__'+regime
                rows[name] = model.predict(cache, hi)
                fits.append({'candidate': name, 'seconds': time.monotonic()-before,
                             'fit_rows': model.fit_rows, 'competition_rows': model.competition_rows,
                             'public_rows': model.public_rows, 'zero_targets': model.zero_targets,
                             'update_frobenius': model.update_frobenius})
                print(f"[graph {ordinal+1}/18] {stamp} {name} MSE={metrics(held.SI, rows[name])['mse']:.2f}; {time.monotonic()-before:.1f}s", flush=True)
        rows.to_csv(path, index=False)
        marker.write_text(json.dumps({'fingerprint': fingerprint, 'fits': fits,
                                     'seconds': time.monotonic()-started, 'frozen_controls_reproduced': True}, indent=2), encoding='utf-8')
        chunks.append(rows)
    if pilot:
        print('[pilot complete] One fold is a runtime/control check, not model selection.', flush=True)
        return
    oof = pd.concat(chunks, ignore_index=True)
    if len(oof) != len(kaggle)*6 or not oof.groupby('original_row').size().eq(6).all():
        raise ValueError('Incomplete OOF')
    oof.to_csv(REPORTS/'oof.csv', index=False)
    ranking = []
    for regime in REGIMES:
        for adapt in [False, True]:
            name = ('lora' if adapt else 'continue')+'__'+regime
            per = {p: metrics(r.SI_true, r[name])['mse'] for p, r in oof.groupby('protocol')}
            ranking.append({'candidate': name, 'regime': regime, 'adapt': adapt,
                            'mean_oof_mse': float(np.mean(list(per.values()))), **per})
    ranking.sort(key=lambda row: row['mean_oof_mse'])
    report = {'ranking': ranking, 'provenance': provenance, 'fingerprint': fingerprint,
              'official_score': None, 'automatic_promotion': False}
    (REPORTS/'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    begin = len(joined)
    for regime in REGIMES:
        fit = joined if regime == 'public_plus_kaggle' else kaggle
        fi = fit.matrix_row.to_numpy(int)
        source = fit.source.eq('kaggle').to_numpy()
        warm = SIModel('chemeleon_domain_log1p', cfg).fit(cache.pooled[fi], fit.SI, fit.weight, source)
        reference = joblib.load(ROOT/'artifacts/current'/('chemeleon_domain_log1p__'+regime+'.joblib'))['model']
        np.testing.assert_allclose(warm.predict(cache.pooled[begin:]), reference.predict(cache.pooled[begin:]), rtol=1e-5, atol=1e-5)
        for row in [r for r in ranking if r['regime'] == regime]:
            model = SIGraphReadout(warm, row['adapt']).fit(cache, fi, fit.SI, fit.weight, source)
            path = ARTIFACTS/(row['candidate']+'.joblib')
            joblib.dump({'model': model, 'provenance': provenance}, path, compress=3)
            restored = joblib.load(path)['model']
            values = restored.predict(cache, np.arange(begin, len(smiles)))
            np.testing.assert_array_equal(values, model.predict(cache, np.arange(begin, len(smiles))))
            submission = REPORTS/('submission_'+row['candidate']+'.csv')
            pd.DataFrame({'index': test['index'], 'SI VACV': values}).to_csv(submission, index=False)
            row.update(artifact=path.relative_to(ROOT).as_posix(), submission=submission.relative_to(ROOT).as_posix(),
                       submission_sha256=sha(submission), update_frobenius=model.update_frobenius)
    (REPORTS/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv', index=False)
    print('[graph complete] '+json.dumps(ranking), flush=True)


def verify():
    report = json.loads((REPORTS/'report.json').read_text())
    test = pd.read_csv(ROOT/'data/raw/kaggle/test.csv', usecols=['index', 'SMILES'])
    cache = prepare([standardize_smiles(s) for s in test.SMILES])
    checks = []
    for row in report['ranking']:
        if sha(ROOT/row['submission']) != row['submission_sha256']:
            raise ValueError('Submission changed')
        bundle = joblib.load(ROOT/row['artifact'])
        if cache.metadata['checkpoint']['sha256'] != bundle['provenance']['encoder']['checkpoint']['sha256']:
            raise ValueError('Graph checkpoint changed')
        model = bundle['model']
        if model.competition_rows != 267 or model.zero_targets != 66:
            raise ValueError('Unexpected competition cohort')
        values = model.predict(cache, np.arange(len(test)))
        saved = pd.read_csv(ROOT/row['submission'])
        np.testing.assert_array_equal(test['index'], saved['index'])
        np.testing.assert_allclose(values, saved['SI VACV'], rtol=1e-4, atol=1e-4)
        checks.append({'candidate': row['candidate'], 'passed': True,
                       'maximum_absolute_difference': float(np.max(np.abs(values-saved['SI VACV'])))})
    result = {'status': 'passed', 'test_columns_read': ['index', 'SMILES'], 'checks': checks}
    (REPORTS/'inference_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    # Canonical module classes are required for fresh-process joblib loading.
    from src.current.graph_readout import run as canonical_run, verify as canonical_verify
    with threadpool_limits(limits=4):
        canonical_verify() if args.verify else canonical_run(args.pilot)
