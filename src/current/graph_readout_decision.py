"""Evaluate the recorded SI graph hypothesis without test labels or uploads."""
import json
import numpy as np
import pandas as pd
from src.current.data import ROOT
from src.current.results import paired_gain
from src.current.graph_readout import REPORTS


def scores(rows, name):
    y, p = rows.SI_true.to_numpy(float), rows[name].to_numpy(float)
    return {key: float(np.mean((y[mask]-p[mask])**2)) for key, mask in
            [('all', np.ones(len(y), bool)), ('zero', y == 0), ('bulk', (y > 0) & (y <= 1000))]}


def main():
    rows = pd.read_csv(REPORTS/'oof.csv')
    baseline = pd.read_csv(ROOT/'reports/current/oof.csv')
    reference = 'chemeleon_domain_log1p__public_plus_kaggle'
    keys = ['original_row', 'protocol', 'repeat', 'fold']
    rows = rows.merge(baseline[keys+[reference]], on=keys, validate='one_to_one')
    if len(rows) != 267*6 or not rows.groupby('original_row').size().eq(6).all():
        raise ValueError('Incomplete aligned OOF')
    report = json.loads((REPORTS/'report.json').read_text())
    reference_scores = scores(rows, reference)
    decisions = []
    for item in report['ranking']:
        name = item['candidate']
        measured = scores(rows, name)
        scaffold = rows[rows.protocol.eq('scaffold')]
        gains = [scores(r, reference)['all']-scores(r, name)['all'] for _, r in scaffold.groupby('repeat')]
        matched = 'continue__'+item['regime'] if item['adapt'] else None
        beats_matched = measured['all'] < scores(rows, matched)['all'] if matched else True
        passes = (measured['all'] < reference_scores['all']
                  and measured['zero'] <= 1.2*reference_scores['zero']
                  and measured['bulk'] <= 1.2*reference_scores['bulk']
                  and sum(gain > 0 for gain in gains) >= 2 and beats_matched)
        final = pd.read_csv(ROOT/item['submission'])['SI VACV'].to_numpy(float)
        decisions.append({'candidate': name, 'mse': measured, 'passes_predeclared_gate': bool(passes),
                          'gain_vs_frozen': paired_gain(rows, reference, name),
                          'matched_control': matched, 'beats_matched_continuation': bool(beats_matched),
                          'gain_vs_matched': paired_gain(rows, matched, name) if matched else None,
                          'scaffold_repeat_gains': gains, 'scaffold_repeats_won': int(sum(gain > 0 for gain in gains)),
                          'final_prediction_max': float(final.max()), 'final_prediction_median': float(np.median(final)),
                          'test_to_oof_squared_prediction_ratio': float(np.mean(final**2)/max(np.mean(rows[name]**2), 1e-12))})
    result = {'reference': reference, 'reference_mse': reference_scores, 'candidates': decisions,
              'independent_confirmation': False, 'test_endpoint_columns_read': [], 'automatic_upload': False,
              'note': 'Predeclared guards are exploratory screening, not evidence of a hidden Kaggle improvement.'}
    (REPORTS/'decision_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
