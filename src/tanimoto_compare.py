"""Tanimoto KRR vs weighted XGB on identical original-target OOF folds."""
import json
import joblib
import numpy as np
import pandas as pd
from src import xgb_workflow as w
from src.local_compare import fold_indices
from src.local_score import score
from src.data.quality_weights import row_weights
from src.models.xgb_two_head import exact_target
from src.models.xgb_weighted import XGBWeightedTwoHead
from src.models.tanimoto_krr import TanimotoKRR
from src.features.fingerprints import MORGAN_BITS


def run():
    parts = w.load_parts()
    frame = pd.concat([parts['train'], parts['validation']], ignore_index=True)
    reference = pd.read_csv(w.ROOT / 'data/processed/kaggle_reference.csv')
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    quality = pd.read_csv(w.ROOT / 'data/processed/public_quality.csv')
    x = w.features(frame)
    fp = x[:, :MORGAN_BITS]
    targets = {k: exact_target(frame, k) for k in ('IC50', 'CC50')}
    weights = row_weights(frame, quality)
    records = []
    for protocol in ('molecule', 'scaffold'):
        for repeat in range(3):
            for fold, indices in fold_indices(frame.standard_smiles.tolist(), protocol, 3, 20261106+repeat).items():
                held = np.asarray(indices)
                fit = np.setdiff1d(np.arange(len(frame)), held)
                krr = TanimotoKRR({'alpha': 1.0}).fit(fp[fit], targets['IC50'][fit], targets['CC50'][fit])
                baseline = XGBWeightedTwoHead().fit(x[fit], targets['IC50'][fit], targets['CC50'][fit], sample_weight=weights[fit])
                rows = reference[reference.standard_smiles.isin(frame.iloc[held].standard_smiles)].copy()
                lookup = pd.Series(np.arange(len(held)), index=frame.iloc[held].standard_smiles)
                positions = rows.standard_smiles.map(lookup).to_numpy(int)
                rows['prediction'] = krr.predict_si(fp[held])[positions]
                rows['weighted_baseline'] = baseline.predict_si(x[held])[positions]
                rows['protocol'], rows['repeat'], rows['fold'] = protocol, repeat, fold
                records.append(rows)
                print(protocol, repeat, fold, score(rows.SI, rows.prediction)['mse'], flush=True)
    oof = pd.concat(records, ignore_index=True)
    report = {'architecture': 'Tanimoto KRR alpha=1', 'folds': 3, 'repeats': 3,
              'holdout_used': False, 'test_data_used': False, 'protocols': {}}
    for protocol, rows in oof.groupby('protocol'):
        if len(rows) != len(reference)*3:
            raise AssertionError('Incomplete original-row OOF coverage')
        report['protocols'][protocol] = {'tanimoto': score(rows.SI, rows.prediction),
            'weighted_baseline': score(rows.SI, rows.weighted_baseline)}
    final = TanimotoKRR({'alpha': 1.0}).fit(fp, targets['IC50'], targets['CC50'])
    joblib.dump({'model': final, 'features': 'standardized Morgan r2 2048',
                 'training_rows': len(frame), 'holdout_used': False}, w.ROOT/'artifacts/tanimoto_krr_final.joblib')
    oof.to_csv(w.ROOT/'reports/tanimoto_oof.csv', index=False)
    (w.ROOT/'reports/tanimoto_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    run()
