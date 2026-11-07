"""Fresh-process SMILES-only reproduction of every saved fine-tuning artifact."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from src.chemeleon_finetune import ROOT, logs_for
from src.chemeleon_states import prepare
from src.data.standardize import standardize_smiles


def main():
    directory = ROOT/'reports/finetune'
    report = json.loads((directory/'report.json').read_text())
    test = pd.read_csv(ROOT/'data/raw/kaggle/test.csv', usecols=['index', 'SMILES'])
    cache = prepare([standardize_smiles(s) for s in test.SMILES])
    checks = []
    for candidate, record in report['final'].items():
        bundle = joblib.load(ROOT/record['artifact'])
        if bundle['encoder']['sha256'] != cache.metadata['checkpoint']['sha256']:
            raise ValueError('Different encoder checkpoint')
        logs = logs_for(bundle['model'], cache, np.arange(len(test)))
        values = np.exp(np.clip(logs[:, 1]-logs[:, 0], -80, 80))
        path = ROOT/record['submission']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Saved submission bytes changed')
        submission = pd.read_csv(path)
        np.testing.assert_array_equal(submission['index'], test['index'])
        old = submission['SI VACV'].to_numpy()
        # Independent graph batches produce small CPU float32 accumulation differences.
        np.testing.assert_allclose(values, old, rtol=1e-4, atol=1e-4)
        checks.append({'candidate': candidate, 'rows': len(values), 'passed': True,
            'max_absolute_difference': float(np.max(np.abs(values-old))),
            'max_relative_difference': float(np.max(np.abs(values-old)/np.maximum(np.abs(old), 1e-8)))})
    registry = json.loads((ROOT/'reports/architectures/kaggle_submissions.json').read_text())['submissions']
    accepted = next(r for r in registry if r['ref'] == 58092002)
    if hashlib.sha256((ROOT/accepted['submission']).read_bytes()).hexdigest() != accepted['submission_sha256']:
        raise ValueError('Preserved Kaggle submission changed')
    result = {'test_columns_read': ['index', 'SMILES'], 'test_endpoints_read': False,
              'encoder_checkpoint_sha256': cache.metadata['checkpoint']['sha256'],
              'preserved_kaggle_submission_sha256': accepted['submission_sha256'], 'checks': checks}
    (directory/'inference_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
