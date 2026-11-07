"""Fresh-process artifact reproduction using only test IDs and SMILES."""
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from src.current.data import ROOT,sha,load
from src.architecture_features import representations
from src.features.fingerprints import featurize
from src.data.standardize import standardize_smiles


def main():
    directory=ROOT/'reports/current'
    report=json.loads((directory/'report.json').read_text())
    kaggle,public,audit=load()
    for relative,expected in report['provenance']['code_sha256'].items():
        source=(ROOT/relative).read_bytes().replace(b'\r\n',b'\n')
        equivalent={hashlib.sha256(source).hexdigest(),hashlib.sha256(source.replace(b'\n',b'\r\n')).hexdigest()}
        if expected not in equivalent:
            raise ValueError('Training source changed: '+relative)
    test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    template=pd.read_csv(ROOT/'data/raw/kaggle/sample_submit.csv',usecols=['index'])
    np.testing.assert_array_equal(test['index'],template['index'])
    raw=featurize(test.SMILES.tolist())
    encoded,metadata=representations([standardize_smiles(s) for s in test.SMILES],'chemeleon',8)
    checks=[]
    for row in report['final']:
        bundle=joblib.load(ROOT/row['artifact'])
        if bundle['provenance']['encoder']['sha256']!=metadata['checkpoint']['sha256']:
            raise ValueError('Different pretrained encoder checkpoint')
        if sha(ROOT/row['submission'])!=row['submission_sha256']:
            raise ValueError('Submission changed')
        matrix=raw if row['method'].startswith('catboost') else encoded
        predicted=bundle['model'].predict(matrix)
        original=pd.read_csv(ROOT/row['submission'])
        np.testing.assert_array_equal(original['index'],test['index'])
        np.testing.assert_allclose(predicted,original['SI VACV'],rtol=1e-4,atol=1e-4)
        if bundle['model'].competition_rows!=len(kaggle) or bundle['model'].zero_targets!=66:
            raise ValueError('Unexpected final competition cohort')
        expected_public=len(public) if row['regime']=='public_plus_kaggle' else 0
        if bundle['model'].public_rows!=expected_public:
            raise ValueError('Unexpected public cohort')
        checks.append({'candidate':row['candidate'],'rows':len(predicted),'passed':True,
            'maximum_absolute_difference':float(np.max(np.abs(predicted-original['SI VACV']))),
            'maximum_relative_difference':float(np.max(np.abs(predicted-original['SI VACV'])/np.maximum(np.abs(original['SI VACV']),1e-8)))})
    current=pd.read_csv(directory/'submission_catboost_direct__kaggle_only.csv')
    legacy=pd.read_csv(ROOT/'reports/testing_main_review/submission_v2_targets_nonnegative.csv')
    np.testing.assert_allclose(current['SI VACV'],legacy['SI VACV'],rtol=1e-12,atol=1e-9)
    result={'status':'passed','artifacts_verified':len(checks),'test_columns_read':['index','SMILES'],
        'test_endpoint_columns_read':[],'holdout_evaluated':False,
        'catboost_control_reproduces_testing_main':True,
        'dataset_version':audit['version'],'checks':checks}
    (directory/'inference_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
