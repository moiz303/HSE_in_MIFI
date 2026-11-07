"""Verify the completed public-pretraining experiment from test IDs/SMILES only."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,sha
from src.current.bio_transfer import design,REPORTS
from src.architecture_features import representations
from src.features.fingerprints import featurize
from src.data.standardize import standardize_smiles


def main():
    report=json.loads((REPORTS/'report.json').read_text())
    test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    raw=featurize(test.SMILES.tolist())
    encoded,encoder=representations([standardize_smiles(s) for s in test.SMILES],'chemeleon',8)
    checks=[]
    for row in report['ranking']:
        if sha(ROOT/row['submission'])!=row['submission_sha256']:raise ValueError('Submission changed')
        bundle=joblib.load(ROOT/row['artifact'])
        if bundle['biology'].__class__.__module__!='src.current.bio_transfer':raise ValueError('Nonportable class')
        if bundle['provenance']['encoder']['checkpoint']['sha256']!=encoder['checkpoint']['sha256']:raise ValueError('Author encoder changed')
        logs=bundle['biology'].predict_logs(encoded)
        values=bundle['head'].predict(design(bundle['kind'],raw,logs,np.ones(len(test))))
        original=pd.read_csv(ROOT/row['submission'])
        np.testing.assert_array_equal(original['index'],test['index'])
        np.testing.assert_allclose(values,original['SI VACV'],rtol=1e-4,atol=1e-4)
        checks.append({'candidate':row['candidate'],'rows':len(values),'maximum_absolute_difference':float(np.max(np.abs(values-original['SI VACV']))),'passed':True})
    result={'status':'passed','test_columns_read':['index','SMILES'],'checks':checks}
    (REPORTS/'inference_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
