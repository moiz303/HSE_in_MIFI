"""Fresh-process check of public latent features and stored train-only projections."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,sha
from src.current.bio_latent import latent_features,make_features,REPORTS
from src.architecture_features import representations
from src.features.fingerprints import featurize
from src.data.standardize import standardize_smiles


def main():
    report=json.loads((REPORTS/'report.json').read_text());test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    raw=featurize(test.SMILES.tolist());encoded,encoder=representations([standardize_smiles(s) for s in test.SMILES],'chemeleon',8)
    checks=[]
    for row in report['ranking']:
        if sha(ROOT/row['submission'])!=row['submission_sha256']:raise ValueError('Submission changed')
        bundle=joblib.load(ROOT/row['artifact']);learner=bundle['biology']
        if learner.__class__.__module__!='src.current.bio_transfer':raise ValueError('Nonportable public model')
        if bundle['provenance']['encoder']['checkpoint']['sha256']!=encoder['checkpoint']['sha256']:raise ValueError('Author checkpoint changed')
        logs=learner.predict_logs(encoded);hidden=latent_features(learner,encoded)
        matrix=make_features(bundle['kind'],raw,encoded,logs,hidden,np.ones(len(test)),bundle['projection'])
        values=bundle['head'].predict(matrix);stored=pd.read_csv(ROOT/row['submission'])
        np.testing.assert_array_equal(stored['index'],test['index']);np.testing.assert_allclose(values,stored['SI VACV'],rtol=1e-4,atol=1e-4)
        checks.append({'candidate':row['candidate'],'rows':len(values),'maximum_absolute_difference':float(np.max(np.abs(values-stored['SI VACV']))),'passed':True})
    result={'status':'passed','test_columns_read':['index','SMILES'],'checks':checks}
    (REPORTS/'inference_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
