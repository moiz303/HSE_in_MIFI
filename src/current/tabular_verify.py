"""Rebuild author TabPFN contexts in a fresh process on IDs/SMILES only."""
import json
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,sha
from src.current.tabular import predictions,REPORTS
from src.architecture_features import representations
from src.features.fingerprints import featurize
from src.data.standardize import standardize_smiles


def main():
    context=joblib.load(ROOT/'artifacts/current_tabular/context.joblib')
    test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    matrix,meta=representations([standardize_smiles(s) for s in test.SMILES],'chemeleon',8)
    if meta['checkpoint']['sha256']!=context['provenance']['encoder']['checkpoint']['sha256']:raise ValueError('Encoder changed')
    projection=context['transforms']
    query=np.column_stack([projection['pca'].transform(projection['scaler'].transform(matrix)),featurize(test.SMILES.tolist())[:,2048:]])
    values=predictions(context['train'],context['targets'],query)
    report=json.loads((REPORTS/'report.json').read_text());checks=[]
    for row in report['ranking']:
        if sha(ROOT/row['submission'])!=row['submission_sha256']:raise ValueError('Submission bytes changed')
        stored=pd.read_csv(ROOT/row['submission'])
        np.testing.assert_array_equal(test['index'],stored['index'])
        prediction=values[row['candidate']]
        np.testing.assert_allclose(prediction,stored['SI VACV'],rtol=1e-3,atol=1e-3)
        checks.append({'candidate':row['candidate'],'maximum_absolute_difference':float(np.max(np.abs(prediction-stored['SI VACV']))),'rows':len(stored),'passed':True})
    result={'status':'passed','test_columns_read':['index','SMILES'],'checks':checks}
    (REPORTS/'inference_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
