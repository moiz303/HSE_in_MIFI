"""Fresh-process numerical reproduction of auxiliary/kernel/GP submissions."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,sha
from src.features.fingerprints import featurize
from src.architecture_features import representations
from src.data.standardize import standardize_smiles


def main():
    test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    raw=featurize(test.SMILES.tolist())
    encoded,encoder=representations([standardize_smiles(s) for s in test.SMILES],'chemeleon',8)
    all_checks=[]
    for name,module,snapshot in [('search','search','source_before_portability_fix.py.txt'),
                                 ('auxiliary','auxiliary','source_at_training.py.txt'),
                                 ('probability','probabilistic','source_at_training.py.txt')]:
        directory=ROOT/('reports/current_'+name)
        report=json.loads((directory/'report.json').read_text())
        before=(directory/snapshot).read_text(encoding='utf-8')
        after=(ROOT/('src/current/'+module+'.py')).read_text(encoding='utf-8')
        if before.split("if __name__=='__main__':")[0]!=after.split("if __name__=='__main__':")[0]:
            raise ValueError('Learning/inference code changed after training: '+module)
        checks=[]
        for row in report['ranking']:
            if 'submission' not in row:continue
            if sha(ROOT/row['submission'])!=row['submission_sha256']:raise ValueError('Submission changed')
            bundle=joblib.load(ROOT/row['artifact']);model=bundle['model']
            if model.__class__.__module__=='__main__':raise ValueError('Nonportable artifact class')
            query=encoded if name=='auxiliary' else raw
            if name=='auxiliary' and bundle['provenance']['encoder']['checkpoint']['sha256']!=encoder['checkpoint']['sha256']:raise ValueError('Encoder changed')
            values=model.predict(query)
            if name=='probability':values=values[row['kind']]
            stored=pd.read_csv(ROOT/row['submission'])
            np.testing.assert_array_equal(stored['index'],test['index'])
            tolerance=1e-4 if name=='auxiliary' else 1e-10
            np.testing.assert_allclose(values,stored['SI VACV'],rtol=tolerance,atol=tolerance)
            checks.append({'candidate':row['candidate'],'rows':len(values),'maximum_absolute_difference':float(np.max(np.abs(values-stored['SI VACV']))),
                'canonical_class':model.__class__.__module__+'.'+model.__class__.__name__,'passed':True})
        result={'status':'passed','test_columns_read':['index','SMILES'],'learning_and_prediction_code_unchanged':True,'checks':checks}
        (directory/'inference_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(name,json.dumps(result),flush=True);all_checks+=checks
    path=ROOT/'reports/current/artifact_portability_fix.json';audit=json.loads(path.read_text())
    audit['requires_fresh_inference_verification']=False;audit['fresh_inference_predictions_verified']=len(all_checks)
    path.write_text(json.dumps(audit,indent=2),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
