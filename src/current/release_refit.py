"""Reviewable final SI-only refit; reserved rows require explicit user release.

Default invocation only prepares a structural/count plan. It neither reads
reserved targets nor trains. --release-reserved-training-rows is a deliberate
operator assertion that the user has authorized consuming the reserve.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,load,sha,read_kaggle_training
from src.current.models import SIModel
from src.data.standardize import standardize_smiles
from src.architecture_features import representations

REPORTS=ROOT/'reports/current_release_refit'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-reserved-training-rows',action='store_true')
    args=parser.parse_args()
    plan=json.loads((ROOT/'configs/current_release_refit.json').read_text())
    cfg=json.loads((ROOT/plan['base_config']).read_text())
    development,public,audit=load()
    train_path=ROOT/'data/raw/kaggle/train_v2.csv'
    if sha(train_path)!=cfg['expected_train_sha256']:raise ValueError('Unexpected competition version')
    structures=pd.read_csv(train_path,usecols=['SMILES'])
    structures['standard_smiles']=structures.SMILES.map(standardize_smiles)
    if len(structures)!=320 or structures.standard_smiles.isna().any():raise ValueError('Invalid full competition structures')
    counts={**plan,'planned_fit_rows':len(structures)+len(public),
        'reserved_rows_to_release':len(structures)-len(development),
        'train_sha256':sha(train_path),'current_dataset_version':audit['version'],
        'no_training_performed':not args.release_reserved_training_rows,
        'reserved_targets_evaluated':False,'kaggle_submitted':False}
    REPORTS.mkdir(parents=True,exist_ok=True)
    (REPORTS/'plan.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    if not args.release_reserved_training_rows:
        print(json.dumps(counts,indent=2));return
    # Only after the explicit release flag do reserved SI values enter fitting.
    competition=read_kaggle_training(train_path)
    y=np.r_[competition.SI.to_numpy(float),public.SI.to_numpy(float)]
    source=np.r_[np.ones(len(competition),bool),np.zeros(len(public),bool)]
    weights=np.r_[np.ones(len(competition)),public.weight.to_numpy(float)]
    test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    template=pd.read_csv(ROOT/'data/raw/kaggle/sample_submit.csv',usecols=['index'])
    np.testing.assert_array_equal(test['index'],template['index'])
    smiles=structures.standard_smiles.tolist()+public.standard_smiles.tolist()+[standardize_smiles(s) for s in test.SMILES]
    encoded,encoder=representations(smiles,'chemeleon',8)
    model=SIModel(plan['method'],cfg).fit(encoded[:len(y)],y,weights,source)
    directory=ROOT/'artifacts/current_release_refit';directory.mkdir(parents=True,exist_ok=True)
    artifact=directory/'chemeleon_domain_log1p_full_competition.joblib'
    joblib.dump({'model':model,'config':cfg,'plan':plan,'data':counts,'encoder':encoder,'reserved_rows_released_for_fit':True},artifact,compress=3)
    restored=joblib.load(artifact)['model'];values=restored.predict(encoded[len(y):])
    np.testing.assert_allclose(values,model.predict(encoded[len(y):]),rtol=1e-6,atol=1e-6)
    submission=REPORTS/'submission_full_competition.csv'
    pd.DataFrame({'index':test['index'],'SI VACV':values}).to_csv(submission,index=False)
    report={**counts,'explicit_operator_release':True,'no_training_performed':False,
        'competition_fit_rows':len(competition),'public_fit_rows':len(public),
        'artifact':artifact.relative_to(ROOT).as_posix(),'submission':submission.relative_to(ROOT).as_posix(),
        'submission_sha256':sha(submission),'code_sha256':sha(Path(__file__)),
        'official_score':None,'independent_validation_score':None,
        'note':'Previously reserved competition rows consumed for frozen final refit only; no new independent validation claim'}
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
