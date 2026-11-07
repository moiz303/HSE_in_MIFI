"""Optional Prior Labs TabPFN-v2 on train-fold PCA CheMeleon + RDKit descriptors.

Original TabPFN: Hollmann et al., Nature (2025), doi:10.1038/s41586-024-08328-6.
Weights: Prior-Labs/TabPFN-v2-reg, Prior Labs License with attribution.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,DIRECTORY,sha
from src.current.workflow import feature_inputs,metrics

CHECKPOINT=ROOT/'artifacts/pretrained/tabpfn-v2-regressor.ckpt'
EXPECTED='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
REPORTS=ROOT/'reports/current_tabular'
ARTIFACTS=ROOT/'artifacts/current_tabular'
QUANTILES=[.025+.05*i for i in range(20)]


def estimator():
    if sha(CHECKPOINT)!=EXPECTED:raise ValueError('Unexpected author checkpoint')
    # Local optional installation avoids changing the architecture environment.
    dependency=ROOT.parent/'tabpfn_deps'
    if dependency.exists():sys.path.insert(0,str(dependency))
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(n_estimators=2,model_path=CHECKPOINT,device='cpu',
        random_state=20261106,n_jobs=1)


def projection(train_encoder,query_encoder,train_raw,query_raw):
    scaler=StandardScaler().fit(train_encoder)
    pca=PCA(n_components=min(32,len(train_encoder)-1),svd_solver='full').fit(scaler.transform(train_encoder))
    train=np.column_stack([pca.transform(scaler.transform(train_encoder)),train_raw[:,2048:]])
    query=np.column_stack([pca.transform(scaler.transform(query_encoder)),query_raw[:,2048:]])
    return train,query,{'scaler':scaler,'pca':pca}


def predictions(train,targets,query):
    raw=estimator().fit(train,targets)
    raw_predictions=np.maximum(raw.predict(query),0.)
    logarithmic=estimator().fit(train,np.log1p(targets))
    # Coarse numerical expectation in original SI units, not exp(E[log SI]).
    quantiles=np.asarray(logarithmic.predict(query,output_type='quantiles',quantiles=QUANTILES+[.5]))
    if quantiles.shape!=(21,len(query)):raise ValueError('Unexpected TabPFN quantile orientation')
    inverse=np.maximum(np.expm1(np.clip(quantiles,-30,30)),0.)
    return {'tabpfn_raw':raw_predictions,'tabpfn_log_median':inverse[-1],
            'tabpfn_log_mean20':inverse[:-1].mean(0)}


def run():
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/current.json').read_text())
    kaggle,public,test,matrices,encoder,audit=feature_inputs(config)
    raw=matrices['catboost_direct'];encoded=matrices['chemeleon']
    provenance={'dataset':audit,'code_sha256':sha(Path(__file__)),'encoder':encoder,
        'tabpfn_checkpoint_sha256':EXPECTED,'tabpfn_version':'2.1.3','estimators':2,
        'target_variants':['raw','log_median','log_mean20'],'quantiles':QUANTILES,
        'features':'32 train-fold PCA CheMeleon components + 13 RDKit descriptors',
        'regime':'kaggle_only','test_endpoints_used':False,'holdout_used':False}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    chunks=[]
    for split in json.loads((DIRECTORY/'splits.json').read_text())['folds']:
        stamp=f"{split['protocol']}_{split['repeat']}_{split['fold']}"
        path=REPORTS/f'fold_{stamp}.csv';marker=path.with_suffix('.json')
        if path.exists() and marker.exists() and json.loads(marker.read_text())['fingerprint']==fingerprint:
            chunks.append(pd.read_csv(path));continue
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])]
        fit=kaggle[~kaggle.original_row.isin(split['held_original_rows'])]
        if set(fit.standard_smiles)&set(held.standard_smiles):raise ValueError('Molecule leakage')
        fi=fit.matrix_row.to_numpy(int);hi=held.matrix_row.to_numpy(int)
        x,q,_=projection(encoded[fi],encoded[hi],raw[fi],raw[hi])
        started=time.monotonic();values=predictions(x,fit.SI.to_numpy(float),q)
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        for name,value in values.items():
            rows[name]=value
            print(f"[tabular] {stamp} {name} MSE={metrics(held.SI,value)['mse']:.3f}",flush=True)
        rows.to_csv(path,index=False);marker.write_text(json.dumps({'fingerprint':fingerprint}),encoding='utf-8')
        chunks.append(rows);print(f'[tabular] seconds={time.monotonic()-started:.1f}',flush=True)
    oof=pd.concat(chunks,ignore_index=True);ranking=[]
    for name in ['tabpfn_raw','tabpfn_log_median','tabpfn_log_mean20']:
        per={p:metrics(r.SI_true,r[name])['mse'] for p,r in oof.groupby('protocol')}
        ranking.append({'candidate':name,'mean_oof_mse':float(np.mean(list(per.values()))),**per})
    ranking.sort(key=lambda r:r['mean_oof_mse'])
    report={'selection':ranking[0]['candidate'],'ranking':ranking,'provenance':provenance,
        'selection_rule':'Mean ALL-row SI MSE across fixed molecule/scaffold OOF',
        'official_kaggle_score':None,'kaggle_submissions_sent':0}
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    oof.to_csv(REPORTS/'oof.csv',index=False)
    count=len(kaggle);start=count+len(public)
    x,q,transforms=projection(encoded[:count],encoded[start:],raw[:count],raw[start:])
    values=predictions(x,kaggle.SI.to_numpy(float),q)
    # TabPFN fit is contextual: store the train context plus projection, not repeated author weights.
    context=ARTIFACTS/'context.joblib'
    joblib.dump({'train':x,'targets':kaggle.SI.to_numpy(float),'transforms':transforms,'provenance':provenance},context,compress=3)
    for row in ranking:
        path=REPORTS/f"submission_{row['candidate']}.csv"
        pd.DataFrame({'index':test['index'],'SI VACV':values[row['candidate']]}).to_csv(path,index=False)
        row.update(submission=path.relative_to(ROOT).as_posix(),submission_sha256=sha(path),artifact=context.relative_to(ROOT).as_posix())
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv',index=False)
    print('[tabular done] '+json.dumps(ranking),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run()
