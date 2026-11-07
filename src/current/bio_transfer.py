"""Public-only endpoint pretraining, followed by independent SI calibration.

Unlike the previous auxiliary experiment, the public predictor is trained first
and frozen before any competition SI head is fitted. Official endpoints are never
read; IC50, EC50 and CC50 stay separate public tasks and bounds stay one-sided.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,DIRECTORY,sha
from src.current.models import SIModel
from src.current.workflow import feature_inputs,select_fit_rows,metrics
from src.current.auxiliary import public_measurements,bound_errors
from src.architecture_features import representations
from src.data.standardize import standardize_smiles
from src.splits import murcko_scaffold

REPORTS=ROOT/'reports/current_bio_transfer'
ARTIFACTS=ROOT/'artifacts/current_bio_transfer'
KINDS=['base_small','bio_small','base_fp','bio_fp']
REGIMES=['kaggle_only','public_plus_kaggle']
EPOCHS=60


def public_network(dim):
    from torch import nn
    return nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Dropout(.1),
        nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3))


class BiologicalEncoder:
    def fit(self,x,measurements):
        import torch
        torch.set_num_threads(4);torch.manual_seed(20261106)
        x=np.asarray(x,float);self.mean=x.mean(0);self.scale=np.maximum(x.std(0),1e-6);self.dim=x.shape[1]
        self.centers=[];self.scales=[]
        for task in range(3):
            exact=measurements.loc[measurements.task.eq(task)&measurements.bound.eq(0),'log_value'].to_numpy()
            if not len(exact):raise ValueError('No exact public observations for task')
            self.centers.append(float(exact.mean()));self.scales.append(max(float(exact.std()),.1))
        tasks=measurements.task.to_numpy(int)
        yy=(measurements.log_value.to_numpy()-np.asarray(self.centers)[tasks])/np.asarray(self.scales)[tasks]
        inputs=torch.from_numpy(np.clip((x-self.mean)/self.scale,-10,10).astype(np.float32))
        target=torch.tensor(yy,dtype=torch.float32);task=torch.tensor(tasks,dtype=torch.long)
        direction=torch.tensor(measurements.bound.to_numpy(),dtype=torch.long)
        weight=torch.tensor(measurements.fit_weight.to_numpy(),dtype=torch.float32)
        model=public_network(self.dim);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        generator=torch.Generator().manual_seed(20261106);model.train()
        for epoch in range(EPOCHS):
            for batch in torch.randperm(len(inputs),generator=generator).split(128):
                value=model(inputs[batch])[torch.arange(len(batch)),task[batch]]
                errors=bound_errors(value,target[batch],direction[batch]).square()
                terms=[]
                for t in range(3):
                    mask=task[batch].eq(t)
                    if mask.any():terms.append((errors[mask]*weight[batch][mask]).sum()/weight[batch][mask].sum())
                loss=torch.stack(terms).mean();optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        self.state={k:v.detach().numpy() for k,v in model.state_dict().items()}
        self.measurement_rows=len(measurements)
        return self
    def predict_logs(self,x):
        import torch
        model=public_network(self.dim);model.load_state_dict({k:torch.from_numpy(v) for k,v in self.state.items()});model.eval()
        inputs=np.clip((np.asarray(x,float)-self.mean)/self.scale,-10,10).astype(np.float32)
        with torch.no_grad():value=model(torch.from_numpy(inputs)).numpy()
        return value*np.asarray(self.scales)[None,:]+np.asarray(self.centers)[None,:]


def design(kind,raw,biology,source):
    raw=np.asarray(raw,float);biology=np.asarray(biology,float)
    if kind not in KINDS or biology.shape!=(len(raw),3):raise ValueError('Invalid design specification')
    base=raw[:,2048:] if kind.endswith('small') else raw
    if kind.startswith('bio'):
        # log concentration predictions, then public log(CC50/IC50), log(CC50/EC50).
        base=np.column_stack([base,biology,biology[:,2]-biology[:,0],biology[:,2]-biology[:,1]])
    return np.column_stack([base,np.asarray(source,dtype=float)])


def run():
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/current.json').read_text());kaggle,public,test,matrices,encoder,audit=feature_inputs(cfg)
    extra=public_measurements();smiles=list(dict.fromkeys(extra.standard_smiles));external,meta=representations(smiles,'chemeleon',8)
    if meta['checkpoint']['sha256']!=encoder['checkpoint']['sha256']:raise ValueError('Encoder checkpoint changed')
    positions={s:i for i,s in enumerate(smiles)};extra['feature_row']=extra.standard_smiles.map(positions).astype(int)
    raw,encoded=matrices['catboost_direct'],matrices['chemeleon']
    provenance={'code_sha256':sha(Path(__file__)),'data':audit,'public_source_sha256':sha(ROOT/'data/experiments/independent_endpoints.csv'),
        'encoder':encoder,'public_pretraining_rows':len(extra),'public_pretraining_structures':len(smiles),
        'epochs':EPOCHS,'task_names':['IC50','EC50','CC50'],'kinds':KINDS,'regimes':REGIMES,
        'config':cfg,'official_training_columns':['SMILES','SI VACV'],'test_endpoint_columns_read':[],
        'holdout_used':False,'public_measurement_bounds':'Directional squared hinge; not substituted as exact observations',
        'selection':'ALL-row SI OOF primary; risk guards zero and positive SI<=1000 MSE <=1.2x fixed CheM reference; no automatic promotion'}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    (REPORTS/'plan.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    chunks=[]
    for ordinal,split in enumerate(json.loads((DIRECTORY/'splits.json').read_text())['folds']):
        stamp=f"{split['protocol']}_{split['repeat']}_{split['fold']}";path=REPORTS/f'fold_{stamp}.csv';marker=path.with_suffix('.json')
        if path.exists() and marker.exists() and json.loads(marker.read_text())['fingerprint']==fingerprint:
            chunks.append(pd.read_csv(path));print('[resume] '+stamp,flush=True);continue
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])]
        supplemental=extra[~extra.standard_smiles.isin(held.standard_smiles)].copy()
        if split['protocol']=='scaffold':supplemental=supplemental[~supplemental.standard_smiles.map(murcko_scaffold).isin(set(held.standard_smiles.map(murcko_scaffold)))]
        started=time.monotonic();print(f'[bio pretrain] fold{ordinal+1}/18 {stamp}, public rows={len(supplemental)}',flush=True)
        biology=BiologicalEncoder().fit(external[supplemental.feature_row.to_numpy(int)],supplemental)
        biological_features=biology.predict_logs(encoded)
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        for regime in REGIMES:
            fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],regime);fi=fit.matrix_row.to_numpy(int);hi=held.matrix_row.to_numpy(int)
            for kind in KINDS:
                head=SIModel('catboost_log1p',cfg).fit(design(kind,raw[fi],biological_features[fi],fit.source.eq('kaggle')),fit.SI,fit.weight,fit.source.eq('kaggle'))
                name=kind+'__'+regime;rows[name]=head.predict(design(kind,raw[hi],biological_features[hi],np.ones(len(held))))
                print(f"[bio OOF] {stamp} {name} MSE={metrics(held.SI,rows[name])['mse']:.2f}",flush=True)
        rows.to_csv(path,index=False);marker.write_text(json.dumps({'fingerprint':fingerprint,'public_fit_rows':len(supplemental),'seconds':time.monotonic()-started}),encoding='utf-8');chunks.append(rows)
    oof=pd.concat(chunks,ignore_index=True);ranking=[]
    for regime in REGIMES:
        for kind in KINDS:
            name=kind+'__'+regime;per={p:metrics(r.SI_true,r[name])['mse'] for p,r in oof.groupby('protocol')}
            ranking.append({'candidate':name,'kind':kind,'regime':regime,'mean_oof_mse':float(np.mean(list(per.values()))),**per})
    ranking.sort(key=lambda r:r['mean_oof_mse']);report={'selected':ranking[0]['candidate'],'ranking':ranking,'provenance':provenance,'official_score':None,'kaggle_submitted':False}
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');oof.to_csv(REPORTS/'oof.csv',index=False)
    biology=BiologicalEncoder().fit(external[extra.feature_row.to_numpy(int)],extra);biological_features=biology.predict_logs(encoded)
    begin=len(kaggle)+len(public)
    for row in ranking:
        fit=pd.concat([kaggle,public],ignore_index=True) if row['regime']=='public_plus_kaggle' else kaggle;fi=fit.matrix_row.to_numpy(int);kind=row['kind']
        head=SIModel('catboost_log1p',cfg).fit(design(kind,raw[fi],biological_features[fi],fit.source.eq('kaggle')),fit.SI,fit.weight,fit.source.eq('kaggle'))
        artifact=ARTIFACTS/(row['candidate']+'.joblib');joblib.dump({'biology':biology,'head':head,'kind':kind,'provenance':provenance},artifact,compress=3)
        restored=joblib.load(artifact);logs=restored['biology'].predict_logs(encoded[begin:]);values=restored['head'].predict(design(kind,raw[begin:],logs,np.ones(len(test))))
        path=REPORTS/('submission_'+row['candidate']+'.csv');pd.DataFrame({'index':test['index'],'SI VACV':values}).to_csv(path,index=False)
        row.update(artifact=artifact.relative_to(ROOT).as_posix(),submission=path.relative_to(ROOT).as_posix(),submission_sha256=sha(path))
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv',index=False)
    print('[bio complete] '+json.dumps(ranking[:3]),flush=True)


if __name__=='__main__':
    from src.current.bio_transfer import run as canonical_run
    with threadpool_limits(limits=4):canonical_run()
