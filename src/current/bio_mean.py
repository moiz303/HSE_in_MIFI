"""Direct SI conditional-mean calibration on frozen public biological features.

Eight predeclared controls: RMSE/Tweedie1.5 x small/fingerprint x two sources.
Public-feature caches are fold-specific and verified by reproducing the previous
log-SI OOF control. The already queued log-SI candidate is never changed.
"""
import copy
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,DIRECTORY,sha
from src.current.workflow import feature_inputs,select_fit_rows,metrics
from src.current.models import SIModel
from src.current.bio_transfer import BiologicalEncoder,design
from src.current.auxiliary import public_measurements
from src.architecture_features import representations
from src.data.standardize import standardize_smiles
from src.splits import murcko_scaffold

REPORTS=ROOT/'reports/current_bio_mean'
ARTIFACTS=ROOT/'artifacts/current_bio_mean'
LOSSES={'rmse':'RMSE','tweedie15':'Tweedie:variance_power=1.5'}
KINDS=['bio_small','bio_fp']
REGIMES=['kaggle_only','public_plus_kaggle']


class MeanCalibration:
    def __init__(self,loss,config):self.loss,self.config=loss,config;self.unit=100.
    def fit(self,x,y,weight,source):
        cfg=copy.deepcopy(self.config);cfg['catboost']['loss_function']=LOSSES[self.loss]
        self.head=SIModel('catboost_direct',cfg).fit(x,np.asarray(y,float)/self.unit,weight,source)
        return self
    def predict(self,x):return self.head.predict(x)*self.unit


def run():
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/current.json').read_text());kaggle,public,test,matrices,encoder,audit=feature_inputs(cfg)
    extra=public_measurements();unique=list(dict.fromkeys(extra.standard_smiles));external,meta=representations(unique,'chemeleon',8)
    assert meta['checkpoint']['sha256']==encoder['checkpoint']['sha256']
    extra['feature_row']=extra.standard_smiles.map({s:i for i,s in enumerate(unique)}).astype(int)
    encoded,raw=matrices['chemeleon'],matrices['catboost_direct']
    previous=pd.read_csv(ROOT/'reports/current_bio_transfer/oof.csv')
    feature_sources={'public_source_sha256':sha(ROOT/'data/experiments/independent_endpoints.csv'),
        'biology_code_sha256':sha(ROOT/'src/current/bio_transfer.py'),
        'public_preparation_code_sha256':sha(ROOT/'src/current/auxiliary.py'),
        'encoded_feature_file_sha256':sha(ROOT/encoder['cache'])}
    provenance={'code_sha256':sha(Path(__file__)),'feature_sources':feature_sources,'data':audit,'config':cfg,
        'losses':LOSSES,'kinds':KINDS,'regimes':REGIMES,'target_unit':100.,'target_transform':'SI/100, no logarithm; restore x100',
        'Tweedie_prediction':'CatBoost default Exponent, verified in installed package; no additional exponential',
        'official_training_columns':['SMILES','SI VACV'],'test_endpoint_columns_read':[],'holdout_used':False,
        'selection':'ALL-row raw-SI OOF, zero and positive<=1000 guards <=1.2x fixed CheM; no automatic promotion',
        'queued_candidate_changed':False}
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
        key=hashlib.sha256(json.dumps({'features':feature_sources,'split':split},sort_keys=True).encode()).hexdigest()
        cache=ROOT/'data/architecture_cache'/('public_bio_fold_'+key+'.npz')
        started=time.monotonic();print(f'[bio mean] fold{ordinal+1}/18 {stamp}',flush=True)
        if cache.exists():biology=np.load(cache)['biology']
        else:
            learner=BiologicalEncoder().fit(external[supplemental.feature_row.to_numpy(int)],supplemental)
            biology=learner.predict_logs(encoded);np.savez_compressed(cache,biology=biology)
        if biology.shape!=(len(encoded),3) or not np.isfinite(biology).all():raise ValueError('Bad public feature cache')
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        # Verify cache/training equivalence before comparing new target objectives.
        fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],'kaggle_only');fi=fit.matrix_row.to_numpy(int);hi=held.matrix_row.to_numpy(int)
        control=SIModel('catboost_log1p',cfg).fit(design('bio_fp',raw[fi],biology[fi],fit.source.eq('kaggle')),fit.SI,fit.weight,fit.source.eq('kaggle'))
        observed=control.predict(design('bio_fp',raw[hi],biology[hi],np.ones(len(held))))
        old=previous[previous.protocol.eq(split['protocol'])&previous.repeat.eq(split['repeat'])&previous.fold.eq(split['fold'])].set_index('original_row')
        np.testing.assert_allclose(observed,held.original_row.map(old['bio_fp__kaggle_only']),rtol=1e-4,atol=1e-4)
        for regime in REGIMES:
            fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],regime);fi=fit.matrix_row.to_numpy(int)
            for kind in KINDS:
                x=design(kind,raw[fi],biology[fi],fit.source.eq('kaggle'));query=design(kind,raw[hi],biology[hi],np.ones(len(held)))
                for loss in LOSSES:
                    name=f'{loss}__{kind}__{regime}'
                    model=MeanCalibration(loss,cfg).fit(x,fit.SI,fit.weight,fit.source.eq('kaggle'));rows[name]=model.predict(query)
                    print(f"[bio mean] {stamp} {name} MSE={metrics(held.SI,rows[name])['mse']:.2f}",flush=True)
        rows.to_csv(path,index=False);marker.write_text(json.dumps({'fingerprint':fingerprint,'feature_cache':cache.relative_to(ROOT).as_posix(),'log_control_reproduced':True,'seconds':time.monotonic()-started}),encoding='utf-8');chunks.append(rows)
    oof=pd.concat(chunks,ignore_index=True);ranking=[]
    for regime in REGIMES:
        for kind in KINDS:
            for loss in LOSSES:
                name=f'{loss}__{kind}__{regime}';per={p:metrics(r.SI_true,r[name])['mse'] for p,r in oof.groupby('protocol')}
                ranking.append({'candidate':name,'kind':kind,'loss':loss,'regime':regime,'mean_oof_mse':float(np.mean(list(per.values()))),**per})
    ranking.sort(key=lambda r:r['mean_oof_mse']);report={'selected':ranking[0]['candidate'],'ranking':ranking,'provenance':provenance,'official_score':None,'kaggle_submitted':False}
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');oof.to_csv(REPORTS/'oof.csv',index=False)
    public_learner=joblib.load(ROOT/'artifacts/current_bio_transfer/bio_fp__kaggle_only.joblib')['biology'];biology=public_learner.predict_logs(encoded)
    begin=len(kaggle)+len(public)
    for row in ranking:
        fit=pd.concat([kaggle,public],ignore_index=True) if row['regime']=='public_plus_kaggle' else kaggle;fi=fit.matrix_row.to_numpy(int)
        model=MeanCalibration(row['loss'],cfg).fit(design(row['kind'],raw[fi],biology[fi],fit.source.eq('kaggle')),fit.SI,fit.weight,fit.source.eq('kaggle'))
        artifact=ARTIFACTS/(row['candidate']+'.joblib');joblib.dump({'biology':public_learner,'head':model,'kind':row['kind'],'provenance':provenance},artifact,compress=3)
        restored=joblib.load(artifact);logs=restored['biology'].predict_logs(encoded[begin:]);values=restored['head'].predict(design(row['kind'],raw[begin:],logs,np.ones(len(test))))
        submission=REPORTS/('submission_'+row['candidate']+'.csv');pd.DataFrame({'index':test['index'],'SI VACV':values}).to_csv(submission,index=False)
        row.update(artifact=artifact.relative_to(ROOT).as_posix(),submission=submission.relative_to(ROOT).as_posix(),submission_sha256=sha(submission))
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv',index=False)
    print('[bio mean complete] '+json.dumps(ranking[:3]),flush=True)


if __name__=='__main__':
    from src.current.bio_mean import run as canonical_run
    with threadpool_limits(limits=4):canonical_run()
