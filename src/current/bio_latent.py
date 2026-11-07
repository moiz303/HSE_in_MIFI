"""Compare learned 32D public-biology features with output-only biological features.

Only representation changes: the same frozen public pretraining and SI log-head
are retained. Fold-specific learned states are cached to avoid repeated public
pretraining in future readout experiments. The queued candidate is unchanged.
"""
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,DIRECTORY,sha
from src.current.workflow import feature_inputs,select_fit_rows,metrics
from src.current.models import SIModel
from src.current.bio_transfer import BiologicalEncoder,public_network,design
from src.current.auxiliary import public_measurements
from src.architecture_features import representations
from src.data.standardize import standardize_smiles
from src.splits import murcko_scaffold

REPORTS=ROOT/'reports/current_bio_latent'
ARTIFACTS=ROOT/'artifacts/current_bio_latent'
KINDS=['latent_small','latent_fp','latent_chem']
REGIMES=['kaggle_only','public_plus_kaggle']


def latent_features(learner,x):
    import torch
    model=public_network(learner.dim);model.load_state_dict({k:torch.from_numpy(v) for k,v in learner.state.items()});model.eval()
    inputs=np.clip((np.asarray(x,float)-learner.mean)/learner.scale,-10,10).astype(np.float32)
    with torch.no_grad():hidden=model[:5](torch.from_numpy(inputs)).numpy()
    if hidden.shape!=(len(x),32) or not np.isfinite(hidden).all():raise ValueError('Invalid public latent features')
    return hidden


def fit_projection(encoded):
    scaler=StandardScaler().fit(encoded)
    pca=PCA(n_components=min(64,len(encoded)-1,encoded.shape[1]),svd_solver='full').fit(scaler.transform(encoded))
    return {'scaler':scaler,'pca':pca}


def make_features(kind,raw,encoded,logs,hidden,source,projection=None):
    if kind=='latent_small':base=np.asarray(raw)[:,2048:]
    elif kind=='latent_fp':base=np.asarray(raw)
    elif kind=='latent_chem':
        if projection is None:raise ValueError('Train-fitted chemical projection required')
        chemical=projection['pca'].transform(projection['scaler'].transform(encoded))
        base=np.column_stack([chemical,np.asarray(raw)[:,2048:]])
    else:raise ValueError('Unknown biological latent design')
    return np.column_stack([base,hidden,logs,logs[:,2]-logs[:,0],logs[:,2]-logs[:,1],np.asarray(source,float)])


def run():
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/current.json').read_text());kaggle,public,test,matrices,encoder,audit=feature_inputs(cfg)
    extra=public_measurements();unique=list(dict.fromkeys(extra.standard_smiles));external,_=representations(unique,'chemeleon',8)
    extra['feature_row']=extra.standard_smiles.map({s:i for i,s in enumerate(unique)}).astype(int)
    raw,encoded=matrices['catboost_direct'],matrices['chemeleon']
    feature_sources={'public_data':sha(ROOT/'data/experiments/independent_endpoints.csv'),
        'public_pretraining_code':sha(ROOT/'src/current/bio_transfer.py'),'public_preparation_code':sha(ROOT/'src/current/auxiliary.py'),
        'encoded_features':sha(ROOT/encoder['cache'])}
    provenance={'code_sha256':sha(Path(__file__)),'feature_sources':feature_sources,'data':audit,'encoder':encoder,'config':cfg,
        'kinds':KINDS,'regimes':REGIMES,'target':'log1p(SI), CatBoost600/depth5; source indicator',
        'novel_factor':'32 public-task-trained hidden features; chemical-input variant uses train-only PCA64; training/splits/objective otherwise fixed',
        'reference':'bio_fp__kaggle_only, mean OOF5014266.299876882',
        'selection':'ALL-row raw SI MSE, zero/bulk <=1.2x fixed Bio reference; no automatic upload',
        'queued_candidate_changed':False,'holdout_used':False,'official_endpoint_columns_read':[]}
    (REPORTS/'plan.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    previous=pd.read_csv(ROOT/'reports/current_bio_transfer/oof.csv');chunks=[]
    for ordinal,split in enumerate(json.loads((DIRECTORY/'splits.json').read_text())['folds']):
        stamp=f"{split['protocol']}_{split['repeat']}_{split['fold']}";path=REPORTS/f'fold_{stamp}.csv';marker=path.with_suffix('.json')
        if path.exists() and marker.exists() and json.loads(marker.read_text())['fingerprint']==fingerprint:
            chunks.append(pd.read_csv(path));print('[resume] '+stamp,flush=True);continue
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])]
        supplemental=extra[~extra.standard_smiles.isin(held.standard_smiles)].copy()
        if split['protocol']=='scaffold':supplemental=supplemental[~supplemental.standard_smiles.map(murcko_scaffold).isin(set(held.standard_smiles.map(murcko_scaffold)))]
        key=hashlib.sha256(json.dumps({'sources':feature_sources,'public_training_row_ids':supplemental.index.tolist()},sort_keys=True).encode()).hexdigest()
        cache=ROOT/'data/architecture_cache'/('public_bio_state_'+key+'.joblib')
        started=time.monotonic();print(f'[latent] fold{ordinal+1}/18 {stamp}, public measurements={len(supplemental)}',flush=True)
        old_key=hashlib.sha256(json.dumps({'sources':feature_sources,'split':split},sort_keys=True).encode()).hexdigest()
        old_cache=ROOT/'data/architecture_cache'/('public_bio_state_'+old_key+'.joblib')
        if cache.exists():learner=joblib.load(cache)['learner']
        elif old_cache.exists():
            saved=joblib.load(old_cache)
            if saved['sources']!=feature_sources or saved['split']!=split or saved['public_fit_rows']!=len(supplemental):raise ValueError('Legacy state mismatch')
            learner=saved['learner']
            joblib.dump({**saved,'public_training_row_ids':supplemental.index.tolist(),'reused_from':old_cache.name},cache,compress=3)
        else:
            learner=BiologicalEncoder().fit(external[supplemental.feature_row.to_numpy(int)],supplemental)
            joblib.dump({'learner':learner,'sources':feature_sources,'public_training_row_ids':supplemental.index.tolist(),'public_fit_rows':len(supplemental)},cache,compress=3)
        logs=learner.predict_logs(encoded);hidden=latent_features(learner,encoded)
        # The correct hidden tap must reconstruct the existing public head.
        reconstructed=(hidden@learner.state['5.weight'].T+learner.state['5.bias'])*np.asarray(learner.scales)+np.asarray(learner.centers)
        np.testing.assert_allclose(reconstructed,logs,rtol=1e-5,atol=1e-5)
        hi=held.matrix_row.to_numpy(int);fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],'kaggle_only');fi=fit.matrix_row.to_numpy(int)
        control=SIModel('catboost_log1p',cfg).fit(design('bio_fp',raw[fi],logs[fi],fit.source.eq('kaggle')),fit.SI,fit.weight,fit.source.eq('kaggle'))
        old=previous[previous.protocol.eq(split['protocol'])&previous.repeat.eq(split['repeat'])&previous.fold.eq(split['fold'])].set_index('original_row')
        np.testing.assert_allclose(control.predict(design('bio_fp',raw[hi],logs[hi],np.ones(len(held)))),held.original_row.map(old['bio_fp__kaggle_only']),rtol=1e-4,atol=1e-4)
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        for regime in REGIMES:
            fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],regime);fi=fit.matrix_row.to_numpy(int)
            for kind in KINDS:
                projection=fit_projection(encoded[fi]) if kind=='latent_chem' else None
                model=SIModel('catboost_log1p',cfg).fit(make_features(kind,raw[fi],encoded[fi],logs[fi],hidden[fi],fit.source.eq('kaggle'),projection),fit.SI,fit.weight,fit.source.eq('kaggle'))
                name=kind+'__'+regime;rows[name]=model.predict(make_features(kind,raw[hi],encoded[hi],logs[hi],hidden[hi],np.ones(len(held)),projection))
                print(f"[latent OOF] {stamp} {name} MSE={metrics(held.SI,rows[name])['mse']:.2f}",flush=True)
        rows.to_csv(path,index=False);marker.write_text(json.dumps({'fingerprint':fingerprint,'state_cache':cache.relative_to(ROOT).as_posix(),'control_reproduced':True,'seconds':time.monotonic()-started}),encoding='utf-8');chunks.append(rows)
    oof=pd.concat(chunks,ignore_index=True);ranking=[]
    for regime in REGIMES:
        for kind in KINDS:
            name=kind+'__'+regime;per={p:metrics(r.SI_true,r[name])['mse'] for p,r in oof.groupby('protocol')}
            ranking.append({'candidate':name,'kind':kind,'regime':regime,'mean_oof_mse':float(np.mean(list(per.values()))),**per})
    ranking.sort(key=lambda r:r['mean_oof_mse']);report={'selected':ranking[0]['candidate'],'ranking':ranking,'provenance':provenance,'official_score':None,'kaggle_submitted':False}
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');oof.to_csv(REPORTS/'oof.csv',index=False)
    learner=joblib.load(ROOT/'artifacts/current_bio_transfer/bio_fp__kaggle_only.joblib')['biology'];logs=learner.predict_logs(encoded);hidden=latent_features(learner,encoded);begin=len(kaggle)+len(public)
    for row in ranking:
        fit=pd.concat([kaggle,public],ignore_index=True) if row['regime']=='public_plus_kaggle' else kaggle;fi=fit.matrix_row.to_numpy(int);kind=row['kind']
        projection=fit_projection(encoded[fi]) if kind=='latent_chem' else None
        model=SIModel('catboost_log1p',cfg).fit(make_features(kind,raw[fi],encoded[fi],logs[fi],hidden[fi],fit.source.eq('kaggle'),projection),fit.SI,fit.weight,fit.source.eq('kaggle'))
        path=ARTIFACTS/(row['candidate']+'.joblib');joblib.dump({'biology':learner,'head':model,'kind':kind,'projection':projection,'provenance':provenance},path,compress=3)
        restored=joblib.load(path);test_logs=restored['biology'].predict_logs(encoded[begin:]);test_hidden=latent_features(restored['biology'],encoded[begin:])
        values=restored['head'].predict(make_features(kind,raw[begin:],encoded[begin:],test_logs,test_hidden,np.ones(len(test)),restored['projection']))
        submission=REPORTS/('submission_'+row['candidate']+'.csv');pd.DataFrame({'index':test['index'],'SI VACV':values}).to_csv(submission,index=False)
        row.update(artifact=path.relative_to(ROOT).as_posix(),submission=submission.relative_to(ROOT).as_posix(),submission_sha256=sha(submission))
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv',index=False)
    print('[latent complete] '+json.dumps(ranking[:3]),flush=True)


if __name__=='__main__':
    from src.current.bio_latent import run as canonical_run
    with threadpool_limits(limits=4):canonical_run()
