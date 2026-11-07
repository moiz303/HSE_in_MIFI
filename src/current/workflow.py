"""Same held-out competition rows for every method/source regime; no endpoint inputs."""
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,DIRECTORY,load,sha
from src.current.models import SIModel
from src.architecture_features import representations
from src.features.fingerprints import featurize
from src.splits import murcko_scaffold

REPORTS=ROOT/'reports/current'
ARTIFACTS=ROOT/'artifacts/current'


def metrics(y,p):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    return {'mse':float(np.mean((y-p)**2)),'mae':float(np.mean(np.abs(y-p))),
            'rmse_log1p':float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2))),'rows':len(y)}


def select_fit_rows(kaggle,public,held_rows,protocol,regime):
    held=kaggle[kaggle.original_row.isin(held_rows)]
    comp_fit=kaggle[~kaggle.original_row.isin(held_rows)].copy()
    if set(comp_fit.standard_smiles)&set(held.standard_smiles):
        raise ValueError('Molecule leakage in competition split')
    if regime=='kaggle_only':
        return comp_fit
    extra=public[~public.standard_smiles.isin(held.standard_smiles)].copy()
    if protocol=='scaffold':
        held_scaffolds=set(held.standard_smiles.map(murcko_scaffold))
        extra=extra[~extra.standard_smiles.map(murcko_scaffold).isin(held_scaffolds)]
        if set(comp_fit.standard_smiles.map(murcko_scaffold))&held_scaffolds:
            raise ValueError('Scaffold leakage in competition split')
    return pd.concat([comp_fit,extra],ignore_index=True)


def feature_inputs(config):
    kaggle,public,audit=load()
    kaggle['matrix_row']=np.arange(len(kaggle))
    public['matrix_row']=np.arange(len(kaggle),len(kaggle)+len(public))
    # Test endpoints cannot enter features or labels: read only identifier and structure.
    test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    template=pd.read_csv(ROOT/'data/raw/kaggle/sample_submit.csv',usecols=['index'])
    if len(test)!=597 or test['index'].isna().any() or test['index'].duplicated().any() or not test['index'].equals(template['index']):
        raise ValueError('Invalid test/template identifier order')
    frame=pd.concat([kaggle,public],ignore_index=True)
    raw_smiles=frame.SMILES.tolist()+test.SMILES.tolist()
    from src.data.standardize import standardize_smiles
    encoder_smiles=[standardize_smiles(s) for s in raw_smiles]
    if any(s is None for s in encoder_smiles):
        raise ValueError('Invalid feature structure')
    matrix,encoder=representations(encoder_smiles,'chemeleon',8)
    matrices={'chemeleon':matrix,'catboost_direct':featurize(raw_smiles)}
    return kaggle,public,test,matrices,encoder,audit


def matrix_for(method,matrices):
    return matrices['catboost_direct' if method.startswith('catboost') else 'chemeleon']


def fit_model(method,frame,matrices,config):
    return SIModel(method,config).fit(matrix_for(method,matrices)[frame.matrix_row.to_numpy(int)],
        frame.SI.to_numpy(float),frame.weight.to_numpy(float),frame.source.eq('kaggle').to_numpy())


def run(config,worker_repeat=None,final_only=False):
    REPORTS.mkdir(parents=True,exist_ok=True)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    kaggle,public,test,matrices,encoder,data_audit=feature_inputs(config)
    splits=json.loads((DIRECTORY/'splits.json').read_text())['folds']
    code={p:sha(ROOT/p) for p in ['src/current/data.py','src/current/models.py','src/current/workflow.py']}
    provenance={'config':config,'data':data_audit,'code_sha256':code,'encoder':encoder['checkpoint']}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    chunks,fit_audits=[],[]
    for split in splits:
        if worker_repeat is not None and split['repeat']!=worker_repeat:
            continue
        stamp=f"{split['protocol']}_{split['repeat']}_{split['fold']}"
        chunk_path=REPORTS/f'oof_{stamp}.csv'
        audit_path=REPORTS/f'audit_{stamp}.json'
        if chunk_path.exists() and audit_path.exists() and json.loads(audit_path.read_text())['fingerprint']==fingerprint:
            chunks.append(pd.read_csv(chunk_path));fit_audits.append(json.loads(audit_path.read_text()))
            print('[resume] '+stamp,flush=True)
            continue
        if final_only:
            raise ValueError('Missing verified fold: '+stamp)
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])].copy()
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'})
        rows['protocol'],rows['repeat'],rows['fold']=split['protocol'],split['repeat'],split['fold']
        audit={'fingerprint':fingerprint,**split,'fits':[]}
        print('[current OOF] '+stamp,flush=True)
        fit_comp=kaggle[~kaggle.original_row.isin(split['held_original_rows'])]
        rows['constant_mean']=float(fit_comp.SI.mean())
        rows['constant_zero']=0.
        for regime in config['regimes']:
            fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],regime)
            for method in config['methods']:
                key=method+'__'+regime
                start=time.monotonic()
                model=fit_model(method,fit,matrices,config)
                values=model.predict(matrix_for(method,matrices)[held.matrix_row.to_numpy(int)])
                rows[key]=values
                audit['fits'].append({'method':method,'regime':regime,'fit_rows':model.fit_rows,
                    'competition_rows':model.competition_rows,'public_rows':model.public_rows,
                    'zero_targets':model.zero_targets,'seconds':time.monotonic()-start})
                print(f"[current OOF] {key}: MSE={metrics(held.SI,values)['mse']:.5f}; {time.monotonic()-start:.1f}s",flush=True)
        rows.to_csv(chunk_path,index=False)
        audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
        chunks.append(rows);fit_audits.append(audit)
    if worker_repeat is not None:
        print('[worker done] '+str(worker_repeat),flush=True)
        return
    oof=pd.concat(chunks,ignore_index=True)
    names=[m+'__'+r for m in config['methods'] for r in config['regimes']]
    summary={}
    for protocol,rows in oof.groupby('protocol'):
        if len(rows)!=len(kaggle)*config['repeats'] or not rows.groupby('original_row').size().eq(config['repeats']).all():
            raise ValueError('Invalid OOF row coverage')
        summary[protocol]={}
        for name in names+['constant_mean','constant_zero']:
            values=metrics(rows.SI_true,rows[name])
            values['repeat_mse']=[metrics(r.SI_true,r[name])['mse'] for _,r in rows.groupby('repeat')]
            for subset,mask in [('zero',rows.SI_true.eq(0)),('positive',rows.SI_true.gt(0))]:
                values[subset]=metrics(rows.loc[mask,'SI_true'],rows.loc[mask,name])
            summary[protocol][name]=values
    selected=min(names,key=lambda n:np.mean([summary[p][n]['mse'] for p in summary]))
    selection={'selected':selected,'rule':config['selection'],'test_endpoints_used':False,
        'official_score_known':False,'automatic_promotion':False,'kaggle_submitted':False}
    (REPORTS/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    oof.to_csv(REPORTS/'oof.csv',index=False)
    print('[OOF selected] '+selected,flush=True)
    final=[]
    # Save all predictions before any separately launched endpoint diagnostics.
    for regime in config['regimes']:
        fit=pd.concat([kaggle,public],ignore_index=True) if regime=='public_plus_kaggle' else kaggle
        for method in config['methods']:
            key=method+'__'+regime
            print('[final fit] '+key,flush=True)
            model=fit_model(method,fit,matrices,config)
            path=ARTIFACTS/(key+'.joblib')
            joblib.dump({'model':model,'provenance':provenance,'method':method,'regime':regime},path,compress=3)
            restored=joblib.load(path)['model']
            query=matrix_for(method,matrices)[len(kaggle)+len(public):]
            values=restored.predict(query)
            np.testing.assert_array_equal(values,model.predict(query))
            submission=REPORTS/('submission_'+key+'.csv')
            pd.DataFrame({'index':test['index'],'SI VACV':values}).to_csv(submission,index=False)
            final.append({'candidate':key,'method':method,'regime':regime,
                'artifact':path.relative_to(ROOT).as_posix(),'submission':submission.relative_to(ROOT).as_posix(),
                'submission_sha256':sha(submission),'fit_rows':model.fit_rows,'competition_rows':model.competition_rows,
                'public_rows':model.public_rows,'zero_targets':model.zero_targets,'official_kaggle_score':None})
    report={'provenance':provenance,'fingerprint':fingerprint,'protocols':summary,'selection':selection,
        'fit_audits':fit_audits,'final':final,'holdout_used':False,'test_endpoint_columns_read':[],
        'kaggle_submitted':False}
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    chosen=next(row for row in final if row['candidate']==selected)
    (REPORTS/'best_candidate.json').write_text(json.dumps({**chosen,**selection},indent=2),encoding='utf-8')
    print('[current done] '+selected,flush=True)


def predict(artifact,test_csv,output):
    bundle=joblib.load(ROOT/artifact)
    test=pd.read_csv(ROOT/test_csv,usecols=['index','SMILES'])
    if test['index'].isna().any() or test['index'].duplicated().any():
        raise ValueError('Invalid inference IDs')
    if bundle['method'].startswith('catboost'):
        matrix=featurize(test.SMILES.tolist())
    else:
        from src.data.standardize import standardize_smiles
        smiles=[standardize_smiles(s) for s in test.SMILES]
        if any(s is None for s in smiles):
            raise ValueError('Invalid inference SMILES')
        matrix,metadata=representations(smiles,'chemeleon',8)
        if metadata['checkpoint']['sha256']!=bundle['provenance']['encoder']['sha256']:
            raise ValueError('Checkpoint differs from model provenance')
    values=bundle['model'].predict(matrix)
    path=ROOT/output
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'index':test['index'],'SI VACV':values}).to_csv(path,index=False)
