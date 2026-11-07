"""Weighted, source-aware extension of the SI Gaussian process to public pairs."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve,solve_triangular
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,DIRECTORY,sha
from src.current.workflow import feature_inputs,select_fit_rows,metrics
from src.current.probabilistic import objective
from src.current.search import similarity

REPORTS=ROOT/'reports/current_power_ensemble'
ARTIFACTS=ROOT/'artifacts/current_power_ensemble'


class PowerEnsemble:
    def __init__(self,power,config):self.power,self.config=power,config
    def fit(self,x,y,weights,source):
        from src.current.models import SIModel
        import copy
        self.models=[]
        for offset in [0,11,22]:
            cfg=copy.deepcopy(self.config);cfg['seed']+=offset
            if self.power==0:
                model=SIModel('chemeleon_domain_log1p',cfg).fit(x,y,weights,source)
            else:
                transformed=np.expm1(self.power*np.log1p(np.asarray(y,float)))/self.power
                model=SIModel('chemeleon_domain',cfg).fit(x,transformed,weights,source)
            self.models.append(model)
        return self
    def individual(self,x):
        values=[]
        for model in self.models:
            prediction=model.predict(x)
            if self.power!=0:prediction=np.expm1(np.log1p(self.power*prediction)/self.power)
            if not np.isfinite(prediction).all():raise ValueError('Nonfinite power inverse')
            values.append(np.maximum(prediction,0.))
        return np.asarray(values)
    def predict(self,x):return self.individual(x).mean(0)


KINDS=[0.,.25,.5]
REGIMES=['kaggle_only','public_plus_kaggle']


def run():
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/current.json').read_text())
    kaggle,public,test,matrices,encoder,audit=feature_inputs(cfg)
    matrix=matrices['chemeleon']
    chunks=[];audits=[]
    folds=json.loads((DIRECTORY/'splits.json').read_text())['folds']
    for ordinal,split in enumerate(folds):
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])]
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        for regime in REGIMES:
            fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],regime)
            for kind in KINDS:
                model=PowerEnsemble(kind,cfg).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI,fit.weight,fit.source.eq('kaggle'))
                rows[f'{kind}_{regime}']=model.predict(matrix[held.matrix_row.to_numpy(int)])
                if kind==0 and regime=='public_plus_kaggle':rows['single_seed_control']=model.models[0].predict(matrix[held.matrix_row.to_numpy(int)])
                audits.append({**split,'regime':regime,'kind':kind,'fit_rows':len(fit)})
        chunks.append(rows);print(f"[power ensemble] fold {ordinal+1}/{len(folds)}",flush=True)
    oof=pd.concat(chunks,ignore_index=True);ranking=[]
    assert len(oof)==267*3*2
    reference=pd.read_csv(ROOT/'reports/current/oof.csv')
    joined=oof.merge(reference[['protocol','repeat','fold','original_row','chemeleon_domain_log1p__public_plus_kaggle']],on=['protocol','repeat','fold','original_row'],validate='one_to_one')
    np.testing.assert_allclose(joined.single_seed_control,joined.chemeleon_domain_log1p__public_plus_kaggle,rtol=1e-4,atol=1e-4)
    for regime in REGIMES:
        for kind in KINDS:
            for outcome in ['ensemble']:
                name=f'{kind}_{regime}';per={p:metrics(r.SI_true,r[name])['mse'] for p,r in oof.groupby('protocol')}
                ranking.append({'candidate':name,'kernel':kind,'outcome':outcome,'regime':regime,'mean_oof_mse':float(np.mean(list(per.values()))),**per})
    ranking.sort(key=lambda r:r['mean_oof_mse'])
    provenance={'data':audit,'code_sha256':sha(Path(__file__)),'objective_code_sha256':sha(ROOT/'src/current/probabilistic.py'),
        'encoder':encoder,'seeds':[cfg['seed']+o for o in [0,11,22]],'target':'Box-Cox of SI+1 powers0,.25,.5; average inverse predictions in SI units; competition-only target scaler',
        'public_weights':'0.5/records per structure -> observation noise divided by weight',
        'hyperparameters':'Baseline160epochs, fixed architecture64-32, seed ensemble only','holdout_used':False,'test_endpoints_used':False}
    report={'selected':ranking[0]['candidate'],'ranking':ranking,'provenance':provenance,'fit_audits':audits,'official_score':None}
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');oof.to_csv(REPORTS/'oof.csv',index=False)
    for regime in REGIMES:
        fit=pd.concat([kaggle,public],ignore_index=True) if regime=='public_plus_kaggle' else kaggle
        query=matrix[len(kaggle)+len(public):]
        for kind in KINDS:
            model=PowerEnsemble(kind,cfg).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI,fit.weight,fit.source.eq('kaggle'))
            artifact=ARTIFACTS/f'{kind}_{regime}.joblib';joblib.dump({'model':model,'provenance':provenance},artifact,compress=3)
            restored=joblib.load(artifact)['model'];values=restored.predict(query)
            for row in ranking:
                if (row['kernel'],row['regime'])!=(kind,regime):continue
                np.testing.assert_allclose(values,model.predict(query),rtol=1e-10,atol=1e-10)
                path=REPORTS/('submission_'+row['candidate']+'.csv');pd.DataFrame({'index':test['index'],'SI VACV':values}).to_csv(path,index=False)
                row.update(artifact=artifact.relative_to(ROOT).as_posix(),submission=path.relative_to(ROOT).as_posix(),submission_sha256=sha(path))
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv',index=False)
    print('[power ensemble done] '+json.dumps(ranking[:3]),flush=True)


def verify():
    from src.features.fingerprints import featurize
    from src.architecture_features import representations
    from src.data.standardize import standardize_smiles
    test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    encoded,_=representations([standardize_smiles(s) for s in test.SMILES],'chemeleon',8)
    query=encoded
    report=json.loads((REPORTS/'report.json').read_text());checks=[]
    for row in report['ranking']:
        assert sha(ROOT/row['submission'])==row['submission_sha256']
        model=joblib.load(ROOT/row['artifact'])['model'];assert model.__class__.__module__=='src.current.power_ensemble'
        value=model.predict(query);stored=pd.read_csv(ROOT/row['submission'])
        np.testing.assert_array_equal(test['index'],stored['index'])
        np.testing.assert_allclose(value,stored['SI VACV'],rtol=1e-4,atol=1e-4)
        checks.append({'candidate':row['candidate'],'passed':True,'maximum_absolute_difference':float(np.max(np.abs(value-stored['SI VACV'])))})
    (REPORTS/'inference_verification.json').write_text(json.dumps({'status':'passed','test_columns':['index','SMILES'],'checks':checks},indent=2),encoding='utf-8')
    print('Fresh process verification passed',len(checks),flush=True)


if __name__=='__main__':
    import sys
    from src.current.power_ensemble import run as canonical_run,verify as canonical_verify
    with threadpool_limits(limits=2):
        if '--verify' in sys.argv:canonical_verify()
        else:canonical_run()
