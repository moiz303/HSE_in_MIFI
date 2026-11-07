"""Tanimoto Gaussian process: train-only empirical-Bayes log-SI uncertainty.

Fit hyperparameters by train marginal likelihood; compare median, latent mean
and noisy observation mean on identical ALL-row development folds.
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve,solve_triangular
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,DIRECTORY,sha
from src.current.workflow import feature_inputs,metrics
from src.current.search import similarity

REPORTS=ROOT/'reports/current_probability'
ARTIFACTS=ROOT/'artifacts/current_probability'


def objective(log_parameters,similarity_matrix,y,gradient=True):
    amplitude,noise=np.exp(log_parameters)
    covariance=amplitude*similarity_matrix+noise*np.eye(len(y))
    factor=cho_factor(covariance,lower=True)
    alpha=cho_solve(factor,y)
    value=.5*y@alpha+np.log(np.diag(factor[0])).sum()+.5*len(y)*np.log(2*np.pi)
    inverse=cho_solve(factor,np.eye(len(y)))
    residual=inverse-np.outer(alpha,alpha)
    derivative=.5*np.array([np.sum(residual*(amplitude*similarity_matrix)),noise*np.trace(residual)])
    return value,derivative


class LogSIProcess:
    def __init__(self,power):self.power=power
    def fit(self,x,y):
        target=np.log1p(np.asarray(y,dtype=float));self.center=float(target.mean());yy=target-self.center
        self.x=np.asarray(x,dtype=float);kernel=similarity(x,x)**self.power
        variance=max(float(np.var(target)),.1)
        fit=minimize(objective,np.log([variance/2,variance/2]),args=(kernel,yy),jac=True,
            bounds=[(np.log(.01),np.log(100.)),(np.log(.05),np.log(100.))],method='L-BFGS-B')
        if not np.isfinite(fit.fun):raise ValueError('Invalid train marginal likelihood')
        self.amplitude,self.noise=np.exp(fit.x);self.optimizer_success=bool(fit.success)
        self.factor=cho_factor(self.amplitude*kernel+self.noise*np.eye(len(yy)),lower=True)
        self.coefficient=cho_solve(self.factor,yy)
        return self
    def predict(self,x):
        cross=self.amplitude*similarity(x,self.x)**self.power
        mean=self.center+cross@self.coefficient
        solved=solve_triangular(self.factor[0],cross.T,lower=True)
        variance=np.maximum(self.amplitude-(solved**2).sum(0),0.)
        result={}
        for name,v in [('median',0.),('latent_mean',variance),('observation_mean',variance+self.noise)]:
            result[name]=np.maximum(np.expm1(np.clip(mean+.5*v,-30,30)),0.)
        return result


def run():
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/current.json').read_text())
    kaggle,public,test,matrices,encoder,audit=feature_inputs(cfg);matrix=matrices['catboost_direct']
    provenance={'data':audit,'code_sha256':sha(Path(__file__)),'similarity_code_sha256':sha(ROOT/'src/current/search.py'),
        'powers':[1.,2.],'fitting':'Train marginal likelihood only, amplitude/noise estimated per fold',
        'official_training_columns':['SMILES','SI VACV'],'test_endpoints_used':False,'holdout_used':False}
    chunks=[];audits=[]
    for split in json.loads((DIRECTORY/'splits.json').read_text())['folds']:
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])];fit=kaggle[~kaggle.original_row.isin(split['held_original_rows'])]
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        for power in [1.,2.]:
            started=time.monotonic();model=LogSIProcess(power).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI)
            for kind,value in model.predict(matrix[held.matrix_row.to_numpy(int)]).items():rows[f'gp_power{power}_{kind}']=value
            audits.append({**split,'power':power,'amplitude':float(model.amplitude),'noise':float(model.noise),'optimizer_success':model.optimizer_success})
            print(f"[probability] {split['protocol']} {split['repeat']} {split['fold']} power={power} seconds={time.monotonic()-started:.1f}",flush=True)
        chunks.append(rows)
    oof=pd.concat(chunks,ignore_index=True);ranking=[]
    for power in [1.,2.]:
        for kind in ['median','latent_mean','observation_mean']:
            name=f'gp_power{power}_{kind}';per={p:metrics(r.SI_true,r[name])['mse'] for p,r in oof.groupby('protocol')}
            ranking.append({'candidate':name,'power':power,'kind':kind,'mean_oof_mse':float(np.mean(list(per.values()))),**per})
    ranking.sort(key=lambda r:r['mean_oof_mse'])
    report={'selected':ranking[0]['candidate'],'ranking':ranking,'provenance':provenance,'fit_audits':audits,'official_kaggle_score':None,'kaggle_submitted':False}
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');oof.to_csv(REPORTS/'oof.csv',index=False)
    for power in [1.,2.]:
        model=LogSIProcess(power).fit(matrix[:len(kaggle)],kaggle.SI)
        artifact=ARTIFACTS/f'gp_power{power}.joblib';joblib.dump({'model':model,'provenance':provenance},artifact,compress=3)
        restored=joblib.load(artifact)['model'];query=matrix[len(kaggle)+len(public):];values=restored.predict(query)
        for row in ranking:
            if row['power']!=power:continue
            kind=row['kind'];np.testing.assert_allclose(values[kind],model.predict(query)[kind],rtol=1e-10,atol=1e-10)
            path=REPORTS/('submission_'+row['candidate']+'.csv');pd.DataFrame({'index':test['index'],'SI VACV':values[kind]}).to_csv(path,index=False)
            row.update(artifact=artifact.relative_to(ROOT).as_posix(),submission=path.relative_to(ROOT).as_posix(),submission_sha256=sha(path))
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv',index=False)
    print('[probability done] '+json.dumps(ranking),flush=True)


if __name__=='__main__':
    from src.current.probabilistic import run as canonical_run
    with threadpool_limits(limits=4):canonical_run()
