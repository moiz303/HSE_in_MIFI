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

REPORTS=ROOT/'reports/current_probability_public'
ARTIFACTS=ROOT/'artifacts/current_probability_public'


class SourceSIProcess:
    def __init__(self,power,rho):self.power,self.rho=power,rho
    def fit(self,x,y,weights,source):
        self.x=np.asarray(x,dtype=float);self.source=np.asarray(source,dtype=bool)
        self.sqrt_weight=np.sqrt(np.asarray(weights,dtype=float))
        if not self.source.any() or (self.sqrt_weight<=0).any():raise ValueError('Competition rows and positive weights required')
        target=np.log1p(np.asarray(y,dtype=float));self.center=float(target[self.source].mean())
        yy=(target-self.center)*self.sqrt_weight
        kernel=similarity(x,x)**self.power*np.where(self.source[:,None]==self.source[None,:],1.,self.rho)
        kernel=kernel*self.sqrt_weight[:,None]*self.sqrt_weight[None,:]
        variance=max(float(np.var(target[self.source])),.1)
        fitted=minimize(objective,np.log([variance/2,variance/2]),args=(kernel,yy),jac=True,
            bounds=[(np.log(.01),np.log(100.)),(np.log(.05),np.log(100.))],method='L-BFGS-B')
        if not np.isfinite(fitted.fun):raise ValueError('Invalid marginal likelihood')
        self.amplitude,self.noise=np.exp(fitted.x);self.optimizer_success=bool(fitted.success)
        self.factor=cho_factor(self.amplitude*kernel+self.noise*np.eye(len(yy)),lower=True)
        self.coefficient=cho_solve(self.factor,yy)
        return self
    def predict(self,x):
        # All inference queries use the competition source; no test metadata is read.
        cross=self.amplitude*similarity(x,self.x)**self.power*np.where(self.source,1.,self.rho)[None,:]*self.sqrt_weight[None,:]
        mean=self.center+cross@self.coefficient
        solved=solve_triangular(self.factor[0],cross.T,lower=True)
        variance=np.maximum(self.amplitude-(solved**2).sum(0),0.)
        return {name:np.maximum(np.expm1(np.clip(mean+.5*v,-30,30)),0.) for name,v in
            [('median',0.),('latent_mean',variance),('observation_mean',variance+self.noise)]}


def run():
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/current.json').read_text());kaggle,public,test,matrices,_,audit=feature_inputs(cfg)
    matrix=matrices['catboost_direct'];chunks=[];audits=[]
    for split in json.loads((DIRECTORY/'splits.json').read_text())['folds']:
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])]
        fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],'public_plus_kaggle')
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        for power in [1.,2.]:
            for rho in [.5,1.]:
                model=SourceSIProcess(power,rho).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI,fit.weight,fit.source.eq('kaggle'))
                for kind,value in model.predict(matrix[held.matrix_row.to_numpy(int)]).items():rows[f'gp_power{power}_rho{rho}_{kind}']=value
                audits.append({**split,'power':power,'rho':rho,'public_rows':int(fit.source.ne('kaggle').sum()),'amplitude':float(model.amplitude),'noise':float(model.noise),'optimizer_success':model.optimizer_success})
        chunks.append(rows);print(f"[public GP] {split['protocol']} {split['repeat']} {split['fold']}",flush=True)
    oof=pd.concat(chunks,ignore_index=True);ranking=[]
    for power in [1.,2.]:
        for rho in [.5,1.]:
            for kind in ['median','latent_mean','observation_mean']:
                name=f'gp_power{power}_rho{rho}_{kind}';per={p:metrics(r.SI_true,r[name])['mse'] for p,r in oof.groupby('protocol')}
                ranking.append({'candidate':name,'power':power,'rho':rho,'kind':kind,'regime':'public_plus_kaggle','mean_oof_mse':float(np.mean(list(per.values()))),**per})
    ranking.sort(key=lambda r:r['mean_oof_mse'])
    provenance={'data':audit,'code_sha256':sha(Path(__file__)),'objective_code_sha256':sha(ROOT/'src/current/probabilistic.py'),
        'public_weights':'0.5/records per structure -> observation noise divided by weight',
        'source_kernel':'Tanimoto^power times source covariance, diagonal 1 off-diagonal rho',
        'hyperparameters':'Train marginal likelihood only','holdout_used':False,'test_endpoints_used':False}
    report={'selected':ranking[0]['candidate'],'ranking':ranking,'provenance':provenance,'fit_audits':audits,'official_score':None}
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');oof.to_csv(REPORTS/'oof.csv',index=False)
    fit=pd.concat([kaggle,public],ignore_index=True);query=matrix[len(fit):]
    for power in [1.,2.]:
        for rho in [.5,1.]:
            model=SourceSIProcess(power,rho).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI,fit.weight,fit.source.eq('kaggle'))
            artifact=ARTIFACTS/f'gp_power{power}_rho{rho}.joblib';joblib.dump({'model':model,'provenance':provenance},artifact,compress=3)
            restored=joblib.load(artifact)['model'];values=restored.predict(query)
            for row in ranking:
                if (row['power'],row['rho'])!=(power,rho):continue
                np.testing.assert_allclose(values[row['kind']],model.predict(query)[row['kind']],rtol=1e-10,atol=1e-10)
                path=REPORTS/('submission_'+row['candidate']+'.csv');pd.DataFrame({'index':test['index'],'SI VACV':values[row['kind']]}).to_csv(path,index=False)
                row.update(artifact=artifact.relative_to(ROOT).as_posix(),submission=path.relative_to(ROOT).as_posix(),submission_sha256=sha(path))
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv',index=False)
    print('[public GP done] '+json.dumps(ranking[:3]),flush=True)


if __name__=='__main__':
    from src.current.probability_public import run as canonical_run
    with threadpool_limits(limits=4):canonical_run()
