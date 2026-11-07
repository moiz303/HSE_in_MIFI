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

REPORTS=ROOT/'reports/current_representation_gp'
ARTIFACTS=ROOT/'artifacts/current_representation_gp'


class RepresentationProcess:
    def __init__(self,kind):self.kind=kind
    def projected(self,x):
        z=np.clip((np.asarray(x[:,2048:],float)-self.mean)/self.scale,-10,10)
        return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    def kernel(self,x,z):
        a,b=self.projected(x),self.projected(z)
        cosine=np.clip(a@b.T,-1,1)
        if self.kind=='cosine':return cosine
        rbf=np.exp(-np.maximum(2.-2.*cosine,0.)/self.length)
        return rbf if self.kind=='rbf' else .5*rbf+.5*similarity(x,z)**2
    def fit(self,x,y,weights,source):
        self.x=np.asarray(x,float);source=np.asarray(source,bool)
        self.mean=self.x[:,2048:].mean(0);self.scale=np.maximum(self.x[:,2048:].std(0),1e-6)
        projected=self.projected(self.x)
        distances=np.maximum(2.-2.*projected@projected.T,0.)
        self.length=max(float(np.median(distances[np.triu_indices(len(x),1)])),.01)
        self.sqrt_weight=np.sqrt(np.asarray(weights,float))
        target=np.log1p(np.asarray(y,float));self.center=float(target[source].mean())
        yy=(target-self.center)*self.sqrt_weight
        kernel=self.kernel(self.x,self.x)*self.sqrt_weight[:,None]*self.sqrt_weight[None,:]
        variance=max(float(np.var(target[source])),.1)
        fitted=minimize(objective,np.log([variance/2,variance/2]),args=(kernel,yy),jac=True,
            bounds=[(np.log(.01),np.log(100.)),(np.log(.05),np.log(100.))],method='L-BFGS-B')
        if not np.isfinite(fitted.fun):raise ValueError('Invalid likelihood')
        self.amplitude,self.noise=np.exp(fitted.x);self.optimizer_success=bool(fitted.success)
        self.factor=cho_factor(self.amplitude*kernel+self.noise*np.eye(len(yy)),lower=True)
        self.coefficient=cho_solve(self.factor,yy)
        return self
    def predict(self,x):
        cross=self.amplitude*self.kernel(x,self.x)*self.sqrt_weight[None,:]
        mean=self.center+cross@self.coefficient
        solved=solve_triangular(self.factor[0],cross.T,lower=True)
        variance=np.maximum(self.amplitude-(solved**2).sum(0),0.)
        return {name:np.maximum(np.expm1(np.clip(mean+.5*v,-30,30)),0.) for name,v in
            [('latent_mean',variance),('observation_mean',variance+self.noise)]}


KINDS=['cosine','rbf','rbf_tanimoto']
REGIMES=['kaggle_only','public_plus_kaggle']


def run():
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/current.json').read_text())
    kaggle,public,test,matrices,encoder,audit=feature_inputs(cfg)
    matrix=np.column_stack([matrices['catboost_direct'][:,:2048],matrices['chemeleon']])
    chunks=[];audits=[]
    folds=json.loads((DIRECTORY/'splits.json').read_text())['folds']
    for ordinal,split in enumerate(folds):
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])]
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        for regime in REGIMES:
            fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],regime)
            for kind in KINDS:
                model=RepresentationProcess(kind).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI,fit.weight,fit.source.eq('kaggle'))
                for outcome,value in model.predict(matrix[held.matrix_row.to_numpy(int)]).items():rows[f'{kind}_{regime}_{outcome}']=value
                audits.append({**split,'regime':regime,'kind':kind,'fit_rows':len(fit),'amplitude':float(model.amplitude),'noise':float(model.noise),'optimizer_success':model.optimizer_success})
        chunks.append(rows);print(f"[representation GP] fold {ordinal+1}/{len(folds)}",flush=True)
    oof=pd.concat(chunks,ignore_index=True);ranking=[]
    assert len(oof)==267*3*2
    for regime in REGIMES:
        for kind in KINDS:
            for outcome in ['latent_mean','observation_mean']:
                name=f'{kind}_{regime}_{outcome}';per={p:metrics(r.SI_true,r[name])['mse'] for p,r in oof.groupby('protocol')}
                ranking.append({'candidate':name,'kernel':kind,'outcome':outcome,'regime':regime,'mean_oof_mse':float(np.mean(list(per.values()))),**per})
    ranking.sort(key=lambda r:r['mean_oof_mse'])
    provenance={'data':audit,'code_sha256':sha(Path(__file__)),'objective_code_sha256':sha(ROOT/'src/current/probabilistic.py'),
        'encoder':encoder,'scaling':'Fit-fold standardized/clipped embeddings, unit normalization, RBF median fit pair distance',
        'public_weights':'0.5/records per structure -> observation noise divided by weight',
        'hyperparameters':'Train marginal likelihood only','holdout_used':False,'test_endpoints_used':False}
    report={'selected':ranking[0]['candidate'],'ranking':ranking,'provenance':provenance,'fit_audits':audits,'official_score':None}
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');oof.to_csv(REPORTS/'oof.csv',index=False)
    for regime in REGIMES:
        fit=pd.concat([kaggle,public],ignore_index=True) if regime=='public_plus_kaggle' else kaggle
        query=matrix[len(kaggle)+len(public):]
        for kind in KINDS:
            model=RepresentationProcess(kind).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI,fit.weight,fit.source.eq('kaggle'))
            artifact=ARTIFACTS/f'{kind}_{regime}.joblib';joblib.dump({'model':model,'provenance':provenance},artifact,compress=3)
            restored=joblib.load(artifact)['model'];values=restored.predict(query)
            for row in ranking:
                if (row['kernel'],row['regime'])!=(kind,regime):continue
                np.testing.assert_allclose(values[row['outcome']],model.predict(query)[row['outcome']],rtol=1e-10,atol=1e-10)
                path=REPORTS/('submission_'+row['candidate']+'.csv');pd.DataFrame({'index':test['index'],'SI VACV':values[row['outcome']]}).to_csv(path,index=False)
                row.update(artifact=artifact.relative_to(ROOT).as_posix(),submission=path.relative_to(ROOT).as_posix(),submission_sha256=sha(path))
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');pd.DataFrame(ranking).to_csv(REPORTS/'comparison.csv',index=False)
    print('[representation GP done] '+json.dumps(ranking[:3]),flush=True)


def verify():
    from src.features.fingerprints import featurize
    from src.architecture_features import representations
    from src.data.standardize import standardize_smiles
    test=pd.read_csv(ROOT/'data/raw/kaggle/test.csv',usecols=['index','SMILES'])
    encoded,_=representations([standardize_smiles(s) for s in test.SMILES],'chemeleon',8)
    query=np.column_stack([featurize(test.SMILES.tolist())[:,:2048],encoded])
    report=json.loads((REPORTS/'report.json').read_text());checks=[]
    for row in report['ranking']:
        assert sha(ROOT/row['submission'])==row['submission_sha256']
        model=joblib.load(ROOT/row['artifact'])['model'];assert model.__class__.__module__=='src.current.representation_gp'
        value=model.predict(query)[row['outcome']];stored=pd.read_csv(ROOT/row['submission'])
        np.testing.assert_array_equal(test['index'],stored['index'])
        np.testing.assert_allclose(value,stored['SI VACV'],rtol=1e-4,atol=1e-4)
        checks.append({'candidate':row['candidate'],'passed':True,'maximum_absolute_difference':float(np.max(np.abs(value-stored['SI VACV'])))})
    (REPORTS/'inference_verification.json').write_text(json.dumps({'status':'passed','test_columns':['index','SMILES'],'checks':checks},indent=2),encoding='utf-8')
    print('Fresh process verification passed',len(checks),flush=True)


if __name__=='__main__':
    import sys
    from src.current.representation_gp import run as canonical_run,verify as canonical_verify
    with threadpool_limits(limits=2):
        if '--verify' in sys.argv:canonical_verify()
        else:canonical_run()
