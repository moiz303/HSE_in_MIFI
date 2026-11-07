"""Second SI-only hypothesis block: kernels, local analogues and nonnegative boosting.

All choices use existing development folds; no endpoint diagnostics are imported.
This is exploratory selection on reused OOF, not independent confirmation.
"""
import json
import hashlib
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.current.data import ROOT, DIRECTORY, sha
from src.current.workflow import feature_inputs, select_fit_rows, metrics

REPORTS=ROOT/'reports/current_search'
ARTIFACTS=ROOT/'artifacts/current_search'


def specifications():
    options=[]
    for target in ('raw','log1p'):
        for alpha in (.1,1.,10.):
            options.append(dict(kind='kernel',target=target,alpha=alpha,power=2.,regime='kaggle_only'))
    for target in ('raw','log1p'):
        for power in (4.,12.):
            options.append(dict(kind='neighbors',target=target,power=power,regime='kaggle_only'))
    for target in ('raw','log1p','capped'):
        for leaf in (1,4):
            options.append(dict(kind='forest',target=target,leaf=leaf,regime='kaggle_only'))
    for loss in ('Tweedie:variance_power=1.3','Tweedie:variance_power=1.7',
                 'Poisson','Huber:delta=100','Quantile:alpha=0.8'):
        options.append(dict(kind='boost',loss=loss,target='log1p' if loss.startswith('Quantile') else 'raw',regime='kaggle_only'))
    for alpha in (1.,10.):
        options.append(dict(kind='kernel',target='log1p',alpha=alpha,power=2.,regime='public_plus_kaggle'))
    for alpha in (1.,10.):
        options.append(dict(kind='hurdle',target='log1p',alpha=alpha,power=2.,regime='kaggle_only'))
    for spec in options:
        spec['name']='__'.join(str(spec[k]).replace(':','_').replace('=','_') for k in spec)
    return options


def similarity(x,z):
    # Only Morgan bits; descriptors have no meaning in a Tanimoto denominator.
    x=np.asarray(x[:,:2048],dtype=float);z=np.asarray(z[:,:2048],dtype=float)
    common=x@z.T
    return common/np.maximum(x.sum(1)[:,None]+z.sum(1)[None,:]-common,1.)


class Candidate:
    def __init__(self,spec):self.spec=spec
    def fit(self,x,y,w):
        spec=self.spec
        self.x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);w=np.asarray(w,dtype=float)
        self.target=spec['target']
        target=np.log1p(y) if self.target=='log1p' else np.minimum(y,5000.) if self.target=='capped' else y
        kind=spec['kind']
        if kind in ('kernel','hurdle'):
            self.prior=float(np.average(target,weights=w)) if self.target=='log1p' else 0.
            kernel=similarity(self.x,self.x)**spec['power']
            self.coefficient=solve(kernel+np.diag(spec['alpha']/w),target-self.prior,assume_a='pos')
            if kind=='hurdle':
                self.scaler=StandardScaler().fit(self.x)
                self.classifier=LogisticRegression(C=.01,max_iter=1000,random_state=20261106).fit(self.scaler.transform(self.x),y>0,sample_weight=w)
                positive=y>0
                self.x=self.x[positive]
                self.prior=float(np.average(target[positive],weights=w[positive]))
                kernel=similarity(self.x,self.x)**spec['power']
                self.coefficient=solve(kernel+np.diag(spec['alpha']/w[positive]),target[positive]-self.prior,assume_a='pos')
        elif kind=='neighbors':
            self.target_values=target;self.weights=w
        elif kind=='forest':
            self.estimator=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=spec['leaf'],max_features=.6,random_state=20261106,n_jobs=4)
            self.estimator.fit(self.x,target,sample_weight=w)
        elif kind=='boost':
            from catboost import CatBoostRegressor
            # Scale only the positive raw-loss targets to keep exponentials stable.
            self.unit=100. if spec['loss'].startswith(('Tweedie','Poisson')) else 1.
            self.estimator=CatBoostRegressor(iterations=800,depth=4,learning_rate=.03,l2_leaf_reg=10,
                loss_function=spec['loss'],random_seed=20261106,thread_count=4,verbose=False,allow_writing_files=False)
            self.estimator.fit(self.x,target/self.unit,sample_weight=w)
        return self
    def predict(self,x):
        kind=self.spec['kind']
        if kind in ('kernel','hurdle'):
            output=self.prior+similarity(x,self.x)**self.spec['power']@self.coefficient
        elif kind=='neighbors':
            weights=similarity(x,self.x)**self.spec['power']*self.weights[None,:]
            output=weights@self.target_values/np.maximum(weights.sum(1),1e-12)
        else:
            output=self.estimator.predict(x)
            if kind=='boost':output=output*self.unit
        output=np.expm1(np.clip(output,-30,30)) if self.target=='log1p' else output
        if kind=='hurdle':output=output*self.classifier.predict_proba(self.scaler.transform(x))[:,1]
        output=np.maximum(output,0.)
        if not np.isfinite(output).all():raise ValueError('Nonfinite predictions')
        return output


def run():
    REPORTS.mkdir(exist_ok=True,parents=True);ARTIFACTS.mkdir(exist_ok=True,parents=True)
    config=json.loads((ROOT/'configs/current.json').read_text())
    kaggle,public,test,matrices,encoder,audit=feature_inputs(config)
    matrix=matrices['catboost_direct']
    specs=specifications()
    provenance={'specifications':specs,'data':audit,'code_sha256':sha(Path(__file__)),
        'feature_code_sha256':sha(ROOT/'src/features/fingerprints.py'),
        'selection':'Mean ALL-row raw-SI MSE over fixed molecule/scaffold development OOF; no endpoint inputs',
        'independent_holdout_used':False,'official_concentrations_used':False}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    chunks=[]
    for split in json.loads((DIRECTORY/'splits.json').read_text())['folds']:
        stamp=f"{split['protocol']}_{split['repeat']}_{split['fold']}"
        path=REPORTS/f'fold_{stamp}.csv';marker=path.with_suffix('.json')
        if path.exists() and marker.exists() and json.loads(marker.read_text())['fingerprint']==fingerprint:
            chunks.append(pd.read_csv(path));continue
        held=kaggle[kaggle.original_row.isin(split['held_original_rows'])]
        rows=held[['original_row','standard_smiles','SI']].rename(columns={'SI':'SI_true'}).copy()
        rows['protocol']=split['protocol'];rows['repeat']=split['repeat'];rows['fold']=split['fold']
        for spec in specs:
            started=time.monotonic()
            fit=select_fit_rows(kaggle,public,split['held_original_rows'],split['protocol'],spec['regime'])
            model=Candidate(spec).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI,fit.weight)
            rows[spec['name']]=model.predict(matrix[held.matrix_row.to_numpy(int)])
            print(f"[search] {stamp} {spec['name']} MSE={metrics(held.SI,rows[spec['name']])['mse']:.3f} seconds={time.monotonic()-started:.1f}",flush=True)
        rows.to_csv(path,index=False);marker.write_text(json.dumps({'fingerprint':fingerprint}),encoding='utf-8');chunks.append(rows)
    oof=pd.concat(chunks,ignore_index=True)
    results=[]
    for spec in specs:
        per={p:metrics(r.SI_true,r[spec['name']])['mse'] for p,r in oof.groupby('protocol')}
        results.append({'candidate':spec['name'],'mean_oof_mse':float(np.mean(list(per.values()))),**per,'spec':spec})
    results.sort(key=lambda r:r['mean_oof_mse'])
    selection={'selected':results[0]['candidate'],'ranking':results,'provenance':provenance,'test_endpoints_used':False,'official_kaggle_score':None,'submissions_sent':0}
    # Freeze the choice before producing ANY final predictions or endpoint diagnostics.
    (REPORTS/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    oof.to_csv(REPORTS/'oof.csv',index=False)
    for row in results[:5]:
        spec=row['spec'];fit=pd.concat([kaggle,public],ignore_index=True) if spec['regime']=='public_plus_kaggle' else kaggle
        model=Candidate(spec).fit(matrix[fit.matrix_row.to_numpy(int)],fit.SI,fit.weight)
        artifact=ARTIFACTS/(spec['name']+'.joblib');joblib.dump({'model':model,'provenance':provenance},artifact,compress=3)
        restored=joblib.load(artifact)['model'];query=matrix[len(kaggle)+len(public):]
        pred=restored.predict(query);np.testing.assert_allclose(pred,model.predict(query),rtol=1e-12,atol=1e-12)
        submission=REPORTS/('submission_'+spec['name']+'.csv')
        pd.DataFrame({'index':test['index'],'SI VACV':pred}).to_csv(submission,index=False)
        row.update(artifact=artifact.relative_to(ROOT).as_posix(),submission=submission.relative_to(ROOT).as_posix(),submission_sha256=sha(submission))
    (REPORTS/'report.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in row.items() if k!='spec'} for row in results]).to_csv(REPORTS/'comparison.csv',index=False)
    print('[search done] '+json.dumps(results[:5]),flush=True)


if __name__=='__main__':
    from src.current.search import run as canonical_run
    with threadpool_limits(limits=4):canonical_run()
