"""Grouped train-vs-test discrimination from frozen SMILES features only.

Importance-weighted OOF is diagnostic: it assumes covariate shift and cannot
correct an assay/target convention shift or reproduce hidden Kaggle labels.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,sha
from src.current.workflow import feature_inputs
from src.current.scoreboard import EXPERIMENTS
from src.data.standardize import standardize_smiles

DIRECTORY=ROOT/'reports/current_covariate_audit'


def density_ratio(probability,n_training,n_test):
    probability=np.clip(np.asarray(probability,dtype=float),.01,.99)
    return np.clip(probability/(1-probability)*n_training/n_test,.1,10.)


def run():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/current.json').read_text())
    train,public,test,matrices,encoder,audit=feature_inputs(config)
    # The source classifier sees no SI, public measurements, or endpoint fields.
    features=np.vstack([matrices['chemeleon'][:len(train)],matrices['chemeleon'][len(train)+len(public):]])
    labels=np.r_[np.zeros(len(train),int),np.ones(len(test),int)]
    smiles=train.standard_smiles.tolist()+[standardize_smiles(s) for s in test.SMILES]
    groups=np.asarray(smiles);probability=np.zeros(len(labels));folds=[]
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261106)
    for number,(fit,held) in enumerate(splitter.split(features,labels,groups)):
        if set(groups[fit])&set(groups[held]):raise ValueError('Molecule leakage in source classifier')
        scaler=StandardScaler().fit(features[fit]);xfit=np.clip(scaler.transform(features[fit]),-10,10)
        estimator=LogisticRegression(C=.01,max_iter=1000,random_state=20261106).fit(xfit,labels[fit])
        probability[held]=estimator.predict_proba(np.clip(scaler.transform(features[held]),-10,10))[:,1]
        folds.append({'fold':number,'fit_rows':len(fit),'held_rows':len(held),'molecule_overlap':0,
            'auc':float(roc_auc_score(labels[held],probability[held]))})
    weights=density_ratio(probability[:len(train)],len(train),len(test))
    weights=weights/weights.mean()
    table=train[['original_row','standard_smiles']].copy();table['source_test_probability']=probability[:len(train)];table['density_weight']=weights
    table.to_csv(DIRECTORY/'development_density_weights.csv',index=False)
    weight_map=table.set_index('original_row').density_weight
    comparisons=[]
    for experiment in EXPERIMENTS:
        directory=ROOT/'reports'/experiment
        if not (directory/'report.json').exists():continue
        report=json.loads((directory/'report.json').read_text());oof=pd.read_csv(directory/'oof.csv')
        candidates=report['final'] if experiment=='current' else report['ranking']
        w=oof.original_row.map(weight_map).to_numpy(float)
        if not np.isfinite(w).all():raise ValueError('Missing source weights')
        for row in candidates:
            name=row['candidate']
            if name=='warm':continue
            loss=(oof[name]-oof.SI_true).to_numpy(float)**2
            comparisons.append({'experiment':experiment,'candidate':name,'raw_oof_mse':float(loss.mean()),
                'density_weighted_oof_mse':float(np.average(loss,weights=w)),
                'weighted_row14_loss_share':float(np.sum(loss[oof.original_row.eq(14)]*w[oof.original_row.eq(14)])/np.sum(loss*w))})
    comparison=pd.DataFrame(comparisons).sort_values('density_weighted_oof_mse')
    comparison.to_csv(DIRECTORY/'weighted_comparison.csv',index=False)
    prior=len(test)/(len(train)+len(test))
    baseline_loss=float(log_loss(labels,np.full(len(labels),prior)))
    report={'source_classifier':'5-fold molecule-grouped OOF logistic regression C=.01; train-fold CheMeleon scaling',
        'source_classifier_uses_SI':False,'source_classifier_uses_test_endpoints':False,
        'test_columns_read':['index','SMILES'],'source_auc':float(roc_auc_score(labels,probability)),
        'source_log_loss':float(log_loss(labels,probability)),'folds':folds,
        'constant_source_prior_log_loss':baseline_loss,
        'density_weights_recommended_for_training':False,
        'recommendation_reason':'Source AUC near chance and source log-loss worse than constant prior; estimated weights reduce effective sample size and do not explain leaderboard disagreement.',
        'weights':'n_train/n_test * p_test/(1-p_test), clip [.1,10], normalize training mean1',
        'effective_sample_size':float(weights.sum()**2/np.sum(weights**2)),
        'row14_weight':float(table.loc[table.original_row.eq(14),'density_weight'].iloc[0]),
        'weight_min':float(weights.min()),'weight_max':float(weights.max()),
        'shared_development_test_structures':len(set(train.standard_smiles)&set(smiles[len(train):])),
        'assumptions':'Requires stable P(SI|SMILES). Does not fix assay/source target shift; estimated weights may increase variance.',
        'changes_primary_score_or_model_selection':False,'kaggle_submitted':False,'holdout_used':False,
        'code_sha256':sha(Path(__file__)),'data_audit':audit,'encoder':encoder}
    (DIRECTORY/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('data_audit','encoder')},indent=2))
    print(comparison.head(5).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
