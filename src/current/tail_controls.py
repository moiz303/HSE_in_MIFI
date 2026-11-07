"""Train-fold target-quantile caps and fixed equal-weight ensemble controls.

No test targets, test endpoints or leaderboard values participate in selection.
This block addresses extrapolation from one extremely large development target.
"""
import json
import argparse
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.current.data import ROOT,DIRECTORY,load,sha
from src.current.results import paired_gain

REPORTS=ROOT/'reports/current_tail'
BASES={
    'domain':('reports/current/oof.csv','chemeleon_domain_log1p__public_plus_kaggle','reports/current/submission_chemeleon_domain_log1p__public_plus_kaggle.csv'),
    'pooled':('reports/current/oof.csv','chemeleon_mlp_log1p__kaggle_only','reports/current/submission_chemeleon_mlp_log1p__kaggle_only.csv'),
    'catboost':('reports/current/oof.csv','catboost_log1p__public_plus_kaggle','reports/current/submission_catboost_log1p__public_plus_kaggle.csv'),
    'quantile':('reports/current_search/oof.csv','boost__Quantile_alpha_0.8__log1p__kaggle_only','reports/current_search/submission_boost__Quantile_alpha_0.8__log1p__kaggle_only.csv'),
    'tabpfn':('reports/current_tabular/oof.csv','tabpfn_raw','reports/current_tabular/submission_tabpfn_raw.csv')}
QUANTILES=[.95,.975,.99]


def capped(prediction,fit_targets,q):
    y=np.asarray(fit_targets,dtype=float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or (y<0).any():raise ValueError('Finite nonnegative fit targets required')
    return np.minimum(np.maximum(prediction,0.),float(np.quantile(y,q)))


def run():
    REPORTS.mkdir(parents=True,exist_ok=True)
    kaggle,_,audit=load()
    keys=['protocol','repeat','fold','original_row']
    original=pd.read_csv(ROOT/'reports/current/oof.csv')
    rows=original[keys+['standard_smiles','SI_true']].copy()
    sources={};test={};input_checksums={}
    for short,(oof,column,submission) in BASES.items():
        table=pd.read_csv(ROOT/oof);merged=rows[keys].merge(table[keys+[column]],on=keys,validate='one_to_one')
        if len(merged)!=len(rows):raise ValueError('Inconsistent base OOF rows')
        sources[short]=merged[column].to_numpy(float)
        prediction=pd.read_csv(ROOT/submission)
        if test:np.testing.assert_array_equal(prediction['index'],next(iter(test.values()))['index'])
        test[short]=prediction
        input_checksums[oof]=sha(ROOT/oof);input_checksums[submission]=sha(ROOT/submission)
    fold_indices={(s['protocol'],s['repeat'],s['fold']):s['held_original_rows'] for s in json.loads((DIRECTORY/'splits.json').read_text())['folds']}
    specs=[]
    for base in BASES:
        for quantile in QUANTILES:
            name=f'{base}_cap_q{quantile}'
            values=np.zeros(len(rows))
            for stamp,part in rows.groupby(['protocol','repeat','fold']):
                targets=kaggle.loc[~kaggle.original_row.isin(fold_indices[stamp]),'SI']
                values[part.index]=capped(sources[base][part.index],targets,quantile)
            rows[name]=values;specs.append({'candidate':name,'kind':'train_quantile_cap','base':base,'quantile':quantile})
    for bases in [('domain','catboost'),('domain','tabpfn'),('domain','catboost','tabpfn')]:
        name='equal_'+('_'.join(bases));rows[name]=np.mean([sources[b] for b in bases],axis=0)
        specs.append({'candidate':name,'kind':'fixed_equal_blend','bases':list(bases)})
    rows['constant_zero']=0.;rows['uncapped_domain']=sources['domain']
    ranking=[]
    for spec in specs:
        name=spec['candidate'];per={p:float(((r[name]-r.SI_true)**2).mean()) for p,r in rows.groupby('protocol')}
        ranking.append({**spec,'mean_oof_mse':float(np.mean(list(per.values()))),**per,
            'gain_vs_uncapped_domain':paired_gain(rows,'uncapped_domain',name)})
    ranking.sort(key=lambda r:r['mean_oof_mse'])
    report={'selected':ranking[0]['candidate'],'ranking':ranking,'input_sha256':input_checksums,
        'data_audit':audit,'code_sha256':sha(Path(__file__)),'test_endpoints_used':False,
        'selection':'ALL-row mean molecule/scaffold MSE; exploratory reused OOF, no independent confirmation',
        'official_kaggle_score':None,'kaggle_submitted':False}
    # Freeze all choices before generating final submissions or endpoint diagnostics.
    (REPORTS/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rows.to_csv(REPORTS/'oof.csv',index=False)
    for row in ranking:
        if row['kind']=='train_quantile_cap':
            prediction=capped(test[row['base']]['SI VACV'],kaggle.SI,row['quantile'])
            row['final_cap']=float(np.quantile(kaggle.SI,row['quantile']))
        else:prediction=np.mean([test[b]['SI VACV'].to_numpy() for b in row['bases']],axis=0)
        path=REPORTS/('submission_'+row['candidate']+'.csv')
        pd.DataFrame({'index':next(iter(test.values()))['index'],'SI VACV':prediction}).to_csv(path,index=False)
        row.update(submission=path.relative_to(ROOT).as_posix(),submission_sha256=sha(path))
    (REPORTS/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in ranking]).to_csv(REPORTS/'comparison.csv',index=False)
    print(json.dumps(ranking[:5],indent=2))


def predict(artifact,test_path,output,quantile=.99):
    from src.current.workflow import predict as base_predict
    kaggle,_,_=load()
    with tempfile.TemporaryDirectory(prefix='vacv_si_prediction_') as directory:
        temporary=Path(directory)/'uncapped.csv'
        base_predict(artifact,test_path,str(temporary))
        frame=pd.read_csv(temporary)
    frame['SI VACV']=capped(frame['SI VACV'],kaggle.SI,quantile)
    path=ROOT/output;path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict-artifact')
    parser.add_argument('--test',default='data/raw/kaggle/test.csv')
    parser.add_argument('--out',default='reports/current_tail/prediction.csv')
    parser.add_argument('--quantile',type=float,choices=QUANTILES,default=.99)
    args=parser.parse_args()
    if args.predict_artifact:predict(args.predict_artifact,args.test,args.out,args.quantile)
    else:run()
