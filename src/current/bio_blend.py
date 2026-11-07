"""Three fixed SI-scale blends; never changes the already queued submission."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.current.data import ROOT,sha
from src.current.results import paired_gain

DIRECTORY=ROOT/'reports/current_bio_blend'
WEIGHTS=[.25,.5,.75]


def run():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    reference='chemeleon_domain_log1p__public_plus_kaggle';bio='bio_fp__kaggle_only'
    paths=['reports/current/oof.csv','reports/current_bio_transfer/oof.csv',
        'reports/current/submission_'+reference+'.csv','reports/current_bio_transfer/submission_'+bio+'.csv']
    a,b=[pd.read_csv(ROOT/p) for p in paths[:2]];keys=['protocol','repeat','fold','original_row']
    rows=a[keys+['standard_smiles','SI_true',reference]].merge(b[keys+['SI_true',bio]],on=keys,validate='one_to_one',suffixes=('','_bio'))
    np.testing.assert_allclose(rows.SI_true,rows.SI_true_bio,rtol=0,atol=1e-9)
    ranking=[]
    for weight in WEIGHTS:
        name=f'bio_weight_{weight}';rows[name]=(1-weight)*rows[reference]+weight*rows[bio]
        per={p:float(np.mean((r.SI_true-r[name])**2)) for p,r in rows.groupby('protocol')}
        y=rows.SI_true;zero=y.eq(0);bulk=y.gt(0)&y.le(1000)
        ranking.append({'candidate':name,'bio_weight':weight,'regime':'fixed_blend',
            'mean_oof_mse':float(np.mean(list(per.values()))),**per,
            'zero_mse':float(np.mean((rows.loc[zero,name]-y[zero])**2)),
            'bulk_mse':float(np.mean((rows.loc[bulk,name]-y[bulk])**2)),
            'gain_vs_chemeleon':paired_gain(rows,reference,name),'gain_vs_bio':paired_gain(rows,bio,name)})
    ranking.sort(key=lambda row:row['mean_oof_mse'])
    report={'selected':ranking[0]['candidate'],'ranking':ranking,'input_sha256':{p:sha(ROOT/p) for p in paths},
        'weights_declared_before_comparison':WEIGHTS,'primary_metric':'ALL original development raw-SI mean molecule/scaffold MSE',
        'independent_confirmation':False,'test_endpoints_used':False,'kaggle_submitted':False,
        'already_queued_candidate_changed':False,'no_new_model_fitting':True,'code_sha256':sha(Path(__file__))}
    (DIRECTORY/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rows.to_csv(DIRECTORY/'oof.csv',index=False)
    first,second=[pd.read_csv(ROOT/p) for p in paths[2:]]
    np.testing.assert_array_equal(first['index'],second['index'])
    for row in ranking:
        weight=row['bio_weight'];values=(1-weight)*first['SI VACV']+weight*second['SI VACV']
        path=DIRECTORY/('submission_'+row['candidate']+'.csv')
        pd.DataFrame({'index':first['index'],'SI VACV':values}).to_csv(path,index=False)
        restored=pd.read_csv(path);np.testing.assert_allclose(restored['SI VACV'],values,rtol=1e-12,atol=1e-12)
        row.update(submission=path.relative_to(ROOT).as_posix(),submission_sha256=sha(path))
    (DIRECTORY/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,dict)} for row in ranking]).to_csv(DIRECTORY/'comparison.csv',index=False)
    print(json.dumps([{k:row[k] for k in ['candidate','mean_oof_mse','zero_mse','bulk_mse']} for row in ranking],indent=2))


if __name__=='__main__':run()
