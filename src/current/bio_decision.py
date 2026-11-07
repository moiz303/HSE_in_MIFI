"""Read-only evaluation of completed biological transfer against fixed controls."""
import json
import numpy as np
import pandas as pd
from src.current.data import ROOT,sha
from src.current.results import paired_gain


def main():
    directory=ROOT/'reports/current_bio_transfer'
    report=json.loads((directory/'report.json').read_text());oof=pd.read_csv(directory/'oof.csv')
    keys=['protocol','repeat','fold','original_row'];reference='chemeleon_domain_log1p__public_plus_kaggle'
    original=pd.read_csv(ROOT/'reports/current/oof.csv')
    merged=oof.merge(original[keys+[reference]],on=keys,validate='one_to_one')
    if len(merged)!=267*6 or not merged.groupby('original_row').size().eq(6).all():raise ValueError('Invalid OOF coverage')
    reference_values=merged[reference].to_numpy(float);y=merged.SI_true.to_numpy(float)
    zero=y==0;bulk=(y>0)&(y<=1000);tail=y>1000
    reference_mse={name:float(np.mean((reference_values[mask]-y[mask])**2)) for name,mask in
        [('all',np.ones(len(y),bool)),('zero',zero),('bulk',bulk),('tail',tail)]}
    rows=[]
    for candidate in report['ranking']:
        name=candidate['candidate'];prediction=merged[name].to_numpy(float)
        values={label:float(np.mean((prediction[mask]-y[mask])**2)) for label,mask in
            [('all',np.ones(len(y),bool)),('zero',zero),('bulk',bulk),('tail',tail)]}
        passes=values['zero']<=1.2*reference_mse['zero'] and values['bulk']<=1.2*reference_mse['bulk']
        gains=[]
        for _,part in merged[merged.protocol.eq('scaffold')].groupby('repeat'):
            gains.append(float(np.mean((part[reference]-part.SI_true)**2-(part[name]-part.SI_true)**2)))
        source=candidate['regime'];kind=candidate['kind']
        matched=('base'+kind[3:]+'__'+source) if kind.startswith('bio') else name
        submission=ROOT/candidate['submission']
        if sha(submission)!=candidate['submission_sha256']:raise ValueError('Submission changed')
        final=pd.read_csv(submission)['SI VACV'].to_numpy(float)
        rows.append({'candidate':name,'mse':values,'passes_predeclared_zero_bulk_guardrails':bool(passes),
            'beats_reference_all_mse':bool(values['all']<reference_mse['all']),
            'gain_vs_reference':paired_gain(merged,reference,name),
            'matched_no_biology_control':matched,'gain_vs_matched_control':paired_gain(merged,matched,name),
            'scaffold_repeat_gains':gains,'scaffold_repeats_won':int(np.sum(np.array(gains)>0)),
            'final_prediction_median':float(np.median(final)),'final_prediction_max':float(final.max()),
            'test_to_oof_squared_prediction_ratio':float(np.mean(final**2)/max(np.mean(prediction**2),1e-12))})
    passing=[row for row in rows if row['passes_predeclared_zero_bulk_guardrails'] and row['beats_reference_all_mse']]
    chosen=min(passing,key=lambda row:row['mse']['all'])['candidate'] if passing else 'retain_reference'
    result={'reference':reference,'reference_mse':reference_mse,'rule_from_training_plan':report['provenance']['selection'],
        'selected_under_training_gate':chosen,'candidates':rows,'test_endpoints_used':False,
        'independent_confirmation':False,'bootstrap':'Exploratory structure clusters; repeated OOF reused across hypotheses',
        'official_kaggle_score':None,'kaggle_upload_automatic':False}
    (directory/'decision_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'selected':chosen,'candidates':[{k:row[k] for k in ['candidate','mse','passes_predeclared_zero_bulk_guardrails','beats_reference_all_mse']} for row in rows]},indent=2))


if __name__=='__main__':main()
