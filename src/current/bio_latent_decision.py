"""Read-only latent-feature comparison with output-only biological controls."""
import json
import numpy as np
import pandas as pd
from src.current.data import ROOT,sha
from src.current.results import paired_gain


def main():
    directory=ROOT/'reports/current_bio_latent';report=json.loads((directory/'report.json').read_text())
    oof=pd.read_csv(directory/'oof.csv');baseline=pd.read_csv(ROOT/'reports/current_bio_transfer/oof.csv')
    keys=['protocol','repeat','fold','original_row'];reference='bio_fp__kaggle_only'
    controls=[k+'__'+s for k in ['bio_small','bio_fp'] for s in ['kaggle_only','public_plus_kaggle']]
    rows=oof.merge(baseline[keys+controls+['SI_true']],on=keys,validate='one_to_one',suffixes=('','_control'))
    np.testing.assert_allclose(rows.SI_true,rows.SI_true_control,rtol=0,atol=1e-9)
    if len(rows)!=267*6:raise ValueError('Incomplete OOF')
    y=rows.SI_true.to_numpy();zero=y==0;bulk=(y>0)&(y<=1000)
    def errors(name):
        prediction=rows[name].to_numpy()
        return {'all':float(np.mean((prediction-y)**2)),'zero':float(np.mean((prediction[zero]-y[zero])**2)),
            'bulk':float(np.mean((prediction[bulk]-y[bulk])**2))}
    reference_mse=errors(reference);comparisons=[]
    for candidate in report['ranking']:
        name=candidate['candidate'];mse=errors(name)
        matched=(candidate['kind'].replace('latent','bio')+'__'+candidate['regime']) if candidate['kind']!='latent_chem' else None
        gains=[float(np.mean((r[reference]-r.SI_true)**2-(r[name]-r.SI_true)**2)) for _,r in rows[rows.protocol.eq('scaffold')].groupby('repeat')]
        eligible=mse['all']<reference_mse['all'] and mse['zero']<=1.2*reference_mse['zero'] and mse['bulk']<=1.2*reference_mse['bulk']
        if sha(ROOT/candidate['submission'])!=candidate['submission_sha256']:raise ValueError('Submission changed')
        final=pd.read_csv(ROOT/candidate['submission'])['SI VACV'].to_numpy()
        comparisons.append({'candidate':name,'mse':mse,'passes_predeclared_gate':bool(eligible),
            'gain_vs_reference':paired_gain(rows,reference,name),'matched_output_only_control':matched,
            'gain_vs_matched_control':paired_gain(rows,matched,name) if matched else None,
            'attribution_note':'Adds only hidden features to the same base' if matched else 'Chemical input uses PCA64; not a one-factor comparison with fingerprint reference',
            'scaffold_repeat_gains':gains,'scaffold_repeats_won':int(np.sum(np.array(gains)>0)),
            'final_prediction_median':float(np.median(final)),'final_prediction_max':float(final.max()),
            'test_to_oof_squared_prediction_ratio':float(np.mean(final**2)/max(np.mean(rows[name].to_numpy()**2),1e-12))})
    passing=[r for r in comparisons if r['passes_predeclared_gate']]
    result={'reference':reference,'reference_mse':reference_mse,'selected_under_predeclared_gate':min(passing,key=lambda r:r['mse']['all'])['candidate'] if passing else 'retain_output_only_bio',
        'candidates':comparisons,'test_endpoints_used':False,'independent_confirmation':False,
        'uncertainty':'Exploratory molecule-cluster bootstrap on reused OOF','queued_candidate_changed':False,'kaggle_upload_automatic':False}
    (directory/'decision_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'selected':result['selected_under_predeclared_gate'],'candidates':[{k:r[k] for k in ['candidate','mse','passes_predeclared_gate']} for r in comparisons]},indent=2))


if __name__=='__main__':main()
