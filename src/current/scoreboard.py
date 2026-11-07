"""One scoreboard of immutable development OOF and hash-matched official results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.current.data import ROOT

EXPERIMENTS=['current','current_search','current_tabular','current_tail','current_auxiliary',
    'current_probability','current_probability_public','current_representation_gp','current_hurdle_gp','current_power_ensemble','current_bio_transfer','current_bio_blend','current_bio_mean','current_bio_latent']


def subset_mse(y,p,mask):
    return float(np.mean((y[mask]-p[mask])**2)) if mask.any() else None


def build(write=True):
    registry=json.loads((ROOT/'reports/current/kaggle_goal_submissions.json').read_text())
    official={r['submission_sha256']:r for r in registry['submissions'] if r.get('official_score') is not None and r.get('ref')}
    original=pd.read_csv(ROOT/'data/current/kaggle_development.csv',usecols=['original_row','SI']).set_index('original_row').SI
    output=[]
    for experiment in EXPERIMENTS:
        directory=ROOT/'reports'/experiment
        if not (directory/'report.json').exists():continue
        report=json.loads((directory/'report.json').read_text());oof=pd.read_csv(directory/'oof.csv')
        if len(oof)!=267*6 or not oof.groupby('original_row').size().eq(6).all():raise ValueError('Invalid OOF row coverage: '+experiment)
        np.testing.assert_allclose(oof.SI_true,oof.original_row.map(original),rtol=0,atol=1e-9)
        candidates=report['final'] if experiment=='current' else report['ranking']
        for candidate in candidates:
            name=candidate['candidate']
            if name=='warm':continue
            if name not in oof:raise ValueError('Candidate has no OOF: '+name)
            y=oof.SI_true.to_numpy(float);p=oof[name].to_numpy(float)
            if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid predictions: '+name)
            per={protocol:float(np.mean((rows.SI_true-rows[name])**2)) for protocol,rows in oof.groupby('protocol')}
            loss=(y-p)**2;extreme=oof.original_row.eq(14).to_numpy()
            known=official.get(candidate.get('submission_sha256'))
            spec=candidate.get('spec',{})
            output.append({'experiment':experiment,'candidate':name,
                'regime':candidate.get('regime',spec.get('regime','kaggle_only' if experiment in ['current_tabular','current_probability'] else 'derived')),
                'mean_oof_mse':float(np.mean(list(per.values()))),'molecule_mse':per['molecule'],'scaffold_mse':per['scaffold'],
                'zero_mse':subset_mse(y,p,y==0),'bulk_mse':subset_mse(y,p,(y>0)&(y<=1000)),
                'tail_mse':subset_mse(y,p,y>1000),'without_row14_mse':subset_mse(y,p,~extreme),
                'row14_loss_share':float(loss[extreme].sum()/loss.sum()),
                'official_kaggle_score':known['official_score'] if known else None,'submission_ref':known['ref'] if known else None,
                'submission':candidate.get('submission'),'submission_sha256':candidate.get('submission_sha256'),'artifact':candidate.get('artifact')})
    table=pd.DataFrame(output).sort_values('mean_oof_mse',ignore_index=True)
    if write:
        table.to_csv(ROOT/'reports/current/all_models.csv',index=False)
        lines=['# Current SI-only scoreboard','',
            'All OOF values use the same 267 original train_v2 development rows, including 66 zeros, 18 molecule/scaffold folds. Lower MSE is better.',
            'One row (SI=36066.667) dominates raw MSE. Subset scores expose this sensitivity; they do not replace official MSE or constitute independent confirmation.',
            'Kaggle scores are joined only by submission SHA256 and verified submission reference. Blank/— means unknown, not zero.','',
            '| Experiment | Candidate | Mean OOF MSE | Zero MSE | Bulk MSE (0<SI<=1000) | Row14 loss share | Kaggle Score |',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in table.to_dict('records'):
            score=f"{row['official_kaggle_score']:.5f}" if pd.notna(row['official_kaggle_score']) else '—'
            lines.append(f"| {row['experiment']} | {row['candidate']} | {row['mean_oof_mse']:.2f} | {row['zero_mse']:.2f} | {row['bulk_mse']:.2f} | {100*row['row14_loss_share']:.2f}% | {score} |")
        lines+=['','Historical concentration-head models are excluded from this strict protocol. No test endpoint labels were read by this report.','']
        (ROOT/'reports/current/all_models.md').write_text('\n'.join(lines),encoding='utf-8')
    return table


if __name__=='__main__':
    table=build();print(table[table.official_kaggle_score.notna()][['experiment','candidate','mean_oof_mse','official_kaggle_score']].to_string(index=False))
