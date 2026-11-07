"""Render immutable OOF results and optional, separately produced endpoint diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.current.data import ROOT,sha


def paired_gain(rows,baseline,candidate,seed=20261106):
    """Exploratory bootstrap of whole structure clusters, not independent repeats."""
    grouped=pd.DataFrame({'structure':rows.standard_smiles,
        'gain':(rows[baseline]-rows.SI_true)**2-(rows[candidate]-rows.SI_true)**2})
    clusters=grouped.groupby('structure').agg(gain=('gain','sum'),count=('gain','size'))
    gains,counts=clusters.gain.to_numpy(),clusters['count'].to_numpy()
    generator=np.random.default_rng(seed)
    samples=generator.integers(0,len(clusters),size=(1000,len(clusters)))
    values=gains[samples].sum(1)/counts[samples].sum(1)
    return {'mean_mse_gain':float(gains.sum()/counts.sum()),
        'ci95_lower':float(np.quantile(values,.025)),'ci95_upper':float(np.quantile(values,.975)),
        'structure_clusters':len(clusters),'independent_holdout_confirmation':False}


def main():
    directory=ROOT/'reports/current'
    report=json.loads((directory/'report.json').read_text())
    diagnostic_path=ROOT/'reports/current_test_diagnostics.json'
    diagnostics=json.loads(diagnostic_path.read_text())['results'] if diagnostic_path.exists() else []
    lookup={row['submission_sha256']:row for row in diagnostics}
    registry_path=directory/'kaggle_goal_submissions.json'
    registry=json.loads(registry_path.read_text()).get('submissions',[]) if registry_path.exists() else []
    official={row['submission_sha256']:row for row in registry if row.get('official_score') is not None and row.get('ref')}
    rows=[]
    for row in report['final']:
        if sha(ROOT/row['submission'])!=row['submission_sha256']:
            raise ValueError('Submission bytes changed')
        name=row['candidate']
        values={p:report['protocols'][p][name]['mse'] for p in ('molecule','scaffold')}
        diagnostic=lookup.get(row['submission_sha256'])
        verified=official.get(row['submission_sha256'])
        rows.append({**row,'official_kaggle_score':verified['official_score'] if verified else row.get('official_kaggle_score'),
            'molecule_oof_mse':values['molecule'],'scaffold_oof_mse':values['scaffold'],
            'mean_oof_mse':float(np.mean(list(values.values()))),
            'molecule_zero_oof_mse':report['protocols']['molecule'][name]['zero']['mse'],
            'scaffold_zero_oof_mse':report['protocols']['scaffold'][name]['zero']['mse'],
            'local_endpoint_subset_mse':diagnostic['mse_ratio_subset'] if diagnostic else None,
            'endpoint_coverage':f"{diagnostic['n_evaluated']}/{diagnostic['n_test']}" if diagnostic else None,
            'oof_selected':name==report['selection']['selected']})
    table=pd.DataFrame(rows)
    table.to_csv(directory/'comparison.csv',index=False)
    oof=pd.read_csv(directory/'oof.csv')
    source_effect=[]
    for method in report['provenance']['config']['methods']:
        a,b=method+'__kaggle_only',method+'__public_plus_kaggle'
        left,right=table.set_index('candidate').loc[a],table.set_index('candidate').loc[b]
        source_effect.append({'method':method,'kaggle_only_mean_oof_mse':float(left.mean_oof_mse),
            'mixed_mean_oof_mse':float(right.mean_oof_mse),
            'public_oof_gain_percent':float(100*(left.mean_oof_mse-right.mean_oof_mse)/left.mean_oof_mse),
            **paired_gain(oof,a,b)})
    pd.DataFrame(source_effect).to_csv(directory/'public_effect.csv',index=False)
    (directory/'oof_uncertainty.json').write_text(json.dumps({'purpose':'Exploratory, selected on reused OOF; not independent confirmation',
        'cluster_unit':'standardized molecular structure, retaining all original rows and repeats together',
        'selected_vs_zero':paired_gain(oof,'constant_zero',report['selection']['selected']),
        'public_effect':source_effect},indent=2),encoding='utf-8')
    lines=['# Актуальное сравнение SMILES → SI (train_v2)', '',
        'Все 267 исходных development-строк, включая 66 нулей. Одинаковые 3 fold × 3 повтора, molecule и scaffold.',
        '53 исходные строки вне development не обучались и не оценивались. Публичные данные: точные IC50/CC50 пары, отдельные от Kaggle.',
        'Из Kaggle в обучении/инференсе используются только SMILES и SI VACV; концентрационные колонки запрещены.', '',
        '| Метод | Данные | OOF molecule MSE | OOF scaffold MSE | Локальный endpoint MSE, 468/597 | Kaggle Score |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        local=f"{row['local_endpoint_subset_mse']:.2f}" if row['local_endpoint_subset_mse'] is not None else '—'
        public=f"{row['official_kaggle_score']:.5f}" if row['official_kaggle_score'] is not None else 'не отправлен'
        lines.append(f"| {row['method']} | {row['regime']} | {row['molecule_oof_mse']:.2f} | {row['scaffold_oof_mse']:.2f} | {local} | {public} |")
    lines+=['',f"Выбор по OOF до тестовых диагностик: **{report['selection']['selected']}**.",
        'Прямое сравнение источников: те же параметры и те же отложенные исходные конкурсные строки.',
        'Локальный endpoint MSE использует другой неполный эталон; это не официальный Score, он не участвует в выборе.',
        'Последующие разрешённые отправки и текущая цель top10 или Score<800000: kaggle_goal_submissions.json. Контроли хвоста: ../current_tail/.', '',
        '| Ориентир | OOF molecule MSE | OOF scaffold MSE |', '|---|---:|---:|']
    for name in ['constant_zero','constant_mean']:
        lines.append(f"| {name} | {report['protocols']['molecule'][name]['mse']:.2f} | {report['protocols']['scaffold'][name]['mse']:.2f} |")
    lines+=['','Нулевые и положительные цели, MAE, log1p RMSE и ошибки каждого повтора доступны в report.json.',
        'Старый CatBoost из testing_main: тот же Kaggle-only SI600/depth5, 267 строк, локальный MSE4187780.78852 с ограничением SI>=0.',
        'Исторический public1614079.37878 относится к прежним концентрационным доменным головам и не переносится на новые SI-головы.',
        'Старые модели вне текущего строгого протокола использования конкурсных концентраций; архивные scores не являются оценками новых моделей.','']
    (directory/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print(table[['candidate','molecule_oof_mse','scaffold_oof_mse','local_endpoint_subset_mse','oof_selected']].to_string(index=False))


if __name__=='__main__':
    main()
