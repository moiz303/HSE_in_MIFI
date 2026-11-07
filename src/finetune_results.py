"""Render completed, OOF-selected ablations plus optional separate endpoint diagnostics."""
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from src import xgb_workflow as workflow
from src.local_compare import fold_indices

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT/'reports/finetune'
    report = json.loads((directory/'report.json').read_text())
    development = workflow.load_parts()
    frame = pd.concat([development['train'], development['validation']], ignore_index=True)
    manifests = [{'protocol': p, 'repeat': r, 'indices': fold_indices(frame.standard_smiles.tolist(),
        p, report['config']['folds'], report['config']['seed']+r)}
        for p in ('molecule', 'scaffold') for r in range(report['config']['repeats'])]
    (directory/'splits.json').write_text(json.dumps({'development_smiles': frame.standard_smiles.tolist(),
        'splits': manifests, 'data_sha256': workflow.data_hashes({'model': {'strategy': 'xgb_weighted'}})}, indent=2), encoding='utf-8')
    diagnostic_path = ROOT/'reports/finetune_test_diagnostics.json'
    diagnostics = json.loads(diagnostic_path.read_text())['results'] if diagnostic_path.exists() else []
    lookup = {Path(r['submission']).name: r for r in diagnostics}
    official = json.loads((ROOT/'reports/architectures/kaggle_submissions.json').read_text())['submissions']
    rows = []
    for name in report['config']['candidates']:
        final = report['final'][name]
        if hashlib.sha256((ROOT/final['submission']).read_bytes()).hexdigest() != final['sha256']:
            raise ValueError('Submission differs from completed fit report')
        diag = lookup.get(Path(final['submission']).name)
        if diag and diag['submission_sha256'] != final['sha256']:
            raise ValueError('Diagnostic references different submission bytes')
        rows.append({'method': name,
            'molecule_measured_oof_mse': report['protocols']['molecule'][name]['measured']['mse'],
            'scaffold_measured_oof_mse': report['protocols']['scaffold'][name]['measured']['mse'],
            'oof_ratio': report['selection']['measured_oof_mean_ratios'][name],
            'local_subset_mse': diag['mse_ratio_subset'] if diag else None,
            'local_rows': diag['n_evaluated'] if diag else None,
            'official_kaggle_mse': next((r['public_score'] for r in official
                if r['submission_sha256'] == final['sha256'] and r.get('public_score') is not None), None),
            'oof_selected': report['selection']['selected'] == name,
            'encoder_update_norm': final['encoder_update_norm'], 'submission': final['submission']})
    table = pd.DataFrame(rows)
    table.to_csv(directory/'comparison.csv', index=False)
    lines = ['# CheMeleon: частичное дообучение и качество данных', '',
        'Одинаковые molecule/scaffold разбиения, 3 fold × 3 повторения; holdout исключён.',
        'OOF: исходные измеренные SI из TRAIN. Локальная диагностика: доступные точные пары теста, отдельно после обучения.',
        'Локальный MSE не является официальным Kaggle Score. Новые файлы не отправлялись.', '',
        '| Метод | OOF molecule MSE | OOF scaffold MSE | Локальный MSE, 468/597 | Kaggle Score |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        local = f"{row['local_subset_mse']:.5f}" if row['local_subset_mse'] is not None else '—'
        official = f"{row['official_kaggle_mse']:.5f}" if row['official_kaggle_mse'] is not None else 'не отправлен'
        lines.append(f"| {row['method']} | {row['molecule_measured_oof_mse']:.5f} | {row['scaffold_measured_oof_mse']:.5f} | {local} | {official} |")
    lines += ['', f"Выбор по заранее заданному OOF-правилу: **{report['selection']['selected']}**.",
        'Официально подтверждённая модель CheMeleon остаётся сохранена отдельно.', '',
        '`continue`: ещё 40 эпох голов; `lora`: те же эпохи + rank-8 адаптер W_o.',
        '`original_*`: доступные точные оригинальные IC50/CC50 имеют приоритет над объединёнными публичными измерениями.',
        'LoRA изменяет финальную атомную проекцию; слои передачи сообщений W_i/W_h заморожены.',
        'Выбор исследовательский: многократные сравнения на тех же OOF не заменяют независимую проверку.']
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    oof = pd.read_csv(directory/'oof.csv')
    measured = oof.target_is_label.astype('string').str.lower().eq('false').fillna(False)
    base = pd.read_csv(directory/'development_error_audit.csv')
    columns = []
    for candidate in report['config']['candidates']:
        for endpoint in ('IC50', 'CC50'):
            columns.append(candidate+'_log_'+endpoint)
    head_logs = oof.loc[measured].groupby('original_row')[columns].mean().reset_index()
    heads = base.merge(head_logs, on='original_row', validate='one_to_one')
    for candidate in report['config']['candidates']:
        for endpoint in ('IC50', 'CC50'):
            name = candidate+'_log_'+endpoint
            truth = pd.to_numeric(heads[endpoint], errors='coerce')
            heads[candidate+'_'+endpoint+'_geomean'] = np.exp(heads[name])
            heads[candidate+'_'+endpoint+'_log10_error'] = (heads[name]-np.log(truth.where(truth.gt(0))))/np.log(10.)
    heads.to_csv(directory/'head_error_audit.csv', index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
