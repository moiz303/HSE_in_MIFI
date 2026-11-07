"""Render completed, OOF-selected ablations plus optional separate endpoint diagnostics."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT/'reports/finetune'
    report = json.loads((directory/'report.json').read_text())
    diagnostic_path = ROOT/'reports/finetune_test_diagnostics.json'
    diagnostics = json.loads(diagnostic_path.read_text())['results'] if diagnostic_path.exists() else []
    lookup = {Path(r['submission']).name: r for r in diagnostics}
    rows = []
    for name in report['config']['candidates']:
        final = report['final'][name]
        diag = lookup.get(Path(final['submission']).name)
        if diag and diag['submission_sha256'] != final['sha256']:
            raise ValueError('Diagnostic references different submission bytes')
        rows.append({'method': name,
            'molecule_measured_oof_mse': report['protocols']['molecule'][name]['measured']['mse'],
            'scaffold_measured_oof_mse': report['protocols']['scaffold'][name]['measured']['mse'],
            'oof_ratio': report['selection']['measured_oof_mean_ratios'][name],
            'local_subset_mse': diag['mse_ratio_subset'] if diag else None,
            'local_rows': diag['n_evaluated'] if diag else None,
            'official_kaggle_mse': 1750683.92577 if name == 'frozen' else None,
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
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
