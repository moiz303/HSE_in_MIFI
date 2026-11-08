"""Fill and execute the skill-scaffolded comparison notebook without a kernel dependency."""
import contextlib
import io
import json
from pathlib import Path
from src.current.data import ROOT


def markdown(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
def code(text):return {'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'execution_count':None,'outputs':[]}


def main():
    path=ROOT/'output/jupyter-notebook/vacv-model-comparison.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    notebook['cells']=[
        markdown('# VACV SI: comparison of reproducible experiments\n\nRead-only report: compare saved OOF and verified Kaggle Scores. No model fitting, test endpoint access or uploads.'),
        code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nroot = Path.cwd().resolve()\nwhile not (root / 'src/current/data.py').exists():\n    if root.parent == root: raise RuntimeError('Run inside the repository')\n    root = root.parent\nsys.path.insert(0, str(root))\nfrom src.current.scoreboard import build\ntable = build(write=False)\nprint(f'{len(table)} evaluated configurations; identical 267 development rows and 18 folds')\n"),
        markdown('## Primary raw-SI OOF MSE\n\nIncludes zeros and every original development row. This is the competition formula on training OOF, not the hidden Kaggle test score. Broad search reuses OOF; there is no independent holdout confirmation.'),
        code("columns = ['experiment','candidate','mean_oof_mse','zero_mse','bulk_mse','row14_loss_share']\nprint(table[columns].head(10).round(2).to_string(index=False))\n"),
        markdown('## Sensitivity to the single extreme observation\n\nRow14 has SI=36066.667 and accounts for about 96% of some candidates\' OOF loss. The following diagnostic does not delete it, change labels or redefine the official metric.'),
        code("print(table.sort_values('without_row14_mse')[columns[:2]+['without_row14_mse','mean_oof_mse']].head(10).round(2).to_string(index=False))\n"),
        markdown('## Paired public + Kaggle / Kaggle-only comparison\n\nSame methods, parameters, competition validation rows and split roles. Public held-out molecule/scaffold overlap was excluded.'),
        code("base = table[table.experiment.eq('current')].copy()\nbase['method'] = base.candidate.str.split('__').str[0]\nprint(base.pivot(index='method',columns='regime',values='mean_oof_mse').round(2).to_string())\n"),
        markdown('## Verified official Kaggle Scores\n\nValues are attached to exact submission bytes and API references. Unknown scores stay missing. Local endpoint-ratio diagnostics are intentionally absent from this report and from model selection.'),
        code("print(table[table.official_kaggle_score.notna()][['experiment','candidate','mean_oof_mse','official_kaggle_score','submission_ref']].to_string(index=False))\nregistry = json.loads((root/'reports/current/kaggle_goal_submissions.json').read_text())\nprint('Current target:', registry['target'])\nprint('Submissions tracked across quota windows:', len(registry['submissions']))\n"),
        markdown('## Interpretation\n\nA better raw OOF score can come from a tiny number of extreme structures while worsening zeros and ordinary SI. The failed mixed GP illustrates why this does not establish hidden-test generalization. Source shift and tail exposure are separate risks. Bootstrap intervals and detailed split/source checks are in `reports/current/protocol_audit.json`.\n\nOfficial competition concentration columns never enter current fitting or inference. Public auxiliary IC50, EC50 and CC50 are separate tasks with independent provenance; censored observations impose directional bounds. Historic concentration-head scores are outside the current protocol.')]
    namespace={'__name__':'__notebook__'};count=0
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        count+=1;stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(compile(''.join(cell['source']),str(path),'exec'),namespace)
        cell['execution_count']=count;cell['outputs']=[{'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(keepends=True)}]
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(f'Executed {count} cells top-to-bottom: {path.relative_to(ROOT)}')


if __name__=='__main__':main()
