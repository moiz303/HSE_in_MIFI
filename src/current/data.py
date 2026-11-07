"""Versioned train_v2 SI and exact public IC50/CC50 pairs with immutable holdout."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.standardize import standardize_smiles
from src.splits import murcko_scaffold

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT/'data/current'
FORBIDDEN = ('CC50_uM', 'IC50_uM_VACV')


def read_kaggle_training(path):
    return pd.read_csv(path,usecols=['SMILES','SI VACV']).rename(columns={'SI VACV':'SI'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def public_pairs(endpoints, doi_mapping):
    """Only exact IC50 and CC50, matched by molecule/document/nonempty cell line."""
    frame = endpoints.copy()
    positive = np.isfinite(frame.value_um) & frame.value_um.gt(0)
    exact = frame.qualifier.fillna('').eq('')
    correct_type = ((frame.endpoint.eq('IC50') & frame.measured_type.eq('IC50')) |
                    (frame.endpoint.eq('CC50') & frame.measured_type.eq('CC50')))
    known_line = frame.cell_line.notna() & frame.cell_line.fillna('').str.strip().ne('')
    keep = positive & exact & correct_type & known_line
    rejected = frame.loc[~keep, ['activity_id','source_id','document_id','endpoint','measured_type','qualifier']].copy()
    rejected['reason'] = np.select([~positive[~keep], ~exact[~keep], ~correct_type[~keep]],
        ['nonpositive_or_nonfinite','censored_measurement','endpoint_type_not_IC50_or_CC50'], default='unknown_cell_line')
    rows = []
    keys = ['standard_smiles','document_id','cell_line']
    for key, group in frame.loc[keep].groupby(keys, sort=True):
        if set(group.endpoint) != {'IC50','CC50'}:
            extra = group[['activity_id','source_id','document_id','endpoint','measured_type','qualifier']].copy()
            extra['reason'] = 'no_exact_partner_same_document_and_cell_line'
            rejected = pd.concat([rejected, extra], ignore_index=True)
            continue
        values = {name: float(np.exp(np.median(np.log(group.loc[group.endpoint.eq(name),'value_um']))))
                  for name in ('IC50','CC50')}
        rows.append({'smiles':key[0], 'SMILES':key[0], 'standard_smiles':key[0], **values,
            'SI': values['CC50']/values['IC50'], 'doi':doi_mapping.get(key[1],''),
            'document_id':key[1], 'cell_line':key[2], 'source':'chembl',
            'source_id':';'.join(sorted(group.source_id.unique())),
            'activity_ids':';'.join(map(str,sorted(group.activity_id.unique()))),
            'assay_ids':';'.join(sorted(group.assay_id.dropna().unique())),
            'IC50_type':'IC50', 'CC50_type':'CC50', 'units':'uM',
            'pair_rule':'exact_same_document_same_cell_line',
            'IC50_qualifier':'', 'CC50_qualifier':''})
    return pd.DataFrame(rows), rejected


def prepare(config):
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    train_path = ROOT/'data/raw/kaggle/train_v2.csv'
    if not train_path.exists():
        raise ValueError('Download authoritative train_v2.csv; legacy train.csv is not accepted')
    # Never load forbidden competition endpoints, even for labels/curation.
    train = read_kaggle_training(train_path)
    train['original_row'] = np.arange(len(train))
    train['standard_smiles'] = train.SMILES.map(standardize_smiles)
    target = pd.to_numeric(train.SI,errors='coerce')
    if len(train) != 320 or train.standard_smiles.isna().any() or not np.isfinite(target).all() or target.lt(0).any():
        raise ValueError('Invalid current competition training data')
    train['SI'] = target
    train['source'],train['weight'] = 'kaggle',1.
    # Reuse roles, never old targets. Legacy-rejected structures stay excluded.
    structures = pd.read_csv(ROOT/'data/processed/dataset.csv',usecols=['standard_smiles'])
    old_manifest = json.loads((ROOT/'data/processed/splits.json').read_text())['indices']
    holdout_structures = set(structures.iloc[old_manifest['holdout']].standard_smiles)
    holdout_scaffolds = set(map(murcko_scaffold,holdout_structures))
    development_structures = set(structures.iloc[old_manifest['train']+old_manifest['validation']].standard_smiles)
    unmapped = set(train.standard_smiles)-development_structures-holdout_structures
    development = train[train.standard_smiles.isin(development_structures)].copy().reset_index(drop=True)
    if set(development.standard_smiles.map(murcko_scaffold)) & holdout_scaffolds:
        raise ValueError('Development overlaps reserved holdout scaffold')
    endpoints_path = ROOT/'data/experiments/independent_endpoints.csv'
    collection_path = ROOT/'data/experiments/independent_collection.json'
    collection = json.loads(collection_path.read_text())
    if collection['target_ids'] != ['CHEMBL613493','CHEMBL613168']:
        raise ValueError('Unreviewed public viral target set')
    raw_pairs = pd.read_csv(ROOT/'data/raw/chembl.csv')
    doi_mapping = raw_pairs.dropna(subset=['doi']).groupby('document_id').doi.first().to_dict()
    collected,rejected = public_pairs(pd.read_csv(endpoints_path),doi_mapping)
    overlap = collected.standard_smiles.isin(set(train.standard_smiles))
    reserved = collected.standard_smiles.map(murcko_scaffold).isin(holdout_scaffolds)
    collected['eligible_for_development'] = ~overlap & ~reserved
    collected['development_exclusion'] = np.select([overlap,reserved],['competition_structure','reserved_holdout_scaffold'],default='')
    public = collected[collected.eligible_for_development].copy().reset_index(drop=True)
    if public.empty:
        raise ValueError('No eligible exact public pairs')
    public['weight'] = config['public_weight']/public.groupby('standard_smiles').standard_smiles.transform('size')
    collected.to_csv(DIRECTORY/'public_dataset.csv',index=False)
    rejected.to_csv(DIRECTORY/'public_rejected.csv',index=False)
    public.to_csv(DIRECTORY/'public_development.csv',index=False)
    development.to_csv(DIRECTORY/'kaggle_development.csv',index=False)
    # Two versioned training cohorts, all original competition rows preserved.
    pd.concat([development,public],ignore_index=True).to_csv(DIRECTORY/'development_mixed.csv',index=False)
    saved = json.loads((ROOT/'reports/architectures/chemeleon/splits.json').read_text())
    old_order = saved['development_smiles']
    folds = []
    for split in saved['splits']:
        for fold, indices in split['indices'].items():
            held_smiles = {old_order[i] for i in indices}
            held = development.loc[development.standard_smiles.isin(held_smiles),'original_row'].astype(int).tolist()
            folds.append({'protocol':split['protocol'],'repeat':split['repeat'],'fold':int(fold),'held_original_rows':held})
    for protocol in ('molecule','scaffold'):
        for repeat in range(config['repeats']):
            covered = [i for fold in folds if fold['protocol']==protocol and fold['repeat']==repeat for i in fold['held_original_rows']]
            if sorted(covered) != sorted(development.original_row.astype(int)):
                raise ValueError('Invalid original-row OOF coverage')
    (DIRECTORY/'splits.json').write_text(json.dumps({'folds':folds,'reserved_competition_rows':len(train)-len(development),
        'holdout_used':False},indent=2),encoding='utf-8')
    source_files = [train_path,endpoints_path,collection_path,ROOT/'data/raw/chembl.csv',
        ROOT/'data/processed/splits.json',ROOT/'data/processed/dataset.csv']
    audit = {'version':'train_v2_exact_public_v1','feature_columns':['SMILES'],
        'kaggle_columns_read':['SMILES','SI VACV'],'forbidden_columns_read':[],
        'kaggle_rows':len(train),'kaggle_development_rows':len(development),
        'reserved_competition_rows':len(train)-len(development),'development_zeros':int(development.SI.eq(0).sum()),
        'legacy_unmapped_competition_structures_excluded':len(unmapped),
        'public_collected_pairs':len(collected),'public_collected_structures':collected.standard_smiles.nunique(),
        'public_fit_pairs':len(public),'public_fit_structures':public.standard_smiles.nunique(),
        'public_missing_doi':int(public.doi.fillna('').eq('').sum()),
        'public_rejection_counts':rejected.reason.value_counts().to_dict(),
        'public_target_types':['IC50','CC50'],'public_virus':'VACV whole-organism targets only',
        'public_weight_rule':str(config['public_weight'])+'/number of records of a structure',
        'competition_targets_aggregated_with_public':False,'holdout_used':False,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in source_files},
        'source_content_sha256':{p.relative_to(ROOT).as_posix():content_sha(p) for p in source_files},
        'output_content_sha256':{p.name:content_sha(p) for p in DIRECTORY.glob('*.csv')}}
    (DIRECTORY/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('[current data] '+json.dumps({k:v for k,v in audit.items() if k not in ('source_sha256','output_sha256','public_rejection_counts')}),flush=True)
    return audit


def load():
    audit = json.loads((DIRECTORY/'audit.json').read_text())
    for relative,expected in audit['source_content_sha256'].items():
        if content_sha(ROOT/relative) != expected:
            raise ValueError('Dataset source changed; rerun prepare: '+relative)
    for relative,expected in audit['output_content_sha256'].items():
        if content_sha(DIRECTORY/relative) != expected:
            raise ValueError('Prepared data changed: '+relative)
    return pd.read_csv(DIRECTORY/'kaggle_development.csv'),pd.read_csv(DIRECTORY/'public_development.csv'),audit


if __name__=='__main__':
    prepare(json.loads((ROOT/'configs/current.json').read_text()))
