"""Published VACV supplement: preserve endpoints; never substitute EC50 for IC50."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.standardize import standardize_smiles
from src.splits import murcko_scaffold


def load_supplement(root):
    root = Path(root)
    raw = pd.read_csv(root/'data/raw/vacv_2023_reporter_observations.csv')
    if len(raw) != 138 or raw.record_id.duplicated().any():
        raise ValueError('Unexpected supplement cardinality or duplicate observations')
    if not raw.units.eq('uM').all() or not raw.endpoint.isin(['EC50', 'CC50']).all():
        raise ValueError('Unsupported endpoint or units')
    rows = raw.copy()
    rows['standard_smiles'] = rows.smiles.map(standardize_smiles)
    if rows.standard_smiles.isna().any():
        raise ValueError('Invalid supplement structure')
    rows['source'] = 'literature_vacv_2023'
    rows['doi'] = rows.publication_doi
    rows['SI'] = np.nan
    rows['IC50'] = np.nan
    rows['IC50_qualifier'] = ''
    rows['CC50'] = np.where(rows.endpoint.eq('CC50'), rows.value, np.nan)
    rows['CC50_qualifier'] = np.where(rows.endpoint.eq('CC50'), rows.relation.replace('=', ''), '')
    rows['EC50'] = np.where(rows.endpoint.eq('EC50'), rows.value, np.nan)
    rows['EC50_qualifier'] = np.where(rows.endpoint.eq('EC50'), rows.relation.replace('=', ''), '')
    rows['target_is_label'] = False
    dataset = pd.read_csv(root/'data/processed/dataset.csv', usecols=['standard_smiles'])
    manifest = json.loads((root/'data/processed/splits.json').read_text())
    indices = manifest.get('indices', manifest.get('split', manifest))
    if 'splits' in manifest:
        indices = manifest['splits']
    held = dataset.iloc[indices['holdout']].standard_smiles.tolist()
    forbidden = {murcko_scaffold(s) for s in held}
    return rows[~rows.standard_smiles.map(murcko_scaffold).isin(forbidden)].reset_index(drop=True)


def eligible_supplement(supplement, held_smiles, protocol='scaffold'):
    held = set(held_smiles)
    if protocol == 'scaffold':
        blocked = {murcko_scaffold(s) for s in held}
        mask = ~supplement.standard_smiles.map(murcko_scaffold).isin(blocked)
    elif protocol == 'molecule':
        mask = ~supplement.standard_smiles.isin(held)
    else:
        raise ValueError('Unknown protocol')
    return supplement.loc[mask].copy().reset_index(drop=True)
