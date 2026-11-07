"""Download a new immutable public snapshot; never reads competition endpoints."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
import requests_cache
from src.current.data import ROOT,public_pairs,sha
from src.data import download
from src.data.independent_endpoints import endpoint_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/public_downloads'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args=parser.parse_args()
    directory=args.output_dir.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Choose an empty output directory; snapshots are immutable')
    directory.mkdir(parents=True,exist_ok=True)
    session=requests_cache.CachedSession(str(ROOT/'data/raw/.http_cache'),expire_after=-1,stale_if_error=True)
    targets=[t for t in download.fetch_targets(session) if t['target_type']=='ORGANISM']
    print('[public] whole-virus targets: '+str([t['target_chembl_id'] for t in targets]),flush=True)
    antiviral=download.fetch_antiviral([t['target_chembl_id'] for t in targets],session)
    documents=sorted({a['document_chembl_id'] for a in antiviral if a.get('document_chembl_id')})
    print(f'[public] antiviral records={len(antiviral)}; documents={len(documents)}',flush=True)
    toxicity=download.fetch_cytotoxicity(documents,session)
    assays=sorted({a['assay_chembl_id'] for a in antiviral+toxicity if a.get('assay_chembl_id')})
    cells=download.fetch_assay_cell_lines(assays,session)
    dois=download.fetch_dois(documents,session)
    endpoints,measurement_rejected=endpoint_rows(antiviral,toxicity,cells)
    pairs,pair_rejected=public_pairs(endpoints,dois)
    endpoints.to_csv(directory/'independent_endpoints.csv',index=False)
    pairs.to_csv(directory/'public_dataset.csv',index=False)
    measurement_rejected.to_csv(directory/'measurement_rejected.csv',index=False)
    pair_rejected.to_csv(directory/'pair_rejected.csv',index=False)
    pd.DataFrame([{'document_id':d,'doi':v} for d,v in sorted(dois.items())]).to_csv(directory/'document_dois.csv',index=False)
    metadata={'provider':'ChEMBL','source_url':download.BASE_URL,'whole_organism_targets':targets,
        'rows':len(endpoints),'exact_same_document_cell_line_pairs':len(pairs),
        'units':'uM','public_target_types':['IC50','CC50'],
        'doi_and_source_activity_ids_retained':True,'competition_columns_read':[],
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'sha256':{p.name:sha(p) for p in directory.glob('*.csv')}}
    (directory/'collection.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('[public done] '+str(directory),flush=True)


if __name__=='__main__':
    main()
