"""Collect independent VACV IC/EC50 and toxicity endpoints without manufacturing pairs."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.data import download
from src.data.standardize import standardize_smiles

ROOT = Path(__file__).resolve().parents[2]


def endpoint_rows(antiviral, toxicity, cell_lines):
    rows, rejected = [], []
    context = {}
    for activity in antiviral:
        key = (activity.get("molecule_chembl_id"), activity.get("document_chembl_id"))
        record, reason = download._measurement(activity, cell_lines)
        if record is not None:
            context.setdefault(key, set()).add(record["cell_line"])
    for endpoint, activities in (("IC50", antiviral), ("CC50", toxicity)):
        seen = set()
        for activity in activities:
            activity_id = activity.get("activity_id")
            if activity_id in seen:
                continue
            seen.add(activity_id)
            record, reason = download._measurement(activity, cell_lines)
            if record is not None and record["validity"] in download.INVALID_VALIDITY_COMMENTS:
                record, reason = None, "ChEMBL validity flag"
            structure = standardize_smiles(record["smiles"]) if record else None
            if record is not None and structure is None:
                reason = "invalid standardized structure"
            if not record or not structure:
                rejected.append({"activity_id": activity_id, "endpoint": endpoint, "reason": reason})
                continue
            molecule, document = activity.get("molecule_chembl_id"), activity.get("document_chembl_id")
            if not molecule or not document:
                rejected.append({"activity_id": activity_id, "endpoint": endpoint, "reason": "missing molecule/document"})
                continue
            same_line = bool(record["cell_line"]) and record["cell_line"] in context.get((molecule, document), set())
            confidence = 1.0 if endpoint == "IC50" else (1.0 if same_line and record["mentions_vaccinia"] else
                .75 if same_line else .5 if record["mentions_vaccinia"] else .25)
            rows.append({"standard_smiles": structure, "endpoint": endpoint, "value_um": record["value_um"],
                "qualifier": record["qualifier"], "weight": .5*confidence, "activity_id": activity_id,
                "source_id": molecule, "document_id": document, "assay_id": activity.get("assay_chembl_id"),
                "target_id": activity.get("target_chembl_id"),
                "measured_type": record["measured_type"], "cell_line": record["cell_line"],
                "assay_description": record["assay_description"], "source": "chembl"})
    return pd.DataFrame(rows), pd.DataFrame(rejected)


def collect():
    import requests_cache
    # Reuse the existing HTTP cache; keep the earlier paired export untouched.
    session = requests_cache.CachedSession(str(ROOT / "data/raw/.http_cache"), expire_after=-1,
                                           stale_if_error=True)
    all_targets = download.fetch_targets(session)
    targets = [t for t in all_targets if t["target_type"] == "ORGANISM"]
    print(f"[independent] whole-virus VACV targets={len(targets)}; excluded protein targets={len(all_targets)-len(targets)}", flush=True)
    antiviral = download.fetch_antiviral([t["target_chembl_id"] for t in targets], session)
    print(f"[independent] antiviral activities={len(antiviral)}", flush=True)
    documents = sorted({a["document_chembl_id"] for a in antiviral if a.get("document_chembl_id")})
    toxicity = download.fetch_cytotoxicity(documents, session)
    print(f"[independent] toxicity activities={len(toxicity)}; documents={len(documents)}", flush=True)
    assays = sorted({a["assay_chembl_id"] for a in antiviral+toxicity if a.get("assay_chembl_id")})
    cells = download.fetch_assay_cell_lines(assays, session)
    rows, rejected = endpoint_rows(antiviral, toxicity, cells)
    directory = ROOT / "data/experiments"
    directory.mkdir(parents=True, exist_ok=True)
    rows.to_csv(directory / "independent_endpoints.csv", index=False)
    rejected.to_csv(directory / "independent_rejected.csv", index=False)
    audit = {"target_ids": [t["target_chembl_id"] for t in targets], "excluded_protein_targets": len(all_targets)-len(targets),
             "raw_antiviral": len(antiviral), "raw_toxicity": len(toxicity), "documents": len(documents),
             "accepted": len(rows), "rejected": len(rejected), "endpoint_counts": rows.endpoint.value_counts().to_dict(),
             "exact_counts": rows.loc[rows.qualifier.eq(""), "endpoint"].value_counts().to_dict(),
             "note": "CC50/TC50/CD50/GI50 and IC50/EC50 types retained explicitly; no SI or missing partners inferred"}
    (directory / "independent_collection.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    collect()
