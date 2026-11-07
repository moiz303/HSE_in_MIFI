#!/usr/bin/env python3
"""Derive CC50/EC50 selectivity ratios from the curated 2023 VACV tables."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/vacv_2023_reporter_observations.csv"
OUTPUT = ROOT / "data/raw/vacv_2023_reporter_si_pairs.csv"
FIELDS = (
    "compound_name", "pubchem_cid", "smiles", "virus_construct",
    "ec50_readout", "ec50_uM", "ec50_relation", "cc50_method",
    "cc50_uM", "cc50_relation", "si_cc50_over_ec50", "si_relation",
    "si_status", "publication_doi", "table_number",
)


def derive_pairs(observations):
    compounds = {}
    for row in observations:
        compounds.setdefault((row["compound_name"], row["table_number"]), []).append(row)

    pairs = []
    for (name, _table), rows in compounds.items():
        ec50s = [row for row in rows if row["endpoint"] == "EC50"]
        cc50s = [row for row in rows if row["endpoint"] == "CC50"]
        for ec50 in ec50s:
            for cc50 in cc50s:
                e_rel, c_rel = ec50["relation"], cc50["relation"]
                e_value, c_value = float(ec50["value"]), float(cc50["value"])
                if e_rel == "=" and c_rel == "=":
                    si, si_rel, status = f"{c_value / e_value:.8g}", "=", "identified"
                elif e_rel == "=" and c_rel == ">":
                    si, si_rel, status = f"{c_value / e_value:.8g}", ">", "identified"
                elif e_rel == ">" and c_rel == "=":
                    si, si_rel, status = f"{c_value / e_value:.8g}", "<", "identified"
                else:
                    si, si_rel, status = "", "", "not_identifiable_from_bounds"
                pairs.append({
                    "compound_name": name,
                    "pubchem_cid": ec50["pubchem_cid"],
                    "smiles": ec50["smiles"],
                    "virus_construct": ec50["virus_construct"],
                    "ec50_readout": ec50["readout"],
                    "ec50_uM": ec50["value"],
                    "ec50_relation": e_rel,
                    "cc50_method": cc50["assay_method"],
                    "cc50_uM": cc50["value"],
                    "cc50_relation": c_rel,
                    "si_cc50_over_ec50": si,
                    "si_relation": si_rel,
                    "si_status": status,
                    "publication_doi": ec50["publication_doi"],
                    "table_number": ec50["table_number"],
                })
    return pairs


def main():
    with SOURCE.open(newline="", encoding="utf-8") as f:
        pairs = derive_pairs(csv.DictReader(f))
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(pairs)
    identified = sum(row["si_status"] == "identified" for row in pairs)
    print(f"Wrote {len(pairs)} contexts ({identified} identified SI ratios) to {OUTPUT}")


if __name__ == "__main__":
    main()
