"""
Generates a small synthetic dataset.csv that MATCHES the data contract
described in TASKA.md (columns: molecule_id, smiles, IC50_uM, CC50_uM, SI),
purely so Participant B can develop and test the whole pipeline
(splits -> features -> models -> evaluation -> OOF) *before* A delivers
the real, standardized dataset.

NOT part of the deliverable pipeline — delete/ignore once
data/processed/dataset.csv from A is available. Swapping it out requires
no code changes elsewhere, since every downstream module only relies on
the column contract, not on this generator.
"""
import numpy as np
import pandas as pd
from rdkit import Chem

rng = np.random.default_rng(42)

# Base scaffolds (valid, aromatic-complete SMILES). Appending a simple
# substituent directly after the final ring-closure atom is a safe SMILES
# operation (it just fills the aromatic carbon's remaining valence), so we
# avoid any fragile mid-string splicing.
SCAFFOLD_POOL = [
    "c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
    "C1CCCCC1", "C1CCNCC1", "c1ccc2[nH]ccc2c1", "c1ccc2ncccc2c1",
    "c1cnc2ccccc2c1", "C1CC2CCC1CC2", "O=C1CCCCC1", "O=C1CCCCN1",
    "c1ccc2c(c1)CCCC2", "c1ccc2c(c1)OCCO2",
]

SUBSTITUENTS = ["", "C", "CC", "CCC", "O", "N", "F", "Cl", "Br",
                "C(=O)O", "C#N", "OC", "C(F)(F)F", "S(=O)(=O)N"]

rows = []
mol_idx = 0
for scaffold in SCAFFOLD_POOL:
    for sub in SUBSTITUENTS:
        smi = scaffold + sub
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        n_heavy = mol.GetNumHeavyAtoms()

        # Synthetic but structurally-plausible potency numbers (log-normal),
        # loosely correlated with molecular size so the baseline has signal
        # to learn and the pipeline is exercised meaningfully end-to-end.
        log_ic50 = rng.normal(loc=1.0 - 0.015 * n_heavy, scale=0.45)   # log10 uM
        log_cc50 = rng.normal(loc=1.8 - 0.01 * n_heavy, scale=0.45)    # log10 uM
        ic50 = 10 ** log_ic50
        cc50 = 10 ** log_cc50
        si = cc50 / ic50

        rows.append({
            "molecule_id": f"MOL{mol_idx:04d}",
            "smiles": canonical,
            "IC50_uM": round(ic50, 4),
            "CC50_uM": round(cc50, 4),
            "SI": round(si, 4),
            "censored_ic50": "",  # placeholder for A's censoring-flag contract (">", "<", "")
            "censored_cc50": "",
        })
        mol_idx += 1

df = pd.DataFrame(rows).drop_duplicates(subset="smiles").reset_index(drop=True)
df["molecule_id"] = [f"MOL{i:04d}" for i in range(len(df))]
df.to_csv("/home/claude/vacv_baseline/data/processed/dataset.csv", index=False)
print(f"Wrote {len(df)} synthetic molecules to data/processed/dataset.csv")
