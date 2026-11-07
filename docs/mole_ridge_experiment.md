# Frozen MolE + Ridge experiment

The graph representation uses the authors' `gin_concat` MolE checkpoint from [Zenodo 10.5281/zenodo.10803099](https://zenodo.org/records/10803099), with the upstream [MolE code](https://github.com/rolayoalarcon/MolE). The extractor follows the upstream atom/bond graph contract and first `GINet.forward` output: concatenated sum pooling of five 200-dimensional GIN layers (1000 features). A smoke comparison against upstream `process_dataset(..., device="cpu")` on ethanol and benzene had maximum absolute difference `2.34e-6`. The retained encoder state omits the 8000-wide projection head and is 3.4 MB. Upstream MIT license is in `third_party/MolE/LICENSE`.

The complete author checkpoint is 803.8 MB. It is not stored in git; `python scripts/download_mole_weights.py` downloads it locally and verifies the Zenodo-published MD5. The author config settings are committed at `artifacts/mole/config.yaml` with whitespace normalized; the download script separately verifies the exact source config hash. Extraction runs in an optional isolated CPU environment from `requirements-mole.txt`; it reads only `train.csv` and `validation.csv` by default. It produced 1007 finite embeddings of width 1000. Feature caches and the full checkpoint are ignored by git.

Each candidate has independent Ridge heads for log10(IC50) and log10(CC50), each fit only on its exact uncensored targets; SI is the exponentiated head difference. Tested inputs were MolE alone, Morgan 2048 + 13 RDKit descriptors, and their concatenation. Standardization is fit within each Ridge training fold. The initial alpha=1 run is preserved under `reports/mole_ridge_alpha1_*` and `artifacts/mole/ridge_alpha1_*.joblib`.

The follow-up selects Ridge alpha from `{100, 1000, 10000}` on three inner scaffold folds drawn only from each outer training partition. Selection uses original competition raw-SI MSE. The chosen alpha is then fit on the outer training partition and scored on the untouched outer fold. Final development refits use the modal alpha from these train-only selections. The fixed validation result is a separate train-only evaluation; the saved development refits include train and validation, so they have no independent validation score. Holdout and test endpoints were not used.

The preserved initial alpha=1 ablation had molecule/scaffold MSE of `5.35e48 / 3.56e31` for MolE, `9.90e9 / 7.89e8` for Morgan, and `99.65M / 33.92M` for fusion. This showed severe extrapolation with weak regularization and motivated the fixed inner-fold alpha grid above.

| Model/features | Molecule OOF raw SI MSE | Scaffold OOF raw SI MSE | Fixed validation raw SI MSE |
|---|---:|---:|---:|
| Weighted XGBoost reference | 29,659,650 | 29,682,392 | — |
| MolE Ridge | 43,298,126 | 10,860,989,982 | 52,036,167 |
| Morgan Ridge | 29,677,364 | 29,680,044 | 51,874,235 |
| MolE + Morgan Ridge | 29,580,752 | 29,697,249 | 51,836,265 |

MolE-only remains unstable under scaffold shift. Fusion is effectively tied to the weighted baseline on the two OOF protocols: a 0.27% molecule-protocol decrease and a 0.05% scaffold-protocol increase. The small molecule difference is not evidence of a robust gain, and the fixed-validation sample contains 67 original rows. The experiment does not change the inference default or establish a Kaggle score.

Reproduction:

```sh
python scripts/download_mole_weights.py
python3 -m venv /Users/f37/HSE_mole_env
/Users/f37/HSE_mole_env/bin/pip install -r requirements-mole.txt
/Users/f37/HSE_mole_env/bin/python -m src.mole_extract --batch-size 32 --device cpu
/Users/f37/HSE_arch_env/bin/python -m src.mole_compare
```

The pretrained checkpoint was trained for antimicrobial prediction. The Zenodo record has no explicit rights value even though the source repository is MIT licensed; preserve attribution and check checkpoint usage terms before redistribution or commercial use.

Final verification: after installing the repository-pinned `catboost==1.2.10` and `requests==2.34.2` into the shared runtime, `python -m unittest discover -s tests -t . -v` passed all 133 tests. Log: `reports/arch3_final_tests.log`. Final fusion artifact reload/inference and exact OOF row/fold joins with the baseline also passed.
