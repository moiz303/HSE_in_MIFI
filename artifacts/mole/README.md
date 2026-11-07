# MolE frozen encoder artifacts

The feature trunk is from the author-published `rolayoalarcon/MolE` `gin_concat` model, Zenodo record [10.5281/zenodo.10803099](https://zenodo.org/records/10803099). The upstream GitHub repository is MIT licensed. The upstream record lists the 803.8 MB checkpoint but leaves the Zenodo rights field blank; retain the source citation with redistributed model files.

`config.yaml` mirrors the author-published settings with whitespace normalized. `encoder_state.pt` contains only atom/bond embeddings, the five GIN layers, and their batch-normalization values used to compute the first `GINet.forward` output. It omits the 8000-wide projection head. The original config and 803.8 MB `model.pth` are downloaded locally when needed and are excluded from git.

To reproduce the upstream checkpoint download, run `python scripts/download_mole_weights.py`. It verifies MD5 `1192c76113766e0d742bc34cb91c81d8` for the config and `3d084daa5f75a0bde23caf7a24a43f8d` for `model.pth`.
