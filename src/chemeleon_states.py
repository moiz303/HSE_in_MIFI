"""Cache the frozen graph prefix for genuine low-rank encoder-readout fine-tuning."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from rdkit import Chem
from src.architecture_features import ROOT, download, CHEMELEON_URL, CHEMELEON_MD5, representations


class AtomStates:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory/"metadata.json").read_text())
        self.inputs = np.load(self.directory/"inputs.npy", mmap_mode="r")
        self.preactivation = np.load(self.directory/"preactivation.npy", mmap_mode="r")
        self.offsets = np.load(self.directory/"offsets.npy")
        self.pooled = np.load(self.directory/"pooled.npy", mmap_mode="r")

    def batch(self, molecule_indices):
        indices = np.asarray(molecule_indices, dtype=int)
        if not len(indices) or (indices < 0).any() or (indices >= len(self.pooled)).any():
            raise ValueError("Invalid molecule indices for graph prefix cache")
        counts = self.offsets[indices+1]-self.offsets[indices]
        z = np.concatenate([self.inputs[self.offsets[i]:self.offsets[i+1]] for i in indices])
        p = np.concatenate([self.preactivation[self.offsets[i]:self.offsets[i+1]] for i in indices])
        groups = np.repeat(np.arange(len(indices)), counts)
        return z, p, groups.astype(np.int64), counts.astype(np.float32)


def prepare(smiles, batch_size=8):
    key = hashlib.sha256(json.dumps({"smiles": smiles, "version": 1, "md5": CHEMELEON_MD5}, sort_keys=True).encode()).hexdigest()
    directory = ROOT / "data/architecture_cache" / f"chemeleon_atoms_{key}"
    if (directory/"metadata.json").exists():
        return AtomStates(directory)
    import torch
    from chemprop import nn, data, featurizers
    torch.set_num_threads(4)
    directory.mkdir(parents=True, exist_ok=True)
    weight_path = ROOT / "artifacts/pretrained/chemeleon_mp.pt"
    checkpoint_meta = download(CHEMELEON_URL, weight_path, md5=CHEMELEON_MD5)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    encoder = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    encoder.eval()
    if encoder.W_d is not None or not isinstance(encoder.tau, torch.nn.ReLU) or encoder.dropout.p != 0:
        raise ValueError("Cached prefix implementation requires the author's ReLU/no-dropout readout")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None or m.GetNumAtoms() == 0 for m in mols):
        raise ValueError("Invalid molecule in prefix cache")
    offsets = np.concatenate([[0], np.cumsum([m.GetNumAtoms() for m in mols])]).astype(np.int64)
    dim_in, dim_out = encoder.W_o.in_features, encoder.W_o.out_features
    inputs = np.lib.format.open_memmap(directory/"inputs.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), dim_in))
    preactivation = np.lib.format.open_memmap(directory/"preactivation.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), dim_out))
    pooled = np.lib.format.open_memmap(directory/"pooled.npy", mode="w+", dtype=np.float32, shape=(len(smiles), dim_out))
    captured = {}
    def hook(module, args, output):
        captured["z"] = args[0].detach().numpy()
        captured["p"] = output.detach().numpy()
    handle = encoder.W_o.register_forward_hook(hook)
    featurizer, aggregator = featurizers.SimpleMoleculeMolGraphFeaturizer(), nn.MeanAggregation()
    started = time.monotonic()
    for begin in range(0, len(mols), batch_size):
        end = min(begin+batch_size, len(mols))
        batch = data.BatchMolGraph([featurizer(m) for m in mols[begin:end]])
        with torch.inference_mode():
            h = encoder(batch)
            vectors = aggregator(h, batch.batch).numpy()
        if len(captured["z"]) != offsets[end]-offsets[begin]:
            raise AssertionError("Atom cache row ordering mismatch")
        inputs[offsets[begin]:offsets[end]] = captured["z"]
        preactivation[offsets[begin]:offsets[end]] = captured["p"]
        pooled[begin:end] = vectors
        if begin % 64 == 0:
            print(f"[prefix] {end}/{len(mols)} molecules; {time.monotonic()-started:.1f}s", flush=True)
    handle.remove()
    inputs.flush(); preactivation.flush(); pooled.flush()
    np.save(directory/"offsets.npy", offsets)
    reference, _ = representations(smiles, "chemeleon", batch_size)
    error = float(np.max(np.abs(pooled-reference)))
    np.testing.assert_allclose(pooled, reference, rtol=1e-5, atol=1e-5)
    # Keep the old pooled vectors exactly, so a zero adapter reproduces the accepted control.
    pooled[:] = reference
    pooled.flush()
    metadata = {"ordered_smiles_sha256": key, "molecules": len(smiles), "atoms": int(offsets[-1]),
        "input_dim": dim_in, "output_dim": dim_out, "checkpoint": checkpoint_meta,
        "frozen_prefix": ["W_i", "W_h"], "adapted_layer": "W_o", "activation": "ReLU",
        "reference_pooling_max_absolute_difference": error, "labels_used": False,
        "bytes": int(offsets[-1])*(dim_in+dim_out)*4, "elapsed_seconds": time.monotonic()-started}
    (directory/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return AtomStates(directory)


if __name__ == "__main__":
    from src.architecture_workflow import inputs
    frame, reference, weights, targets, test, smiles = inputs()
    cache = prepare(smiles)
    print(json.dumps(cache.metadata, indent=2))
