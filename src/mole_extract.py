"""Extract deterministic MolE embeddings using the separate encoder environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.features.mole_encoder import MolEEncoder, mol_batch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/mole/checkpoints"
ENCODER_WEIGHTS = ROOT / "artifacts/mole/encoder_state.pt"
OUTPUT = ROOT / "data/interim/mole_development_embeddings.npz"


def extract(smiles: list[str], checkpoint_dir: Path = CHECKPOINT, batch_size: int = 32,
            device: str = "cpu") -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config_path = ROOT / "artifacts/mole/config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "gin_concat":
        raise ValueError(f"Expected author gin_concat checkpoint; got {config.get('model_type')!r}")
    model_config = config["model"]
    model = MolEEncoder(num_layer=model_config["num_layer"], emb_dim=model_config["emb_dim"],
                        drop_ratio=model_config["drop_ratio"]).to(device)
    weights_path = ENCODER_WEIGHTS if ENCODER_WEIGHTS.exists() else checkpoint_dir / "model.pth"
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True,
                       mmap=weights_path.name == "model.pth")
    encoder_state = {key: tensor for key, tensor in state.items() if key in model.state_dict()}
    if set(encoder_state) != set(model.state_dict()):
        missing = sorted(set(model.state_dict()) - set(encoder_state))
        raise ValueError(f"MolE checkpoint is missing encoder weights: {missing}")
    model.load_state_dict(encoder_state, strict=True)
    if weights_path.name == "model.pth":
        torch.save(encoder_state, ENCODER_WEIGHTS)
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            graphs = mol_batch(smiles[start:start + batch_size], device)
            result.append(model(graphs).cpu().numpy())
    embeddings = np.concatenate(result, axis=0) if result else np.empty((0, model.output_dim))
    if embeddings.shape != (len(smiles), model.output_dim) or not np.isfinite(embeddings).all():
        raise ValueError("MolE returned invalid embeddings")
    return embeddings.astype(np.float32)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional development-only CSV with standard_smiles")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.input is None:
        frame = pd.concat((pd.read_csv(ROOT / "data/processed/train.csv"),
                           pd.read_csv(ROOT / "data/processed/validation.csv")), ignore_index=True)
    else:
        frame = pd.read_csv(args.input)
    if "standard_smiles" not in frame or frame.standard_smiles.isna().any():
        raise ValueError("Input needs nonempty standard_smiles rows")
    smiles = frame.standard_smiles.astype(str).tolist()
    matrix = extract(smiles, batch_size=args.batch_size, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, standard_smiles=np.asarray(smiles), embeddings=matrix)
    metadata = {"source": "MolE gin_concat author checkpoint, Zenodo 10.5281/zenodo.10803099",
                "embedding_contract": "first forward output: concatenated sum-pooled 5 GIN layers x 200 dims",
                "rows": len(smiles), "dimensions": matrix.shape[1],
                "smiles_sha256": hashlib.sha256("\n".join(smiles).encode()).hexdigest(),
                "checkpoint_md5": "3d084daa5f75a0bde23caf7a24a43f8d"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
