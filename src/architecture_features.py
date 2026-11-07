"""Frozen author checkpoints and structure-only, row-preserving representations."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_MD5 = "6a80b54fdb7de37ef0374d302f01e8ce"


def ranged_download(url, temporary, size):
    """Bounded parallel ranges, with an atomic completion journal for interrupted transfers."""
    marker = temporary.with_suffix(temporary.suffix+".ranges.json")
    if marker.exists():
        progress = json.loads(marker.read_text())
        if progress["url"] != url or progress["size"] != size:
            raise ValueError("Partial download journal does not match the requested checkpoint")
    else:
        prefix = temporary.stat().st_size if temporary.exists() else 0
        if prefix > size:
            raise ValueError("Partial checkpoint is larger than the published file")
        progress = {"url": url, "size": size, "prefix": prefix, "complete": []}
        temporary.touch(exist_ok=True)
    def save_progress():
        pending = marker.with_suffix(marker.suffix+".tmp")
        pending.write_text(json.dumps(progress), encoding="utf-8")
        pending.replace(marker)
    save_progress()
    step = 8*1024*1024
    starts = [start for start in range(progress["prefix"], size, step) if start not in progress["complete"]]
    def fetch(start):
        end = min(start+step, size)-1
        for attempt in range(4):
            try:
                session = requests.Session()
                session.trust_env = False
                response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 90), stream=True)
                response.raise_for_status()
                expected = f"bytes {start}-{end}/{size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Author server did not honor the exact checkpoint byte range")
                if len(response.content) != end-start+1:
                    raise ValueError("Truncated checkpoint range")
                return start, response.content
            except requests.RequestException:
                if attempt == 3:
                    raise
                print(f"[weights retry] offset={start} attempt={attempt+2}/4", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, start) for start in starts]
        for future in as_completed(futures):
            start, content = future.result()
            with temporary.open("r+b") as output:
                output.seek(start)
                output.write(content)
            progress["complete"].append(start)
            save_progress()
            count = progress["prefix"]+sum(min(step, size-start) for start in progress["complete"])
            print(f"[weights] verified ranges {count/1024**2:.1f}/{size/1024**2:.1f} MiB", flush=True)
    return marker


def download(url, path, md5=None, sha256=None, size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix+".part")
        print(f"[weights] downloading {url}", flush=True)
        marker = None
        if size is not None:
            marker = ranged_download(url, temporary, size)
        else:
            session = requests.Session()
            # This laptop's environment proxy times out on author weight hosts; use verified direct TLS.
            session.trust_env = False
            with session.get(url, timeout=(20, 90), stream=True) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for block in response.iter_content(1024*1024):
                        output.write(block)
        content = temporary.read_bytes()
    else:
        content = path.read_bytes()
    if md5 and hashlib.md5(content).hexdigest() != md5:
        raise ValueError("Author checkpoint MD5 mismatch; refusing to load")
    digest = hashlib.sha256(content).hexdigest()
    if sha256 and digest != sha256:
        raise ValueError("Author checkpoint SHA256 mismatch; refusing to load")
    if not path.exists():
        temporary.replace(path)
        if marker is not None:
            marker.unlink()
    return {"url": url, "file": path.relative_to(ROOT).as_posix(), "sha256": digest, "bytes": len(content)}


def chemeleon(smiles, random=False, batch_size=8):
    import torch
    from chemprop import data, featurizers, nn
    torch.set_num_threads(4)
    path = ROOT / "artifacts/pretrained/chemeleon_mp.pt"
    meta = download(CHEMELEON_URL, path, md5=CHEMELEON_MD5)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    torch.manual_seed(20261106)
    model = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
    if not random:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    aggregator = nn.MeanAggregation()
    output = []
    started = time.monotonic()
    for begin in range(0, len(smiles), batch_size):
        graphs = [featurizer(Chem.MolFromSmiles(s)) for s in smiles[begin:begin+batch_size]]
        batch = data.BatchMolGraph(graphs)
        with torch.inference_mode():
            encoded = aggregator(model(batch), batch.batch).cpu().numpy()
        output.append(encoded)
        if begin % (batch_size*8) == 0:
            print(f"[encoder] {'random control' if random else 'CheMeleon'} {min(begin+batch_size,len(smiles))}/{len(smiles)}; {time.monotonic()-started:.1f}s", flush=True)
    result = np.concatenate(output)
    return result, {"checkpoint": meta, "random_control": random, "frozen": True,
                    "dimensions": result.shape[1], "elapsed_seconds": time.monotonic()-started}


def unimol(smiles, batch_size=8):
    import torch
    from unittest.mock import patch
    torch.set_num_threads(4)
    directory = ROOT / "artifacts/pretrained/unimol"
    if (directory / "source.json").exists():
        source = json.loads((directory / "source.json").read_text())
    else:
        session = requests.Session()
        session.trust_env = False
        response = session.get("https://huggingface.co/api/models/dptech/Uni-Mol-Models", params={"blobs": "true"}, timeout=(20, 60))
        response.raise_for_status()
        payload = response.json()
        source = {"revision": payload["sha"], "files": {f["rfilename"]: f.get("lfs", {}) for f in payload["siblings"]}}
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    metas = []
    for filename in ("mol_pre_no_h_220816.pt", "mol.dict.txt"):
        url = f"https://huggingface.co/dptech/Uni-Mol-Models/resolve/{source['revision']}/{filename}"
        metas.append(download(url, directory / filename, sha256=source["files"][filename].get("sha256"),
                              size=source["files"][filename].get("size")))
    weights = directory / "mol_pre_no_h_220816.pt"
    default_load = torch.load
    def load_author_checkpoint(path, *args, **kwargs):
        if Path(path).resolve() == weights.resolve():
            # Only the hash-checked author's legacy checkpoint may contain training metadata.
            kwargs.setdefault("weights_only", False)
        return default_load(path, *args, **kwargs)
    from unimol_tools import UniMolRepr
    with patch("torch.load", load_author_checkpoint):
        encoder = UniMolRepr(data_type="molecule", batch_size=batch_size, remove_hs=True,
            use_cuda=False, model_name="unimolv1", pretrained_model_path=str(weights),
            pretrained_dict_path=str(directory / "mol.dict.txt"), max_atoms=256)
    checkpoint = default_load(weights, map_location="cpu", weights_only=False)
    checkpoint = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    encoder_keys = {k: v for k, v in encoder.model.state_dict().items() if not k.startswith("classification_head")}
    if not all(k in checkpoint and tuple(checkpoint[k].shape) == tuple(v.shape) for k, v in encoder_keys.items()):
        raise ValueError("Uni-Mol encoder checkpoint coverage/shape mismatch")
    encoder.params.update(seed=42, multi_process=False, num_workers=0)
    output, failed = [], []
    started = time.monotonic()
    for begin in range(0, len(smiles), 32):
        chunk = smiles[begin:begin+32]
        try:
            encoded = encoder.get_repr(chunk)
            encoded = encoded["cls_repr"] if isinstance(encoded, dict) else encoded
            encoded = np.asarray(encoded, dtype=np.float32)
            if encoded.shape != (len(chunk), 512) or not np.isfinite(encoded).all():
                raise ValueError("Invalid Uni-Mol representation shape or values")
        except Exception as error:
            print(f"[Uni-Mol] retrying chunk molecule-by-molecule: {type(error).__name__}: {error}", flush=True)
            encoded = []
            for position, smile in enumerate(chunk):
                try:
                    value = encoder.get_repr([smile])
                    value = value["cls_repr"] if isinstance(value, dict) else value
                    value = np.asarray(value, dtype=np.float32).reshape(1, 512)[0]
                    if not np.isfinite(value).all():
                        raise ValueError("nonfinite embedding")
                except Exception:
                    value = np.zeros(512, dtype=np.float32)
                    failed.append(begin+position)
                encoded.append(value)
            encoded = np.asarray(encoded)
        output.append(encoded)
        if len(failed) > max(16, int(.05*min(begin+32, len(smiles)))):
            raise RuntimeError("Uni-Mol encoder failed systematically; refusing to score a zero-vector substitute")
        print(f"[encoder] Uni-Mol {min(begin+32,len(smiles))}/{len(smiles)}; failed={len(failed)}; {time.monotonic()-started:.1f}s", flush=True)
    if len(failed) > .05*len(smiles):
        raise RuntimeError("Uni-Mol exception fallback exceeds the predeclared five-percent limit")
    result = np.concatenate(output)
    failure_flags = np.zeros((len(smiles), 1), dtype=np.float32)
    failure_flags[failed] = 1
    return np.concatenate([result, failure_flags], axis=1), {"checkpoints": metas, "revision": source["revision"],
        "frozen": True, "dimensions": 513, "conformers": 1, "seed": 42, "encoder_keys_verified": len(encoder_keys),
        "zero_embedding_fallback_rows": failed, "elapsed_seconds": time.monotonic()-started}


def representations(smiles, name, batch_size=8):
    directory = ROOT / "data/architecture_cache"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps({"smiles": smiles, "encoder": name, "version": 1}, sort_keys=True).encode()).hexdigest()
    path = directory / f"{name}_{key}.npz"
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists():
        cached = np.load(path)
        if cached["matrix"].shape[0] != len(smiles):
            raise ValueError("Cached representation row mismatch")
        return cached["matrix"], json.loads(metadata.read_text(encoding="utf-8"))
    if name in ("chemeleon", "random_chemeleon"):
        matrix, audit = chemeleon(smiles, random=name.startswith("random"), batch_size=batch_size)
    elif name == "unimol":
        matrix, audit = unimol(smiles, batch_size)
    else:
        raise ValueError(name)
    if len(matrix) != len(smiles) or not np.isfinite(matrix).all():
        raise ValueError("Representation rows/values invalid")
    np.savez_compressed(path, matrix=matrix.astype(np.float32))
    audit.update({"smiles_sha256": key, "rows": len(smiles), "cache": path.relative_to(ROOT).as_posix()})
    metadata.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return matrix, audit


def main():
    import pandas as pd
    from src.data.standardize import standardize_smiles
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", choices=["chemeleon", "random_chemeleon", "unimol"])
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(ROOT / f"data/processed/{p}.csv") for p in ("train", "validation")], ignore_index=True)
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv", usecols=["index", "SMILES"])
    smiles = frame.standard_smiles.tolist()+[standardize_smiles(s) for s in test.SMILES]
    matrix, audit = representations(smiles, args.encoder)
    directory = ROOT / "reports/architectures"
    directory.mkdir(exist_ok=True)
    (directory / f"{args.encoder}_representation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[encoder done] {args.encoder} shape={matrix.shape}", flush=True)


if __name__ == "__main__":
    main()
