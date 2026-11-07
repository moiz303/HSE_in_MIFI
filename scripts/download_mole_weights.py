"""Download the author-published MolE checkpoint and verify its MD5."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifacts/mole/checkpoints"
CONFIG_PATH = DESTINATION / "config.yaml"
FILES = {
    "model.pth": ("https://zenodo.org/records/10803099/files/model.pth?download=1",
                  "3d084daa5f75a0bde23caf7a24a43f8d"),
}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    config_url = "https://zenodo.org/records/10803099/files/config.yaml?download=1"
    if not CONFIG_PATH.exists():
        urllib.request.urlretrieve(config_url, CONFIG_PATH)
    config_md5 = file_md5(CONFIG_PATH)
    if config_md5 != "1192c76113766e0d742bc34cb91c81d8":
        raise ValueError(f"config.yaml: unexpected MD5 {config_md5}")
    print(f"config.yaml: MD5 verified ({config_md5})", flush=True)
    for filename, (url, expected_md5) in FILES.items():
        path = DESTINATION / filename
        if not path.exists():
            print(f"Downloading {filename} from Zenodo", flush=True)
            urllib.request.urlretrieve(url, path)
        digest = file_md5(path)
        if digest != expected_md5:
            raise ValueError(f"{filename}: expected MD5 {expected_md5}, got {digest}")
        print(f"{filename}: MD5 verified ({digest})", flush=True)


if __name__ == "__main__":
    main()
