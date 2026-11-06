"""Записывает конфигурацию и полный вывод каждого CLI-запуска."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "rdkit", "joblib",
    "chembl-webresource-client", "kaggle",
)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _metadata(command, config, inputs):
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    hashes = {}
    for value in inputs:
        path = Path(value)
        label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        hashes[label] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )

    return {
        "command": list(command),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": {
            "global": config.get("seed"),
            "split": config.get("splits", {}).get("seed"),
            "model_random_state": config.get("model", {}).get("params", {}).get("random_state"),
        },
        "packages": versions,
        "config": config,
        "input_sha256": hashes,
    }


@contextmanager
def record_run(command, config, inputs=(), reports_dir: Path | None = None):
    """Mirror console output to a log and save run metadata beside it."""
    directory = reports_dir or ROOT / "reports" / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = directory / f"run_{run_id}.log"
    metadata_path = directory / f"run_{run_id}.json"
    record = _metadata(command, config, inputs)
    record.update({"started_utc": started.isoformat(), "status": "running"})
    metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    with log_path.open("w", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f"[run] metadata: {metadata_path}")
            try:
                yield
            except BaseException as error:
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
                print(f"[run] failed: {record['error']}", file=sys.stderr)
                raise
            else:
                record["status"] = "success"
            finally:
                record["finished_utc"] = datetime.now(timezone.utc).isoformat()
                metadata_path.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"[run] log: {log_path}")
