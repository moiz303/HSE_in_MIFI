"""Сохраняет команду, конфиг и чексуммы входных файлов запуска."""

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def record_run(command, config, inputs=(), reports_dir=None):
    directory = reports_dir or ROOT / "reports" / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    path = directory / f"run_{started.strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    hashes = {}
    for source in map(Path, inputs):
        if source.is_file():
            with source.open("rb") as stream:
                hashes[str(source)] = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            hashes[str(source)] = None
    record = {
        "command": list(command),
        "config": config,
        "input_sha256": hashes,
        "started_utc": started.isoformat(),
        "status": "running",
    }

    def save():
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    save()
    print(f"[run] {path}")
    try:
        yield
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        record["status"] = "success"
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
