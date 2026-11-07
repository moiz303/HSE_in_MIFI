"""Параметры, чексуммы и результат запуска сохраняются при успехе и ошибке."""
import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

from src.run_record import record_run


class RunRecord(unittest.TestCase):
    def test_success_and_failure(self):
        config = {"seed": 7, "model": {"params": {"random_state": 9}}}
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, missing = root / "input.csv", root / "missing.csv"
                source.write_text("x\n1\n", encoding="utf-8")
                expected = self.assertRaisesRegex(RuntimeError, "failure") if fail else nullcontext()
                with expected, record_run(["test"], config, [source, missing], reports_dir=root):
                    path = next(root.glob("*.json"))
                    self.assertEqual(json.loads(path.read_text())["status"], "running")
                    if fail:
                        raise RuntimeError("failure")
                record = json.loads(path.read_text())
                self.assertEqual(record["status"], "failed" if fail else "success")
                self.assertEqual(record["config"], config)
                self.assertEqual(record["command"], ["test"])
                self.assertEqual(record["input_sha256"], {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest(), str(missing): None})
                self.assertIn("started_utc", record)
                self.assertIn("finished_utc", record)
                if fail:
                    self.assertEqual(record["error"], "RuntimeError: failure")
