"""Run records contain enough information to identify a CLI invocation."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.run_record import record_run


class RunRecord(unittest.TestCase):
    def test_metadata_and_console_log_are_saved(self):
        config = {"seed": 7, "splits": {"seed": 8}, "model": {"params": {"random_state": 9}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            source.write_text("x\n1\n", encoding="utf-8")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with record_run(["python", "-m", "src.pipeline", "test"], config,
                                [source], reports_dir=root / "runs"):
                    print("synthetic run output")

            metadata_path = next((root / "runs").glob("*.json"))
            log_path = next((root / "runs").glob("*.log"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "success")
            self.assertEqual(metadata["config"], config)
            self.assertEqual(metadata["seed"], {
                "global": 7, "split": 8, "model_random_state": 9,
            })
            self.assertIn("numpy", metadata["packages"])
            self.assertEqual(
                metadata["input_sha256"][str(source)],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertIn("synthetic run output", log_path.read_text(encoding="utf-8"))

    def test_failed_run_is_marked_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "expected failure"):
                    with record_run(["test"], {}, reports_dir=root):
                        raise RuntimeError("expected failure")
            metadata_path = next(root.glob("*.json"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("expected failure", metadata["error"])


if __name__ == "__main__":
    unittest.main()
