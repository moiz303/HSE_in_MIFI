import hashlib
from pathlib import Path
import tempfile
import unittest
from src.hypothesis_results import official_score_for


class OfficialScores(unittest.TestCase):
    def test_completed_score_requires_matching_bytes_and_competition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"submission.csv"
            path.write_bytes(b"index,SI VACV\n1,2\n")
            row = {"competition": "molecular-activity-prediction-hack", "status": "SubmissionStatus.COMPLETE",
                "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "public_score": 123.45}
            self.assertEqual(official_score_for(path, [row]), 123.45)
            self.assertIsNone(official_score_for(path, [{**row, "status": "SubmissionStatus.PENDING"}]))
            self.assertIsNone(official_score_for(path, [{**row, "competition": "other"}]))
            path.write_bytes(b"index,SI VACV\n1,3\n")
            self.assertIsNone(official_score_for(path, [row]))
