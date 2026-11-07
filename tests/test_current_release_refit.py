"""Default planning must never consume the protected competition targets."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from src.current import release_refit


class ReservedRowsGuard(unittest.TestCase):
    def test_default_only_reads_structures_and_never_trains(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'configs').mkdir();(root/'data/raw/kaggle').mkdir(parents=True)
            train=root/'data/raw/kaggle/train_v2.csv'
            pd.DataFrame({'SMILES':['CC']*320,'SI VACV':['DO_NOT_PARSE_RESERVED_TARGETS']*320,
                'CC50_uM':['FORBIDDEN']*320,'IC50_uM_VACV':['FORBIDDEN']*320}).to_csv(train,index=False)
            (root/'configs/current.json').write_text(json.dumps({'expected_train_sha256':hashlib.sha256(train.read_bytes()).hexdigest()}))
            (root/'configs/current_release_refit.json').write_text(json.dumps({'base_config':'configs/current.json','release_authorized':False}))
            with patch.object(release_refit,'ROOT',root),patch.object(release_refit,'REPORTS',root/'reports/release'),\
                 patch.object(release_refit,'load',return_value=(pd.DataFrame(index=range(267)),pd.DataFrame(index=range(80)),{'version':'fixture'})),\
                 patch.object(release_refit,'read_kaggle_training',side_effect=AssertionError('Reserved targets parsed')),\
                 patch.object(release_refit,'SIModel',side_effect=AssertionError('Training without release')),\
                 patch('sys.argv',['release_refit']):
                release_refit.main()
            report=json.loads((root/'reports/release/plan.json').read_text())
            self.assertTrue(report['no_training_performed'])
            self.assertFalse(report['reserved_targets_evaluated'])
            self.assertEqual(report['reserved_rows_to_release'],53)
            self.assertEqual(report['planned_fit_rows'],400)


if __name__=='__main__':unittest.main()
