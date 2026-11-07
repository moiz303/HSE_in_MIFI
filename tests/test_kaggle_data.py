"""Тесты запрета на арифметику и загрузки данных соревнования.

Организаторы запретили выводить целевую переменную из колонок тестового файла.
Эти тесты закрепляют запрет в коде: если кто-то снова начнёт считать SI из
CC50_uM и IC50_uM_VACV на пути сабмита, тест покраснеет.
"""

import unittest
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.data import kaggle_data as kd


class ForbiddenFeatures(unittest.TestCase):
    def test_cc50_in_features_is_rejected(self):
        frame = pd.DataFrame({"SMILES": ["CCO"], "CC50_uM": [10.0]})
        with self.assertRaises(ValueError) as caught:
            kd.assert_no_forbidden_features(frame)
        self.assertIn("CC50_uM", str(caught.exception))

    def test_ic50_in_features_is_rejected(self):
        frame = pd.DataFrame({"SMILES": ["CCO"], "IC50_uM_VACV": [1.0]})
        with self.assertRaises(ValueError):
            kd.assert_no_forbidden_features(frame)

    def test_smiles_only_frame_passes(self):
        frame = pd.DataFrame({"index": [0], "SMILES": ["CCO"], "strain": ["WR"]})
        kd.assert_no_forbidden_features(frame)

    def test_the_arithmetic_helpers_are_gone(self):
        # Прямая проверка, что путь не вернулся под старыми именами.
        for name in ("selectivity_from_columns", "verify_rule",
                     "INACTIVE_MARKER", "NO_CC50_VALUE"):
            self.assertFalse(
                hasattr(kd, name),
                f"{name} восстановлен, а арифметика по тесту запрещена организаторами",
            )


class TrainingFrame(unittest.TestCase):
    def test_current_targets_override_legacy_training_file_and_keep_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pd.DataFrame({'SMILES':['CCO'],'SI VACV':[10000.]}).to_csv(directory/'train.csv',index=False)
            pd.DataFrame({'SMILES':['CCO'],'SI VACV':[0.]}).to_csv(directory/'train_v2.csv',index=False)
            self.assertEqual(kd.load('train',directory)['SI VACV'].tolist(),[0.])
            self.assertEqual(pd.read_csv(directory/'train.csv')['SI VACV'].tolist(),[10000.])

    def test_legacy_only_directory_cannot_silently_train_on_obsolete_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pd.DataFrame({'SMILES':['CCO'],'SI VACV':[10000.]}).to_csv(directory/'train.csv',index=False)
            with self.assertRaisesRegex(SystemExit,'train_v2.csv'):
                kd.load('train',directory)

    def test_cached_legacy_archive_still_fetches_corrected_training_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'inputs'
            directory.mkdir()
            with zipfile.ZipFile(directory/'example.zip','w') as archive:
                archive.writestr('train.csv','SMILES,SI VACV\nCCO,10000\n')
            def download(args, **kwargs):
                self.assertIn('train_v2.csv',args)
                pd.DataFrame({'SMILES':['CCO'],'SI VACV':[0.]}).to_csv(directory/'train_v2.csv',index=False)
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            def isolated_path(value):
                return Path(tmp)/'reports' if str(value) == 'reports' else Path(value)
            with patch.object(kd.subprocess,'run',side_effect=download) as call, patch.object(kd,'Path',side_effect=isolated_path):
                kd.download('example',directory)
            self.assertEqual(call.call_count,1)
            self.assertEqual(kd.load('train',directory)['SI VACV'].tolist(),[0.])

    def test_columns_are_renamed_to_the_contract(self):
        raw = pd.DataFrame({
            "SMILES": ["CCO"], "CC50_uM": [100.0], "IC50_uM_VACV": [4.0],
            "SI VACV": [25.0], "Culture": ["Vero"], "Title": ["x"], "strain": ["WR"],
        })

        class Stub(dict):
            pass

        # Подменяем чтение файла: сети и данных соревнования в тестах нет.
        original = kd.load
        kd.load = lambda name, directory=kd.KAGGLE_DIR: raw.copy()
        try:
            frame = kd.training_frame()
        finally:
            kd.load = original

        for column in ("smiles", "CC50", "IC50", "SI", "cell_line", "source", "virus"):
            self.assertIn(column, frame.columns)
        self.assertEqual(frame.iloc[0]["source"], "kaggle_train")
        self.assertEqual(frame.iloc[0]["virus"], "vaccinia virus")
        self.assertAlmostEqual(frame.iloc[0]["SI"], 25.0)


if __name__ == "__main__":
    unittest.main()
