"""Offline tests for low-rank graph readout equivalence and independent exact targets."""
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.models.architecture_heads import FrozenHeads
from src.models.chemeleon_tune import ReadoutTune, pooled_delta
from src.oof_error_audit import original_priority_targets


class TinyCache:
    def __init__(self):
        rng = np.random.default_rng(18)
        self.inputs = rng.normal(size=(24, 5)).astype('float32')
        self.p = (self.inputs @ rng.normal(size=(5, 6))).astype('float32')
        self.pooled = np.maximum(self.p, 0).reshape(12, 2, 6).mean(1)

    def batch(self, indices):
        atoms = (np.asarray(indices)[:, None]*2+np.arange(2)).ravel()
        return self.inputs[atoms], self.p[atoms], np.repeat(np.arange(len(indices)), 2), np.full(len(indices), 2, dtype='float32')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional architecture environment required')
class TuneTests(unittest.TestCase):
    def setUp(self):
        self.cache = TinyCache()
        self.indices = np.arange(12)
        self.targets = np.exp(np.random.default_rng(4).normal(size=(12, 2)))
        self.targets[0, 0] = np.nan
        self.targets[1, 1] = np.nan
        self.warm = FrozenHeads(settings={'hidden': 8, 'dropout': .1, 'epochs': 3, 'batch_size': 4,
            'learning_rate': .001, 'weight_decay': .01}).fit(self.cache.pooled, self.targets, np.ones(12), np.ones(12))
        self.cfg = {'rank': 2, 'alpha': 2., 'epochs': 2, 'batch_size': 4, 'head_lr': .001,
                    'encoder_lr': .003, 'feature_anchor': .01, 'weight_decay': .01}

    def test_low_rank_delta_matches_merged_readout_with_relu_pooling(self):
        import torch
        rng = np.random.default_rng(51)
        z = torch.tensor(rng.normal(size=(8, 5)), dtype=torch.float32)
        w = torch.tensor(rng.normal(size=(6, 5)), dtype=torch.float32)
        bias = torch.tensor(rng.normal(size=6), dtype=torch.float32)
        a = torch.tensor(rng.normal(size=(2, 5)), dtype=torch.float32)
        b = torch.tensor(rng.normal(size=(6, 2)), dtype=torch.float32)
        p = z @ w.T+bias
        actual = pooled_delta(z, p, torch.arange(2).repeat_interleave(4), torch.full((2,), 4.), a, b, .5)
        expected = (torch.relu(z @ (w+.5*b@a).T+bias)-torch.relu(p)).reshape(2, 4, 6).mean(1)
        torch.testing.assert_close(actual, expected)

    def test_zero_adapter_reproduces_warm_control(self):
        cfg = dict(self.cfg, epochs=0)
        model = ReadoutTune(self.warm, cfg).fit(self.cache, self.indices, self.targets, np.ones(12))
        np.testing.assert_allclose(model.predict_logs(self.cache, self.indices), self.warm.predict_logs(self.cache.pooled), rtol=1e-6, atol=1e-7)

    def test_adapter_trains_and_serialization_reproduces_inference(self):
        original_bias = self.warm.state['output.bias'].copy()
        model = ReadoutTune(self.warm, self.cfg).fit(self.cache, self.indices, self.targets, np.ones(12))
        self.assertGreater(model.update_frobenius, 0)
        self.assertEqual(model.n_train, [11, 11])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_array_equal(model.predict_logs(self.cache, self.indices), joblib.load(path).predict_logs(self.cache, self.indices))
        control = ReadoutTune(self.warm, self.cfg, False).fit(self.cache, self.indices, self.targets, np.ones(12))
        self.assertEqual(control.update_frobenius, 0)
        np.testing.assert_array_equal(self.warm.state['output.bias'], original_bias)
        self.assertFalse(np.array_equal(model.warm.state['output.bias'], self.warm.state['output.bias']))


class OriginalPolicyTests(unittest.TestCase):
    def test_cache_hit_does_not_import_chemprop(self):
        from src import chemeleon_states
        sentinel = object()
        with patch.object(Path, 'exists', return_value=True), patch.object(chemeleon_states, 'AtomStates', return_value=sentinel), patch.dict('sys.modules', {'chemprop': None}):
            self.assertIs(chemeleon_states.prepare(['C']), sentinel)

    def test_original_exact_heads_unmasked_independently_public_fallback_preserved(self):
        frame = pd.DataFrame({'standard_smiles': ['C', 'CC', 'CCC'], 'source': ['kaggle_train;chembl']*3,
            'IC50': [3., 4., 5.], 'CC50': [30., 40., 50.],
            'IC50_qualifier': ['>', '', ''], 'CC50_qualifier': ['', '>', '']})
        ref = pd.DataFrame({'standard_smiles': ['C', 'CC', 'CCC', 'outside'],
                            'IC50': ['2', '>8', None, '999'], 'CC50': [None, '80', '<100', '999'], 'SI': [99999]*4})
        targets, changes = original_priority_targets(frame, ref)
        np.testing.assert_allclose(targets, [[2, 30], [4, 80], [5, 50]])
        self.assertEqual(changes['IC50']['unmasked_exact'], 1)
        self.assertEqual(changes['CC50']['unmasked_exact'], 1)

    def test_oof_selection_falls_back_and_uses_measured_targets(self):
        from src.chemeleon_finetune import select
        rows = pd.DataFrame([{'protocol': p, 'repeat': r, 'SI_true': y, 'target_is_label': flag,
                              'frozen': pred, 'better': new, 'worse': bad}
            for p in ('molecule', 'scaffold') for r in range(3)
            for y, flag, pred, new, bad in [(10, False, 8, 9, 7), (200, False, 150, 175, 140),
                                           (10000, True, 9900, 1, 10000)]])
        self.assertEqual(select(rows, ['frozen', 'better', 'worse'])['selected'], 'better')
        self.assertEqual(select(rows, ['frozen', 'worse'])['selected'], 'frozen')


if __name__ == '__main__':
    unittest.main()
