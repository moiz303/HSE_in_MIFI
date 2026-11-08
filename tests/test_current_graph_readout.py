import copy
import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch not installed')
class StrictGraphReadout(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from src.current.models import SIModel
        rng = np.random.default_rng(11)
        x = rng.normal(size=(12, 8)).astype(np.float32)
        self.cache = SimpleNamespace(pooled=x, inputs=x)
        self.si = np.array([0., 1., 2., 10., 0., 3., 5., 7., 0., 20., 4., 8.])
        self.cfg = {'seed': 12, 'neural': {'hidden': 8, 'dropout': 0., 'epochs': 2,
                                         'learning_rate': .001, 'weight_decay': .01, 'batch_size': 8}}
        self.warm = SIModel('chemeleon_domain_log1p', self.cfg).fit(x, self.si, np.ones(12), np.ones(12, bool))

    def test_zero_epoch_adapter_reproduces_single_si_head_and_keeps_zeros(self):
        from src.current.graph_readout import SIGraphReadout, SETTINGS
        warm_state = copy.deepcopy(self.warm.state)
        model = SIGraphReadout(self.warm, False, {**SETTINGS, 'epochs': 0})
        model.fit(self.cache, np.arange(12), self.si, np.ones(12), np.ones(12, bool))
        np.testing.assert_allclose(model.predict(self.cache, np.arange(12)), self.warm.predict(self.cache.pooled),
                                   rtol=1e-6, atol=1e-6)
        self.assertEqual(model.zero_targets, 3)
        for key in warm_state:
            np.testing.assert_array_equal(warm_state[key], self.warm.state[key])

    def test_invalid_si_and_missing_competition_rows_are_rejected(self):
        from src.current.graph_readout import SIGraphReadout
        for si, source in [(np.full(12, np.nan), np.ones(12, bool)),
                           (-np.ones(12), np.ones(12, bool)),
                           (self.si, np.zeros(12, bool))]:
            with self.assertRaises(ValueError):
                SIGraphReadout(self.warm, False).fit(self.cache, np.arange(12), si, np.ones(12), source)

    def test_adapter_updates_and_portable_si_model_keeps_original_warm(self):
        import tempfile
        from pathlib import Path
        import joblib
        from src.current.graph_readout import SIGraphReadout, SETTINGS
        from tests.test_chemeleon_finetune import TinyCache
        from src.current.models import SIModel
        cache = TinyCache()
        source = np.array([False]*4+[True]*8)
        warm = SIModel('chemeleon_domain_log1p', self.cfg).fit(cache.pooled, self.si, np.ones(12), source)
        original = copy.deepcopy(warm.state)
        cfg = {**SETTINGS, 'epochs': 2, 'batch_size': 4, 'rank': 2}
        model = SIGraphReadout(warm, True, cfg).fit(cache, np.arange(12), self.si, np.ones(12), source)
        self.assertGreater(model.update_frobenius, 0)
        self.assertEqual((model.competition_rows, model.public_rows, model.zero_targets), (8, 4, 3))
        self.assertEqual(model.warm.state['output.bias'].shape, (2,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'si.joblib'
            joblib.dump(model, path)
            np.testing.assert_array_equal(model.predict(cache, np.arange(12)),
                                          joblib.load(path).predict(cache, np.arange(12)))
        for key in original:
            np.testing.assert_array_equal(original[key], warm.state[key])


if __name__ == '__main__':
    unittest.main()
