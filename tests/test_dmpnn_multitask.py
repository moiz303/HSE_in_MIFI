import unittest

try:
    import torch
except ImportError:
    torch = None

from src.models.dmpnn_multitask import DMPNNMultiTask, batch_graphs, bounded_mse, featurize_smiles


@unittest.skipIf(torch is None, "optional PyTorch runtime is not installed")
class DMPNNGraphs(unittest.TestCase):
    def test_feature_shapes_and_forward_for_bonded_and_isolated_molecules(self):
        graphs = [featurize_smiles("CCO"), featurize_smiles("c1ccccc1"), featurize_smiles("C")]
        self.assertEqual(graphs[0].atoms.shape[1], 32)
        self.assertEqual(graphs[0].bonds.shape[1], 7)
        self.assertEqual(graphs[2].bonds.shape, (0, 7))
        output = DMPNNMultiTask(hidden_size=16, depth=2)(batch_graphs(graphs))
        self.assertEqual(tuple(output.shape), (3, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_empty_or_invalid_smiles_is_rejected(self):
        for smiles in ("", "not smiles"):
            with self.subTest(smiles=smiles), self.assertRaises(ValueError):
                featurize_smiles(smiles)

    def test_censored_loss_only_penalizes_predictions_outside_the_bound(self):
        prediction = torch.tensor([0.0, 2.0, 0.0, 2.0])
        target = [1.0, 1.0, 1.0, 1.0]
        loss = bounded_mse(prediction, target, [">", ">", "<", "<"])
        self.assertEqual(loss.tolist(), [1.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
