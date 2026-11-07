"""Independent posterior checks for weighted public-data Gaussian processes."""
import unittest

import numpy as np
from scipy.linalg import solve

from src.current.probabilistic import LogSIProcess
from src.current.probability_public import SourceSIProcess
from src.current.search import similarity


class WeightedGaussianProcessContracts(unittest.TestCase):
    @staticmethod
    def examples():
        train = np.zeros((5, 2061), dtype=float)
        for row, bits in enumerate(([0, 1], [0, 2], [1, 3], [2, 4], [0, 4])):
            train[row, bits] = 1.0
        query = np.zeros((3, 2061), dtype=float)
        for row, bits in enumerate(([0, 4], [2, 3], [1, 3])):
            query[row, bits] = 1.0
        # Non-bit descriptor values must not enter the Tanimoto covariance.
        train[:, 2048:] = 1e4
        query[:, 2048:] = -1e5
        return train, np.array([0.0, 2.0, 7.0, 3.0, 10.0]), query

    def test_unit_weights_single_source_reproduce_unweighted_gp(self):
        train, target, query = self.examples()
        for power in (1.0, 2.0):
            with self.subTest(power=power):
                original = LogSIProcess(power).fit(train, target)
                weighted = SourceSIProcess(power, 0.5).fit(
                    train, target, np.ones(len(target)), np.ones(len(target), dtype=bool)
                )
                np.testing.assert_allclose(
                    [weighted.amplitude, weighted.noise],
                    [original.amplitude, original.noise],
                    rtol=1e-10,
                    atol=1e-10,
                )
                for kind, prediction in original.predict(query).items():
                    np.testing.assert_allclose(
                        weighted.predict(query)[kind], prediction, rtol=1e-10, atol=1e-10
                    )

    def test_heterogeneous_weights_match_direct_unwhitened_posterior(self):
        train, target, query = self.examples()
        weights = np.array([1.0, 1.0, 0.25, 0.5, 0.75])
        source = np.array([True, True, False, False, False])
        power, rho = 2.0, 0.5
        model = SourceSIProcess(power, rho).fit(train, target, weights, source)

        source_covariance = np.where(source[:, None] == source[None, :], 1.0, rho)
        covariance = (
            model.amplitude * similarity(train, train) ** power * source_covariance
            + model.noise * np.diag(1.0 / weights)
        )
        cross = (
            model.amplitude
            * similarity(query, train) ** power
            * np.where(source, 1.0, rho)[None, :]
        )
        centered = np.log1p(target) - model.center
        posterior_mean = model.center + cross @ solve(covariance, centered, assume_a="pos")
        posterior_variance = model.amplitude - np.sum(
            cross * solve(covariance, cross.T, assume_a="pos").T, axis=1
        )
        self.assertTrue((posterior_variance >= -1e-12).all())
        predicted = model.predict(query)
        for kind, variance in (
            ("median", 0.0),
            ("latent_mean", posterior_variance),
            ("observation_mean", posterior_variance + model.noise),
        ):
            expected = np.maximum(np.expm1(posterior_mean + 0.5 * variance), 0.0)
            np.testing.assert_allclose(predicted[kind], expected, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
