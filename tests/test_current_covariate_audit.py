import unittest
import numpy as np
from src.current.covariate_audit import density_ratio


class SourceRatio(unittest.TestCase):
    def test_unchanged_source_prior_gives_unit_ratio(self):
        ntrain,ntest=267,597
        prior=ntest/(ntrain+ntest)
        np.testing.assert_allclose(density_ratio(np.full(5,prior),ntrain,ntest),np.ones(5),rtol=1e-12)

    def test_extreme_probabilities_are_bounded(self):
        values=density_ratio(np.array([0.,1.,.5]),267,597)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values>=.1).all());self.assertTrue((values<=10.).all())


if __name__=='__main__':unittest.main()
