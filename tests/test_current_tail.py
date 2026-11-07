import unittest
import numpy as np
from src.current.tail_controls import capped


class TailControls(unittest.TestCase):
    def test_threshold_depends_on_fit_targets_only(self):
        target=[0.,1.,10.,100.]
        prediction=np.array([2.,10000.])
        np.testing.assert_array_equal(capped(prediction,target,.95),[2.,np.quantile(target,.95)])
        # More extreme query rows cannot change the cap on existing queries.
        np.testing.assert_array_equal(capped(np.r_[prediction,1e20],target,.95)[:2],capped(prediction,target,.95))

    def test_invalid_targets_fail(self):
        for target in ([],[-1],[np.nan]):
            with self.assertRaises(ValueError):capped([1],target,.99)


if __name__=='__main__':unittest.main()
