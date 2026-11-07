import unittest
import numpy as np
from src.current.hurdle_gp import HurdleProcess


class HurdleContracts(unittest.TestCase):
    def test_public_positive_rows_do_not_change_zero_classifier(self):
        x=np.eye(6,2061);y=np.array([0.,1.,0.,3.,0.,10.])
        base=HurdleProcess(1.).fit(x,y,np.ones(6),np.ones(6,bool))
        augmented=HurdleProcess(1.).fit(np.vstack([x,np.ones((1,2061))*1000]),np.r_[y,1e6],np.ones(7),np.r_[np.ones(6,bool),False])
        np.testing.assert_array_equal(base.scaler.mean_,augmented.scaler.mean_)
        np.testing.assert_array_equal(base.probability(x),augmented.probability(x))
        self.assertEqual(augmented.classifier_public_rows,0)

    def test_positive_gp_uses_log_not_log1p_and_no_minus_one(self):
        x=np.eye(4,2061);model=HurdleProcess(1.).fit(x,np.full(4,3.),np.ones(4),np.ones(4,bool))
        self.assertAlmostEqual(model.center,np.log(3.))
        predictions=model.predict(x)
        self.assertTrue((predictions['latent_mean']>=3.-1e-10).all())
        self.assertTrue((predictions['observation_mean']>=predictions['latent_mean']).all())


if __name__=='__main__':unittest.main()
