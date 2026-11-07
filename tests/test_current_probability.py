import unittest
import numpy as np
from src.current.probabilistic import objective,LogSIProcess


class GaussianProcessContracts(unittest.TestCase):
    def test_marginal_likelihood_gradient_matches_finite_difference(self):
        kernel=np.eye(4)*.8+np.ones((4,4))*.2
        y=np.array([-1.,.5,2.,-1.5]);theta=np.log([1.3,.7])
        _,gradient=objective(theta,kernel,y)
        finite=[]
        for i in range(2):
            step=np.zeros(2);step[i]=1e-5
            finite.append((objective(theta+step,kernel,y)[0]-objective(theta-step,kernel,y)[0])/2e-5)
        np.testing.assert_allclose(gradient,finite,rtol=1e-6,atol=1e-7)

    def test_posterior_uncertainty_increases_original_scale_mean(self):
        x=np.eye(5,2048)
        model=LogSIProcess(2.).fit(x,[0.,1.,10.,100.,1000.])
        values=model.predict(x)
        self.assertTrue(np.isfinite(values['observation_mean']).all())
        self.assertTrue((values['median']<=values['latent_mean']+1e-10).all())
        self.assertTrue((values['latent_mean']<=values['observation_mean']+1e-10).all())


if __name__=='__main__':unittest.main()
