import importlib.util
import unittest
import numpy as np
if importlib.util.find_spec('torch'):
    from src.current.bio_latent import latent_features
    from src.current.bio_transfer import BiologicalEncoder,public_network
else:latent_features=None


@unittest.skipUnless(latent_features is not None,'Optional architecture environment')
class BiologicalLatentContract(unittest.TestCase):
    def test_hidden_features_reconstruct_original_public_predictions(self):
        learner=BiologicalEncoder();learner.dim=8;learner.mean=np.zeros(8);learner.scale=np.ones(8)
        learner.centers=[1.,2.,3.];learner.scales=[2.,3.,4.]
        model=public_network(8);learner.state={k:v.detach().numpy() for k,v in model.state_dict().items()}
        x=np.random.default_rng(7).normal(size=(7,8));hidden=latent_features(learner,x)
        self.assertEqual(hidden.shape,(7,32))
        reconstructed=(hidden@learner.state['5.weight'].T+learner.state['5.bias'])*np.array(learner.scales)+np.array(learner.centers)
        np.testing.assert_allclose(reconstructed,learner.predict_logs(x),rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
