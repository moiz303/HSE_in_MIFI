import importlib.util
import unittest
import numpy as np
if importlib.util.find_spec('torch'):
    from src.current.bio_latent import latent_features,fit_projection,make_features
    from src.current.bio_transfer import BiologicalEncoder,public_network
else:latent_features=None


@unittest.skipUnless(latent_features is not None,'Optional architecture environment')
class BiologicalLatentContract(unittest.TestCase):
    def test_chemical_projection_is_required_and_query_does_not_refit_it(self):
        rng=np.random.default_rng(5);train=rng.normal(size=(12,8));query=rng.normal(size=(4,8))*1000
        raw=np.zeros((4,2061));logs=np.zeros((4,3));hidden=np.zeros((4,32))
        with self.assertRaisesRegex(ValueError,'Train-fitted'):
            make_features('latent_chem',raw,query,logs,hidden,np.ones(4))
        projection=fit_projection(train);before=projection['scaler'].mean_.copy()
        values=make_features('latent_chem',raw,query,logs,hidden,np.ones(4),projection)
        self.assertEqual(values.shape,(4,8+13+32+3+2+1))
        np.testing.assert_array_equal(projection['scaler'].mean_,before)
        np.testing.assert_allclose(before,train.mean(0))

    def test_hidden_features_reconstruct_original_public_predictions(self):
        learner=BiologicalEncoder();learner.dim=8;learner.mean=np.zeros(8);learner.scale=np.ones(8)
        learner.centers=[1.,2.,3.];learner.scales=[2.,3.,4.]
        model=public_network(8);learner.state={k:v.detach().numpy() for k,v in model.state_dict().items()}
        x=np.random.default_rng(7).normal(size=(7,8));hidden=latent_features(learner,x)
        self.assertEqual(hidden.shape,(7,32))
        reconstructed=(hidden@learner.state['5.weight'].T+learner.state['5.bias'])*np.array(learner.scales)+np.array(learner.centers)
        np.testing.assert_allclose(reconstructed,learner.predict_logs(x),rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
