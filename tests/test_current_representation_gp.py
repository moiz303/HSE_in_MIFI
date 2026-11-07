import unittest
import numpy as np
from src.current.representation_gp import RepresentationProcess


class RepresentationGPContracts(unittest.TestCase):
    def inputs(self):
        rng=np.random.default_rng(22)
        return np.column_stack([rng.integers(0,2,(12,2048)),rng.normal(size=(12,20))])

    def test_kernel_is_positive_semidefinite(self):
        x=self.inputs()
        for kind in ['cosine','rbf','rbf_tanimoto']:
            model=RepresentationProcess(kind).fit(x,np.arange(12),np.ones(12),np.ones(12,bool))
            kernel=model.kernel(x,x)
            self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(),-1e-10)
            np.testing.assert_allclose(np.diag(kernel),1.,atol=1e-10)

    def test_query_distribution_does_not_change_scaler(self):
        x=self.inputs();model=RepresentationProcess('rbf').fit(x,np.arange(12),np.ones(12),np.ones(12,bool))
        before=model.mean.copy();first=model.predict(x[:2])['observation_mean']
        model.predict(x*1000)
        np.testing.assert_array_equal(before,model.mean)
        np.testing.assert_array_equal(first,model.predict(x[:2])['observation_mean'])


if __name__=='__main__':unittest.main()
