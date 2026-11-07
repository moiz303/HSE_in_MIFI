"""Contract checks for the second SI-only model block."""
import pickle
import unittest
import numpy as np
from src.current.search import Candidate, similarity, specifications


class SearchContracts(unittest.TestCase):
    def test_tanimoto_ignores_descriptor_units(self):
        x=np.zeros((2,2061));x[0,0]=1;x[1,1]=1
        x[:,2048:]=1e9
        np.testing.assert_array_equal(similarity(x,x),np.eye(2))

    def test_log_neighbors_preserve_target_scale(self):
        x=np.ones((2,2048))
        model=Candidate({'kind':'neighbors','target':'log1p','power':4.}).fit(x,[0,8],[1,1])
        # Geometric averaging of SI+1 must be inverted, never treated as raw SI.
        np.testing.assert_allclose(model.predict(x),[2,2])

    def test_kernel_zero_targets_and_serialization(self):
        x=np.eye(3,2048)
        for target in ('raw','log1p'):
            model=Candidate({'kind':'kernel','target':target,'alpha':1.,'power':2.}).fit(x,[0,0,0],[1,1,1])
            np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(x),np.zeros(3))

    def test_specifications_have_unique_names(self):
        specs=specifications()
        self.assertEqual(len(specs),len({row['name'] for row in specs}))


if __name__=='__main__':unittest.main()
