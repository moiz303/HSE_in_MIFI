import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional architecture environment')
class TabularProjection(unittest.TestCase):
    def test_query_rows_cannot_change_fitted_projection(self):
        from src.current.tabular import projection
        rng=np.random.default_rng(9)
        train=rng.normal(size=(40,50));query=rng.normal(size=(5,50))
        raw_train=rng.normal(size=(40,2061));raw_query=rng.normal(size=(5,2061))
        a,_,first=projection(train,query,raw_train,raw_query)
        b,_,second=projection(train,query*1e6,raw_train,raw_query*1e6)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(first['scaler'].mean_,train.mean(0))
        np.testing.assert_array_equal(first['pca'].components_,second['pca'].components_)
        self.assertEqual(a.shape,(40,45))


if __name__=='__main__':unittest.main()
