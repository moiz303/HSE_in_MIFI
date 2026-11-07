import unittest
import importlib.util
import numpy as np
if importlib.util.find_spec('torch'):
    from src.current.bio_transfer import design
else:
    design=None


@unittest.skipUnless(design is not None,'Optional architecture environment')
class BiologicalDesign(unittest.TestCase):
    def test_public_log_ratios_and_source_indicator_are_explicit(self):
        raw=np.zeros((2,2061));logs=np.array([[1.,2.,5.],[3.,4.,9.]])
        result=design('bio_small',raw,logs,[1,0])
        self.assertEqual(result.shape,(2,19))
        np.testing.assert_array_equal(result[:,-3:],[[4.,3.,1.],[6.,5.,0.]])

    def test_control_features_cannot_see_auxiliary_predictions(self):
        raw=np.arange(2*2061).reshape(2,2061)
        first=design('base_fp',raw,np.zeros((2,3)),[1,1])
        second=design('base_fp',raw,np.full((2,3),1e10),[1,1])
        np.testing.assert_array_equal(first,second)


if __name__=='__main__':unittest.main()
