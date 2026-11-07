import copy
import importlib.util
import unittest
import numpy as np
if importlib.util.find_spec('torch'):
    from src.current.bio_mean import MeanCalibration,LOSSES
else:MeanCalibration=None


@unittest.skipUnless(MeanCalibration is not None,'Optional architecture environment')
class MeanCalibrationContracts(unittest.TestCase):
    def test_original_si_units_and_zeros_are_preserved(self):
        from src.current.models import SIModel
        cfg={'catboost':{'iterations':20,'depth':2,'learning_rate':.1,'loss_function':'RMSE','verbose':False,'allow_writing_files':False,'random_seed':42,'thread_count':1}}
        x=np.arange(12,dtype=float).reshape(-1,1);y=np.array([0.,0.,1.,3.,10.,100.,0.,5.,50.,200.,1000.,36066.667]);w=np.ones(12);source=np.ones(12,bool)
        for loss in ['rmse','tweedie15']:
            model=MeanCalibration(loss,cfg).fit(x,y,w,source)
            local=copy.deepcopy(cfg);local['catboost']['loss_function']=LOSSES[loss]
            direct=SIModel('catboost_direct',local).fit(x,y/100.,w,source)
            np.testing.assert_array_equal(model.predict(x),direct.predict(x)*100.)
            self.assertEqual(model.head.zero_targets,3)
            self.assertTrue(np.isfinite(model.predict(x)).all())
            self.assertTrue((model.predict(x)>=0).all())


if __name__=='__main__':unittest.main()
