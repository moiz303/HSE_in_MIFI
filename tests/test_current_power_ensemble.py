import copy
import importlib.util
import json
import unittest
import numpy as np
from src.current.data import ROOT
from src.current.power_ensemble import PowerEnsemble
from src.current.models import SIModel


class PowerEnsembleContracts(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional architecture environment')
    def test_log_control_seed_matches_baseline_and_inverse_mean(self):
        cfg=json.loads((ROOT/'configs/current.json').read_text());cfg=copy.deepcopy(cfg);cfg['neural']['epochs']=2
        rng=np.random.default_rng(4);x=rng.normal(size=(12,8));y=np.array([0,1,0,5,10,50,100,200,0,1,3,0.])
        source=np.ones(12,bool);w=np.ones(12)
        ensemble=PowerEnsemble(0.,cfg).fit(x,y,w,source)
        baseline=SIModel('chemeleon_domain_log1p',cfg).fit(x,y,w,source)
        np.testing.assert_array_equal(ensemble.models[0].predict(x),baseline.predict(x))
        np.testing.assert_array_equal(ensemble.predict(x),ensemble.individual(x).mean(0))

    def test_power_inverse_roundtrip_including_zero(self):
        y=np.array([0.,.1,1.,100.,36066.667])
        for power in [.25,.5]:
            z=np.expm1(power*np.log1p(y))/power
            inverse=np.expm1(np.log1p(power*z)/power)
            np.testing.assert_allclose(inverse,y,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
