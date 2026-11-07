import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional architecture environment')
class PublicAuxiliaryContracts(unittest.TestCase):
    def test_censored_measurements_have_one_sided_penalties(self):
        import torch
        from src.current.auxiliary import bound_errors
        pred=torch.tensor([1.,3.,1.,3.,1.,3.])
        direction=torch.tensor([1,1,-1,-1,0,0])
        np.testing.assert_array_equal(bound_errors(pred,torch.full((6,),2.),direction).numpy(),[-1,0,0,1,-1,1])

    def test_public_tasks_stay_distinct_and_official_structures_are_excluded(self):
        from src.current.auxiliary import public_measurements
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for relative in ['data/experiments','data/processed','data/raw/kaggle']:(root/relative).mkdir(parents=True,exist_ok=True)
            public=pd.DataFrame({'standard_smiles':['CCOc1ccccc1','COc1ccccc1','CCc1ccccc1','Oc1ccccc1','c1ccncc1','CCCc1ccccc1'],
                'measured_type':['IC50','EC50','CC50','IC50','CC50','GI50'],'qualifier':['','>','','','',''],
                'value_um':[1.,2.,3.,4.,5.,6.],'document_id':['D']*6})
            public.to_csv(root/'data/experiments/independent_endpoints.csv',index=False)
            pd.DataFrame({'standard_smiles':['c1ccncc1']}).to_csv(root/'data/processed/dataset.csv',index=False)
            (root/'data/processed/splits.json').write_text(json.dumps({'indices':{'holdout':[0]}}))
            # No official concentration fields exist in this input at all.
            pd.DataFrame({'SMILES':['Oc1ccccc1']}).to_csv(root/'data/raw/kaggle/train_v2.csv',index=False)
            pd.DataFrame({'document_id':['D'],'doi':['10.0/example']}).to_csv(root/'data/raw/chembl.csv',index=False)
            with patch('src.current.auxiliary.ROOT',root):result=public_measurements()
            self.assertEqual(result.measured_type.tolist(),['IC50','EC50','CC50'])
            self.assertEqual(result.task.tolist(),[0,1,2])
            self.assertEqual(result.bound.tolist(),[0,1,0])
            self.assertTrue(result.doi.eq('10.0/example').all())


if __name__=='__main__':unittest.main()
