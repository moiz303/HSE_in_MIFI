import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from src.current.data import public_pairs,read_kaggle_training
from src.current.workflow import select_fit_rows
from src.current.models import SIModel
from src.current.results import paired_gain
from src.current.robustness import (
    _inner_splits, _weighted_frame, apply_calibration, build_variants,
    calibration_candidates,
)
from src.splits import murcko_scaffold


class DataTests(unittest.TestCase):
    def test_root_entrypoint_prepares_only_current_smiles_si_protocol(self):
        import train_model
        with patch('src.current.cli.prepare') as prepare:
            train_model.main(['--stage','prepare'])
        config=prepare.call_args.args[0]
        self.assertEqual(config['feature_columns'],['SMILES'])
        self.assertEqual(config['kaggle_target_column'],'SI VACV')
        self.assertFalse(config['kaggle_submission_authorized'])

    def test_forbidden_training_endpoint_changes_have_no_effect_on_ingested_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame=pd.DataFrame({'SMILES':['C','CC'],'SI VACV':[0.,2.],
                'CC50_uM':[3.,4.],'IC50_uM_VACV':[1.,2.],'Culture':['A','B']})
            a,b=Path(tmp)/'a.csv',Path(tmp)/'b.csv'
            frame.to_csv(a,index=False)
            frame['CC50_uM']=[-100,1e9];frame['IC50_uM_VACV']=[0,999]
            frame['Culture']=['X','Y'];frame.to_csv(b,index=False)
            pd.testing.assert_frame_equal(read_kaggle_training(a),read_kaggle_training(b))
            self.assertEqual(read_kaggle_training(a).columns.tolist(),['SMILES','SI'])
            self.assertEqual(read_kaggle_training(a).SI.tolist(),[0.,2.])

    def test_public_pairs_require_exact_same_document_line_and_correct_types(self):
        rows=[]
        for molecule,doc,line,endpoint,kind,value,mark in [
            ('C','doc','Vero','IC50','IC50',2.,''),('C','doc','Vero','CC50','CC50',10.,''),
            ('CC','a','Vero','IC50','IC50',1.,''),('CC','b','Vero','CC50','CC50',10.,''),
            ('CCC','a','Vero','IC50','EC50',1.,''),('CCC','a','Vero','CC50','CC50',10.,''),
            ('CCCC','a','Vero','IC50','IC50',1.,'>'),('CCCC','a','Vero','CC50','CC50',10.,''),
            ('CCCCC','a','Vero','IC50','IC50',1.,''),('CCCCC','a','HEL','CC50','CC50',10.,'')]:
            rows.append(dict(standard_smiles=molecule,document_id=doc,cell_line=line,endpoint=endpoint,
                measured_type=kind,value_um=value,qualifier=mark,activity_id=len(rows),source_id='id',assay_id='assay'))
        accepted,rejected=public_pairs(pd.DataFrame(rows),{'doc':'10/example'})
        self.assertEqual(len(accepted),1)
        self.assertAlmostEqual(accepted.SI.iloc[0],5.)
        self.assertEqual(accepted.doi.iloc[0],'10/example')
        self.assertIn('censored_measurement',set(rejected.reason))
        self.assertIn('endpoint_type_not_IC50_or_CC50',set(rejected.reason))

    def test_held_competition_scaffold_cannot_enter_public_fit(self):
        kaggle=pd.DataFrame({'original_row':[0,1],'standard_smiles':['c1ccccc1','C1CCCCC1'],
                             'SI':[0.,10.],'source':['kaggle']*2})
        public=pd.DataFrame({'standard_smiles':['Oc1ccccc1','c1ccncc1'],'SI':[100.,2.],'source':['chembl']*2})
        result=select_fit_rows(kaggle,public,[0],'scaffold','public_plus_kaggle')
        self.assertEqual(set(result.standard_smiles),{'C1CCCCC1','c1ccncc1'})
        only=select_fit_rows(kaggle,public,[0],'scaffold','kaggle_only')
        self.assertTrue(only.source.eq('kaggle').all())


class RidgeTests(unittest.TestCase):
    def test_bootstrap_does_not_treat_duplicated_repeats_as_independent_molecules(self):
        frame=pd.DataFrame({'standard_smiles':['C','CC','CCC'],'SI_true':[0.,1.,3.],
            'base':[3.,3.,3.],'new':[1.,2.,2.]})
        first=paired_gain(frame,'base','new')
        repeated=paired_gain(pd.concat([frame]*6,ignore_index=True),'base','new')
        self.assertEqual(first,repeated)
        self.assertEqual(first['structure_clusters'],3)

    def test_zero_targets_retained_and_scaling_is_fit_competition_only(self):
        config={'ridge_alpha':100.}
        x=np.random.default_rng(9).normal(size=(8,4))
        y=np.array([0.,1.,2.,3.,1000.,2000.,3000.,4000.])
        source=np.array([True]*4+[False]*4)
        model=SIModel('chemeleon_ridge',config).fit(x,y,np.ones(8),source)
        self.assertEqual(model.zero_targets,1)
        self.assertEqual(model.y_mean,1.5)
        self.assertTrue((model.predict(x)>=0).all())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'m.joblib';joblib.dump(model,path)
            np.testing.assert_array_equal(model.predict(x),joblib.load(path).predict(x))


class RobustnessTests(unittest.TestCase):
    def test_calibration_candidates_are_nonnegative_and_include_zero_shrinkage(self):
        y=np.array([0.,2.,4.,6.])
        prediction=np.array([1.,2.,3.,4.])
        candidates=calibration_candidates(y,prediction,[0.,.5,1.])
        self.assertIn('affine_ols',[row['name'] for row in candidates])
        shrinkage=next(row for row in candidates if row['name']=='affine_0.5')
        calibrated=apply_calibration(prediction,shrinkage)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue((calibrated>=0).all())
        self.assertTrue((apply_calibration(prediction,{'slope':1.,'intercept':-10.})==0).all())

    def test_public_weight_is_normalized_per_structure(self):
        frame=pd.DataFrame({
            'source':['kaggle','chembl','chembl','chembl'],
            'standard_smiles':['C','CC','CC','CCC'],
            'weight':[1.,.5,.5,.5],
        })
        weighted=_weighted_frame(frame,2.)
        public=weighted.source.ne('kaggle')
        totals=weighted.loc[public].groupby('standard_smiles').weight.sum()
        self.assertEqual(totals.to_dict(),{'CC':2.,'CCC':2.})
        self.assertEqual(weighted.loc[~public,'weight'].tolist(),[1.])

    def test_inner_folds_keep_molecules_and_scaffolds_together(self):
        frame=pd.DataFrame({
            'standard_smiles':['c1ccccc1','c1ccccc1','c1ccncc1',
                               'C1CCCCC1','C1CCCCC1','C1CCNCC1',
                               'CCO','CCN','CCC','CCF','CCCl','CCBr'],
        })
        for protocol in ('molecule','scaffold'):
            for train_indices,validation_indices in _inner_splits(frame,protocol,3,7):
                train=frame.iloc[train_indices].standard_smiles
                validation=frame.iloc[validation_indices].standard_smiles
                self.assertTrue(set(train).isdisjoint(set(validation)))
                if protocol=='scaffold':
                    train_scaffolds=set(train.map(murcko_scaffold))
                    validation_scaffolds=set(validation.map(murcko_scaffold))
                    self.assertTrue(train_scaffolds.isdisjoint(validation_scaffolds))

    def test_search_space_covers_source_weights_hurdle_ensemble_and_tuning(self):
        settings={
            'public_weights':[0.,.5,2.], 'ensemble_public_weight':.5,
            'ensemble_seeds':[1,2,3], 'seed':1, 'base_catboost_depth':5,
            'base_catboost_iterations':600, 'catboost_depths':[4,6],
            'catboost_iterations':[400,800], 'tuned_public_weight':.5,
        }
        variants=build_variants(settings)
        names=[variant['name'] for variant in variants]
        self.assertEqual(len(names),len(set(names)))
        self.assertTrue(any('ensemble' in name for name in names))
        self.assertTrue(any(variant['kind']=='hurdle' for variant in variants))
        self.assertTrue(any('d4_i800' in name for name in names))
        self.assertTrue(any('public_2' in name for name in names))


@unittest.skipUnless(importlib.util.find_spec('torch'),'Architecture environment required')
class NeuralTests(unittest.TestCase):
    def test_direct_and_domain_heads_train_on_zero_without_concentration_labels(self):
        config={'seed':42,'neural':{'epochs':2,'batch_size':8,'hidden':8,'dropout':0.,
                                  'learning_rate':.001,'weight_decay':.01}}
        x=np.random.default_rng(3).normal(size=(12,4))
        y=np.array([0.,0.,1.,2.,3.,4.]*2)
        for kind in ['chemeleon_mlp','chemeleon_domain','chemeleon_mlp_log1p','chemeleon_domain_log1p']:
            model=SIModel(kind,config).fit(x,y,np.ones(12),np.array([True]*6+[False]*6))
            self.assertEqual(model.zero_targets,4)
            self.assertEqual(model.public_rows,6)
            self.assertTrue(np.isfinite(model.predict(x)).all())


if __name__=='__main__':
    unittest.main()
