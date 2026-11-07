import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from src import hypotheses as suite
from src.data.independent_endpoints import endpoint_rows
from tests.test_download import activity
from src.models.xgb_weighted import XGBWeightedTwoHead


class IndependentEndpoints(unittest.TestCase):
    def test_user_submission_threshold_is_strict_and_smaller_is_better(self):
        from src.hypothesis_results import eligible
        self.assertTrue(eligible(89., 100.))
        self.assertFalse(eligible(90., 100.))
        self.assertFalse(eligible(110., 100.))

    def test_unpaired_endpoints_keep_provenance_without_made_up_partners(self):
        inhibition = [activity(activity_id=1, canonical_smiles="CCO")]
        toxicity = [activity(activity_id=2, standard_type="CC50", canonical_smiles="CCC",
            molecule_chembl_id="CHEMBL2", standard_value="2000")]
        rows, rejected = endpoint_rows(inhibition, toxicity, {})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(set(rows.endpoint), {"IC50", "CC50"})
        self.assertNotIn("SI", rows)
        self.assertAlmostEqual(rows.loc[rows.endpoint.eq("CC50"), "value_um"].iloc[0], 2)
        bad = activity(activity_id=3, data_validity_comment="Potential author error")
        rows, rejected = endpoint_rows(inhibition+[bad], [], {})
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rejected), 1)

    def test_extra_sources_respect_both_structure_and_scaffold_exclusions(self):
        extra = pd.DataFrame({"standard_smiles": ["c1ccccc1", "Oc1ccccc1", "C1CCCCC1"]})
        self.assertEqual(len(suite.exclude_groups(extra, ["c1ccccc1"], "molecule")), 2)
        result = suite.exclude_groups(extra, ["c1ccccc1"], "scaffold")
        self.assertEqual(result.standard_smiles.tolist(), ["C1CCCCC1"])


class Models(unittest.TestCase):
    def setUp(self):
        self.smiles = ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1", "c1ccsc1", "c1ccoc1"]
        self.frame = pd.DataFrame({"standard_smiles": self.smiles, "source": "kaggle_train",
            "IC50": np.arange(1, 7, dtype=float), "CC50": np.arange(1, 7)*10.,
            "IC50_qualifier": "", "CC50_qualifier": "", "n_measurements": 1})
        self.ref = self.frame.assign(SI=10., target_is_label=False)
        self.extra = pd.DataFrame({"standard_smiles": ["CCCC", "CCCC"], "endpoint": ["IC50", "CC50"],
            "value_um": [2., 20.], "weight": [.5, .25]})
        self.raw = pd.DataFrame({"standard_smiles": self.smiles, "IC50": 1., "CC50": 10.,
            "IC50_qualifier": "", "CC50_qualifier": "", "cell_line_ic50": "Vero", "cell_line_cc50": "Vero",
            "cc50_cell_line_match": "same_cell_line"})

    def test_bounds_reject_ambiguous_aggregates_and_preserve_direction(self):
        frame = self.frame.copy()
        frame["IC50_qualifier"] = ["", ">", "<=", ">;<", ">", "~"]
        frame.loc[4, "n_measurements"] = 2
        lo, hi = suite.endpoint_bounds(frame, "IC50", True)
        np.testing.assert_allclose(lo[:3], [1, 2, 0])
        np.testing.assert_allclose(hi[:3], [1, np.inf, 3])
        self.assertTrue(np.isnan(lo[3:]).all())
        lo, hi = suite.endpoint_bounds(frame, "IC50", False)
        self.assertEqual(np.isfinite(lo).sum(), 1)

    def test_native_baseline_matches_existing_sklearn_model(self):
        params = {"n_estimators": 3, "n_jobs": 1, "min_child_weight": 1}
        matrix = suite.feature_matrix(self.smiles)
        weights = np.array([1, .5, .5, .25, .25, 1.])
        baseline = XGBWeightedTwoHead(params).fit(matrix, self.frame.IC50, self.frame.CC50, sample_weight=weights)
        with patch.dict(suite.PARAMETER_PROFILES, {"default": params}):
            model = suite.HypothesisModel({}).fit(self.frame, matrix, weights, self.ref, self.extra,
                suite.feature_matrix(self.extra.standard_smiles.tolist()), self.raw)
        np.testing.assert_array_equal(model.predict_si(matrix), baseline.predict_si(matrix))

    def test_feature_representations_differ_and_are_finite(self):
        matrices = [suite.feature_matrix(self.smiles, p) for p in ("default", "counts", "radius3", "feature_morgan")]
        for matrix in matrices:
            self.assertEqual(matrix.shape, (6, 2061))
            self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue((matrices[1][:, :2048] > 1).any())
        self.assertFalse(np.array_equal(matrices[0], matrices[3]))

    def test_all_model_families_have_independent_masks_and_portable_native_heads(self):
        small = {name: {"n_estimators": 3, "n_jobs": 1, "min_child_weight": 1} for name in suite.PARAMETER_PROFILES}
        with patch.dict(suite.PARAMETER_PROFILES, small):
            for settings in ({}, {"expanded": True}, {"aft": True, "censored": True},
                             {"direct_alpha": .25}, {"clean": True}, {"ic_params": "shallow", "cc_params": "deeper"}):
                model = suite.HypothesisModel(settings).fit(self.frame, suite.feature_matrix(self.smiles),
                    np.ones(6), self.ref, self.extra, suite.feature_matrix(self.extra.standard_smiles.tolist()), self.raw)
                self.assertEqual(model.audit["IC50"]["n"], 7 if settings.get("expanded") else 6)
                prediction = model.predict_si(suite.feature_matrix(self.smiles))
                self.assertTrue(np.isfinite(prediction).all())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"model.joblib"
                    joblib.dump(model, path)
                    restored = joblib.load(path)
                    np.testing.assert_array_equal(prediction, restored.predict_si(suite.feature_matrix(self.smiles)))

    def test_direct_si_fit_excludes_validation_and_conditional_labels(self):
        ref = pd.concat([self.ref, self.ref.iloc[[0]].assign(target_is_label=True, SI=10000)], ignore_index=True)
        small = {"default": {"n_estimators": 3, "n_jobs": 1, "min_child_weight": 1}}
        with patch.dict(suite.PARAMETER_PROFILES, small), patch.dict(suite.DEFAULT_PARAMS, {"n_estimators": 3, "n_jobs": 1}):
            model = suite.HypothesisModel({"direct_alpha": .25}).fit(self.frame.iloc[:4],
                suite.feature_matrix(self.smiles[:4]), np.ones(4), ref, self.extra,
                suite.feature_matrix(self.extra.standard_smiles.tolist()), self.raw)
        self.assertEqual(model.audit["SI"]["measured_competition_rows"], 4)

    def test_strict_targets_prefer_original_endpoints_and_do_not_keep_bad_public_pairs(self):
        frame = self.frame.copy()
        frame.loc[0, "source"] = "chembl"
        raw = self.raw.copy()
        raw.loc[0, "cell_line_cc50"] = "HEL"
        result = suite.strict_targets(frame, raw, self.ref)
        self.assertTrue(np.isnan(result["CC50"][0]))
        self.assertEqual(result["IC50"][1], self.ref.IC50.iloc[1])

    def test_selection_is_based_on_oof_and_flags_ordinary_value_damage(self):
        base = {"mse": 100., "measured": {"mse": 100.}, "low_si": {"mse": 100.}}
        good = {"mse": 90., "measured": {"mse": 90.}, "low_si": {"mse": 110.}}
        protocols = {p: {"baseline": copy.deepcopy(base), "new": copy.deepcopy(good)} for p in ("molecule", "scaffold")}
        self.assertTrue(suite.choose_candidate(protocols, ["new"])["oof_safe"])
        protocols["scaffold"]["new"]["low_si"]["mse"] = 121.
        self.assertFalse(suite.choose_candidate(protocols, ["new"])["oof_safe"])
