import unittest
import numpy as np
import pandas as pd
from baseline import run


class SumModel:
    """Small deterministic predictor for exercising the inference boundary."""

    def predict(self, features):
        return np.asarray(features, dtype=float).sum(axis=1)


def small_tables():
    train = pd.DataFrame(
        {
            "SMILES": ["CCO", "c1ccccc1"],
            "SI VACV": [0.0, 2.5],
            "CC50_uM": [None, 10.0],
            "IC50_uM_VACV": [None, 4.0],
        }
    )
    test = pd.DataFrame(
        {
            "index": ["7", "3"],
            "SMILES": ["CCO", "c1ccccc1"],
            "CC50_uM": [None, 12.0],
            "IC50_uM_VACV": [5.0, None],
        }
    )
    template = pd.DataFrame({"index": ["7", "3"], "SI VACV": [123.0, 456.0]})
    return train, test, template


class DirectCatBoostBaselineTests(unittest.TestCase):
    def test_features_and_predictions_ignore_forbidden_test_endpoints(self):
        _, test, _ = small_tables()
        changed = test.copy(deep=True)
        changed["CC50_uM"] = [999999.0, -123.0]
        changed["IC50_uM_VACV"] = [-900.0, 0.0]
        original_features = run.features_from_frame(test)
        changed_features = run.features_from_frame(changed)
        np.testing.assert_array_equal(original_features, changed_features)
        model = SumModel()
        np.testing.assert_array_equal(
            run.predict_frame(model, test),
            run.predict_frame(model, changed),
        )

    def test_missing_forbidden_endpoint_columns_are_not_required(self):
        train, test, template = small_tables()
        test = test.drop(columns=["CC50_uM", "IC50_uM_VACV"])
        targets = run.validate_tables(train, test, template)
        self.assertEqual(targets.tolist(), [0.0, 2.5])

    def test_submission_has_exact_schema_order_and_preserves_template_order(self):
        _, _, template = small_tables()
        submission = run.build_submission(template, np.array([8.5, 1.25]))
        self.assertEqual(submission.columns.tolist(), ["index", "SI VACV"])
        self.assertEqual(submission["index"].tolist(), ["7", "3"])
        self.assertEqual(submission["SI VACV"].tolist(), [8.5, 1.25])
        self.assertEqual(template["SI VACV"].tolist(), [123.0, 456.0])

    def test_missing_required_header_has_actionable_error(self):
        train, test, template = small_tables()
        with self.assertRaisesRegex(ValueError, "missing required column.*SMILES"):
            run.validate_tables(
                train.drop(columns=["SMILES"]),
                test,
                template,
            )

    def test_unparseable_smiles_is_rejected(self):
        train, test, template = small_tables()
        test.loc[1, "SMILES"] = "not a SMILES"
        with self.assertRaisesRegex(ValueError, "unparseable SMILES"):
            run.validate_tables(train, test, template)

    def test_non_numeric_target_is_rejected_but_zero_is_kept(self):
        train, test, template = small_tables()
        targets = run.validate_tables(train, test, template)
        self.assertEqual(targets.tolist(), [0.0, 2.5])
        train["SI VACV"] = train["SI VACV"].astype(object)
        train.loc[1, "SI VACV"] = "not numeric"
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            run.validate_tables(train, test, template)

    def test_template_ids_must_match_test_ids_in_order(self):
        train, test, template = small_tables()
        template.loc[0, "index"] = "wrong-id"
        with self.assertRaisesRegex(ValueError, "IDs do not match in row order"):
            run.validate_tables(train, test, template)

    def test_sample_target_values_are_replaced_without_being_used(self):
        _, _, template = small_tables()
        changed_template = template.copy(deep=True)
        changed_template["SI VACV"] = ["untrusted", None]
        first = run.build_submission(template, np.array([8.5, 1.25]))
        second = run.build_submission(changed_template, np.array([8.5, 1.25]))
        pd.testing.assert_frame_equal(first, second)

    def test_expected_row_count_errors_name_the_required_size(self):
        train, test, template = small_tables()
        with self.assertRaisesRegex(ValueError, "expected 320"):
            run.validate_tables(
                train,
                test,
                template,
                expected_train_rows=320,
            )


if __name__ == "__main__":
    unittest.main()
