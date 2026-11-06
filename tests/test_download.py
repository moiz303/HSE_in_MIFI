"""Тесты выгрузки ChEMBL. Стадия 1, владелец A.

Сети нет: фикстуры — минимальные записи той же формы, что отдаёт API. Если CI
начнёт ходить в EBI, билд будет краснеть при каждой недоступности сервиса.

Фикстуры заданы прямо в тесте, а не отдельными json: так видно, какое именно
поле проверяется, и не нужно держать в репозитории срезы чужих ответов.
"""

import unittest

from src.data import download, schema

# Запись активности ChEMBL в том виде, в котором её отдаёт activity.json.
# canonical_smiles приходит прямо в записи активности, отдельный запрос
# молекул не нужен.
def activity(**overrides):
    record = {
        "molecule_chembl_id": "CHEMBL1",
        "document_chembl_id": "CHEMBLDOC",
        "assay_chembl_id": "CHEMBLASSAY",
        "standard_type": "EC50",
        "standard_value": "1000.0",
        "standard_units": "nM",
        "standard_relation": "=",
        "canonical_smiles": "CCO",
        "assay_description": "Antiviral activity against Vaccinia virus in Vero cells",
        "target_pref_name": "Vaccinia virus",
        "data_validity_comment": None,
    }
    record.update(overrides)
    return record


class Measurement(unittest.TestCase):
    def test_nanomolar_becomes_micromolar(self):
        record, reason = download._measurement(activity(), {})
        self.assertEqual(reason, "")
        self.assertAlmostEqual(record["value_um"], 1.0)

    def test_mass_units_use_molecular_weight(self):
        # 10 µg/mL при массе этанола 46.07 г/моль → ~217 µM
        record, reason = download._measurement(
            activity(standard_value="10", standard_units="ug.mL-1"), {}
        )
        self.assertEqual(reason, "")
        self.assertAlmostEqual(record["value_um"], 10 / 46.069 * 1e3, places=1)

    def test_mass_units_written_with_a_space(self):
        # ChEMBL пишет и "ug.mL-1", и "ug mL-1" — обе формы должны работать,
        # иначе молча теряются десятки строк.
        spaced, reason = download._measurement(
            activity(standard_value="10", standard_units="ug mL-1"), {}
        )
        self.assertEqual(reason, "")
        dotted, _ = download._measurement(
            activity(standard_value="10", standard_units="ug.mL-1"), {}
        )
        self.assertAlmostEqual(spaced["value_um"], dotted["value_um"])

    def test_missing_units_are_rejected_with_a_reason(self):
        record, reason = download._measurement(activity(standard_units=None), {})
        self.assertIsNone(record)
        self.assertIn("единиц", reason)

    def test_zero_value_is_rejected(self):
        record, reason = download._measurement(activity(standard_value="0.0"), {})
        self.assertIsNone(record)
        self.assertIn("положительное", reason)

    def test_value_outside_plausible_range_is_rejected(self):
        record, reason = download._measurement(activity(standard_value="1e15"), {})
        self.assertIsNone(record)
        self.assertIn("диапазон", reason)

    def test_censoring_sign_is_preserved(self):
        record, _ = download._measurement(activity(standard_relation=">"), {})
        self.assertEqual(record["qualifier"], ">")

    def test_exact_measurement_has_empty_qualifier(self):
        record, _ = download._measurement(activity(standard_relation="="), {})
        self.assertEqual(record["qualifier"], "")

    def test_virus_target_is_not_used_as_a_cell_line(self):
        # У антивирусной записи цель — вирус. Подставлять её как линию клеток
        # нельзя: тогда пара "та же линия" собирается по несуществующему признаку.
        record, _ = download._measurement(
            activity(assay_description="Antiviral activity against Vaccinia virus"), {}
        )
        self.assertEqual(record["cell_line"], "")

    def test_admet_is_not_a_cell_line(self):
        record, _ = download._measurement(
            activity(
                standard_type="CC50",
                target_pref_name="ADMET",
                assay_description="Cytotoxicity measured by MTT",
            ),
            {},
        )
        self.assertEqual(record["cell_line"], "")

    def test_toxicity_falls_back_to_target_as_cell_line(self):
        record, _ = download._measurement(
            activity(
                standard_type="CC50",
                target_pref_name="Vero 76",
                assay_description="Cytotoxicity by neutral red uptake",
            ),
            {},
        )
        self.assertEqual(record["cell_line"], "Vero 76")

    def test_assay_record_wins_over_description(self):
        record, _ = download._measurement(activity(), {"CHEMBLASSAY": "BSC-40"})
        self.assertEqual(record["cell_line"], "BSC-40")


class CellLineFromText(unittest.TestCase):
    def test_hela_is_not_confused_with_hel(self):
        self.assertEqual(download.cell_line_from_text("in HeLa cells"), "HeLa")
        self.assertEqual(download.cell_line_from_text("in HEL cells"), "HEL")

    def test_common_lines(self):
        self.assertEqual(download.cell_line_from_text("BHK-21 monolayer"), "BHK-21")
        self.assertEqual(download.cell_line_from_text("cat CRFK cells"), "CRFK")
        self.assertEqual(download.cell_line_from_text("african green monkey Vero"), "Vero")

    def test_no_cell_line_mentioned(self):
        self.assertEqual(download.cell_line_from_text("Inhibitory activity against VV"), "")
        self.assertEqual(download.cell_line_from_text(None), "")


class Representative(unittest.TestCase):
    def record(self, value, qualifier=""):
        return {
            "value_um": value,
            "qualifier": qualifier,
            "cell_line": "Vero",
            "measured_type": "EC50",
            "validity": "",
            "smiles": "CCO",
            "assay_description": "",
            "mentions_vaccinia": True,
        }

    def test_exact_measurements_win_over_censored(self):
        result = download._representative(
            [self.record(5.0), self.record(500.0, ">"), self.record(900.0, ">")]
        )
        self.assertAlmostEqual(result["value_um"], 5.0)
        self.assertEqual(result["qualifier"], "")
        self.assertEqual(result["n_measurements"], 3)
        self.assertEqual(result["n_censored"], 2)

    def test_median_is_taken_in_log_space(self):
        result = download._representative(
            [self.record(1.0), self.record(10.0), self.record(1000.0)]
        )
        self.assertAlmostEqual(result["value_um"], 10.0)

    def test_all_censored_keeps_the_sign(self):
        result = download._representative([self.record(100.0, ">"), self.record(100.0, ">")])
        self.assertEqual(result["qualifier"], ">")


class PickCytotoxicity(unittest.TestCase):
    def record(self, value, cell_line, vaccinia):
        return {
            "value_um": value,
            "qualifier": "",
            "cell_line": cell_line,
            "measured_type": "CC50",
            "validity": "",
            "smiles": "CCO",
            "assay_description": "",
            "mentions_vaccinia": vaccinia,
        }

    def test_vaccinia_arm_in_the_same_cell_line_wins(self):
        candidates = [
            self.record(10.0, "MDCK", False),
            self.record(20.0, "HEL", True),
            self.record(30.0, "HEL", False),
        ]
        chosen, rule = download._pick_cytotoxicity(candidates, {"HEL"})
        self.assertEqual(rule, "vaccinia_same_cell_line")
        self.assertAlmostEqual(chosen["value_um"], 20.0)

    def test_vaccinia_arm_in_another_line_beats_a_different_virus(self):
        candidates = [self.record(10.0, "MDCK", True), self.record(30.0, "HEL", False)]
        chosen, rule = download._pick_cytotoxicity(candidates, {"Vero"})
        self.assertEqual(rule, "vaccinia_other_cell_line")
        self.assertAlmostEqual(chosen["value_um"], 10.0)

    def test_same_cell_line_when_no_arm_mentions_vaccinia(self):
        candidates = [self.record(10.0, "MDCK", False), self.record(30.0, "Vero", False)]
        chosen, rule = download._pick_cytotoxicity(candidates, {"Vero"})
        self.assertEqual(rule, "same_cell_line")
        self.assertAlmostEqual(chosen["value_um"], 30.0)

    def test_unknown_cell_line_falls_back_to_the_document(self):
        candidates = [self.record(10.0, "MDCK", False)]
        chosen, rule = download._pick_cytotoxicity(candidates, set())
        self.assertEqual(rule, "any_in_document")
        self.assertAlmostEqual(chosen["value_um"], 10.0)


class BuildTable(unittest.TestCase):
    def test_pair_is_assembled_per_molecule_and_document(self):
        antiviral = [activity(standard_value="1000.0")]
        cytotoxic = [activity(standard_type="CC50", standard_value="50000.0",
                              target_pref_name="Vero")]
        table, excluded = download.build_table(
            antiviral, cytotoxic, {}, {"CHEMBLDOC": "10.1/abc"}
        )
        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        self.assertAlmostEqual(row["IC50"], 1.0)
        self.assertAlmostEqual(row["CC50"], 50.0)
        self.assertAlmostEqual(row["SI"], 50.0)
        self.assertEqual(row["doi"], "10.1/abc")
        self.assertEqual(row["source"], "chembl")
        self.assertEqual(row["assay_type"], "EC50")
        self.assertTrue(excluded.empty)

    def test_activity_without_a_pair_goes_to_the_journal(self):
        table, excluded = download.build_table([activity()], [], {}, {})
        self.assertTrue(table.empty)
        self.assertEqual(len(excluded), 1)
        self.assertIn("парной цитотоксичности", excluded.iloc[0]["reason"])

    def test_different_documents_do_not_pair(self):
        antiviral = [activity(document_chembl_id="DOC1")]
        cytotoxic = [activity(standard_type="CC50", document_chembl_id="DOC2")]
        table, _ = download.build_table(antiviral, cytotoxic, {}, {})
        self.assertTrue(table.empty)

    def test_si_artifact_is_flagged(self):
        # Неактивное соединение: и активность, и токсичность упёрлись в одну
        # максимальную концентрацию, поэтому SI = 1 — артефакт протокола.
        antiviral = [activity(standard_relation=">", standard_value="82000.0")]
        cytotoxic = [activity(standard_type="CC50", standard_relation=">",
                              standard_value="82000.0", target_pref_name="Vero")]
        table, _ = download.build_table(antiviral, cytotoxic, {}, {})
        row = table.iloc[0]
        self.assertAlmostEqual(row["SI"], 1.0)
        self.assertTrue(bool(row["si_artifact"]))

    def test_honest_pair_is_not_flagged_as_artifact(self):
        antiviral = [activity(standard_value="1000.0")]
        cytotoxic = [activity(standard_type="CC50", standard_relation=">",
                              standard_value="50000.0", target_pref_name="Vero")]
        table, _ = download.build_table(antiviral, cytotoxic, {}, {})
        self.assertFalse(bool(table.iloc[0]["si_artifact"]))

    def test_chembl_flagged_values_are_dropped_with_a_reason(self):
        antiviral = [activity(data_validity_comment="Outside typical range")]
        cytotoxic = [activity(standard_type="CC50", target_pref_name="Vero")]
        table, excluded = download.build_table(antiviral, cytotoxic, {}, {})
        self.assertTrue(table.empty)
        self.assertTrue(any("недостоверн" in r for r in excluded["reason"]))

    def test_flagged_values_are_kept_when_asked(self):
        antiviral = [activity(data_validity_comment="Outside typical range")]
        cytotoxic = [activity(standard_type="CC50", standard_value="50000.0",
                              target_pref_name="Vero")]
        table, _ = download.build_table(antiviral, cytotoxic, {}, {}, drop_invalid=False)
        self.assertEqual(len(table), 1)

    def test_si_matches_the_contract_formula(self):
        antiviral = [activity(standard_value="2000.0")]
        cytotoxic = [activity(standard_type="CC50", standard_value="40000.0",
                              target_pref_name="Vero")]
        table, _ = download.build_table(antiviral, cytotoxic, {}, {})
        row = table.iloc[0]
        self.assertAlmostEqual(row["SI"], schema.selectivity_index(row["IC50"], row["CC50"]))


if __name__ == "__main__":
    unittest.main()
