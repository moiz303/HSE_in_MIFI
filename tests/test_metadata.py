"""Тесты нормализации метаданных. Текст в источниках грязный, и порядок
шаблонов критичен: BSC-40 не должен попасть в BSC-1, а HeLa в HEL.
"""

import unittest

import numpy as np
import pandas as pd

from src.features import metadata as m


class CellLines(unittest.TestCase):
    def test_messy_spellings_map_to_one_token(self):
        for text in ("Vero", "culture Vero", "Vaccinia virus, Vero cells", "VERO"):
            self.assertEqual(m.normalize_cell_line(text), "Vero", text)

    def test_hela_is_not_confused_with_hel(self):
        self.assertEqual(m.normalize_cell_line("HELA cells"), "HeLa")
        self.assertEqual(m.normalize_cell_line("HeLa cells"), "HeLa")
        self.assertEqual(m.normalize_cell_line("cells HEL"), "HEL")

    def test_bsc40_is_not_confused_with_bsc1(self):
        self.assertEqual(m.normalize_cell_line("BSC-40 "), "BSC-40")
        self.assertEqual(m.normalize_cell_line("BSC40"), "BSC-40")
        self.assertEqual(m.normalize_cell_line("cells BSC1"), "BSC-1")

    def test_human_foreskin_fibroblast_is_hff(self):
        self.assertEqual(m.normalize_cell_line("Human foreskin fibroblast"), "HFF")
        self.assertEqual(m.normalize_cell_line("Vaccinia virus, HFF cells"), "HFF")

    def test_unknown_when_nothing_matches(self):
        self.assertEqual(m.normalize_cell_line("не указано"), m.UNKNOWN)
        self.assertEqual(m.normalize_cell_line(None), m.UNKNOWN)

    def test_falls_back_to_the_next_field(self):
        # Культура часто упомянута только внутри текста штамма.
        self.assertEqual(
            m.normalize_cell_line("", "Vaccinia virus (strain Copenhagen). culture Vero"),
            "Vero",
        )


class Strains(unittest.TestCase):
    def test_known_strains(self):
        self.assertEqual(m.normalize_strain("Vaccinia virus (strain Copenhagen)"), "Copenhagen")
        self.assertEqual(m.normalize_strain("Vaccinia virus Western Reserve"), "WR")
        self.assertEqual(m.normalize_strain("VV Lister"), "Lister")
        self.assertEqual(m.normalize_strain("Vaccinia virus Tiantan strain"), "Tiantan")

    def test_plain_vaccinia_has_no_strain(self):
        self.assertEqual(m.normalize_strain("Vaccinia virus"), m.UNKNOWN)


class Flags(unittest.TestCase):
    def test_panel_papers_are_flagged(self):
        flags = m.panel_flags("Vaccinia virus, cowpox virus (strain_Grishak), ectromelia virus")
        self.assertEqual(flags["cowpox"], 1)
        self.assertEqual(flags["ectromelia"], 1)
        self.assertEqual(flags["multi_virus"], 1)

    def test_single_virus_is_not_flagged(self):
        flags = m.panel_flags("Vaccinia virus", "Vero")
        self.assertEqual(flags["cowpox"], 0)
        self.assertEqual(flags["ectromelia"], 0)


class Matrix(unittest.TestCase):
    FRAME = pd.DataFrame({
        "Culture": ["Vero", "cells HEL", "Human foreskin fibroblast"],
        "strain": ["Vaccinia virus (strain Copenhagen). culture Vero",
                   "Vaccinia virus", "VV Lister, cowpox virus"],
    })

    def test_width_matches_the_declared_one(self):
        for cells in (True, False):
            for strain in (True, False):
                for flags in (True, False):
                    if not (cells or strain or flags):
                        continue
                    matrix = m.metadata_matrix(
                        self.FRAME, use_cells=cells, use_strain=strain, use_flags=flags
                    )
                    self.assertEqual(matrix.shape[1], m.width(cells, strain, flags))

    def test_one_hot_has_exactly_one_cell_line_per_row(self):
        matrix = m.metadata_matrix(self.FRAME, use_strain=False, use_flags=False)
        np.testing.assert_array_equal(matrix.sum(axis=1), np.ones(len(self.FRAME)))

    def test_matrix_is_finite_and_numeric(self):
        matrix = m.metadata_matrix(self.FRAME)
        self.assertTrue(np.isfinite(matrix).all())

    def test_missing_columns_do_not_crash(self):
        matrix = m.metadata_matrix(pd.DataFrame({"smiles": ["CCO", "CCC"]}))
        self.assertEqual(matrix.shape[0], 2)
