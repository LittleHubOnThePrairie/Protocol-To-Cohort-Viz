"""Tests for CtNormalizer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.sdtm.ct_normalizer import CtNormalizer


@pytest.fixture()
def ct():
    return CtNormalizer()


class TestPhaseNormalization:
    def test_phase_iii_exact(self, ct):
        result = ct.normalize("PHASE", "Phase III")
        assert result.mapped is True
        assert result.tsvalcd == "PHASE3"
        assert result.nci_code == "C15962"

    def test_phase_iii_uppercase(self, ct):
        result = ct.normalize("PHASE", "PHASE III")
        assert result.mapped is True
        assert result.tsvalcd == "PHASE3"

    def test_phase_i(self, ct):
        result = ct.normalize("PHASE", "Phase I")
        assert result.mapped is True
        assert result.tsvalcd == "PHASE1"

    def test_phase_ii_iii(self, ct):
        result = ct.normalize("PHASE", "Phase II/III")
        assert result.mapped is True
        assert result.tsvalcd == "PHASE2-3"

    def test_phase_iv(self, ct):
        result = ct.normalize("PHASE", "Phase IV")
        assert result.mapped is True
        assert result.tsvalcd == "PHASE4"

    def test_unmapped_phase_returns_false(self, ct):
        result = ct.normalize("PHASE", "Phase X")
        assert result.mapped is False
        assert result.tsvalcd == ""
        assert result.nci_code == ""
        assert result.original_value == "Phase X"


class TestBlindingNormalization:
    def test_double_blind(self, ct):
        result = ct.normalize("BLIND", "Double-Blind")
        assert result.mapped is True
        assert result.tsvalcd == "DOUBLE-BLIND"

    def test_open_label(self, ct):
        result = ct.normalize("BLIND", "Open-Label")
        assert result.mapped is True
        assert result.tsvalcd == "OPEN-LABEL"

    def test_open_label_space(self, ct):
        result = ct.normalize("BLIND", "Open Label")
        assert result.mapped is True
        assert result.tsvalcd == "OPEN-LABEL"


class TestDesignNormalization:
    def test_parallel(self, ct):
        result = ct.normalize("STYPE", "Parallel")
        assert result.mapped is True
        assert result.tsvalcd == "PARALLEL"

    def test_crossover(self, ct):
        result = ct.normalize("STYPE", "Crossover")
        assert result.mapped is True
        assert result.tsvalcd == "CROSSOVER"


class TestTrialType:
    def test_interventional(self, ct):
        result = ct.normalize("TTYPE", "INTERVENTIONAL")
        assert result.mapped is True
        assert result.tsvalcd == "INTERVENTIONAL"

    def test_unknown_parmcd_not_mapped(self, ct):
        # Parmcd not in table → always unmapped
        result = ct.normalize("UNKNOWN_PARM", "some value")
        assert result.mapped is False


class TestNormalizePhaseFreeText:
    def test_phase_in_text(self, ct):
        result = ct.normalize_phase(
            "A Phase III, double-blind study of Drug X in hypertension"
        )
        assert result.mapped is True
        assert result.tsvalcd == "PHASE3"

    def test_phase_ii_in_text(self, ct):
        result = ct.normalize_phase("Phase II/III adaptive trial")
        assert result.mapped is True

    def test_no_phase_in_text(self, ct):
        result = ct.normalize_phase("Randomised controlled trial with no phase mention")
        assert result.mapped is False
        assert result.tsval == ""

    def test_placeholder_prefix(self):
        ct_prefix = CtNormalizer(placeholder_prefix="REVIEW: ")
        result = ct_prefix.normalize("PHASE", "Phase X")
        assert result.tsval.startswith("REVIEW: ")
