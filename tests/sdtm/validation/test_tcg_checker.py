"""Tests for TcgChecker (PTCV-23) — FDA TCG v5.9 Appendix B."""

from __future__ import annotations

import pandas as pd
import pytest

from ptcv.sdtm.validation.models import TcgParameter
from ptcv.sdtm.validation.tcg_checker import TcgChecker, _TCG_REQUIRED


class TestTcgCheckerAllMissing:
    """When TS has no TSPARMCD column, all parameters are missing."""

    def test_empty_df_all_missing(self):
        checker = TcgChecker(pd.DataFrame())
        params, passed, missing = checker.check()
        assert passed is False
        assert len(missing) == len(_TCG_REQUIRED)
        assert len(params) == len(_TCG_REQUIRED)

    def test_all_params_are_tcgparameter(self):
        checker = TcgChecker(pd.DataFrame())
        params, _, _ = checker.check()
        assert all(isinstance(p, TcgParameter) for p in params)

    def test_all_params_required_by_tcg(self):
        checker = TcgChecker(pd.DataFrame())
        params, _, _ = checker.check()
        assert all(p.required_by == "FDA TCG v5.9 Appendix B" for p in params)


class TestTcgCheckerPartialPresent:
    """When TS has some required parameters."""

    @pytest.fixture()
    def ts_with_some_params(self) -> pd.DataFrame:
        """TS with TITLE and PHASE present (only 2 of 22 required)."""
        return pd.DataFrame([
            {"STUDYID": "NCT001", "DOMAIN": "TS", "TSSEQ": 1.0,
             "TSPARMCD": "TITLE", "TSPARM": "Trial Title",
             "TSVAL": "Phase III", "TSVALCD": "", "TSVALNF": "", "TSVALUNIT": ""},
            {"STUDYID": "NCT001", "DOMAIN": "TS", "TSSEQ": 2.0,
             "TSPARMCD": "PHASE", "TSPARM": "Trial Phase",
             "TSVAL": "PHASE III", "TSVALCD": "C49688", "TSVALNF": "", "TSVALUNIT": ""},
        ])

    def test_passed_is_false_when_missing(self, ts_with_some_params):
        checker = TcgChecker(ts_with_some_params)
        _, passed, _ = checker.check()
        assert passed is False

    def test_present_params_not_in_missing(self, ts_with_some_params):
        checker = TcgChecker(ts_with_some_params)
        _, _, missing = checker.check()
        assert "TITLE" not in missing
        assert "PHASE" not in missing

    def test_missing_params_count_correct(self, ts_with_some_params):
        checker = TcgChecker(ts_with_some_params)
        _, _, missing = checker.check()
        # 22 required total, 2 present → 20 missing
        assert len(missing) == len(_TCG_REQUIRED) - 2

    def test_present_flag_on_param_record(self, ts_with_some_params):
        checker = TcgChecker(ts_with_some_params)
        params, _, _ = checker.check()
        title_param = next(p for p in params if p.tsparmcd == "TITLE")
        assert title_param.present is True
        assert title_param.missing is False


class TestTcgCheckerAllPresent:
    """When TS has all required parameters, passed=True."""

    @pytest.fixture()
    def ts_all_required(self) -> pd.DataFrame:
        rows = []
        for i, (parmcd, parm) in enumerate(_TCG_REQUIRED, start=1):
            rows.append({
                "STUDYID": "NCT001",
                "DOMAIN": "TS",
                "TSSEQ": float(i),
                "TSPARMCD": parmcd,
                "TSPARM": parm,
                "TSVAL": "Test Value",
                "TSVALCD": "",
                "TSVALNF": "",
                "TSVALUNIT": "",
            })
        return pd.DataFrame(rows)

    def test_passed_true_when_all_present(self, ts_all_required):
        checker = TcgChecker(ts_all_required)
        _, passed, missing = checker.check()
        assert passed is True
        assert missing == []

    def test_all_params_present_flag_true(self, ts_all_required):
        checker = TcgChecker(ts_all_required)
        params, _, _ = checker.check()
        assert all(p.present for p in params)
        assert all(not p.missing for p in params)


class TestTcgCheckerGherkin:
    """PTCV-23 Scenario: FDA TCG Appendix B completeness check."""

    def test_tcg_completeness_json_structure(self, ts_df):
        """Verify the shape of check() output matches tcg_completeness.json."""
        checker = TcgChecker(ts_df)
        params, passed, missing = checker.check()

        # Output must have a passed boolean
        assert isinstance(passed, bool)

        # Output must have a missing_params list
        assert isinstance(missing, list)
        assert all(isinstance(m, str) for m in missing)

        # Each param record must match TcgParameter fields
        for p in params:
            assert hasattr(p, "tsparmcd")
            assert hasattr(p, "tsparm")
            assert hasattr(p, "required_by")
            assert hasattr(p, "present")
            assert hasattr(p, "missing")
            # present and missing must be complements
            assert p.present == (not p.missing)

    def test_missing_params_are_tsparmcd_codes(self, ts_df):
        """Missing entries are TSPARMCD strings (not full names)."""
        checker = TcgChecker(ts_df)
        _, _, missing = checker.check()
        required_codes = {parmcd for parmcd, _ in _TCG_REQUIRED}
        for m in missing:
            assert m in required_codes, f"Unexpected missing code: {m!r}"
