"""Tests for GX checkpoint runner (PTCV-120).

Feature: GX Checkpoint Runner

  Scenario: Valid VS DataFrame passes all expectations
    Given a VS DataFrame with correct columns, no nulls in required
        fields, and valid codelist values
    When validate_domain("VS", df) is called
    Then success is True
    And failed is 0

  Scenario: Missing required column detected
    Given a VS DataFrame missing the VSTESTCD column
    When validate_domain("VS", df) is called
    Then success is False
    And failed_expectations includes expect_column_to_exist
        for VSTESTCD

  Scenario: Out-of-range values detected
    Given a VS DataFrame with VSSTRESN = 999 (above range max)
    When validate_domain("VS", df) is called
    Then failed_expectations includes
        expect_column_values_to_be_between for VSSTRESN

  Scenario: Full validation report across multiple domains
    Given DM and VS DataFrames
    When validate_all is called
    Then the ValidationReport has domain_results for both DM and VS
    And overall_success reflects whether all domains passed
"""

from __future__ import annotations

import pytest
import pandas as pd

from ptcv.expectations.checkpoint_runner import (
    DomainValidationResult,
    FailedExpectation,
    SdtmValidator,
    ValidationReport,
    validate_all,
    validate_domain,
)
from ptcv.expectations.suite_builder import SuiteConfig
from ptcv.mock_data.sdtm_metadata import get_domain_spec


# -------------------------------------------------------------------
# Helpers — build minimal valid DataFrames
# -------------------------------------------------------------------


def _minimal_vs_df() -> pd.DataFrame:
    """Return a minimal valid VS DataFrame."""
    return pd.DataFrame({
        "STUDYID": ["STUDY01"],
        "DOMAIN": ["VS"],
        "USUBJID": ["SUBJ-001"],
        "VSSEQ": [1],
        "VSTESTCD": ["SYSBP"],
        "VSTEST": ["Systolic Blood Pressure"],
        "VSORRES": ["120"],
        "VSORRESU": ["mmHg"],
        "VSSTRESC": ["120"],
        "VSSTRESN": [120.0],
        "VSSTRESU": ["mmHg"],
        "VSSTAT": [None],
        "VISITNUM": [1],
        "VISIT": ["Screening"],
        "VSDTC": ["2024-01-15"],
        "VSDY": [1],
    })


def _minimal_dm_df() -> pd.DataFrame:
    """Return a minimal valid DM DataFrame."""
    return pd.DataFrame({
        "STUDYID": ["STUDY01"],
        "DOMAIN": ["DM"],
        "USUBJID": ["SUBJ-001"],
        "SUBJID": ["001"],
        "RFSTDTC": ["2024-01-15"],
        "RFENDTC": ["2024-06-15"],
        "SITEID": ["SITE-01"],
        "AGE": [45],
        "AGEU": ["YEARS"],
        "SEX": ["M"],
        "RACE": ["WHITE"],
        "ETHNIC": ["NOT HISPANIC OR LATINO"],
        "ARMCD": ["TRT"],
        "ARM": ["Treatment"],
        "COUNTRY": ["USA"],
        "DMDTC": ["2024-01-15"],
        "DMDY": [1],
    })


# -------------------------------------------------------------------
# TestDataModels
# -------------------------------------------------------------------


class TestDataModels:
    """Data model integrity tests."""

    def test_failed_expectation_frozen(self) -> None:
        fe = FailedExpectation(
            column="X", expectation_type="test",
            observed_value=None, details="fail",
        )
        with pytest.raises(AttributeError):
            fe.column = "Y"  # type: ignore[misc]

    def test_domain_validation_result_frozen(self) -> None:
        dvr = DomainValidationResult(
            domain_code="VS", success=True,
            total_expectations=10, passed=10, failed=0,
            failed_expectations=(),
        )
        with pytest.raises(AttributeError):
            dvr.success = False  # type: ignore[misc]

    def test_validation_report_frozen(self) -> None:
        vr = ValidationReport(
            domain_results={}, overall_success=True,
            total_passed=0, total_failed=0,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            vr.overall_success = False  # type: ignore[misc]


# -------------------------------------------------------------------
# TestValidVSDataFrame
# -------------------------------------------------------------------


class TestValidVSDataFrame:
    """Valid VS DataFrame passes all expectations."""

    def test_valid_vs_passes(self) -> None:
        df = _minimal_vs_df()
        result = validate_domain("VS", df)
        assert result.success is True
        assert result.failed == 0

    def test_valid_vs_counts(self) -> None:
        df = _minimal_vs_df()
        result = validate_domain("VS", df)
        assert result.total_expectations > 0
        assert result.passed == result.total_expectations

    def test_domain_code_uppercase(self) -> None:
        df = _minimal_vs_df()
        result = validate_domain("vs", df)
        assert result.domain_code == "VS"


# -------------------------------------------------------------------
# TestMissingColumn
# -------------------------------------------------------------------


class TestMissingColumn:
    """Missing required column detected."""

    def test_missing_vstestcd(self) -> None:
        df = _minimal_vs_df().drop(columns=["VSTESTCD"])
        result = validate_domain("VS", df)
        assert result.success is False
        types = {
            fe.expectation_type
            for fe in result.failed_expectations
            if fe.column == "VSTESTCD"
        }
        assert "expect_column_to_exist" in types

    def test_missing_column_details(self) -> None:
        df = _minimal_vs_df().drop(columns=["VSTESTCD"])
        result = validate_domain("VS", df)
        col_failures = [
            fe for fe in result.failed_expectations
            if (fe.column == "VSTESTCD"
                and fe.expectation_type == "expect_column_to_exist")
        ]
        assert len(col_failures) == 1
        assert "VSTESTCD" in col_failures[0].details


# -------------------------------------------------------------------
# TestOutOfRangeValues
# -------------------------------------------------------------------


class TestOutOfRangeValues:
    """Out-of-range values detected."""

    def test_vsstresn_above_max(self) -> None:
        df = _minimal_vs_df()
        df["VSSTRESN"] = [999.0]
        result = validate_domain("VS", df)
        between_failures = [
            fe for fe in result.failed_expectations
            if (fe.expectation_type
                == "expect_column_values_to_be_between"
                and fe.column == "VSSTRESN")
        ]
        assert len(between_failures) == 1

    def test_dm_age_below_min(self) -> None:
        df = _minimal_dm_df()
        df["AGE"] = [5]
        result = validate_domain("DM", df)
        between_failures = [
            fe for fe in result.failed_expectations
            if (fe.expectation_type
                == "expect_column_values_to_be_between"
                and fe.column == "AGE")
        ]
        assert len(between_failures) == 1

    def test_out_of_range_observed_value(self) -> None:
        df = _minimal_vs_df()
        df["VSSTRESN"] = [999.0]
        result = validate_domain("VS", df)
        between_failures = [
            fe for fe in result.failed_expectations
            if (fe.expectation_type
                == "expect_column_values_to_be_between"
                and fe.column == "VSSTRESN")
        ]
        assert 999.0 in between_failures[0].observed_value


# -------------------------------------------------------------------
# TestInvalidCodelistValues
# -------------------------------------------------------------------


class TestInvalidCodelistValues:
    """Invalid codelist values detected."""

    def test_invalid_vstestcd(self) -> None:
        df = _minimal_vs_df()
        df["VSTESTCD"] = ["INVALID"]
        result = validate_domain("VS", df)
        set_failures = [
            fe for fe in result.failed_expectations
            if (fe.expectation_type
                == "expect_column_values_to_be_in_set"
                and fe.column == "VSTESTCD")
        ]
        assert len(set_failures) == 1

    def test_invalid_sex_codelist(self) -> None:
        df = _minimal_dm_df()
        df["SEX"] = ["X"]
        result = validate_domain("DM", df)
        set_failures = [
            fe for fe in result.failed_expectations
            if (fe.expectation_type
                == "expect_column_values_to_be_in_set"
                and fe.column == "SEX")
        ]
        assert len(set_failures) == 1


# -------------------------------------------------------------------
# TestNullValues
# -------------------------------------------------------------------


class TestNullValues:
    """Null value detection in required fields."""

    def test_null_required_field(self) -> None:
        df = _minimal_vs_df()
        df["USUBJID"] = [None]
        result = validate_domain("VS", df)
        null_failures = [
            fe for fe in result.failed_expectations
            if (fe.expectation_type
                == "expect_column_values_to_not_be_null"
                and fe.column == "USUBJID")
        ]
        assert len(null_failures) == 1

    def test_custom_mostly_threshold(self) -> None:
        df = pd.concat([_minimal_vs_df()] * 10, ignore_index=True)
        df.loc[0, "USUBJID"] = None  # 10% null
        cfg = SuiteConfig(mostly=0.85)
        result = validate_domain("VS", df, config=cfg)
        # 90% non-null > 85% threshold → should pass
        null_failures = [
            fe for fe in result.failed_expectations
            if (fe.expectation_type
                == "expect_column_values_to_not_be_null"
                and fe.column == "USUBJID")
        ]
        assert len(null_failures) == 0


# -------------------------------------------------------------------
# TestValidateAll
# -------------------------------------------------------------------


class TestValidateAll:
    """Full validation report across multiple domains."""

    def test_both_domains_present(self) -> None:
        report = validate_all({
            "VS": _minimal_vs_df(),
            "DM": _minimal_dm_df(),
        })
        assert "VS" in report.domain_results
        assert "DM" in report.domain_results

    def test_overall_success_when_all_pass(self) -> None:
        report = validate_all({
            "VS": _minimal_vs_df(),
            "DM": _minimal_dm_df(),
        })
        assert report.overall_success is True

    def test_overall_failure_when_one_fails(self) -> None:
        bad_vs = _minimal_vs_df().drop(columns=["VSTESTCD"])
        report = validate_all({
            "VS": bad_vs,
            "DM": _minimal_dm_df(),
        })
        assert report.overall_success is False
        assert report.domain_results["VS"].success is False
        assert report.domain_results["DM"].success is True

    def test_total_counts(self) -> None:
        report = validate_all({
            "VS": _minimal_vs_df(),
            "DM": _minimal_dm_df(),
        })
        assert report.total_passed == (
            report.domain_results["VS"].passed
            + report.domain_results["DM"].passed
        )
        assert report.total_failed == 0

    def test_timestamp_present(self) -> None:
        report = validate_all({"VS": _minimal_vs_df()})
        assert report.timestamp is not None
        assert "T" in report.timestamp  # ISO 8601


# -------------------------------------------------------------------
# TestSdtmValidator
# -------------------------------------------------------------------


class TestSdtmValidator:
    """GX-powered SdtmValidator tests."""

    def test_raises_import_error_without_gx(self) -> None:
        with pytest.raises(ImportError, match="great_expectations"):
            SdtmValidator(suites={})


# -------------------------------------------------------------------
# TestEdgeCases
# -------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_dataframe(self) -> None:
        spec = get_domain_spec("VS")
        columns = [v.name for v in spec.variables]
        df = pd.DataFrame(columns=columns)
        result = validate_domain("VS", df)
        # Columns exist, no data to violate constraints
        assert result.success is True

    def test_validate_all_empty_dict(self) -> None:
        report = validate_all({})
        assert report.overall_success is True
        assert report.total_passed == 0
        assert report.total_failed == 0
