"""Tests for mock data panel component — PTCV-122.

Tests the data models, CSV ZIP export, and render functions
for the Streamlit mock data panel. Streamlit rendering calls
are not tested directly (would require ``streamlit.testing``),
but we verify the data layer and helper functions.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ui.components.mock_data_panel import (
    DomainValidationResult,
    MockPipelineResult,
    _build_csv_zip,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_df(domain: str, n_rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "STUDYID": ["MOCK-001"] * n_rows,
        "DOMAIN": [domain] * n_rows,
        "USUBJID": [f"MOCK-001-{i:04d}" for i in range(n_rows)],
    })


def _make_result(
    domains: dict[str, int] | None = None,
    validation: dict[str, bool] | None = None,
) -> MockPipelineResult:
    """Build a MockPipelineResult for testing.

    Args:
        domains: Mapping of domain code to row count.
        validation: Mapping of domain code to passed flag.
    """
    domains = domains or {"VS": 10, "LB": 15}
    dfs = {code: _make_df(code, n) for code, n in domains.items()}

    val_results: dict[str, DomainValidationResult] = {}
    if validation:
        for code, passed in validation.items():
            failed = []
            if not passed:
                failed = [{
                    "column": "VSTESTCD",
                    "expectation": "values in set",
                    "observed": "INVALID",
                }]
            val_results[code] = DomainValidationResult(
                domain_code=code,
                passed=passed,
                total_expectations=5,
                failed_expectations=failed,
            )

    return MockPipelineResult(
        domain_dataframes=dfs,
        validation_results=val_results,
        run_id="test-run-001",
        num_subjects=10,
        synthesizer_type="gaussian_copula",
    )


# -----------------------------------------------------------------------
# Scenario: Mock data panel renders after SoA extraction
# -----------------------------------------------------------------------


class TestMockPipelineResult:
    """MockPipelineResult data model."""

    def test_domain_dataframes_accessible(self) -> None:
        result = _make_result()
        assert "VS" in result.domain_dataframes
        assert "LB" in result.domain_dataframes

    def test_dataframe_has_rows(self) -> None:
        result = _make_result({"VS": 10})
        assert len(result.domain_dataframes["VS"]) == 10

    def test_run_id_preserved(self) -> None:
        result = _make_result()
        assert result.run_id == "test-run-001"

    def test_num_subjects(self) -> None:
        result = _make_result()
        assert result.num_subjects == 10

    def test_synthesizer_type(self) -> None:
        result = _make_result()
        assert result.synthesizer_type == "gaussian_copula"


# -----------------------------------------------------------------------
# Scenario: Validation results displayed per domain
# -----------------------------------------------------------------------


class TestDomainValidationResult:
    """DomainValidationResult data model."""

    def test_passed_result(self) -> None:
        val = DomainValidationResult(
            domain_code="VS",
            passed=True,
            total_expectations=5,
            failed_expectations=[],
        )
        assert val.passed is True
        assert val.total_expectations == 5
        assert len(val.failed_expectations) == 0

    def test_failed_result(self) -> None:
        val = DomainValidationResult(
            domain_code="LB",
            passed=False,
            total_expectations=5,
            failed_expectations=[{
                "column": "LBTESTCD",
                "expectation": "values in set",
                "observed": "INVALID_CODE",
            }],
        )
        assert val.passed is False
        assert len(val.failed_expectations) == 1
        assert val.failed_expectations[0]["column"] == "LBTESTCD"

    def test_validation_in_pipeline_result(self) -> None:
        result = _make_result(
            domains={"VS": 5, "LB": 5},
            validation={"VS": True, "LB": False},
        )
        assert result.validation_results["VS"].passed is True
        assert result.validation_results["LB"].passed is False

    def test_failed_expectation_has_observed_value(self) -> None:
        result = _make_result(
            domains={"VS": 5},
            validation={"VS": False},
        )
        fail = result.validation_results["VS"].failed_expectations[0]
        assert "observed" in fail
        assert fail["observed"] == "INVALID"


# -----------------------------------------------------------------------
# Scenario: CSV download available
# -----------------------------------------------------------------------


class TestCsvZipExport:
    """CSV ZIP export for download button."""

    def test_builds_valid_zip(self) -> None:
        dfs = {
            "DM": _make_df("DM", 3),
            "VS": _make_df("VS", 5),
            "LB": _make_df("LB", 4),
        }
        zip_bytes = _build_csv_zip(dfs)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
            assert names == {"DM.csv", "LB.csv", "VS.csv"}

    def test_csv_contains_data(self) -> None:
        dfs = {"VS": _make_df("VS", 5)}
        zip_bytes = _build_csv_zip(dfs)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_content = zf.read("VS.csv").decode("utf-8")
            assert "STUDYID" in csv_content
            assert "MOCK-001" in csv_content

    def test_csv_row_count(self) -> None:
        dfs = {"VS": _make_df("VS", 10)}
        zip_bytes = _build_csv_zip(dfs)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_content = zf.read("VS.csv").decode("utf-8")
            # Header + 10 data rows.
            lines = csv_content.strip().split("\n")
            assert len(lines) == 11

    def test_empty_domains(self) -> None:
        zip_bytes = _build_csv_zip({})

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert len(zf.namelist()) == 0

    def test_single_domain(self) -> None:
        dfs = {"AE": _make_df("AE", 2)}
        zip_bytes = _build_csv_zip(dfs)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert zf.namelist() == ["AE.csv"]


# -----------------------------------------------------------------------
# Scenario: Configuration controls
# -----------------------------------------------------------------------


class TestConfigurationDefaults:
    """Verify MockPipelineResult captures config values."""

    def test_num_subjects_configurable(self) -> None:
        result = MockPipelineResult(
            domain_dataframes={},
            validation_results={},
            run_id="r1",
            num_subjects=50,
            synthesizer_type="gaussian_copula",
        )
        assert result.num_subjects == 50

    def test_synthesizer_type_configurable(self) -> None:
        result = MockPipelineResult(
            domain_dataframes={},
            validation_results={},
            run_id="r1",
            num_subjects=10,
            synthesizer_type="ctgan",
        )
        assert result.synthesizer_type == "ctgan"
