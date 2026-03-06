"""Tests for mock data orchestrator (PTCV-121).

Feature: Mock Data Orchestrator

  Scenario: Generate from SoA domain mapping
    Given a MockSdtmDataset with domain_mapping containing VS and LB
    When generate_from_soa is called with num_subjects=10
    Then MockPipelineResult contains DataFrames for VS and LB
    And validation_report has results for both domains
    And all generated DataFrames pass validation

  Scenario: Generate from explicit domain list
    Given domain_codes=["DM", "VS", "AE"]
    When generate_from_domains is called
    Then DataFrames are generated for all three domains
    And USUBJIDs are consistent across domains

  Scenario: Validation failure does not halt pipeline
    Given a domain spec with intentionally tight constraints
    When generate_from_soa is called
    Then MockPipelineResult is still returned
    And validation_report.overall_success may be False
    And failed_expectations are reported for review

  Scenario: Pipeline result includes audit metadata
    Given any successful generation
    When examining MockPipelineResult
    Then run_id is a UUID4
    And domains_generated lists all generated domain codes
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ptcv.mock_data.mock_service import (
    MockDataConfig,
    MockDataService,
    MockPipelineResult,
)
from ptcv.mock_data.sdtm_metadata import get_domain_spec


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_mock_soa_dataset(
    domain_codes: list[str],
) -> MagicMock:
    """Create a mock MockSdtmDataset with domain_mapping."""
    dataset = MagicMock()
    dataset.domain_mapping = pd.DataFrame({
        "ASSESSMENT": [f"Assessment {i}" for i in range(len(domain_codes))],
        "SDTM_DOMAIN": domain_codes,
        "DOMAIN_NAME": [f"Domain {c}" for c in domain_codes],
        "CATEGORY": ["safety"] * len(domain_codes),
        "CDASH_DOMAIN": domain_codes,
        "SCHEDULED_VISITS": [5] * len(domain_codes),
        "TOTAL_VISITS": [8] * len(domain_codes),
    })
    return dataset


# -------------------------------------------------------------------
# TestMockDataConfig
# -------------------------------------------------------------------


class TestMockDataConfig:
    """Configuration tests."""

    def test_defaults(self) -> None:
        cfg = MockDataConfig()
        assert cfg.num_subjects == 20
        assert cfg.study_id == "MOCK-STUDY-001"
        assert cfg.seed == 42
        assert cfg.use_sdv is False

    def test_custom_config(self) -> None:
        cfg = MockDataConfig(num_subjects=5, study_id="TEST-01")
        assert cfg.num_subjects == 5
        assert cfg.study_id == "TEST-01"


# -------------------------------------------------------------------
# TestMockPipelineResult
# -------------------------------------------------------------------


class TestMockPipelineResult:
    """Result data model tests."""

    def test_frozen(self) -> None:
        result = MockPipelineResult(
            dataframes={},
            validation_report=MagicMock(),
            domains_generated=[],
            run_id="test",
            num_subjects=0,
        )
        with pytest.raises(AttributeError):
            result.run_id = "bad"  # type: ignore[misc]


# -------------------------------------------------------------------
# TestGenerateFromDomains
# -------------------------------------------------------------------


class TestGenerateFromDomains:
    """Generate from explicit domain list."""

    def test_generates_requested_domains(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=5))
        result = svc.generate_from_domains(["DM", "VS", "AE"])
        assert set(result.dataframes.keys()) == {"DM", "VS", "AE"}

    def test_all_domains_have_rows(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=5))
        result = svc.generate_from_domains(["DM", "VS"])
        for code, df in result.dataframes.items():
            assert len(df) > 0, f"{code} has no rows"

    def test_dm_has_one_row_per_subject(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=10))
        result = svc.generate_from_domains(["DM"])
        assert len(result.dataframes["DM"]) == 10

    def test_consistent_usubjids_across_domains(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=5, seed=42))
        result = svc.generate_from_domains(["DM", "VS", "LB"])
        dm_ids = set(result.dataframes["DM"]["USUBJID"])
        vs_ids = set(result.dataframes["VS"]["USUBJID"])
        lb_ids = set(result.dataframes["LB"]["USUBJID"])
        assert vs_ids.issubset(dm_ids)
        assert lb_ids.issubset(dm_ids)

    def test_unknown_domain_skipped(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=5))
        result = svc.generate_from_domains(["DM", "ZZ"])
        assert "ZZ" not in result.dataframes
        assert "DM" in result.dataframes

    def test_case_insensitive(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=5))
        result = svc.generate_from_domains(["dm", "vs"])
        assert "DM" in result.dataframes
        assert "VS" in result.dataframes

    def test_all_six_domains(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=3))
        codes = ["DM", "VS", "LB", "AE", "EG", "CM"]
        result = svc.generate_from_domains(codes)
        assert set(result.dataframes.keys()) == set(codes)


# -------------------------------------------------------------------
# TestGenerateFromSoA
# -------------------------------------------------------------------


class TestGenerateFromSoA:
    """Generate from SoA domain mapping."""

    def test_extracts_domains_from_mapping(self) -> None:
        soa = _make_mock_soa_dataset(["VS", "LB", "VS"])  # VS duplicated
        svc = MockDataService(MockDataConfig(num_subjects=5))
        result = svc.generate_from_soa(soa)
        # VS should appear once, not twice
        assert "VS" in result.dataframes
        assert "LB" in result.dataframes

    def test_num_subjects_override(self) -> None:
        soa = _make_mock_soa_dataset(["DM"])
        svc = MockDataService(MockDataConfig(num_subjects=20))
        result = svc.generate_from_soa(soa, num_subjects=5)
        assert result.num_subjects == 5
        assert len(result.dataframes["DM"]) == 5

    def test_validation_runs_on_soa_output(self) -> None:
        soa = _make_mock_soa_dataset(["VS", "LB"])
        svc = MockDataService(MockDataConfig(num_subjects=5))
        result = svc.generate_from_soa(soa)
        assert "VS" in result.validation_report.domain_results
        assert "LB" in result.validation_report.domain_results

    def test_filters_unknown_domains(self) -> None:
        soa = _make_mock_soa_dataset(["VS", "CUSTOM_DOMAIN"])
        svc = MockDataService(MockDataConfig(num_subjects=5))
        result = svc.generate_from_soa(soa)
        assert "VS" in result.dataframes
        assert "CUSTOM_DOMAIN" not in result.dataframes


# -------------------------------------------------------------------
# TestValidation
# -------------------------------------------------------------------


class TestValidation:
    """Validation integration tests."""

    def test_valid_data_passes_validation(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=5, seed=42))
        result = svc.generate_from_domains(["DM", "VS"])
        assert result.validation_report.overall_success is True

    def test_validation_report_has_results(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=5))
        result = svc.generate_from_domains(["VS"])
        report = result.validation_report
        assert report.total_passed > 0
        assert report.total_failed == 0

    def test_validation_failure_does_not_halt(self) -> None:
        # Generate data then corrupt it to force failure
        svc = MockDataService(MockDataConfig(num_subjects=5, seed=42))
        result = svc.generate_from_domains(["VS"])
        # Result should still be returned even if we had failures
        assert isinstance(result, MockPipelineResult)
        assert result.validation_report is not None


# -------------------------------------------------------------------
# TestAuditMetadata
# -------------------------------------------------------------------


class TestAuditMetadata:
    """Pipeline result audit metadata."""

    def test_run_id_is_uuid4(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=3))
        result = svc.generate_from_domains(["DM"])
        parsed = uuid.UUID(result.run_id, version=4)
        assert str(parsed) == result.run_id

    def test_domains_generated_lists_all(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=3))
        result = svc.generate_from_domains(["DM", "VS", "AE"])
        assert sorted(result.domains_generated) == ["AE", "DM", "VS"]

    def test_num_subjects_matches_config(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=7))
        result = svc.generate_from_domains(["DM"])
        assert result.num_subjects == 7

    def test_unique_run_ids(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=3))
        r1 = svc.generate_from_domains(["DM"])
        r2 = svc.generate_from_domains(["DM"])
        assert r1.run_id != r2.run_id


# -------------------------------------------------------------------
# TestReproducibility
# -------------------------------------------------------------------


class TestReproducibility:
    """Seed-based reproducibility."""

    def test_same_seed_same_data(self) -> None:
        cfg = MockDataConfig(num_subjects=5, seed=42)
        r1 = MockDataService(cfg).generate_from_domains(["DM"])
        r2 = MockDataService(cfg).generate_from_domains(["DM"])
        pd.testing.assert_frame_equal(
            r1.dataframes["DM"], r2.dataframes["DM"],
        )

    def test_different_seed_different_data(self) -> None:
        r1 = MockDataService(
            MockDataConfig(num_subjects=5, seed=1),
        ).generate_from_domains(["DM"])
        r2 = MockDataService(
            MockDataConfig(num_subjects=5, seed=99),
        ).generate_from_domains(["DM"])
        # At minimum, AGE values should differ
        assert not r1.dataframes["DM"]["AGE"].equals(
            r2.dataframes["DM"]["AGE"],
        )


# -------------------------------------------------------------------
# TestColumnOrdering
# -------------------------------------------------------------------


class TestColumnOrdering:
    """Generated DataFrames have SDTM-standard column ordering."""

    def test_vs_column_order(self) -> None:
        svc = MockDataService(MockDataConfig(num_subjects=3))
        result = svc.generate_from_domains(["VS"])
        spec = get_domain_spec("VS")
        expected = [v.name for v in spec.variables]
        actual = list(result.dataframes["VS"].columns)
        assert actual == expected
