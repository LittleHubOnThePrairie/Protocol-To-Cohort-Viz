"""Unit tests for synthetic SDTM data generator (PTCV-54).

Tests synthetic patient-level data generation from SoA-derived schemas,
covering all 4 GHERKIN acceptance criteria.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import numpy as np
import pandas as pd

from ptcv.soa_extractor.models import RawSoaTable
from ptcv.sdtm.soa_mapper import MockSdtmDataset, SoaToSdtmMapper
from ptcv.sdtm.synthetic_generator import (
    SyntheticConfig,
    SyntheticSdtmGenerator,
    SyntheticSdtmResult,
)


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------

def _make_mock_dataset(studyid: str = "NCT_TEST") -> MockSdtmDataset:
    """Build a MockSdtmDataset using the SoaToSdtmMapper with realistic data."""
    import json
    from ptcv.ich_parser.models import IchSection

    soa_table = RawSoaTable(
        visit_headers=[
            "Screening", "Baseline", "Week 2", "Week 4",
            "Week 8", "Week 12", "Follow-up", "End of Study",
        ],
        day_headers=[
            "Day -14 to -1", "Day 1", "Day 15 ± 3", "Day 29 ± 3",
            "Day 57 ± 3", "Day 85 ± 3", "Day 113 ± 7", "",
        ],
        activities=[
            ("Informed Consent", [True, False, False, False, False, False, False, False]),
            ("Physical Exam", [True, True, False, True, False, True, False, True]),
            ("Complete Blood Count", [True, True, True, True, True, True, True, True]),
            ("Chemistry Panel", [True, True, True, True, True, True, True, True]),
            ("Vital Signs", [True, True, True, True, True, True, True, True]),
            ("12-lead ECG", [True, False, True, False, True, False, False, True]),
            ("Adverse Events", [False, True, True, True, True, True, True, True]),
            ("PK Samples", [False, True, True, True, True, True, False, False]),
        ],
        section_code="B.4",
    )

    b4_section = IchSection(
        run_id="run-001",
        source_run_id="",
        source_sha256="a" * 64,
        registry_id=studyid,
        section_code="B.4",
        section_name="Trial Design",
        content_json=json.dumps({
            "text_excerpt": (
                "This is a parallel-group, double-blind study.\n"
                "Arm A: Drug X 10 mg once daily\n"
                "Arm B: Placebo once daily\n"
                "Screening period followed by treatment then follow-up."
            ),
        }),
        confidence_score=0.88,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
    )

    mapper = SoaToSdtmMapper()
    return mapper.map(soa_table, sections=[b4_section], studyid=studyid)


@pytest.fixture
def mock_dataset() -> MockSdtmDataset:
    return _make_mock_dataset()


@pytest.fixture
def default_config() -> SyntheticConfig:
    return SyntheticConfig(
        n_subjects=100,
        randomization_ratio=[2.0, 1.0],
        dropout_rate=0.15,
        seed=42,
    )


@pytest.fixture
def small_config() -> SyntheticConfig:
    """Small config for faster tests."""
    return SyntheticConfig(
        n_subjects=20,
        randomization_ratio=[1.0, 1.0],
        dropout_rate=0.15,
        seed=42,
    )


# -----------------------------------------------------------------------
# Scenario 1: Synthetic subjects generated with demographics
# -----------------------------------------------------------------------

class TestDmGeneration:
    """Scenario: Synthetic subjects generated with demographics."""

    def test_dm_has_correct_row_count(
        self, mock_dataset: MockSdtmDataset, default_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=default_config)
        result = gen.generate(mock_dataset)
        assert len(result.dm) == 100

    def test_dm_has_required_variables(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        required = {"STUDYID", "DOMAIN", "USUBJID", "AGE", "SEX", "RACE"}
        assert required.issubset(set(result.dm.columns))

    def test_each_subject_has_usubjid(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert all(result.dm["USUBJID"].str.len() > 0)
        # All USUBJIDs unique
        assert result.dm["USUBJID"].nunique() == len(result.dm)

    def test_ages_in_plausible_range(
        self, mock_dataset: MockSdtmDataset, default_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=default_config)
        result = gen.generate(mock_dataset)
        assert all(result.dm["AGE"] >= 18)
        assert all(result.dm["AGE"] <= 85)

    def test_sex_values_valid(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert set(result.dm["SEX"]).issubset({"M", "F"})

    def test_race_values_valid(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        valid_races = {
            "WHITE", "BLACK OR AFRICAN AMERICAN", "ASIAN",
            "AMERICAN INDIAN OR ALASKA NATIVE", "MULTIPLE",
        }
        assert set(result.dm["RACE"]).issubset(valid_races)

    def test_arm_allocation_follows_ratio(
        self, mock_dataset: MockSdtmDataset,
    ) -> None:
        cfg = SyntheticConfig(
            n_subjects=1000,
            randomization_ratio=[2.0, 1.0],
            seed=42,
        )
        gen = SyntheticSdtmGenerator(config=cfg)
        result = gen.generate(mock_dataset)
        arm_counts = result.dm["ARMCD"].value_counts()
        # With 2:1 ratio and 1000 subjects, expect ~667:333
        # Allow 10% tolerance
        total = len(result.dm)
        first_arm = arm_counts.iloc[0]
        assert first_arm / total > 0.55, (
            f"Expected ~2/3 in first arm, got {first_arm}/{total}"
        )

    def test_two_arms_present(
        self, mock_dataset: MockSdtmDataset, default_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=default_config)
        result = gen.generate(mock_dataset)
        unique_arms = result.dm["ARMCD"].nunique()
        assert unique_arms == 2

    def test_dm_domain_is_dm(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert all(result.dm["DOMAIN"] == "DM")


# -----------------------------------------------------------------------
# Scenario 2: Visit instances follow TV schedule
# -----------------------------------------------------------------------

class TestSvGeneration:
    """Scenario: Visit instances follow TV schedule."""

    def test_sv_produced(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert isinstance(result.sv, pd.DataFrame)
        assert not result.sv.empty

    def test_sv_has_required_variables(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        required = {"STUDYID", "DOMAIN", "USUBJID", "VISITNUM", "VISIT", "SVSTDTC"}
        assert required.issubset(set(result.sv.columns))

    def test_visit_records_per_subject(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        # Each subject should have at least 2 visits (screening + baseline)
        visits_per_subj = result.sv.groupby("USUBJID").size()
        assert all(visits_per_subj >= 2)

    def test_some_subjects_have_fewer_visits_dropout(
        self, mock_dataset: MockSdtmDataset,
    ) -> None:
        cfg = SyntheticConfig(
            n_subjects=200,
            dropout_rate=0.20,
            seed=42,
        )
        gen = SyntheticSdtmGenerator(config=cfg)
        result = gen.generate(mock_dataset)
        visits_per_subj = result.sv.groupby("USUBJID").size()
        max_visits = len(mock_dataset.tv)
        # Some subjects should have fewer than max visits
        assert any(visits_per_subj < max_visits)

    def test_visit_dates_are_iso_format(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        for dtc in result.sv["SVSTDTC"]:
            # ISO format: YYYY-MM-DD
            assert len(str(dtc)) == 10
            parts = str(dtc).split("-")
            assert len(parts) == 3

    def test_sv_domain_is_sv(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert all(result.sv["DOMAIN"] == "SV")

    def test_zero_dropout_all_subjects_complete(
        self, mock_dataset: MockSdtmDataset,
    ) -> None:
        cfg = SyntheticConfig(
            n_subjects=50,
            dropout_rate=0.0,
            seed=42,
        )
        gen = SyntheticSdtmGenerator(config=cfg)
        result = gen.generate(mock_dataset)
        max_visits = len(mock_dataset.tv)
        visits_per_subj = result.sv.groupby("USUBJID").size()
        assert all(visits_per_subj == max_visits)


# -----------------------------------------------------------------------
# Scenario 3: Lab data generated for assessment visits
# -----------------------------------------------------------------------

class TestLbGeneration:
    """Scenario: Lab data generated for assessment visits."""

    def test_lb_produced(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert isinstance(result.lb, pd.DataFrame)
        assert not result.lb.empty

    def test_lb_has_required_variables(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        required = {
            "STUDYID", "DOMAIN", "USUBJID", "LBTESTCD",
            "LBTEST", "LBORRES", "LBORRESU", "VISITNUM",
        }
        assert required.issubset(set(result.lb.columns))

    def test_lb_values_clinically_plausible(
        self, mock_dataset: MockSdtmDataset, default_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=default_config)
        result = gen.generate(mock_dataset)
        # All numeric values should be non-negative
        values = result.lb["LBORRES"].astype(float)
        assert all(values >= 0)

    def test_lb_has_reference_ranges(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert "LBORNRLO" in result.lb.columns
        assert "LBORNRHI" in result.lb.columns

    def test_lb_has_visitnum(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert all(pd.notna(result.lb["VISITNUM"]))
        assert all(result.lb["VISITNUM"] > 0)

    def test_lb_has_visit_name(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert "VISIT" in result.lb.columns
        assert all(result.lb["VISIT"].str.len() > 0)

    def test_lb_domain_is_lb(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert all(result.lb["DOMAIN"] == "LB")

    def test_multiple_lab_tests_per_visit(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        # Each visit should have multiple lab tests
        first_subj = result.lb["USUBJID"].iloc[0]
        first_visit = result.lb[
            result.lb["USUBJID"] == first_subj
        ]["VISITNUM"].iloc[0]
        tests_at_visit = result.lb[
            (result.lb["USUBJID"] == first_subj)
            & (result.lb["VISITNUM"] == first_visit)
        ]
        assert len(tests_at_visit) > 1

    def test_lbtestcd_values_known(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        known_tests = {"WBC", "RBC", "HGB", "PLT", "ALT", "AST", "CREAT", "GLUC"}
        assert set(result.lb["LBTESTCD"]).issubset(known_tests)


# -----------------------------------------------------------------------
# Scenario 4: Reproducible with seed
# -----------------------------------------------------------------------

class TestReproducibility:
    """Scenario: Reproducible with seed."""

    def test_same_seed_same_dm(self, mock_dataset: MockSdtmDataset) -> None:
        cfg = SyntheticConfig(n_subjects=50, seed=12345)
        gen = SyntheticSdtmGenerator(config=cfg)

        result1 = gen.generate(mock_dataset)
        result2 = gen.generate(mock_dataset, config=cfg)

        pd.testing.assert_frame_equal(result1.dm, result2.dm)

    def test_same_seed_same_sv(self, mock_dataset: MockSdtmDataset) -> None:
        cfg = SyntheticConfig(n_subjects=50, seed=12345)
        gen = SyntheticSdtmGenerator(config=cfg)

        result1 = gen.generate(mock_dataset)
        result2 = gen.generate(mock_dataset, config=cfg)

        pd.testing.assert_frame_equal(result1.sv, result2.sv)

    def test_same_seed_same_lb(self, mock_dataset: MockSdtmDataset) -> None:
        cfg = SyntheticConfig(n_subjects=50, seed=12345)
        gen = SyntheticSdtmGenerator(config=cfg)

        result1 = gen.generate(mock_dataset)
        result2 = gen.generate(mock_dataset, config=cfg)

        pd.testing.assert_frame_equal(result1.lb, result2.lb)

    def test_different_seed_different_dm(
        self, mock_dataset: MockSdtmDataset,
    ) -> None:
        gen = SyntheticSdtmGenerator()
        cfg1 = SyntheticConfig(n_subjects=50, seed=111)
        cfg2 = SyntheticConfig(n_subjects=50, seed=222)

        result1 = gen.generate(mock_dataset, config=cfg1)
        result2 = gen.generate(mock_dataset, config=cfg2)

        # DM data should differ (different random draws)
        assert not result1.dm["AGE"].equals(result2.dm["AGE"])


# -----------------------------------------------------------------------
# Edge cases and result structure
# -----------------------------------------------------------------------

class TestResultStructure:
    """Result structure and edge cases."""

    def test_returns_synthetic_result(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert isinstance(result, SyntheticSdtmResult)

    def test_domains_property(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        domains = result.domains
        assert "DM" in domains
        assert "SV" in domains
        assert "LB" in domains

    def test_studyid_consistent(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        for name, df in result.domains.items():
            if not df.empty:
                assert all(df["STUDYID"] == mock_dataset.studyid), (
                    f"STUDYID mismatch in {name}"
                )

    def test_config_preserved(
        self, mock_dataset: MockSdtmDataset, small_config: SyntheticConfig,
    ) -> None:
        gen = SyntheticSdtmGenerator(config=small_config)
        result = gen.generate(mock_dataset)
        assert result.config is small_config

    def test_single_arm_dataset(self) -> None:
        """Dataset with only one arm still works."""
        soa = RawSoaTable(
            visit_headers=["Screening", "Baseline"],
            day_headers=["Day -7", "Day 1"],
            activities=[("Lab Tests", [True, True])],
            section_code="B.4",
        )
        mapper = SoaToSdtmMapper()
        dataset = mapper.map(soa, studyid="SINGLE_ARM")
        cfg = SyntheticConfig(n_subjects=10, seed=42)
        gen = SyntheticSdtmGenerator(config=cfg)
        result = gen.generate(dataset)
        assert len(result.dm) == 10
        # All subjects in single arm
        assert result.dm["ARMCD"].nunique() == 1
