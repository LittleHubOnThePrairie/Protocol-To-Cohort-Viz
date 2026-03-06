"""Unit tests for SoA-to-SDTM mapper (PTCV-53).

Tests the mapping from RawSoaTable data to mock SDTM Trial Design
domains (TV, TA, TE, SE) covering all 5 GHERKIN acceptance criteria.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pandas as pd

from ptcv.ich_parser.models import IchSection
from ptcv.soa_extractor.models import RawSoaTable
from ptcv.sdtm.soa_mapper import (
    MockSdtmDataset,
    SoaToSdtmMapper,
    _DOMAIN_TO_CATEGORY,
    _classify_assessment,
)
from ptcv.sdtm.models import SoaCellMetadata


# -----------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------

def _make_section(
    code: str, name: str, text_excerpt: str
) -> IchSection:
    return IchSection(
        run_id="run-001",
        source_run_id="",
        source_sha256="a" * 64,
        registry_id="NCT_TEST",
        section_code=code,
        section_name=name,
        content_json=json.dumps({"text_excerpt": text_excerpt}),
        confidence_score=0.85,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
    )


# Realistic SoA table with screening, baseline, treatment, follow-up visits
SOA_TABLE = RawSoaTable(
    visit_headers=[
        "Screening",
        "Baseline",
        "Week 2",
        "Week 4",
        "Week 8",
        "Follow-up",
        "End of Study",
    ],
    day_headers=[
        "Day -14 to -1",
        "Day 1",
        "Day 15 ± 3",
        "Day 29 ± 3",
        "Day 57 ± 3",
        "Day 85 ± 7",
        "",
    ],
    activities=[
        ("Informed Consent", [True, False, False, False, False, False, False]),
        ("Physical Exam", [True, True, True, True, True, False, True]),
        ("12-lead ECG", [True, False, True, False, True, False, True]),
        ("Complete Blood Count", [True, True, True, True, True, True, True]),
        ("Vital Signs", [True, True, True, True, True, True, True]),
        ("Adverse Events", [False, True, True, True, True, True, True]),
        ("PK Samples", [False, True, True, True, True, False, False]),
        ("ECOG Performance", [True, True, False, False, False, False, False]),
        ("Concomitant Medications", [False, True, True, True, True, True, True]),
    ],
    section_code="B.4",
)

# B.4 section with treatment arm info
B4_SECTION = _make_section(
    "B.4",
    "Trial Design",
    (
        "This is a parallel-group, double-blind study.\n"
        "Arm A: Drug X 10 mg once daily\n"
        "Arm B: Placebo once daily\n"
        "Screening period followed by treatment period then follow-up."
    ),
)

# Minimal SoA table with only 3 visits
MINIMAL_SOA = RawSoaTable(
    visit_headers=["Screening", "Baseline", "Follow-up"],
    day_headers=["Day -7", "Day 1", "Day 30"],
    activities=[
        ("Physical Exam", [True, True, True]),
        ("Lab Tests", [True, True, True]),
    ],
    section_code="B.4",
)


# -----------------------------------------------------------------------
# Scenario 1: Visit schedule maps to TV domain
# -----------------------------------------------------------------------

class TestTvDomainMapping:
    """Scenario: Visit schedule maps to TV domain."""

    def test_tv_dataframe_produced(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert isinstance(result.tv, pd.DataFrame)
        assert not result.tv.empty

    def test_tv_has_required_variables(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        required_cols = {"STUDYID", "DOMAIN", "VISITNUM", "VISIT", "VISITDY"}
        assert required_cols.issubset(set(result.tv.columns))

    def test_each_visit_has_visitnum_visit_visitdy(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        for _, row in result.tv.iterrows():
            assert pd.notna(row["VISITNUM"])
            assert pd.notna(row["VISIT"])
            assert row["VISIT"].strip() != ""
            assert pd.notna(row["VISITDY"])

    def test_visits_ordered_chronologically(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        visit_days = list(result.tv["VISITDY"])
        assert visit_days == sorted(visit_days)

    def test_visitnum_sequential(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        visit_nums = list(result.tv["VISITNUM"])
        expected = [float(i) for i in range(1, len(visit_nums) + 1)]
        assert visit_nums == expected

    def test_screening_before_baseline(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        visits = list(result.tv["VISIT"])
        day_vals = list(result.tv["VISITDY"])
        # Screening should have a negative day offset
        screening_idx = next(
            i for i, v in enumerate(visits) if "Screening" in v
        )
        assert day_vals[screening_idx] < 0

    def test_tv_domain_column_is_tv(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert all(result.tv["DOMAIN"] == "TV")

    def test_visit_count_matches_soa_columns(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert len(result.tv) == len(SOA_TABLE.visit_headers)


# -----------------------------------------------------------------------
# Scenario 2: Treatment arms map to TA domain
# -----------------------------------------------------------------------

class TestTaDomainMapping:
    """Scenario: Treatment arms map to TA domain."""

    def test_ta_dataframe_produced(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        assert isinstance(result.ta, pd.DataFrame)
        assert not result.ta.empty

    def test_ta_has_required_variables(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        required_cols = {"STUDYID", "DOMAIN", "ARMCD", "ARM", "TAETORD"}
        assert required_cols.issubset(set(result.ta.columns))

    def test_each_arm_has_armcd_and_arm(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        for _, row in result.ta.iterrows():
            assert row["ARMCD"].strip() != ""
            assert row["ARM"].strip() != ""
            assert pd.notna(row["TAETORD"])

    def test_two_arms_detected(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        unique_arms = result.ta["ARMCD"].unique()
        assert len(unique_arms) == 2

    def test_element_order_present(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        assert all(result.ta["TAETORD"] > 0)

    def test_ta_domain_column_is_ta(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        assert all(result.ta["DOMAIN"] == "TA")

    def test_ta_without_sections_uses_fallback(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[], studyid="NCT_TEST")
        # Should still produce TA with at least one "Treatment" arm
        assert not result.ta.empty


# -----------------------------------------------------------------------
# Scenario 3: Study phases map to TE domain
# -----------------------------------------------------------------------

class TestTeDomainMapping:
    """Scenario: Study phases map to TE domain."""

    def test_te_dataframe_produced(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert isinstance(result.te, pd.DataFrame)
        assert not result.te.empty

    def test_te_has_required_variables(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        required_cols = {"STUDYID", "DOMAIN", "ETCD", "ELEMENT"}
        assert required_cols.issubset(set(result.te.columns))

    def test_screening_treatment_followup_present(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        elements = set(result.te["ELEMENT"])
        assert "Screening" in elements
        assert "Treatment" in elements
        assert "Follow-up" in elements

    def test_te_etcd_codes_valid(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        valid_codes = {"SCRN", "TRT", "FUP", "EOS", "RUN", "RAND"}
        for etcd in result.te["ETCD"]:
            assert etcd in valid_codes, f"Unexpected ETCD: {etcd}"

    def test_te_domain_column_is_te(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert all(result.te["DOMAIN"] == "TE")

    def test_te_enriched_by_b4_text(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        elements = set(result.te["ELEMENT"])
        # B.4 text mentions screening, treatment, follow-up
        assert "Screening" in elements
        assert "Treatment" in elements


# -----------------------------------------------------------------------
# Scenario 4: Assessment rows map to SDTM domain indicators
# -----------------------------------------------------------------------

class TestDomainMapping:
    """Scenario: Assessment rows map to SDTM domain indicators."""

    def test_domain_mapping_produced(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert isinstance(result.domain_mapping, pd.DataFrame)
        assert not result.domain_mapping.empty

    def test_ecg_mapped_to_eg(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        ecg_rows = result.domain_mapping[
            result.domain_mapping["ASSESSMENT"].str.contains("ECG", case=False)
        ]
        assert not ecg_rows.empty
        assert all(ecg_rows["SDTM_DOMAIN"] == "EG")

    def test_blood_count_mapped_to_lb(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        lab_rows = result.domain_mapping[
            result.domain_mapping["ASSESSMENT"].str.contains(
                "Blood Count", case=False
            )
        ]
        assert not lab_rows.empty
        assert all(lab_rows["SDTM_DOMAIN"] == "LB")

    def test_vital_signs_mapped_to_vs(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        vs_rows = result.domain_mapping[
            result.domain_mapping["ASSESSMENT"].str.contains(
                "Vital", case=False
            )
        ]
        assert not vs_rows.empty
        assert all(vs_rows["SDTM_DOMAIN"] == "VS")

    def test_pk_samples_mapped_to_pc(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        pk_rows = result.domain_mapping[
            result.domain_mapping["ASSESSMENT"].str.contains("PK", case=False)
        ]
        assert not pk_rows.empty
        assert all(pk_rows["SDTM_DOMAIN"] == "PC")

    def test_adverse_events_mapped_to_ae(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        ae_rows = result.domain_mapping[
            result.domain_mapping["ASSESSMENT"].str.contains(
                "Adverse", case=False
            )
        ]
        assert not ae_rows.empty
        assert all(ae_rows["SDTM_DOMAIN"] == "AE")

    def test_concomitant_mapped_to_cm(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        cm_rows = result.domain_mapping[
            result.domain_mapping["ASSESSMENT"].str.contains(
                "Concomitant", case=False
            )
        ]
        assert not cm_rows.empty
        assert all(cm_rows["SDTM_DOMAIN"] == "CM")

    def test_mapping_has_scheduled_count(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert "SCHEDULED_VISITS" in result.domain_mapping.columns
        assert "TOTAL_VISITS" in result.domain_mapping.columns
        # Informed Consent is scheduled at 1 visit only
        consent_row = result.domain_mapping[
            result.domain_mapping["ASSESSMENT"].str.contains("Consent")
        ].iloc[0]
        assert consent_row["SCHEDULED_VISITS"] == 1

    def test_classify_assessment_defaults_to_fa(self) -> None:
        domain_code, _ = _classify_assessment("Unknown Assessment Type XYZ")
        assert domain_code == "FA"

    def test_classify_physical_exam_to_pe(self) -> None:
        domain_code, _ = _classify_assessment("Physical Exam")
        assert domain_code == "PE"


# -----------------------------------------------------------------------
# Scenario 5: Output is valid mock SDTM structure
# -----------------------------------------------------------------------

class TestMockSdtmStructure:
    """Scenario: Output is valid mock SDTM structure."""

    def test_returns_mock_sdtm_dataset(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        assert isinstance(result, MockSdtmDataset)

    def test_all_domains_present(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        domains = result.domains
        assert "TV" in domains
        assert "TA" in domains
        assert "TE" in domains
        assert "SE" in domains

    def test_studyid_consistent_across_domains(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(
            SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST"
        )
        for name, df in result.domains.items():
            if not df.empty:
                assert all(df["STUDYID"] == "NCT_TEST"), (
                    f"STUDYID mismatch in {name}"
                )

    def test_domain_columns_match_cdisc_naming(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        # All column names should be uppercase (CDISC convention)
        for name, df in result.domains.items():
            for col in df.columns:
                assert col == col.upper(), (
                    f"Column {col} in {name} is not uppercase"
                )

    def test_all_required_tv_variables(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        tv_required = {"STUDYID", "DOMAIN", "VISITNUM", "VISIT", "VISITDY"}
        assert tv_required.issubset(set(result.tv.columns))

    def test_all_required_ta_variables(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        ta_required = {"STUDYID", "DOMAIN", "ARMCD", "ARM", "TAETORD", "ETCD", "ELEMENT"}
        assert ta_required.issubset(set(result.ta.columns))

    def test_all_required_te_variables(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        te_required = {"STUDYID", "DOMAIN", "ETCD", "ELEMENT"}
        assert te_required.issubset(set(result.te.columns))

    def test_se_domain_present(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert isinstance(result.se, pd.DataFrame)
        assert not result.se.empty
        assert all(result.se["DOMAIN"] == "SE")

    def test_se_has_required_variables(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        se_required = {"STUDYID", "DOMAIN", "SESEQ", "SESTDY", "SEENDY"}
        assert se_required.issubset(set(result.se.columns))

    def test_serializable_to_csv(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=[B4_SECTION], studyid="NCT_TEST")
        csvs = result.to_csv()
        assert "TV" in csvs
        assert "TA" in csvs
        assert "TE" in csvs
        assert "SE" in csvs
        assert "DOMAIN_MAP" in csvs
        # CSV strings should be non-empty
        for name, csv_str in csvs.items():
            assert len(csv_str) > 0, f"Empty CSV for {name}"

    def test_run_id_auto_generated(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert result.run_id
        assert len(result.run_id) > 10  # UUID is ~36 chars

    def test_run_id_explicit(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST", run_id="test-run-42")
        assert result.run_id == "test-run-42"


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_minimal_soa_table(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(MINIMAL_SOA, studyid="TEST")
        assert len(result.tv) == 3
        assert not result.te.empty

    def test_no_sections_produces_valid_output(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, sections=None, studyid="TEST")
        assert isinstance(result, MockSdtmDataset)
        assert not result.tv.empty
        assert not result.te.empty

    def test_empty_activities_produces_empty_mapping(self) -> None:
        empty_soa = RawSoaTable(
            visit_headers=["Screening", "Baseline"],
            day_headers=["Day -7", "Day 1"],
            activities=[],
            section_code="B.4",
        )
        mapper = SoaToSdtmMapper()
        result = mapper.map(empty_soa, studyid="TEST")
        assert result.domain_mapping.empty
        # TV should still have visits
        assert len(result.tv) == 2


# -----------------------------------------------------------------------
# Cell-level metadata (PTCV-53 grooming enrichment from PTCV-1 research)
# -----------------------------------------------------------------------

class TestSoaCellMetadata:
    """Cell-level SoA metadata matrix tests."""

    def test_soa_matrix_populated(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert len(result.soa_matrix) > 0

    def test_soa_matrix_keys_are_tuples(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        for key in result.soa_matrix:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], int)
            assert isinstance(key[1], str)

    def test_cell_status_defaults_to_required(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        for cell in result.soa_matrix.values():
            assert cell.status == "required"

    def test_cell_condition_defaults_to_empty(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        for cell in result.soa_matrix.values():
            assert cell.condition == ""

    def test_cell_category_safety(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        # Vital Signs → VS → safety
        vs_cells = [
            c for c in result.soa_matrix.values()
            if c.cdash_domain == "VS"
        ]
        assert len(vs_cells) > 0
        assert all(c.category == "safety" for c in vs_cells)

    def test_cell_category_efficacy(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        # ECOG Performance → QS → efficacy
        qs_cells = [
            c for c in result.soa_matrix.values()
            if c.cdash_domain == "QS"
        ]
        assert len(qs_cells) > 0
        assert all(c.category == "efficacy" for c in qs_cells)

    def test_cell_category_operational(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        # Informed Consent → DS → operational
        ds_cells = [
            c for c in result.soa_matrix.values()
            if c.cdash_domain == "DS"
        ]
        assert len(ds_cells) > 0
        assert all(c.category == "operational" for c in ds_cells)

    def test_cell_cdash_domain_matches_classify(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        for cell in result.soa_matrix.values():
            expected_domain, _ = _classify_assessment(cell.assessment)
            assert cell.cdash_domain == expected_domain

    def test_cell_timing_window_from_tv(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        # Build expected windows from TV
        tv_windows: dict[int, tuple[int, int]] = {}
        for _, row in result.tv.iterrows():
            vnum = int(row["VISITNUM"])
            tv_windows[vnum] = (int(row["TVSTRL"]), int(row["TVENRL"]))
        for cell in result.soa_matrix.values():
            expected = tv_windows.get(cell.visitnum, (0, 0))
            assert cell.timing_window_days == expected

    def test_unscheduled_cells_absent(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        # Informed Consent is only at visit 1 (Screening)
        consent_cells = [
            c for c in result.soa_matrix.values()
            if "Consent" in c.assessment
        ]
        assert len(consent_cells) == 1
        assert consent_cells[0].visitnum == 1

    def test_domain_mapping_has_category_column(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert "CATEGORY" in result.domain_mapping.columns
        categories = set(result.domain_mapping["CATEGORY"])
        assert categories.issubset({"safety", "efficacy", "operational"})

    def test_domain_mapping_has_cdash_domain_column(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        assert "CDASH_DOMAIN" in result.domain_mapping.columns
        # CDASH_DOMAIN should match SDTM_DOMAIN
        for _, row in result.domain_mapping.iterrows():
            assert row["CDASH_DOMAIN"] == row["SDTM_DOMAIN"]

    def test_soa_matrix_csv_serialization(self) -> None:
        mapper = SoaToSdtmMapper()
        result = mapper.map(SOA_TABLE, studyid="NCT_TEST")
        csvs = result.to_csv()
        assert "SOA_MATRIX" in csvs
        assert "VISITNUM" in csvs["SOA_MATRIX"]
        assert "CATEGORY" in csvs["SOA_MATRIX"]

    def test_empty_activities_empty_matrix(self) -> None:
        empty_soa = RawSoaTable(
            visit_headers=["Screening", "Baseline"],
            day_headers=["Day -7", "Day 1"],
            activities=[],
            section_code="B.4",
        )
        mapper = SoaToSdtmMapper()
        result = mapper.map(empty_soa, studyid="TEST")
        assert len(result.soa_matrix) == 0
