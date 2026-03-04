"""Unit tests for priority_sections (PTCV-52).

Tests targeted extraction for ICH E6(R3) B.4, B.5, B.10, B.14 sections,
covering all 4 GHERKIN acceptance criteria.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.extraction.models import ExtractedTable, TextBlock
from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.priority_sections import (
    PRIORITY_SECTIONS,
    PrioritySectionResult,
    extract_priority_sections,
    _is_soa_candidate,
    _map_sections_to_pages,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_text_block(
    text: str, page: int, block_index: int = 0
) -> TextBlock:
    return TextBlock(
        run_id="run-001",
        source_registry_id="NCT_TEST",
        source_sha256="a" * 64,
        page_number=page,
        block_index=block_index,
        text=text,
        block_type="paragraph",
    )


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


def _make_table(
    header: list[str],
    rows: list[list[str]],
    page: int,
    table_index: int = 0,
) -> ExtractedTable:
    return ExtractedTable(
        run_id="run-001",
        source_registry_id="NCT_TEST",
        source_sha256="a" * 64,
        page_number=page,
        extractor_used="pdfplumber",
        table_index=table_index,
        header_row=json.dumps(header),
        data_rows=json.dumps(rows),
    )


# Realistic text blocks simulating a multi-page protocol
PROTOCOL_BLOCKS = [
    # Pages 1-4: General info (B.1)
    _make_text_block("1. GENERAL INFORMATION Protocol Title: Test Study", 1),
    _make_text_block("Sponsor: Test Pharma Inc.", 1, 1),
    _make_text_block("Protocol version 3.0", 2),
    _make_text_block("Date: January 2024", 3),
    # Pages 5-10: Trial Design (B.4) — this is a priority section
    _make_text_block("4. STUDY DESIGN This is a multicenter randomized study", 5),
    _make_text_block("The study has three phases: screening, treatment, follow-up", 5, 1),
    _make_text_block("Patients will be randomized 1:1 to active or placebo", 6),
    _make_text_block("See Section B.5 for eligibility criteria", 7),
    _make_text_block("Treatment duration is 12 weeks with 4-week follow-up", 8),
    _make_text_block("The schedule of activities is shown in Table 1", 9),
    _make_text_block("Study visits occur at screening, baseline, week 2, 4, 8, 12", 10),
    # Pages 11-15: Selection of Subjects (B.5) — priority section
    _make_text_block("5. SELECTION OF SUBJECTS Inclusion Criteria", 11),
    _make_text_block("Age >= 18 years, confirmed diagnosis", 11, 1),
    _make_text_block("Exclusion: pregnant, prior treatment", 12),
    _make_text_block("Refer to Section B.4 for study design", 13),
    # Pages 16-18: Treatment (B.6) — NOT priority
    _make_text_block("6. TREATMENT OF SUBJECTS Dosing regimen", 16),
    _make_text_block("Drug X 100mg daily for 12 weeks", 17),
    # Pages 19-22: Assessment of Efficacy (B.10) — priority
    _make_text_block("10. ASSESSMENT OF EFFICACY Primary endpoint", 19),
    _make_text_block("Primary: Change in biomarker at Week 12", 19, 1),
    _make_text_block("Secondary: Overall response rate", 20),
    _make_text_block("Tumor assessments every 8 weeks per RECIST 1.1", 21),
    # Pages 23-25: Quality (B.11) — NOT priority
    _make_text_block("11. QUALITY ASSURANCE GCP compliance", 23),
]

SECTIONS = [
    _make_section("B.1", "General Information",
                  "1. GENERAL INFORMATION Protocol Title: Test Study"),
    _make_section("B.4", "Trial Design",
                  "4. STUDY DESIGN This is a multicenter randomized study"),
    _make_section("B.5", "Selection of Subjects",
                  "5. SELECTION OF SUBJECTS Inclusion Criteria"),
    _make_section("B.6", "Treatment of Subjects",
                  "6. TREATMENT OF SUBJECTS Dosing regimen"),
    _make_section("B.10", "Assessment of Efficacy",
                  "10. ASSESSMENT OF EFFICACY Primary endpoint"),
    _make_section("B.11", "Quality Control and Quality Assurance",
                  "11. QUALITY ASSURANCE GCP compliance"),
]

# SoA table on page 9 (within B.4)
SOA_TABLE = _make_table(
    header=["Assessment", "Screening", "Baseline", "Week 2", "Week 4", "Week 8", "Follow-up"],
    rows=[
        ["Physical Exam", "X", "X", "X", "X", "X", "X"],
        ["ECG", "X", "", "X", "", "X", ""],
        ["Lab Tests", "X", "X", "X", "X", "X", "X"],
    ],
    page=9,
)

# Non-SoA table on page 12 (within B.5)
NON_SOA_TABLE = _make_table(
    header=["Criterion", "Requirement"],
    rows=[
        ["Age", ">= 18 years"],
        ["Diagnosis", "Confirmed by biopsy"],
    ],
    page=12,
    table_index=1,
)

# Efficacy table on page 20 (within B.10)
EFFICACY_TABLE = _make_table(
    header=["Endpoint", "Timepoint", "Method"],
    rows=[
        ["Biomarker", "Week 12", "Blood draw"],
        ["Response", "Week 8, 16, 24", "CT scan"],
    ],
    page=20,
    table_index=2,
)


class TestPrioritySectionsConfig:
    """Verify PRIORITY_SECTIONS constant."""

    def test_contains_required_codes(self) -> None:
        assert "B.4" in PRIORITY_SECTIONS
        assert "B.5" in PRIORITY_SECTIONS
        assert "B.10" in PRIORITY_SECTIONS
        assert "B.14" in PRIORITY_SECTIONS

    def test_excludes_non_priority(self) -> None:
        assert "B.1" not in PRIORITY_SECTIONS
        assert "B.6" not in PRIORITY_SECTIONS
        assert "B.11" not in PRIORITY_SECTIONS


class TestPrioritySectionExtraction:
    """Scenario: Priority sections extracted without truncation."""

    def test_returns_only_priority_sections(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [SOA_TABLE, NON_SOA_TABLE, EFFICACY_TABLE],
        )
        codes = {r.section_code for r in results}
        # B.4, B.5, B.10 are present; B.14 is not in SECTIONS
        assert codes == {"B.4", "B.5", "B.10"}

    def test_full_text_not_truncated(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [],
        )
        b4 = next(r for r in results if r.section_code == "B.4")
        # Full text should include content from pages 5-10
        assert "multicenter randomized study" in b4.full_text
        assert "schedule of activities" in b4.full_text.lower()
        # Should be longer than the 2000-char truncation limit
        assert len(b4.full_text) > len("4. STUDY DESIGN This is a multicenter randomized study")

    def test_b5_full_text_includes_all_pages(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [],
        )
        b5 = next(r for r in results if r.section_code == "B.5")
        assert "Inclusion Criteria" in b5.full_text
        assert "Exclusion" in b5.full_text
        assert "Refer to Section B.4" in b5.full_text


class TestTableToSectionMapping:
    """Scenario: Tables mapped to parent sections by page range."""

    def test_soa_table_mapped_to_b4(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [SOA_TABLE, NON_SOA_TABLE, EFFICACY_TABLE],
        )
        b4 = next(r for r in results if r.section_code == "B.4")
        assert SOA_TABLE in b4.tables

    def test_non_soa_table_mapped_to_b5(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [SOA_TABLE, NON_SOA_TABLE, EFFICACY_TABLE],
        )
        b5 = next(r for r in results if r.section_code == "B.5")
        assert NON_SOA_TABLE in b5.tables

    def test_efficacy_table_mapped_to_b10(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [SOA_TABLE, NON_SOA_TABLE, EFFICACY_TABLE],
        )
        b10 = next(r for r in results if r.section_code == "B.10")
        assert EFFICACY_TABLE in b10.tables

    def test_page_ranges_set(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [],
        )
        b4 = next(r for r in results if r.section_code == "B.4")
        assert b4.page_range[0] > 0
        assert b4.page_range[1] >= b4.page_range[0]


class TestSoaCandidateFlagging:
    """Scenario: SoA candidate tables flagged."""

    def test_soa_table_flagged_as_candidate(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [SOA_TABLE, NON_SOA_TABLE, EFFICACY_TABLE],
        )
        b4 = next(r for r in results if r.section_code == "B.4")
        assert SOA_TABLE in b4.soa_candidate_tables

    def test_non_soa_table_not_flagged(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [SOA_TABLE, NON_SOA_TABLE, EFFICACY_TABLE],
        )
        b5 = next(r for r in results if r.section_code == "B.5")
        assert NON_SOA_TABLE not in b5.soa_candidate_tables

    def test_is_soa_candidate_with_visit_headers(self) -> None:
        soa = _make_table(
            ["Activity", "Screening", "Baseline", "Day 1", "Week 2", "Follow-up"],
            [["ECG", "X", "X", "X", "X", "X"]], page=1,
        )
        assert _is_soa_candidate(soa) is True

    def test_is_soa_candidate_non_visit_headers(self) -> None:
        non_soa = _make_table(
            ["Column A", "Column B"],
            [["val1", "val2"]], page=1,
        )
        assert _is_soa_candidate(non_soa) is False


class TestBackwardCompatibility:
    """Scenario: Non-priority sections still extracted at current quality."""

    def test_non_priority_sections_excluded_from_results(self) -> None:
        """Only priority sections appear in output."""
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [],
        )
        codes = {r.section_code for r in results}
        assert "B.1" not in codes
        assert "B.6" not in codes
        assert "B.11" not in codes

    def test_no_text_blocks_returns_fallback(self) -> None:
        """When no text blocks available, uses content_json text."""
        results = extract_priority_sections(
            SECTIONS, [], [],
        )
        # Should still produce results for priority sections
        priority_in_sections = [
            s for s in SECTIONS if s.section_code in PRIORITY_SECTIONS
        ]
        assert len(results) == len(priority_in_sections)
        # Text comes from content_json fallback
        b4 = next(r for r in results if r.section_code == "B.4")
        assert "multicenter" in b4.full_text

    def test_no_tables_returns_empty_lists(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, None,
        )
        for r in results:
            assert r.tables == []
            assert r.soa_candidate_tables == []

    def test_cross_references_detected(self) -> None:
        results = extract_priority_sections(
            SECTIONS, PROTOCOL_BLOCKS, [],
        )
        b4 = next(r for r in results if r.section_code == "B.4")
        # B.4 text contains "See Section B.5"
        assert any("B.5" in ref for ref in b4.cross_references)


class TestMapSectionsToPages:
    """Tests for the page mapping helper."""

    def test_maps_sections_by_text_match(self) -> None:
        page_ranges = _map_sections_to_pages(SECTIONS, PROTOCOL_BLOCKS)
        assert "B.4" in page_ranges
        assert page_ranges["B.4"][0] == 5  # B.4 starts on page 5

    def test_returns_empty_for_no_blocks(self) -> None:
        assert _map_sections_to_pages(SECTIONS, []) == {}

    def test_returns_empty_for_no_sections(self) -> None:
        assert _map_sections_to_pages([], PROTOCOL_BLOCKS) == {}
