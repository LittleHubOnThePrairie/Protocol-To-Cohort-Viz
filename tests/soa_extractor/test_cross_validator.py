"""Tests for SoA cross-validation framework (PTCV-216).

Tests completeness checking, alignment validation, fuzzy matching,
and the cross_validate combined function.
"""

from __future__ import annotations

import pytest

from ptcv.soa_extractor.cross_validator import (
    AssessmentMatch,
    CompletenessReport,
    AlignmentReport,
    CrossValidationResult,
    GENERAL_ASSESSMENTS,
    ONCOLOGY_PHASE12_ASSESSMENTS,
    check_alignment,
    check_completeness,
    cross_validate,
)
from ptcv.soa_extractor.models import RawSoaTable


def _make_table(
    activities: list[tuple[str, list[bool]]],
    visit_headers: list[str] | None = None,
) -> RawSoaTable:
    """Helper to build a RawSoaTable for testing."""
    if visit_headers is None:
        n = max((len(flags) for _, flags in activities), default=3)
        visit_headers = [f"V{i}" for i in range(1, n + 1)]
    return RawSoaTable(
        visit_headers=visit_headers,
        day_headers=[],
        activities=activities,
        section_code="B.4",
    )


class TestCheckCompleteness:
    """Tests for check_completeness function."""

    def test_full_coverage(self):
        """Test 100% coverage when all assessments match."""
        activities = [
            (name, [True, False, True])
            for name in GENERAL_ASSESSMENTS
        ]
        table = _make_table(activities)
        result = check_completeness(table, "general")

        assert result.found_count == len(GENERAL_ASSESSMENTS)
        assert result.coverage_ratio == 1.0
        assert result.missing == []
        assert not result.needs_escalation

    def test_partial_coverage(self):
        """Test partial coverage reports missing assessments."""
        activities = [
            ("Physical Exam", [True, True]),
            ("Vital Signs", [True, True]),
            ("ECG", [True, False]),
        ]
        table = _make_table(activities)
        result = check_completeness(table, "general")

        assert result.found_count == 3
        assert result.coverage_ratio == 3 / len(GENERAL_ASSESSMENTS)
        assert "Hematology" in result.missing
        assert "Chemistry" in result.missing

    def test_zero_coverage_triggers_escalation(self):
        """Test escalation when no assessments match."""
        activities = [
            ("Unknown Assessment A", [True]),
            ("Unknown Assessment B", [False]),
        ]
        table = _make_table(activities)
        result = check_completeness(table, "general", min_coverage=0.5)

        assert result.found_count == 0
        assert result.coverage_ratio == 0.0
        assert result.needs_escalation

    def test_fuzzy_match_abbreviations(self):
        """Test fuzzy matching handles common abbreviations."""
        activities = [
            ("PE", [True]),
            ("Vitals", [True]),
            ("Electrocardiogram", [True]),
        ]
        table = _make_table(activities)
        result = check_completeness(table, "general")

        assert result.found_count >= 3

    def test_fuzzy_match_substring(self):
        """Test fuzzy matching handles substring matches."""
        activities = [
            ("Physical Examination (Complete)", [True, True]),
            ("12-Lead ECG", [True, False]),
        ]
        table = _make_table(activities)
        result = check_completeness(table, "general")

        assert any(m.found and "Physical" in m.expected for m in result.matches)
        assert any(m.found and "ECG" in m.expected for m in result.matches)

    def test_extra_activities_reported(self):
        """Test activities not in reference are listed as extra."""
        activities = [
            ("Physical Exam", [True]),
            ("Custom Biomarker Panel", [True]),
            ("Genomic Sequencing", [True]),
        ]
        table = _make_table(activities)
        result = check_completeness(table, "general")

        assert "Custom Biomarker Panel" in result.extra
        assert "Genomic Sequencing" in result.extra

    def test_oncology_reference(self):
        """Test oncology reference list is larger than general."""
        assert len(ONCOLOGY_PHASE12_ASSESSMENTS) > len(GENERAL_ASSESSMENTS)

    def test_footnote_markers_ignored(self):
        """Test footnote markers don't block matching."""
        activities = [
            ("Physical Exam^2", [True]),
            ("ECG¹", [True]),
        ]
        table = _make_table(activities)
        result = check_completeness(table, "general")

        pe_match = [m for m in result.matches if m.expected == "Physical Exam"]
        assert pe_match[0].found


class TestCheckAlignment:
    """Tests for check_alignment function."""

    def test_all_aligned(self):
        """Test no issues when all rows match header count."""
        table = _make_table(
            activities=[
                ("A", [True, False, True]),
                ("B", [False, True, True]),
            ],
            visit_headers=["V1", "V2", "V3"],
        )
        result = check_alignment(table)

        assert result.misaligned_count == 0
        assert result.misalignment_ratio == 0.0
        assert result.issues == []

    def test_misaligned_detected(self):
        """Test misaligned rows are detected."""
        table = RawSoaTable(
            visit_headers=["V1", "V2", "V3"],
            day_headers=[],
            activities=[
                ("A", [True, False, True]),      # OK: 3
                ("B", [True, False]),             # Bad: 2
                ("C", [True, False, True, True]), # Bad: 4
            ],
            section_code="B.4",
        )
        result = check_alignment(table)

        assert result.misaligned_count == 2
        assert result.header_count == 3
        assert result.issues[0].activity_name == "B"
        assert result.issues[0].actual_cells == 2
        assert result.issues[1].activity_name == "C"
        assert result.issues[1].actual_cells == 4

    def test_empty_table(self):
        """Test empty table has no issues."""
        table = _make_table([], visit_headers=["V1"])
        result = check_alignment(table)

        assert result.total_rows == 0
        assert result.misaligned_count == 0


class TestCrossValidate:
    """Tests for cross_validate combined function."""

    def test_no_escalation_good_table(self):
        """Test no escalation for a well-formed table."""
        activities = [
            (name, [True, False, True])
            for name in GENERAL_ASSESSMENTS
        ]
        table = _make_table(activities, ["Screening", "Day 1", "EOT"])
        result = cross_validate(table, "general")

        assert not result.needs_escalation
        assert result.escalation_reasons == []

    def test_escalation_for_low_coverage(self):
        """Test escalation when coverage is below threshold."""
        table = _make_table(
            [("Unknown", [True])],
            visit_headers=["V1"],
        )
        result = cross_validate(table, "general", min_coverage=0.5)

        assert result.needs_escalation
        assert any("coverage" in r.lower() for r in result.escalation_reasons)

    def test_escalation_for_misalignment(self):
        """Test escalation when too many rows are misaligned."""
        table = RawSoaTable(
            visit_headers=["V1", "V2", "V3"],
            day_headers=[],
            activities=[
                ("A", [True]),       # Misaligned
                ("B", [True]),       # Misaligned
                ("C", [True]),       # Misaligned
                ("D", [True, False, True]),  # OK
            ],
            section_code="B.4",
        )
        result = cross_validate(
            table, "general", max_misalignment=0.2,
        )

        assert result.needs_escalation
        assert any(
            "misalignment" in r.lower()
            for r in result.escalation_reasons
        )

    def test_both_checks_present(self):
        """Test both completeness and alignment are populated."""
        table = _make_table(
            [("Physical Exam", [True, False])],
            visit_headers=["V1", "V2"],
        )
        result = cross_validate(table)

        assert result.completeness is not None
        assert result.alignment is not None
