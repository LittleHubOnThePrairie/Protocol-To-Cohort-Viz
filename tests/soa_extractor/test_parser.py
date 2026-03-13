"""Tests for SoaTableParser."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.ich_parser.models import IchSection
from ptcv.soa_extractor.parser import SoaTableParser


def make_section(text: str, code: str = "B.4") -> IchSection:
    return IchSection(
        run_id="r1",
        source_run_id="",
        source_sha256="a" * 64,
        registry_id="NCT0001",
        section_code=code,
        section_name="Design",
        content_json=json.dumps({"text": text}),
        confidence_score=0.9,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
    )


@pytest.fixture
def parser():
    return SoaTableParser()


class TestMarkdownTableParsing:
    SIMPLE_TABLE = """\
| Assessment | Screening | Baseline | Week 2 |
|------------|-----------|----------|--------|
| ECG        | X         |          | X      |
| Labs       | X         | X        | X      |
"""

    def test_finds_table(self, parser):
        table = parser.parse([make_section(self.SIMPLE_TABLE)])
        assert table is not None

    def test_visit_headers_extracted(self, parser):
        table = parser.parse([make_section(self.SIMPLE_TABLE)])
        assert table is not None
        assert "Screening" in table.visit_headers
        assert "Baseline" in table.visit_headers
        assert "Week 2" in table.visit_headers

    def test_activities_extracted(self, parser):
        table = parser.parse([make_section(self.SIMPLE_TABLE)])
        assert table is not None
        names = [name for name, _ in table.activities]
        assert "ECG" in names
        assert "Labs" in names

    def test_scheduled_flags_correct(self, parser):
        table = parser.parse([make_section(self.SIMPLE_TABLE)])
        assert table is not None
        ecg_flags = next(flags for name, flags in table.activities if name == "ECG")
        # Screening=X, Baseline=empty, Week2=X
        assert ecg_flags[0] is True
        assert ecg_flags[1] is False
        assert ecg_flags[2] is True

    def test_day_row_detected(self, parser):
        text = """\
| Assessment | Screening | Baseline |
| Day        | -14 to -1 | 1        |
|------------|-----------|----------|
| ECG        | X         | X        |
"""
        table = parser.parse([make_section(text)])
        assert table is not None
        assert len(table.day_headers) > 0

    def test_returns_none_for_no_table(self, parser):
        text = "This protocol has no schedule table. Just text."
        table = parser.parse([make_section(text)])
        assert table is None

    def test_section_code_preserved(self, parser):
        table = parser.parse([make_section(self.SIMPLE_TABLE, code="B.7")])
        assert table is not None
        assert table.section_code == "B.7"

    def test_priority_ordering(self, parser):
        """B.4 section is tried before B.2."""
        b2 = make_section("No table here", code="B.2")
        b4 = make_section(self.SIMPLE_TABLE, code="B.4")
        table = parser.parse([b2, b4])
        assert table is not None
        assert table.section_code == "B.4"


class TestPTCV128RejectionGuards:
    """Regression tests for PTCV-128: prose text misclassified as SoA."""

    MMRM_TEXT = (
        "VISIT SCHEDULE\n\n"
        "The visit schedule is described in section 8.2.\n\n"
        "1. MMRM will be used for the analysis for the change of UACR\n"
        "2. Descriptive statistics will be provided for the change of UACR\n"
        "3. The primary endpoint is the change from baseline\n"
        "4. Safety will be assessed at each study visit\n"
        "5. Adverse events will be monitored throughout the trial\n"
    )

    def test_aligned_parser_rejects_prose_with_soa_keyword(self, parser):
        """Prose containing 'visit schedule' should not become a table."""
        section = make_section(self.MMRM_TEXT, code="B.8")
        table = parser.parse([section])
        assert table is None

    def test_pipe_table_rejects_no_markers(self, parser):
        """Pipe table with no X/✓ markers should be rejected."""
        text = """\
| Category | Description | Notes |
|----------|-------------|-------|
| Stat     | MMRM model  | primary |
| Stat     | Descriptive | secondary |
"""
        section = make_section(text, code="B.10")
        table = parser.parse([section])
        assert table is None

    def test_aligned_parser_accepts_valid_visit_headers(self, parser):
        """Headers matching visit patterns (Day1, Week2) should parse."""
        # SoA keyword on a short line so it doesn't become the header
        text = (
            "Visit schedule\n"
            "Activity Day1 Week2 Week4\n"
            "ECG X X X\n"
            "Labs X X X\n"
        )
        section = make_section(text, code="B.8")
        table = parser.parse([section])
        assert table is not None
        assert "Day1" in table.visit_headers or "Week2" in table.visit_headers

    def test_aligned_parser_rejects_english_word_headers(self, parser):
        """Common English words as headers must be rejected."""
        text = (
            "Assessment schedule for the trial.\n\n"
            "Analysis will be performed using MMRM\n"
            "Statistics will be provided for endpoints\n"
            "Results will be summarised in tables\n"
        )
        section = make_section(text, code="B.8")
        table = parser.parse([section])
        assert table is None


class TestAlignedTableHeaderValidation:
    """PTCV-151: Stricter header validation for aligned-table parser."""

    def test_rejects_single_visit_header(self, parser):
        """Text with only 1 visit-like token in headers → None."""
        text = (
            "Schedule of activities for this study.\n\n"
            "Assessment Screening B double-blind, placebo\n"
            "ECG X X X\n"
            "Labs X X X\n"
        )
        section = make_section(text, code="B.4")
        table = parser.parse([section])
        assert table is None

    def test_requires_minimum_three_visit_columns(self, parser):
        """Fewer than 3 visit columns → None (even if all are visit-like)."""
        text = (
            "Visit schedule\n\n"
            "Assessment Screening Baseline\n"
            "ECG X X\n"
            "Labs X X\n"
        )
        section = make_section(text, code="B.4")
        table = parser.parse([section])
        assert table is None

    def test_accepts_three_valid_visit_headers(self, parser):
        """Three visit-like headers should be accepted."""
        text = (
            "Visit schedule\n\n"
            "Assessment Screening Baseline Week4\n"
            "ECG X X X\n"
            "Labs X X X\n"
        )
        section = make_section(text, code="B.4")
        table = parser.parse([section])
        assert table is not None
        assert len(table.visit_headers) == 3

    def test_rejects_mostly_prose_headers(self, parser):
        """Headers that are mostly non-visit words → None."""
        text = (
            "Study procedures summary\n\n"
            "Method Statistical Primary Secondary Screening Endpoint\n"
            "Analysis X X X X X\n"
            "Review X X X X X\n"
        )
        section = make_section(text, code="B.8")
        table = parser.parse([section])
        assert table is None


class TestSampleSoaFromConftest:
    def test_full_sample_parses(self, parser, sample_ich_sections):
        table = parser.parse(sample_ich_sections)
        assert table is not None
        assert len(table.visit_headers) >= 4
        assert len(table.activities) >= 4

    def test_unscheduled_column_present(self, parser, sample_ich_sections):
        table = parser.parse(sample_ich_sections)
        assert table is not None
        headers_lower = [h.lower() for h in table.visit_headers]
        assert any("unscheduled" in h for h in headers_lower)

    def test_early_termination_column_present(self, parser, sample_ich_sections):
        table = parser.parse(sample_ich_sections)
        assert table is not None
        headers_lower = [h.lower() for h in table.visit_headers]
        assert any("early" in h or "termination" in h for h in headers_lower)
