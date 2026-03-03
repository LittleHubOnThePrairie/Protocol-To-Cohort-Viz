"""Tests for blended table discovery (PTCV-38)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.soa_extractor.table_discovery import (
    TableDiscovery,
    _SCHEDULE_HEADING_RE,
)


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------


class TestScheduleHeadingRegex:
    """Tests for _SCHEDULE_HEADING_RE pattern matching."""

    @pytest.mark.parametrize(
        "text",
        [
            "Schedule of Activities",
            "SCHEDULE OF ASSESSMENTS",
            "Visit Schedule",
            "Study Calendar",
            "study calendar",
            "Study Timeline",
            "Treatment Calendar",
            "Assessment Calendar",
            "Study Procedures",
            "Trial Procedures",
            "Schedule of Visits",
        ],
    )
    def test_matches_schedule_headings(self, text: str) -> None:
        assert _SCHEDULE_HEADING_RE.search(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Informed Consent",
            "Study Design",
            "Primary Endpoint",
            "Randomisation",
        ],
    )
    def test_no_match_for_non_schedule(self, text: str) -> None:
        assert _SCHEDULE_HEADING_RE.search(text) is None


# ---------------------------------------------------------------------------
# Candidate page detection
# ---------------------------------------------------------------------------


class TestFindCandidatePages:
    """Tests for TableDiscovery._find_candidate_pages()."""

    def test_finds_page_with_heading(self) -> None:
        discovery = TableDiscovery()
        blocks = [
            {"page_number": 5, "text": "Schedule of Activities"},
            {"page_number": 6, "text": "Regular text"},
        ]
        pages = discovery._find_candidate_pages(blocks)
        assert 5 in pages

    def test_includes_next_two_pages(self) -> None:
        discovery = TableDiscovery()
        blocks = [
            {"page_number": 10, "text": "Study Calendar"},
        ]
        pages = discovery._find_candidate_pages(blocks)
        assert pages == [10, 11, 12]

    def test_empty_blocks_returns_empty(self) -> None:
        discovery = TableDiscovery()
        pages = discovery._find_candidate_pages([])
        assert pages == []

    def test_no_headings_returns_empty(self) -> None:
        discovery = TableDiscovery()
        blocks = [
            {"page_number": 1, "text": "Introduction"},
            {"page_number": 2, "text": "Background"},
        ]
        pages = discovery._find_candidate_pages(blocks)
        assert pages == []

    def test_deduplicated_and_sorted(self) -> None:
        discovery = TableDiscovery()
        blocks = [
            {"page_number": 5, "text": "Schedule of Activities"},
            {"page_number": 6, "text": "Visit Schedule"},
        ]
        pages = discovery._find_candidate_pages(blocks)
        # 5,6,7 from first + 6,7,8 from second → 5,6,7,8
        assert pages == [5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Header similarity
# ---------------------------------------------------------------------------


class TestHeaderSimilarity:
    """Tests for TableDiscovery._header_similarity()."""

    def test_identical_headers(self) -> None:
        sim = TableDiscovery._header_similarity(
            ["Visit", "Day 1", "Week 2"],
            ["Visit", "Day 1", "Week 2"],
        )
        assert sim == 1.0

    def test_completely_different(self) -> None:
        sim = TableDiscovery._header_similarity(
            ["Visit", "Day 1"],
            ["Procedure", "Category"],
        )
        assert sim == 0.0

    def test_partial_overlap(self) -> None:
        sim = TableDiscovery._header_similarity(
            ["Visit", "Day 1", "Week 2"],
            ["Visit", "Day 1", "Month 1"],
        )
        # Jaccard: {visit, day 1} / {visit, day 1, week 2, month 1} = 2/4
        assert sim == pytest.approx(0.5)

    def test_empty_headers_both(self) -> None:
        assert TableDiscovery._header_similarity([], []) == 1.0

    def test_one_empty(self) -> None:
        assert TableDiscovery._header_similarity(["Visit"], []) == 0.0

    def test_case_insensitive(self) -> None:
        sim = TableDiscovery._header_similarity(
            ["VISIT", "DAY 1"],
            ["visit", "day 1"],
        )
        assert sim == 1.0


# ---------------------------------------------------------------------------
# Multi-page table merging
# ---------------------------------------------------------------------------


class TestMergeMultiPage:
    """Tests for TableDiscovery._merge_multi_page()."""

    def test_merges_consecutive_same_header(self) -> None:
        discovery = TableDiscovery(min_header_similarity=0.80)
        tables = [
            {
                "page": 36,
                "header": ["Activity", "Screening", "Day 1"],
                "rows": [["ECG", "X", "X"]],
                "extractor": "camelot_stream",
                "cell_count": 6,
            },
            {
                "page": 37,
                "header": ["Activity", "Screening", "Day 1"],
                "rows": [["Labs", "X", ""]],
                "extractor": "camelot_stream",
                "cell_count": 6,
            },
        ]
        merged = discovery._merge_multi_page(tables)
        assert len(merged) == 1
        assert len(merged[0]["rows"]) == 2

    def test_no_merge_different_headers(self) -> None:
        discovery = TableDiscovery(min_header_similarity=0.80)
        tables = [
            {
                "page": 5,
                "header": ["Activity", "Screening"],
                "rows": [["ECG", "X"]],
                "extractor": "camelot_stream",
                "cell_count": 4,
            },
            {
                "page": 6,
                "header": ["Procedure", "Category"],
                "rows": [["Blood draw", "Lab"]],
                "extractor": "camelot_stream",
                "cell_count": 4,
            },
        ]
        merged = discovery._merge_multi_page(tables)
        assert len(merged) == 2

    def test_no_merge_distant_pages(self) -> None:
        discovery = TableDiscovery(min_header_similarity=0.80)
        tables = [
            {
                "page": 5,
                "header": ["Activity", "Screening", "Day 1"],
                "rows": [["ECG", "X", "X"]],
                "extractor": "camelot_stream",
                "cell_count": 6,
            },
            {
                "page": 20,
                "header": ["Activity", "Screening", "Day 1"],
                "rows": [["Labs", "X", ""]],
                "extractor": "camelot_stream",
                "cell_count": 6,
            },
        ]
        merged = discovery._merge_multi_page(tables)
        assert len(merged) == 2

    def test_empty_list(self) -> None:
        discovery = TableDiscovery()
        assert discovery._merge_multi_page([]) == []

    def test_three_page_merge(self) -> None:
        discovery = TableDiscovery(min_header_similarity=0.80)
        header = ["Activity", "Visit 1", "Visit 2"]
        tables = [
            {"page": 10, "header": header, "rows": [["ECG", "X", ""]],
             "extractor": "camelot_stream", "cell_count": 6},
            {"page": 11, "header": header, "rows": [["Labs", "", "X"]],
             "extractor": "camelot_stream", "cell_count": 6},
            {"page": 12, "header": header, "rows": [["Vitals", "X", "X"]],
             "extractor": "camelot_stream", "cell_count": 6},
        ]
        merged = discovery._merge_multi_page(tables)
        assert len(merged) == 1
        assert len(merged[0]["rows"]) == 3


# ---------------------------------------------------------------------------
# RawSoaTable conversion
# ---------------------------------------------------------------------------


class TestToRawSoaTables:
    """Tests for TableDiscovery._to_raw_soa_tables()."""

    def test_basic_conversion(self) -> None:
        discovery = TableDiscovery()
        tables = [{
            "page": 5,
            "header": ["Assessment", "Screening", "Baseline", "Week 2"],
            "rows": [
                ["ECG", "X", "", "X"],
                ["Labs", "X", "X", "X"],
            ],
            "extractor": "camelot_stream",
            "cell_count": 12,
        }]
        results = discovery._to_raw_soa_tables(tables)
        assert len(results) == 1
        raw = results[0]
        assert raw.visit_headers == ["Screening", "Baseline", "Week 2"]
        assert len(raw.activities) == 2

    def test_day_row_detected(self) -> None:
        discovery = TableDiscovery()
        tables = [{
            "page": 5,
            "header": ["Assessment", "Screening", "Baseline"],
            "rows": [
                ["Day", "-14 to -1", "1"],
                ["ECG", "X", "X"],
            ],
            "extractor": "camelot_stream",
            "cell_count": 9,
        }]
        results = discovery._to_raw_soa_tables(tables)
        assert len(results) == 1
        assert len(results[0].day_headers) == 2
        assert len(results[0].activities) == 1

    def test_scheduled_flags_correct(self) -> None:
        discovery = TableDiscovery()
        tables = [{
            "page": 5,
            "header": ["Activity", "V1", "V2", "V3"],
            "rows": [
                ["ECG", "X", "", "x"],
                ["Labs", "", "Yes", ""],
            ],
            "extractor": "camelot_stream",
            "cell_count": 12,
        }]
        results = discovery._to_raw_soa_tables(tables)
        ecg = results[0].activities[0]
        assert ecg[0] == "ECG"
        assert ecg[1] == [True, False, True]

        labs = results[0].activities[1]
        assert labs[1] == [False, True, False]

    def test_skips_too_few_columns(self) -> None:
        discovery = TableDiscovery()
        tables = [{
            "page": 5,
            "header": ["Activity"],
            "rows": [["ECG"]],
            "extractor": "camelot_stream",
            "cell_count": 2,
        }]
        assert discovery._to_raw_soa_tables(tables) == []

    def test_skips_no_activities(self) -> None:
        discovery = TableDiscovery()
        tables = [{
            "page": 5,
            "header": ["Activity", "V1", "V2"],
            "rows": [["", "", ""]],
            "extractor": "camelot_stream",
            "cell_count": 6,
        }]
        assert discovery._to_raw_soa_tables(tables) == []

    def test_section_code_is_b4(self) -> None:
        discovery = TableDiscovery()
        tables = [{
            "page": 5,
            "header": ["Activity", "V1", "V2"],
            "rows": [["ECG", "X", ""]],
            "extractor": "camelot_stream",
            "cell_count": 6,
        }]
        results = discovery._to_raw_soa_tables(tables)
        assert results[0].section_code == "B.4"

    def test_flags_padded_to_header_length(self) -> None:
        discovery = TableDiscovery()
        tables = [{
            "page": 5,
            "header": ["Activity", "V1", "V2", "V3"],
            "rows": [["ECG", "X"]],
            "extractor": "camelot_stream",
            "cell_count": 8,
        }]
        results = discovery._to_raw_soa_tables(tables)
        flags = results[0].activities[0][1]
        assert len(flags) == 3
        assert flags == [True, False, False]


# ---------------------------------------------------------------------------
# Integration: discover() with mocked Camelot
# ---------------------------------------------------------------------------


class TestDiscoverWithMockedCamelot:
    """Tests for the full discover() flow with mocked Camelot."""

    def _make_mock_camelot_table(
        self, header: list[str], rows: list[list[str]]
    ) -> MagicMock:
        """Build a mock Camelot table with a .df attribute."""
        import pandas as pd

        df = pd.DataFrame([header] + rows)
        mock_tbl = MagicMock()
        mock_tbl.df = df
        return mock_tbl

    def test_discover_with_camelot_results(self) -> None:
        discovery = TableDiscovery()
        blocks = [
            {"page_number": 5, "text": "Schedule of Activities"},
        ]

        mock_tbl = self._make_mock_camelot_table(
            ["Assessment", "Screening", "Day 1"],
            [["ECG", "X", "X"], ["Labs", "X", ""]],
        )

        # Patch _camelot_stream_extract to return pre-built raw tables
        discovery._camelot_stream_extract = MagicMock(  # type: ignore[method-assign]
            return_value=[{
                "page": 5,
                "header": ["Assessment", "Screening", "Day 1"],
                "rows": [["ECG", "X", "X"], ["Labs", "X", ""]],
                "extractor": "camelot_stream",
                "cell_count": 9,
            }]
        )
        results = discovery.discover(
            pdf_bytes=b"%PDF-fake",
            text_blocks=blocks,
            page_count=10,
        )

        assert len(results) >= 1
        assert results[0].visit_headers == ["Screening", "Day 1"]
        assert len(results[0].activities) == 2

    def test_discover_no_candidates_returns_empty(self) -> None:
        discovery = TableDiscovery()
        blocks = [
            {"page_number": 1, "text": "Introduction"},
        ]
        results = discovery.discover(
            pdf_bytes=b"%PDF-fake",
            text_blocks=blocks,
            page_count=5,
        )
        assert results == []

    def test_discover_camelot_empty_triggers_tatr_fallback(self) -> None:
        discovery = TableDiscovery()
        blocks = [
            {"page_number": 5, "text": "Study Calendar"},
        ]

        # Mock camelot returning empty, TATR also not installed
        discovery._camelot_stream_extract = MagicMock(  # type: ignore[method-assign]
            return_value=[]
        )

        results = discovery.discover(
            pdf_bytes=b"%PDF-fake",
            text_blocks=blocks,
            page_count=10,
        )
        assert results == []


# ---------------------------------------------------------------------------
# Lazy import: Table Transformer
# ---------------------------------------------------------------------------


class TestTatrLazyImport:
    """Tests that Table Transformer is a lazy optional dependency."""

    def test_no_import_error_without_tatr(self) -> None:
        """System works without table-transformer installed."""
        discovery = TableDiscovery()
        results = discovery._tatr_extract(b"%PDF-fake", [5, 6])
        assert results == []

    def test_camelot_import_error_handled(self) -> None:
        """Camelot import error is handled gracefully."""
        discovery = TableDiscovery()
        with patch.dict("sys.modules", {"camelot": None}):
            results = discovery._camelot_stream_extract(
                b"%PDF-fake", [5]
            )
            assert results == []


# ---------------------------------------------------------------------------
# Keyword expansion (Part A)
# ---------------------------------------------------------------------------


class TestSoaKeywordExpansion:
    """Verify expanded _SOA_KEYWORDS in parser.py."""

    def test_study_calendar_in_parser_keywords(self) -> None:
        from ptcv.soa_extractor.parser import _SOA_KEYWORDS

        assert _SOA_KEYWORDS.search("Study Calendar") is not None

    def test_study_timeline_in_parser_keywords(self) -> None:
        from ptcv.soa_extractor.parser import _SOA_KEYWORDS

        assert _SOA_KEYWORDS.search("Study Timeline") is not None

    def test_treatment_calendar_in_parser_keywords(self) -> None:
        from ptcv.soa_extractor.parser import _SOA_KEYWORDS

        assert _SOA_KEYWORDS.search("Treatment Calendar") is not None

    def test_assessment_calendar_in_parser_keywords(self) -> None:
        from ptcv.soa_extractor.parser import _SOA_KEYWORDS

        assert _SOA_KEYWORDS.search("Assessment Calendar") is not None

    def test_original_keywords_still_match(self) -> None:
        from ptcv.soa_extractor.parser import _SOA_KEYWORDS

        assert _SOA_KEYWORDS.search("Schedule of Activities") is not None
        assert _SOA_KEYWORDS.search("Visit Schedule") is not None


class TestClassifierB4Expansion:
    """Verify expanded B.4 patterns in classifier.py."""

    def test_study_calendar_in_b4_patterns(self) -> None:
        from ptcv.ich_parser.classifier import _ICH_SECTIONS

        patterns = _ICH_SECTIONS["B.4"]["patterns"]
        import re

        combined = "|".join(patterns)
        assert re.search(combined, "study calendar", re.IGNORECASE)

    def test_schedule_of_activities_in_b4_patterns(self) -> None:
        from ptcv.ich_parser.classifier import _ICH_SECTIONS

        patterns = _ICH_SECTIONS["B.4"]["patterns"]
        import re

        combined = "|".join(patterns)
        assert re.search(
            combined, "schedule of activities", re.IGNORECASE
        )

    def test_study_calendar_in_b4_keywords(self) -> None:
        from ptcv.ich_parser.classifier import _ICH_SECTIONS

        keywords = _ICH_SECTIONS["B.4"]["keywords"]
        assert "study calendar" in keywords

    def test_schedule_of_activities_in_b4_keywords(self) -> None:
        from ptcv.ich_parser.classifier import _ICH_SECTIONS

        keywords = _ICH_SECTIONS["B.4"]["keywords"]
        assert "schedule of activities" in keywords
