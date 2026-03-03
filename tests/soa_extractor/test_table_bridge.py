"""Unit tests for table_bridge (PTCV-51).

Tests the ExtractedTable-to-RawSoaTable bridge that enables the SoA
pipeline to consume pre-extracted tables from tables.parquet, covering
all 5 GHERKIN acceptance criteria.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.extraction.models import ExtractedTable
from ptcv.soa_extractor.table_bridge import (
    extracted_table_to_raw_soa,
    filter_soa_tables,
    is_soa_table,
)


def _make_extracted_table(
    header: list[str],
    rows: list[list[str]],
    page_number: int = 1,
    table_index: int = 0,
) -> ExtractedTable:
    """Helper to build an ExtractedTable with JSON-serialised fields."""
    return ExtractedTable(
        run_id="run-001",
        source_registry_id="NCT00112827",
        source_sha256="a" * 64,
        page_number=page_number,
        extractor_used="pdfplumber",
        table_index=table_index,
        header_row=json.dumps(header),
        data_rows=json.dumps(rows),
    )


# -----------------------------------------------------------------------
# A realistic SoA table header and data
# -----------------------------------------------------------------------
SOA_HEADER = [
    "Assessment",
    "Screening\n(Day -30 to -1)",
    "Admission\n(Day -2)",
    "Baseline\n(Day 1)",
    "Week 2",
    "Week 4\n(Day 29)",
    "Follow-up\n(Day 57)",
    "End of Study",
]

SOA_ROWS = [
    ["Day", "-30 to -1", "-2", "1", "15 ± 3", "29 ± 3", "57 ± 3", ""],
    ["Informed Consent", "X", "", "", "", "", "", ""],
    ["Physical Exam", "X", "", "X", "X", "X", "", "X"],
    ["ECG", "X", "", "", "X", "", "", "X"],
    ["Lab Tests", "X", "", "X", "X", "X", "X", "X"],
    ["Vital Signs", "X", "X", "X", "X", "X", "X", "X"],
    ["ECOG Performance", "X", "", "X", "", "", "", ""],
]

# A non-SoA table (abbreviations list)
NON_SOA_HEADER = ["Abbreviation", "Definition"]
NON_SOA_ROWS = [
    ["AE", "Adverse Event"],
    ["SAE", "Serious Adverse Event"],
    ["ECG", "Electrocardiogram"],
]


class TestIsSoaTable:
    """Tests for is_soa_table header matching."""

    def test_soa_header_matches(self) -> None:
        assert is_soa_table(SOA_HEADER) is True

    def test_non_soa_header_rejected(self) -> None:
        assert is_soa_table(NON_SOA_HEADER) is False

    def test_too_few_columns_rejected(self) -> None:
        assert is_soa_table(["A", "B", "C"]) is False

    def test_minimal_soa_header(self) -> None:
        header = ["Activity", "Screening", "Baseline", "Day 1", "Week 2", "Follow-up"]
        assert is_soa_table(header) is True

    def test_visit_number_headers(self) -> None:
        header = ["Procedure", "Visit 1", "Visit 2", "Visit 3", "Visit 4", "Visit 5"]
        assert is_soa_table(header) is True


class TestExtractedTableToRawSoa:
    """Scenario: Pre-extracted tables with visit columns are identified."""

    def test_soa_table_converts(self) -> None:
        """Given a table with visit columns, it converts to RawSoaTable."""
        table = _make_extracted_table(SOA_HEADER, SOA_ROWS)
        raw = extracted_table_to_raw_soa(table)
        assert raw is not None
        assert len(raw.visit_headers) == 7  # 8 cols minus activity label
        assert raw.section_code == "tables.parquet"
        assert len(raw.activities) > 0

    def test_activities_have_correct_flags(self) -> None:
        table = _make_extracted_table(SOA_HEADER, SOA_ROWS)
        raw = extracted_table_to_raw_soa(table)
        assert raw is not None
        # "Informed Consent" should be X at Screening only
        consent = next(a for a in raw.activities if "Consent" in a[0])
        assert consent[1][0] is True  # Screening
        assert consent[1][1] is False  # Admission

    def test_day_row_detected(self) -> None:
        """Day/window sub-header row is separated from activities."""
        table = _make_extracted_table(SOA_HEADER, SOA_ROWS)
        raw = extracted_table_to_raw_soa(table)
        assert raw is not None
        assert len(raw.day_headers) > 0
        # First activity should not be the day row
        assert "Consent" in raw.activities[0][0] or "Exam" in raw.activities[0][0]

    def test_non_soa_table_returns_none(self) -> None:
        table = _make_extracted_table(NON_SOA_HEADER, NON_SOA_ROWS)
        raw = extracted_table_to_raw_soa(table)
        assert raw is None

    def test_too_few_data_rows_returns_none(self) -> None:
        """Tables with < 3 data rows are rejected."""
        table = _make_extracted_table(SOA_HEADER, SOA_ROWS[:1])
        raw = extracted_table_to_raw_soa(table)
        assert raw is None

    def test_invalid_json_returns_none(self) -> None:
        table = ExtractedTable(
            run_id="run-001",
            source_registry_id="NCT_TEST",
            source_sha256="a" * 64,
            page_number=1,
            extractor_used="pdfplumber",
            table_index=0,
            header_row="not valid json",
            data_rows="also not json",
        )
        raw = extracted_table_to_raw_soa(table)
        assert raw is None

    def test_multiline_cell_values_cleaned(self) -> None:
        """Newlines in cell values are collapsed to spaces."""
        table = _make_extracted_table(SOA_HEADER, SOA_ROWS)
        raw = extracted_table_to_raw_soa(table)
        assert raw is not None
        # "Screening\n(Day -30 to -1)" should become "Screening (Day -30 to -1)"
        assert "\n" not in raw.visit_headers[0]


class TestFilterSoaTables:
    """Scenario: Pre-extracted tables take priority over text parsing."""

    def test_filters_soa_from_mixed_list(self) -> None:
        soa = _make_extracted_table(SOA_HEADER, SOA_ROWS, table_index=0)
        non_soa = _make_extracted_table(NON_SOA_HEADER, NON_SOA_ROWS, table_index=1)
        results = filter_soa_tables([soa, non_soa])
        assert len(results) == 1
        assert results[0].section_code == "tables.parquet"

    def test_empty_list_returns_empty(self) -> None:
        assert filter_soa_tables([]) == []

    def test_no_soa_tables_returns_empty(self) -> None:
        non_soa = _make_extracted_table(NON_SOA_HEADER, NON_SOA_ROWS)
        assert filter_soa_tables([non_soa]) == []

    def test_multiple_soa_tables_all_returned(self) -> None:
        soa1 = _make_extracted_table(SOA_HEADER, SOA_ROWS, page_number=5)
        soa2 = _make_extracted_table(SOA_HEADER, SOA_ROWS, page_number=12)
        results = filter_soa_tables([soa1, soa2])
        assert len(results) == 2


class TestMultiPageSoaTable:
    """Scenario: Multi-page SoA tables are handled correctly."""

    def test_merged_continuation_rows_preserved(self) -> None:
        """PdfExtractor merges multi-page tables; bridge preserves all rows."""
        # Simulate a merged table: original + continuation rows
        extended_rows = SOA_ROWS + [
            ["Chest X-ray", "X", "", "", "", "", "", "X"],
            ["CT Scan", "", "", "X", "", "X", "", "X"],
            ["Adverse Events", "", "X", "X", "X", "X", "X", "X"],
        ]
        table = _make_extracted_table(SOA_HEADER, extended_rows)
        raw = extracted_table_to_raw_soa(table)
        assert raw is not None
        # Day row is excluded from activities, so count = total - 1
        assert len(raw.activities) == len(extended_rows) - 1


class TestBackwardCompatibility:
    """Scenario: Backward compatibility with text-based parsing."""

    def test_extractor_accepts_none_extracted_tables(self) -> None:
        """SoaExtractor.extract() works without extracted_tables param."""
        from ptcv.ich_parser.models import IchSection
        from ptcv.soa_extractor.extractor import SoaExtractor
        from ptcv.storage import FilesystemAdapter

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gw = FilesystemAdapter(root=Path(tmp))
            gw.initialise()
            extractor = SoaExtractor(gateway=gw)

            sections = [
                IchSection(
                    run_id="test-run",
                    source_run_id="",
                    source_sha256="a" * 64,
                    registry_id="NCT_TEST",
                    section_code="B.4",
                    section_name="Trial Design",
                    content_json=json.dumps({"text": "No SoA table here."}),
                    confidence_score=0.9,
                    review_required=False,
                    legacy_format=False,
                    extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
                ),
            ]

            # Should not raise — extracted_tables defaults to None
            result = extractor.extract(
                sections=sections,
                registry_id="NCT_TEST",
            )
            assert result.run_id  # valid run_id produced
            assert result.activity_count == 0  # no SoA found

    def test_extractor_prefers_extracted_tables(self) -> None:
        """When extracted_tables has SoA, text parsing is skipped."""
        from ptcv.ich_parser.models import IchSection
        from ptcv.soa_extractor.extractor import SoaExtractor
        from ptcv.storage import FilesystemAdapter

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gw = FilesystemAdapter(root=Path(tmp))
            gw.initialise()
            extractor = SoaExtractor(gateway=gw)

            # Section has no SoA table text
            sections = [
                IchSection(
                    run_id="test-run",
                    source_run_id="",
                    source_sha256="a" * 64,
                    registry_id="NCT_TEST",
                    section_code="B.4",
                    section_name="Trial Design",
                    content_json=json.dumps({"text": "No SoA table here."}),
                    confidence_score=0.9,
                    review_required=False,
                    legacy_format=False,
                    extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
                ),
            ]

            # But extracted_tables has a real SoA
            soa_table = _make_extracted_table(SOA_HEADER, SOA_ROWS)

            result = extractor.extract(
                sections=sections,
                registry_id="NCT_TEST",
                extracted_tables=[soa_table],
            )
            # SoA was found via the bridge (GHERKIN Scenario 3)
            assert result.activity_count > 0
            assert result.timepoint_count > 0
            assert result.epoch_count > 0
