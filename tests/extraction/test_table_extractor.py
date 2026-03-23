"""Tests for universal table extraction (PTCV-223).

Tests Camelot/pdfplumber extraction, multi-method merge, multi-page
continuity, and ExtractedTable output format.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from ptcv.extraction.table_extractor import (
    _merge_methods,
    _merge_multi_page,
    _header_similarity,
    _is_continuation,
    extract_all_tables,
)
from ptcv.extraction.models import ExtractedTable


class TestMergeMethods:
    """Tests for _merge_methods best-per-page selection."""

    def test_single_method(self):
        """Test single method results pass through."""
        tables = [
            {"page": 1, "header": ["A"], "rows": [["1"]], "extractor": "stream", "cell_count": 10},
        ]
        result = _merge_methods(tables)
        assert len(result) == 1

    def test_best_per_page_kept(self):
        """Test highest cell count per page wins."""
        stream = [
            {"page": 1, "header": ["A", "B"], "rows": [["1", "2"]], "extractor": "stream", "cell_count": 4},
        ]
        lattice = [
            {"page": 1, "header": ["A", "B", "C"], "rows": [["1", "2", "3"], ["4", "5", "6"]], "extractor": "lattice", "cell_count": 9},
        ]
        result = _merge_methods(stream, lattice)
        assert len(result) == 1
        assert result[0]["extractor"] == "lattice"
        assert result[0]["cell_count"] == 9

    def test_different_pages_both_kept(self):
        """Test tables from different pages are both kept."""
        a = [{"page": 1, "header": ["A"], "rows": [], "extractor": "s", "cell_count": 10}]
        b = [{"page": 5, "header": ["B"], "rows": [], "extractor": "l", "cell_count": 8}]
        result = _merge_methods(a, b)
        assert len(result) == 2

    def test_empty_inputs(self):
        """Test empty inputs return empty."""
        assert _merge_methods([], [], []) == []

    def test_sorted_by_page(self):
        """Test output sorted by page number."""
        a = [{"page": 5, "header": ["A"], "rows": [], "extractor": "s", "cell_count": 10}]
        b = [{"page": 1, "header": ["B"], "rows": [], "extractor": "l", "cell_count": 8}]
        result = _merge_methods(a, b)
        assert result[0]["page"] == 1
        assert result[1]["page"] == 5


class TestHeaderSimilarity:
    """Tests for Jaccard header similarity."""

    def test_identical(self):
        assert _header_similarity(["A", "B"], ["A", "B"]) == 1.0

    def test_disjoint(self):
        assert _header_similarity(["A", "B"], ["C", "D"]) == 0.0

    def test_partial_overlap(self):
        sim = _header_similarity(["A", "B", "C"], ["A", "B", "D"])
        assert 0.4 < sim < 0.7

    def test_both_empty(self):
        assert _header_similarity([], []) == 1.0


class TestIsContinuation:
    """Tests for continuation signal detection."""

    def test_continued_text(self):
        prev = {"page": 1, "header": ["A", "B", "C", "D", "E"], "rows": []}
        cont = {"page": 2, "header": ["Table 11 (continued)", "B", "C", "D", "E"], "rows": []}
        assert _is_continuation(prev, cont)

    def test_same_high_col_count(self):
        prev = {"page": 1, "header": ["A", "B", "C", "D", "E", "F"], "rows": []}
        cont = {"page": 2, "header": ["G", "H", "I", "J", "K", "L"], "rows": []}
        assert _is_continuation(prev, cont)

    def test_different_col_count_no_signal(self):
        prev = {"page": 1, "header": ["A", "B"], "rows": []}
        cont = {"page": 2, "header": ["C", "D", "E", "F"], "rows": []}
        assert not _is_continuation(prev, cont)


class TestMergeMultiPage:
    """Tests for multi-page table merging."""

    def test_merge_consecutive_similar_headers(self):
        tables = [
            {"page": 1, "header": ["A", "B"], "rows": [["1", "2"]], "extractor": "s", "cell_count": 4},
            {"page": 2, "header": ["A", "B"], "rows": [["3", "4"]], "extractor": "s", "cell_count": 4},
        ]
        result = _merge_multi_page(tables)
        assert len(result) == 1
        assert len(result[0]["rows"]) == 2

    def test_no_merge_different_headers(self):
        tables = [
            {"page": 1, "header": ["A", "B"], "rows": [["1", "2"]], "extractor": "s", "cell_count": 4},
            {"page": 2, "header": ["X", "Y"], "rows": [["3", "4"]], "extractor": "s", "cell_count": 4},
        ]
        result = _merge_multi_page(tables)
        assert len(result) == 2

    def test_skip_repeated_header_on_continuation(self):
        tables = [
            {"page": 1, "header": ["A", "B", "C", "D", "E"], "rows": [["1", "2", "3", "4", "5"]], "extractor": "s", "cell_count": 10},
            {"page": 2, "header": ["A", "B", "C", "D", "E"], "rows": [["A", "B", "C", "D", "E"], ["6", "7", "8", "9", "10"]], "extractor": "s", "cell_count": 15},
        ]
        result = _merge_multi_page(tables)
        assert len(result) == 1
        # Should have 2 data rows (original + continuation minus repeated header)
        assert len(result[0]["rows"]) == 2

    def test_empty_input(self):
        assert _merge_multi_page([]) == []


class TestExtractAllTables:
    """Tests for extract_all_tables main function."""

    def test_with_existing_tables_only(self):
        """Test passthrough when no Camelot/pdfplumber available."""
        existing = [
            ExtractedTable(
                run_id="r1",
                source_registry_id="NCT00001",
                source_sha256="a" * 64,
                page_number=1,
                extractor_used="pymupdf4llm",
                table_index=0,
                header_row=json.dumps(["Col1", "Col2", "Col3"]),
                data_rows=json.dumps([["a", "b", "c"], ["d", "e", "f"]]),
            ),
        ]

        with patch(
            "ptcv.extraction.table_extractor._camelot_extract",
            return_value=[],
        ), patch(
            "ptcv.extraction.table_extractor._pdfplumber_extract",
            return_value=[],
        ):
            result = extract_all_tables(
                pdf_bytes=b"%PDF-1.4 stub",
                page_count=5,
                run_id="r1",
                registry_id="NCT00001",
                source_sha256="a" * 64,
                existing_tables=existing,
            )

        assert len(result) == 1
        assert result[0].extractor_used == "pymupdf4llm"

    def test_camelot_supplements_existing(self):
        """Test Camelot finds tables on pages where pymupdf4llm didn't."""
        existing = [
            ExtractedTable(
                run_id="r1",
                source_registry_id="NCT00001",
                source_sha256="a" * 64,
                page_number=1,
                extractor_used="pymupdf4llm",
                table_index=0,
                header_row=json.dumps(["A", "B", "C"]),
                data_rows=json.dumps([["1", "2", "3"], ["4", "5", "6"]]),
            ),
        ]
        plumber_result = [
            {
                "page": 5,
                "header": ["X", "Y", "Z"],
                "rows": [["a", "b", "c"], ["d", "e", "f"]],
                "extractor": "pdfplumber_table",
                "cell_count": 9,
            },
        ]

        with patch(
            "ptcv.extraction.table_extractor._camelot_extract",
            return_value=[],
        ), patch(
            "ptcv.extraction.table_extractor._pdfplumber_extract",
            return_value=plumber_result,
        ):
            result = extract_all_tables(
                pdf_bytes=b"%PDF-1.4 stub",
                page_count=10,
                run_id="r1",
                registry_id="NCT00001",
                source_sha256="a" * 64,
                existing_tables=existing,
            )

        assert len(result) == 2
        pages = [t.page_number for t in result]
        assert 1 in pages
        assert 5 in pages

    def test_output_format(self):
        """Test output is properly formatted ExtractedTable objects."""
        plumber_result = [
            {
                "page": 3,
                "header": ["Assessment", "V1", "V2"],
                "rows": [["Exam", "X", ""], ["Labs", "", "X"]],
                "extractor": "pdfplumber_table",
                "cell_count": 9,
            },
        ]

        with patch(
            "ptcv.extraction.table_extractor._camelot_extract",
            return_value=[],
        ), patch(
            "ptcv.extraction.table_extractor._pdfplumber_extract",
            return_value=plumber_result,
        ):
            result = extract_all_tables(
                pdf_bytes=b"%PDF",
                page_count=5,
                run_id="run-001",
                registry_id="NCT12345678",
                source_sha256="b" * 64,
            )

        assert len(result) == 1
        t = result[0]
        assert t.run_id == "run-001"
        assert t.source_registry_id == "NCT12345678"
        assert t.page_number == 3
        assert t.extractor_used == "pdfplumber_table"
        assert t.table_index == 0

        header = json.loads(t.header_row)
        assert header == ["Assessment", "V1", "V2"]
        rows = json.loads(t.data_rows)
        assert len(rows) == 2

    def test_no_tables_found(self):
        """Test empty result when no tables detected."""
        with patch(
            "ptcv.extraction.table_extractor._camelot_extract",
            return_value=[],
        ), patch(
            "ptcv.extraction.table_extractor._pdfplumber_extract",
            return_value=[],
        ):
            result = extract_all_tables(
                pdf_bytes=b"%PDF",
                page_count=3,
                run_id="r1",
                registry_id="NCT00001",
                source_sha256="a" * 64,
            )

        assert result == []
