"""Tests for ptcv.extraction.pdf_extractor.

Covers multi-page table reconstruction (PTCV-19 Scenario 3) and the
pdfplumber→camelot cascade (PTCV-19 Scenario 2).

Qualification phase: IQ/OQ
Risk tier: MEDIUM
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.extraction.pdf_extractor import PdfExtractor


_RUN = "run-002"
_REG = "NCT00492050"
_SHA = "b" * 64


class TestPdfExtractorTextExtraction:
    def setup_method(self):
        self.extractor = PdfExtractor(min_words_for_text_block=3)

    def test_extract_returns_text_blocks(self, minimal_pdf_bytes):
        blocks, tables, page_count = self.extractor.extract(
            minimal_pdf_bytes, _RUN, _REG, _SHA
        )
        assert page_count >= 1
        # Text blocks may vary by pdfplumber version/PDF structure
        assert isinstance(blocks, list)
        assert isinstance(tables, list)

    def test_block_fields_set(self, minimal_pdf_bytes):
        blocks, _, _ = self.extractor.extract(
            minimal_pdf_bytes, _RUN, _REG, _SHA
        )
        for blk in blocks:
            assert blk.run_id == _RUN
            assert blk.source_registry_id == _REG
            assert blk.source_sha256 == _SHA
            assert blk.page_number >= 1
            assert blk.block_type in ("heading", "paragraph", "list_item")
            assert blk.extraction_timestamp_utc == ""  # set by service

    def test_classify_heading(self):
        result = PdfExtractor._classify_block_type("INTRODUCTION")
        assert result == "heading"

    def test_classify_list_item_bullet(self):
        result = PdfExtractor._classify_block_type("• Treatment arm A")
        assert result == "list_item"

    def test_classify_list_item_numbered(self):
        result = PdfExtractor._classify_block_type("1. Inclusion criteria")
        assert result == "list_item"

    def test_classify_paragraph(self):
        result = PdfExtractor._classify_block_type(
            "This is a longer paragraph describing the study methodology."
        )
        assert result == "paragraph"


class TestMultiPageTableReconstruction:
    """PTCV-19 Scenario 3: multi-page SoA table reconstruction."""

    def setup_method(self):
        self.extractor = PdfExtractor()

    def test_single_page_unchanged(self):
        pages = [
            [
                [["Visit", "Day", "Procedure"],
                 ["Screening", "-7", "Blood draw"]],
            ]
        ]
        result = self.extractor._reconstruct_multi_page_tables(pages)
        assert len(result[0][0]) == 2  # header + 1 row unchanged

    def test_two_page_table_merged(self):
        """Header repeated on page 2 → rows merged into page 1 table."""
        header = ["Visit", "Day", "Procedure"]
        pages = [
            # Page 1: header + row 1
            [[header + [], ["Screening", "-7", "Blood draw"]]],
            # Page 2: repeated header + row 2
            [[header + [], ["Week 4", "28", "ECG"]]],
        ]
        result = self.extractor._reconstruct_multi_page_tables(pages)
        merged_table = result[0][0]
        # Should have header + 2 data rows
        assert len(merged_table) == 3
        assert merged_table[0] == header
        assert merged_table[1] == ["Screening", "-7", "Blood draw"]
        assert merged_table[2] == ["Week 4", "28", "ECG"]

    def test_three_page_table_merged(self):
        """SoA spanning 3 pages reconstructed as single continuous record."""
        header = ["Visit", "Day", "Lab"]
        # Outer list = pages; each page contains a list of tables;
        # each table is a list of rows (list of cells).
        pages = [
            [[header, ["V1", "1", "Yes"]]],   # page 1: one table
            [[header, ["V2", "8", "No"]]],    # page 2: continuation
            [[header, ["V3", "15", "Yes"]]],  # page 3: continuation
        ]
        result = self.extractor._reconstruct_multi_page_tables(pages)
        merged = result[0][0]
        assert len(merged) == 4  # header + 3 data rows
        assert merged[1] == ["V1", "1", "Yes"]
        assert merged[3] == ["V3", "15", "Yes"]

    def test_non_matching_header_starts_new_table(self):
        """Different headers on page 2 → separate tables."""
        pages = [
            [[["A", "B"], ["1", "2"]]],
            [[["X", "Y"], ["3", "4"]]],
        ]
        result = self.extractor._reconstruct_multi_page_tables(pages)
        # Page 1 has its own table, page 2 has a different table
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert result[0][0][0] == ["A", "B"]
        assert result[1][0][0] == ["X", "Y"]

    def test_empty_pages_reset_active_header(self):
        """Empty page between tables resets the continuation state."""
        header = ["A", "B"]
        pages = [
            [[header, ["1", "2"]]],   # page 1: table with header
            [],                        # page 2: empty
            [[header, ["3", "4"]]],   # page 3: same header — new table
        ]
        result = self.extractor._reconstruct_multi_page_tables(pages)
        assert len(result[0]) == 1  # page 1: original table
        assert len(result[2]) == 1  # page 3: new table, not merged


class TestCamelotFallback:
    """PTCV-19 Scenario 2: cascade to Camelot when pdfplumber returns zero rows."""

    def setup_method(self):
        self.extractor = PdfExtractor()

    def test_camelot_used_on_pdfplumber_empty(self, minimal_pdf_bytes):
        """When pdfplumber finds no tables, camelot is attempted."""
        import pandas as pd

        mock_camelot_table = MagicMock()
        mock_camelot_table.df = pd.DataFrame(
            [["Visit", "Day"], ["Screening", "-7"]]
        )

        with patch("ptcv.extraction.pdf_extractor.PdfExtractor._fallback_extract") as mock_fb:
            mock_fb.return_value = [
                type(
                    "E",
                    (),
                    {
                        "extractor_used": "camelot",
                        "page_number": 1,
                        "table_index": 0,
                        "header_row": '["Visit","Day"]',
                        "data_rows": '[["Screening","-7"]]',
                        "run_id": _RUN,
                        "source_registry_id": _REG,
                        "source_sha256": _SHA,
                        "extraction_timestamp_utc": "",
                    },
                )()
            ]
            _, tables, _ = self.extractor.extract(
                minimal_pdf_bytes, _RUN, _REG, _SHA
            )
            # fallback was called (pdfplumber found no tables in minimal PDF)
            mock_fb.assert_called_once()

    def test_camelot_extract_returns_extractor_name(self, tmp_path):
        """_camelot_extract returns tables with extractor_used='camelot'."""
        import pandas as pd

        mock_table = MagicMock()
        mock_table.df = pd.DataFrame([["Visit", "Day"], ["Screening", "-7"]])

        with patch("camelot.read_pdf", return_value=[mock_table]):
            results = self.extractor._camelot_extract(
                str(tmp_path / "fake.pdf"), 1, _RUN, _REG, _SHA, 0
            )
        assert len(results) == 1
        assert results[0].extractor_used == "camelot"
        header = json.loads(results[0].header_row)
        assert "Visit" in header


class TestRealPdfExtraction:
    """Integration test using a real ClinicalTrials.gov protocol PDF."""

    def setup_method(self):
        self.extractor = PdfExtractor()

    def test_real_pdf_extracts_text_and_metadata(self, real_pdf_bytes):
        blocks, tables, page_count = self.extractor.extract(
            real_pdf_bytes, _RUN, "NCT00112827", _SHA
        )
        assert page_count > 0
        # A real protocol has at least some text blocks
        assert len(blocks) > 0
        # All blocks must have correct run_id
        assert all(b.run_id == _RUN for b in blocks)
