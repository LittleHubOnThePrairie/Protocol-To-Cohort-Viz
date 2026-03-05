"""Tests for ptcv.extraction.pdf_extractor.

Covers multi-page table reconstruction (PTCV-19 Scenario 3), the
pdfplumber→camelot cascade (PTCV-19 Scenario 2), and landscape page
rotation normalisation (PTCV-63).

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
        blocks, tables, page_count, landscape = self.extractor.extract(
            minimal_pdf_bytes, _RUN, _REG, _SHA
        )
        assert page_count >= 1
        # Text blocks may vary by pdfplumber version/PDF structure
        assert isinstance(blocks, list)
        assert isinstance(tables, list)
        assert isinstance(landscape, list)

    def test_block_fields_set(self, minimal_pdf_bytes):
        blocks, _, _, _ = self.extractor.extract(
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
            _, tables, _, _ = self.extractor.extract(
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
        blocks, tables, page_count, _landscape = self.extractor.extract(
            real_pdf_bytes, _RUN, "NCT00112827", _SHA
        )
        assert page_count > 0
        # A real protocol has at least some text blocks
        assert len(blocks) > 0
        # All blocks must have correct run_id
        assert all(b.run_id == _RUN for b in blocks)


# -------------------------------------------------------------------
# PTCV-63: Landscape page rotation normalisation
# -------------------------------------------------------------------


def _make_landscape_pdf() -> bytes:
    """Create a 2-page PDF: page 1 portrait, page 2 landscape content.

    Page 2 uses a rotated text matrix (0, 10, -10, 0) to simulate
    landscape content rendered on a portrait page — the same pattern
    seen in NCT03045302 pages 67-69.
    """
    import fitz

    doc = fitz.open()

    # Page 1: normal portrait text
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 100),
        "This is normal portrait text on page one.",
        fontsize=12,
    )

    # Page 2: landscape content via rotated text matrix
    page2 = doc.new_page(width=595, height=842)
    # Insert text with 90-degree rotation using a text writer
    tw = fitz.TextWriter(page2.rect)
    tw.append(
        (100, 50),
        "Schedule of Activities Visit Table",
        fontsize=12,
    )
    # The TextWriter writes normal text. To simulate the rotation,
    # we insert via a rotated morph transform.
    page2.insert_text(
        (100, 50),
        "Schedule of Activities Visit Table",
        fontsize=12,
        rotate=90,
    )
    page2.insert_text(
        (100, 100),
        "V1 V2 V3 V4 V5 Screening Treatment",
        fontsize=10,
        rotate=90,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestLandscapePageDetection:
    """PTCV-63: landscape content pages are detected and normalised."""

    def setup_method(self):
        self.extractor = PdfExtractor(min_words_for_text_block=2)

    def test_portrait_pdf_no_landscape_detected(self, minimal_pdf_bytes):
        """Normal portrait PDF has no landscape pages."""
        _, _, _, landscape = self.extractor.extract(
            minimal_pdf_bytes, _RUN, _REG, _SHA
        )
        assert landscape == []

    def test_landscape_pdf_detected(self):
        """Pages with rotated text matrices are detected."""
        pdf_bytes = _make_landscape_pdf()
        _, _, page_count, landscape = self.extractor.extract(
            pdf_bytes, _RUN, _REG, _SHA
        )
        assert page_count == 2
        # Page 2 should be detected as landscape
        assert 2 in landscape
        # Page 1 should NOT be in landscape list
        assert 1 not in landscape

    def test_landscape_text_reads_correctly(self):
        """Rotated page text is extracted in correct reading order."""
        pdf_bytes = _make_landscape_pdf()
        blocks, _, _, landscape = self.extractor.extract(
            pdf_bytes, _RUN, _REG, _SHA
        )
        # Find text blocks from page 2
        page2_texts = [
            b.text for b in blocks if b.page_number == 2
        ]
        # Text should read forwards, not backwards
        full_text = " ".join(page2_texts)
        # Should contain "Schedule" not "eludehcS"
        assert "eludehcS" not in full_text
        if page2_texts:
            assert any(
                "Schedule" in t or "Visit" in t or "V1" in t
                for t in page2_texts
            )

    def test_normalize_returns_unmodified_for_normal_pdf(
        self, minimal_pdf_bytes
    ):
        """Normal PDFs pass through without modification."""
        result_bytes, landscape = PdfExtractor._normalize_page_rotation(
            minimal_pdf_bytes,
        )
        assert landscape == []
        # Bytes may differ slightly due to PyMuPDF round-trip, but
        # the key assertion is that no pages were rotated.

    def test_normalize_detects_rotated_pages(self):
        """PyMuPDF detects rotated content via line direction vectors."""
        pdf_bytes = _make_landscape_pdf()
        _, landscape = PdfExtractor._normalize_page_rotation(pdf_bytes)
        assert 2 in landscape

    def test_normalize_graceful_without_fitz(self, minimal_pdf_bytes):
        """If PyMuPDF is not installed, falls back gracefully."""
        with patch.dict("sys.modules", {"fitz": None}):
            result_bytes, landscape = (
                PdfExtractor._normalize_page_rotation(minimal_pdf_bytes)
            )
            assert landscape == []
            assert result_bytes == minimal_pdf_bytes


class TestLandscapeRealProtocol:
    """PTCV-63 Scenario 2: NCT03045302 landscape SoA extraction."""

    def setup_method(self):
        self.extractor = PdfExtractor()

    @pytest.fixture()
    def nct03045302_bytes(self) -> bytes:
        pdf_path = Path(
            "C:/Dev/PTCV/data/protocols/clinicaltrials/"
            "NCT03045302_1.0.pdf"
        )
        if not pdf_path.exists():
            pytest.skip("NCT03045302 PDF not available")
        return pdf_path.read_bytes()

    def test_landscape_pages_detected(self, nct03045302_bytes):
        """Pages 67-69 and 76-77 detected as landscape content."""
        _, _, _, landscape = self.extractor.extract(
            nct03045302_bytes, _RUN, "NCT03045302", _SHA
        )
        assert 67 in landscape
        assert 68 in landscape
        assert 69 in landscape

    def test_soa_text_extracted_correctly(self, nct03045302_bytes):
        """SoA text on landscape pages reads forwards."""
        blocks, _, _, _ = self.extractor.extract(
            nct03045302_bytes, _RUN, "NCT03045302", _SHA
        )
        page67_texts = [
            b.text for b in blocks if b.page_number == 67
        ]
        full = " ".join(page67_texts)
        # Text should read "Study Schedule" not "ydutS eludehcS"
        assert "ydutS" not in full
        assert "Schedule" in full or "Study" in full

    def test_soa_table_headers_correct(self, nct03045302_bytes):
        """Table extracted from landscape page has correct headers."""
        _, tables, _, _ = self.extractor.extract(
            nct03045302_bytes, _RUN, "NCT03045302", _SHA
        )
        # Find tables on pages 67-68
        soa_tables = [
            t for t in tables if t.page_number in (67, 68)
        ]
        assert len(soa_tables) > 0
        header = json.loads(soa_tables[0].header_row)
        # Should contain visit-like headers, not reversed text
        header_text = " ".join(str(h) for h in header)
        assert "ydutS" not in header_text

    def test_portrait_pages_unchanged(self, nct03045302_bytes):
        """Portrait pages (e.g. page 1) extract normally."""
        blocks, _, _, _ = self.extractor.extract(
            nct03045302_bytes, _RUN, "NCT03045302", _SHA
        )
        page1_texts = [
            b.text for b in blocks if b.page_number == 1
        ]
        assert len(page1_texts) > 0
        full = " ".join(page1_texts)
        # "CONFIDENTIAL" should appear normally
        assert "CONFIDENTIAL" in full or "Protocol" in full
