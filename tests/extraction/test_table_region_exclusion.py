"""Tests for table-region exclusion from text blocks (PTCV-224).

Tests the in_table flag on TextBlock and the table-cell overlap
detection logic.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from ptcv.extraction.models import TextBlock
from ptcv.extraction.pdf_extractor import (
    _collect_table_cell_texts,
    _line_overlaps_table,
)


class TestTextBlockInTable:
    """Tests for the in_table field on TextBlock."""

    def test_default_false(self):
        """Test in_table defaults to False."""
        block = TextBlock(
            run_id="r1",
            source_registry_id="NCT00001",
            source_sha256="a" * 64,
            page_number=1,
            block_index=0,
            text="Some prose text",
            block_type="paragraph",
        )
        assert block.in_table is False

    def test_explicit_true(self):
        """Test in_table can be set to True."""
        block = TextBlock(
            run_id="r1",
            source_registry_id="NCT00001",
            source_sha256="a" * 64,
            page_number=1,
            block_index=0,
            text="Drug A 100mg BID",
            block_type="paragraph",
            in_table=True,
        )
        assert block.in_table is True


class TestCollectTableCellTexts:
    """Tests for _collect_table_cell_texts helper."""

    def test_extracts_cell_texts(self):
        """Test table cell texts are collected."""
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [
            [
                ["Assessment", "Visit 1", "Visit 2"],
                ["Physical Exam", "X", "X"],
                ["ECG", "", "X"],
            ],
        ]

        result = _collect_table_cell_texts(mock_page)

        assert "assessment" in result
        assert "physical exam" in result
        assert "ecg" in result
        assert "visit 1" in result
        # Single-char "X" and "" are excluded (len < 2)
        assert "x" not in result

    def test_no_tables(self):
        """Test empty set when no tables found."""
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = []

        result = _collect_table_cell_texts(mock_page)
        assert result == set()

    def test_extract_tables_exception(self):
        """Test graceful handling when extract_tables fails."""
        mock_page = MagicMock()
        mock_page.extract_tables.side_effect = RuntimeError("fail")

        result = _collect_table_cell_texts(mock_page)
        assert result == set()

    def test_none_cells_handled(self):
        """Test None cells are skipped."""
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [
            [[None, "Value", None]],
        ]

        result = _collect_table_cell_texts(mock_page)
        assert "value" in result


class TestLineOverlapsTable:
    """Tests for _line_overlaps_table helper."""

    def test_exact_match(self):
        """Test exact cell text match."""
        cells = {"physical exam", "ecg", "assessment"}
        assert _line_overlaps_table("Physical Exam", cells) is True

    def test_no_match(self):
        """Test no overlap with unrelated text."""
        cells = {"physical exam", "ecg"}
        assert _line_overlaps_table(
            "This study investigates a novel compound",
            cells,
        ) is False

    def test_word_overlap_above_threshold(self):
        """Test word-level overlap > 50% flags as table."""
        cells = {"drug", "100mg", "bid", "dosing schedule"}
        # Line: "Drug A 100mg BID daily" — 3 of 4 words match
        assert _line_overlaps_table(
            "Drug 100mg BID daily",
            cells,
        ) is True

    def test_word_overlap_below_threshold(self):
        """Test word-level overlap < 50% is not flagged."""
        cells = {"exam"}
        assert _line_overlaps_table(
            "The primary endpoint is overall survival at 12 months",
            cells,
        ) is False

    def test_empty_cells(self):
        """Test empty cell set returns False."""
        assert _line_overlaps_table("any text", set()) is False

    def test_single_word_line(self):
        """Test single-word lines are not flagged via word overlap."""
        cells = {"assessment", "visit"}
        # Single word — exact match works but word overlap needs >= 2 words
        assert _line_overlaps_table("Assessment", cells) is True  # exact match
        assert _line_overlaps_table("Z", cells) is False


class TestFilterInTable:
    """Tests for downstream filtering of in_table blocks."""

    def test_filter_clean_prose(self):
        """Test filtering in_table=False gives clean prose."""
        blocks = [
            TextBlock("r", "N", "s", 1, 0, "Study design", "heading"),
            TextBlock("r", "N", "s", 1, 1, "Drug A 100mg", "paragraph", in_table=True),
            TextBlock("r", "N", "s", 1, 2, "Drug B 200mg", "paragraph", in_table=True),
            TextBlock("r", "N", "s", 1, 3, "Patients were randomized", "paragraph"),
        ]

        clean = [b for b in blocks if not b.in_table]
        assert len(clean) == 2
        assert clean[0].text == "Study design"
        assert clean[1].text == "Patients were randomized"

    def test_all_blocks_available(self):
        """Test all blocks still available without filtering."""
        blocks = [
            TextBlock("r", "N", "s", 1, 0, "Text", "paragraph"),
            TextBlock("r", "N", "s", 1, 1, "Table", "paragraph", in_table=True),
        ]
        assert len(blocks) == 2
