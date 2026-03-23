"""Tests for column header validation (PTCV-215).

Verifies visit header matching, row alignment detection,
multi-column header splitting, and escalation logic.
"""

from __future__ import annotations

import pytest

from ptcv.soa_extractor.column_validator import (
    ColumnValidationResult,
    validate_columns,
    split_multi_column_header,
)


class TestValidateColumns:
    """Tests for validate_columns function."""

    def test_all_visit_headers_matched(self):
        """Test 100% match ratio for standard visit headers."""
        header = [
            "Assessment", "Screening", "Baseline", "Day 1",
            "Week 2", "Week 4", "EOT", "Follow-up",
        ]
        rows = [
            ["Physical Exam", "X", "X", "X", "X", "X", "X", "X"],
            ["Vitals", "X", "X", "X", "X", "X", "", "X"],
        ]
        result = validate_columns(header, rows)

        assert result.header_count == 8
        assert result.visit_headers_matched >= 5
        assert result.match_ratio > 0.5
        assert not result.needs_escalation
        assert result.misaligned_rows == []

    def test_low_match_ratio_triggers_escalation(self):
        """Test escalation when few headers match visit patterns."""
        header = [
            "Assessment", "Column A", "Column B", "Column C",
            "Column D", "Column E",
        ]
        rows = [["Row1", "a", "b", "c", "d", "e"]]
        result = validate_columns(header, rows, min_match_ratio=0.3)

        assert result.match_ratio < 0.3
        assert result.needs_escalation
        assert "Low visit header match ratio" in result.escalation_reason

    def test_misaligned_rows_detected(self):
        """Test detection of rows with wrong cell count."""
        header = ["Assessment", "V1", "V2", "V3"]
        rows = [
            ["Exam", "X", "X", "X"],      # OK
            ["Labs", "X", "X"],            # Missing 1 cell
            ["ECG", "X", "X", "X", "X"],   # Extra cell
        ]
        result = validate_columns(header, rows)

        assert len(result.misaligned_rows) == 2
        assert 1 in result.misaligned_rows
        assert 2 in result.misaligned_rows

    def test_high_misalignment_triggers_escalation(self):
        """Test escalation when too many rows are misaligned."""
        header = ["Assessment", "Screening", "Day 1"]
        rows = [
            ["A", "X"],          # Misaligned
            ["B", "X"],          # Misaligned
            ["C", "X"],          # Misaligned
            ["D", "X", "X"],     # OK
        ]
        result = validate_columns(
            header, rows, max_misaligned_ratio=0.2
        )

        assert result.needs_escalation
        assert "misalignment" in result.escalation_reason.lower()

    def test_multi_column_header_detected(self):
        """Test detection of multi-column headers."""
        header = [
            "Assessment", "Screening",
            "Future Cycles (even) (odd)", "EOT",
        ]
        rows = [["Exam", "X", "X", "X"]]
        result = validate_columns(header, rows)

        assert len(result.multi_column_headers) == 1
        assert "even" in result.multi_column_headers[0].lower()

    def test_empty_header(self):
        """Test empty header triggers escalation."""
        result = validate_columns([], [])
        assert result.needs_escalation
        assert "No column headers" in result.escalation_reason

    def test_no_rows(self):
        """Test validation with headers but no data rows."""
        header = ["Assessment", "Screening", "Day 1"]
        result = validate_columns(header, [])

        assert result.header_count == 3
        assert result.misaligned_rows == []
        assert not result.needs_escalation

    def test_single_column_not_escalated(self):
        """Test that a 2-column table with no visit match doesn't escalate."""
        header = ["Name", "Value"]
        rows = [["A", "1"], ["B", "2"]]
        result = validate_columns(header, rows, min_match_ratio=0.3)

        # Only 1 visit header to match; < 3 so skip ratio check
        assert not result.needs_escalation


class TestSplitMultiColumnHeader:
    """Tests for split_multi_column_header function."""

    def test_even_odd_split(self):
        """Test splitting 'X (even) (odd)' pattern."""
        result = split_multi_column_header(
            "Future Cycles (even) (odd)"
        )
        assert result == [
            "Future Cycles (even)",
            "Future Cycles (odd)",
        ]

    def test_slash_split(self):
        """Test splitting 'A / B' pattern."""
        result = split_multi_column_header("Cycle 1 / Cycle 2")
        assert result == ["Cycle 1", "Cycle 2"]

    def test_pipe_split(self):
        """Test splitting 'A | B' pattern."""
        result = split_multi_column_header("Arm A | Arm B")
        assert result == ["Arm A", "Arm B"]

    def test_no_split_needed(self):
        """Test single header returned as-is."""
        result = split_multi_column_header("Screening")
        assert result == ["Screening"]

    def test_no_split_short_tokens(self):
        """Test that very short tokens don't trigger split."""
        result = split_multi_column_header("A / B")
        assert result == ["A / B"]  # Too short to split

    def test_case_insensitive_even_odd(self):
        """Test even/odd detection is case-insensitive."""
        result = split_multi_column_header(
            "Cycles (Even) (Odd)"
        )
        assert len(result) == 2


class TestTableDiscoveryLattice:
    """Tests for Camelot Lattice integration in TableDiscovery."""

    def test_camelot_disabled_by_default(self):
        """Test Camelot disabled by default (PTCV-244)."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        td = TableDiscovery()
        assert td._enable_camelot is False
        assert td._enable_lattice is False

    def test_camelot_can_be_enabled(self):
        """Test Camelot can be opted in."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        td = TableDiscovery(enable_camelot=True, enable_lattice=True)
        assert td._enable_camelot is True
        assert td._enable_lattice is True

    def test_merge_camelot_modes_stream_only(self):
        """Test merge with only stream results."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        stream = [
            {"page": 1, "header": ["A"], "rows": [], "extractor": "camelot_stream", "cell_count": 10},
        ]
        result = TableDiscovery._merge_camelot_modes(stream, [])
        assert len(result) == 1
        assert result[0]["extractor"] == "camelot_stream"

    def test_merge_camelot_modes_lattice_only(self):
        """Test merge with only lattice results."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        lattice = [
            {"page": 1, "header": ["A"], "rows": [], "extractor": "camelot_lattice", "cell_count": 12},
        ]
        result = TableDiscovery._merge_camelot_modes([], lattice)
        assert len(result) == 1
        assert result[0]["extractor"] == "camelot_lattice"

    def test_merge_camelot_modes_keeps_higher_cells(self):
        """Test merge keeps result with higher cell count per page."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        stream = [
            {"page": 1, "header": ["A", "B"], "rows": [["1", "2"]], "extractor": "camelot_stream", "cell_count": 4},
        ]
        lattice = [
            {"page": 1, "header": ["A", "B", "C"], "rows": [["1", "2", "3"], ["4", "5", "6"]], "extractor": "camelot_lattice", "cell_count": 9},
        ]
        result = TableDiscovery._merge_camelot_modes(stream, lattice)
        assert len(result) == 1
        assert result[0]["extractor"] == "camelot_lattice"
        assert result[0]["cell_count"] == 9

    def test_merge_camelot_modes_different_pages(self):
        """Test merge includes results from different pages."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        stream = [
            {"page": 1, "header": ["A"], "rows": [], "extractor": "camelot_stream", "cell_count": 10},
        ]
        lattice = [
            {"page": 2, "header": ["B"], "rows": [], "extractor": "camelot_lattice", "cell_count": 8},
        ]
        result = TableDiscovery._merge_camelot_modes(stream, lattice)
        assert len(result) == 2
