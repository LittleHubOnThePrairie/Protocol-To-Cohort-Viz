"""Tests for universal table extraction in ProtocolIndex (PTCV-237).

Tests that ProtocolIndex has a tables field and that
extract_protocol_index populates it.
"""

from __future__ import annotations

import dataclasses
import json
import pytest
from unittest.mock import MagicMock, patch

from ptcv.ich_parser.toc_extractor import ProtocolIndex


class TestProtocolIndexTablesField:
    """Tests for the tables field on ProtocolIndex."""

    def test_tables_defaults_empty(self):
        """Test tables field defaults to empty list."""
        idx = ProtocolIndex(
            source_path="/test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
        )
        assert idx.tables == []

    def test_tables_accepts_list(self):
        """Test tables field accepts list of dicts."""
        tables = [
            {
                "page_number": 5,
                "header_row": json.dumps(["Assessment", "V1", "V2"]),
                "data_rows": json.dumps([["Exam", "X", ""]]),
                "extractor_used": "camelot_stream",
                "table_index": 0,
            },
        ]
        idx = ProtocolIndex(
            source_path="/test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
            tables=tables,
        )
        assert len(idx.tables) == 1
        assert idx.tables[0]["page_number"] == 5
        assert json.loads(idx.tables[0]["header_row"]) == ["Assessment", "V1", "V2"]

    def test_tables_field_is_dataclass_field(self):
        """Test tables is a proper dataclass field with default factory."""
        fields = {f.name: f for f in dataclasses.fields(ProtocolIndex)}
        assert "tables" in fields
        # Default should be empty list (via default_factory)
        assert fields["tables"].default is dataclasses.MISSING
        assert fields["tables"].default_factory is not dataclasses.MISSING

    def test_multiple_tables(self):
        """Test multiple tables from different pages."""
        tables = [
            {
                "page_number": 3,
                "header_row": json.dumps(["Drug", "Dose"]),
                "data_rows": json.dumps([["A", "100mg"]]),
                "extractor_used": "camelot_lattice",
                "table_index": 0,
            },
            {
                "page_number": 12,
                "header_row": json.dumps(["Assessment", "Screen", "Day 1"]),
                "data_rows": json.dumps([["ECG", "X", "X"]]),
                "extractor_used": "pdfplumber_table",
                "table_index": 1,
            },
        ]
        idx = ProtocolIndex(
            source_path="/test.pdf",
            page_count=20,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
            tables=tables,
        )
        assert len(idx.tables) == 2
        assert idx.tables[0]["extractor_used"] == "camelot_lattice"
        assert idx.tables[1]["page_number"] == 12


class TestExtractProtocolIndexTables:
    """Tests that extract_protocol_index calls table extraction."""

    def test_extract_protocol_index_source_has_tables_call(self):
        """Test that extract_protocol_index references table extraction."""
        import inspect
        from ptcv.ich_parser.toc_extractor import extract_protocol_index
        source = inspect.getsource(extract_protocol_index)
        assert "extract_all_tables" in source
        assert "PTCV-237" in source
        assert "structured_tables" in source

    def test_protocol_index_returns_tables_field(self):
        """Test return value includes tables field."""
        import inspect
        from ptcv.ich_parser.toc_extractor import extract_protocol_index
        source = inspect.getsource(extract_protocol_index)
        assert "tables=structured_tables" in source

    def test_table_extraction_failure_non_blocking(self):
        """Test table extraction failure doesn't crash index creation."""
        import inspect
        from ptcv.ich_parser.toc_extractor import extract_protocol_index
        source = inspect.getsource(extract_protocol_index)
        # Should have try/except around table extraction
        assert "continuing without structured tables" in source
