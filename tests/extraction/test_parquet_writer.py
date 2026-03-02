"""Tests for ptcv.extraction.parquet_writer.

Qualification phase: IQ/OQ
Risk tier: MEDIUM (data pipeline I/O)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.extraction.models import ExtractionMetadata, ExtractedTable, TextBlock
from ptcv.extraction.parquet_writer import (
    metadata_to_parquet,
    parquet_to_metadata,
    parquet_to_tables,
    parquet_to_text_blocks,
    tables_to_parquet,
    text_blocks_to_parquet,
)


_TS = "2025-01-01T00:00:00+00:00"
_RUN = "run-001"
_REG = "NCT00112827"
_SHA = "a" * 64


class TestTextBlocksParquet:
    def _make_block(self, idx: int = 0) -> TextBlock:
        return TextBlock(
            run_id=_RUN,
            source_registry_id=_REG,
            source_sha256=_SHA,
            page_number=1,
            block_index=idx,
            text="Clinical trial text block.",
            block_type="paragraph",
            extraction_timestamp_utc=_TS,
        )

    def test_round_trip_single_block(self):
        blk = self._make_block()
        data = text_blocks_to_parquet([blk])
        result = parquet_to_text_blocks(data)
        assert len(result) == 1
        assert result[0].text == blk.text
        assert result[0].run_id == _RUN
        assert result[0].block_type == "paragraph"

    def test_round_trip_empty_list(self):
        data = text_blocks_to_parquet([])
        result = parquet_to_text_blocks(data)
        assert result == []

    def test_round_trip_multiple_blocks(self):
        blocks = [self._make_block(i) for i in range(5)]
        data = text_blocks_to_parquet(blocks)
        result = parquet_to_text_blocks(data)
        assert len(result) == 5
        assert [b.block_index for b in result] == list(range(5))

    def test_missing_timestamp_raises(self):
        blk = self._make_block()
        blk.extraction_timestamp_utc = ""
        with pytest.raises(ValueError, match="extraction_timestamp_utc"):
            text_blocks_to_parquet([blk])

    def test_returns_bytes(self):
        data = text_blocks_to_parquet([self._make_block()])
        assert isinstance(data, bytes)
        assert data[:4] == b"PAR1"  # Parquet magic


class TestTablesParquet:
    def _make_table(self, idx: int = 0) -> ExtractedTable:
        return ExtractedTable(
            run_id=_RUN,
            source_registry_id=_REG,
            source_sha256=_SHA,
            page_number=3,
            extractor_used="pdfplumber",
            table_index=idx,
            header_row='["Visit", "Day", "Procedure"]',
            data_rows='[["Screening", "Day -7", "Blood draw"]]',
            extraction_timestamp_utc=_TS,
        )

    def test_round_trip(self):
        tbl = self._make_table()
        data = tables_to_parquet([tbl])
        result = parquet_to_tables(data)
        assert len(result) == 1
        assert result[0].extractor_used == "pdfplumber"
        assert result[0].header_row == tbl.header_row

    def test_round_trip_empty(self):
        data = tables_to_parquet([])
        result = parquet_to_tables(data)
        assert result == []

    def test_missing_timestamp_raises(self):
        tbl = self._make_table()
        tbl.extraction_timestamp_utc = ""
        with pytest.raises(ValueError, match="extraction_timestamp_utc"):
            tables_to_parquet([tbl])

    def test_extractor_used_camelot(self):
        tbl = self._make_table()
        tbl.extractor_used = "camelot"
        data = tables_to_parquet([tbl])
        result = parquet_to_tables(data)
        assert result[0].extractor_used == "camelot"


class TestMetadataParquet:
    def _make_meta(self) -> ExtractionMetadata:
        return ExtractionMetadata(
            run_id=_RUN,
            source_registry_id=_REG,
            source_sha256=_SHA,
            format_detected="pdf",
            extractor_used="pdfplumber",
            page_count=42,
            table_count=7,
            text_block_count=120,
            extraction_timestamp_utc=_TS,
        )

    def test_round_trip(self):
        meta = self._make_meta()
        data = metadata_to_parquet(meta)
        result = parquet_to_metadata(data)
        assert result.run_id == _RUN
        assert result.page_count == 42
        assert result.table_count == 7
        assert result.format_detected == "pdf"

    def test_missing_timestamp_raises(self):
        meta = self._make_meta()
        meta.extraction_timestamp_utc = ""
        with pytest.raises(ValueError, match="extraction_timestamp_utc"):
            metadata_to_parquet(meta)
