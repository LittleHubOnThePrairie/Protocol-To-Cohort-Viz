"""PTCV extraction data models.

Represents the output of the protocol extraction stage (PTCV-19):
text blocks, tables, and per-run metadata written to Parquet.

Risk tier: MEDIUM — data pipeline output (no patient data).

Regulatory references:
- ALCOA+ Contemporaneous: extraction_timestamp_utc on every record
- ALCOA+ Traceable: source_sha256 links to PTCV-18 download artifact
- ALCOA+ Original: re-run creates new run_id; prior Parquet preserved
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class TextBlock:
    """One extracted text block from a protocol document.

    Attributes:
        run_id: UUID4 for this extraction pipeline run.
        source_registry_id: Trial identifier (e.g. "NCT00112827").
        source_sha256: SHA-256 of the source protocol file from PTCV-18.
        page_number: 1-based page where the block was found.
            0 for non-page-based formats (e.g. CTR-XML).
        block_index: 0-based position of this block on the page.
        text: Extracted text content.
        block_type: Structural role: "heading", "paragraph", "list_item",
            or "other".
        extraction_timestamp_utc: ISO 8601 UTC timestamp, set just before
            the Parquet artifact is written (ALCOA+ Contemporaneous).
    """

    run_id: str
    source_registry_id: str
    source_sha256: str
    page_number: int
    block_index: int
    text: str
    block_type: str
    extraction_timestamp_utc: str = ""


@dataclasses.dataclass
class ExtractedTable:
    """One extracted table from a protocol document.

    Attributes:
        run_id: UUID4 for this extraction run.
        source_registry_id: Trial identifier.
        source_sha256: SHA-256 of the source protocol file.
        page_number: 1-based page where the table starts.
        extractor_used: Name of the extractor that produced this row:
            "pymupdf4llm", "pdfplumber", "camelot", "tabula", or
            "claude_vision" (PTCV-172).
        table_index: 0-based position of this table in the document.
        header_row: Column header cells serialised as a JSON array string.
        data_rows: Table data rows serialised as a JSON array-of-arrays.
        extraction_timestamp_utc: ISO 8601 UTC timestamp.
    """

    run_id: str
    source_registry_id: str
    source_sha256: str
    page_number: int
    extractor_used: str
    table_index: int
    header_row: str
    data_rows: str
    extraction_timestamp_utc: str = ""


@dataclasses.dataclass
class ExtractionMetadata:
    """Per-run extraction summary written to metadata.parquet.

    Attributes:
        run_id: UUID4 for this extraction run.
        source_registry_id: Trial identifier.
        source_sha256: SHA-256 of the source protocol file.
        format_detected: Detected file format: "pdf", "ctr-xml",
            "word", or "unknown".
        extractor_used: Primary extractor that produced most tables,
            or "ctr-xml" / "word" for non-PDF formats.
        page_count: Total page count (0 for non-paged formats).
        table_count: Number of tables extracted.
        text_block_count: Number of text blocks extracted.
        extraction_timestamp_utc: ISO 8601 UTC timestamp.
    """

    run_id: str
    source_registry_id: str
    source_sha256: str
    format_detected: str
    extractor_used: str
    page_count: int
    table_count: int
    text_block_count: int
    extraction_timestamp_utc: str = ""
    landscape_pages: str = ""


@dataclasses.dataclass
class ExtractionResult:
    """Return value from ExtractionService.extract().

    Attributes:
        run_id: UUID4 for this extraction run.
        registry_id: Trial identifier.
        format_detected: "pdf", "ctr-xml", "word", or "unknown".
        text_artifact_key: Storage key for text_blocks.parquet.
        tables_artifact_key: Storage key for tables.parquet.
        metadata_artifact_key: Storage key for metadata.parquet.
        text_artifact_sha256: SHA-256 of text_blocks.parquet.
        tables_artifact_sha256: SHA-256 of tables.parquet.
        metadata_artifact_sha256: SHA-256 of metadata.parquet.
        source_sha256: SHA-256 of the source protocol file.
        table_count: Number of extracted tables.
        text_block_count: Number of extracted text blocks.
        lineage_run_id: run_id used for all three lineage records
            (equals run_id).
    """

    run_id: str
    registry_id: str
    format_detected: str
    text_artifact_key: str
    tables_artifact_key: str
    metadata_artifact_key: str
    text_artifact_sha256: str
    tables_artifact_sha256: str
    metadata_artifact_sha256: str
    source_sha256: str
    table_count: int
    text_block_count: int
    lineage_run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.lineage_run_id is None:
            self.lineage_run_id = self.run_id
