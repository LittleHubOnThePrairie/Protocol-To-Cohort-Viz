"""Parquet serialisers for PTCV-19 extraction output.

Three schemas are defined, one per output artifact:
  - text_blocks.parquet: TextBlock rows
  - tables.parquet:      ExtractedTable rows
  - metadata.parquet:    ExtractionMetadata rows (single row per run)

All serialisation uses PyArrow + Snappy compression, consistent with
the rest of the PTCV data pipeline.

Risk tier: MEDIUM — data pipeline I/O.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from .models import ExtractionMetadata, ExtractedTable, TextBlock

# ---- text_blocks.parquet schema ------------------------------------

_TEXT_BLOCKS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("source_registry_id", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("page_number", pa.int32(), nullable=False),
        pa.field("block_index", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("block_type", pa.string(), nullable=False),
        pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
    ]
)

# ---- tables.parquet schema ----------------------------------------

_TABLES_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("source_registry_id", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("page_number", pa.int32(), nullable=False),
        pa.field("extractor_used", pa.string(), nullable=False),
        pa.field("table_index", pa.int32(), nullable=False),
        pa.field("header_row", pa.string(), nullable=False),
        pa.field("data_rows", pa.string(), nullable=False),
        pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
    ]
)

# ---- metadata.parquet schema --------------------------------------

_METADATA_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("source_registry_id", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("format_detected", pa.string(), nullable=False),
        pa.field("extractor_used", pa.string(), nullable=False),
        pa.field("page_count", pa.int32(), nullable=False),
        pa.field("table_count", pa.int32(), nullable=False),
        pa.field("text_block_count", pa.int32(), nullable=False),
        pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
    ]
)


# -----------------------------------------------------------------------
# Public serialisation functions
# -----------------------------------------------------------------------


def text_blocks_to_parquet(blocks: list["TextBlock"]) -> bytes:
    """Serialise TextBlock rows to Parquet bytes.

    An empty list is valid and produces a zero-row Parquet file.

    Args:
        blocks: List of TextBlock instances. All must have
            extraction_timestamp_utc set (non-empty).

    Returns:
        Snappy-compressed Parquet bytes.

    Raises:
        ValueError: If any block has an empty extraction_timestamp_utc.
    """
    for blk in blocks:
        if not blk.extraction_timestamp_utc:
            raise ValueError(
                f"extraction_timestamp_utc is empty for block "
                f"{blk.block_index} in run {blk.run_id}."
            )

    table = pa.table(
        {
            "run_id": [b.run_id for b in blocks],
            "source_registry_id": [b.source_registry_id for b in blocks],
            "source_sha256": [b.source_sha256 for b in blocks],
            "page_number": pa.array(
                [b.page_number for b in blocks], type=pa.int32()
            ),
            "block_index": pa.array(
                [b.block_index for b in blocks], type=pa.int32()
            ),
            "text": [b.text for b in blocks],
            "block_type": [b.block_type for b in blocks],
            "extraction_timestamp_utc": [
                b.extraction_timestamp_utc for b in blocks
            ],
        },
        schema=_TEXT_BLOCKS_SCHEMA,
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def tables_to_parquet(tables: list["ExtractedTable"]) -> bytes:
    """Serialise ExtractedTable rows to Parquet bytes.

    An empty list is valid and produces a zero-row Parquet file.

    Args:
        tables: List of ExtractedTable instances. All must have
            extraction_timestamp_utc set.

    Returns:
        Snappy-compressed Parquet bytes.

    Raises:
        ValueError: If any table has an empty extraction_timestamp_utc.
    """
    for tbl in tables:
        if not tbl.extraction_timestamp_utc:
            raise ValueError(
                f"extraction_timestamp_utc is empty for table "
                f"{tbl.table_index} in run {tbl.run_id}."
            )

    table = pa.table(
        {
            "run_id": [t.run_id for t in tables],
            "source_registry_id": [t.source_registry_id for t in tables],
            "source_sha256": [t.source_sha256 for t in tables],
            "page_number": pa.array(
                [t.page_number for t in tables], type=pa.int32()
            ),
            "extractor_used": [t.extractor_used for t in tables],
            "table_index": pa.array(
                [t.table_index for t in tables], type=pa.int32()
            ),
            "header_row": [t.header_row for t in tables],
            "data_rows": [t.data_rows for t in tables],
            "extraction_timestamp_utc": [
                t.extraction_timestamp_utc for t in tables
            ],
        },
        schema=_TABLES_SCHEMA,
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def metadata_to_parquet(metadata: "ExtractionMetadata") -> bytes:
    """Serialise a single ExtractionMetadata record to Parquet bytes.

    Args:
        metadata: ExtractionMetadata instance with all fields set.

    Returns:
        Snappy-compressed Parquet bytes (single-row file).

    Raises:
        ValueError: If extraction_timestamp_utc is empty.
    """
    if not metadata.extraction_timestamp_utc:
        raise ValueError(
            f"extraction_timestamp_utc is empty for run {metadata.run_id}."
        )

    table = pa.table(
        {
            "run_id": [metadata.run_id],
            "source_registry_id": [metadata.source_registry_id],
            "source_sha256": [metadata.source_sha256],
            "format_detected": [metadata.format_detected],
            "extractor_used": [metadata.extractor_used],
            "page_count": pa.array([metadata.page_count], type=pa.int32()),
            "table_count": pa.array(
                [metadata.table_count], type=pa.int32()
            ),
            "text_block_count": pa.array(
                [metadata.text_block_count], type=pa.int32()
            ),
            "extraction_timestamp_utc": [metadata.extraction_timestamp_utc],
        },
        schema=_METADATA_SCHEMA,
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


# -----------------------------------------------------------------------
# Round-trip readers (for testing and downstream consumers)
# -----------------------------------------------------------------------


def parquet_to_text_blocks(data: bytes) -> list["TextBlock"]:
    """Deserialise text_blocks.parquet bytes to TextBlock instances."""
    from .models import TextBlock

    tbl = pq.read_table(io.BytesIO(data), schema=_TEXT_BLOCKS_SCHEMA)
    result: list[TextBlock] = []
    for i in range(len(tbl)):
        row = {col: tbl.column(col)[i].as_py() for col in tbl.column_names}
        result.append(
            TextBlock(
                run_id=row["run_id"],
                source_registry_id=row["source_registry_id"],
                source_sha256=row["source_sha256"],
                page_number=int(row["page_number"]),
                block_index=int(row["block_index"]),
                text=row["text"],
                block_type=row["block_type"],
                extraction_timestamp_utc=row["extraction_timestamp_utc"],
            )
        )
    return result


def parquet_to_tables(data: bytes) -> list["ExtractedTable"]:
    """Deserialise tables.parquet bytes to ExtractedTable instances."""
    from .models import ExtractedTable

    tbl = pq.read_table(io.BytesIO(data), schema=_TABLES_SCHEMA)
    result: list[ExtractedTable] = []
    for i in range(len(tbl)):
        row = {col: tbl.column(col)[i].as_py() for col in tbl.column_names}
        result.append(
            ExtractedTable(
                run_id=row["run_id"],
                source_registry_id=row["source_registry_id"],
                source_sha256=row["source_sha256"],
                page_number=int(row["page_number"]),
                extractor_used=row["extractor_used"],
                table_index=int(row["table_index"]),
                header_row=row["header_row"],
                data_rows=row["data_rows"],
                extraction_timestamp_utc=row["extraction_timestamp_utc"],
            )
        )
    return result


def parquet_to_metadata(data: bytes) -> "ExtractionMetadata":
    """Deserialise metadata.parquet bytes to ExtractionMetadata."""
    from .models import ExtractionMetadata

    tbl = pq.read_table(io.BytesIO(data), schema=_METADATA_SCHEMA)
    if len(tbl) == 0:
        raise ValueError("metadata.parquet is empty")
    row = {col: tbl.column(col)[0].as_py() for col in tbl.column_names}
    return ExtractionMetadata(
        run_id=row["run_id"],
        source_registry_id=row["source_registry_id"],
        source_sha256=row["source_sha256"],
        format_detected=row["format_detected"],
        extractor_used=row["extractor_used"],
        page_count=int(row["page_count"]),
        table_count=int(row["table_count"]),
        text_block_count=int(row["text_block_count"]),
        extraction_timestamp_utc=row["extraction_timestamp_utc"],
    )
