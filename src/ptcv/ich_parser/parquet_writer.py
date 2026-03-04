"""DuckDB + Parquet serialiser for ICH section output.

Serialises a list of IchSection instances to Parquet bytes using
PyArrow, matching the schema defined in the PTCV-20 spec. The bytes
are then stored via StorageGateway.put_artifact().

Schema enforces:
- confidence_score: float32, required (no nulls) — ALCOA+ Accurate
- All other columns: nullable string / bool / timestamp

Risk tier: MEDIUM — data pipeline output (no patient data).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from .models import IchSection


# Parquet schema for sections.parquet (PTCV-20 spec)
_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("source_run_id", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("registry_id", pa.string(), nullable=False),
        pa.field("section_code", pa.string(), nullable=False),
        pa.field("section_name", pa.string(), nullable=False),
        pa.field("content_json", pa.string(), nullable=False),
        # confidence_score is required non-null (ALCOA+ Accurate)
        pa.field("confidence_score", pa.float32(), nullable=False),
        pa.field("review_required", pa.bool_(), nullable=False),
        pa.field("legacy_format", pa.bool_(), nullable=False),
        pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
        pa.field("content_text", pa.string(), nullable=True),
    ]
)


def sections_to_parquet(sections: list["IchSection"]) -> bytes:
    """Serialise IchSection rows to Parquet bytes.

    Args:
        sections: Non-empty list of IchSection instances. All must have
            extraction_timestamp_utc set (non-empty string).

    Returns:
        Parquet file contents as bytes.

    Raises:
        ValueError: If sections is empty or any confidence_score is None.
    [PTCV-20 Scenario: Confidence score persisted as required column]
    """
    if not sections:
        raise ValueError("sections list must not be empty")

    for sec in sections:
        if sec.confidence_score is None:
            raise ValueError(
                f"confidence_score is None for section {sec.section_code} "
                f"in run {sec.run_id}. ALCOA+ Accurate: this field is required."
            )
        if not sec.extraction_timestamp_utc:
            raise ValueError(
                f"extraction_timestamp_utc is empty for {sec.section_code}. "
                "Set timestamp before calling sections_to_parquet()."
            )

    table = pa.table(
        {
            "run_id": [s.run_id for s in sections],
            "source_run_id": [s.source_run_id for s in sections],
            "source_sha256": [s.source_sha256 for s in sections],
            "registry_id": [s.registry_id for s in sections],
            "section_code": [s.section_code for s in sections],
            "section_name": [s.section_name for s in sections],
            "content_json": [s.content_json for s in sections],
            "confidence_score": pa.array(
                [s.confidence_score for s in sections], type=pa.float32()
            ),
            "review_required": [s.review_required for s in sections],
            "legacy_format": [s.legacy_format for s in sections],
            "extraction_timestamp_utc": [
                s.extraction_timestamp_utc for s in sections
            ],
            "content_text": [
                s.content_text or None for s in sections
            ],
        },
        schema=_SCHEMA,
    )

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def parquet_to_sections(data: bytes) -> list["IchSection"]:
    """Deserialise Parquet bytes to a list of IchSection instances.

    Args:
        data: Parquet file contents as bytes.

    Returns:
        List of IchSection instances.
    """
    from .models import IchSection

    table = pq.read_table(io.BytesIO(data))
    columns = set(table.column_names)
    sections: list[IchSection] = []
    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in columns}
        sections.append(
            IchSection(
                run_id=row["run_id"],
                source_run_id=row["source_run_id"],
                source_sha256=row["source_sha256"],
                registry_id=row["registry_id"],
                section_code=row["section_code"],
                section_name=row["section_name"],
                content_json=row["content_json"],
                confidence_score=float(row["confidence_score"]),
                review_required=bool(row["review_required"]),
                legacy_format=bool(row["legacy_format"]),
                extraction_timestamp_utc=row["extraction_timestamp_utc"],
                content_text=row.get("content_text", "") or "",
            )
        )
    return sections
