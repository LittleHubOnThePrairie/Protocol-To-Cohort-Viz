"""USDM Parquet writer — serialises five USDM entity tables to Parquet.

Produces:
  usdm/{run_id}/epochs.parquet
  usdm/{run_id}/timepoints.parquet
  usdm/{run_id}/activities.parquet
  usdm/{run_id}/scheduled_instances.parquet
  usdm/{run_id}/synonym_mappings.parquet

Each schema is enforced at write time. The timepoints schema enforces
non-null constraints on the seven required USDM attributes (ALCOA+ Accurate).
All tables use Snappy compression.

Risk tier: MEDIUM — data pipeline storage (no patient data).

Regulatory references:
- ALCOA+ Accurate: seven timepoint attributes are non-nullable in schema
- ALCOA+ Original: caller passes immutable=False; re-runs write new run_ids
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import io

from .models import (
    SynonymMapping,
    UsdmActivity,
    UsdmEpoch,
    UsdmScheduledInstance,
    UsdmTimepoint,
)


# ---------------------------------------------------------------------------
# PyArrow schemas
# ---------------------------------------------------------------------------

_EPOCH_SCHEMA = pa.schema([
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("source_run_id", pa.string(), nullable=False),
    pa.field("source_sha256", pa.string(), nullable=False),
    pa.field("registry_id", pa.string(), nullable=False),
    pa.field("epoch_id", pa.string(), nullable=False),
    pa.field("epoch_name", pa.string(), nullable=False),
    pa.field("epoch_type", pa.string(), nullable=False),
    pa.field("order", pa.int32(), nullable=False),
    pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
])

_TIMEPOINT_SCHEMA = pa.schema([
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("source_run_id", pa.string(), nullable=False),
    pa.field("source_sha256", pa.string(), nullable=False),
    pa.field("registry_id", pa.string(), nullable=False),
    pa.field("timepoint_id", pa.string(), nullable=False),
    pa.field("epoch_id", pa.string(), nullable=False),
    # Seven required USDM attributes — all nullable=False (ALCOA+ Accurate)
    pa.field("visit_name", pa.string(), nullable=False),
    pa.field("visit_type", pa.string(), nullable=False),
    pa.field("day_offset", pa.int32(), nullable=False),
    pa.field("window_early", pa.int32(), nullable=False),
    pa.field("window_late", pa.int32(), nullable=False),
    pa.field("mandatory", pa.bool_(), nullable=False),
    pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
    # Optional attributes
    pa.field("repeat_cycle", pa.string(), nullable=True),
    pa.field("conditional_rule", pa.string(), nullable=True),
    pa.field("review_required", pa.bool_(), nullable=False),
])

_ACTIVITY_SCHEMA = pa.schema([
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("source_run_id", pa.string(), nullable=False),
    pa.field("source_sha256", pa.string(), nullable=False),
    pa.field("registry_id", pa.string(), nullable=False),
    pa.field("activity_id", pa.string(), nullable=False),
    pa.field("activity_name", pa.string(), nullable=False),
    pa.field("activity_type", pa.string(), nullable=False),
    pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
])

_INSTANCE_SCHEMA = pa.schema([
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("source_run_id", pa.string(), nullable=False),
    pa.field("source_sha256", pa.string(), nullable=False),
    pa.field("registry_id", pa.string(), nullable=False),
    pa.field("instance_id", pa.string(), nullable=False),
    pa.field("activity_id", pa.string(), nullable=False),
    pa.field("timepoint_id", pa.string(), nullable=False),
    pa.field("scheduled", pa.bool_(), nullable=False),
    pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
])

_SYNONYM_SCHEMA = pa.schema([
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("original_text", pa.string(), nullable=False),
    pa.field("canonical_label", pa.string(), nullable=False),
    pa.field("method", pa.string(), nullable=False),
    pa.field("confidence", pa.float32(), nullable=False),
    pa.field("review_required", pa.bool_(), nullable=False),
    pa.field("extraction_timestamp_utc", pa.string(), nullable=False),
])


# ---------------------------------------------------------------------------
# Writer class
# ---------------------------------------------------------------------------

class UsdmParquetWriter:
    """Serialise USDM entity collections to Parquet bytes.

    All methods return Snappy-compressed Parquet bytes suitable for
    passing directly to StorageGateway.put_artifact().
    """

    # ------------------------------------------------------------------
    # Serialise (dataclass → bytes)
    # ------------------------------------------------------------------

    def epochs_to_parquet(self, epochs: list[UsdmEpoch]) -> bytes:
        """Serialise UsdmEpoch records to Parquet.

        Args:
            epochs: Non-empty list of UsdmEpoch instances.

        Returns:
            Snappy-compressed Parquet bytes.

        Raises:
            ValueError: If epochs is empty.
        """
        if not epochs:
            raise ValueError("epochs must not be empty")
        table = pa.table(
            {
                "run_id": [e.run_id for e in epochs],
                "source_run_id": [e.source_run_id for e in epochs],
                "source_sha256": [e.source_sha256 for e in epochs],
                "registry_id": [e.registry_id for e in epochs],
                "epoch_id": [e.epoch_id for e in epochs],
                "epoch_name": [e.epoch_name for e in epochs],
                "epoch_type": [e.epoch_type for e in epochs],
                "order": [e.order for e in epochs],
                "extraction_timestamp_utc": [
                    e.extraction_timestamp_utc for e in epochs
                ],
            },
            schema=_EPOCH_SCHEMA,
        )
        return _write(table)

    def timepoints_to_parquet(self, timepoints: list[UsdmTimepoint]) -> bytes:
        """Serialise UsdmTimepoint records to Parquet.

        Validates that all seven required attributes are non-null before
        writing (ALCOA+ Accurate). Missing values raise ValueError.

        Args:
            timepoints: Non-empty list of UsdmTimepoint instances.

        Returns:
            Snappy-compressed Parquet bytes.

        Raises:
            ValueError: If timepoints is empty or any required field is empty.
        [PTCV-21 Scenario: All 7 required SoA attributes present]
        """
        if not timepoints:
            raise ValueError("timepoints must not be empty")
        for tp in timepoints:
            if not tp.visit_name:
                raise ValueError(
                    f"timepoint {tp.timepoint_id}: visit_name is required"
                )
            if not tp.visit_type:
                raise ValueError(
                    f"timepoint {tp.timepoint_id}: visit_type is required"
                )
            if not tp.extraction_timestamp_utc:
                raise ValueError(
                    f"timepoint {tp.timepoint_id}: extraction_timestamp_utc is required"
                )
        table = pa.table(
            {
                "run_id": [t.run_id for t in timepoints],
                "source_run_id": [t.source_run_id for t in timepoints],
                "source_sha256": [t.source_sha256 for t in timepoints],
                "registry_id": [t.registry_id for t in timepoints],
                "timepoint_id": [t.timepoint_id for t in timepoints],
                "epoch_id": [t.epoch_id for t in timepoints],
                "visit_name": [t.visit_name for t in timepoints],
                "visit_type": [t.visit_type for t in timepoints],
                "day_offset": [t.day_offset for t in timepoints],
                "window_early": [t.window_early for t in timepoints],
                "window_late": [t.window_late for t in timepoints],
                "mandatory": [t.mandatory for t in timepoints],
                "extraction_timestamp_utc": [
                    t.extraction_timestamp_utc for t in timepoints
                ],
                "repeat_cycle": [t.repeat_cycle for t in timepoints],
                "conditional_rule": [t.conditional_rule for t in timepoints],
                "review_required": [t.review_required for t in timepoints],
            },
            schema=_TIMEPOINT_SCHEMA,
        )
        return _write(table)

    def activities_to_parquet(self, activities: list[UsdmActivity]) -> bytes:
        """Serialise UsdmActivity records to Parquet.

        Args:
            activities: Non-empty list of UsdmActivity instances.

        Returns:
            Snappy-compressed Parquet bytes.

        Raises:
            ValueError: If activities is empty.
        """
        if not activities:
            raise ValueError("activities must not be empty")
        table = pa.table(
            {
                "run_id": [a.run_id for a in activities],
                "source_run_id": [a.source_run_id for a in activities],
                "source_sha256": [a.source_sha256 for a in activities],
                "registry_id": [a.registry_id for a in activities],
                "activity_id": [a.activity_id for a in activities],
                "activity_name": [a.activity_name for a in activities],
                "activity_type": [a.activity_type for a in activities],
                "extraction_timestamp_utc": [
                    a.extraction_timestamp_utc for a in activities
                ],
            },
            schema=_ACTIVITY_SCHEMA,
        )
        return _write(table)

    def instances_to_parquet(
        self, instances: list[UsdmScheduledInstance]
    ) -> bytes:
        """Serialise UsdmScheduledInstance records to Parquet.

        Args:
            instances: Non-empty list of UsdmScheduledInstance instances.

        Returns:
            Snappy-compressed Parquet bytes.

        Raises:
            ValueError: If instances is empty.
        """
        if not instances:
            raise ValueError("instances must not be empty")
        table = pa.table(
            {
                "run_id": [i.run_id for i in instances],
                "source_run_id": [i.source_run_id for i in instances],
                "source_sha256": [i.source_sha256 for i in instances],
                "registry_id": [i.registry_id for i in instances],
                "instance_id": [i.instance_id for i in instances],
                "activity_id": [i.activity_id for i in instances],
                "timepoint_id": [i.timepoint_id for i in instances],
                "scheduled": [i.scheduled for i in instances],
                "extraction_timestamp_utc": [
                    i.extraction_timestamp_utc for i in instances
                ],
            },
            schema=_INSTANCE_SCHEMA,
        )
        return _write(table)

    def synonyms_to_parquet(self, synonyms: list[SynonymMapping]) -> bytes:
        """Serialise SynonymMapping records to Parquet.

        Args:
            synonyms: Non-empty list of SynonymMapping instances.

        Returns:
            Snappy-compressed Parquet bytes.

        Raises:
            ValueError: If synonyms is empty.
        [PTCV-21 Scenario: Synonym resolution logged in synonym_mappings]
        """
        if not synonyms:
            raise ValueError("synonyms must not be empty")
        table = pa.table(
            {
                "run_id": [s.run_id for s in synonyms],
                "original_text": [s.original_text for s in synonyms],
                "canonical_label": [s.canonical_label for s in synonyms],
                "method": [s.method for s in synonyms],
                "confidence": [s.confidence for s in synonyms],
                "review_required": [s.review_required for s in synonyms],
                "extraction_timestamp_utc": [
                    s.extraction_timestamp_utc for s in synonyms
                ],
            },
            schema=_SYNONYM_SCHEMA,
        )
        return _write(table)

    # ------------------------------------------------------------------
    # Deserialise (bytes → dataclass)
    # ------------------------------------------------------------------

    @staticmethod
    def parquet_to_timepoints(data: bytes) -> list[UsdmTimepoint]:
        """Deserialise Parquet bytes to UsdmTimepoint list."""
        buf = pa.BufferReader(data)
        table = pq.read_table(buf)
        result: list[UsdmTimepoint] = []
        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.schema.names}
            result.append(
                UsdmTimepoint(
                    run_id=row["run_id"],
                    source_run_id=row["source_run_id"],
                    source_sha256=row["source_sha256"],
                    registry_id=row["registry_id"],
                    timepoint_id=row["timepoint_id"],
                    epoch_id=row["epoch_id"],
                    visit_name=row["visit_name"],
                    visit_type=row["visit_type"],
                    day_offset=int(row["day_offset"]),
                    window_early=int(row["window_early"]),
                    window_late=int(row["window_late"]),
                    mandatory=bool(row["mandatory"]),
                    repeat_cycle=row.get("repeat_cycle") or "",
                    conditional_rule=row.get("conditional_rule") or "",
                    review_required=bool(row["review_required"]),
                    extraction_timestamp_utc=row["extraction_timestamp_utc"],
                )
            )
        return result

    @staticmethod
    def parquet_to_synonyms(data: bytes) -> list[SynonymMapping]:
        """Deserialise Parquet bytes to SynonymMapping list."""
        buf = pa.BufferReader(data)
        table = pq.read_table(buf)
        result: list[SynonymMapping] = []
        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.schema.names}
            result.append(
                SynonymMapping(
                    run_id=row["run_id"],
                    original_text=row["original_text"],
                    canonical_label=row["canonical_label"],
                    method=row["method"],
                    confidence=float(row["confidence"]),
                    review_required=bool(row["review_required"]),
                    extraction_timestamp_utc=row["extraction_timestamp_utc"],
                )
            )
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _write(table: pa.Table) -> bytes:
    """Write a PyArrow Table to Snappy-compressed Parquet bytes."""
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()
