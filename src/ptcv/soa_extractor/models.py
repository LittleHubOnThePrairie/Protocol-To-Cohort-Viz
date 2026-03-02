"""PTCV SoA extractor data models.

CDISC USDM v4.0 output types for the Schedule of Activities pipeline.
Each entity type maps to one Parquet file under usdm/{run_id}/.

Risk tier: MEDIUM — data pipeline output (no patient data).

Regulatory references:
- ALCOA+ Accurate: seven required timepoint attributes are non-null
- ALCOA+ Contemporaneous: extraction_timestamp_utc captured at write time
- ALCOA+ Traceable: source_sha256 links back to PTCV-20 ICH sections artifact
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class UsdmEpoch:
    """CDISC USDM v4.0 Epoch — top-level study period grouping.

    Attributes:
        run_id: UUID4 for this SoA extraction run.
        source_run_id: run_id from the PTCV-20 ICH parse that produced input.
        source_sha256: SHA-256 of the PTCV-20 sections.parquet artifact.
        registry_id: Trial registry identifier (EUCT-Code or NCT-ID).
        epoch_id: Unique identifier within this run.
        epoch_name: Human-readable name, e.g. "Screening".
        epoch_type: One of Screening/Treatment/Follow-up/End of Study.
        order: Epoch sequence position (1-based).
        extraction_timestamp_utc: ISO 8601 UTC timestamp of the write.
    """

    run_id: str
    source_run_id: str
    source_sha256: str
    registry_id: str
    epoch_id: str
    epoch_name: str
    epoch_type: str
    order: int
    extraction_timestamp_utc: str = ""


@dataclasses.dataclass
class UsdmTimepoint:
    """CDISC USDM v4.0 Timepoint (protocol visit).

    Seven attributes are mandatory (ALCOA+ Accurate). Missing any one
    of these sets review_required=True for the row.

    Attributes:
        run_id: UUID4 for this SoA extraction run.
        source_run_id: run_id from PTCV-20.
        source_sha256: SHA-256 of the PTCV-20 sections.parquet artifact.
        registry_id: Trial identifier.
        timepoint_id: Unique ID within this run.
        epoch_id: Foreign key to UsdmEpoch.epoch_id.
        visit_name: Canonical resolved visit name. [REQUIRED]
        visit_type: One of 9 visit types. [REQUIRED]
        day_offset: VISITDY equivalent (relative day from Day 1). [REQUIRED]
        window_early: TVSTRL equivalent, days before day_offset. [REQUIRED]
        window_late: TVENRL equivalent, days after day_offset. [REQUIRED]
        mandatory: Whether the visit is a protocol-required visit. [REQUIRED]
        repeat_cycle: Cycle definition if repeating, e.g. "C2" (empty otherwise).
        conditional_rule: soaTransition rule: "PRN" or "EARLY_TERM" or "".
        review_required: True if any mandatory attribute was inferred/missing.
        extraction_timestamp_utc: ISO 8601 UTC timestamp. [REQUIRED]
    """

    run_id: str
    source_run_id: str
    source_sha256: str
    registry_id: str
    timepoint_id: str
    epoch_id: str
    visit_name: str              # mandatory
    visit_type: str              # mandatory
    day_offset: int              # mandatory
    window_early: int            # mandatory
    window_late: int             # mandatory
    mandatory: bool              # mandatory
    repeat_cycle: str = ""
    conditional_rule: str = ""
    review_required: bool = False
    extraction_timestamp_utc: str = ""   # mandatory


@dataclasses.dataclass
class UsdmActivity:
    """CDISC USDM v4.0 Activity (assessment, procedure, or lab test).

    Attributes:
        run_id: UUID4 for this SoA extraction run.
        source_run_id: run_id from PTCV-20.
        source_sha256: SHA-256 of the PTCV-20 sections.parquet artifact.
        registry_id: Trial identifier.
        activity_id: Unique ID within this run.
        activity_name: Canonical activity name from the SoA row header.
        activity_type: Category: Assessment/Procedure/Lab/Vital/ECG/Other.
        extraction_timestamp_utc: ISO 8601 UTC timestamp of the write.
    """

    run_id: str
    source_run_id: str
    source_sha256: str
    registry_id: str
    activity_id: str
    activity_name: str
    activity_type: str
    extraction_timestamp_utc: str = ""


@dataclasses.dataclass
class UsdmScheduledInstance:
    """CDISC USDM v4.0 ScheduledActivityInstance (Activity × Timepoint).

    One row for every (activity, timepoint) pair where the activity is
    scheduled at that visit. Unscheduled (X not present) pairs are omitted.

    Attributes:
        run_id: UUID4 for this SoA extraction run.
        source_run_id: run_id from PTCV-20.
        source_sha256: SHA-256 of the PTCV-20 sections.parquet artifact.
        registry_id: Trial identifier.
        instance_id: Unique ID within this run.
        activity_id: FK to UsdmActivity.activity_id.
        timepoint_id: FK to UsdmTimepoint.timepoint_id.
        scheduled: True (always True in the stored table; absent entries omitted).
        extraction_timestamp_utc: ISO 8601 UTC timestamp of the write.
    """

    run_id: str
    source_run_id: str
    source_sha256: str
    registry_id: str
    instance_id: str
    activity_id: str
    timepoint_id: str
    scheduled: bool = True
    extraction_timestamp_utc: str = ""


@dataclasses.dataclass
class SynonymMapping:
    """Visit synonym resolution audit record.

    One row per visit header resolved. Persisted in synonym_mappings.parquet
    and — when confidence < 0.80 — also written to review_queue.db.

    Attributes:
        run_id: UUID4 for this SoA extraction run.
        original_text: Raw visit header text as found in the SoA table.
        canonical_label: Resolved canonical visit type or name.
        method: Resolution method: "lookup", "regex", "spacy_ner", "default".
        confidence: Resolver confidence 0.0–1.0.
        review_required: True when confidence < 0.80.
        extraction_timestamp_utc: ISO 8601 UTC timestamp of the write.
    """

    run_id: str
    original_text: str
    canonical_label: str
    method: str
    confidence: float
    review_required: bool = False
    extraction_timestamp_utc: str = ""


@dataclasses.dataclass
class RawSoaTable:
    """Intermediate parsed SoA table before USDM mapping.

    Attributes:
        visit_headers: Ordered visit column headers from the SoA table.
        day_headers: Optional second row with day/window info per visit.
        activities: List of (activity_name, scheduled_flags) where
            scheduled_flags[i] is True if the activity is scheduled at
            visit_headers[i].
        section_code: ICH section code the table was extracted from.
    """

    visit_headers: list[str]
    day_headers: list[str]
    activities: list[tuple[str, list[bool]]]
    section_code: str


@dataclasses.dataclass
class ExtractResult:
    """Result returned by SoaExtractor.extract().

    Attributes:
        run_id: UUID4 for this SoA extraction run.
        registry_id: Trial identifier.
        epoch_count: Number of USDM epochs written.
        timepoint_count: Number of USDM timepoints written.
        activity_count: Number of USDM activities written.
        instance_count: Number of scheduled instances written.
        synonym_mapping_count: Number of synonym mappings recorded.
        review_count: Number of synonym mappings flagged for review.
        artifact_keys: Dict mapping artifact name to storage key.
        source_sha256: SHA-256 passed from upstream ICH artifact.
    """

    run_id: str
    registry_id: str
    epoch_count: int
    timepoint_count: int
    activity_count: int
    instance_count: int
    synonym_mapping_count: int
    review_count: int
    artifact_keys: dict[str, str]
    source_sha256: str

    def __repr__(self) -> str:
        return (
            f"ExtractResult(run_id={self.run_id!r}, "
            f"registry_id={self.registry_id!r}, "
            f"timepoints={self.timepoint_count}, "
            f"activities={self.activity_count}, "
            f"review={self.review_count})"
        )
