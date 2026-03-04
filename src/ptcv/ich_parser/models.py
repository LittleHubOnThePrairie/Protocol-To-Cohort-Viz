"""PTCV ICH Parser data models.

IchSection is the core output unit — one row per classified ICH E6(R3)
Appendix B section per protocol. ReviewQueueEntry is written to SQLite
for any section where confidence_score < 0.70.

Risk tier: MEDIUM — data pipeline output (no patient data).

Regulatory references:
- ALCOA+ Accurate: confidence_score is required and non-null
- ALCOA+ Contemporaneous: extraction_timestamp_utc captured at write time
- ALCOA+ Traceable: source_run_id / source_sha256 link back to PTCV-19
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class IchSection:
    """One classified ICH E6(R3) Appendix B section from a protocol.

    Attributes:
        run_id: UUID4 for this ICH-parse pipeline run.
        source_run_id: run_id from the PTCV-19 extraction that produced
            the input text. Empty string if input was provided directly.
        source_sha256: SHA-256 of the PTCV-19 extraction Parquet artifact
            (or of the raw text bytes when source_run_id is empty).
        registry_id: Trial registry identifier (EUCT-Code or NCT-ID).
        section_code: ICH E6(R3) section code, e.g. "B.1", "B.7".
        section_name: Human-readable section name, e.g. "Treatment".
        content_json: Extracted section content serialised as a JSON
            string (keys depend on section type).
        confidence_score: RAG or rule-based confidence 0.0–1.0. Required
            non-null column (ALCOA+ Accurate).
        review_required: True when confidence_score < 0.70.
        legacy_format: True when fallback section detection was used
            (pre-ICH-E6(R3) protocol or non-standard headings).
        extraction_timestamp_utc: ISO 8601 UTC timestamp of the write.
            Populated by IchParser immediately before storing.
        content_text: Full assembled text for this section from the
            original protocol (PTCV-64). Empty string for legacy
            classification-only results. When populated, contains the
            complete un-truncated text assigned to this ICH section.
    """

    run_id: str
    source_run_id: str
    source_sha256: str
    registry_id: str
    section_code: str
    section_name: str
    content_json: str
    confidence_score: float
    review_required: bool
    legacy_format: bool
    extraction_timestamp_utc: str = ""
    content_text: str = ""


@dataclasses.dataclass
class ReviewQueueEntry:
    """Row written to review_queue.db for low-confidence sections.

    Attributes:
        run_id: UUID4 for this ICH-parse pipeline run.
        registry_id: Trial registry identifier.
        section_code: ICH section code of the flagged section.
        confidence_score: Confidence at the time of flagging.
        content_json: Section content as serialised JSON string.
        queue_timestamp_utc: ISO 8601 UTC insertion time.
    """

    run_id: str
    registry_id: str
    section_code: str
    confidence_score: float
    content_json: str
    queue_timestamp_utc: str
    id: Optional[int] = None
