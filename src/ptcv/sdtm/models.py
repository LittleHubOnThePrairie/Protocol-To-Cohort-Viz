"""PTCV SDTM data models.

Covers generation result and CT review queue entries for PTCV-22.

Risk tier: MEDIUM — data pipeline output (no patient data).

Regulatory references:
- ALCOA+ Traceable: source_sha256 links each artifact to upstream USDM
- ALCOA+ Contemporaneous: generation_timestamp_utc captured at write time
- ALCOA+ Original: new run_id per amendment; prior XPT artifacts WORM-locked
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class CtReviewQueueEntry:
    """Row written to ct_review_queue.db for unmapped CT terms.

    Attributes:
        run_id: UUID4 for this SDTM generation run.
        registry_id: Trial registry identifier.
        domain: SDTM domain name (e.g. "TS").
        variable: Variable name within domain (e.g. "TSVAL").
        original_value: The raw value that could not be mapped to a CT term.
        ct_lookup_attempted: True when the static CT table was queried.
        queue_timestamp_utc: ISO 8601 UTC insertion time.
        id: Auto-incremented SQLite row ID (0 before insert).
    """

    run_id: str
    registry_id: str
    domain: str
    variable: str
    original_value: str
    ct_lookup_attempted: bool
    queue_timestamp_utc: str
    id: int = 0


@dataclasses.dataclass
class SdtmGenerationResult:
    """Result returned by SdtmService.generate().

    Attributes:
        run_id: UUID4 for this SDTM generation run.
        registry_id: Trial registry identifier.
        artifact_keys: Mapping from domain (e.g. "ts") to storage key.
        artifact_sha256s: Mapping from domain to ArtifactRecord sha256.
        source_sha256: SHA-256 of the upstream timepoints.parquet artifact.
        domain_row_counts: Mapping from domain name to row count.
        ct_unmapped_count: Number of CT values written to review queue.
        generation_timestamp_utc: ISO 8601 UTC timestamp of the write.
    """

    run_id: str
    registry_id: str
    artifact_keys: dict[str, str]
    artifact_sha256s: dict[str, str]
    source_sha256: str
    domain_row_counts: dict[str, int]
    ct_unmapped_count: int
    generation_timestamp_utc: str

    def __repr__(self) -> str:
        domains = list(self.artifact_keys.keys())
        return (
            f"SdtmGenerationResult(run_id={self.run_id!r}, "
            f"registry_id={self.registry_id!r}, "
            f"domains={domains}, "
            f"ct_unmapped={self.ct_unmapped_count})"
        )
