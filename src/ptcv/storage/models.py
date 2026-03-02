"""PTCV Storage Gateway data models.

ArtifactRecord and LineageRecord are the core data transfer objects shared
by all StorageGateway implementations (FilesystemAdapter, LocalStorageAdapter).

Risk tier: MEDIUM — data pipeline storage.

Regulatory references:
- ALCOA+ Attributable: user field on every record
- ALCOA+ Traceable: run_id links artifacts to the download pipeline run
- 21 CFR 11.10(e): audit trail for storage operations via lineage chain
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ArtifactRecord:
    """Metadata returned after a successful put_artifact() call.

    Attributes:
        key: Storage key / relative path (e.g.
            "protocols/clinicaltrials/NCT12345678_00.pdf").
        version_id: Object version identifier. Non-empty for MinIO-backed
            storage; empty string for FilesystemAdapter (no versioning).
        sha256: Hex-encoded SHA-256 digest of the stored bytes.
        content_type: MIME type (e.g. "application/pdf").
        timestamp_utc: ISO 8601 UTC timestamp of the store operation.
        run_id: UUID4 linking this artifact to the download pipeline run.
        user: Identifier of the actor that triggered the store operation.
    """

    key: str
    version_id: str
    sha256: str
    content_type: str
    timestamp_utc: str
    run_id: str
    user: str


@dataclasses.dataclass
class LineageRecord:
    """Append-only provenance record stored in the SQLite lineage chain.

    One LineageRecord is written per put_artifact() call, linking the
    produced artifact to its upstream source artifact (if any).

    Attributes:
        id: Auto-incremented row ID (set to 0 before INSERT; populated
            by query helpers after INSERT).
        run_id: UUID4 shared with the corresponding ArtifactRecord.
        stage: Pipeline stage name (e.g. "download", "extraction").
        artifact_key: Storage key of the produced artifact.
        version_id: Version ID of the produced artifact (matches
            ArtifactRecord.version_id).
        sha256: Hex SHA-256 of the produced artifact bytes.
        source_hash: SHA-256 of the upstream artifact bytes. Empty
            string for initial ingestion stages (no upstream artifact).
        user: Actor identifier (matches ArtifactRecord.user).
        timestamp_utc: ISO 8601 UTC timestamp.
        registry_id: Trial registry identifier (e.g. "NCT00112827").
            Optional — populated for protocol download stages.
        amendment_number: Protocol amendment number (e.g. "00").
            Optional — populated for protocol download stages.
        source: Registry source name (e.g. "ClinicalTrials.gov",
            "EU-CTR"). Optional.
    """

    id: int
    run_id: str
    stage: str
    artifact_key: str
    version_id: str
    sha256: str
    source_hash: str
    user: str
    timestamp_utc: str
    registry_id: Optional[str] = None
    amendment_number: Optional[str] = None
    source: Optional[str] = None
