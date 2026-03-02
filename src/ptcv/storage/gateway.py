"""StorageGateway abstract base class.

All storage backends (FilesystemAdapter, LocalStorageAdapter) implement
this interface. Services depend on StorageGateway — they never import a
concrete adapter directly. This decouples production MinIO WORM storage
from the local filesystem used in development and tests.

Risk tier: MEDIUM — data pipeline storage interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .models import ArtifactRecord, LineageRecord


class StorageGateway(ABC):
    """Abstract interface for PTCV artifact storage.

    Implementations must guarantee:
    - SHA-256 is computed over the bytes *before* writing.
    - Every put_artifact() call appends one LineageRecord to the lineage
      chain (SQLite triggers prevent UPDATE/DELETE).
    - When immutable=True, a second write to the same key raises
      FileExistsError (ALCOA+ Original principle).
    """

    @abstractmethod
    def initialise(self) -> None:
        """Create storage structures (directories, bucket, schema).

        Safe to call repeatedly — implementations must be idempotent.
        [PTCV-29 Scenario: Initialise WORM bucket and lineage schema]
        """

    @abstractmethod
    def put_artifact(
        self,
        key: str,
        data: bytes,
        content_type: str,
        run_id: str,
        source_hash: str,
        user: str,
        immutable: bool = False,
        stage: str = "download",
        registry_id: Optional[str] = None,
        amendment_number: Optional[str] = None,
        source: Optional[str] = None,
    ) -> ArtifactRecord:
        """Store bytes and record lineage.

        Args:
            key: Storage key / relative path for the artifact.
            data: Raw bytes to store.
            content_type: MIME type string.
            run_id: UUID4 that groups all artifacts from one pipeline run.
            source_hash: SHA-256 of the upstream artifact. Pass "" for
                initial ingestion (no upstream).
            user: Actor identifier for audit trail.
            immutable: If True, raise FileExistsError when the key
                already exists (ALCOA+ Original principle).
            stage: Pipeline stage name for the lineage record (e.g.
                "download", "extraction", "ich_parse", "usdm_map",
                "sdtm_gen", "validation"). Default "download".
            registry_id: Trial registry identifier (lineage metadata).
            amendment_number: Protocol amendment number (lineage metadata).
            source: Registry source name (lineage metadata).

        Returns:
            ArtifactRecord describing the stored artifact.

        Raises:
            FileExistsError: If immutable=True and key already exists.
        [PTCV-29 Scenario: put_artifact records lock and lineage]
        """

    @abstractmethod
    def get_artifact(self, key: str, version_id: str = "") -> bytes:
        """Retrieve stored bytes.

        Args:
            key: Storage key used in put_artifact().
            version_id: Specific version to retrieve. If empty, returns
                the latest version.

        Returns:
            Raw bytes of the stored artifact.

        Raises:
            FileNotFoundError: If the key does not exist.
        """

    @abstractmethod
    def list_versions(self, key: str) -> list[ArtifactRecord]:
        """List all stored versions of a key.

        For FilesystemAdapter this always returns a single-element list
        (no versioning). For LocalStorageAdapter it returns one entry per
        MinIO object version.

        Args:
            key: Storage key to query.

        Returns:
            List of ArtifactRecord, ordered oldest-first.
        """

    @abstractmethod
    def get_lineage(self, run_id: str) -> list[LineageRecord]:
        """Return all lineage records for a pipeline run.

        Args:
            run_id: UUID4 run identifier.

        Returns:
            List of LineageRecord ordered by insertion time.
        [PTCV-29 Scenario: get_lineage returns run records]
        """
