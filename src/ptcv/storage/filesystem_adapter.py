"""FilesystemAdapter — local filesystem + SQLite StorageGateway.

Default adapter for development and testing. Requires no external
services. Artifacts are stored under ``{root}/{key}`` using the same
directory layout as the legacy FilestoreManager. SHA-256 is computed
before writing. An append-only SQLite database at ``{root}/lineage.db``
records every put_artifact() call.

Risk tier: MEDIUM — data pipeline storage (dev/test).

Regulatory references:
- ALCOA+ Original: immutable=True raises FileExistsError on duplicate
- ALCOA+ Traceable: every write appended to lineage.db (no UPDATE/DELETE)
- 21 CFR 11.10(e): lineage chain serves as the audit trail
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .gateway import StorageGateway
from .models import ArtifactRecord, LineageRecord
from . import _lineage_db


class FilesystemAdapter(StorageGateway):
    """StorageGateway backed by the local filesystem and SQLite.

    File layout under ``root``:

    .. code-block:: text

        root/
            eu-ctr/                # EU-CTR protocol files
            clinicaltrials/        # ClinicalTrials.gov protocol files
            metadata/              # Metadata JSON files
            lineage.db             # SQLite append-only lineage chain

    Args:
        root: Filesystem root directory. All keys are resolved relative
            to this path.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._db_path = root / "lineage.db"

    # ------------------------------------------------------------------
    # StorageGateway interface
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Create directories and SQLite schema. Idempotent.

        [PTCV-29 Scenario: Initialise WORM bucket and lineage schema]
        """
        for subdir in ("eu-ctr", "clinicaltrials", "metadata"):
            (self.root / subdir).mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            _lineage_db.create_schema(conn)
        finally:
            conn.close()

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
        """Write bytes to ``{root}/{key}`` and append a lineage record.

        Args:
            key: Relative storage path (e.g.
                "clinicaltrials/NCT00112827_00.pdf").
            data: Raw bytes to store.
            content_type: MIME type string.
            run_id: UUID4 pipeline run identifier.
            source_hash: SHA-256 of the upstream artifact (or "").
            user: Actor identifier.
            immutable: Raise FileExistsError if file already exists.
            stage: Pipeline stage name for the lineage record.
            registry_id: Optional trial identifier for lineage.
            amendment_number: Optional amendment number for lineage.
            source: Optional registry source name for lineage.

        Returns:
            ArtifactRecord with SHA-256 and empty version_id.

        Raises:
            FileExistsError: If immutable=True and file already exists.
        [PTCV-29 Scenario: put_artifact records lock and lineage]
        """
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)

        if immutable and dest.exists():
            raise FileExistsError(
                f"Artifact already exists at {dest}. "
                "ALCOA+ Original: do not overwrite immutable artifact."
            )

        sha256 = hashlib.sha256(data).hexdigest()
        dest.write_bytes(data)

        timestamp = datetime.now(timezone.utc).isoformat()
        artifact = ArtifactRecord(
            key=key,
            version_id="",
            sha256=sha256,
            content_type=content_type,
            timestamp_utc=timestamp,
            run_id=run_id,
            user=user,
        )

        lineage = LineageRecord(
            id=0,
            run_id=run_id,
            stage=stage,
            artifact_key=key,
            version_id="",
            sha256=sha256,
            source_hash=source_hash,
            user=user,
            timestamp_utc=timestamp,
            registry_id=registry_id,
            amendment_number=amendment_number,
            source=source,
        )
        conn = self._connect()
        try:
            _lineage_db.insert_record(conn, lineage)
        finally:
            conn.close()

        return artifact

    def get_artifact(self, key: str, version_id: str = "") -> bytes:
        """Read and return bytes stored at ``{root}/{key}``.

        Args:
            key: Storage key.
            version_id: Ignored (no versioning on filesystem).

        Returns:
            Raw bytes.

        Raises:
            FileNotFoundError: If the key does not exist.
        """
        path = self.root / key
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")
        return path.read_bytes()

    def list_versions(self, key: str) -> list[ArtifactRecord]:
        """Return a single-element list for the stored artifact.

        Args:
            key: Storage key.

        Returns:
            List with one ArtifactRecord, or empty list if not found.
        """
        path = self.root / key
        if not path.exists():
            return []
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        mtime = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        return [
            ArtifactRecord(
                key=key,
                version_id="",
                sha256=sha256,
                content_type="application/octet-stream",
                timestamp_utc=mtime,
                run_id="",
                user="",
            )
        ]

    def get_lineage(self, run_id: str) -> list[LineageRecord]:
        """Return all lineage records for a pipeline run.

        Args:
            run_id: UUID4 run identifier.

        Returns:
            List of LineageRecord ordered by insertion sequence.
        [PTCV-29 Scenario: get_lineage returns run records]
        """
        conn = self._connect()
        try:
            return _lineage_db.query_by_run_id(conn, run_id)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection to lineage.db."""
        return sqlite3.connect(str(self._db_path))
