"""LocalStorageAdapter — MinIO WORM + SQLite StorageGateway.

Production adapter for GxP-compliant object storage. Uses MinIO with
Compliance-mode Object Lock (WORM — Write Once, Read Many) and an
append-only SQLite lineage chain. Every put_artifact(immutable=True)
call sets a 99-year COMPLIANCE retention lock on the object, preventing
any deletion or overwrite — even by the bucket owner.

Risk tier: MEDIUM — data pipeline storage (production).

Regulatory references:
- ALCOA+ Original: COMPLIANCE retention lock prevents overwrite
- ALCOA+ Traceable: every write appended to lineage.db (no UPDATE/DELETE)
- 21 CFR 11.10(e): lineage chain serves as the audit trail

Dependencies:
    minio>=7.2.0 (install separately; not required for dev/test)
"""

from __future__ import annotations

import hashlib
import io
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .gateway import StorageGateway
from .models import ArtifactRecord, LineageRecord
from . import _lineage_db

# MinIO is an optional production dependency.
# Tests mock it via unittest.mock.patch("ptcv.storage.local_adapter.Minio").
try:
    from minio import Minio  # type: ignore[import]
    from minio.commonconfig import COMPLIANCE  # type: ignore[import]
    from minio.objectlockconfig import ObjectLockConfig  # type: ignore[import]
    from minio.retention import Retention  # type: ignore[import]
    _MINIO_AVAILABLE = True
except ImportError:
    _MINIO_AVAILABLE = False
    Minio = None  # type: ignore[misc,assignment]
    COMPLIANCE = "COMPLIANCE"  # type: ignore[assignment]

    class ObjectLockConfig:  # type: ignore[misc]
        """Stub when minio is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class Retention:  # type: ignore[misc]
        """Stub when minio is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass


# 99-year WORM retention (ALCOA+ Original)
_RETENTION_YEARS = 99


class LocalStorageAdapter(StorageGateway):
    """StorageGateway backed by MinIO WORM and SQLite lineage.

    Args:
        endpoint: MinIO server host:port (e.g. "localhost:9000").
        access_key: MinIO access key.
        secret_key: MinIO secret key.
        bucket: Bucket name (default "protocols").
        db_path: Path to SQLite lineage database file.
        secure: Use HTTPS. Default False for local dev.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str = "protocols",
        db_path: Optional[Path] = None,
        secure: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._bucket = bucket
        self._db_path = db_path or Path("lineage.db")
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    # ------------------------------------------------------------------
    # StorageGateway interface
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Create bucket with Object Lock config and SQLite schema.

        Creates the bucket only if it does not already exist. If the
        bucket exists, no changes are made to its configuration. SQLite
        schema creation is also idempotent.

        [PTCV-29 Scenario: Initialise WORM bucket and lineage schema]
        """
        if not self._client.bucket_exists(self._bucket):
            lock_config = ObjectLockConfig(COMPLIANCE, str(_RETENTION_YEARS), "Years")
            self._client.make_bucket(
                self._bucket, object_lock=True
            )
            self._client.set_object_lock_config(self._bucket, lock_config)
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
        registry_id: Optional[str] = None,
        amendment_number: Optional[str] = None,
        source: Optional[str] = None,
    ) -> ArtifactRecord:
        """Upload bytes to MinIO and append a lineage record.

        When immutable=True, a COMPLIANCE retention lock is applied to
        the uploaded object (99-year lock). When immutable=False, the
        object is uploaded without a retention lock (metadata artifacts).

        Args:
            key: MinIO object key.
            data: Raw bytes to upload.
            content_type: MIME type string.
            run_id: UUID4 pipeline run identifier.
            source_hash: SHA-256 of the upstream artifact (or "").
            user: Actor identifier.
            immutable: Apply COMPLIANCE retention lock if True.
            registry_id: Optional trial identifier for lineage.
            amendment_number: Optional amendment number for lineage.
            source: Optional registry source name for lineage.

        Returns:
            ArtifactRecord with SHA-256 and MinIO version_id.

        Raises:
            FileExistsError: If immutable=True and key already exists.
        [PTCV-29 Scenario: put_artifact records lock and lineage]
        """
        sha256 = hashlib.sha256(data).hexdigest()
        length = len(data)

        if immutable:
            try:
                stat = self._client.stat_object(self._bucket, key)
                raise FileExistsError(
                    f"Immutable artifact already exists: {key} "
                    f"(version {stat.version_id}). "
                    "ALCOA+ Original: do not overwrite WORM artifact."
                )
            except Exception as exc:
                # Re-raise FileExistsError; swallow "not found" errors
                if isinstance(exc, FileExistsError):
                    raise

        result = self._client.put_object(
            self._bucket,
            key,
            io.BytesIO(data),
            length,
            content_type=content_type,
        )
        version_id = result.version_id or ""

        if immutable:
            retain_until = datetime.now(timezone.utc) + timedelta(
                days=_RETENTION_YEARS * 365
            )
            self._client.set_object_retention(
                self._bucket,
                key,
                Retention(COMPLIANCE, retain_until),
                version_id=version_id,
            )

        timestamp = datetime.now(timezone.utc).isoformat()
        artifact = ArtifactRecord(
            key=key,
            version_id=version_id,
            sha256=sha256,
            content_type=content_type,
            timestamp_utc=timestamp,
            run_id=run_id,
            user=user,
        )

        lineage = LineageRecord(
            id=0,
            run_id=run_id,
            stage="download",
            artifact_key=key,
            version_id=version_id,
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
        """Download object bytes from MinIO.

        Args:
            key: MinIO object key.
            version_id: Specific version to retrieve. If empty, the
                latest version is returned.

        Returns:
            Raw bytes of the stored object.

        Raises:
            FileNotFoundError: If the object does not exist.
        """
        kwargs: dict = {}
        if version_id:
            kwargs["version_id"] = version_id
        response = self._client.get_object(self._bucket, key, **kwargs)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def list_versions(self, key: str) -> list[ArtifactRecord]:
        """List all MinIO object versions for the given key.

        Args:
            key: MinIO object key.

        Returns:
            List of ArtifactRecord, one per version, oldest-first.
        """
        objects = self._client.list_objects(
            self._bucket, prefix=key, include_version=True
        )
        records: list[ArtifactRecord] = []
        for obj in objects:
            if obj.object_name == key:
                records.append(
                    ArtifactRecord(
                        key=obj.object_name or key,
                        version_id=obj.version_id or "",
                        sha256="",  # not available from list_objects
                        content_type="application/octet-stream",
                        timestamp_utc=(
                            obj.last_modified.isoformat()
                            if obj.last_modified
                            else ""
                        ),
                        run_id="",
                        user="",
                    )
                )
        return records

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
        """Open a SQLite connection to the lineage database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self._db_path))
