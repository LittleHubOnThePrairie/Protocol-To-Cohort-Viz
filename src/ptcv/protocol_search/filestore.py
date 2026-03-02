"""PTCV Local Filestore Manager.

FilestoreManager is now a thin compatibility shim over FilesystemAdapter
(from ptcv.storage). All previous tests and callers continue to work
unchanged. New code should use StorageGateway / FilesystemAdapter
directly instead of FilestoreManager.

Implements ALCOA+ Original principle: raw downloaded bytes are
written once and never overwritten. SHA-256 is computed before
writing and stored in metadata.

Risk tier: MEDIUM — data pipeline storage.

Regulatory references:
- ALCOA+ Original: raw source data preserved immutably (via FilesystemAdapter)
- ALCOA+ Complete: metadata accompanies every stored file
- 21 CFR 11.10(e): audit trail on storage operations (lineage.db)
"""

import dataclasses
import hashlib
import json
import uuid
from pathlib import Path
from typing import Optional

from .models import ProtocolMetadata


_FORMAT_EXT: dict[str, str] = {
    "PDF": "pdf",
    "CTR-XML": "xml",
    "DOCX": "docx",
}

_CONTENT_TYPES: dict[str, str] = {
    "PDF": "application/pdf",
    "CTR-XML": "application/xml",
    "DOCX": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_DEFAULT_ROOT = Path("C:/Dev/PTCV/data/protocols")


def _make_protocol_key(
    registry_id: str,
    amendment_number: str,
    fmt: str,
    source: str,
) -> str:
    """Return the relative storage key for a protocol file."""
    ext = _FORMAT_EXT.get(fmt, "pdf")
    subdir = "eu-ctr" if source == "EU-CTR" else "clinicaltrials"
    return f"{subdir}/{registry_id}_{amendment_number}.{ext}"


def _content_type(fmt: str) -> str:
    """Return MIME type for the given format key."""
    return _CONTENT_TYPES.get(fmt, "application/octet-stream")


def _serialise(metadata: ProtocolMetadata) -> bytes:
    """Serialise ProtocolMetadata to UTF-8 JSON bytes."""
    return json.dumps(
        dataclasses.asdict(metadata), indent=2, ensure_ascii=False
    ).encode("utf-8")


def _new_run_id() -> str:
    """Generate a UUID4 run identifier."""
    return str(uuid.uuid4())


class FilestoreManager:
    """Compatibility shim over FilesystemAdapter.

    Preserves the original FilestoreManager public API so that all
    existing tests and callers continue to work without modification.
    Internally delegates all I/O to a FilesystemAdapter instance.

    Args:
        root: Override the default filestore root. Only for testing.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        from ptcv.storage import FilesystemAdapter  # local import

        _root = root or _DEFAULT_ROOT
        self._adapter = FilesystemAdapter(root=_root)
        self._adapter.initialise()

        # Public attributes preserved for test introspection
        self.root = _root
        self.eu_ctr_dir = _root / "eu-ctr"
        self.clinicaltrials_dir = _root / "clinicaltrials"
        self.metadata_dir = _root / "metadata"

    def ensure_directories(self) -> None:
        """Create all required subdirectories if they do not exist.

        Delegates to FilesystemAdapter.initialise() which is idempotent.
        [PTCV-18 Scenario: Filestore directories are created if missing]
        """
        self._adapter.initialise()

    @staticmethod
    def compute_sha256(content: bytes) -> str:
        """Compute SHA-256 hex digest of byte content.

        Used at the download boundary per ALCOA+ Consistent principle.

        Args:
            content: Raw bytes to hash.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        return hashlib.sha256(content).hexdigest()

    def protocol_path(
        self,
        registry_id: str,
        amendment_number: str,
        fmt: str,
        source: str,
    ) -> Path:
        """Compute the canonical path for a protocol file.

        This is a pure path computation — no I/O is performed.

        Args:
            registry_id: Trial registry identifier.
            amendment_number: Amendment number (e.g., "00", "01").
            fmt: Format key ("PDF", "CTR-XML", "DOCX").
            source: Registry source ("EU-CTR" or "ClinicalTrials.gov").

        Returns:
            Absolute Path for the protocol file.
        """
        key = _make_protocol_key(registry_id, amendment_number, fmt, source)
        return self.root / key

    def metadata_path(
        self,
        registry_id: str,
        amendment_number: str,
    ) -> Path:
        """Compute the canonical path for a metadata JSON file.

        This is a pure path computation — no I/O is performed.

        Args:
            registry_id: Trial registry identifier.
            amendment_number: Amendment number.

        Returns:
            Absolute Path for the metadata JSON file.
        """
        return self.metadata_dir / f"{registry_id}_{amendment_number}.json"

    def save_protocol(
        self,
        content: bytes,
        registry_id: str,
        amendment_number: str,
        fmt: str,
        source: str,
    ) -> tuple[Path, str]:
        """Write protocol bytes to filestore. Returns (path, sha256).

        Does NOT overwrite if the file already exists — preserving the
        original per ALCOA+ Original principle. Raises FileExistsError
        if the same registry_id + amendment already stored.

        Args:
            content: Raw downloaded bytes.
            registry_id: Trial identifier.
            amendment_number: Amendment number.
            fmt: File format ("PDF", "CTR-XML", "DOCX").
            source: Registry source.

        Returns:
            Tuple of (saved Path, hex SHA-256 of content).

        Raises:
            FileExistsError: If the protocol file already exists.
        """
        key = _make_protocol_key(registry_id, amendment_number, fmt, source)
        artifact = self._adapter.put_artifact(
            key=key,
            data=content,
            content_type=_content_type(fmt),
            run_id=_new_run_id(),
            source_hash="",
            user="filestore",
            immutable=True,
            registry_id=registry_id,
            amendment_number=amendment_number,
            source=source,
        )
        return self.root / artifact.key, artifact.sha256

    def save_metadata(self, metadata: ProtocolMetadata) -> Path:
        """Write metadata JSON to filestore. Returns saved path.

        Metadata is written fresh on every call (overwrites allowed —
        metadata is derived, not raw source data).

        Args:
            metadata: ProtocolMetadata instance to serialise.

        Returns:
            Path to the saved metadata JSON file.
        """
        key = (
            f"metadata/{metadata.registry_id}_{metadata.amendment_number}.json"
        )
        self._adapter.put_artifact(
            key=key,
            data=_serialise(metadata),
            content_type="application/json",
            run_id=_new_run_id(),
            source_hash="",
            user="filestore",
            immutable=False,
        )
        return self.root / key
