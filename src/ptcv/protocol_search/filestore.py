"""PTCV Local Filestore Manager.

Manages the PTCV-specific protocol filestore at
C:/Dev/PTCV/data/protocols/ with subdirectories for EU-CTR,
ClinicalTrials.gov, and metadata JSON files.

Implements ALCOA+ Original principle: raw downloaded bytes are
written once and never overwritten. SHA-256 is computed before
writing and stored in metadata.

Risk tier: MEDIUM — data pipeline storage.

Regulatory references:
- ALCOA+ Original: raw source data preserved immutably
- ALCOA+ Complete: metadata accompanies every stored file
- 21 CFR 11.10(e): audit trail on storage operations
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Optional

from .models import ProtocolMetadata


_FORMAT_EXT: dict[str, str] = {
    "PDF": "pdf",
    "CTR-XML": "xml",
    "DOCX": "docx",
}

_DEFAULT_ROOT = Path("C:/Dev/PTCV/data/protocols")


class FilestoreManager:
    """Manages the PTCV protocol filestore directory tree.

    Ensures the eu-ctr/, clinicaltrials/, and metadata/ subdirectories
    exist before any file operation. Files are written once — no
    in-place overwrites of stored protocol content.

    Args:
        root: Override the default filestore root. Only for testing.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or _DEFAULT_ROOT
        self.eu_ctr_dir = self.root / "eu-ctr"
        self.clinicaltrials_dir = self.root / "clinicaltrials"
        self.metadata_dir = self.root / "metadata"

    def ensure_directories(self) -> None:
        """Create all required subdirectories if they do not exist.

        Safe to call repeatedly — uses mkdir(exist_ok=True).
        [PTCV-18 Scenario: Filestore directories are created if missing]
        """
        for directory in (
            self.eu_ctr_dir,
            self.clinicaltrials_dir,
            self.metadata_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

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

        Args:
            registry_id: Trial registry identifier.
            amendment_number: Amendment number (e.g., "00", "01").
            fmt: Format key ("PDF", "CTR-XML", "DOCX").
            source: Registry source ("EU-CTR" or "ClinicalTrials.gov").

        Returns:
            Absolute Path for the protocol file.
        """
        ext = _FORMAT_EXT.get(fmt, "pdf")
        filename = f"{registry_id}_{amendment_number}.{ext}"
        directory = (
            self.eu_ctr_dir
            if source == "EU-CTR"
            else self.clinicaltrials_dir
        )
        return directory / filename

    def metadata_path(
        self,
        registry_id: str,
        amendment_number: str,
    ) -> Path:
        """Compute the canonical path for a metadata JSON file.

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
        self.ensure_directories()
        path = self.protocol_path(registry_id, amendment_number, fmt, source)
        if path.exists():
            raise FileExistsError(
                f"Protocol already stored at {path}. "
                "ALCOA+ Original: do not overwrite source data."
            )
        file_hash = self.compute_sha256(content)
        path.write_bytes(content)
        return path, file_hash

    def save_metadata(self, metadata: ProtocolMetadata) -> Path:
        """Write metadata JSON to filestore. Returns saved path.

        Metadata is written fresh on every call (overwrites allowed —
        metadata is derived, not raw source data).

        Args:
            metadata: ProtocolMetadata instance to serialise.

        Returns:
            Path to the saved metadata JSON file.
        """
        self.ensure_directories()
        path = self.metadata_path(
            metadata.registry_id, metadata.amendment_number
        )
        path.write_text(
            json.dumps(dataclasses.asdict(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
