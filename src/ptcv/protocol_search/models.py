"""Data models for clinical trial protocol search and download.

Implements PTCV-18 metadata schema for EU-CTR and ClinicalTrials.gov
protocols stored in the PTCV filestore.

Risk tier: MEDIUM — structured metadata, no PHI.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SearchResult:
    """A single trial returned from a registry search.

    Attributes:
        registry_id: Registry-specific identifier
            (e.g., "2024-123456-10-00" for EU-CTR, "NCT12345678" for CT.gov).
        title: Full trial title.
        source: Registry name ("EU-CTR" or "ClinicalTrials.gov").
        sponsor: Sponsor organisation name (no PHI — organisation only).
        phase: Trial phase (e.g., "Phase 2").
        condition: Therapeutic area or condition studied.
        status: Trial status (e.g., "Authorised", "Ongoing").
        url: Direct URL to trial record.
    """

    registry_id: str
    title: str
    source: str
    sponsor: str = ""
    phase: str = ""
    condition: str = ""
    status: str = ""
    url: str = ""


@dataclass
class ProtocolMetadata:
    """Metadata written alongside every downloaded protocol file.

    Conforms to the PTCV-18 metadata JSON schema. SHA-256 hash of the
    downloaded file is stored here for ALCOA+ integrity verification.

    Attributes:
        source: Registry name ("EU-CTR", "ClinicalTrials.gov", "WHO-ICTRP").
        registry_id: Registry-specific trial identifier.
        version: Protocol version string.
        amendment_number: Amendment number (zero-padded, e.g., "00", "01").
        title: Trial title (no PHI).
        sponsor: Sponsor organisation name.
        phase: Trial phase.
        condition: Therapeutic area.
        download_timestamp: ISO 8601 UTC timestamp of download.
        file_path: Absolute path to the stored protocol file.
        format: File format ("PDF", "CTR-XML", "DOCX").
        legacy_source: True for pre-2023 EudraCT trials not yet in CTIS.
        file_hash_sha256: SHA-256 digest of the stored file content.
    """

    source: str
    registry_id: str
    version: str = "1.0"
    amendment_number: str = "00"
    title: str = ""
    sponsor: str = ""
    phase: str = ""
    condition: str = ""
    download_timestamp: str = ""
    file_path: str = ""
    format: str = "PDF"
    legacy_source: bool = False
    file_hash_sha256: str = ""

    def __post_init__(self) -> None:
        """Set download_timestamp to current UTC if not provided."""
        if not self.download_timestamp:
            self.download_timestamp = (
                datetime.now(timezone.utc).isoformat()
            )


@dataclass
class DownloadResult:
    """Result of a single protocol download operation.

    Attributes:
        success: Whether the download and storage succeeded.
        registry_id: Identifier of the trial downloaded.
        file_path: Absolute path of the saved protocol file.
        metadata_path: Absolute path of the saved metadata JSON.
        file_hash_sha256: SHA-256 of the downloaded file.
        error: Error message if success is False.
    """

    success: bool
    registry_id: str
    file_path: str = ""
    metadata_path: str = ""
    file_hash_sha256: str = ""
    error: Optional[str] = None
