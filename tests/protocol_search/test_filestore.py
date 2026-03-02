"""Tests for PTCV-18 local filestore manager.

Qualification phase: IQ/OQ (unit and integration)
Regulatory requirement: ALCOA+ Original (raw data preserved, never
  overwritten), ALCOA+ Complete (metadata accompanies every file),
  PTCV-18 Scenario: Filestore directories are created if missing.
Risk tier: MEDIUM
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.protocol_search.filestore import FilestoreManager
from ptcv.protocol_search.models import ProtocolMetadata


class TestFilestoreManagerDirectories:
    """IQ: Directory creation and structure."""

    def test_ensure_directories_creates_all_subdirs(self, tmp_path):
        """PTCV-18 Scenario: Filestore directories created if missing."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        assert not (tmp_path / "protocols").exists()

        fs.ensure_directories()

        assert (tmp_path / "protocols" / "eu-ctr").is_dir()
        assert (tmp_path / "protocols" / "clinicaltrials").is_dir()
        assert (tmp_path / "protocols" / "metadata").is_dir()

    def test_ensure_directories_idempotent(self, tmp_path):
        """Repeated calls to ensure_directories do not raise."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        fs.ensure_directories()
        fs.ensure_directories()  # Should not raise


class TestFilestoreManagerSha256:
    """IQ: SHA-256 hash computation at download boundary (ALCOA+)."""

    def test_compute_sha256_known_value(self):
        """SHA-256 of known bytes matches expected digest."""
        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()
        fs = FilestoreManager()
        assert fs.compute_sha256(content) == expected

    def test_compute_sha256_empty_bytes(self):
        """SHA-256 of empty bytes is well-defined."""
        expected = hashlib.sha256(b"").hexdigest()
        assert FilestoreManager.compute_sha256(b"") == expected


class TestFilestoreManagerPaths:
    """IQ: Canonical path computation for protocols and metadata."""

    def test_eu_ctr_protocol_path_pdf(self, tmp_path):
        """EU-CTR PDF path uses eu-ctr/ subdirectory."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        path = fs.protocol_path("2024-123456-10-00", "00", "PDF", "EU-CTR")
        assert "eu-ctr" in str(path)
        assert path.name == "2024-123456-10-00_00.pdf"

    def test_eu_ctr_protocol_path_xml(self, tmp_path):
        """EU-CTR CTR-XML path uses .xml extension."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        path = fs.protocol_path("2024-123456-10-00", "01", "CTR-XML", "EU-CTR")
        assert path.name == "2024-123456-10-00_01.xml"

    def test_clinicaltrials_protocol_path(self, tmp_path):
        """ClinicalTrials.gov path uses clinicaltrials/ subdirectory."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        path = fs.protocol_path("NCT12345678", "1.0", "PDF", "ClinicalTrials.gov")
        assert "clinicaltrials" in str(path)
        assert path.name == "NCT12345678_1.0.pdf"

    def test_metadata_path(self, tmp_path):
        """Metadata JSON path uses metadata/ subdirectory."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        path = fs.metadata_path("2024-123456-10-00", "00")
        assert "metadata" in str(path)
        assert path.name == "2024-123456-10-00_00.json"


class TestFilestoreManagerSaveProtocol:
    """OQ: Protocol storage with integrity verification."""

    def test_save_protocol_eu_ctr(self, tmp_path):
        """PTCV-18 Scenario: EU-CTR protocol saved to eu-ctr/ subdirectory."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        content = b"%PDF-1.4 mock content"
        path, sha256 = fs.save_protocol(
            content=content,
            registry_id="2024-123456-10-00",
            amendment_number="00",
            fmt="PDF",
            source="EU-CTR",
        )

        assert path.exists()
        assert "eu-ctr" in str(path)
        assert path.read_bytes() == content
        assert sha256 == hashlib.sha256(content).hexdigest()

    def test_save_protocol_clinicaltrials(self, tmp_path):
        """PTCV-18 Scenario: ClinicalTrials.gov protocol saved to clinicaltrials/."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        content = b'{"nctId": "NCT12345678"}'
        path, sha256 = fs.save_protocol(
            content=content,
            registry_id="NCT12345678",
            amendment_number="1.0",
            fmt="PDF",
            source="ClinicalTrials.gov",
        )

        assert path.exists()
        assert "clinicaltrials" in str(path)
        assert sha256 == hashlib.sha256(content).hexdigest()

    def test_save_protocol_no_overwrite(self, tmp_path):
        """ALCOA+ Original: saving same file twice raises FileExistsError."""
        fs = FilestoreManager(root=tmp_path / "protocols")
        content = b"content"
        fs.save_protocol(
            content=content,
            registry_id="2024-123456-10-00",
            amendment_number="00",
            fmt="PDF",
            source="EU-CTR",
        )
        with pytest.raises(FileExistsError):
            fs.save_protocol(
                content=b"different content",
                registry_id="2024-123456-10-00",
                amendment_number="00",
                fmt="PDF",
                source="EU-CTR",
            )

    def test_save_metadata_writes_json(self, tmp_path):
        """PTCV-18 Scenario: metadata JSON written with required fields."""
        import json

        fs = FilestoreManager(root=tmp_path / "protocols")
        meta = ProtocolMetadata(
            source="EU-CTR",
            registry_id="2024-123456-10-00",
            amendment_number="00",
            title="Test Trial",
            file_path="/some/path/file.pdf",
            file_hash_sha256="abc123",
        )
        path = fs.save_metadata(meta)

        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["source"] == "EU-CTR"
        assert data["registry_id"] == "2024-123456-10-00"
        assert data["file_hash_sha256"] == "abc123"
        assert "download_timestamp" in data
