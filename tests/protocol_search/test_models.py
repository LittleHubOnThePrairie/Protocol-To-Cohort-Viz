"""Tests for PTCV-18 protocol search data models.

Qualification phase: IQ (Installation Qualification — structural correctness)
Regulatory requirement: PTCV-18 metadata schema conformance
  (ALCOA+ Complete — metadata accompanies every stored file)
Risk tier: LOW
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.protocol_search.models import DownloadResult, ProtocolMetadata, SearchResult


class TestProtocolMetadata:
    """IQ: ProtocolMetadata dataclass schema and defaults."""

    def test_required_fields_set(self):
        """Metadata stores source and registry_id."""
        m = ProtocolMetadata(source="EU-CTR", registry_id="2024-123456-10-00")
        assert m.source == "EU-CTR"
        assert m.registry_id == "2024-123456-10-00"

    def test_defaults(self):
        """Metadata defaults match PTCV-18 schema."""
        m = ProtocolMetadata(source="EU-CTR", registry_id="X")
        assert m.version == "1.0"
        assert m.amendment_number == "00"
        assert m.format == "PDF"
        assert m.legacy_source is False
        assert m.file_hash_sha256 == ""

    def test_download_timestamp_auto_set(self):
        """download_timestamp is set to non-empty UTC ISO string."""
        m = ProtocolMetadata(source="EU-CTR", registry_id="X")
        assert m.download_timestamp
        assert "T" in m.download_timestamp  # ISO 8601 format

    def test_download_timestamp_not_overwritten(self):
        """Explicitly provided timestamp is preserved."""
        ts = "2026-01-01T00:00:00+00:00"
        m = ProtocolMetadata(source="EU-CTR", registry_id="X", download_timestamp=ts)
        assert m.download_timestamp == ts

    def test_legacy_source_flag(self):
        """legacy_source=True for pre-CTIS EudraCT trials."""
        m = ProtocolMetadata(source="EU-CTR", registry_id="2019-001234-56", legacy_source=True)
        assert m.legacy_source is True


class TestSearchResult:
    """IQ: SearchResult dataclass schema."""

    def test_required_fields(self):
        """SearchResult stores registry_id, title, source."""
        r = SearchResult(registry_id="NCT12345678", title="Trial A", source="ClinicalTrials.gov")
        assert r.registry_id == "NCT12345678"
        assert r.title == "Trial A"
        assert r.source == "ClinicalTrials.gov"

    def test_optional_defaults_empty(self):
        """Optional fields default to empty strings."""
        r = SearchResult(registry_id="X", title="Y", source="EU-CTR")
        assert r.sponsor == ""
        assert r.phase == ""
        assert r.condition == ""
        assert r.status == ""
        assert r.url == ""


class TestDownloadResult:
    """IQ: DownloadResult dataclass schema."""

    def test_success_result(self):
        """DownloadResult captures paths and hash on success."""
        r = DownloadResult(
            success=True,
            registry_id="2024-123456-10-00",
            file_path="/data/eu-ctr/2024-123456-10-00_00.pdf",
            metadata_path="/data/metadata/2024-123456-10-00_00.json",
            file_hash_sha256="abc123",
        )
        assert r.success is True
        assert r.file_hash_sha256 == "abc123"
        assert r.error is None

    def test_failure_result(self):
        """DownloadResult captures error message on failure."""
        r = DownloadResult(
            success=False,
            registry_id="X",
            error="HTTP 404",
        )
        assert r.success is False
        assert r.error == "HTTP 404"
        assert r.file_path == ""
