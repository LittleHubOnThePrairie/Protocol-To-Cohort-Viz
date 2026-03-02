"""Tests for PTCV-29 storage gateway data models.

Qualification phase: IQ (installation qualification)
Regulatory requirement: ALCOA+ Traceable — ArtifactRecord and LineageRecord
  carry run_id, user, sha256, and timestamp for full provenance.
Risk tier: MEDIUM
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.storage.models import ArtifactRecord, LineageRecord


class TestArtifactRecord:
    """IQ: ArtifactRecord field creation and defaults."""

    def test_creation_all_fields(self):
        """ArtifactRecord stores all required fields."""
        rec = ArtifactRecord(
            key="clinicaltrials/NCT00112827_00.pdf",
            version_id="v1",
            sha256="abc123",
            content_type="application/pdf",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            run_id="run-001",
            user="tester",
        )
        assert rec.key == "clinicaltrials/NCT00112827_00.pdf"
        assert rec.version_id == "v1"
        assert rec.sha256 == "abc123"
        assert rec.content_type == "application/pdf"
        assert rec.run_id == "run-001"
        assert rec.user == "tester"

    def test_filesystem_version_id_is_empty(self):
        """FilesystemAdapter artifacts use empty string for version_id."""
        rec = ArtifactRecord(
            key="eu-ctr/2024-123456-10-00_00.pdf",
            version_id="",
            sha256="deadbeef",
            content_type="application/pdf",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            run_id="run-002",
            user="ptcv-service",
        )
        assert rec.version_id == ""

    def test_equality(self):
        """Two ArtifactRecords with identical fields are equal (dataclass)."""
        kwargs = dict(
            key="k",
            version_id="v",
            sha256="s",
            content_type="application/pdf",
            timestamp_utc="t",
            run_id="r",
            user="u",
        )
        assert ArtifactRecord(**kwargs) == ArtifactRecord(**kwargs)


class TestLineageRecord:
    """IQ: LineageRecord field creation and optional fields."""

    def test_creation_with_optional_fields(self):
        """LineageRecord stores all fields including optional registry metadata."""
        rec = LineageRecord(
            id=1,
            run_id="run-001",
            stage="download",
            artifact_key="clinicaltrials/NCT00112827_00.pdf",
            version_id="",
            sha256="abc123",
            source_hash="",
            user="ptcv-service",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            registry_id="NCT00112827",
            amendment_number="00",
            source="ClinicalTrials.gov",
        )
        assert rec.run_id == "run-001"
        assert rec.stage == "download"
        assert rec.registry_id == "NCT00112827"
        assert rec.amendment_number == "00"
        assert rec.source == "ClinicalTrials.gov"

    def test_optional_fields_default_none(self):
        """Optional fields default to None when not provided."""
        rec = LineageRecord(
            id=0,
            run_id="run-002",
            stage="download",
            artifact_key="eu-ctr/2024-123456-10-00_00.pdf",
            version_id="",
            sha256="def456",
            source_hash="",
            user="ptcv-service",
            timestamp_utc="2026-01-01T00:00:00+00:00",
        )
        assert rec.registry_id is None
        assert rec.amendment_number is None
        assert rec.source is None

    def test_id_zero_before_insert(self):
        """id=0 is the sentinel value used before SQLite INSERT."""
        rec = LineageRecord(
            id=0,
            run_id="r",
            stage="download",
            artifact_key="k",
            version_id="",
            sha256="s",
            source_hash="",
            user="u",
            timestamp_utc="t",
        )
        assert rec.id == 0
