"""Tests for PTCV-18 EU-CTR (CTIS) Protocol Search and Download Service.

Qualification phase: OQ (operational qualification)
Regulatory requirement: ALCOA+ Audit trail, ALCOA+ Original (no overwrite),
  ALCOA+ Complete (metadata accompanies every file),
  PTCV-18 Scenarios: Search EU-CTR, Download EU-CTR, Handle legacy EudraCT.
Risk tier: MEDIUM
"""

import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.protocol_search.eu_ctr_service import CTISService
from ptcv.protocol_search.models import DownloadResult, SearchResult


class TestCTISServiceInit:
    """IQ: Service initialisation wires defaults."""

    def test_default_filestore_and_audit_created(self, tmp_path):
        """Service creates internal filestore and audit logger when none given."""
        svc = CTISService()
        assert svc._filestore is not None
        assert svc._audit is not None
        assert svc._timeout == 30

    def test_custom_injected_filestore_used(self, ctis_service):
        """Injected FilestoreManager is stored on the service."""
        assert ctis_service._filestore is not None


class TestCTISServiceSearch:
    """OQ: PTCV-18 Scenario: Search EU-CTR protocols by condition."""

    def _mock_response(self, data: dict):
        """Build a mock response suitable for _post."""
        return data

    def test_search_returns_search_results(self, ctis_service):
        """PTCV-18 Scenario: Search EU-CTR by condition returns SearchResult list."""
        payload = {
            "data": [
                {
                    "ctNumber": "2024-123456-10-00",
                    "trialTitle": "Test Trial",
                    "sponsorName": "Sponsor A",
                    "trialPhase": "Phase 2",
                    "medicalCondition": "oncology",
                    "trialStatus": "Authorised",
                }
            ]
        }
        with patch.object(ctis_service, "_post", return_value=payload):
            results = ctis_service.search(condition="oncology")

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].registry_id == "2024-123456-10-00"
        assert results[0].title == "Test Trial"
        assert results[0].source == "EU-CTR"
        assert results[0].sponsor == "Sponsor A"

    def test_search_uses_results_key_fallback(self, ctis_service):
        """Search handles 'results' key as fallback when 'data' absent."""
        payload = {
            "results": [
                {
                    "ctNumber": "2023-111111-10-00",
                    "trialTitle": "Fallback Trial",
                    "sponsorName": "",
                    "trialPhase": "",
                    "medicalCondition": "cardiology",
                    "trialStatus": "Ongoing",
                }
            ]
        }
        with patch.object(ctis_service, "_post", return_value=payload):
            results = ctis_service.search(condition="cardiology")

        assert len(results) == 1
        assert results[0].registry_id == "2023-111111-10-00"

    def test_search_empty_results(self, ctis_service):
        """Search returns empty list when no trials match."""
        with patch.object(ctis_service, "_post", return_value={"data": []}):
            results = ctis_service.search(condition="nonexistent")

        assert results == []

    def test_search_network_error_returns_empty(self, ctis_service):
        """Network failure in search is swallowed and returns empty list."""
        with patch.object(
            ctis_service, "_post", side_effect=Exception("Connection refused")
        ):
            results = ctis_service.search(condition="oncology")

        assert results == []

    def test_search_writes_audit_entry(self, ctis_service, tmp_path):
        """PTCV-18: Search action is written to the audit log."""
        with patch.object(ctis_service, "_post", return_value={"data": []}):
            ctis_service.search(condition="oncology", who="tester")

        audit_file = tmp_path / "audit.jsonl"
        assert audit_file.exists()
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry["action"] == "SEARCH"
        assert entry["user_id"] == "tester"


class TestCTISServiceDownload:
    """OQ: PTCV-18 Scenario: Download EU-CTR protocol to PTCV filestore."""

    def test_download_eu_ctr_success(self, ctis_service, tmp_path):
        """PTCV-18 Scenario: EU-CTR protocol downloaded and stored."""
        content = b"%PDF-1.4 mock eu-ctr content"

        with patch.object(ctis_service, "_get_bytes", return_value=content):
            result = ctis_service.download(
                euct_code="2024-123456-10-00",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="protocol_ingestion",
            )

        assert result.success is True
        assert result.registry_id == "2024-123456-10-00"
        assert result.file_hash_sha256 is not None
        assert len(result.file_hash_sha256) == 64  # SHA-256 hex digest

        # File must exist under eu-ctr/ subdir
        file_path = Path(result.file_path)
        assert file_path.exists()
        assert "eu-ctr" in str(file_path)
        assert file_path.read_bytes() == content

        # Metadata JSON must exist
        metadata_path = Path(result.metadata_path)
        assert metadata_path.exists()
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert data["registry_id"] == "2024-123456-10-00"
        assert data["source"] == "EU-CTR"
        assert data["legacy_source"] is False

    def test_download_writes_audit_entry(self, ctis_service, tmp_path):
        """PTCV-18: Download action is written to audit log."""
        content = b"content bytes"
        with patch.object(ctis_service, "_get_bytes", return_value=content):
            ctis_service.download(
                euct_code="2024-000001-01-00",
                amendment_number="00",
                fmt="PDF",
                who="auditor",
                why="gcp_review",
            )

        audit_file = tmp_path / "audit.jsonl"
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        download_entries = [
            json.loads(line)
            for line in lines
            if json.loads(line).get("action") == "DOWNLOAD"
        ]
        assert len(download_entries) >= 1
        entry = download_entries[-1]
        assert entry["user_id"] == "auditor"
        assert entry["reason"] == "gcp_review"

    def test_download_http_error_returns_failure(self, ctis_service):
        """HTTP error from CTIS returns DownloadResult with success=False."""
        err = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=None
        )
        with patch.object(ctis_service, "_get_bytes", side_effect=err):
            result = ctis_service.download(
                euct_code="2024-999999-10-00",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is False
        assert "404" in result.error

    def test_download_generic_error_returns_failure(self, ctis_service):
        """Non-HTTP exception returns DownloadResult with success=False."""
        with patch.object(
            ctis_service, "_get_bytes", side_effect=Exception("timeout")
        ):
            result = ctis_service.download(
                euct_code="2024-888888-10-00",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is False
        assert "timeout" in result.error

    def test_download_no_overwrite_raises_file_exists(self, ctis_service):
        """ALCOA+ Original: downloading the same protocol twice returns failure."""
        content = b"original"
        with patch.object(ctis_service, "_get_bytes", return_value=content):
            first = ctis_service.download(
                euct_code="2024-123456-10-00",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="test",
            )
        assert first.success is True

        with patch.object(ctis_service, "_get_bytes", return_value=b"new content"):
            second = ctis_service.download(
                euct_code="2024-123456-10-00",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert second.success is False
        assert "already stored" in second.error.lower()


class TestCTISServiceLegacyEudraCT:
    """OQ: PTCV-18 Scenario: Handle legacy EudraCT protocols."""

    def test_legacy_eudract_pattern_detected(self, ctis_service):
        """EudraCT number (3 segments) triggers legacy fallback without HTTP call."""
        with patch.object(
            ctis_service,
            "_download_legacy_eudract",
            return_value=DownloadResult(
                success=True,
                registry_id="2019-001234-56",
                file_path="/some/path",
                metadata_path="/some/meta",
                file_hash_sha256="abc",
            ),
        ) as mock_legacy:
            result = ctis_service.download(
                euct_code="2019-001234-56",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="test",
            )

        mock_legacy.assert_called_once()
        assert result.success is True

    def test_legacy_fallback_r_not_installed(self, ctis_service):
        """FileNotFoundError from R subprocess returns success=False with message."""
        with patch("subprocess.run", side_effect=FileNotFoundError("Rscript")):
            result = ctis_service._download_legacy_eudract(
                eudract_number="2019-001234-56",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is False
        assert "R" in result.error or "ctrdata" in result.error

    def test_legacy_fallback_r_nonzero_exit(self, ctis_service):
        """Non-zero Rscript exit code returns success=False with stderr."""
        import subprocess

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: could not find function ctrLoadQueryIntoDb"

        with patch("subprocess.run", return_value=mock_result):
            result = ctis_service._download_legacy_eudract(
                eudract_number="2019-001234-56",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is False
        assert "ctrLoadQueryIntoDb" in result.error

    def test_legacy_fallback_timeout(self, ctis_service):
        """subprocess.TimeoutExpired returns success=False."""
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="Rscript", timeout=60),
        ):
            result = ctis_service._download_legacy_eudract(
                eudract_number="2019-001234-56",
                amendment_number="00",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is False

    def test_legacy_fallback_audit_written_on_failure(self, ctis_service, tmp_path):
        """PTCV-18: Legacy fallback failure is written to audit log."""
        with patch("subprocess.run", side_effect=FileNotFoundError("Rscript")):
            ctis_service._download_legacy_eudract(
                eudract_number="2019-001234-56",
                amendment_number="00",
                fmt="PDF",
                who="auditor",
                why="gcp_review",
            )

        audit_file = tmp_path / "audit.jsonl"
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        entries = [json.loads(line) for line in lines]
        download_entries = [e for e in entries if e.get("action") == "DOWNLOAD"]
        assert any(not e.get("after", {}).get("success", True) for e in download_entries)
