"""Tests for PTCV-18 ClinicalTrials.gov Protocol Search and Download Service.

Qualification phase: OQ (operational qualification)
Regulatory requirement: ALCOA+ Audit trail, ALCOA+ Original (no overwrite),
  ALCOA+ Complete (metadata accompanies every file),
  PTCV-18 Scenarios: Search ClinicalTrials.gov, Download ClinicalTrials.gov.
Risk tier: MEDIUM
"""

import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.protocol_search.clinicaltrials_service import ClinicalTrialsService
from ptcv.protocol_search.models import SearchResult


class TestClinicalTrialsServiceInit:
    """IQ: Service initialisation wires defaults."""

    def test_default_dependencies_created(self):
        """Service creates internal gateway and audit logger when none given."""
        svc = ClinicalTrialsService()
        assert svc._gateway is not None
        assert svc._audit is not None
        assert svc._timeout == 30

    def test_custom_injected_dependencies_stored(self, ct_service):
        """Injected dependencies are stored on the service (PTCV-29)."""
        assert ct_service._gateway is not None
        assert ct_service._audit is not None


class TestClinicalTrialsServiceSearch:
    """OQ: PTCV-18 Scenario: Search ClinicalTrials.gov protocols."""

    def _study_stub(self, nct_id: str, title: str = "Test Study") -> dict:
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct_id,
                    "briefTitle": title,
                    "officialTitle": title,
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "descriptionModule": {},
                "designModule": {"phases": ["PHASE2"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Test Sponsor"}
                },
                "conditionsModule": {"conditions": ["Oncology"]},
            }
        }

    def test_search_returns_search_results(self, ct_service):
        """PTCV-18 Scenario: Search CT.gov returns SearchResult list."""
        payload = {
            "studies": [self._study_stub("NCT12345678", "Oncology Trial")]
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search(condition="oncology")

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].registry_id == "NCT12345678"
        assert results[0].title == "Oncology Trial"
        assert results[0].source == "ClinicalTrials.gov"
        assert results[0].sponsor == "Test Sponsor"
        assert results[0].status == "RECRUITING"

    def test_search_pagination_follows_next_token(self, ct_service):
        """PTCV-18: Search follows nextPageToken until no more pages."""
        page1 = {
            "studies": [self._study_stub("NCT11111111")],
            "nextPageToken": "token_page2",
        }
        page2 = {
            "studies": [self._study_stub("NCT22222222")],
        }

        def side_effect(url, params):
            if params.get("pageToken") == "token_page2":
                return page2
            return page1

        with patch.object(ct_service, "_get_json", side_effect=side_effect):
            results = ct_service.search(condition="oncology")

        assert len(results) == 2
        ids = {r.registry_id for r in results}
        assert "NCT11111111" in ids
        assert "NCT22222222" in ids

    def test_search_respects_max_results(self, ct_service):
        """max_results stops pagination after the requested count."""
        page1 = {
            "studies": [
                self._study_stub("NCT11111111"),
                self._study_stub("NCT22222222"),
                self._study_stub("NCT33333333"),
            ],
            "nextPageToken": "more",
        }
        with patch.object(ct_service, "_get_json", return_value=page1):
            results = ct_service.search(condition="oncology", max_results=2)

        assert len(results) == 2

    def test_search_empty_studies(self, ct_service):
        """Empty studies array returns empty list."""
        with patch.object(ct_service, "_get_json", return_value={"studies": []}):
            results = ct_service.search(condition="nothing")

        assert results == []

    def test_search_network_error_returns_empty(self, ct_service):
        """Network exception in search is swallowed and returns empty list."""
        with patch.object(
            ct_service, "_get_json", side_effect=Exception("DNS failure")
        ):
            results = ct_service.search(condition="oncology")

        assert results == []

    def test_search_condition_filter_included_in_params(self, ct_service):
        """Condition filter is passed as query.cond parameter."""
        captured_params: list = []

        def capture(url, params):
            captured_params.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search(condition="diabetes", phase="PHASE3")

        assert captured_params
        params = captured_params[0]
        assert params.get("query.cond") == "diabetes"
        assert params.get("query.term") == "PHASE3"

    def test_search_writes_audit_entry(self, ct_service, tmp_path):
        """PTCV-18: Search action is written to audit log."""
        with patch.object(ct_service, "_get_json", return_value={"studies": []}):
            ct_service.search(condition="oncology", who="tester")

        audit_file = tmp_path / "audit.jsonl"
        assert audit_file.exists()
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["action"] == "SEARCH"
        assert entry["user_id"] == "tester"


class TestClinicalTrialsServiceDownload:
    """OQ: PTCV-18 Scenario: Download ClinicalTrials.gov protocol to PTCV filestore."""

    def _study_response(self, nct_id: str) -> dict:
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct_id,
                    "briefTitle": "Study Title",
                    "officialTitle": "Official Study Title",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"phases": ["PHASE2"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "BigPharma Inc"}
                },
                "conditionsModule": {"conditions": ["Diabetes"]},
            }
        }

    def test_download_clinicaltrials_success(self, ct_service, tmp_path):
        """PTCV-18 Scenario: CT.gov protocol saved to clinicaltrials/ subdirectory."""
        study_data = self._study_response("NCT12345678")

        with patch.object(ct_service, "_get_json", return_value=study_data):
            result = ct_service.download(
                nct_id="NCT12345678",
                version="1.0",
                fmt="PDF",
                who="tester",
                why="protocol_ingestion",
            )

        assert result.success is True
        assert result.registry_id == "NCT12345678"
        assert result.file_hash_sha256 is not None
        assert len(result.file_hash_sha256) == 64

        # File must exist under clinicaltrials/ subdir
        file_path = Path(result.file_path)
        assert file_path.exists()
        assert "clinicaltrials" in str(file_path)

        # Stored content must be valid JSON
        stored = json.loads(file_path.read_text(encoding="utf-8"))
        assert stored["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"

        # Metadata JSON must exist
        metadata_path = Path(result.metadata_path)
        assert metadata_path.exists()
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert meta["registry_id"] == "NCT12345678"
        assert meta["source"] == "ClinicalTrials.gov"

    def test_download_writes_audit_entry(self, ct_service, tmp_path):
        """PTCV-18: Download action is written to audit log."""
        study_data = self._study_response("NCT00000001")
        with patch.object(ct_service, "_get_json", return_value=study_data):
            ct_service.download(
                nct_id="NCT00000001",
                version="1.0",
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
        assert "file_path" in entry.get("after", {})

    def test_download_http_error_returns_failure(self, ct_service):
        """HTTP error from CT.gov API returns DownloadResult with success=False."""
        err = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=None
        )
        with patch.object(ct_service, "_get_json", side_effect=err):
            result = ct_service.download(
                nct_id="NCT99999999",
                version="1.0",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is False
        assert "404" in result.error

    def test_download_generic_exception_returns_failure(self, ct_service):
        """Non-HTTP exception returns DownloadResult with success=False."""
        with patch.object(
            ct_service, "_get_json", side_effect=Exception("SSL error")
        ):
            result = ct_service.download(
                nct_id="NCT88888888",
                version="1.0",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is False
        assert "SSL error" in result.error

    def test_download_no_overwrite_raises_file_exists(self, ct_service):
        """ALCOA+ Original: downloading the same NCT ID twice returns failure."""
        study_data = self._study_response("NCT12345678")

        with patch.object(ct_service, "_get_json", return_value=study_data):
            first = ct_service.download(
                nct_id="NCT12345678",
                version="1.0",
                fmt="PDF",
                who="tester",
                why="test",
            )
        assert first.success is True

        with patch.object(ct_service, "_get_json", return_value=study_data):
            second = ct_service.download(
                nct_id="NCT12345678",
                version="1.0",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert second.success is False
        assert "already exists" in second.error.lower()

    def test_download_sha256_stored_in_metadata(self, ct_service, tmp_path):
        """ALCOA+ Consistent: SHA-256 in metadata matches file content hash."""
        import hashlib

        study_data = self._study_response("NCT77777777")
        with patch.object(ct_service, "_get_json", return_value=study_data):
            result = ct_service.download(
                nct_id="NCT77777777",
                version="1.0",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is True
        file_path = Path(result.file_path)
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert result.file_hash_sha256 == actual_hash

        meta = json.loads(
            Path(result.metadata_path).read_text(encoding="utf-8")
        )
        assert meta["file_hash_sha256"] == actual_hash

    def test_download_metadata_has_download_timestamp(self, ct_service, tmp_path):
        """ALCOA+ Contemporaneous: metadata includes download_timestamp."""
        study_data = self._study_response("NCT66666666")
        with patch.object(ct_service, "_get_json", return_value=study_data):
            result = ct_service.download(
                nct_id="NCT66666666",
                version="1.0",
                fmt="PDF",
                who="tester",
                why="test",
            )

        assert result.success is True
        meta = json.loads(
            Path(result.metadata_path).read_text(encoding="utf-8")
        )
        assert "download_timestamp" in meta
        assert meta["download_timestamp"]  # not empty

    def test_download_fallback_sets_actual_format_json(self, ct_service):
        """No largeDocumentModule → actual_format is 'JSON'."""
        study_data = self._study_response("NCT55555555")
        with patch.object(ct_service, "_get_json", return_value=study_data):
            result = ct_service.download(
                nct_id="NCT55555555", version="1.0", fmt="PDF", who="t", why="t"
            )

        assert result.success is True
        assert result.actual_format == "JSON"


class TestClinicalTrialsServiceLargeDocModule:
    """OQ: PTCV-28 largeDocumentModule pathway for protocol PDFs and SAPs."""

    def _study_with_large_docs(
        self, nct_id: str, large_docs: list[dict]
    ) -> dict:
        """Build a study response that includes largeDocumentModule."""
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct_id,
                    "briefTitle": "Large Doc Study",
                    "officialTitle": "Large Doc Study Official",
                },
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"phases": ["PHASE3"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Sponsor Co"}
                },
                "conditionsModule": {"conditions": ["Oncology"]},
            },
            "documentSection": {
                "largeDocumentModule": {
                    "largeDocs": large_docs,
                }
            },
        }

    def test_protocol_pdf_downloaded_when_prot_present(self, ct_service, tmp_path):
        """PTCV-28 Scenario: protocol PDF fetched from ProvidedDocs URL."""
        fake_pdf = b"%PDF-1.7 fake protocol content"
        study = self._study_with_large_docs(
            "NCT10000001",
            [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
        )
        with (
            patch.object(ct_service, "_get_json", return_value=study),
            patch.object(ct_service, "_get_bytes", return_value=fake_pdf),
        ):
            result = ct_service.download(
                nct_id="NCT10000001", version="1.0", fmt="PDF",
                who="tester", why="test",
            )

        assert result.success is True
        assert result.actual_format == "PDF"
        assert Path(result.file_path).read_bytes() == fake_pdf

    def test_provided_docs_url_built_correctly(self, ct_service, tmp_path):
        """ProvidedDocs URL uses last-two chars of NCT ID as path segment."""
        captured_urls: list[str] = []
        fake_pdf = b"%PDF-1.4 content"
        study = self._study_with_large_docs(
            "NCT12345678",
            [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
        )

        def capture_get_bytes(url: str) -> bytes:
            captured_urls.append(url)
            return fake_pdf

        with (
            patch.object(ct_service, "_get_json", return_value=study),
            patch.object(ct_service, "_get_bytes", side_effect=capture_get_bytes),
        ):
            ct_service.download(
                nct_id="NCT12345678", version="1.0", fmt="PDF",
                who="tester", why="test",
            )

        assert len(captured_urls) == 1
        assert captured_urls[0] == (
            "https://clinicaltrials.gov/ProvidedDocs/78/NCT12345678/Prot_000.pdf"
        )

    def test_prot_sap_satisfies_pdf_request(self, ct_service, tmp_path):
        """Prot_SAP typeAbbrev is accepted when no Prot entry is present."""
        fake_pdf = b"%PDF-1.7 combined prot_sap"
        study = self._study_with_large_docs(
            "NCT10000002",
            [{"typeAbbrev": "Prot_SAP", "filename": "Prot_SAP_000.pdf"}],
        )
        with (
            patch.object(ct_service, "_get_json", return_value=study),
            patch.object(ct_service, "_get_bytes", return_value=fake_pdf),
        ):
            result = ct_service.download(
                nct_id="NCT10000002", version="1.0", fmt="PDF",
                who="tester", why="test",
            )

        assert result.success is True
        assert result.actual_format == "PDF"

    def test_prot_takes_priority_over_prot_sap(self, ct_service, tmp_path):
        """Dedicated Prot entry is preferred over combined Prot_SAP."""
        captured_urls: list[str] = []

        def capture_get_bytes(url: str) -> bytes:
            captured_urls.append(url)
            return b"%PDF-1.7"

        study = self._study_with_large_docs(
            "NCT10000003",
            [
                {"typeAbbrev": "Prot_SAP", "filename": "Prot_SAP_000.pdf"},
                {"typeAbbrev": "Prot", "filename": "Prot_000.pdf"},
            ],
        )
        with (
            patch.object(ct_service, "_get_json", return_value=study),
            patch.object(ct_service, "_get_bytes", side_effect=capture_get_bytes),
        ):
            ct_service.download(
                nct_id="NCT10000003", version="1.0", fmt="PDF",
                who="tester", why="test",
            )

        assert len(captured_urls) == 1
        assert "Prot_000.pdf" in captured_urls[0]

    def test_sap_downloaded_when_fmt_sap(self, ct_service, tmp_path):
        """PTCV-28 Scenario: SAP PDF downloaded when fmt='SAP'."""
        fake_sap = b"%PDF-1.7 statistical analysis plan"
        study = self._study_with_large_docs(
            "NCT10000004",
            [
                {"typeAbbrev": "SAP", "filename": "SAP_000.pdf"},
                {"typeAbbrev": "Prot", "filename": "Prot_000.pdf"},
            ],
        )
        with (
            patch.object(ct_service, "_get_json", return_value=study),
            patch.object(ct_service, "_get_bytes", return_value=fake_sap),
        ):
            result = ct_service.download(
                nct_id="NCT10000004", version="1.0", fmt="SAP",
                who="tester", why="test",
            )

        assert result.success is True
        assert result.actual_format == "SAP"
        assert Path(result.file_path).read_bytes() == fake_sap

    def test_sap_takes_priority_over_prot_sap(self, ct_service, tmp_path):
        """Dedicated SAP entry is preferred over combined Prot_SAP for SAP request."""
        captured_urls: list[str] = []

        def capture_get_bytes(url: str) -> bytes:
            captured_urls.append(url)
            return b"%PDF-1.7"

        study = self._study_with_large_docs(
            "NCT10000005",
            [
                {"typeAbbrev": "Prot_SAP", "filename": "Prot_SAP_000.pdf"},
                {"typeAbbrev": "SAP", "filename": "SAP_000.pdf"},
            ],
        )
        with (
            patch.object(ct_service, "_get_json", return_value=study),
            patch.object(ct_service, "_get_bytes", side_effect=capture_get_bytes),
        ):
            ct_service.download(
                nct_id="NCT10000005", version="1.0", fmt="SAP",
                who="tester", why="test",
            )

        assert len(captured_urls) == 1
        assert "SAP_000.pdf" in captured_urls[0]

    def test_fallback_to_json_when_no_large_docs(self, ct_service, tmp_path):
        """PTCV-28 Scenario: registration JSON stored when no largeDocumentModule."""
        study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT10000006",
                    "briefTitle": "No Docs Study",
                    "officialTitle": "No Docs Study",
                },
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"phases": ["PHASE2"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Sponsor"}
                },
                "conditionsModule": {"conditions": ["Diabetes"]},
            }
        }
        with patch.object(ct_service, "_get_json", return_value=study):
            result = ct_service.download(
                nct_id="NCT10000006", version="1.0", fmt="PDF",
                who="tester", why="test",
            )

        assert result.success is True
        assert result.actual_format == "JSON"
        # Stored content must be valid JSON registration record
        stored = json.loads(Path(result.file_path).read_bytes())
        assert (
            stored["protocolSection"]["identificationModule"]["nctId"]
            == "NCT10000006"
        )

    def test_fallback_to_json_when_pdf_fetch_fails(self, ct_service, tmp_path):
        """PTCV-28: HTTP error on ProvidedDocs URL falls back to registration JSON."""
        study = self._study_with_large_docs(
            "NCT10000007",
            [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
        )
        with (
            patch.object(ct_service, "_get_json", return_value=study),
            patch.object(
                ct_service,
                "_get_bytes",
                side_effect=Exception("connection refused"),
            ),
        ):
            result = ct_service.download(
                nct_id="NCT10000007", version="1.0", fmt="PDF",
                who="tester", why="test",
            )

        assert result.success is True
        assert result.actual_format == "JSON"

    def test_metadata_records_effective_format(self, ct_service, tmp_path):
        """PTCV-28: metadata format field reflects actual stored format."""
        fake_pdf = b"%PDF-1.7"
        study = self._study_with_large_docs(
            "NCT10000008",
            [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
        )
        with (
            patch.object(ct_service, "_get_json", return_value=study),
            patch.object(ct_service, "_get_bytes", return_value=fake_pdf),
        ):
            result = ct_service.download(
                nct_id="NCT10000008", version="1.0", fmt="PDF",
                who="tester", why="test",
            )

        assert result.success is True
        meta = json.loads(
            Path(result.metadata_path).read_text(encoding="utf-8")
        )
        assert meta["format"] == "PDF"

    def test_find_large_doc_returns_none_for_empty(self, ct_service):
        """_find_large_doc returns None when no largeDocs present."""
        study = {"protocolSection": {}}
        assert ct_service._find_large_doc(study, ("Prot",)) is None

    def test_find_large_doc_returns_none_for_no_match(self, ct_service):
        """_find_large_doc returns None when typeAbbrev does not match."""
        study = self._study_with_large_docs(
            "NCT99",
            [{"typeAbbrev": "ICF", "filename": "ICF_000.pdf"}],
        )
        assert ct_service._find_large_doc(study, ("Prot", "Prot_SAP")) is None

    def test_build_provided_docs_url(self, ct_service):
        """_build_provided_docs_url produces correct URL from NCT ID and filename."""
        url = ct_service._build_provided_docs_url("NCT12345678", "Prot_000.pdf")
        assert url == (
            "https://clinicaltrials.gov/ProvidedDocs/78/NCT12345678/Prot_000.pdf"
        )


class TestClinicalTrialsServiceSearchPdfAvailable:
    """Tests for ClinicalTrialsService.search_pdf_available() (PTCV-28)."""

    def _make_study(self, nct_id: str, large_docs: list[dict]) -> dict:
        """Build a minimal study dict as returned by the v2 search API."""
        study: dict = {
            "protocolSection": {
                "identificationModule": {"nctId": nct_id}
            }
        }
        if large_docs:
            study["documentSection"] = {
                "largeDocumentModule": {"largeDocs": large_docs}
            }
        return study

    def test_returns_nct_ids_with_prot_doc(self, ct_service):
        """NCT IDs with Prot typeAbbrev are returned."""
        studies = [
            self._make_study(
                "NCT11111111",
                [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
            ),
            self._make_study("NCT22222222", []),  # no large docs
        ]
        with patch.object(
            ct_service, "_get_json", return_value={"studies": studies}
        ):
            result = ct_service.search_pdf_available(condition="oncology")

        assert result == ["NCT11111111"]

    def test_returns_empty_when_no_pdf_available(self, ct_service):
        """Empty list returned when no studies have matching large docs."""
        studies = [
            self._make_study("NCT33333333", []),
            self._make_study(
                "NCT44444444",
                [{"typeAbbrev": "ICF", "filename": "ICF_000.pdf"}],
            ),
        ]
        with patch.object(
            ct_service, "_get_json", return_value={"studies": studies}
        ):
            result = ct_service.search_pdf_available(condition="test")

        assert result == []

    def test_prot_sap_satisfies_pdf_fmt(self, ct_service):
        """Prot_SAP typeAbbrev counts as PDF-available."""
        studies = [
            self._make_study(
                "NCT55555555",
                [{"typeAbbrev": "Prot_SAP", "filename": "ProtSAP_000.pdf"}],
            )
        ]
        with patch.object(
            ct_service, "_get_json", return_value={"studies": studies}
        ):
            result = ct_service.search_pdf_available(fmt="PDF")

        assert "NCT55555555" in result

    def test_sap_fmt_matches_sap_abbrev(self, ct_service):
        """SAP typeAbbrev is returned when fmt='SAP'."""
        studies = [
            self._make_study(
                "NCT66666666",
                [{"typeAbbrev": "SAP", "filename": "SAP_000.pdf"}],
            ),
            self._make_study(
                "NCT77777777",
                [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
            ),
        ]
        with patch.object(
            ct_service, "_get_json", return_value={"studies": studies}
        ):
            result = ct_service.search_pdf_available(fmt="SAP")

        assert "NCT66666666" in result
        assert "NCT77777777" not in result  # Prot does not satisfy SAP

    def test_unknown_fmt_returns_empty_without_api_call(self, ct_service):
        """Unrecognised fmt returns [] immediately without HTTP request."""
        with patch.object(ct_service, "_get_json") as mock_get:
            result = ct_service.search_pdf_available(fmt="UNKNOWN")

        assert result == []
        mock_get.assert_not_called()

    def test_pagination_followed(self, ct_service):
        """Second page is fetched when nextPageToken is present."""
        page1 = {
            "studies": [
                self._make_study(
                    "NCT88888881",
                    [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
                )
            ],
            "nextPageToken": "tok123",
        }
        page2 = {
            "studies": [
                self._make_study(
                    "NCT88888882",
                    [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
                )
            ],
        }
        with patch.object(
            ct_service, "_get_json", side_effect=[page1, page2]
        ):
            result = ct_service.search_pdf_available(
                condition="test", max_results=2
            )

        assert "NCT88888881" in result
        assert "NCT88888882" in result

    def test_stops_at_max_results(self, ct_service):
        """No more pages fetched once max_results studies have been scanned."""
        # Each page has 1 study; max_results=1 → only one page fetched.
        page1 = {
            "studies": [
                self._make_study(
                    "NCT99999991",
                    [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
                )
            ],
            "nextPageToken": "tok456",
        }
        with patch.object(
            ct_service, "_get_json", return_value=page1
        ) as mock_get:
            ct_service.search_pdf_available(condition="test", max_results=1)

        assert mock_get.call_count == 1

    def test_api_error_returns_partial_results(self, ct_service):
        """If HTTP fails mid-pagination, already-found IDs are returned."""
        page1 = {
            "studies": [
                self._make_study(
                    "NCT10101010",
                    [{"typeAbbrev": "Prot", "filename": "Prot_000.pdf"}],
                )
            ],
            "nextPageToken": "tokFail",
        }
        with patch.object(
            ct_service,
            "_get_json",
            side_effect=[page1, Exception("timeout")],
        ):
            result = ct_service.search_pdf_available(
                condition="test", max_results=200
            )

        assert "NCT10101010" in result
