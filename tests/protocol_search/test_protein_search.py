"""Tests for PTCV-190 protein-based protocol search and CSV index.

Qualification phase: OQ (operational qualification)
Regulatory requirement: PTCV-190 Scenarios — protein search, PDF
  filtering, CSV index generation, protein deduplication.
Risk tier: MEDIUM
"""

import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.protocol_search.clinicaltrials_service import ClinicalTrialsService
from ptcv.protocol_search.models import ProteinSearchResult
from ptcv.protocol_search.protein_search import write_protein_csv_index


def _study_with_protein(
    nct_id: str,
    title: str = "Test Study",
    has_large_doc: bool = True,
    conditions: list[str] | None = None,
    phases: list[str] | None = None,
    sponsor: str = "Test Sponsor",
    status: str = "RECRUITING",
    start_date: str = "2024-01-15",
    results_date: str = "",
    references: list[dict] | None = None,
) -> dict:
    """Build a v2 API study stub for protein search tests."""
    study: dict = {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": title,
                "officialTitle": title,
            },
            "statusModule": {
                "overallStatus": status,
                "startDateStruct": {"date": start_date},
            },
            "designModule": {
                "phases": phases or ["PHASE2"],
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": sponsor},
            },
            "conditionsModule": {
                "conditions": conditions or ["Oncology"],
            },
            "referencesModule": {
                "references": references or [],
            },
        },
    }
    if results_date:
        study["protocolSection"]["statusModule"][
            "resultsFirstPostDateStruct"
        ] = {"date": results_date}
    if has_large_doc:
        study["documentSection"] = {
            "largeDocumentModule": {
                "largeDocs": [
                    {
                        "typeAbbrev": "Prot",
                        "hasProtocol": True,
                        "filename": "Prot_000.pdf",
                    }
                ]
            }
        }
    return study


class TestSearchByProteins:
    """OQ: PTCV-190 Scenario: Search by protein list."""

    def test_single_protein_returns_results(self, ct_service):
        """Search for one protein returns matching trials."""
        payload = {
            "studies": [
                _study_with_protein("NCT11111111", "VEGF Trial"),
            ],
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_by_proteins(["VEGF"])

        assert len(results) == 1
        assert results[0].registry_id == "NCT11111111"
        assert results[0].matched_proteins == ["VEGF"]
        assert isinstance(results[0], ProteinSearchResult)

    def test_multiple_proteins_deduplicates(self, ct_service):
        """Same NCT ID from two proteins is deduplicated."""
        payload = {
            "studies": [
                _study_with_protein("NCT11111111", "VEGF+PD1 Trial"),
            ],
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_by_proteins(["VEGF", "PD-1"])

        assert len(results) == 1
        assert set(results[0].matched_proteins) == {"VEGF", "PD-1"}

    def test_different_proteins_different_trials(self, ct_service):
        """Different proteins matching different trials both appear."""
        call_count = [0]

        def side_effect(url, params):
            call_count[0] += 1
            protein = params.get("query.term", "")
            if protein == "VEGF":
                return {
                    "studies": [
                        _study_with_protein("NCT11111111", "VEGF Trial")
                    ]
                }
            return {
                "studies": [
                    _study_with_protein("NCT22222222", "HER2 Trial")
                ]
            }

        with patch.object(
            ct_service, "_get_json", side_effect=side_effect
        ):
            results = ct_service.search_by_proteins(["VEGF", "HER2"])

        assert len(results) == 2
        ids = {r.registry_id for r in results}
        assert ids == {"NCT11111111", "NCT22222222"}

    def test_pdf_only_excludes_no_pdf(self, ct_service):
        """pdf_only=True excludes trials without protocol PDFs."""
        payload = {
            "studies": [
                _study_with_protein(
                    "NCT11111111", "Has PDF", has_large_doc=True
                ),
                _study_with_protein(
                    "NCT22222222", "No PDF", has_large_doc=False
                ),
            ],
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_by_proteins(
                ["VEGF"], pdf_only=True
            )

        assert len(results) == 1
        assert results[0].registry_id == "NCT11111111"

    def test_pdf_only_false_includes_all(self, ct_service):
        """pdf_only=False includes trials without protocol PDFs."""
        payload = {
            "studies": [
                _study_with_protein(
                    "NCT11111111", "Has PDF", has_large_doc=True
                ),
                _study_with_protein(
                    "NCT22222222", "No PDF", has_large_doc=False
                ),
            ],
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_by_proteins(
                ["VEGF"], pdf_only=False
            )

        assert len(results) == 2

    def test_empty_proteins_returns_empty(self, ct_service):
        """Empty protein list returns no results."""
        results = ct_service.search_by_proteins([])
        assert results == []

    def test_network_error_returns_partial(self, ct_service):
        """Network error on one protein still returns other results."""
        call_count = [0]

        def side_effect(url, params):
            protein = params.get("query.term", "")
            if protein == "VEGF":
                return {
                    "studies": [
                        _study_with_protein("NCT11111111", "VEGF Trial")
                    ]
                }
            raise Exception("DNS failure")

        with patch.object(
            ct_service, "_get_json", side_effect=side_effect
        ):
            results = ct_service.search_by_proteins(["VEGF", "BAD"])

        assert len(results) == 1
        assert results[0].registry_id == "NCT11111111"


class TestProteinResultMetadata:
    """OQ: PTCV-190 metadata extraction for protein results."""

    def test_year_from_start_date(self, ct_service):
        """Year is extracted from startDateStruct."""
        payload = {
            "studies": [
                _study_with_protein(
                    "NCT11111111", start_date="2023-06-01"
                ),
            ],
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_by_proteins(["VEGF"])

        assert results[0].year == "2023"

    def test_outcome_has_results(self, ct_service):
        """Outcome shows 'Has Results' when resultsFirstPostDate exists."""
        payload = {
            "studies": [
                _study_with_protein(
                    "NCT11111111", results_date="2025-01-01"
                ),
            ],
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_by_proteins(["VEGF"])

        assert results[0].outcome == "Has Results"

    def test_outcome_falls_back_to_status(self, ct_service):
        """Outcome falls back to overallStatus when no results."""
        payload = {
            "studies": [
                _study_with_protein(
                    "NCT11111111", status="COMPLETED"
                ),
            ],
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_by_proteins(["VEGF"])

        assert results[0].outcome == "COMPLETED"

    def test_publications_from_pmid(self, ct_service):
        """PubMed references are formatted as URLs."""
        refs = [
            {"pmid": "12345678", "citation": "Smith et al 2024"},
            {"pmid": "87654321", "citation": "Jones et al 2024"},
        ]
        payload = {
            "studies": [
                _study_with_protein(
                    "NCT11111111", references=refs
                ),
            ],
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_by_proteins(["VEGF"])

        pubs = results[0].publications
        assert "pubmed.ncbi.nlm.nih.gov/12345678" in pubs
        assert "pubmed.ncbi.nlm.nih.gov/87654321" in pubs

    def test_audit_trail_written(self, ct_service, tmp_path):
        """Protein search writes audit log entry."""
        payload = {"studies": []}
        with patch.object(ct_service, "_get_json", return_value=payload):
            ct_service.search_by_proteins(
                ["VEGF"], who="test-user"
            )

        audit_file = tmp_path / "audit.jsonl"
        assert audit_file.exists()
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["action"] == "SEARCH"
        assert "VEGF" in entry["reason"]


class TestWriteProteinCSVIndex:
    """OQ: PTCV-190 Scenario: Generate CSV index."""

    def _make_results(self) -> list[ProteinSearchResult]:
        return [
            ProteinSearchResult(
                registry_id="NCT11111111",
                title="VEGF Inhibitor Trial",
                sponsor="Pharma Co",
                phase="PHASE2",
                condition="Oncology",
                status="COMPLETED",
                url="https://clinicaltrials.gov/study/NCT11111111",
                matched_proteins=["VEGF"],
                year="2023",
                outcome="Has Results",
                publications="https://pubmed.ncbi.nlm.nih.gov/12345/",
                has_protocol_pdf=True,
            ),
            ProteinSearchResult(
                registry_id="NCT22222222",
                title="PD-1 Checkpoint Trial",
                sponsor="BioTech Inc",
                phase="PHASE3",
                condition="Melanoma",
                status="RECRUITING",
                url="https://clinicaltrials.gov/study/NCT22222222",
                matched_proteins=["PD-1", "PD-L1"],
                year="2024",
                outcome="RECRUITING",
                publications="",
                has_protocol_pdf=True,
            ),
        ]

    def test_csv_written_with_correct_columns(self, tmp_path):
        """CSV has all required columns from the PTCV-190 spec."""
        results = self._make_results()
        csv_path = tmp_path / "index.csv"
        write_protein_csv_index(results, csv_path)

        assert csv_path.exists()
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        expected_cols = {
            "nct_id", "trial_name", "condition", "year",
            "phase", "sponsor", "outcome", "publications",
            "matched_proteins", "url", "has_pdf", "file_path",
        }
        assert set(rows[0].keys()) == expected_cols

    def test_csv_content_matches_results(self, tmp_path):
        """CSV row content matches the input ProteinSearchResult."""
        results = self._make_results()
        csv_path = tmp_path / "index.csv"
        write_protein_csv_index(results, csv_path)

        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        row = rows[0]
        assert row["nct_id"] == "NCT11111111"
        assert row["trial_name"] == "VEGF Inhibitor Trial"
        assert row["year"] == "2023"
        assert row["sponsor"] == "Pharma Co"
        assert row["outcome"] == "Has Results"
        assert row["has_pdf"] == "yes"

    def test_csv_matched_proteins_semicolon_joined(self, tmp_path):
        """Multiple matched proteins are semicolon-separated."""
        results = self._make_results()
        csv_path = tmp_path / "index.csv"
        write_protein_csv_index(results, csv_path)

        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # Second row has two proteins
        assert rows[1]["matched_proteins"] == "PD-1; PD-L1"

    def test_csv_includes_download_paths(self, tmp_path):
        """Download paths are included when provided."""
        results = self._make_results()
        csv_path = tmp_path / "index.csv"
        paths = {"NCT11111111": "/data/NCT11111111_1.0.pdf"}
        write_protein_csv_index(results, csv_path, download_paths=paths)

        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["file_path"] == "/data/NCT11111111_1.0.pdf"
        assert rows[1]["file_path"] == ""

    def test_csv_creates_parent_directories(self, tmp_path):
        """Parent directories are created if they don't exist."""
        results = self._make_results()
        csv_path = tmp_path / "nested" / "dir" / "index.csv"
        write_protein_csv_index(results, csv_path)

        assert csv_path.exists()

    def test_empty_results_writes_header_only(self, tmp_path):
        """Empty results list writes CSV with header but no rows."""
        csv_path = tmp_path / "index.csv"
        write_protein_csv_index([], csv_path)

        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 0
