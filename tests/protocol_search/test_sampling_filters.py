"""Tests for PTCV-39 sponsor and date filtering on ClinicalTrialsService.

Qualification phase: OQ (operational qualification)
Regulatory requirement: PTCV-39 — ICH E6(R3)-aligned sampling strategy.
Risk tier: MEDIUM — search filter construction.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.protocol_search.clinicaltrials_service import ClinicalTrialsService


class TestBuildAdvancedFilter:
    """Unit tests for _build_advanced_filter() static method."""

    def test_empty_when_no_filters(self):
        """No filter components produces empty string."""
        assert ClinicalTrialsService._build_advanced_filter() == ""

    def test_sponsor_class_only(self):
        """Single sponsor class filter."""
        result = ClinicalTrialsService._build_advanced_filter(
            sponsor_class="INDUSTRY"
        )
        assert result == "AREA[LeadSponsorClass]INDUSTRY"

    def test_date_from_only(self):
        """Single date range filter."""
        result = ClinicalTrialsService._build_advanced_filter(
            date_from="10/01/2024"
        )
        assert result == "AREA[StudyFirstPostDate]RANGE[10/01/2024, MAX]"

    def test_has_protocol_doc_only(self):
        """Single protocol doc filter."""
        result = ClinicalTrialsService._build_advanced_filter(
            has_protocol_doc=True
        )
        assert result == "AREA[LargeDocHasProtocol]true"

    def test_combined_sponsor_and_date(self):
        """Sponsor class + date range joined with AND."""
        result = ClinicalTrialsService._build_advanced_filter(
            sponsor_class="INDUSTRY",
            date_from="10/01/2024",
        )
        assert "AREA[LeadSponsorClass]INDUSTRY" in result
        assert "AREA[StudyFirstPostDate]RANGE[10/01/2024, MAX]" in result
        assert " AND " in result

    def test_all_three_filters(self):
        """All three filters joined with AND."""
        result = ClinicalTrialsService._build_advanced_filter(
            sponsor_class="INDUSTRY",
            date_from="01/01/2024",
            has_protocol_doc=True,
        )
        parts = result.split(" AND ")
        assert len(parts) == 3
        assert "AREA[LeadSponsorClass]INDUSTRY" in parts
        assert "AREA[StudyFirstPostDate]RANGE[01/01/2024, MAX]" in parts
        assert "AREA[LargeDocHasProtocol]true" in parts

    def test_other_sponsor_classes(self):
        """Non-INDUSTRY sponsor classes are formatted correctly."""
        for cls in ("NIH", "FED", "OTHER", "NETWORK", "INDIV"):
            result = ClinicalTrialsService._build_advanced_filter(
                sponsor_class=cls
            )
            assert result == f"AREA[LeadSponsorClass]{cls}"

    def test_has_protocol_doc_false_excluded(self):
        """has_protocol_doc=False does not add a filter clause."""
        result = ClinicalTrialsService._build_advanced_filter(
            sponsor_class="INDUSTRY",
            has_protocol_doc=False,
        )
        assert "LargeDocHasProtocol" not in result


class TestSearchWithSponsorAndDateFilters:
    """OQ: PTCV-39 Scenario: Sponsor-filtered and temporal filtering."""

    def _study_stub(
        self,
        nct_id: str,
        sponsor: str = "Pfizer",
        sponsor_class: str = "INDUSTRY",
    ) -> dict:
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct_id,
                    "briefTitle": "Test Study",
                    "officialTitle": "Test Study",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"phases": ["PHASE3"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {
                        "name": sponsor,
                        "class": sponsor_class,
                    }
                },
                "conditionsModule": {"conditions": ["Oncology"]},
            }
        }

    def test_sponsor_class_filter_passed_to_api(self, ct_service):
        """sponsor_class param adds filter.advanced to API call."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search(sponsor_class="INDUSTRY")

        assert captured
        advanced = captured[0].get("filter.advanced", "")
        assert "AREA[LeadSponsorClass]INDUSTRY" in advanced

    def test_date_from_filter_passed_to_api(self, ct_service):
        """date_from param adds date range to filter.advanced."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search(date_from="10/01/2024")

        assert captured
        advanced = captured[0].get("filter.advanced", "")
        assert "AREA[StudyFirstPostDate]RANGE[10/01/2024, MAX]" in advanced

    def test_has_protocol_doc_filter_passed_to_api(self, ct_service):
        """has_protocol_doc=True adds LargeDocHasProtocol to filter."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search(has_protocol_doc=True)

        assert captured
        advanced = captured[0].get("filter.advanced", "")
        assert "AREA[LargeDocHasProtocol]true" in advanced

    def test_combined_filters_all_present(self, ct_service):
        """All filters combined into single filter.advanced param."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search(
                sponsor_class="INDUSTRY",
                date_from="10/01/2024",
                has_protocol_doc=True,
            )

        advanced = captured[0]["filter.advanced"]
        assert "LeadSponsorClass" in advanced
        assert "StudyFirstPostDate" in advanced
        assert "LargeDocHasProtocol" in advanced

    def test_no_advanced_filter_when_no_new_params(self, ct_service):
        """Legacy calls without new params do not add filter.advanced."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search(condition="oncology")

        assert "filter.advanced" not in captured[0]

    def test_search_returns_results_with_filters(self, ct_service):
        """Filtered search still returns valid SearchResult objects."""
        payload = {
            "studies": [self._study_stub("NCT99990001", "Pfizer", "INDUSTRY")]
        }
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search(
                sponsor_class="INDUSTRY",
                date_from="10/01/2024",
            )

        assert len(results) == 1
        assert results[0].registry_id == "NCT99990001"
        assert results[0].sponsor == "Pfizer"

    def test_fields_include_lead_sponsor_class(self, ct_service):
        """API fields parameter includes LeadSponsorClass."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search(sponsor_class="INDUSTRY")

        fields = captured[0]["fields"]
        assert "LeadSponsorClass" in fields

    def test_audit_log_includes_new_filter_params(self, ct_service, tmp_path):
        """Audit reason includes sponsor_class and date_from."""
        import json

        with patch.object(
            ct_service, "_get_json", return_value={"studies": []}
        ):
            ct_service.search(
                sponsor_class="INDUSTRY",
                date_from="10/01/2024",
                who="tester",
            )

        audit_file = tmp_path / "audit.jsonl"
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert "sponsor_class='INDUSTRY'" in entry["reason"]
        assert "date_from='10/01/2024'" in entry["reason"]


class TestSearchPdfAvailableWithFilters:
    """PTCV-39: search_pdf_available() accepts sponsor/date filters."""

    def _make_study(self, nct_id: str, large_docs: list[dict]) -> dict:
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

    def test_sponsor_class_filter_forwarded(self, ct_service):
        """sponsor_class passed to filter.advanced on search_pdf_available."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search_pdf_available(
                sponsor_class="INDUSTRY",
                date_from="01/01/2024",
            )

        advanced = captured[0].get("filter.advanced", "")
        assert "AREA[LeadSponsorClass]INDUSTRY" in advanced
        assert "StudyFirstPostDate" in advanced


class TestSearchIndustryProtocols:
    """PTCV-39: search_industry_protocols() convenience method."""

    def _study_stub(self, nct_id: str) -> dict:
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct_id,
                    "briefTitle": "Industry Trial",
                    "officialTitle": "Industry Trial",
                },
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"phases": ["PHASE3"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {
                        "name": "Big Pharma",
                        "class": "INDUSTRY",
                    }
                },
                "conditionsModule": {"conditions": ["Oncology"]},
            }
        }

    def test_defaults_to_industry_and_post_oct_2024(self, ct_service):
        """Default call uses INDUSTRY + 10/01/2024 + has_protocol_doc."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search_industry_protocols()

        advanced = captured[0]["filter.advanced"]
        assert "AREA[LeadSponsorClass]INDUSTRY" in advanced
        assert "AREA[StudyFirstPostDate]RANGE[10/01/2024, MAX]" in advanced
        assert "AREA[LargeDocHasProtocol]true" in advanced

    def test_custom_date_from(self, ct_service):
        """Custom date_from overrides default."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search_industry_protocols(date_from="01/01/2025")

        advanced = captured[0]["filter.advanced"]
        assert "01/01/2025" in advanced

    def test_returns_search_results(self, ct_service):
        """Convenience method returns list of SearchResult."""
        payload = {"studies": [self._study_stub("NCT88880001")]}
        with patch.object(ct_service, "_get_json", return_value=payload):
            results = ct_service.search_industry_protocols(max_results=10)

        assert len(results) == 1
        assert results[0].registry_id == "NCT88880001"

    def test_condition_and_phase_forwarded(self, ct_service):
        """Optional condition and phase are forwarded to search."""
        captured: list[dict] = []

        def capture(url, params):
            captured.append(params)
            return {"studies": []}

        with patch.object(ct_service, "_get_json", side_effect=capture):
            ct_service.search_industry_protocols(
                condition="oncology", phase="PHASE3"
            )

        assert captured[0].get("query.cond") == "oncology"
        assert captured[0].get("query.term") == "PHASE3"
