"""Tests for trial curator (PTCV-85).

Covers GHERKIN scenarios:
  - Query filters by industry sponsor
  - Multi-indication filtering
  - Dose-ranging identification
  - Full protocol download
  - Output manifest
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ptcv.protocol_search.trial_curator import (
    QualifyingTrial,
    _DOSE_KEYWORDS,
    _DOSE_PATTERN,
    _extract_conditions,
    _extract_nct_id,
    _extract_phase,
    _extract_sponsor,
    _extract_status,
    _extract_title,
    _has_protocol_pdf,
    _search_raw,
    curate_trials,
    is_dose_ranging,
    is_multi_indication,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic study dicts
# ---------------------------------------------------------------------------

def _make_study(
    nct_id: str = "NCT00112827",
    title: str = "A Phase 2 Dose-Ranging Study of Drug X",
    sponsor: str = "Pfizer",
    conditions: list[str] | None = None,
    arms: list[dict] | None = None,
    interventions: list[dict] | None = None,
    phases: list[str] | None = None,
    status: str = "COMPLETED",
    brief_summary: str = "",
    has_prot_pdf: bool = True,
) -> dict:
    """Build a realistic ClinicalTrials.gov v2 study dict."""
    if conditions is None:
        conditions = ["Breast Cancer", "Lung Cancer"]
    if phases is None:
        phases = ["PHASE2"]
    if arms is None:
        arms = [
            {
                "label": "Drug X 10 mg",
                "type": "EXPERIMENTAL",
                "description": "10 mg oral daily",
                "interventionNames": ["Drug X"],
            },
            {
                "label": "Drug X 25 mg",
                "type": "EXPERIMENTAL",
                "description": "25 mg oral daily",
                "interventionNames": ["Drug X"],
            },
            {
                "label": "Placebo",
                "type": "PLACEBO_COMPARATOR",
                "description": "Matching placebo",
                "interventionNames": ["Placebo"],
            },
        ]
    if interventions is None:
        interventions = [
            {
                "name": "Drug X",
                "type": "DRUG",
                "armGroupLabels": ["Drug X 10 mg", "Drug X 25 mg"],
            },
        ]

    large_docs = []
    if has_prot_pdf:
        large_docs = [
            {
                "typeAbbrev": "Prot",
                "hasProtocol": True,
                "filename": "Prot_000.pdf",
            }
        ]

    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "officialTitle": title,
                "briefTitle": title[:60],
            },
            "statusModule": {"overallStatus": status},
            "designModule": {"phases": phases},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {
                    "name": sponsor,
                    "class": "INDUSTRY",
                }
            },
            "conditionsModule": {"conditions": conditions},
            "armsInterventionsModule": {
                "armGroups": arms,
                "interventions": interventions,
            },
            "descriptionModule": {
                "briefSummary": brief_summary,
            },
        },
        "documentSection": {
            "largeDocumentModule": {
                "largeDocs": large_docs,
            }
        },
    }


# ---------------------------------------------------------------------------
# TestMultiIndication
# ---------------------------------------------------------------------------

class TestMultiIndication:
    """Scenario: Multi-indication filtering."""

    def test_two_conditions_qualifies(self) -> None:
        assert is_multi_indication(["Breast Cancer", "Lung Cancer"]) is True

    def test_three_conditions_qualifies(self) -> None:
        assert is_multi_indication(
            ["Melanoma", "NSCLC", "Renal Cell Carcinoma"]
        ) is True

    def test_single_condition_excluded(self) -> None:
        assert is_multi_indication(["Breast Cancer"]) is False

    def test_empty_conditions_excluded(self) -> None:
        assert is_multi_indication([]) is False

    def test_duplicate_conditions_counted_once(self) -> None:
        assert is_multi_indication(["NSCLC", "NSCLC"]) is False


# ---------------------------------------------------------------------------
# TestDoseRanging
# ---------------------------------------------------------------------------

class TestDoseRanging:
    """Scenario: Dose-ranging identification."""

    def test_keyword_in_title(self) -> None:
        study = _make_study(
            title="A Phase 2 Dose-Ranging Study of Drug X",
            arms=[],
            interventions=[],
        )
        flag, signal, arms = is_dose_ranging(study)
        assert flag is True
        assert signal == "keyword"

    def test_keyword_dose_finding(self) -> None:
        study = _make_study(
            title="Phase 1 Dose-Finding Study",
            arms=[],
            interventions=[],
        )
        flag, signal, _ = is_dose_ranging(study)
        assert flag is True
        assert signal == "keyword"

    def test_keyword_dose_escalation(self) -> None:
        study = _make_study(
            title="Normal title",
            brief_summary="This is a dose escalation study.",
            arms=[],
            interventions=[],
        )
        flag, signal, _ = is_dose_ranging(study)
        assert flag is True
        assert signal == "keyword"

    def test_keyword_dose_response(self) -> None:
        study = _make_study(
            title="Phase 2 Dose-Response Evaluation",
            arms=[],
            interventions=[],
        )
        flag, signal, _ = is_dose_ranging(study)
        assert flag is True

    def test_arms_based_detection(self) -> None:
        study = _make_study(
            title="A Study of Drug Y in Solid Tumors",
            arms=[
                {
                    "label": "Drug Y 50 mg",
                    "type": "EXPERIMENTAL",
                    "description": "50 mg once daily",
                    "interventionNames": ["Drug Y"],
                },
                {
                    "label": "Drug Y 100 mg",
                    "type": "EXPERIMENTAL",
                    "description": "100 mg once daily",
                    "interventionNames": ["Drug Y"],
                },
            ],
        )
        flag, signal, arm_labels = is_dose_ranging(study)
        assert flag is True
        assert signal == "arms"
        assert "Drug Y 50 mg" in arm_labels
        assert "Drug Y 100 mg" in arm_labels

    def test_placebo_arms_excluded(self) -> None:
        """Placebo arms should not count toward dose-ranging."""
        study = _make_study(
            title="A Study of Drug Z",
            arms=[
                {
                    "label": "Drug Z 10 mg",
                    "type": "EXPERIMENTAL",
                    "description": "10 mg",
                    "interventionNames": ["Drug Z"],
                },
                {
                    "label": "Placebo",
                    "type": "PLACEBO_COMPARATOR",
                    "description": "Matching placebo",
                    "interventionNames": ["Placebo"],
                },
            ],
        )
        flag, _, _ = is_dose_ranging(study)
        assert flag is False

    def test_single_dose_arm_not_dose_ranging(self) -> None:
        study = _make_study(
            title="Efficacy of Drug A",
            arms=[
                {
                    "label": "Drug A 200 mg",
                    "type": "EXPERIMENTAL",
                    "description": "200 mg daily",
                    "interventionNames": ["Drug A"],
                },
            ],
        )
        flag, _, _ = is_dose_ranging(study)
        assert flag is False

    def test_no_arms_no_keywords_not_dose_ranging(self) -> None:
        study = _make_study(
            title="Efficacy Study",
            arms=[],
            interventions=[],
        )
        flag, signal, _ = is_dose_ranging(study)
        assert flag is False
        assert signal == ""

    def test_different_interventions_not_dose_ranging(self) -> None:
        """Two different drugs are not dose-ranging."""
        study = _make_study(
            title="Combination Study",
            arms=[
                {
                    "label": "Drug A 10 mg",
                    "type": "EXPERIMENTAL",
                    "description": "10 mg Drug A",
                    "interventionNames": ["Drug A"],
                },
                {
                    "label": "Drug B 50 mg",
                    "type": "EXPERIMENTAL",
                    "description": "50 mg Drug B",
                    "interventionNames": ["Drug B"],
                },
            ],
        )
        flag, _, _ = is_dose_ranging(study)
        assert flag is False


# ---------------------------------------------------------------------------
# TestDosePatternRegex
# ---------------------------------------------------------------------------

class TestDosePatternRegex:
    """Verify dose extraction regex catches common patterns."""

    def test_mg(self) -> None:
        assert _DOSE_PATTERN.findall("10 mg daily") == ["10"]

    def test_mg_no_space(self) -> None:
        assert _DOSE_PATTERN.findall("100mg") == ["100"]

    def test_mcg(self) -> None:
        assert _DOSE_PATTERN.findall("200 mcg") == ["200"]

    def test_mg_per_kg(self) -> None:
        assert _DOSE_PATTERN.findall("5 mg/kg") == ["5"]

    def test_decimal(self) -> None:
        assert _DOSE_PATTERN.findall("0.5 mg") == ["0.5"]

    def test_iu(self) -> None:
        assert _DOSE_PATTERN.findall("1000 IU") == ["1000"]


# ---------------------------------------------------------------------------
# TestDoseKeywordRegex
# ---------------------------------------------------------------------------

class TestDoseKeywordRegex:
    """Verify keyword detection regex."""

    def test_dose_ranging(self) -> None:
        assert _DOSE_KEYWORDS.search("dose-ranging") is not None

    def test_dose_finding(self) -> None:
        assert _DOSE_KEYWORDS.search("Dose Finding") is not None

    def test_dose_escalation(self) -> None:
        assert _DOSE_KEYWORDS.search("dose escalation") is not None

    def test_dose_response(self) -> None:
        assert _DOSE_KEYWORDS.search("dose-response") is not None

    def test_dose_titration(self) -> None:
        assert _DOSE_KEYWORDS.search("dose titration") is not None

    def test_no_match(self) -> None:
        assert _DOSE_KEYWORDS.search("fixed dose study") is None


# ---------------------------------------------------------------------------
# TestExtractors
# ---------------------------------------------------------------------------

class TestExtractors:
    """Tests for data extraction helpers."""

    def test_extract_nct_id(self) -> None:
        study = _make_study(nct_id="NCT99887766")
        assert _extract_nct_id(study) == "NCT99887766"

    def test_extract_conditions(self) -> None:
        study = _make_study(conditions=["Asthma", "COPD"])
        assert _extract_conditions(study) == ["Asthma", "COPD"]

    def test_extract_title(self) -> None:
        study = _make_study(title="My Trial Title")
        assert _extract_title(study) == "My Trial Title"

    def test_extract_sponsor(self) -> None:
        study = _make_study(sponsor="Novartis")
        assert _extract_sponsor(study) == "Novartis"

    def test_extract_phase(self) -> None:
        study = _make_study(phases=["PHASE2", "PHASE3"])
        assert _extract_phase(study) == "PHASE2, PHASE3"

    def test_extract_status(self) -> None:
        study = _make_study(status="RECRUITING")
        assert _extract_status(study) == "RECRUITING"

    def test_has_protocol_pdf_true(self) -> None:
        study = _make_study(has_prot_pdf=True)
        assert _has_protocol_pdf(study) is True

    def test_has_protocol_pdf_false(self) -> None:
        study = _make_study(has_prot_pdf=False)
        assert _has_protocol_pdf(study) is False

    def test_extract_from_empty_study(self) -> None:
        assert _extract_nct_id({}) == ""
        assert _extract_conditions({}) == []
        assert _extract_title({}) == ""
        assert _extract_sponsor({}) == ""
        assert _extract_phase({}) == ""
        assert _extract_status({}) == ""
        assert _has_protocol_pdf({}) is False


# ---------------------------------------------------------------------------
# TestCurateTrials
# ---------------------------------------------------------------------------

class TestCurateTrials:
    """Scenario: Full curation pipeline with mocked API/downloads."""

    def _mock_search_raw(self, studies: list[dict]) -> Any:
        """Patch _search_raw to return given studies."""
        return patch(
            "ptcv.protocol_search.trial_curator._search_raw",
            return_value=studies,
        )

    def test_qualifying_trial_included(self) -> None:
        study = _make_study()
        service = MagicMock()
        service.download.return_value = MagicMock(
            success=True,
            file_path="/data/NCT00112827_1.0.pdf",
            error=None,
        )
        with self._mock_search_raw([study]):
            results = curate_trials(service, max_search=10)
        assert len(results) == 1
        assert results[0].nct_id == "NCT00112827"
        assert results[0].file_path == "/data/NCT00112827_1.0.pdf"

    def test_single_condition_excluded(self) -> None:
        study = _make_study(conditions=["Breast Cancer"])
        service = MagicMock()
        with self._mock_search_raw([study]):
            results = curate_trials(service, max_search=10)
        assert len(results) == 0

    def test_non_dose_ranging_excluded(self) -> None:
        study = _make_study(
            title="Efficacy Study",
            arms=[
                {
                    "label": "Drug X 10 mg",
                    "type": "EXPERIMENTAL",
                    "description": "10 mg",
                    "interventionNames": ["Drug X"],
                },
            ],
        )
        service = MagicMock()
        with self._mock_search_raw([study]):
            results = curate_trials(service, max_search=10)
        assert len(results) == 0

    def test_no_protocol_pdf_excluded(self) -> None:
        study = _make_study(has_prot_pdf=False)
        service = MagicMock()
        with self._mock_search_raw([study]):
            results = curate_trials(service, max_search=10)
        assert len(results) == 0

    def test_download_error_captured(self) -> None:
        study = _make_study()
        service = MagicMock()
        service.download.return_value = MagicMock(
            success=False,
            file_path="",
            error="HTTP 404",
        )
        with self._mock_search_raw([study]):
            results = curate_trials(service, max_search=10)
        assert len(results) == 1
        assert results[0].download_error == "HTTP 404"
        assert results[0].file_path == ""

    def test_max_download_respected(self) -> None:
        studies = [
            _make_study(nct_id=f"NCT0000000{i}")
            for i in range(5)
        ]
        service = MagicMock()
        service.download.return_value = MagicMock(
            success=True, file_path="/data/test.pdf", error=None
        )
        with self._mock_search_raw(studies):
            results = curate_trials(
                service, max_search=10, max_download=2
            )
        assert len(results) == 5
        assert service.download.call_count == 2

    def test_multiple_qualifying_trials(self) -> None:
        studies = [
            _make_study(nct_id="NCT00000001", sponsor="Pfizer"),
            _make_study(nct_id="NCT00000002", sponsor="Roche"),
            _make_study(
                nct_id="NCT00000003",
                conditions=["Asthma"],  # single condition
            ),
        ]
        service = MagicMock()
        service.download.return_value = MagicMock(
            success=True, file_path="/data/test.pdf", error=None
        )
        with self._mock_search_raw(studies):
            results = curate_trials(service, max_search=10)
        assert len(results) == 2
        nct_ids = {r.nct_id for r in results}
        assert nct_ids == {"NCT00000001", "NCT00000002"}


# ---------------------------------------------------------------------------
# TestWriteManifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    """Scenario: Output manifest."""

    def test_writes_valid_json(self) -> None:
        trials = [
            QualifyingTrial(
                nct_id="NCT00112827",
                title="Test Study",
                sponsor="Pfizer",
                conditions=["Cancer A", "Cancer B"],
                dose_arms=["10 mg", "25 mg"],
                dose_ranging_signal="arms",
                phase="PHASE2",
                status="COMPLETED",
                url="https://clinicaltrials.gov/study/NCT00112827",
                file_path="/data/NCT00112827_1.0.pdf",
            ),
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            write_manifest(trials, path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_qualifying"] == 1
            assert data["downloaded"] == 1
            assert data["download_errors"] == 0
            assert len(data["trials"]) == 1
            assert data["trials"][0]["nct_id"] == "NCT00112827"
        finally:
            os.unlink(path)

    def test_counts_download_errors(self) -> None:
        trials = [
            QualifyingTrial(
                nct_id="NCT001",
                title="T1",
                sponsor="S",
                conditions=["A", "B"],
                dose_arms=[],
                dose_ranging_signal="keyword",
                phase="",
                status="",
                url="",
                file_path="/ok.pdf",
            ),
            QualifyingTrial(
                nct_id="NCT002",
                title="T2",
                sponsor="S",
                conditions=["A", "B"],
                dose_arms=[],
                dose_ranging_signal="keyword",
                phase="",
                status="",
                url="",
                download_error="HTTP 500",
            ),
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            write_manifest(trials, path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_qualifying"] == 2
            assert data["downloaded"] == 1
            assert data["download_errors"] == 1
        finally:
            os.unlink(path)

    def test_empty_manifest(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            write_manifest([], path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_qualifying"] == 0
            assert data["trials"] == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestSearchRaw
# ---------------------------------------------------------------------------

class TestSearchRaw:
    """Tests for _search_raw with mocked HTTP."""

    def test_builds_correct_filter(self) -> None:
        """Verify the API request includes industry + protocol filters."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"studies": [], "nextPageToken": ""}
        ).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as m:
            _search_raw(max_results=10)
            called_url = m.call_args[0][0].full_url
            assert "INDUSTRY" in called_url
            assert "LargeDocHasProtocol" in called_url

    def test_pagination(self) -> None:
        """Verify pagination follows nextPageToken."""
        page1 = {
            "studies": [_make_study(nct_id="NCT001")],
            "nextPageToken": "token2",
        }
        page2 = {
            "studies": [_make_study(nct_id="NCT002")],
            "nextPageToken": "",
        }

        responses = iter([
            self._mock_http_response(page1),
            self._mock_http_response(page2),
        ])

        with patch(
            "urllib.request.urlopen",
            side_effect=lambda *a, **kw: next(responses),
        ):
            results = _search_raw(max_results=10)
        assert len(results) == 2

    def test_max_results_cap(self) -> None:
        """Verify max_results limits returned studies."""
        page = {
            "studies": [
                _make_study(nct_id=f"NCT{i:08d}") for i in range(5)
            ],
            "nextPageToken": "",
        }
        with patch(
            "urllib.request.urlopen",
            return_value=self._mock_http_response(page),
        ):
            results = _search_raw(max_results=3)
        assert len(results) == 3

    @staticmethod
    def _mock_http_response(data: dict) -> MagicMock:
        resp = MagicMock()
        resp.read.return_value = json.dumps(data).encode("utf-8")
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp
