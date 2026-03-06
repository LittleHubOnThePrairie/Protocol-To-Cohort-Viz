"""Tests for CDISC Library REST API integration (PTCV-58).

Covers all 4 GHERKIN acceptance criteria scenarios:

  Scenario 1: CT values validated against authoritative codelist
  Scenario 2: Unknown CT values flagged with suggestions
  Scenario 3: Biomedical concepts used for domain mapping
  Scenario 4: Offline fallback to local mappings

All tests use mocked HTTP responses — no real API calls are made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest

from ptcv.sdtm.cdisc_library import (
    CdiscLibraryClient,
    CdiscLibraryNormalizer,
    CodelistTerm,
    CtValidationResult,
    DomainMapping,
    _CdiscLibraryCache,
)
from ptcv.sdtm.ct_normalizer import CtLookupResult, CtNormalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cache_db(tmp_path: Path) -> Path:
    """Temporary SQLite cache database path."""
    return tmp_path / "test_cache.db"


@pytest.fixture()
def mock_phase_codelist() -> list[dict]:
    """Mock CDISC Library API response for Phase codelist (C66737)."""
    return {
        "terms": [
            {
                "conceptId": "C15720",
                "submissionValue": "PHASE I TRIAL",
                "preferredTerm": "Phase I Trial",
                "synonyms": "Phase 1",
            },
            {
                "conceptId": "C15961",
                "submissionValue": "PHASE II TRIAL",
                "preferredTerm": "Phase II Trial",
                "synonyms": "Phase 2",
            },
            {
                "conceptId": "C15962",
                "submissionValue": "PHASE III TRIAL",
                "preferredTerm": "Phase III Trial",
                "synonyms": "Phase 3",
            },
            {
                "conceptId": "C15963",
                "submissionValue": "PHASE IV TRIAL",
                "preferredTerm": "Phase IV Trial",
                "synonyms": "Phase 4",
            },
            {
                "conceptId": "C54723",
                "submissionValue": "PHASE 0 TRIAL",
                "preferredTerm": "Phase 0 Trial",
                "synonyms": "",
            },
        ],
    }


@pytest.fixture()
def mock_blinding_codelist() -> dict:
    """Mock CDISC Library API response for Blinding codelist (C66735)."""
    return {
        "terms": [
            {
                "conceptId": "C15228",
                "submissionValue": "DOUBLE BLIND",
                "preferredTerm": "Double Blind",
                "synonyms": "Double-Blind",
            },
            {
                "conceptId": "C49659",
                "submissionValue": "OPEN LABEL",
                "preferredTerm": "Open Label",
                "synonyms": "Open-Label|Unblinded",
            },
            {
                "conceptId": "C15229",
                "submissionValue": "SINGLE BLIND",
                "preferredTerm": "Single Blind",
                "synonyms": "Single-Blind",
            },
        ],
    }


@pytest.fixture()
def client(cache_db: Path) -> CdiscLibraryClient:
    """Client with fake API key and temp cache."""
    return CdiscLibraryClient(
        api_key="test-key-12345",
        cache_db=cache_db,
    )


@pytest.fixture()
def normalizer(cache_db: Path) -> CdiscLibraryNormalizer:
    """Enhanced normalizer with fake API key."""
    return CdiscLibraryNormalizer(
        api_key="test-key-12345",
        cache_db=cache_db,
    )


def _mock_api_response(client: CdiscLibraryClient, codelist_data: dict) -> None:
    """Seed the client's cache with mock codelist data."""
    # Directly cache the response to avoid HTTP calls
    for pkg in ["sdtmct-2024-03-29"]:
        for codelist_code in ["C66737", "C66735", "C66739", "C66738"]:
            key = (
                f"https://library.cdisc.org/api"
                f"/mdr/ct/packages/{pkg}/codelists/{codelist_code}"
            )
            client._cache.put(key, codelist_data)


# ---------------------------------------------------------------------------
# Scenario 1: CT values validated against authoritative codelist
# ---------------------------------------------------------------------------

class TestCtValidation:
    """Scenario: CT values validated against authoritative codelist."""

    def test_valid_phase_iii_matches(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Given CDISC Library has Phase III, PHASE III TRIAL passes."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "PHASE III TRIAL")
        assert result.valid is True
        assert result.matched_term is not None
        assert result.matched_term.code == "C15962"

    def test_valid_value_no_review_queue(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Valid values should not need review queue entry."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "Phase III Trial")
        assert result.valid is True
        assert len(result.suggestions) == 0

    def test_case_insensitive_matching(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Matching should be case-insensitive."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "phase iii trial")
        assert result.valid is True

    def test_preferred_term_also_matches(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Match by preferred term (not just submission value)."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "Phase II Trial")
        assert result.valid is True
        assert result.matched_term is not None
        assert result.matched_term.code == "C15961"

    def test_source_is_cache_when_cached(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Source should be 'cache' when using cached API data."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "Phase III Trial")
        assert result.source == "cache"

    def test_validation_result_has_package_version(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Package version should be populated in result."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "Phase III Trial")
        assert result.package_version == "sdtmct-2024-03-29"

    def test_blinding_codelist_validation(
        self, client: CdiscLibraryClient, mock_blinding_codelist: dict,
    ) -> None:
        """Blinding values should validate against C66735."""
        _mock_api_response(client, mock_blinding_codelist)
        result = client.validate_ct_value("BLIND", "DOUBLE BLIND")
        assert result.valid is True
        assert result.matched_term is not None
        assert result.matched_term.code == "C15228"


# ---------------------------------------------------------------------------
# Scenario 2: Unknown CT values flagged with suggestions
# ---------------------------------------------------------------------------

class TestUnknownCtSuggestions:
    """Scenario: Unknown CT values flagged with suggestions."""

    def test_unknown_value_flagged_invalid(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Unknown value should be flagged as invalid."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "Phase X")
        assert result.valid is False
        assert result.matched_term is None

    def test_suggestions_provided_for_unknown(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Unknown values should get closest matching suggestions."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "Phase III Study")
        assert result.valid is False
        assert len(result.suggestions) > 0

    def test_suggestions_limited_to_three(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Suggestions should be limited to 3 max."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "Something")
        assert len(result.suggestions) <= 3

    def test_closest_suggestion_ranked_first(
        self, client: CdiscLibraryClient, mock_phase_codelist: dict,
    ) -> None:
        """Most similar term should be the first suggestion."""
        _mock_api_response(client, mock_phase_codelist)
        result = client.validate_ct_value("PHASE", "Phase III Study")
        assert len(result.suggestions) > 0
        # "PHASE III TRIAL" should be most similar to "Phase III Study"
        assert "III" in result.suggestions[0].submission_value

    def test_unknown_parmcd_returns_invalid(
        self, client: CdiscLibraryClient,
    ) -> None:
        """Unknown TSPARMCD should return invalid with no suggestions."""
        result = client.validate_ct_value("UNKNOWN_PARM", "some value")
        assert result.valid is False
        assert result.source == "fallback"
        assert len(result.suggestions) == 0

    def test_normalizer_returns_unmapped_for_unknown(
        self, normalizer: CdiscLibraryNormalizer, mock_phase_codelist: dict,
    ) -> None:
        """Enhanced normalizer should return unmapped for unknown values."""
        _mock_api_response(normalizer._client, mock_phase_codelist)
        result = normalizer.normalize("PHASE", "Phase X")
        assert result.mapped is False
        assert result.tsvalcd == ""

    def test_validate_with_suggestions_returns_details(
        self, normalizer: CdiscLibraryNormalizer, mock_phase_codelist: dict,
    ) -> None:
        """validate_with_suggestions should return full CtValidationResult."""
        _mock_api_response(normalizer._client, mock_phase_codelist)
        result = normalizer.validate_with_suggestions("PHASE", "Phase III Study")
        assert isinstance(result, CtValidationResult)
        assert len(result.suggestions) > 0


# ---------------------------------------------------------------------------
# Scenario 3: Biomedical concepts used for domain mapping
# ---------------------------------------------------------------------------

class TestBiomedicalConceptLookup:
    """Scenario: Biomedical concepts used for domain mapping."""

    def test_cbc_maps_to_lb(self, client: CdiscLibraryClient) -> None:
        """Complete Blood Count should map to LB domain."""
        result = client.lookup_domain("Complete Blood Count")
        assert result.domain_code == "LB"
        assert result.domain_name == "Laboratory Test Results"

    def test_cbc_suggests_testcd(self, client: CdiscLibraryClient) -> None:
        """CBC should suggest LBTESTCD value."""
        result = client.lookup_domain("Complete Blood Count")
        assert result.testcd_suggestion == "CBC"

    def test_vital_signs_maps_to_vs(self, client: CdiscLibraryClient) -> None:
        """Vital Signs should map to VS domain."""
        result = client.lookup_domain("Vital Signs")
        assert result.domain_code == "VS"

    def test_ecg_maps_to_eg(self, client: CdiscLibraryClient) -> None:
        """ECG should map to EG domain."""
        result = client.lookup_domain("12-lead ECG")
        assert result.domain_code == "EG"

    def test_blood_pressure_suggests_sysbp(
        self, client: CdiscLibraryClient,
    ) -> None:
        """Blood Pressure should suggest SYSBP testcd."""
        result = client.lookup_domain("Blood Pressure")
        assert result.domain_code == "VS"
        assert result.testcd_suggestion == "SYSBP"

    def test_adverse_event_maps_to_ae(
        self, client: CdiscLibraryClient,
    ) -> None:
        """Adverse Event should map to AE domain."""
        result = client.lookup_domain("Adverse Event Monitoring")
        assert result.domain_code == "AE"

    def test_unknown_assessment_falls_back(
        self, client: CdiscLibraryClient,
    ) -> None:
        """Unrecognised assessment should fall back to FA domain."""
        result = client.lookup_domain("Completely Unknown Assessment XYZ")
        assert result.domain_code == "FA"
        assert result.source == "fallback"

    def test_normalizer_map_assessment(
        self, normalizer: CdiscLibraryNormalizer,
    ) -> None:
        """Enhanced normalizer's map_assessment should work."""
        result = normalizer.map_assessment("Chemistry Panel")
        assert result.domain_code == "LB"
        assert result.testcd_suggestion == "CHEM"

    def test_hba1c_maps_to_lb(self, client: CdiscLibraryClient) -> None:
        """HbA1c should map to LB with specific testcd."""
        result = client.lookup_domain("HbA1c Test")
        assert result.domain_code == "LB"
        assert result.testcd_suggestion == "HBA1C"

    def test_pharmacokinetic_maps_to_pc(
        self, client: CdiscLibraryClient,
    ) -> None:
        """Pharmacokinetic sampling should map to PC domain."""
        result = client.lookup_domain("Pharmacokinetic Sampling")
        assert result.domain_code == "PC"

    def test_domain_mapping_confidence(
        self, client: CdiscLibraryClient,
    ) -> None:
        """Local mappings should have confidence 1.0."""
        result = client.lookup_domain("Complete Blood Count")
        assert result.confidence == 1.0
        assert result.source == "local"


# ---------------------------------------------------------------------------
# Scenario 4: Offline fallback to local mappings
# ---------------------------------------------------------------------------

class TestOfflineFallback:
    """Scenario: Offline fallback to local mappings."""

    def test_no_api_key_uses_fallback(self, cache_db: Path) -> None:
        """Without API key, normalizer falls back to CtNormalizer."""
        normalizer = CdiscLibraryNormalizer(
            api_key="", cache_db=cache_db,
        )
        result = normalizer.normalize("PHASE", "Phase III")
        # Local CtNormalizer should handle this
        assert result.mapped is True
        assert result.nci_code == "C15962"

    def test_fallback_logs_warning(
        self, cache_db: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Fallback should log a warning about using local data."""
        import logging
        normalizer = CdiscLibraryNormalizer(
            api_key="", cache_db=cache_db,
        )
        with caplog.at_level(logging.WARNING, logger="ptcv.sdtm.cdisc_library"):
            normalizer.normalize("PHASE", "Phase III")
        assert any("fallback" in r.message.lower() for r in caplog.records)

    def test_api_not_configured_property(self, cache_db: Path) -> None:
        """api_available should be False without API key."""
        normalizer = CdiscLibraryNormalizer(
            api_key="", cache_db=cache_db,
        )
        assert normalizer.api_available is False

    def test_api_configured_property(
        self, normalizer: CdiscLibraryNormalizer,
    ) -> None:
        """api_available should be True with API key."""
        assert normalizer.api_available is True

    def test_fallback_phase_iii_still_maps(self, cache_db: Path) -> None:
        """Phase III should map via local fallback."""
        normalizer = CdiscLibraryNormalizer(
            api_key="", cache_db=cache_db,
        )
        result = normalizer.normalize("PHASE", "Phase III")
        assert result.mapped is True
        assert result.tsvalcd == "PHASE3"

    def test_fallback_blinding_maps(self, cache_db: Path) -> None:
        """Double-blind should map via local fallback."""
        normalizer = CdiscLibraryNormalizer(
            api_key="", cache_db=cache_db,
        )
        result = normalizer.normalize("BLIND", "Double-Blind")
        assert result.mapped is True
        assert result.nci_code == "C15228"

    def test_fallback_design_maps(self, cache_db: Path) -> None:
        """Parallel should map via local fallback."""
        normalizer = CdiscLibraryNormalizer(
            api_key="", cache_db=cache_db,
        )
        result = normalizer.normalize("STYPE", "Parallel")
        assert result.mapped is True

    def test_client_configured_with_key(
        self, client: CdiscLibraryClient,
    ) -> None:
        """Client should report configured when key present."""
        assert client.configured is True

    def test_client_not_configured_without_key(
        self, cache_db: Path,
    ) -> None:
        """Client should report not configured without key."""
        with patch.dict("os.environ", {}, clear=True):
            c = CdiscLibraryClient(api_key="", cache_db=cache_db)
            assert c.configured is False


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestCache:
    """SQLite cache behavior."""

    def test_cache_initialise_creates_db(self, cache_db: Path) -> None:
        """Cache initialization should create the database file."""
        cache = _CdiscLibraryCache(db_path=cache_db)
        cache.initialise()
        assert cache_db.exists()

    def test_cache_put_and_get(self, cache_db: Path) -> None:
        """Stored values should be retrievable."""
        cache = _CdiscLibraryCache(db_path=cache_db)
        cache.initialise()
        cache.put("test-key", {"data": "value"})
        result = cache.get("test-key")
        assert result == {"data": "value"}

    def test_cache_miss_returns_none(self, cache_db: Path) -> None:
        """Missing keys should return None."""
        cache = _CdiscLibraryCache(db_path=cache_db)
        cache.initialise()
        assert cache.get("nonexistent") is None

    def test_cache_expired_returns_none(self, cache_db: Path) -> None:
        """Expired entries should return None."""
        cache = _CdiscLibraryCache(db_path=cache_db, ttl_hours=0)
        cache.initialise()
        cache.put("expired-key", {"data": "old"})
        # TTL=0 means entries expire immediately
        import time
        time.sleep(0.01)
        assert cache.get("expired-key") is None

    def test_cache_overwrite_existing(self, cache_db: Path) -> None:
        """Overwriting should update the cached value."""
        cache = _CdiscLibraryCache(db_path=cache_db)
        cache.initialise()
        cache.put("key", {"version": 1})
        cache.put("key", {"version": 2})
        result = cache.get("key")
        assert result == {"version": 2}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and integration."""

    def test_codelist_term_dataclass(self) -> None:
        """CodelistTerm should be a valid dataclass."""
        term = CodelistTerm(
            code="C15962",
            submission_value="PHASE III TRIAL",
            preferred_term="Phase III Trial",
            codelist_code="C66737",
        )
        assert term.code == "C15962"
        assert term.synonyms == ""

    def test_empty_codelist_returns_empty_list(
        self, client: CdiscLibraryClient,
    ) -> None:
        """Empty API response should return empty term list."""
        client._cache.put(
            "https://library.cdisc.org/api/mdr/ct/packages/sdtmct-2024-03-29"
            "/codelists/C99999",
            {"terms": []},
        )
        terms = client.get_codelist_terms("C99999")
        assert terms == []

    def test_list_packages_empty_when_offline(
        self, cache_db: Path,
    ) -> None:
        """list_ct_packages should return empty list when offline."""
        client = CdiscLibraryClient(api_key="", cache_db=cache_db)
        assert client.list_ct_packages() == []

    def test_normalizer_api_match_overrides_fallback(
        self, normalizer: CdiscLibraryNormalizer, mock_phase_codelist: dict,
    ) -> None:
        """When API has a match, it should take precedence."""
        _mock_api_response(normalizer._client, mock_phase_codelist)
        result = normalizer.normalize("PHASE", "Phase III Trial")
        assert result.mapped is True
        assert result.nci_code == "C15962"
        assert result.tsval == "PHASE III TRIAL"

    def test_normalizer_falls_back_when_api_no_match(
        self, normalizer: CdiscLibraryNormalizer, mock_phase_codelist: dict,
    ) -> None:
        """When API has no match, fallback should be tried."""
        _mock_api_response(normalizer._client, mock_phase_codelist)
        # "Phase III" won't match "PHASE III TRIAL" exactly,
        # but local fallback has "PHASE III" → C15962
        result = normalizer.normalize("PHASE", "Phase III")
        assert result.mapped is True
        assert result.nci_code == "C15962"

    def test_domain_mapping_dataclass(self) -> None:
        """DomainMapping should be a valid dataclass."""
        mapping = DomainMapping(
            assessment_name="CBC",
            domain_code="LB",
            domain_name="Laboratory Test Results",
            testcd_suggestion="CBC",
            source="local",
        )
        assert mapping.confidence == 1.0

    def test_validation_result_dataclass(self) -> None:
        """CtValidationResult should be a valid dataclass."""
        result = CtValidationResult(
            original_value="Phase X",
            valid=False,
            matched_term=None,
            suggestions=[],
            source="fallback",
        )
        assert result.codelist_name == ""
        assert result.package_version == ""
