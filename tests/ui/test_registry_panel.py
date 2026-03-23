"""Unit tests for registry_panel and catalog enrichment (PTCV-207).

Tests the CT.gov metadata parser, catalog enrichment, sidebar
search, and graceful fallback when registry cache is missing.

Qualification: Installation Qualification (IQ)
Regulatory: N/A — UI display logic, no regulated data mutation
Risk Tier: LOW — metadata display only
Module: ptcv.ui.components.registry_panel
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ptcv.ui.components.protocol_catalog import (
    ProtocolEntry,
    QualityScores,
    TherapeuticArea,
    load_protocol_catalog,
)
from ptcv.ui.components.registry_panel import (
    RegistryMetadata,
    load_all_registry_metadata,
    load_registry_metadata,
    parse_registry_json,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _sample_ctgov_json(
    nct_id: str = "NCT00004088",
    official_title: str = "A Phase II Trial of Drug X",
    conditions: list[str] | None = None,
    phase: str = "PHASE2",
    status: str = "COMPLETED",
    sponsor: str = "City of Hope",
    enrollment: int = 77,
) -> dict[str, Any]:
    """Build a minimal CT.gov API response JSON."""
    if conditions is None:
        conditions = ["Multiple Myeloma"]
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "officialTitle": official_title,
                "briefTitle": f"Trial {nct_id}",
            },
            "statusModule": {
                "overallStatus": status,
                "startDateStruct": {"date": "2020-01-15"},
                "primaryCompletionDateStruct": {
                    "date": "2023-06-30",
                    "type": "ACTUAL",
                },
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": [phase],
                "designInfo": {
                    "allocation": "RANDOMIZED",
                    "interventionModel": "PARALLEL",
                    "maskingInfo": {"masking": "DOUBLE"},
                },
                "enrollmentInfo": {
                    "count": enrollment,
                    "type": "ACTUAL",
                },
            },
            "conditionsModule": {"conditions": conditions},
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "Overall Survival"},
                    {"measure": "Progression-Free Survival"},
                ],
                "secondaryOutcomes": [
                    {"measure": "Quality of Life"},
                ],
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {
                    "name": sponsor,
                    "class": "OTHER",
                },
            },
        },
    }


@pytest.fixture
def registry_cache_dir(tmp_path: Path) -> Path:
    """Create a temp registry cache with sample data."""
    cache = tmp_path / "registry_cache"
    cache.mkdir()
    for nct_id, condition in [
        ("NCT00004088", ["Multiple Myeloma"]),
        ("NCT00074490", ["Breast Cancer"]),
        ("NCT01512251", ["Melanoma"]),
    ]:
        data = _sample_ctgov_json(
            nct_id=nct_id,
            conditions=condition,
        )
        (cache / f"{nct_id}.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
    return cache


# ------------------------------------------------------------------
# parse_registry_json
# ------------------------------------------------------------------

class TestParseRegistryJson:
    """Tests for parse_registry_json()."""

    def test_parses_all_fields(self) -> None:
        raw = _sample_ctgov_json()
        meta = parse_registry_json(raw)

        assert meta.nct_id == "NCT00004088"
        assert meta.official_title == "A Phase II Trial of Drug X"
        assert meta.brief_title == "Trial NCT00004088"
        assert meta.sponsor == "City of Hope"
        assert meta.phase == "PHASE2"
        assert meta.status == "COMPLETED"
        assert meta.conditions == ["Multiple Myeloma"]
        assert meta.enrollment == 77
        assert meta.enrollment_type == "ACTUAL"
        assert meta.study_type == "INTERVENTIONAL"
        assert meta.start_date == "2020-01-15"
        assert meta.completion_date == "2023-06-30"
        assert meta.allocation == "RANDOMIZED"
        assert meta.masking == "DOUBLE"
        assert meta.intervention_model == "PARALLEL"

    def test_primary_outcomes(self) -> None:
        meta = parse_registry_json(_sample_ctgov_json())
        assert len(meta.primary_outcomes) == 2
        assert "Overall Survival" in meta.primary_outcomes

    def test_secondary_outcomes(self) -> None:
        meta = parse_registry_json(_sample_ctgov_json())
        assert len(meta.secondary_outcomes) == 1
        assert "Quality of Life" in meta.secondary_outcomes

    def test_display_title_prefers_official(self) -> None:
        meta = parse_registry_json(_sample_ctgov_json())
        assert meta.display_title == "A Phase II Trial of Drug X"

    def test_display_title_falls_back_to_brief(self) -> None:
        raw = _sample_ctgov_json()
        raw["protocolSection"]["identificationModule"][
            "officialTitle"
        ] = ""
        meta = parse_registry_json(raw)
        assert meta.display_title == "Trial NCT00004088"

    def test_phase_display(self) -> None:
        meta = parse_registry_json(_sample_ctgov_json())
        assert meta.phase_display == "Phase 2"

    def test_phase_display_multi(self) -> None:
        raw = _sample_ctgov_json()
        raw["protocolSection"]["designModule"]["phases"] = [
            "PHASE1",
            "PHASE2",
        ]
        meta = parse_registry_json(raw)
        assert meta.phase_display == "Phase 1 / Phase 2"

    def test_ctgov_url(self) -> None:
        meta = parse_registry_json(_sample_ctgov_json())
        assert meta.ctgov_url == (
            "https://clinicaltrials.gov/study/NCT00004088"
        )

    def test_empty_json(self) -> None:
        meta = parse_registry_json({})
        assert meta.nct_id == ""
        assert meta.conditions == []
        assert meta.primary_outcomes == []

    def test_missing_modules(self) -> None:
        """Handles missing modules gracefully."""
        raw = {"protocolSection": {"identificationModule": {
            "nctId": "NCT99999999",
        }}}
        meta = parse_registry_json(raw)
        assert meta.nct_id == "NCT99999999"
        assert meta.sponsor == ""
        assert meta.enrollment == 0


# ------------------------------------------------------------------
# load_registry_metadata
# ------------------------------------------------------------------

class TestLoadRegistryMetadata:
    """Tests for load_registry_metadata()."""

    def test_loads_cached_file(
        self, registry_cache_dir: Path,
    ) -> None:
        meta = load_registry_metadata(
            registry_cache_dir, "NCT00004088",
        )
        assert meta is not None
        assert meta.nct_id == "NCT00004088"

    def test_returns_none_for_missing(
        self, registry_cache_dir: Path,
    ) -> None:
        meta = load_registry_metadata(
            registry_cache_dir, "NCT99999999",
        )
        assert meta is None

    def test_returns_none_for_missing_dir(
        self, tmp_path: Path,
    ) -> None:
        meta = load_registry_metadata(
            tmp_path / "nonexistent", "NCT00004088",
        )
        assert meta is None

    def test_returns_none_for_invalid_json(
        self, tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "NCT00001111.json").write_text(
            "not valid json", encoding="utf-8",
        )
        meta = load_registry_metadata(cache, "NCT00001111")
        assert meta is None


# ------------------------------------------------------------------
# load_all_registry_metadata
# ------------------------------------------------------------------

class TestLoadAllRegistryMetadata:
    """Tests for load_all_registry_metadata()."""

    def test_loads_all_files(
        self, registry_cache_dir: Path,
    ) -> None:
        all_meta = load_all_registry_metadata(
            registry_cache_dir,
        )
        assert len(all_meta) == 3
        assert "NCT00004088" in all_meta
        assert "NCT00074490" in all_meta

    def test_empty_dir(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        cache.mkdir()
        assert load_all_registry_metadata(cache) == {}

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert load_all_registry_metadata(
            tmp_path / "nope"
        ) == {}


# ------------------------------------------------------------------
# Catalog enrichment (PTCV-207)
# ------------------------------------------------------------------

def _write_metadata(
    metadata_dir: Path,
    registry_id: str,
    condition: str,
    title: str = "Test Protocol",
) -> None:
    """Write a minimal metadata JSON file."""
    meta: dict[str, Any] = {
        "source": "ClinicalTrials.gov",
        "registry_id": registry_id,
        "version": "1.0",
        "amendment_number": "1.0",
        "title": title,
        "condition": condition,
    }
    path = metadata_dir / f"{registry_id}_1.0.json"
    path.write_text(json.dumps(meta), encoding="utf-8")


def _write_pdf(ct_dir: Path, registry_id: str) -> None:
    """Write a stub PDF file."""
    (ct_dir / f"{registry_id}_1.0.pdf").write_bytes(
        b"%PDF-fake"
    )


def _write_cache(
    cache_dir: Path,
    nct_id: str,
    title: str = "CT.gov Official Title",
    conditions: list[str] | None = None,
    sponsor: str = "Acme Pharma",
) -> None:
    """Write a registry cache JSON file."""
    data = _sample_ctgov_json(
        nct_id=nct_id,
        official_title=title,
        conditions=conditions or ["Breast Cancer"],
        sponsor=sponsor,
    )
    (cache_dir / f"{nct_id}.json").write_text(
        json.dumps(data), encoding="utf-8",
    )


class TestCatalogEnrichment:
    """Tests for registry cache enrichment in catalog loading."""

    def test_title_enriched_from_registry(
        self, tmp_path: Path,
    ) -> None:
        """CT.gov official title overrides metadata title."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        cache_dir = ct_dir / "registry_cache"
        cache_dir.mkdir()

        _write_metadata(
            meta_dir, "NCT001", "Cancer",
            title="Old Title",
        )
        _write_pdf(ct_dir, "NCT001")
        _write_cache(
            cache_dir, "NCT001",
            title="New Official Title from CT.gov",
        )

        catalog = load_protocol_catalog(tmp_path)
        entry = list(catalog.values())[0][0]
        assert "New Official Title" in entry.title

    def test_condition_enriched_from_registry(
        self, tmp_path: Path,
    ) -> None:
        """CT.gov conditions override metadata condition."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        cache_dir = ct_dir / "registry_cache"
        cache_dir.mkdir()

        _write_metadata(
            meta_dir, "NCT001", "Unknown Condition",
        )
        _write_pdf(ct_dir, "NCT001")
        _write_cache(
            cache_dir, "NCT001",
            conditions=["Heart Failure"],
        )

        catalog = load_protocol_catalog(tmp_path)
        # Should be reclassified to Cardiovascular
        assert TherapeuticArea.CARDIOVASCULAR in catalog

    def test_sponsor_and_phase_populated(
        self, tmp_path: Path,
    ) -> None:
        """Sponsor and phase are set from registry cache."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        cache_dir = ct_dir / "registry_cache"
        cache_dir.mkdir()

        _write_metadata(meta_dir, "NCT001", "Cancer")
        _write_pdf(ct_dir, "NCT001")
        _write_cache(
            cache_dir, "NCT001", sponsor="Pfizer",
        )

        catalog = load_protocol_catalog(tmp_path)
        entry = list(catalog.values())[0][0]
        assert entry.sponsor == "Pfizer"
        assert entry.phase  # Should have phase from cache

    def test_fallback_without_registry_cache(
        self, tmp_path: Path,
    ) -> None:
        """Catalog works normally without registry cache dir."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        # No registry_cache directory

        _write_metadata(
            meta_dir, "NCT001", "Breast Cancer",
            title="Original Title",
        )
        _write_pdf(ct_dir, "NCT001")

        catalog = load_protocol_catalog(tmp_path)
        entry = catalog[TherapeuticArea.ONCOLOGY][0]
        assert entry.title == "Original Title"
        assert entry.sponsor == ""

    def test_partial_enrichment(
        self, tmp_path: Path,
    ) -> None:
        """Only entries with cache are enriched; others untouched."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        cache_dir = ct_dir / "registry_cache"
        cache_dir.mkdir()

        _write_metadata(
            meta_dir, "NCT001", "Breast Cancer",
            title="Has Cache",
        )
        _write_pdf(ct_dir, "NCT001")
        _write_cache(cache_dir, "NCT001", title="Enriched")

        _write_metadata(
            meta_dir, "NCT002", "Colon Cancer",
            title="No Cache",
        )
        _write_pdf(ct_dir, "NCT002")
        # No cache for NCT002

        catalog = load_protocol_catalog(tmp_path)
        entries = catalog[TherapeuticArea.ONCOLOGY]
        titles = {e.registry_id: e.title for e in entries}
        assert "Enriched" in titles["NCT001"]
        assert titles["NCT002"] == "No Cache"


# ------------------------------------------------------------------
# Sidebar search (PTCV-207)
# ------------------------------------------------------------------

class TestSidebarSearch:
    """Tests for _matches_search in file_browser."""

    def test_matches_title(self) -> None:
        from ptcv.ui.components.file_browser import (
            _matches_search,
        )

        entry = ProtocolEntry(
            registry_id="NCT001",
            title="Phase II Study of Drug X",
            condition="Cancer",
            therapeutic_area=TherapeuticArea.ONCOLOGY,
            registry_source="clinicaltrials",
            filename="NCT001_1.0.pdf",
            file_path=Path("/fake/NCT001_1.0.pdf"),
            sponsor="Pfizer",
        )
        assert _matches_search(entry, "drug x")
        assert _matches_search(entry, "phase ii")

    def test_matches_nct_id(self) -> None:
        from ptcv.ui.components.file_browser import (
            _matches_search,
        )

        entry = ProtocolEntry(
            registry_id="NCT00004088",
            title="Title",
            condition="Cancer",
            therapeutic_area=TherapeuticArea.ONCOLOGY,
            registry_source="clinicaltrials",
            filename="NCT00004088_1.0.pdf",
            file_path=Path("/fake/NCT00004088_1.0.pdf"),
        )
        assert _matches_search(entry, "nct00004088")

    def test_matches_condition(self) -> None:
        from ptcv.ui.components.file_browser import (
            _matches_search,
        )

        entry = ProtocolEntry(
            registry_id="NCT001",
            title="Title",
            condition="Multiple Myeloma",
            therapeutic_area=TherapeuticArea.ONCOLOGY,
            registry_source="clinicaltrials",
            filename="NCT001_1.0.pdf",
            file_path=Path("/fake/NCT001_1.0.pdf"),
        )
        assert _matches_search(entry, "myeloma")

    def test_matches_sponsor(self) -> None:
        from ptcv.ui.components.file_browser import (
            _matches_search,
        )

        entry = ProtocolEntry(
            registry_id="NCT001",
            title="Title",
            condition="Cancer",
            therapeutic_area=TherapeuticArea.ONCOLOGY,
            registry_source="clinicaltrials",
            filename="NCT001_1.0.pdf",
            file_path=Path("/fake/NCT001_1.0.pdf"),
            sponsor="City of Hope",
        )
        assert _matches_search(entry, "city of hope")

    def test_no_match(self) -> None:
        from ptcv.ui.components.file_browser import (
            _matches_search,
        )

        entry = ProtocolEntry(
            registry_id="NCT001",
            title="Title",
            condition="Cancer",
            therapeutic_area=TherapeuticArea.ONCOLOGY,
            registry_source="clinicaltrials",
            filename="NCT001_1.0.pdf",
            file_path=Path("/fake/NCT001_1.0.pdf"),
        )
        assert not _matches_search(entry, "diabetes")


# ------------------------------------------------------------------
# RegistryMetadata dataclass
# ------------------------------------------------------------------

class TestRegistryMetadataDefaults:
    """Tests for RegistryMetadata default values."""

    def test_defaults(self) -> None:
        meta = RegistryMetadata()
        assert meta.nct_id == ""
        assert meta.conditions == []
        assert meta.primary_outcomes == []
        assert meta.enrollment == 0

    def test_phase_display_empty(self) -> None:
        meta = RegistryMetadata()
        assert meta.phase_display == "N/A"
