"""Tests for registry graceful degradation (PTCV-199).

Covers all GHERKIN scenarios:
- EU-CTR protocol with no NCT ID
- ClinicalTrials.gov API is unreachable
- Partial metadata (missing modules)
- Pipeline metrics track registry coverage

Qualification phase: OQ (operational qualification)
Risk tier: LOW
"""

import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from ptcv.registry.cross_validator import CrossValidator, SectionStatus
from ptcv.registry.fallback import (
    RegistryEnrichmentResult,
    RegistryPipelineMetrics,
    collect_batch_metrics,
    detect_nct_id,
    try_registry_enrichment,
)
from ptcv.registry.ich_mapper import (
    MappedRegistrySection,
    MetadataToIchMapper,
)
from ptcv.registry.metadata_fetcher import RegistryMetadataFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_metadata() -> dict[str, Any]:
    """Minimal CT.gov metadata with all modules present."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT01512251",
                "officialTitle": "BKM120 + Vemurafenib Trial",
                "briefTitle": "BKM120 Phase 1/2",
                "organization": {
                    "fullName": "UCSF",
                },
            },
            "statusModule": {
                "overallStatus": "Completed",
                "startDateStruct": {"date": "2012-02"},
            },
            "conditionsModule": {
                "conditions": ["Melanoma"],
            },
            "descriptionModule": {
                "briefSummary": "Phase 1/2 trial.",
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE1"],
                "enrollmentInfo": {"count": 30, "type": "ACTUAL"},
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"name": "BKM120", "type": "DRUG"},
                ],
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "MTD", "timeFrame": "28 days"},
                ],
            },
            "eligibilityModule": {
                "eligibilityCriteria": "BRAF V600E/K",
                "sex": "ALL",
                "minimumAge": "18 Years",
            },
            "referencesModule": {
                "references": [
                    {"pmid": "12345", "citation": "Smith 2015"},
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Scenario 1: EU-CTR protocol with no NCT ID
# ---------------------------------------------------------------------------


class TestEuCtrNoNctId:
    """Scenario: EU-CTR protocol with no NCT ID."""

    def test_eudract_id_not_detected_as_nct(self) -> None:
        """EudraCT number is not mistaken for NCT ID."""
        result = detect_nct_id(
            registry_id="EUCTR2020-001234-56"
        )
        assert result is None

    def test_enrichment_skipped_for_eu_ctr(self) -> None:
        """EU-CTR protocol skips registry enrichment without error."""
        result = try_registry_enrichment(
            registry_id="EUCTR2020-001234-56",
            filename="protocol_eu_ctr.pdf",
        )
        assert result.registry_available is False
        assert result.skip_reason != ""
        assert "NCT ID" in result.skip_reason

    def test_no_api_call_for_eu_ctr(self) -> None:
        """No API call is made when NCT ID is absent."""
        mock_fetcher = MagicMock(spec=RegistryMetadataFetcher)
        try_registry_enrichment(
            registry_id="EUCTR2020-001234-56",
            fetcher=mock_fetcher,
        )
        mock_fetcher.fetch.assert_not_called()

    def test_empty_identifiers_return_no_nct(self) -> None:
        """All-empty inputs return None."""
        assert detect_nct_id() is None

    def test_nct_id_detected_from_registry_id(self) -> None:
        """Valid NCT ID in registry_id is detected."""
        assert detect_nct_id(
            registry_id="NCT01512251"
        ) == "NCT01512251"

    def test_nct_id_detected_from_filename(self) -> None:
        """NCT ID embedded in filename is detected."""
        assert detect_nct_id(
            filename="NCT01512251_1.0.pdf"
        ) == "NCT01512251"

    def test_nct_id_detected_from_metadata(self) -> None:
        """NCT ID in metadata dict is detected."""
        assert detect_nct_id(
            metadata={"nctId": "NCT01512251"}
        ) == "NCT01512251"


# ---------------------------------------------------------------------------
# Scenario 2: ClinicalTrials.gov API is unreachable
# ---------------------------------------------------------------------------


class TestApiUnreachable:
    """Scenario: ClinicalTrials.gov API is unreachable."""

    def test_fetch_returns_none_on_timeout(self) -> None:
        """API timeout → enrichment returns registry_available=False."""
        mock_fetcher = MagicMock(spec=RegistryMetadataFetcher)
        mock_fetcher.fetch.return_value = None

        result = try_registry_enrichment(
            registry_id="NCT01512251",
            fetcher=mock_fetcher,
        )

        assert result.registry_available is False
        assert result.registry_id == "NCT01512251"
        assert "no data" in result.skip_reason.lower()

    def test_fetch_exception_caught(self) -> None:
        """Unexpected exception in fetch → graceful degradation."""
        mock_fetcher = MagicMock(spec=RegistryMetadataFetcher)
        mock_fetcher.fetch.side_effect = ConnectionError(
            "DNS resolution failed"
        )

        result = try_registry_enrichment(
            registry_id="NCT01512251",
            fetcher=mock_fetcher,
        )

        assert result.registry_available is False
        assert "failed" in result.skip_reason.lower()

    def test_no_error_raised_on_api_failure(self) -> None:
        """API failure never raises — always returns a result."""
        mock_fetcher = MagicMock(spec=RegistryMetadataFetcher)
        mock_fetcher.fetch.side_effect = OSError("Connection refused")

        # Should not raise
        result = try_registry_enrichment(
            registry_id="NCT01512251",
            fetcher=mock_fetcher,
        )
        assert isinstance(result, RegistryEnrichmentResult)

    def test_downstream_continues_without_registry(self) -> None:
        """Cross-validator works with empty registry sections."""
        from ptcv.ich_parser.models import IchSection

        pdf_sections = [
            IchSection(
                run_id="r1",
                source_run_id="s1",
                source_sha256="h1",
                registry_id="NCT01512251",
                section_code="B.4",
                section_name="Trial Design",
                content_json="{}",
                confidence_score=0.8,
                review_required=False,
                legacy_format=False,
                content_text="Open-label study",
            ),
        ]
        validator = CrossValidator()
        report = validator.validate(
            "NCT01512251", pdf_sections, []
        )

        assert report.pdf_only_count == 1
        assert report.missing_from_pdf_count == 0


# ---------------------------------------------------------------------------
# Scenario 3: Partial metadata (missing modules)
# ---------------------------------------------------------------------------


class TestPartialMetadata:
    """Scenario: Partial metadata (missing modules)."""

    def test_missing_arms_omits_interventions_but_maps_rest(
        self,
    ) -> None:
        """Null ArmsInterventionsModule → B.4 still maps (without
        interventions), other sections unaffected."""
        metadata = _full_metadata()
        del metadata["protocolSection"]["armsInterventionsModule"]

        mapper = MetadataToIchMapper()
        sections = mapper.map(metadata)
        codes = {s.section_code for s in sections}

        # B.4 should still exist (from DesignModule fields)
        assert "B.4" in codes
        # Other sections unaffected
        assert "B.1" in codes
        assert "B.3" in codes
        assert "B.5" in codes

    def test_missing_outcomes_omits_b8(self) -> None:
        """Null OutcomesModule → B.8 omitted, rest preserved."""
        metadata = _full_metadata()
        metadata["protocolSection"]["outcomesModule"] = {}

        mapper = MetadataToIchMapper()
        sections = mapper.map(metadata)
        codes = {s.section_code for s in sections}

        assert "B.8" not in codes
        assert "B.1" in codes
        assert "B.5" in codes

    def test_enrichment_succeeds_with_partial_data(self) -> None:
        """Partial metadata still produces enrichment result."""
        metadata = _full_metadata()
        del metadata["protocolSection"]["outcomesModule"]
        del metadata["protocolSection"]["referencesModule"]

        mock_fetcher = MagicMock(spec=RegistryMetadataFetcher)
        mock_fetcher.fetch.return_value = metadata

        result = try_registry_enrichment(
            registry_id="NCT01512251",
            fetcher=mock_fetcher,
        )

        assert result.registry_available is True
        assert result.registry_sections_mapped > 0
        # Should have fewer sections than full metadata
        mapped_codes = {
            s.section_code for s in result.mapped_sections
        }
        assert "B.8" not in mapped_codes
        assert "B.16" not in mapped_codes

    def test_mapper_exception_caught_gracefully(self) -> None:
        """Exception during mapping → graceful degradation."""
        mock_fetcher = MagicMock(spec=RegistryMetadataFetcher)
        mock_fetcher.fetch.return_value = _full_metadata()

        mock_mapper = MagicMock(spec=MetadataToIchMapper)
        mock_mapper.map.side_effect = ValueError("Bad data")

        result = try_registry_enrichment(
            registry_id="NCT01512251",
            fetcher=mock_fetcher,
            mapper=mock_mapper,
        )

        assert result.registry_available is False
        assert "mapping failed" in result.skip_reason.lower()


# ---------------------------------------------------------------------------
# Scenario 4: Pipeline metrics track registry coverage
# ---------------------------------------------------------------------------


class TestPipelineMetrics:
    """Scenario: Pipeline metrics track registry coverage."""

    def test_coverage_7_of_10(self) -> None:
        """7 enriched + 3 skipped → coverage=0.7."""
        results = []
        for i in range(7):
            results.append(RegistryEnrichmentResult(
                registry_id=f"NCT{i:08d}",
                registry_available=True,
                registry_sections_mapped=5,
            ))
        for i in range(3):
            results.append(RegistryEnrichmentResult(
                skip_reason="No NCT ID",
            ))

        metrics = collect_batch_metrics(results)

        assert metrics.total_protocols == 10
        assert metrics.registry_enriched == 7
        assert metrics.registry_skipped == 3
        assert metrics.registry_coverage == pytest.approx(0.7)

    def test_all_enriched(self) -> None:
        """All protocols enriched → coverage=1.0."""
        results = [
            RegistryEnrichmentResult(
                registry_id="NCT00000001",
                registry_available=True,
            )
            for _ in range(5)
        ]
        metrics = collect_batch_metrics(results)
        assert metrics.registry_coverage == pytest.approx(1.0)

    def test_none_enriched(self) -> None:
        """No protocols enriched → coverage=0.0."""
        results = [
            RegistryEnrichmentResult(skip_reason="No NCT ID")
            for _ in range(5)
        ]
        metrics = collect_batch_metrics(results)
        assert metrics.registry_coverage == pytest.approx(0.0)

    def test_empty_batch(self) -> None:
        """Empty batch → coverage=0.0, no division error."""
        metrics = collect_batch_metrics([])
        assert metrics.registry_coverage == 0.0
        assert metrics.total_protocols == 0

    def test_per_protocol_logs_available(self) -> None:
        """Per-protocol results are preserved in metrics."""
        results = [
            RegistryEnrichmentResult(
                registry_id="NCT00000001",
                registry_available=True,
                registry_sections_mapped=5,
            ),
            RegistryEnrichmentResult(
                skip_reason="API timeout",
            ),
        ]
        metrics = collect_batch_metrics(results)

        assert len(metrics.per_protocol) == 2
        assert metrics.per_protocol[0].registry_available is True
        assert metrics.per_protocol[1].registry_available is False

    def test_update_coverage_recalculates(self) -> None:
        """update_coverage() recalculates from per_protocol."""
        metrics = RegistryPipelineMetrics()
        metrics.per_protocol = [
            RegistryEnrichmentResult(registry_available=True),
            RegistryEnrichmentResult(registry_available=False),
            RegistryEnrichmentResult(registry_available=True),
        ]
        metrics.update_coverage()

        assert metrics.total_protocols == 3
        assert metrics.registry_enriched == 2
        assert metrics.registry_coverage == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# detect_nct_id edge cases
# ---------------------------------------------------------------------------


class TestDetectNctId:
    """Edge cases for NCT ID detection."""

    def test_priority_registry_id_over_filename(self) -> None:
        """registry_id takes priority over filename."""
        result = detect_nct_id(
            registry_id="NCT11111111",
            filename="NCT22222222_1.0.pdf",
        )
        assert result == "NCT11111111"

    def test_filename_fallback(self) -> None:
        """Filename used when registry_id has no NCT."""
        result = detect_nct_id(
            registry_id="EUCTR2020-001234",
            filename="NCT22222222_1.0.pdf",
        )
        assert result == "NCT22222222"

    def test_metadata_nct_id_key(self) -> None:
        """Metadata with nct_id key (underscore) detected."""
        result = detect_nct_id(
            metadata={"nct_id": "NCT33333333"}
        )
        assert result == "NCT33333333"

    def test_metadata_registry_id_key(self) -> None:
        """Metadata with registry_id containing NCT detected."""
        result = detect_nct_id(
            metadata={"registry_id": "NCT44444444"}
        )
        assert result == "NCT44444444"

    def test_non_nct_metadata_returns_none(self) -> None:
        """Metadata without NCT patterns returns None."""
        result = detect_nct_id(
            metadata={"registry_id": "EUCTR2020-001234-56"}
        )
        assert result is None
