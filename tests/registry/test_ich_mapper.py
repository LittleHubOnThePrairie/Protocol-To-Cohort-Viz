"""Tests for MetadataToIchMapper (PTCV-195).

Covers all GHERKIN scenarios:
- Map CT.gov fields to ICH section codes
- Handle missing optional modules
- Multi-value field concatenation
- Quality rating assignment
"""

import json
from typing import Any

import pytest

from ptcv.registry.ich_mapper import (
    QUALITY_CONTEXTUAL,
    QUALITY_DIRECT,
    QUALITY_PARTIAL,
    MappedRegistrySection,
    MetadataToIchMapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_metadata() -> dict[str, Any]:
    """Return a complete CT.gov study JSON with all relevant modules."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT01512251",
                "officialTitle": (
                    "A Phase 1/2 Trial of BKM120 Combined With "
                    "Vemurafenib in BRAFV600E/K Mutant Melanoma"
                ),
                "briefTitle": "BKM120 + Vemurafenib in Melanoma",
                "organization": {
                    "fullName": "University of California, San Francisco",
                },
            },
            "statusModule": {
                "overallStatus": "Completed",
                "startDateStruct": {"date": "2012-02"},
                "completionDateStruct": {"date": "2016-12"},
            },
            "conditionsModule": {
                "conditions": ["Melanoma", "BRAF V600E Mutation"],
                "keywords": ["PI3K", "BRAF", "melanoma"],
            },
            "descriptionModule": {
                "briefSummary": (
                    "This study evaluates BKM120 combined with "
                    "vemurafenib in advanced melanoma."
                ),
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE1", "PHASE2"],
                "designInfo": {
                    "allocation": "NON_RANDOMIZED",
                    "interventionModel": "SEQUENTIAL",
                    "primaryPurpose": "TREATMENT",
                    "maskingInfo": {"masking": "NONE"},
                },
                "enrollmentInfo": {
                    "count": 30,
                    "type": "ACTUAL",
                },
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "name": "BKM120",
                        "type": "DRUG",
                        "description": "Pan-class I PI3K inhibitor",
                    },
                    {
                        "name": "Vemurafenib",
                        "type": "DRUG",
                        "description": "BRAF V600E inhibitor",
                    },
                ],
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": "Maximum Tolerated Dose",
                        "timeFrame": "Cycle 1 (28 days)",
                    },
                    {
                        "measure": "Dose Limiting Toxicities",
                        "timeFrame": "Cycle 1 (28 days)",
                    },
                    {
                        "measure": "Overall Response Rate",
                        "timeFrame": "Up to 2 years",
                    },
                ],
                "secondaryOutcomes": [
                    {
                        "measure": "Progression Free Survival",
                        "timeFrame": "Up to 3 years",
                    },
                    {
                        "measure": "Overall Survival",
                        "timeFrame": "Up to 5 years",
                    },
                ],
            },
            "eligibilityModule": {
                "eligibilityCriteria": (
                    "Inclusion: BRAF V600E/K mutation\n"
                    "Exclusion: Prior PI3K inhibitor"
                ),
                "sex": "ALL",
                "minimumAge": "18 Years",
                "maximumAge": "N/A",
            },
            "referencesModule": {
                "references": [
                    {
                        "pmid": "25678901",
                        "citation": "Smith et al. J Clin Oncol. 2015",
                    },
                    {
                        "pmid": "26789012",
                        "citation": "Jones et al. NEJM. 2016",
                    },
                ],
            },
        },
    }


@pytest.fixture
def mapper() -> MetadataToIchMapper:
    return MetadataToIchMapper()


@pytest.fixture
def full_metadata() -> dict[str, Any]:
    return _full_metadata()


# ---------------------------------------------------------------------------
# Scenario: Map CT.gov fields to ICH section codes
# ---------------------------------------------------------------------------


class TestMapFields:
    """Scenario: Map CT.gov fields to ICH section codes."""

    def test_map_returns_all_sections(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Full metadata produces sections for all mapped codes."""
        sections = mapper.map(full_metadata)
        codes = {s.section_code for s in sections}

        expected = {"B.1", "B.2", "B.3", "B.4", "B.5",
                    "B.8", "B.10", "B.16"}
        assert codes == expected

    def test_each_section_has_source_registry(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Every mapped section has source='registry'."""
        sections = mapper.map(full_metadata)
        for sec in sections:
            assert sec.source == "registry"

    def test_each_section_has_quality_rating(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Every mapped section has a quality_rating > 0."""
        sections = mapper.map(full_metadata)
        for sec in sections:
            assert 0.0 < sec.quality_rating <= 1.0

    def test_each_section_has_nonempty_content(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """No section is produced with empty content_text."""
        sections = mapper.map(full_metadata)
        for sec in sections:
            assert sec.content_text.strip()
            assert sec.content_json.strip()

    def test_content_json_is_valid_json(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """content_json is parseable JSON for every section."""
        sections = mapper.map(full_metadata)
        for sec in sections:
            parsed = json.loads(sec.content_json)
            assert isinstance(parsed, dict)

    def test_b3_contains_official_title(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """B.3 section includes the official trial title."""
        sections = mapper.map(full_metadata)
        b3 = next(s for s in sections if s.section_code == "B.3")
        assert "BKM120" in b3.content_text
        assert "Phase 1/2" in b3.content_text

    def test_b4_contains_design_and_interventions(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """B.4 includes study type, phase, and interventions."""
        sections = mapper.map(full_metadata)
        b4 = next(s for s in sections if s.section_code == "B.4")
        assert "INTERVENTIONAL" in b4.content_text
        assert "BKM120" in b4.content_text
        assert "Vemurafenib" in b4.content_text

    def test_b5_contains_eligibility(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """B.5 includes eligibility criteria and demographics."""
        sections = mapper.map(full_metadata)
        b5 = next(s for s in sections if s.section_code == "B.5")
        assert "BRAF V600E/K" in b5.content_text
        assert "18 Years" in b5.content_text


# ---------------------------------------------------------------------------
# Scenario: Handle missing optional modules
# ---------------------------------------------------------------------------


class TestMissingModules:
    """Scenario: Handle missing optional modules."""

    def test_missing_outcomes_omits_b8(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Empty OutcomesModule → no B.8 section."""
        full_metadata["protocolSection"]["outcomesModule"] = {}
        sections = mapper.map(full_metadata)
        codes = {s.section_code for s in sections}
        assert "B.8" not in codes

    def test_missing_outcomes_preserves_others(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Missing OutcomesModule doesn't affect other sections."""
        full_metadata["protocolSection"]["outcomesModule"] = {}
        sections = mapper.map(full_metadata)
        codes = {s.section_code for s in sections}
        assert "B.1" in codes
        assert "B.3" in codes
        assert "B.4" in codes
        assert "B.5" in codes

    def test_missing_conditions_omits_b2(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Empty ConditionsModule → no B.2 section."""
        full_metadata["protocolSection"]["conditionsModule"] = {}
        sections = mapper.map(full_metadata)
        codes = {s.section_code for s in sections}
        assert "B.2" not in codes

    def test_missing_references_omits_b16(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Empty ReferencesModule → no B.16 section."""
        full_metadata["protocolSection"]["referencesModule"] = {}
        sections = mapper.map(full_metadata)
        codes = {s.section_code for s in sections}
        assert "B.16" not in codes

    def test_empty_protocol_section(
        self, mapper: MetadataToIchMapper
    ) -> None:
        """Empty protocolSection returns no sections."""
        sections = mapper.map({"protocolSection": {}})
        assert sections == []

    def test_no_protocol_section(
        self, mapper: MetadataToIchMapper
    ) -> None:
        """Missing protocolSection key returns no sections."""
        sections = mapper.map({})
        assert sections == []

    def test_missing_enrollment_omits_b10(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """No enrollmentInfo → no B.10 section."""
        full_metadata["protocolSection"]["designModule"].pop(
            "enrollmentInfo", None
        )
        sections = mapper.map(full_metadata)
        codes = {s.section_code for s in sections}
        assert "B.10" not in codes


# ---------------------------------------------------------------------------
# Scenario: Multi-value field concatenation
# ---------------------------------------------------------------------------


class TestMultiValueConcatenation:
    """Scenario: Multi-value field concatenation."""

    def test_b8_contains_all_primary_outcomes(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """B.8 lists all 3 primary outcomes."""
        sections = mapper.map(full_metadata)
        b8 = next(s for s in sections if s.section_code == "B.8")
        assert "Maximum Tolerated Dose" in b8.content_text
        assert "Dose Limiting Toxicities" in b8.content_text
        assert "Overall Response Rate" in b8.content_text

        parsed = json.loads(b8.content_json)
        assert len(parsed["primary_outcomes"]) == 3

    def test_b8_contains_all_secondary_outcomes(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """B.8 also includes secondary outcomes (efficacy, not safety)."""
        sections = mapper.map(full_metadata)
        b8 = next(s for s in sections if s.section_code == "B.8")
        assert "Progression Free Survival" in b8.content_text
        assert "Overall Survival" in b8.content_text

        parsed = json.loads(b8.content_json)
        assert len(parsed["secondary_outcomes"]) == 2

    def test_b4_contains_all_interventions(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """B.4 lists both interventions with types."""
        sections = mapper.map(full_metadata)
        b4 = next(s for s in sections if s.section_code == "B.4")
        assert "BKM120" in b4.content_text
        assert "Vemurafenib" in b4.content_text
        assert "DRUG" in b4.content_text

    def test_b8_from_secondary_only(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Secondary-only outcomes still produce B.8."""
        full_metadata["protocolSection"]["outcomesModule"] = {
            "secondaryOutcomes": [
                {"measure": "PFS", "timeFrame": "3 years"},
            ],
        }
        sections = mapper.map(full_metadata)
        b8 = next(s for s in sections if s.section_code == "B.8")
        assert "PFS" in b8.content_text

    def test_b16_contains_all_references(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """B.16 lists both references with PMIDs."""
        sections = mapper.map(full_metadata)
        b16 = next(s for s in sections if s.section_code == "B.16")
        assert "25678901" in b16.content_text
        assert "26789012" in b16.content_text
        assert "Smith" in b16.content_text
        assert "Jones" in b16.content_text


# ---------------------------------------------------------------------------
# Scenario: Quality rating assignment
# ---------------------------------------------------------------------------


class TestQualityRatings:
    """Scenario: Quality rating assignment."""

    def test_direct_sections_get_0_9(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Direct-mapped sections (B.3, B.4, B.5, B.8, B.16) = 0.9."""
        sections = mapper.map(full_metadata)
        direct_codes = {"B.3", "B.4", "B.5", "B.8", "B.16"}
        for sec in sections:
            if sec.section_code in direct_codes:
                assert sec.quality_rating == QUALITY_DIRECT, (
                    f"{sec.section_code} should be Direct (0.9), "
                    f"got {sec.quality_rating}"
                )

    def test_partial_sections_get_0_6(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Partial-mapped sections (B.1, B.10) = 0.6."""
        sections = mapper.map(full_metadata)
        partial_codes = {"B.1", "B.10"}
        for sec in sections:
            if sec.section_code in partial_codes:
                assert sec.quality_rating == QUALITY_PARTIAL, (
                    f"{sec.section_code} should be Partial (0.6), "
                    f"got {sec.quality_rating}"
                )

    def test_contextual_sections_get_0_3(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Contextual-mapped sections (B.2) = 0.3."""
        sections = mapper.map(full_metadata)
        b2 = next(s for s in sections if s.section_code == "B.2")
        assert b2.quality_rating == QUALITY_CONTEXTUAL


# ---------------------------------------------------------------------------
# MappedRegistrySection.to_ich_kwargs
# ---------------------------------------------------------------------------


class TestToIchKwargs:
    """MappedRegistrySection → IchSection conversion."""

    def test_to_ich_kwargs_has_all_required_fields(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """to_ich_kwargs returns all IchSection fields."""
        sections = mapper.map(full_metadata)
        kwargs = sections[0].to_ich_kwargs(
            run_id="test-run",
            registry_id="NCT01512251",
        )

        required = {
            "run_id", "source_run_id", "source_sha256",
            "registry_id", "section_code", "section_name",
            "content_json", "confidence_score", "review_required",
            "legacy_format", "content_text",
        }
        assert required.issubset(kwargs.keys())

    def test_review_required_for_low_confidence(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Sections with quality < 0.7 have review_required=True."""
        sections = mapper.map(full_metadata)
        b2 = next(s for s in sections if s.section_code == "B.2")
        kwargs = b2.to_ich_kwargs()
        assert kwargs["review_required"] is True  # 0.3 < 0.7

    def test_review_not_required_for_high_confidence(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Sections with quality >= 0.7 have review_required=False."""
        sections = mapper.map(full_metadata)
        b3 = next(s for s in sections if s.section_code == "B.3")
        kwargs = b3.to_ich_kwargs()
        assert kwargs["review_required"] is False  # 0.9 >= 0.7

    def test_legacy_format_always_false(
        self, mapper: MetadataToIchMapper, full_metadata: dict[str, Any]
    ) -> None:
        """Registry-derived sections are never legacy format."""
        sections = mapper.map(full_metadata)
        for sec in sections:
            kwargs = sec.to_ich_kwargs()
            assert kwargs["legacy_format"] is False
