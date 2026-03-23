"""Tests for query-level registry injection (PTCV-259).

GHERKIN Scenarios:
  Feature: Query-level registry injection

    Scenario: Protocol title from registry
      Given B.1.1 query has LOW CONFIDENCE from PDF extraction
      And CT.gov has officialTitle = "A Phase 2 Study of Drug X"
      When query-level registry injection runs
      Then B.1.1 content is set to the official title
      And confidence is 0.85
      And extraction_method is "registry_direct"

    Scenario: Endpoints from registry
      Given B.4.1 query has LOW CONFIDENCE
      And CT.gov has 3 primary outcomes
      When query-level registry injection runs
      Then B.4.1 content lists all 3 primary outcomes
      And each outcome includes measure and timeFrame

    Scenario: No registry override for HIGH confidence PDF
      Given B.5.1 query has HIGH CONFIDENCE from PDF
      And CT.gov also has eligibility criteria
      When query-level registry injection runs
      Then B.5.1 keeps its PDF-extracted content
      And registry data is not injected

    Scenario: Query with no registry mapping skipped
      Given B.12 (Ethics) query has LOW CONFIDENCE
      And no registry field maps to B.12
      When query-level registry injection runs
      Then B.12 is unchanged (stays LOW CONFIDENCE)
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.query_extractor import QueryExtraction
from ptcv.registry.query_injector import (
    QUERY_TO_REGISTRY_FIELD,
    format_registry_answer,
    inject_registry_answers,
    navigate_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REGISTRY: dict[str, Any] = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT12345678",
            "officialTitle": "A Phase 2 Study of Drug X in Solid Tumors",
            "briefTitle": "Drug X Phase 2",
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {
                "name": "Acme Pharma Inc.",
                "class": "INDUSTRY",
            },
        },
        "contactsLocationsModule": {
            "overallOfficials": [
                {
                    "name": "Dr. Jane Smith",
                    "role": "PRINCIPAL_INVESTIGATOR",
                    "affiliation": "University Hospital",
                },
            ],
        },
        "descriptionModule": {
            "briefSummary": (
                "This is a Phase 2 study to evaluate the efficacy "
                "and safety of Drug X in patients with solid tumors."
            ),
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {
                    "measure": "Overall Response Rate",
                    "timeFrame": "24 weeks",
                    "description": "Per RECIST 1.1",
                },
                {
                    "measure": "Duration of Response",
                    "timeFrame": "Up to 2 years",
                    "description": "Time from first response to progression",
                },
                {
                    "measure": "Progression-Free Survival",
                    "timeFrame": "Up to 2 years",
                    "description": "Time from randomization to progression",
                },
            ],
            "secondaryOutcomes": [
                {
                    "measure": "Overall Survival",
                    "timeFrame": "Up to 5 years",
                    "description": "",
                },
            ],
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\n"
                "- Age >= 18 years\n"
                "- ECOG PS 0-1\n"
                "Exclusion Criteria:\n"
                "- Prior therapy with Drug X"
            ),
        },
        "designModule": {
            "enrollmentInfo": {
                "count": 120,
                "type": "ESTIMATED",
            },
        },
    },
}


def _make_extraction(
    query_id: str,
    section_id: str,
    content: str,
    confidence: float,
    extraction_method: str = "heuristic",
) -> QueryExtraction:
    """Build a QueryExtraction fixture."""
    return QueryExtraction(
        query_id=query_id,
        section_id=section_id,
        content=content,
        confidence=confidence,
        extraction_method=extraction_method,
        source_section=section_id,
    )


# ---------------------------------------------------------------------------
# navigate_json tests
# ---------------------------------------------------------------------------


class TestNavigateJson:

    def test_simple_path(self):
        data = {"a": {"b": {"c": "value"}}}
        assert navigate_json(data, "a.b.c") == "value"

    def test_missing_key(self):
        data = {"a": {"b": 1}}
        assert navigate_json(data, "a.x.y") is None

    def test_non_dict_intermediate(self):
        data = {"a": "string"}
        assert navigate_json(data, "a.b") is None

    def test_registry_official_title(self):
        result = navigate_json(
            SAMPLE_REGISTRY,
            "protocolSection.identificationModule.officialTitle",
        )
        assert result == "A Phase 2 Study of Drug X in Solid Tumors"

    def test_registry_lead_sponsor(self):
        result = navigate_json(
            SAMPLE_REGISTRY,
            "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
        )
        assert result == "Acme Pharma Inc."


# ---------------------------------------------------------------------------
# format_registry_answer tests
# ---------------------------------------------------------------------------


class TestFormatRegistryAnswer:

    def test_string_value(self):
        result = format_registry_answer("A Phase 2 Study")
        assert result == "A Phase 2 Study"

    def test_outcomes_list(self):
        outcomes = [
            {"measure": "ORR", "timeFrame": "24 weeks", "description": "RECIST"},
            {"measure": "PFS", "timeFrame": "2 years", "description": ""},
        ]
        result = format_registry_answer(outcomes)
        assert "1. ORR" in result
        assert "Time Frame: 24 weeks" in result
        assert "2. PFS" in result

    def test_officials_list(self):
        officials = [
            {"name": "Dr. Smith", "role": "PI", "affiliation": "Hospital"},
        ]
        result = format_registry_answer(officials)
        assert "Dr. Smith (PI)" in result
        assert "Hospital" in result

    def test_enrollment_dict(self):
        result = format_registry_answer({"count": 120, "type": "ESTIMATED"})
        assert "120" in result
        assert "ESTIMATED" in result

    def test_empty_list(self):
        assert format_registry_answer([]) == ""

    def test_empty_string(self):
        assert format_registry_answer("  ") == ""


# ---------------------------------------------------------------------------
# Scenario: Protocol title from registry
# ---------------------------------------------------------------------------


class TestProtocolTitleFromRegistry:
    """B.1.1 LOW confidence → replaced with officialTitle."""

    def test_content_set_to_official_title(self):
        extractions = [
            _make_extraction(
                "B.1.1.q1", "B.1.1",
                "Some fuzzy title extraction", 0.45,
            ),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        assert len(result) == 1
        assert result[0].content == "A Phase 2 Study of Drug X in Solid Tumors"

    def test_confidence_is_085(self):
        extractions = [
            _make_extraction("B.1.1.q1", "B.1.1", "fuzzy", 0.45),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)
        assert result[0].confidence == 0.85

    def test_extraction_method_is_registry_direct(self):
        extractions = [
            _make_extraction("B.1.1.q1", "B.1.1", "fuzzy", 0.45),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)
        assert result[0].extraction_method == "registry_direct"

    def test_verbatim_content_preserves_original(self):
        extractions = [
            _make_extraction("B.1.1.q1", "B.1.1", "original text", 0.45),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)
        assert result[0].verbatim_content == "original text"


# ---------------------------------------------------------------------------
# Scenario: Endpoints from registry
# ---------------------------------------------------------------------------


class TestEndpointsFromRegistry:
    """B.4.1 LOW confidence → replaced with primary outcomes."""

    def test_all_three_outcomes_listed(self):
        extractions = [
            _make_extraction(
                "B.4.1.q1", "B.4.1",
                "Some vague endpoint text", 0.40,
            ),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        content = result[0].content
        assert "Overall Response Rate" in content
        assert "Duration of Response" in content
        assert "Progression-Free Survival" in content

    def test_each_outcome_includes_measure_and_timeframe(self):
        extractions = [
            _make_extraction("B.4.1.q1", "B.4.1", "vague", 0.40),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        content = result[0].content
        assert "Time Frame: 24 weeks" in content
        assert "Time Frame: Up to 2 years" in content

    def test_secondary_outcomes_injected(self):
        extractions = [
            _make_extraction("B.4.1.q2", "B.4.1", "vague", 0.40),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        content = result[0].content
        assert "Overall Survival" in content


# ---------------------------------------------------------------------------
# Scenario: No registry override for HIGH confidence PDF
# ---------------------------------------------------------------------------


class TestNoOverrideForHighConfidence:
    """B.5.1 HIGH confidence → keeps PDF content."""

    def test_high_confidence_not_overridden(self):
        original_content = (
            "Inclusion Criteria:\n"
            "- Confirmed solid tumor\n"
            "- Age 18+\n"
            "Exclusion Criteria:\n"
            "- Brain metastases"
        )
        extractions = [
            _make_extraction(
                "B.5.1.q1", "B.5.1",
                original_content, 0.88,
            ),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        assert result[0].content == original_content
        assert result[0].confidence == 0.88
        assert result[0].extraction_method == "heuristic"

    def test_borderline_confidence_not_overridden(self):
        """Confidence exactly at threshold should NOT be replaced."""
        extractions = [
            _make_extraction("B.5.1.q1", "B.5.1", "PDF content", 0.70),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        assert result[0].content == "PDF content"
        assert result[0].confidence == 0.70


# ---------------------------------------------------------------------------
# Scenario: Query with no registry mapping skipped
# ---------------------------------------------------------------------------


class TestUnmappedQuerySkipped:
    """B.12 (Ethics) has no registry mapping → unchanged."""

    def test_unmapped_query_unchanged(self):
        extractions = [
            _make_extraction(
                "B.12.q1", "B.12",
                "Ethics committee approval pending", 0.35,
            ),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        assert result[0].content == "Ethics committee approval pending"
        assert result[0].confidence == 0.35
        assert result[0].extraction_method == "heuristic"

    def test_multiple_unmapped_queries_all_unchanged(self):
        extractions = [
            _make_extraction("B.12.q1", "B.12", "ethics", 0.35),
            _make_extraction("B.14.q1", "B.14", "data handling", 0.40),
            _make_extraction("B.15.q1", "B.15", "safety reporting", 0.30),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        for orig, injected in zip(extractions, result):
            assert injected.content == orig.content
            assert injected.confidence == orig.confidence


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_registry_metadata(self):
        extractions = [
            _make_extraction("B.1.1.q1", "B.1.1", "title", 0.40),
        ]
        result = inject_registry_answers(extractions, {})
        assert result[0].content == "title"

    def test_registry_field_missing(self):
        """Registry exists but the specific field is None."""
        partial_registry = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    # officialTitle intentionally missing
                },
            },
        }
        extractions = [
            _make_extraction("B.1.1.q1", "B.1.1", "title", 0.40),
        ]
        result = inject_registry_answers(extractions, partial_registry)
        assert result[0].content == "title"  # Not overridden

    def test_mixed_queries_selective_injection(self):
        """Only low-confidence mapped queries are replaced."""
        extractions = [
            _make_extraction("B.1.1.q1", "B.1.1", "fuzzy title", 0.45),
            _make_extraction("B.5.1.q1", "B.5.1", "good criteria", 0.85),
            _make_extraction("B.12.q1", "B.12", "ethics", 0.30),
            _make_extraction("B.1.2.q1", "B.1.2", "unknown sponsor", 0.50),
        ]
        result = inject_registry_answers(extractions, SAMPLE_REGISTRY)

        # B.1.1 → injected (low confidence, mapped)
        assert result[0].extraction_method == "registry_direct"
        # B.5.1 → kept (high confidence)
        assert result[1].extraction_method == "heuristic"
        # B.12 → kept (no mapping)
        assert result[2].extraction_method == "heuristic"
        # B.1.2 → injected (low confidence, mapped)
        assert result[3].extraction_method == "registry_direct"
        assert result[3].content == "Acme Pharma Inc."

    def test_original_list_not_mutated(self):
        """inject_registry_answers should not mutate the input list."""
        extractions = [
            _make_extraction("B.1.1.q1", "B.1.1", "fuzzy", 0.45),
        ]
        original_copy = list(extractions)
        inject_registry_answers(extractions, SAMPLE_REGISTRY)
        assert extractions == original_copy

    def test_custom_confidence_threshold(self):
        """Custom threshold should be respected."""
        extractions = [
            _make_extraction("B.1.1.q1", "B.1.1", "title", 0.60),
        ]
        # With threshold 0.50, 0.60 is above → not replaced
        result = inject_registry_answers(
            extractions, SAMPLE_REGISTRY,
            confidence_threshold=0.50,
        )
        assert result[0].content == "title"

        # With threshold 0.70, 0.60 is below → replaced
        result = inject_registry_answers(
            extractions, SAMPLE_REGISTRY,
            confidence_threshold=0.70,
        )
        assert result[0].extraction_method == "registry_direct"


# ---------------------------------------------------------------------------
# Mapping coverage
# ---------------------------------------------------------------------------


class TestMappingCoverage:
    """Verify all documented mappings are present."""

    def test_all_expected_mappings_exist(self):
        expected = {
            "B.1.1.q1", "B.1.2.q1", "B.1.3.q1",
            "B.3.q1", "B.3.q2",
            "B.4.1.q1", "B.4.1.q2",
            "B.5.1.q1",
            "B.10.q1",
        }
        assert set(QUERY_TO_REGISTRY_FIELD.keys()) == expected

    def test_all_field_paths_resolve_in_sample(self):
        """Every mapped field path should resolve in sample registry."""
        for query_id, path in QUERY_TO_REGISTRY_FIELD.items():
            value = navigate_json(SAMPLE_REGISTRY, path)
            assert value is not None, (
                f"{query_id} → {path} resolved to None"
            )

    def test_b3_maps_to_brief_summary_not_title(self):
        """PTCV-285: B.3.q1 should map to briefSummary (richer) not briefTitle."""
        path = QUERY_TO_REGISTRY_FIELD["B.3.q1"]
        assert "briefSummary" in path
        value = navigate_json(SAMPLE_REGISTRY, path)
        assert "Phase 2 study" in value

    def test_b3_q2_maps_to_secondary_outcomes(self):
        """PTCV-285: B.3.q2 maps to secondaryOutcomes for endpoint detail."""
        path = QUERY_TO_REGISTRY_FIELD["B.3.q2"]
        assert "secondaryOutcomes" in path
        value = navigate_json(SAMPLE_REGISTRY, path)
        assert isinstance(value, list)
        assert len(value) >= 1
