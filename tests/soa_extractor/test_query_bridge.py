"""Tests for query_bridge — PTCV-112.

Verifies the adapter that converts AssembledProtocol sections into
IchSection objects for the SoA extraction pipeline.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    AssembledSection,
    CoverageReport,
    QueryExtractionHit,
    SourceReference,
)
from ptcv.soa_extractor.query_bridge import (
    assembled_to_sections,
    get_assembled_protocol,
    has_query_pipeline_results,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_hit(
    query_id: str,
    section_id: str,
    parent: str,
    content: str,
    confidence: float = 0.85,
) -> QueryExtractionHit:
    return QueryExtractionHit(
        query_id=query_id,
        section_id=section_id,
        parent_section=parent,
        query_text=f"Query for {section_id}",
        extracted_content=content,
        confidence=confidence,
    )


def _make_section(
    code: str,
    name: str,
    hits: list[QueryExtractionHit] | None = None,
    populated: bool = True,
) -> AssembledSection:
    hits = hits or []
    avg = (
        sum(h.confidence for h in hits) / len(hits) if hits else 0.0
    )
    return AssembledSection(
        section_code=code,
        section_name=name,
        populated=populated,
        hits=hits,
        average_confidence=avg,
        is_gap=not populated,
        has_low_confidence=any(h.confidence < 0.70 for h in hits),
        required_query_count=len(hits),
        answered_required_count=len(hits),
    )


def _make_protocol(
    sections: list[AssembledSection],
) -> AssembledProtocol:
    return AssembledProtocol(
        sections=sections,
        coverage=CoverageReport(
            total_sections=16,
            populated_count=len(sections),
            gap_count=16 - len(sections),
            average_confidence=0.80,
            high_confidence_count=0,
            medium_confidence_count=len(sections),
            low_confidence_count=0,
            total_queries=10,
            answered_queries=5,
            required_queries=10,
            answered_required=5,
            gap_sections=[],
            low_confidence_sections=[],
        ),
        source_traceability={},
    )


SOA_TABLE = (
    "| Assessment | Screening | Week 2 |\n"
    "|------------|-----------|--------|\n"
    "| ECG        | X         | X      |\n"
    "| Labs       | X         |        |\n"
)


# -----------------------------------------------------------------------
# TestAssembledToSections
# -----------------------------------------------------------------------


class TestAssembledToSections:
    """assembled_to_sections() converts relevant sections."""

    def test_extracts_b4_and_b7(self) -> None:
        """B.4 and B.7 sections are converted."""
        protocol = _make_protocol([
            _make_section("B.1", "General", [
                _make_hit("B.1.q1", "B.1", "B.1", "Title"),
            ]),
            _make_section("B.4", "Trial Design", [
                _make_hit("B.4.q1", "B.4", "B.4", SOA_TABLE),
            ]),
            _make_section("B.7", "Schedule of Activities", [
                _make_hit("B.7.q1", "B.7", "B.7", "Visit schedule"),
            ]),
        ])

        result = assembled_to_sections(
            protocol, registry_id="NCT001",
        )

        assert len(result) == 2
        codes = {s.section_code for s in result}
        assert codes == {"B.4", "B.7"}

    def test_content_json_has_text_field(self) -> None:
        """content_json wraps content as {"text": "..."}."""
        protocol = _make_protocol([
            _make_section("B.4", "Trial Design", [
                _make_hit("B.4.q1", "B.4", "B.4", SOA_TABLE),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 1

        data = json.loads(result[0].content_json)
        assert "text" in data
        assert "ECG" in data["text"]

    def test_skips_unpopulated_sections(self) -> None:
        """Sections with populated=False are not included."""
        protocol = _make_protocol([
            _make_section(
                "B.4", "Trial Design", populated=False,
            ),
            _make_section("B.7", "Schedule", [
                _make_hit("B.7.q1", "B.7", "B.7", "data"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 1
        assert result[0].section_code == "B.7"

    def test_skips_empty_content(self) -> None:
        """Sections where all hits have empty content are skipped."""
        protocol = _make_protocol([
            _make_section("B.4", "Trial Design", [
                _make_hit("B.4.q1", "B.4", "B.4", ""),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 0

    def test_concatenates_multiple_hits(self) -> None:
        """Multiple hits in one section are joined."""
        protocol = _make_protocol([
            _make_section("B.7", "Schedule", [
                _make_hit("B.7.q1", "B.7", "B.7", "Part A"),
                _make_hit("B.7.q2", "B.7", "B.7", "Part B"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 1
        data = json.loads(result[0].content_json)
        assert "Part A" in data["text"]
        assert "Part B" in data["text"]

    def test_preserves_registry_id(self) -> None:
        """registry_id is passed through to IchSection."""
        protocol = _make_protocol([
            _make_section("B.4", "Design", [
                _make_hit("B.4.q1", "B.4", "B.4", "content"),
            ]),
        ])

        result = assembled_to_sections(
            protocol, registry_id="NCT02856802",
        )
        assert result[0].registry_id == "NCT02856802"

    def test_source_run_id_is_query_pipeline(self) -> None:
        """source_run_id is set to 'query_pipeline' for traceability."""
        protocol = _make_protocol([
            _make_section("B.4", "Design", [
                _make_hit("B.4.q1", "B.4", "B.4", "content"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert result[0].source_run_id == "query_pipeline"

    def test_confidence_from_section_average(self) -> None:
        """confidence_score reflects the section average."""
        protocol = _make_protocol([
            _make_section("B.4", "Design", [
                _make_hit("B.4.q1", "B.4", "B.4", "text", 0.90),
                _make_hit("B.4.q2", "B.4", "B.4", "more", 0.80),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert result[0].confidence_score == pytest.approx(0.85)

    def test_returns_empty_when_no_relevant_sections(self) -> None:
        """Protocol with only B.1 returns empty list."""
        protocol = _make_protocol([
            _make_section("B.1", "General", [
                _make_hit("B.1.q1", "B.1", "B.1", "Title"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert result == []


# -----------------------------------------------------------------------
# TestSessionStateHelpers
# -----------------------------------------------------------------------


class TestSessionStateHelpers:
    """has_query_pipeline_results and get_assembled_protocol."""

    def test_has_results_empty(self) -> None:
        assert has_query_pipeline_results({}) is False
        assert has_query_pipeline_results(
            {"query_cache": {}}
        ) is False

    def test_has_results_populated(self) -> None:
        assert has_query_pipeline_results(
            {"query_cache": {"abc_v1_t1": {"assembled": "x"}}}
        ) is True

    def test_get_assembled_found(self) -> None:
        assembled = _make_protocol([])
        state = {
            "query_cache": {
                "sha123_v2_t1": {"assembled": assembled},
            },
        }
        result = get_assembled_protocol(state, "sha123")
        assert result is assembled

    def test_get_assembled_not_found(self) -> None:
        state = {
            "query_cache": {
                "sha999_v2_t1": {"assembled": "proto"},
            },
        }
        result = get_assembled_protocol(state, "sha123")
        assert result is None

    def test_get_assembled_no_cache(self) -> None:
        result = get_assembled_protocol({}, "sha123")
        assert result is None
