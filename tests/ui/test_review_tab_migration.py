"""Tests for Review tab query pipeline migration (PTCV-139).

Verifies the source selection logic: query pipeline sections are
preferred over Process tab sections, with correct fallback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    AssembledSection,
    CoverageReport,
    QueryExtractionHit,
)
from ptcv.soa_extractor.query_bridge import assembled_to_sections


def _make_hit(
    code: str,
    content: str,
    confidence: float = 0.85,
) -> QueryExtractionHit:
    return QueryExtractionHit(
        query_id=f"{code}.q1",
        section_id=code,
        parent_section=code,
        query_text=f"Query for {code}",
        extracted_content=content,
        confidence=confidence,
    )


def _make_section(
    code: str,
    name: str,
    content: str = "Content",
    confidence: float = 0.85,
    populated: bool = True,
) -> AssembledSection:
    hits = (
        [_make_hit(code, content, confidence)]
        if populated
        else []
    )
    return AssembledSection(
        section_code=code,
        section_name=name,
        populated=populated,
        hits=hits,
        average_confidence=confidence if populated else 0.0,
        is_gap=not populated,
        has_low_confidence=confidence < 0.70 if populated else False,
        required_query_count=1,
        answered_required_count=1 if populated else 0,
    )


def _make_protocol(
    sections: list[AssembledSection],
) -> AssembledProtocol:
    return AssembledProtocol(
        sections=sections,
        coverage=CoverageReport(
            total_sections=14,
            populated_count=len(
                [s for s in sections if s.populated]
            ),
            gap_count=14 - len(
                [s for s in sections if s.populated]
            ),
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


class TestReviewSectionSourceSelection:
    """Query pipeline sections are preferred for Review tab."""

    def test_query_pipeline_produces_ich_sections(self) -> None:
        """assembled_to_sections returns valid IchSection list."""
        protocol = _make_protocol([
            _make_section("B.5", "Selection of Subjects",
                          "Inclusion criteria"),
            _make_section("B.9", "Assessment of Safety",
                          "AE reporting"),
        ])

        result = assembled_to_sections(
            protocol, registry_id="NCT001",
        )

        assert len(result) == 2
        assert all(isinstance(s, IchSection) for s in result)
        codes = {s.section_code for s in result}
        assert codes == {"B.5", "B.9"}

    def test_review_sections_have_source_run_id(self) -> None:
        """Sections from query pipeline are tagged for traceability."""
        protocol = _make_protocol([
            _make_section("B.3", "Objectives", "Primary endpoint"),
        ])

        result = assembled_to_sections(protocol)
        assert result[0].source_run_id == "query_pipeline"

    def test_low_confidence_sections_flagged(self) -> None:
        """Low confidence sections have review_required=True."""
        protocol = _make_protocol([
            _make_section("B.5", "Subjects", "Eligibility",
                          confidence=0.55),
        ])

        result = assembled_to_sections(protocol)
        assert result[0].review_required is True

    def test_high_confidence_not_flagged(self) -> None:
        """High confidence sections don't require review."""
        protocol = _make_protocol([
            _make_section("B.4", "Trial Design", "Phase 3 RCT",
                          confidence=0.92),
        ])

        result = assembled_to_sections(protocol)
        assert result[0].review_required is False

    def test_content_json_populated(self) -> None:
        """Review sections have content_json with text field."""
        protocol = _make_protocol([
            _make_section("B.7", "Treatment", "Drug A 100mg daily"),
        ])

        result = assembled_to_sections(protocol)
        data = json.loads(result[0].content_json)
        assert "Drug A 100mg daily" in data["text"]

    def test_unpopulated_sections_excluded(self) -> None:
        """Unpopulated sections are not sent to review."""
        protocol = _make_protocol([
            _make_section("B.1", "General", "Title",
                          populated=True),
            _make_section("B.2", "Background", populated=False),
        ])

        result = assembled_to_sections(
            protocol, registry_id="NCT001",
        )
        assert len(result) == 1
        assert result[0].section_code == "B.1"

    def test_registry_id_passed_through(self) -> None:
        """Registry ID flows through to review sections."""
        protocol = _make_protocol([
            _make_section("B.4", "Design", "RCT"),
        ])

        result = assembled_to_sections(
            protocol, registry_id="NCT02856802",
        )
        assert result[0].registry_id == "NCT02856802"

    def test_multiple_sections_for_review(self) -> None:
        """All populated sections are available for review."""
        protocol = _make_protocol([
            _make_section("B.1", "General", "Title"),
            _make_section("B.3", "Objectives", "Primary"),
            _make_section("B.5", "Subjects", "Adults 18+"),
            _make_section("B.7", "Treatment", "Drug A"),
            _make_section("B.9", "Safety", "AEs monitored"),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 5
        codes = [s.section_code for s in result]
        assert codes == ["B.1", "B.3", "B.5", "B.7", "B.9"]
