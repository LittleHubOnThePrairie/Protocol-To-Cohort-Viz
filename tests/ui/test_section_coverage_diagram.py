"""Tests for section coverage diagram (PTCV-270).

Tests color-coded block generation from AssembledProtocol data.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ui.components.section_coverage_diagram import (
    SectionBlock,
    build_section_blocks,
)
from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    AssembledSection,
    CoverageReport,
    QueryExtractionHit,
    SourceReference,
)


def _make_section(
    code: str,
    name: str,
    confidence: float = 0.90,
    populated: bool = True,
    is_gap: bool = False,
    hit_count: int = 3,
) -> AssembledSection:
    hits = [
        QueryExtractionHit(
            query_id=f"{code}.q{i}",
            section_id=code,
            parent_section=code,
            query_text="",
            extracted_content="content",
            confidence=confidence,
            source=SourceReference(),
        )
        for i in range(hit_count)
    ]
    return AssembledSection(
        section_code=code,
        section_name=name,
        populated=populated,
        hits=hits if populated else [],
        average_confidence=confidence if populated else 0.0,
        is_gap=is_gap,
        has_low_confidence=confidence < 0.70 if populated else False,
        required_query_count=3,
        answered_required_count=hit_count if populated else 0,
    )


def _make_assembled(sections: list[AssembledSection]) -> AssembledProtocol:
    return AssembledProtocol(
        sections=sections,
        coverage=CoverageReport(
            total_sections=len(sections),
            populated_count=sum(1 for s in sections if s.populated),
            gap_count=sum(1 for s in sections if s.is_gap),
            average_confidence=0.85,
            high_confidence_count=0,
            medium_confidence_count=0,
            low_confidence_count=0,
            total_queries=0,
            answered_queries=0,
            required_queries=0,
            answered_required=0,
            gap_sections=[],
            low_confidence_sections=[],
        ),
        source_traceability={},
    )


class TestBuildSectionBlocks:
    def test_all_16_sections(self):
        """GHERKIN: Diagram shows all 16 ICH sections."""
        sections = [
            _make_section(f"B.{i}", f"Section {i}")
            for i in range(1, 17)
        ]
        assembled = _make_assembled(sections)
        blocks = build_section_blocks(assembled)
        assert len(blocks) == 16

    def test_high_confidence_green(self):
        sections = [_make_section("B.4", "Trial Design", confidence=0.92)]
        blocks = build_section_blocks(_make_assembled(sections))
        assert blocks[0].status == "high"
        assert blocks[0].color == "#28a745"

    def test_moderate_confidence_amber(self):
        sections = [_make_section("B.3", "Objectives", confidence=0.75)]
        blocks = build_section_blocks(_make_assembled(sections))
        assert blocks[0].status == "moderate"
        assert blocks[0].color == "#ffc107"

    def test_low_confidence_red(self):
        sections = [_make_section("B.11", "Direct Access", confidence=0.50)]
        blocks = build_section_blocks(_make_assembled(sections))
        assert blocks[0].status == "low"
        assert blocks[0].color == "#dc3545"

    def test_gap_section_red(self):
        """GHERKIN: Gap sections appear in red."""
        sections = [_make_section(
            "B.14", "Financing", populated=False, is_gap=True,
        )]
        blocks = build_section_blocks(_make_assembled(sections))
        assert blocks[0].status == "gap"
        assert blocks[0].is_gap is True
        assert blocks[0].color == "#dc3545"

    def test_hit_count_preserved(self):
        sections = [_make_section("B.5", "Subjects", hit_count=5)]
        blocks = build_section_blocks(_make_assembled(sections))
        assert blocks[0].hit_count == 5

    def test_confidence_preserved(self):
        sections = [_make_section("B.4", "Design", confidence=0.92)]
        blocks = build_section_blocks(_make_assembled(sections))
        assert blocks[0].confidence == 0.92

    def test_mixed_statuses(self):
        """GHERKIN: Populated sections green/amber, gaps red."""
        sections = [
            _make_section("B.1", "General", confidence=0.90),
            _make_section("B.2", "Background", confidence=0.75),
            _make_section("B.3", "Objectives", confidence=0.50),
            _make_section("B.11", "Direct Access", populated=False, is_gap=True),
        ]
        blocks = build_section_blocks(_make_assembled(sections))
        statuses = [b.status for b in blocks]
        assert statuses == ["high", "moderate", "low", "gap"]

    def test_boundary_high(self):
        """Exactly 0.85 is high confidence."""
        sections = [_make_section("B.1", "General", confidence=0.85)]
        blocks = build_section_blocks(_make_assembled(sections))
        assert blocks[0].status == "high"

    def test_boundary_moderate(self):
        """Exactly 0.70 is moderate confidence."""
        sections = [_make_section("B.1", "General", confidence=0.70)]
        blocks = build_section_blocks(_make_assembled(sections))
        assert blocks[0].status == "moderate"


class TestSectionBlock:
    def test_dataclass_fields(self):
        block = SectionBlock(
            code="B.4", name="Trial Design",
            confidence=0.92, status="high",
            color="#28a745", hit_count=3, is_gap=False,
        )
        assert block.code == "B.4"
        assert block.confidence == 0.92
        assert block.is_gap is False
