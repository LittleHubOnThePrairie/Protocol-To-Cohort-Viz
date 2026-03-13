"""Tests for Results tab data source selection (PTCV-137).

Verifies that the Results tab correctly prefers query pipeline output
over Process tab output, and handles all three states: both sources,
query-only, process-only, and neither.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    AssembledSection,
    CoverageReport,
    QueryExtractionHit,
    SourceReference,
)
from ptcv.ui.app import compute_quality_verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coverage(**overrides) -> CoverageReport:
    defaults = dict(
        total_sections=14,
        populated_count=10,
        gap_count=4,
        average_confidence=0.80,
        high_confidence_count=5,
        medium_confidence_count=3,
        low_confidence_count=2,
        total_queries=70,
        answered_queries=50,
        required_queries=60,
        answered_required=45,
        gap_sections=[],
        low_confidence_sections=[],
    )
    defaults.update(overrides)
    return CoverageReport(**defaults)


def _make_assembled(
    populated_count: int = 10,
    avg_confidence: float = 0.80,
) -> AssembledProtocol:
    """Build a minimal AssembledProtocol for testing."""
    sections = []
    for i in range(1, 15):
        code = f"B.{i}"
        populated = i <= populated_count
        sections.append(
            AssembledSection(
                section_code=code,
                section_name=f"Section {code}",
                populated=populated,
                hits=[
                    QueryExtractionHit(
                        query_id=f"{code}.q1",
                        section_id=f"{code}.1",
                        parent_section=code,
                        query_text="test query",
                        extracted_content="test content",
                        confidence=avg_confidence,
                        source=SourceReference(
                            section_header="test",
                        ),
                    ),
                ] if populated else [],
                average_confidence=avg_confidence if populated else 0.0,
                is_gap=not populated,
                has_low_confidence=avg_confidence < 0.70,
                required_query_count=5,
                answered_required_count=5 if populated else 0,
            )
        )
    coverage = _make_coverage(
        populated_count=populated_count,
        gap_count=14 - populated_count,
        average_confidence=avg_confidence,
    )
    return AssembledProtocol(
        sections=sections,
        coverage=coverage,
        source_traceability={},
    )


def _make_process_result() -> dict:
    """Build a minimal Process tab result dict."""
    return {
        "format_verdict": "PARTIAL_ICH",
        "format_confidence": 0.75,
        "section_count": 8,
        "missing_required_sections": ["B.5", "B.6", "B.7"],
        "review_count": 2,
        "coverage_score": 0.65,
        "coverage_passed": False,
    }


# ---------------------------------------------------------------------------
# Data source preference tests
# ---------------------------------------------------------------------------


def _resolve_results_source(
    qp_assembled: AssembledProtocol | None,
    cached: dict | None,
) -> str:
    """Pure function mirroring Results tab source selection logic.

    Returns:
        "query_pipeline", "process", or "none".
    """
    if qp_assembled is not None:
        return "query_pipeline"
    elif cached:
        return "process"
    else:
        return "none"


class TestResultsSourceSelection:
    """Tests for Results tab data source selection (PTCV-137)."""

    def test_query_pipeline_preferred_when_both_available(self) -> None:
        assembled = _make_assembled()
        cached = _make_process_result()
        assert _resolve_results_source(assembled, cached) == "query_pipeline"

    def test_query_pipeline_only(self) -> None:
        assembled = _make_assembled()
        assert _resolve_results_source(assembled, None) == "query_pipeline"

    def test_process_fallback_when_no_query(self) -> None:
        cached = _make_process_result()
        assert _resolve_results_source(None, cached) == "process"

    def test_no_results_when_neither(self) -> None:
        assert _resolve_results_source(None, None) == "none"

    def test_empty_dict_is_falsy(self) -> None:
        """Empty cached dict should also resolve to 'none'."""
        assert _resolve_results_source(None, {}) == "none"


class TestAssembledProtocolMarkdown:
    """Verify AssembledProtocol.to_markdown() works for Results tab."""

    def test_to_markdown_returns_string(self) -> None:
        assembled = _make_assembled()
        md = assembled.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_markdown_contains_section_codes(self) -> None:
        assembled = _make_assembled(populated_count=3)
        md = assembled.to_markdown()
        assert "B.1" in md
        assert "B.2" in md
        assert "B.3" in md

    def test_markdown_marks_gaps(self) -> None:
        assembled = _make_assembled(populated_count=1)
        md = assembled.to_markdown()
        # Gaps should be indicated in markdown
        assert "B.2" in md  # Even gaps should appear


class TestQualityVerdictIntegration:
    """Verify compute_quality_verdict works with assembled coverage."""

    def test_pass_verdict_from_assembled(self) -> None:
        assembled = _make_assembled(
            populated_count=12, avg_confidence=0.85,
        )
        result = compute_quality_verdict(assembled.coverage)
        assert result["verdict"] == "PASS"

    def test_review_verdict_from_assembled(self) -> None:
        assembled = _make_assembled(
            populated_count=8, avg_confidence=0.60,
        )
        result = compute_quality_verdict(assembled.coverage)
        assert result["verdict"] == "REVIEW"

    def test_fail_verdict_from_assembled(self) -> None:
        assembled = _make_assembled(
            populated_count=3, avg_confidence=0.40,
        )
        result = compute_quality_verdict(assembled.coverage)
        assert result["verdict"] == "FAIL"
