"""Tests for Quality tab query pipeline integration (PTCV-138).

Tests the compute_quality_verdict() pure function and the
_display_query_quality() function contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.template_assembler import CoverageReport
from ptcv.ui.app import compute_quality_verdict


def _make_coverage(
    total_sections: int = 14,
    populated_count: int = 10,
    gap_count: int = 4,
    average_confidence: float = 0.80,
    high_confidence_count: int = 5,
    medium_confidence_count: int = 3,
    low_confidence_count: int = 2,
    total_queries: int = 70,
    answered_queries: int = 50,
    required_queries: int = 60,
    answered_required: int = 45,
    gap_sections: list[str] | None = None,
    low_confidence_sections: list[str] | None = None,
) -> CoverageReport:
    return CoverageReport(
        total_sections=total_sections,
        populated_count=populated_count,
        gap_count=gap_count,
        average_confidence=average_confidence,
        high_confidence_count=high_confidence_count,
        medium_confidence_count=medium_confidence_count,
        low_confidence_count=low_confidence_count,
        total_queries=total_queries,
        answered_queries=answered_queries,
        required_queries=required_queries,
        answered_required=answered_required,
        gap_sections=gap_sections or [],
        low_confidence_sections=low_confidence_sections or [],
    )


class TestComputeQualityVerdict:
    """compute_quality_verdict() returns correct verdict."""

    def test_pass_high_coverage_high_confidence(self) -> None:
        """>=75% coverage and >=0.70 confidence → PASS."""
        cov = _make_coverage(
            total_sections=14, populated_count=12,
            average_confidence=0.85,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "PASS"
        assert result["coverage_pct"] == pytest.approx(85.71, abs=0.1)

    def test_review_moderate_coverage(self) -> None:
        """50-74% coverage → REVIEW."""
        cov = _make_coverage(
            total_sections=14, populated_count=8,
            average_confidence=0.80,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "REVIEW"
        assert result["coverage_pct"] == pytest.approx(57.14, abs=0.1)

    def test_review_high_coverage_low_confidence(self) -> None:
        """>=75% coverage but <0.70 confidence → REVIEW."""
        cov = _make_coverage(
            total_sections=14, populated_count=12,
            average_confidence=0.55,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "REVIEW"

    def test_fail_low_coverage(self) -> None:
        """<50% coverage → FAIL."""
        cov = _make_coverage(
            total_sections=14, populated_count=5,
            average_confidence=0.90,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "FAIL"
        assert result["coverage_pct"] == pytest.approx(35.71, abs=0.1)

    def test_query_pct_calculated(self) -> None:
        """query_pct is answered/total * 100."""
        cov = _make_coverage(
            total_queries=100, answered_queries=75,
        )
        result = compute_quality_verdict(cov)
        assert result["query_pct"] == pytest.approx(75.0)

    def test_zero_sections_returns_fail(self) -> None:
        """0 total_sections → 0% coverage → FAIL."""
        cov = _make_coverage(
            total_sections=0, populated_count=0,
            total_queries=0, answered_queries=0,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "FAIL"
        assert result["coverage_pct"] == 0.0
        assert result["query_pct"] == 0.0

    def test_all_sections_populated(self) -> None:
        """14/14 populated with high confidence → PASS."""
        cov = _make_coverage(
            total_sections=14, populated_count=14, gap_count=0,
            average_confidence=0.90,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "PASS"
        assert result["coverage_pct"] == 100.0

    def test_boundary_75_percent(self) -> None:
        """Exactly at 75% boundary (10.5/14 rounds to ~75%)."""
        # 11/14 = 78.6% → PASS with sufficient confidence
        cov = _make_coverage(
            total_sections=14, populated_count=11,
            average_confidence=0.75,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "PASS"

    def test_boundary_50_percent(self) -> None:
        """Exactly at 50% boundary → REVIEW."""
        cov = _make_coverage(
            total_sections=14, populated_count=7,
            average_confidence=0.80,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "REVIEW"

    def test_boundary_below_50_percent(self) -> None:
        """Just below 50% → FAIL."""
        cov = _make_coverage(
            total_sections=14, populated_count=6,
            average_confidence=0.80,
        )
        result = compute_quality_verdict(cov)
        assert result["verdict"] == "FAIL"
