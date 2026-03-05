"""Tests for benchmark viewer UI helpers (PTCV-95).

Pure-Python tests — no Streamlit dependency.
"""

from __future__ import annotations

from types import SimpleNamespace

from ptcv.ui.components.benchmark_viewer import (
    format_metrics_comparison,
    format_recommendation,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _metrics(
    coverage_pct: float = 50.0,
    avg_confidence: float = 0.75,
    avg_span_length: float = 500.0,
    gap_rate: float = 0.20,
    section_count: int = 8,
    high_confidence_pct: float = 40.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        coverage_pct=coverage_pct,
        avg_confidence=avg_confidence,
        avg_span_length=avg_span_length,
        gap_rate=gap_rate,
        section_count=section_count,
        high_confidence_pct=high_confidence_pct,
    )


# ---------------------------------------------------------------------------
# TestFormatMetricsComparison
# ---------------------------------------------------------------------------


class TestFormatMetricsComparison:
    """Tests for format_metrics_comparison()."""

    def test_positive_delta(self) -> None:
        tf = _metrics(coverage_pct=40.0)
        qd = _metrics(coverage_pct=60.0)
        rows = format_metrics_comparison(tf, qd)
        coverage_row = rows[0]
        assert coverage_row["metric"] == "Coverage (%)"
        assert coverage_row["text_first"] == "40.0"
        assert coverage_row["query_driven"] == "60.0"
        assert coverage_row["delta"] == "+20.0"

    def test_negative_delta(self) -> None:
        tf = _metrics(avg_confidence=0.90)
        qd = _metrics(avg_confidence=0.70)
        rows = format_metrics_comparison(tf, qd)
        conf_row = rows[1]
        assert conf_row["metric"] == "Avg Confidence"
        assert "-0.200" in conf_row["delta"]

    def test_equal_metrics(self) -> None:
        tf = _metrics()
        qd = _metrics()
        rows = format_metrics_comparison(tf, qd)
        for row in rows:
            # Deltas should be zero (with sign)
            delta = row["delta"]
            assert "0" in delta

    def test_none_metrics_returns_empty(self) -> None:
        assert format_metrics_comparison(None, None) == []
        assert format_metrics_comparison(_metrics(), None) == []
        assert format_metrics_comparison(None, _metrics()) == []

    def test_six_rows_returned(self) -> None:
        rows = format_metrics_comparison(_metrics(), _metrics())
        assert len(rows) == 6


# ---------------------------------------------------------------------------
# TestFormatRecommendation
# ---------------------------------------------------------------------------


class TestFormatRecommendation:
    """Tests for format_recommendation()."""

    def test_replace(self) -> None:
        badge, text = format_recommendation(
            "replace", "Query-driven outperforms"
        )
        assert badge == "success"
        assert "Replace" in text

    def test_supplement(self) -> None:
        badge, text = format_recommendation(
            "supplement", "Mixed results"
        )
        assert badge == "warning"
        assert "Supplement" in text

    def test_keep_text_first(self) -> None:
        badge, text = format_recommendation(
            "keep_text_first", "Text-first is better"
        )
        assert badge == "info"
        assert "Keep" in text

    def test_unknown_defaults_to_info(self) -> None:
        badge, text = format_recommendation(
            "unknown_value", "Something"
        )
        assert badge == "info"
