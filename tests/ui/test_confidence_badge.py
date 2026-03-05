"""Tests for confidence badge helpers (PTCV-95).

Pure-Python tests — no Streamlit dependency.
"""

from __future__ import annotations

from ptcv.ui.components.confidence_badge import (
    HIGH_THRESHOLD,
    REVIEW_THRESHOLD,
    confidence_color,
    confidence_icon,
    confidence_label,
    format_confidence,
)


class TestConfidenceColor:
    """Color mapping for confidence scores."""

    def test_high_confidence(self) -> None:
        assert confidence_color(0.90) == "green"

    def test_exactly_high_threshold(self) -> None:
        assert confidence_color(HIGH_THRESHOLD) == "green"

    def test_review_confidence(self) -> None:
        assert confidence_color(0.75) == "orange"

    def test_exactly_review_threshold(self) -> None:
        assert confidence_color(REVIEW_THRESHOLD) == "orange"

    def test_low_confidence(self) -> None:
        assert confidence_color(0.50) == "red"

    def test_zero(self) -> None:
        assert confidence_color(0.0) == "red"

    def test_perfect(self) -> None:
        assert confidence_color(1.0) == "green"


class TestConfidenceLabel:
    """Label text for confidence tiers."""

    def test_high(self) -> None:
        assert confidence_label(0.90) == "HIGH"

    def test_review(self) -> None:
        assert confidence_label(0.75) == "REVIEW"

    def test_low(self) -> None:
        assert confidence_label(0.50) == "LOW"

    def test_boundary_just_below_high(self) -> None:
        assert confidence_label(0.849) == "REVIEW"

    def test_boundary_just_below_review(self) -> None:
        assert confidence_label(0.699) == "LOW"


class TestConfidenceIcon:
    """Icon characters for Streamlit metrics."""

    def test_high_icon(self) -> None:
        assert confidence_icon(0.90) == "+"

    def test_review_icon(self) -> None:
        assert confidence_icon(0.75) == "~"

    def test_low_icon(self) -> None:
        assert confidence_icon(0.50) == "-"


class TestFormatConfidence:
    """Formatted confidence string."""

    def test_high_format(self) -> None:
        assert format_confidence(0.90) == "0.90 (HIGH)"

    def test_review_format(self) -> None:
        assert format_confidence(0.75) == "0.75 (REVIEW)"

    def test_low_format(self) -> None:
        assert format_confidence(0.50) == "0.50 (LOW)"

    def test_zero_format(self) -> None:
        assert format_confidence(0.0) == "0.00 (LOW)"

    def test_rounding(self) -> None:
        result = format_confidence(0.8567)
        assert result == "0.86 (HIGH)"
