"""Shared confidence scoring helpers for UI components (PTCV-95).

Pure-Python module — no Streamlit dependency, fully testable.
Provides color, label, and formatting for confidence scores
used across query pipeline, benchmark, and refinement panels.
"""

from __future__ import annotations

# Thresholds aligned with SectionMatcher (section_matcher.py).
HIGH_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.70


def confidence_color(score: float) -> str:
    """Return a color name for the given confidence score.

    Args:
        score: Confidence value between 0.0 and 1.0.

    Returns:
        ``'green'`` for high, ``'orange'`` for review,
        ``'red'`` for low confidence.
    """
    if score >= HIGH_THRESHOLD:
        return "green"
    if score >= REVIEW_THRESHOLD:
        return "orange"
    return "red"


def confidence_label(score: float) -> str:
    """Return a text label for the confidence tier.

    Args:
        score: Confidence value between 0.0 and 1.0.

    Returns:
        ``'HIGH'``, ``'REVIEW'``, or ``'LOW'``.
    """
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "LOW"


def confidence_icon(score: float) -> str:
    """Return a status icon character for Streamlit metrics.

    Args:
        score: Confidence value between 0.0 and 1.0.

    Returns:
        Checkmark, warning, or cross icon.
    """
    if score >= HIGH_THRESHOLD:
        return "+"
    if score >= REVIEW_THRESHOLD:
        return "~"
    return "-"


def format_confidence(score: float) -> str:
    """Return a formatted confidence string.

    Example: ``'0.85 (HIGH)'``.

    Args:
        score: Confidence value between 0.0 and 1.0.

    Returns:
        Formatted string with score and tier label.
    """
    return f"{score:.2f} ({confidence_label(score)})"
