"""Tests for refinement panel UI helpers (PTCV-95).

Pure-Python tests — no Streamlit dependency.
"""

from __future__ import annotations

from types import SimpleNamespace

from ptcv.ui.components.refinement_panel import (
    BIAS_FLAG_THRESHOLD,
    format_calibration_table,
    format_synonym_table,
    format_trend_summary,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _band(
    range_low: float = 0.0,
    range_high: float = 0.10,
    count: int = 5,
    predicted_avg: float = 0.05,
    actual_accuracy: float = 0.04,
    bias: float = 0.01,
) -> SimpleNamespace:
    return SimpleNamespace(
        range_low=range_low,
        range_high=range_high,
        count=count,
        predicted_avg=predicted_avg,
        actual_accuracy=actual_accuracy,
        bias=bias,
    )


def _calibration_report(
    bands: list | None = None,
    total_entries: int = 0,
    overall_accuracy: float = 0.0,
    bias_detected: bool = False,
    suggested_threshold_adjustment: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        bands=bands or [],
        total_entries=total_entries,
        overall_accuracy=overall_accuracy,
        bias_detected=bias_detected,
        suggested_threshold_adjustment=suggested_threshold_adjustment,
    )


def _trend_report(
    total_corrections: int = 0,
    total_calibration_entries: int = 0,
    common_corrections: list | None = None,
    new_synonyms: list | None = None,
    accuracy_by_month: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        total_corrections=total_corrections,
        total_calibration_entries=total_calibration_entries,
        common_corrections=common_corrections or [],
        new_synonyms=new_synonyms or [],
        accuracy_by_month=accuracy_by_month or [],
    )


# ---------------------------------------------------------------------------
# TestFormatCalibrationTable
# ---------------------------------------------------------------------------


class TestFormatCalibrationTable:
    """Tests for format_calibration_table()."""

    def test_empty(self) -> None:
        report = _calibration_report()
        assert format_calibration_table(report) == []

    def test_single_band(self) -> None:
        band = _band(0.70, 0.80, 10, 0.75, 0.72, 0.03)
        report = _calibration_report(bands=[band])
        rows = format_calibration_table(report)
        assert len(rows) == 1
        assert rows[0]["range"] == "0.70-0.80"
        assert rows[0]["count"] == 10
        assert rows[0]["flagged"] is False

    def test_bias_flagging(self) -> None:
        """Bands with |bias| > 0.15 should be flagged."""
        band = _band(bias=0.20)
        report = _calibration_report(bands=[band])
        rows = format_calibration_table(report)
        assert rows[0]["flagged"] is True

    def test_negative_bias_flagged(self) -> None:
        band = _band(bias=-0.18)
        report = _calibration_report(bands=[band])
        rows = format_calibration_table(report)
        assert rows[0]["flagged"] is True

    def test_exactly_at_threshold_not_flagged(self) -> None:
        band = _band(bias=BIAS_FLAG_THRESHOLD)
        report = _calibration_report(bands=[band])
        rows = format_calibration_table(report)
        assert rows[0]["flagged"] is False

    def test_multiple_bands(self) -> None:
        bands = [
            _band(0.0, 0.10, 3, 0.05, 0.03, 0.02),
            _band(0.70, 0.80, 10, 0.75, 0.72, 0.03),
            _band(0.90, 1.00, 8, 0.95, 0.78, 0.17),
        ]
        report = _calibration_report(bands=bands)
        rows = format_calibration_table(report)
        assert len(rows) == 3
        assert rows[2]["flagged"] is True  # 0.17 > 0.15


# ---------------------------------------------------------------------------
# TestFormatTrendSummary
# ---------------------------------------------------------------------------


class TestFormatTrendSummary:
    """Tests for format_trend_summary()."""

    def test_empty(self) -> None:
        report = _trend_report()
        summary = format_trend_summary(report)
        assert summary["total_corrections"] == 0
        assert summary["common_corrections"] == []
        assert summary["synonyms"] == []
        assert summary["accuracy_by_month"] == []

    def test_with_data(self) -> None:
        report = _trend_report(
            total_corrections=15,
            total_calibration_entries=50,
            common_corrections=[
                ("inclusion criteria", "B.5", 5),
                ("study design", "B.4", 3),
            ],
            new_synonyms=[
                ("eligibility", "B.5"),
            ],
            accuracy_by_month=[
                ("2026-03", 0.85),
                ("2026-02", 0.78),
            ],
        )
        summary = format_trend_summary(report)
        assert summary["total_corrections"] == 15
        assert summary["total_calibration"] == 50
        assert len(summary["common_corrections"]) == 2
        assert summary["common_corrections"][0]["header"] == "inclusion criteria"
        assert summary["common_corrections"][0]["count"] == 5
        assert len(summary["synonyms"]) == 1
        assert summary["synonyms"][0]["section_code"] == "B.5"
        assert len(summary["accuracy_by_month"]) == 2
        assert summary["accuracy_by_month"][0]["month"] == "2026-03"

    def test_monthly_accuracy_formatted_as_percent(self) -> None:
        report = _trend_report(
            accuracy_by_month=[("2026-01", 0.923)],
        )
        summary = format_trend_summary(report)
        assert summary["accuracy_by_month"][0]["accuracy"] == "92.3%"


# ---------------------------------------------------------------------------
# TestFormatSynonymTable
# ---------------------------------------------------------------------------


class TestFormatSynonymTable:
    """Tests for format_synonym_table()."""

    def test_empty(self) -> None:
        assert format_synonym_table({}) == []

    def test_with_entries(self) -> None:
        boosts = {
            "inclusion criteria": "B.5",
            "assessment of efficacy": "B.7",
        }
        rows = format_synonym_table(boosts)
        assert len(rows) == 2
        # Sorted by header
        assert rows[0]["header"] == "assessment of efficacy"
        assert rows[0]["section_code"] == "B.7"
        assert rows[1]["header"] == "inclusion criteria"

    def test_single_entry(self) -> None:
        rows = format_synonym_table({"study design": "B.4"})
        assert len(rows) == 1
        assert rows[0]["header"] == "study design"
