"""Tests for content format auto-detection (PTCV-257).

Feature: Adaptive extraction strategy selection

  Scenario: Table content extracted despite text_long hint
  Scenario: YAML hint used when detection is ambiguous
  Scenario: No regression on well-matched queries
"""

from __future__ import annotations

import pytest

from ptcv.ich_parser.content_format_detector import (
    FormatDetection,
    detect_content_format,
    select_strategy,
    should_detect,
)


# -----------------------------------------------------------------------
# detect_content_format tests
# -----------------------------------------------------------------------


class TestDetectContentFormat:
    """Tests for format detection logic."""

    def test_pipe_table_detected(self) -> None:
        text = (
            "| Visit | Day 1 | Week 2 | Week 4 |\n"
            "|-------|-------|--------|--------|\n"
            "| Labs  | X     | X      |        |\n"
            "| ECG   |       | X      | X      |\n"
        )
        result = detect_content_format(text)
        assert result.detected_type == "table"
        assert result.confidence >= 0.70

    def test_aligned_table_detected(self) -> None:
        text = (
            "Assessment          Visit 1    Visit 2    Visit 3\n"
            "Physical Exam       X          X          X\n"
            "Blood Chemistry     X                     X\n"
            "Urinalysis          X          X          X\n"
        )
        result = detect_content_format(text)
        assert result.detected_type == "table"
        assert result.confidence >= 0.55

    def test_bulleted_list_detected(self) -> None:
        text = (
            "Inclusion Criteria:\n"
            "- Age >= 18 years\n"
            "- Confirmed diagnosis of condition X\n"
            "- ECOG performance status 0-2\n"
            "- Adequate organ function\n"
            "- Written informed consent\n"
        )
        result = detect_content_format(text)
        assert result.detected_type == "list"
        assert result.confidence >= 0.60

    def test_numbered_list_detected(self) -> None:
        text = (
            "1. Patient must be at least 18 years old\n"
            "2. Confirmed histological diagnosis\n"
            "3. ECOG status 0-1\n"
            "4. Life expectancy > 6 months\n"
        )
        result = detect_content_format(text)
        assert result.detected_type == "list"

    def test_short_text_detected(self) -> None:
        text = "Protocol ABC-123 Version 2.0"
        result = detect_content_format(text)
        assert result.detected_type == "text_short"

    def test_prose_detected_as_text_long(self) -> None:
        text = (
            "This is a randomized, double-blind, placebo-controlled "
            "study designed to evaluate the efficacy and safety of "
            "drug X in patients with condition Y. The study will "
            "enroll approximately 500 participants across 50 sites "
            "in North America and Europe. Participants will be "
            "randomized 2:1 to receive drug X or placebo for "
            "24 weeks, followed by a 12-week follow-up period."
        )
        result = detect_content_format(text)
        assert result.detected_type == "text_long"

    def test_empty_text(self) -> None:
        result = detect_content_format("")
        assert result.detected_type == "text_long"
        assert result.confidence == 0.0

    def test_whitespace_only(self) -> None:
        result = detect_content_format("   \n\n  ")
        assert result.detected_type == "text_long"
        assert result.confidence == 0.0

    def test_few_list_items_not_false_positive(self) -> None:
        """Prose with 1-2 occasional numbered refs shouldn't be 'list'."""
        text = (
            "The study drug is administered orally once daily. "
            "See 1. dosing schedule in the appendix for details. "
            "Patients should take the drug with food. Treatment "
            "continues until disease progression or unacceptable "
            "toxicity is observed in the patient population."
        )
        result = detect_content_format(text)
        assert result.detected_type != "list"


# -----------------------------------------------------------------------
# select_strategy tests
# -----------------------------------------------------------------------


class TestSelectStrategy:
    """Tests for strategy selection logic."""

    def test_table_overrides_text_long_hint(self) -> None:
        """Scenario: Table content extracted despite text_long hint."""
        detected = FormatDetection(
            detected_type="table",
            confidence=0.85,
            signals={"pipe_lines": 4},
        )
        result = select_strategy("text_long", detected)
        assert result == "table"

    def test_yaml_hint_used_when_ambiguous(self) -> None:
        """Scenario: YAML hint used when detection is ambiguous."""
        detected = FormatDetection(
            detected_type="list",
            confidence=0.40,  # Below threshold
            signals={"list_items": 2},
        )
        result = select_strategy("text_long", detected)
        assert result == "text_long"

    def test_no_regression_when_types_match(self) -> None:
        """Scenario: No regression on well-matched queries."""
        detected = FormatDetection(
            detected_type="text_long",
            confidence=0.50,
            signals={},
        )
        result = select_strategy("text_long", detected)
        assert result == "text_long"

    def test_identifier_never_overridden(self) -> None:
        """Regex-specific types should not be overridden."""
        detected = FormatDetection(
            detected_type="list",
            confidence=0.95,
            signals={},
        )
        result = select_strategy("identifier", detected)
        assert result == "identifier"

    def test_date_never_overridden(self) -> None:
        detected = FormatDetection(
            detected_type="table",
            confidence=0.95,
            signals={},
        )
        result = select_strategy("date", detected)
        assert result == "date"

    def test_numeric_never_overridden(self) -> None:
        detected = FormatDetection(
            detected_type="text_long",
            confidence=0.90,
            signals={},
        )
        result = select_strategy("numeric", detected)
        assert result == "numeric"

    def test_enum_never_overridden(self) -> None:
        detected = FormatDetection(
            detected_type="list",
            confidence=0.90,
            signals={},
        )
        result = select_strategy("enum", detected)
        assert result == "enum"

    def test_list_overrides_text_long(self) -> None:
        detected = FormatDetection(
            detected_type="list",
            confidence=0.80,
            signals={"list_items": 6},
        )
        result = select_strategy("text_long", detected)
        assert result == "list"

    def test_custom_threshold(self) -> None:
        detected = FormatDetection(
            detected_type="table",
            confidence=0.60,
            signals={},
        )
        # Default threshold (0.65) → should NOT override
        assert select_strategy("text_long", detected) == "text_long"
        # Lower threshold → should override
        assert select_strategy(
            "text_long", detected, confidence_threshold=0.55,
        ) == "table"


# -----------------------------------------------------------------------
# should_detect tests
# -----------------------------------------------------------------------


class TestShouldDetect:
    """Tests for skip-detection logic."""

    def test_text_long_should_detect(self) -> None:
        assert should_detect("text_long") is True

    def test_text_short_should_detect(self) -> None:
        assert should_detect("text_short") is True

    def test_list_should_detect(self) -> None:
        assert should_detect("list") is True

    def test_table_should_detect(self) -> None:
        assert should_detect("table") is True

    def test_statement_should_detect(self) -> None:
        assert should_detect("statement") is True

    def test_identifier_skips(self) -> None:
        assert should_detect("identifier") is False

    def test_date_skips(self) -> None:
        assert should_detect("date") is False

    def test_numeric_skips(self) -> None:
        assert should_detect("numeric") is False

    def test_enum_skips(self) -> None:
        assert should_detect("enum") is False
