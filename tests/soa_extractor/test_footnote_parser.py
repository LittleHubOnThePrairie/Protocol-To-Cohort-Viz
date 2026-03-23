"""Tests for SoA footnote parser (PTCV-218).

Tests footnote marker extraction, cross-referencing, and
various marker formats (caret, unicode, symbol, parenthetical).
"""

from __future__ import annotations

import pytest

from ptcv.soa_extractor.footnote_parser import (
    FootnoteMarker,
    ParsedAssessment,
    Footnote,
    FootnoteReport,
    parse_assessment_name,
    extract_footnotes,
    parse_and_crossref,
)


class TestParseAssessmentName:
    """Tests for parse_assessment_name function."""

    def test_no_markers(self):
        """Test name without markers is returned clean."""
        result = parse_assessment_name("Physical Exam")
        assert result.clean_name == "Physical Exam"
        assert result.original_name == "Physical Exam"
        assert not result.has_footnotes

    def test_caret_marker(self):
        """Test caret-style footnote: Hematology^2."""
        result = parse_assessment_name("Hematology^2")
        assert result.clean_name == "Hematology"
        assert result.has_footnotes
        assert len(result.markers) == 1
        assert result.markers[0].marker_id == "2"

    def test_unicode_superscript(self):
        """Test unicode superscript: ECG¹."""
        result = parse_assessment_name("ECG¹")
        assert result.clean_name == "ECG"
        assert result.markers[0].marker_id == "1"

    def test_multiple_superscripts(self):
        """Test multiple unicode superscripts: Labs¹²."""
        result = parse_assessment_name("Labs¹²")
        assert result.clean_name == "Labs"
        assert len(result.markers) >= 1

    def test_asterisk_marker(self):
        """Test asterisk marker: Vitals*."""
        result = parse_assessment_name("Vitals*")
        assert result.clean_name == "Vitals"
        assert result.has_footnotes
        assert result.markers[0].marker_id == "*"

    def test_dagger_marker(self):
        """Test dagger marker: Coagulation†."""
        result = parse_assessment_name("Coagulation†")
        assert result.clean_name == "Coagulation"
        assert result.markers[0].marker_id == "†"

    def test_see_note_style(self):
        """Test parenthetical: Labs (see note 3)."""
        result = parse_assessment_name("Labs (see note 3)")
        assert result.clean_name == "Labs"
        assert result.has_footnotes
        assert result.markers[0].marker_id == "3"

    def test_multiple_markers(self):
        """Test multiple markers on one name."""
        result = parse_assessment_name("Hematology^2^5")
        assert result.clean_name == "Hematology"
        assert len(result.markers) == 2

    def test_preserves_original(self):
        """Test original name is preserved."""
        result = parse_assessment_name("ECG^13")
        assert result.original_name == "ECG^13"


class TestExtractFootnotes:
    """Tests for extract_footnotes function."""

    def test_numbered_footnotes(self):
        """Test numbered footnote lines: 1. Text..."""
        text = """
1. Perform at screening only
2. CBC with differential required
3. Only if clinically indicated
"""
        result = extract_footnotes(text)
        assert len(result) == 3
        assert result[0].footnote_id == "1"
        assert "screening" in result[0].text
        assert result[1].footnote_id == "2"
        assert result[2].footnote_id == "3"

    def test_symbol_footnotes(self):
        """Test symbol footnote lines: * Text..."""
        text = "* Perform fasting\n† Only at baseline"
        result = extract_footnotes(text)
        assert len(result) == 2
        assert result[0].footnote_id == "*"
        assert result[1].footnote_id == "†"

    def test_parenthetical_numbered(self):
        """Test parenthetical numbers: 1) Text..."""
        text = "1) First note\n2) Second note"
        result = extract_footnotes(text)
        assert len(result) == 2

    def test_no_footnotes(self):
        """Test text without footnote patterns."""
        text = "This is just regular paragraph text."
        result = extract_footnotes(text)
        assert result == []

    def test_empty_text(self):
        """Test empty input."""
        assert extract_footnotes("") == []

    def test_letter_footnotes(self):
        """Test letter-style footnotes: a. Text..."""
        text = "a. First\nb. Second"
        result = extract_footnotes(text)
        assert len(result) == 2
        assert result[0].footnote_id == "a"


class TestParseAndCrossref:
    """Tests for parse_and_crossref function."""

    def test_all_markers_resolved(self):
        """Test all markers find matching footnotes."""
        names = ["Hematology^1", "ECG^2", "Vitals"]
        footnotes = "1. CBC with differential\n2. 12-lead ECG required"

        report = parse_and_crossref(names, footnotes)

        assert len(report.parsed_assessments) == 3
        assert len(report.footnotes) == 2
        assert report.unresolved_markers == []

    def test_unresolved_markers_detected(self):
        """Test unresolved markers are reported."""
        names = ["Hematology^1", "ECG^3"]
        footnotes = "1. CBC required"  # Note 3 is missing

        report = parse_and_crossref(names, footnotes)

        assert "3" in report.unresolved_markers
        assert "1" not in report.unresolved_markers

    def test_no_footnotes_text(self):
        """Test with markers but no footnote text."""
        names = ["Exam^1", "Labs^2"]
        report = parse_and_crossref(names)

        assert len(report.unresolved_markers) == 2

    def test_no_markers(self):
        """Test with clean names and no footnotes."""
        names = ["Physical Exam", "Vitals"]
        report = parse_and_crossref(names)

        assert report.unresolved_markers == []
        assert all(not p.has_footnotes for p in report.parsed_assessments)


class TestTableContinuity:
    """Tests for continuation signal detection in TableDiscovery."""

    def test_continuation_text_in_header(self):
        """Test 'continued' text in header triggers merge."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        prev = {
            "page": 51,
            "header": ["Assessment", "V1", "V2", "V3", "V4"],
            "rows": [["Exam", "X", "", "X", ""]],
        }
        cont = {
            "page": 52,
            "header": [
                "Table 11a (continued)", "V1", "V2", "V3", "V4",
            ],
            "rows": [["Labs", "X", "X", "", "X"]],
        }

        assert TableDiscovery._is_continuation_signal(prev, cont)

    def test_contd_abbreviation(self):
        """Test cont'd abbreviation detected."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        prev = {"page": 1, "header": ["A", "B", "C", "D", "E"], "rows": []}
        cont = {
            "page": 2,
            "header": ["(cont'd)", "B", "C", "D", "E"],
            "rows": [],
        }

        assert TableDiscovery._is_continuation_signal(prev, cont)

    def test_same_column_count_is_signal(self):
        """Test same high column count on consecutive page."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        prev = {
            "page": 10,
            "header": ["A", "B", "C", "D", "E", "F"],
            "rows": [],
        }
        cont = {
            "page": 11,
            "header": ["G", "H", "I", "J", "K", "L"],
            "rows": [],
        }

        # 6 columns on consecutive pages → continuation signal
        assert TableDiscovery._is_continuation_signal(prev, cont)

    def test_different_column_count_not_signal(self):
        """Test different column counts don't trigger."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        prev = {
            "page": 10,
            "header": ["A", "B"],
            "rows": [],
        }
        cont = {
            "page": 11,
            "header": ["C", "D", "E", "F"],
            "rows": [],
        }

        assert not TableDiscovery._is_continuation_signal(prev, cont)

    def test_low_column_count_not_signal(self):
        """Test same but low column count doesn't trigger."""
        from ptcv.soa_extractor.table_discovery import TableDiscovery

        prev = {"page": 1, "header": ["A", "B"], "rows": []}
        cont = {"page": 2, "header": ["C", "D"], "rows": []}

        # Only 2 columns — not enough for same-count signal
        assert not TableDiscovery._is_continuation_signal(prev, cont)
