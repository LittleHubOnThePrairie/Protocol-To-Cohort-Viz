"""Tests for protocol TOC and section header extraction (PTCV-89).

Feature: Protocol TOC and section header extraction

  Scenario: TOC extracted from protocol with explicit table of contents
  Scenario: Section headers detected in document body
  Scenario: Content spans map sections to their text
  Scenario: Protocol without TOC falls back to header detection
  Scenario: Multi-level section hierarchy preserved
  Scenario: Extraction works across the qualifying protocol corpus
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.toc_extractor import (
    ProtocolIndex,
    SectionHeader,
    TOCEntry,
    _LIST_ENTRY_RE,
    _LIST_OF_HEADER_RE,
    _clean_title,
    _count_toc_lines,
    _detect_body_headers,
    _is_page_boilerplate,
    _is_toc_page,
    _normalise_number,
    _parse_toc_page,
    _resolve_content_spans,
    _resolve_toc_pages,
    _section_level,
    extract_protocol_index,
    strip_toc_lines,
)


# -----------------------------------------------------------------------
# Helper fixtures
# -----------------------------------------------------------------------

@pytest.fixture()
def toc_page_text() -> str:
    """A realistic TOC page."""
    return textwrap.dedent("""\
        TABLE OF CONTENTS
        1   PROTOCOL SUMMARY .......................... 9
        1.1 Synopsis .................................. 9
        1.2 Schema .................................... 13
        2   INTRODUCTION .............................. 16
        2.1 Study Rationale ........................... 16
        2.2 Background ................................ 17
        3   OBJECTIVES AND ENDPOINTS .................. 22
        4   STUDY DESIGN .............................. 23
        4.1 Overall Design ............................ 23
        4.2 Number of Participants .................... 25
        5   STUDY POPULATION .......................... 28
        5.1 Inclusion Criteria ........................ 28
        5.2 Exclusion Criteria ........................ 29
    """)


@pytest.fixture()
def body_page_text() -> str:
    """Body text with section headers."""
    return textwrap.dedent("""\
        1   PROTOCOL SUMMARY
        This section describes the protocol summary.
        The study is a randomised controlled trial.

        1.1 Synopsis
        The study evaluates drug X in population Y.

        2   INTRODUCTION
        This section introduces the study.

        2.1 Study Rationale
        Drug X has shown efficacy in preclinical models.
    """)


# -----------------------------------------------------------------------
# Unit tests: helper functions
# -----------------------------------------------------------------------

class TestSectionLevel:
    def test_level_1(self):
        assert _section_level("1") == 1

    def test_level_2(self):
        assert _section_level("1.1") == 2

    def test_level_3(self):
        assert _section_level("1.1.1") == 3

    def test_level_4(self):
        assert _section_level("10.2.3.1") == 4


class TestNormaliseNumber:
    def test_trailing_dot(self):
        assert _normalise_number("3.2.") == "3.2"

    def test_trailing_dot_zero(self):
        assert _normalise_number("1.0") == "1"

    def test_no_change(self):
        assert _normalise_number("10.2.3") == "10.2.3"

    def test_ten_stays(self):
        assert _normalise_number("10") == "10"


class TestCleanTitle:
    def test_dot_leaders_stripped(self):
        assert _clean_title("Synopsis ........... 9") == "Synopsis"

    def test_whitespace_collapsed(self):
        assert _clean_title("Study   Rationale") == "Study Rationale"

    def test_clean_title_unchanged(self):
        assert _clean_title("Inclusion Criteria") == "Inclusion Criteria"


# -----------------------------------------------------------------------
# Scenario: TOC extracted from protocol with explicit table of contents
# -----------------------------------------------------------------------

class TestTOCExtraction:
    """Given a protocol PDF with a 'Table of Contents' page,
    When the TOC extractor processes the PDF,
    Then toc_entries contains all listed sections with titles and page refs,
    And section hierarchy (level) is correctly assigned.
    """

    def test_is_toc_page_positive(self, toc_page_text: str):
        assert _is_toc_page(toc_page_text) is True

    def test_is_toc_page_negative(self, body_page_text: str):
        assert _is_toc_page(body_page_text) is False

    def test_count_toc_lines(self, toc_page_text: str):
        count = _count_toc_lines(toc_page_text)
        assert count >= 10  # 13 TOC entries in fixture

    def test_parse_toc_entries_count(self, toc_page_text: str):
        entries = _parse_toc_page(toc_page_text)
        assert len(entries) == 13

    def test_parse_toc_entries_fields(self, toc_page_text: str):
        entries = _parse_toc_page(toc_page_text)
        first = entries[0]
        assert first.number == "1"
        assert first.title == "PROTOCOL SUMMARY"
        assert first.page_ref == 9
        assert first.level == 1

    def test_parse_toc_hierarchy(self, toc_page_text: str):
        entries = _parse_toc_page(toc_page_text)
        levels = {e.number: e.level for e in entries}
        assert levels["1"] == 1
        assert levels["1.1"] == 2
        assert levels["2.1"] == 2
        assert levels["4.2"] == 2

    def test_toc_no_duplicates(self, toc_page_text: str):
        entries = _parse_toc_page(toc_page_text)
        numbers = [e.number for e in entries]
        assert len(numbers) == len(set(numbers))


# -----------------------------------------------------------------------
# Scenario: Section headers detected in document body
# -----------------------------------------------------------------------

class TestBodyHeaderDetection:
    """Given a protocol PDF,
    When the header extractor scans the document body,
    Then section_headers contains detected headers with hierarchy levels,
    And headers match TOC entries where a TOC exists.
    """

    def test_detect_headers(self, body_page_text: str):
        page_texts = [(10, body_page_text)]
        headers = _detect_body_headers(page_texts, body_page_text)
        numbers = [h.number for h in headers]
        assert "1" in numbers
        assert "1.1" in numbers
        assert "2" in numbers
        assert "2.1" in numbers

    def test_header_hierarchy(self, body_page_text: str):
        page_texts = [(10, body_page_text)]
        headers = _detect_body_headers(page_texts, body_page_text)
        levels = {h.number: h.level for h in headers}
        assert levels["1"] == 1
        assert levels["1.1"] == 2
        assert levels["2"] == 1

    def test_skips_toc_pages(self, body_page_text: str):
        # If page 10 is in toc_pages, no headers detected
        page_texts = [(10, body_page_text)]
        headers = _detect_body_headers(
            page_texts, body_page_text, toc_pages=[10],
        )
        assert len(headers) == 0

    def test_skips_front_matter(self, body_page_text: str):
        # If body_start_page is 11 and text is on page 10, skip it
        page_texts = [(10, body_page_text)]
        headers = _detect_body_headers(
            page_texts, body_page_text, body_start_page=11,
        )
        assert len(headers) == 0

    def test_filters_date_lines(self):
        text = "24 January 2023\n15 March 2021\n1 INTRODUCTION"
        page_texts = [(10, text)]
        headers = _detect_body_headers(page_texts, text)
        numbers = [h.number for h in headers]
        assert "24" not in numbers
        assert "15" not in numbers
        assert "1" in numbers

    def test_filters_large_section_numbers(self):
        text = "1800 Century Park East Suite 600\n1 INTRODUCTION"
        page_texts = [(10, text)]
        headers = _detect_body_headers(page_texts, text)
        numbers = [h.number for h in headers]
        assert "1800" not in numbers
        assert "1" in numbers

    def test_toc_number_filtering(self):
        text = "1 INTRODUCTION\n99 Stray Number Line"
        page_texts = [(10, text)]
        # With toc_numbers, only allow leading components from TOC
        headers = _detect_body_headers(
            page_texts, text, toc_numbers={"1", "2", "3"},
        )
        numbers = [h.number for h in headers]
        assert "1" in numbers
        assert "99" not in numbers

    def test_deduplicates_across_pages(self):
        text1 = "1 INTRODUCTION\n2 METHODS"
        text2 = "1 INTRODUCTION\n3 RESULTS"
        full = text1 + "\n" + text2
        page_texts = [(10, text1), (20, text2)]
        headers = _detect_body_headers(page_texts, full)
        # Section "1" appears on pages 10 and 20, but dedup keeps first
        ones = [h for h in headers if h.number == "1"]
        assert len(ones) == 1
        assert ones[0].page == 10


# -----------------------------------------------------------------------
# Scenario: Content spans map sections to their text
# -----------------------------------------------------------------------

class TestCharOffsetAvoidsTOC:
    """PTCV-109: char_offset must point to body, not TOC (PTCV-109).

    When the same heading text appears in both the TOC and the body,
    the char_offset should resolve to the body occurrence so that
    content_spans contain actual section content.
    """

    def test_offset_points_to_body_not_toc(self) -> None:
        """Headers offset past the TOC region."""
        toc_text = (
            "TABLE OF CONTENTS\n"
            "1 INTRODUCTION .............. 5\n"
            "2 METHODS ................... 10\n"
        )
        body_text = (
            "1 INTRODUCTION\n"
            "This section introduces the study.\n\n"
            "2 METHODS\n"
            "The study uses a randomised design.\n"
        )
        full_text = toc_text + "\n" + body_text
        page_offsets = {1: 0, 5: len(toc_text) + 1}

        page_texts = [(1, toc_text), (5, body_text)]
        headers = _detect_body_headers(
            page_texts,
            full_text,
            toc_pages=[1],
            body_start_page=2,
            page_offsets=page_offsets,
        )

        for h in headers:
            assert h.char_offset >= len(toc_text), (
                f"Header {h.number} offset {h.char_offset} "
                f"points into TOC area (< {len(toc_text)})"
            )

    def test_content_spans_nonempty_with_page_offsets(self) -> None:
        """Content spans should have real content when offsets are correct."""
        toc_text = (
            "TABLE OF CONTENTS\n"
            "1 SUMMARY .............. 3\n"
            "2 BACKGROUND ........... 5\n"
        )
        body_text = (
            "1 SUMMARY\n"
            "This protocol evaluates drug X for condition Y.\n\n"
            "2 BACKGROUND\n"
            "Drug X has shown efficacy in preclinical studies.\n"
        )
        full_text = toc_text + "\n" + body_text
        page_offsets = {1: 0, 3: len(toc_text) + 1}

        page_texts = [(1, toc_text), (3, body_text)]
        headers = _detect_body_headers(
            page_texts,
            full_text,
            toc_pages=[1],
            body_start_page=2,
            page_offsets=page_offsets,
        )
        spans = _resolve_content_spans(headers, full_text)

        assert len(spans) >= 2
        for num, content in spans.items():
            assert len(content) > 0, (
                f"Section {num} has empty content span"
            )
        assert "drug x" in spans.get("1", "").lower()
        assert "preclinical" in spans.get("2", "").lower()


class TestContentSpans:
    """Given an extracted ProtocolIndex,
    When content_spans is queried for a specific section,
    Then the returned text starts at the section header
    And ends before the next section,
    And no content is duplicated across adjacent sections.
    """

    def test_spans_cover_all_headers(self):
        headers = [
            SectionHeader("1", "INTRO", 1, 10, 0),
            SectionHeader("2", "METHODS", 1, 15, 100),
            SectionHeader("3", "RESULTS", 1, 20, 200),
        ]
        full_text = "x" * 300
        spans = _resolve_content_spans(headers, full_text)
        assert "1" in spans
        assert "2" in spans
        assert "3" in spans

    def test_no_overlap(self):
        headers = [
            SectionHeader("1", "INTRO", 1, 10, 0),
            SectionHeader("2", "METHODS", 1, 15, 100),
            SectionHeader("3", "RESULTS", 1, 20, 200),
        ]
        full_text = "a" * 50 + "b" * 50 + "c" * 50 + "d" * 150
        spans = _resolve_content_spans(headers, full_text)
        # Span 1 ends where span 2 starts
        total_len = sum(len(v) for v in spans.values())
        # No duplicated content (total <= full_text length)
        assert total_len <= len(full_text)

    def test_empty_headers(self):
        spans = _resolve_content_spans([], "some text")
        assert spans == {}

    def test_unresolved_offsets_skipped(self):
        headers = [
            SectionHeader("1", "INTRO", 1, 10, -1),  # unresolved
        ]
        spans = _resolve_content_spans(headers, "some text")
        assert spans == {}


# -----------------------------------------------------------------------
# Scenario: Protocol without TOC falls back to header detection
# -----------------------------------------------------------------------

class TestNoTOCFallback:
    """Given a protocol PDF with no explicit TOC page,
    When the TOC extractor processes the PDF,
    Then section_headers are still detected from body content,
    And a warning is logged that no TOC was found.
    """

    def test_no_toc_synthesises_entries(self, body_page_text: str):
        """Simulate a PDF with no TOC but with body headers."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = body_page_text
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("ptcv.ich_parser.toc_extractor.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = mock_pdf
            with patch("pathlib.Path.exists", return_value=True):
                idx = extract_protocol_index("/fake/protocol.pdf")

        assert idx.toc_found is False
        assert len(idx.toc_entries) > 0
        assert len(idx.section_headers) > 0

    def test_no_toc_logs_warning(self, body_page_text: str, caplog):
        """A warning should be logged when no TOC is found."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = body_page_text
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("ptcv.ich_parser.toc_extractor.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = mock_pdf
            with patch("pathlib.Path.exists", return_value=True):
                import logging
                with caplog.at_level(logging.WARNING):
                    extract_protocol_index("/fake/protocol.pdf")

        assert any("No Table of Contents" in r.message for r in caplog.records)


# -----------------------------------------------------------------------
# Scenario: Multi-level section hierarchy preserved
# -----------------------------------------------------------------------

class TestMultiLevelHierarchy:
    """Given a protocol with sections 1, 1.1, 1.1.1, 1.2,
    When the TOC extractor processes the PDF,
    Then levels are correctly assigned (1=H1, 1.1=H2, 1.1.1=H3),
    And parent-child relationships are navigable.
    """

    def test_hierarchy_levels(self):
        entries = [
            TOCEntry(level=1, number="1", title="Summary"),
            TOCEntry(level=2, number="1.1", title="Synopsis"),
            TOCEntry(level=3, number="1.1.1", title="Detail"),
            TOCEntry(level=2, number="1.2", title="Schema"),
        ]
        idx = ProtocolIndex(
            source_path="test.pdf",
            page_count=50,
            toc_entries=entries,
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[2],
        )
        assert idx.toc_entries[0].level == 1  # H1
        assert idx.toc_entries[1].level == 2  # H2
        assert idx.toc_entries[2].level == 3  # H3

    def test_get_children(self):
        entries = [
            TOCEntry(level=1, number="1", title="Summary"),
            TOCEntry(level=2, number="1.1", title="Synopsis"),
            TOCEntry(level=3, number="1.1.1", title="Detail"),
            TOCEntry(level=2, number="1.2", title="Schema"),
        ]
        idx = ProtocolIndex(
            source_path="test.pdf",
            page_count=50,
            toc_entries=entries,
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[2],
        )
        children = idx.get_children("1")
        assert len(children) == 2
        assert children[0].number == "1.1"
        assert children[1].number == "1.2"

    def test_get_section_text(self):
        idx = ProtocolIndex(
            source_path="test.pdf",
            page_count=50,
            toc_entries=[],
            section_headers=[],
            content_spans={"1": "Section one text", "1.1": "Subsection"},
            full_text="",
            toc_found=True,
            toc_pages=[2],
        )
        assert idx.get_section_text("1") == "Section one text"
        assert idx.get_section_text("1.1") == "Subsection"
        assert idx.get_section_text("99") is None


# -----------------------------------------------------------------------
# Scenario: TOC page resolution from body headers
# -----------------------------------------------------------------------

class TestTOCPageResolution:
    def test_resolve_toc_pages(self):
        entries = [
            TOCEntry(level=1, number="1", title="Summary", page_ref=9),
            TOCEntry(level=1, number="2", title="Intro", page_ref=16),
            TOCEntry(level=1, number="3", title="Methods", page_ref=22),
        ]
        headers = [
            SectionHeader("1", "Summary", 1, 10, 100),
            SectionHeader("2", "Intro", 1, 17, 500),
            SectionHeader("3", "Methods", 1, 23, 900),
        ]
        resolved = _resolve_toc_pages(entries, headers)
        assert resolved[0].page_start == 10
        assert resolved[0].char_offset_start == 100
        assert resolved[0].page_end == 17  # next section's page_start
        assert resolved[1].page_start == 17
        assert resolved[2].page_start == 23

    def test_unmatched_toc_entry(self):
        entries = [
            TOCEntry(level=1, number="99", title="Missing", page_ref=50),
        ]
        headers = [
            SectionHeader("1", "Summary", 1, 10, 100),
        ]
        resolved = _resolve_toc_pages(entries, headers)
        assert resolved[0].page_start == 0  # unresolved


# -----------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_protocol_index("/nonexistent/protocol.pdf")


# -----------------------------------------------------------------------
# Scenario: Extraction works across the qualifying protocol corpus
# -----------------------------------------------------------------------

_PROTOCOL_DIR = Path(__file__).resolve().parents[2] / "data" / "protocols" / "clinicaltrials"


@pytest.mark.skipif(
    not _PROTOCOL_DIR.exists(),
    reason="Protocol corpus not available",
)
class TestCorpusExtraction:
    """Given the qualifying protocol corpus,
    When the TOC extractor processes each protocol,
    Then at least 80% produce a ProtocolIndex with >= 10 sections.
    """

    def test_corpus_coverage(self):
        pdfs = sorted(_PROTOCOL_DIR.glob("*.pdf"))[:30]  # sample
        assert len(pdfs) >= 15, "Need at least 15 protocols for corpus test"

        pass_count = 0
        for pdf in pdfs:
            try:
                idx = extract_protocol_index(pdf)
                if len(idx.section_headers) >= 10:
                    pass_count += 1
            except Exception:
                pass  # count as failure

        pct = pass_count / len(pdfs)
        assert pct >= 0.80, (
            f"Only {pct:.0%} ({pass_count}/{len(pdfs)}) protocols "
            f"produced >=10 sections (need >=80%)"
        )


# -----------------------------------------------------------------------
# PTCV-107: Enhanced TOC detection and line stripping
# -----------------------------------------------------------------------


class TestEnhancedTocDetection:
    """Tests for strengthened _is_toc_page heuristic (PTCV-107)."""

    def test_detects_page_with_header(self) -> None:
        """Classic TOC page with header is detected."""
        text = (
            "TABLE OF CONTENTS\n"
            "1 Introduction .............. 5\n"
            "2 Objectives ................ 8\n"
        )
        assert _is_toc_page(text) is True

    def test_detects_sub_toc_without_header(self) -> None:
        """Page with >50% TOC lines but no header is detected."""
        text = (
            "4 Population ................. 15\n"
            "5 Endpoints .................. 20\n"
            "6 Statistical Methods ........ 25\n"
            "7 Safety ..................... 30\n"
        )
        assert _is_toc_page(text) is True

    def test_rejects_body_content(self) -> None:
        """Normal body content is not flagged as TOC."""
        text = (
            "4 STUDY POPULATION\n"
            "Patients eligible for this study must meet all of the\n"
            "following inclusion criteria and none of the exclusion\n"
            "criteria. Written informed consent is required.\n"
        )
        assert _is_toc_page(text) is False

    def test_rejects_short_page(self) -> None:
        """Pages with fewer than 3 non-empty lines are not flagged."""
        text = "1 Introduction .............. 5\n"
        assert _is_toc_page(text) is False

    def test_mixed_content_below_threshold(self) -> None:
        """Page with <50% TOC lines is not flagged."""
        text = (
            "1 Introduction .............. 5\n"
            "This section describes the study background.\n"
            "The rationale for this trial is based on\n"
            "preclinical evidence of efficacy.\n"
        )
        assert _is_toc_page(text) is False


class TestStripTocLines:
    """Tests for strip_toc_lines utility (PTCV-107)."""

    def test_removes_dotted_leader_lines(self) -> None:
        text = (
            "7.2 Inclusion Criteria\n"
            "7.3 Exclusion Criteria .......................... 20\n"
            "Patients must meet all inclusion criteria.\n"
        )
        result = strip_toc_lines(text)
        assert "Exclusion Criteria ........." not in result
        assert "Patients must meet" in result
        assert "7.2 Inclusion Criteria" in result

    def test_removes_appendix_toc_lines(self) -> None:
        text = (
            "Protocol content here.\n"
            "A  APPENDIX TITLE .......................... 45\n"
            "More content follows.\n"
        )
        result = strip_toc_lines(text)
        assert "APPENDIX TITLE" not in result
        assert "Protocol content here" in result
        assert "More content follows" in result

    def test_preserves_normal_content(self) -> None:
        text = (
            "4 STUDY POPULATION\n"
            "Patients eligible for this study must meet all of\n"
            "the following inclusion criteria.\n"
        )
        result = strip_toc_lines(text)
        assert result.strip() == text.strip()

    def test_preserves_blank_lines(self) -> None:
        text = "Line one.\n\nLine three.\n"
        result = strip_toc_lines(text)
        assert result == text

    def test_empty_input(self) -> None:
        assert strip_toc_lines("") == ""

    def test_content_spans_cleaned(self) -> None:
        """_resolve_content_spans strips TOC lines from spans."""
        headers = [
            SectionHeader(
                number="7",
                title="ELIGIBILITY",
                level=1,
                page=10,
                char_offset=0,
            ),
            SectionHeader(
                number="8",
                title="TREATMENT",
                level=1,
                page=15,
                char_offset=200,
            ),
        ]
        # Simulate full_text with TOC lines embedded in a section
        full_text = (
            "7 ELIGIBILITY\n"
            "7.1 Inclusion Criteria .............. 12\n"
            "7.2 Exclusion Criteria .............. 14\n"
            "Patients must be 18 years or older.\n"
            "Written informed consent required.\n"
        ) + ("x" * 100) + (
            "\n8 TREATMENT\n"
            "Drug X administered orally.\n"
        )
        # Adjust offset for header 8
        headers[1] = SectionHeader(
            number="8",
            title="TREATMENT",
            level=1,
            page=15,
            char_offset=full_text.index("8 TREATMENT"),
        )
        spans = _resolve_content_spans(headers, full_text)
        # Section 7 should NOT contain dotted-leader TOC lines
        assert "............" not in spans["7"]
        # But should still have body content
        assert "Patients must be 18" in spans["7"]


# -----------------------------------------------------------------------
# PTCV-108: List-of-Tables/Figures and page boilerplate stripping
# -----------------------------------------------------------------------


class TestListEntryRegex:
    """Tests for _LIST_ENTRY_RE pattern (PTCV-108)."""

    def test_matches_table_entry(self) -> None:
        line = "Table 10-1 Schedule of Assessments ............ 30"
        assert _LIST_ENTRY_RE.match(line) is not None

    def test_matches_figure_entry(self) -> None:
        line = "Figure 6-1 Trial Design ....................... 19"
        assert _LIST_ENTRY_RE.match(line) is not None

    def test_matches_listing_entry(self) -> None:
        line = "Listing 14.1.1 Demographics ................... 42"
        assert _LIST_ENTRY_RE.match(line) is not None

    def test_matches_exhibit_entry(self) -> None:
        line = "Exhibit A-1 Informed Consent Form ............. 55"
        assert _LIST_ENTRY_RE.match(line) is not None

    def test_matches_wide_whitespace_variant(self) -> None:
        line = "Table 10-2 eDiary Assessments              33"
        assert _LIST_ENTRY_RE.match(line) is not None

    def test_case_insensitive(self) -> None:
        line = "TABLE 10-1 Schedule of Assessments ............ 30"
        assert _LIST_ENTRY_RE.match(line) is not None

    def test_no_match_section_number(self) -> None:
        """Regular section number lines should NOT match."""
        line = "10.1 Study Endpoints .............. 22"
        assert _LIST_ENTRY_RE.match(line) is None

    def test_no_match_plain_text(self) -> None:
        line = "Table salt was used as a placebo comparator."
        assert _LIST_ENTRY_RE.match(line) is None


class TestListOfHeaderRegex:
    """Tests for _LIST_OF_HEADER_RE pattern (PTCV-108)."""

    def test_matches_list_of_tables(self) -> None:
        assert _LIST_OF_HEADER_RE.match("LIST OF IN-TEXT TABLES") is not None

    def test_matches_list_of_figures(self) -> None:
        assert _LIST_OF_HEADER_RE.match("List of Figures") is not None

    def test_matches_list_of_abbreviations(self) -> None:
        assert _LIST_OF_HEADER_RE.match("List of Abbreviations") is not None

    def test_no_match_normal_text(self) -> None:
        assert _LIST_OF_HEADER_RE.match("Patients were listed by site.") is None


class TestPageBoilerplate:
    """Tests for _is_page_boilerplate helper (PTCV-108)."""

    def test_proprietary_confidential(self) -> None:
        line = (
            "Proprietary and Confidential Page 11 of 59 "
            "Dr. Reddy's Laboratories, Ltd. Protocol No. DFN-02-CD-012"
        )
        assert _is_page_boilerplate(line) is True

    def test_confidential_standalone(self) -> None:
        assert _is_page_boilerplate("CONFIDENTIAL") is True

    def test_page_x_of_y(self) -> None:
        assert _is_page_boilerplate("Page 3 of 120") is True

    def test_page_within_line(self) -> None:
        line = "Protocol ABC-123 — Page 42 of 100 — Final"
        assert _is_page_boilerplate(line) is True

    def test_normal_text_not_matched(self) -> None:
        assert _is_page_boilerplate(
            "Patients received the study drug orally."
        ) is False

    def test_word_page_in_context(self) -> None:
        """'page' in normal prose should NOT trigger boilerplate."""
        assert _is_page_boilerplate(
            "Refer to page 5 for the inclusion criteria."
        ) is False


class TestStripTocLinesExtended:
    """Tests for PTCV-108 additions to strip_toc_lines."""

    def test_strips_table_list_entries(self) -> None:
        text = (
            "10 ASSESSMENTS\n"
            "Table 10-1 Schedule of Assessments ............ 30\n"
            "Table 10-2 eDiary Assessments ................. 33\n"
            "The schedule of assessments is summarised below.\n"
        )
        result = strip_toc_lines(text)
        assert "Table 10-1" not in result
        assert "Table 10-2" not in result
        assert "schedule of assessments is summarised" in result

    def test_strips_figure_list_entries(self) -> None:
        text = (
            "6 STUDY DESIGN\n"
            "Figure 6-1 Trial Design ....................... 19\n"
            "The study uses a randomised double-blind design.\n"
        )
        result = strip_toc_lines(text)
        assert "Figure 6-1" not in result
        assert "randomised double-blind" in result

    def test_strips_list_of_header(self) -> None:
        text = (
            "LIST OF IN-TEXT TABLES\n"
            "Table 10-1 Schedule of Assessments ............ 30\n"
            "Some actual content follows.\n"
        )
        result = strip_toc_lines(text)
        assert "LIST OF IN-TEXT TABLES" not in result
        assert "Table 10-1" not in result
        assert "Some actual content" in result

    def test_strips_page_boilerplate(self) -> None:
        text = (
            "5.1 Inclusion Criteria\n"
            "Proprietary and Confidential Page 11 of 59 "
            "Dr. Reddy's Laboratories\n"
            "Patients must be 18 years or older.\n"
        )
        result = strip_toc_lines(text)
        assert "Proprietary and Confidential" not in result
        assert "Patients must be 18" in result

    def test_preserves_legitimate_content(self) -> None:
        """Normal clinical content is not stripped."""
        text = (
            "5.1 Inclusion Criteria\n"
            "Patients must meet the following criteria:\n"
            "  1. Age >= 18 years\n"
            "  2. Written informed consent\n"
            "Table salt was used as placebo comparator.\n"
        )
        result = strip_toc_lines(text)
        assert result.strip() == text.strip()

    def test_content_spans_cleaned_extended(self) -> None:
        """_resolve_content_spans strips all junk from spans."""
        headers = [
            SectionHeader("10", "ASSESSMENTS", 1, 20, 0),
            SectionHeader("11", "TREATMENT", 1, 25, 300),
        ]
        full_text = (
            "10 ASSESSMENTS\n"
            "Table 10-1 Schedule of Assessments ............ 30\n"
            "Figure 10-1 Study Flow ........................ 31\n"
            "Proprietary and Confidential Page 20 of 59\n"
            "The schedule of assessments is provided below.\n"
            "All patients undergo screening visit.\n"
        ) + ("x" * 100) + (
            "\n11 TREATMENT\n"
            "Drug X is administered orally.\n"
        )
        headers[1] = SectionHeader(
            number="11",
            title="TREATMENT",
            level=1,
            page=25,
            char_offset=full_text.index("11 TREATMENT"),
        )
        spans = _resolve_content_spans(headers, full_text)
        assert "Table 10-1" not in spans["10"]
        assert "Figure 10-1" not in spans["10"]
        assert "Proprietary and Confidential" not in spans["10"]
        assert "schedule of assessments is provided" in spans["10"]
