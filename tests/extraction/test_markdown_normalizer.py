"""Tests for pymupdf4llm markdown normalizer — PTCV-176.

Qualification phase: IQ/OQ
Risk tier: MEDIUM
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.extraction.markdown_normalizer import (
    NormalizedResult,
    NormalizationStats,
    TOCItem,
    normalize_markdown,
)


# ---------------------------------------------------------------------------
# TestBoilerplateRemoval
# ---------------------------------------------------------------------------


class TestBoilerplateRemoval:
    """Boilerplate page headers/footers are removed."""

    def test_removes_confidential(self) -> None:
        text = "Intro paragraph\n\nCONFIDENTIAL\n\nNext paragraph"
        result = normalize_markdown(text)
        assert "CONFIDENTIAL" not in result.text
        assert "Intro paragraph" in result.text
        assert result.stats.boilerplate_lines_removed >= 1

    def test_removes_page_x_of_y(self) -> None:
        text = "Some content\nPage 3 of 163\nMore content"
        result = normalize_markdown(text)
        assert "Page 3 of 163" not in result.text
        assert "Some content" in result.text

    def test_removes_standalone_page_number(self) -> None:
        text = "Content above\n\nPage 42\n\nContent below"
        result = normalize_markdown(text)
        assert "Page 42" not in result.text

    def test_removes_copyright(self) -> None:
        text = "Header\n\u00a9 2023 Some Corp\nBody text"
        result = normalize_markdown(text)
        assert "\u00a9 2023" not in result.text
        assert "Body text" in result.text

    def test_removes_approval_stamp(self) -> None:
        text = "Content\nApproved on 14 Feb 2023 GMT\nMore content"
        result = normalize_markdown(text)
        assert "Approved on" not in result.text

    def test_removes_orphan_numeric_id(self) -> None:
        text = "Content\n\n2535\n\nMore content"
        result = normalize_markdown(text)
        assert "\n2535\n" not in result.text

    def test_preserves_real_content(self) -> None:
        text = (
            "## 5.1 Inclusion Criteria\n\n"
            "Individuals 18 to 75 years of age\n\n"
            "Individuals with diagnosed type 2 diabetes"
        )
        result = normalize_markdown(text)
        assert "Inclusion Criteria" in result.text
        assert "18 to 75 years" in result.text
        assert result.stats.boilerplate_lines_removed == 0

    def test_removes_clinical_trial_protocol(self) -> None:
        text = "Clinical Trial Protocol\nActual content here"
        result = normalize_markdown(text)
        assert "Clinical Trial Protocol" not in result.text
        assert "Actual content here" in result.text


# ---------------------------------------------------------------------------
# TestHeadingNormalization
# ---------------------------------------------------------------------------


class TestHeadingNormalization:
    """Bold markers stripped from ATX headings."""

    def test_strips_bold_from_heading(self) -> None:
        text = "##### **Title Page**"
        result = normalize_markdown(text)
        assert result.text.strip() == "##### Title Page"
        assert result.stats.headings_cleaned >= 1

    def test_strips_bold_various_levels(self) -> None:
        text = "## **Introduction**\n### **Methods**\n# **Abstract**"
        result = normalize_markdown(text)
        assert "## Introduction" in result.text
        assert "### Methods" in result.text
        assert "# Abstract" in result.text

    def test_merges_compound_bold_heading(self) -> None:
        text = "##### **1. Protocol Summary** **1.1. Synopsis**"
        result = normalize_markdown(text)
        assert "##### 1. Protocol Summary" in result.text
        assert "1.1. Synopsis" in result.text
        assert "**" not in result.text

    def test_preserves_non_heading_bold(self) -> None:
        text = "This is **bold text** in a paragraph."
        result = normalize_markdown(text)
        assert "**bold text**" in result.text

    def test_preserves_heading_without_bold(self) -> None:
        text = "## Plain Heading"
        result = normalize_markdown(text)
        assert result.text.strip() == "## Plain Heading"


# ---------------------------------------------------------------------------
# TestBoldToHeadingPromotion
# ---------------------------------------------------------------------------


class TestBoldToHeadingPromotion:
    """Standalone bold-numbered lines promoted to ATX headings."""

    def test_promotes_top_level_section(self) -> None:
        text = "**1. Protocol Summary**"
        result = normalize_markdown(text)
        assert result.text.strip() == "## 1. Protocol Summary"

    def test_promotes_subsection(self) -> None:
        text = "**1.2. Background**"
        result = normalize_markdown(text)
        assert result.text.strip() == "### 1.2. Background"

    def test_promotes_deep_subsection(self) -> None:
        text = "**1.2.3. Detail**"
        result = normalize_markdown(text)
        assert result.text.strip() == "#### 1.2.3. Detail"

    def test_promotes_four_part_section(self) -> None:
        text = "**1.2.3.4. Deep**"
        result = normalize_markdown(text)
        assert result.text.strip() == "##### 1.2.3.4. Deep"

    def test_promotes_split_bold(self) -> None:
        """pymupdf4llm split-bold: **1.** **Title Text**"""
        text = "**1.** **Title Text**"
        result = normalize_markdown(text)
        assert result.text.strip() == "## 1. Title Text"

    def test_does_not_promote_non_numbered(self) -> None:
        text = "**Some bold text**"
        result = normalize_markdown(text)
        assert "**Some bold text**" in result.text
        assert result.stats.headings_promoted == 0

    def test_does_not_promote_table_rows(self) -> None:
        text = "|**1. Col**|Value|"
        result = normalize_markdown(text)
        assert "|**1. Col**|Value|" in result.text
        assert result.stats.headings_promoted == 0

    def test_does_not_promote_existing_heading(self) -> None:
        text = "## **1. Title**"
        result = normalize_markdown(text)
        # Handled by _normalize_headings (bold stripped), not promoted
        assert "## 1. Title" in result.text
        assert result.stats.headings_promoted == 0

    def test_stats_tracks_promotions(self) -> None:
        text = (
            "**1. First Section**\n\n"
            "Some content.\n\n"
            "**1.1. Sub Section**\n\n"
            "More content."
        )
        result = normalize_markdown(text)
        assert result.stats.headings_promoted == 2

    def test_end_to_end_heading_count_increases(self) -> None:
        """On realistic sample, heading count goes up after promotion."""
        text = (
            "## Introduction\n\n"
            "**1. Protocol Summary**\n\n"
            "Overview text.\n\n"
            "**1.1. Synopsis**\n\n"
            "Synopsis text.\n\n"
            "**1.2. Background**\n\n"
            "Background text.\n\n"
            "**2. Objectives**\n\n"
            "Objective text."
        )
        result = normalize_markdown(text)
        heading_lines = [
            line for line in result.text.split("\n") if line.startswith("#")
        ]
        # 1 existing + 4 promoted = 5
        assert len(heading_lines) == 5
        assert result.stats.headings_promoted == 4


# ---------------------------------------------------------------------------
# TestTableCleanup
# ---------------------------------------------------------------------------


class TestTableCleanup:
    """<br> tags replaced with spaces in table cells."""

    def test_replaces_br_in_table_row(self) -> None:
        text = "|Section 1.1 Protocol<br>Synopsis|Updated text|"
        result = normalize_markdown(text)
        assert "<br>" not in result.text
        assert "Protocol Synopsis" in result.text
        assert result.stats.br_tags_removed >= 1

    def test_replaces_br_slash_in_table(self) -> None:
        text = "|Cell with<br/>break|Other cell|"
        result = normalize_markdown(text)
        assert "<br/>" not in result.text

    def test_leaves_non_table_br_alone(self) -> None:
        text = "Paragraph with <br> in it"
        result = normalize_markdown(text)
        assert "<br>" in result.text

    def test_counts_replacements(self) -> None:
        text = "|A<br>B<br>C|D<br>E|"
        result = normalize_markdown(text)
        assert result.stats.br_tags_removed == 3

    def test_handles_separator_row(self) -> None:
        text = "|---|---|---|\n|A|B|C|"
        result = normalize_markdown(text)
        assert "|---|---|---|" in result.text


# ---------------------------------------------------------------------------
# TestTocExtraction
# ---------------------------------------------------------------------------


class TestTocExtraction:
    """TOC dot-leader lines extracted and removed."""

    def test_extracts_bold_toc_line(self) -> None:
        text = "**1.** **Protocol Summary ........................................11**"
        result = normalize_markdown(text, extract_toc=True)
        assert len(result.toc) == 1
        assert result.toc[0].number == "1"
        assert "Protocol Summary" in result.toc[0].title
        assert result.toc[0].page == 11

    def test_extracts_plain_toc_line(self) -> None:
        text = "1.1. Synopsis ..........................................................11"
        result = normalize_markdown(text, extract_toc=True)
        assert len(result.toc) == 1
        assert result.toc[0].number == "1.1"
        assert result.toc[0].title == "Synopsis"
        assert result.toc[0].page == 11

    def test_removes_toc_from_output(self) -> None:
        text = (
            "Table of Contents\n\n"
            "**1.** **Protocol Summary ........11**\n"
            "1.1. Synopsis ..........11\n\n"
            "## 1. Protocol Summary"
        )
        result = normalize_markdown(text, extract_toc=True)
        assert "........" not in result.text
        assert "## 1. Protocol Summary" in result.text
        assert result.stats.toc_lines_removed == 2

    def test_removes_toc_without_extracting(self) -> None:
        text = "1.1. Synopsis ..........11\n\nContent"
        result = normalize_markdown(text, extract_toc=False)
        assert len(result.toc) == 0
        assert "........" not in result.text
        assert result.stats.toc_lines_removed == 1

    def test_multi_level_toc(self) -> None:
        text = (
            "**1.** **Summary ....11**\n"
            "1.1. Synopsis ..........11\n"
            "1.2. Schema ..............16\n"
            "**2.** **Introduction ....21**\n"
        )
        result = normalize_markdown(text, extract_toc=True)
        assert len(result.toc) == 4
        pages = [item.page for item in result.toc]
        assert pages == [11, 11, 16, 21]


# ---------------------------------------------------------------------------
# TestWhitespaceNormalization
# ---------------------------------------------------------------------------


class TestWhitespaceNormalization:
    """Excessive blank lines collapsed."""

    def test_collapses_excess_newlines(self) -> None:
        text = "Para 1\n\n\n\n\nPara 2"
        result = normalize_markdown(text)
        # Should be at most 2 blank lines (3 newlines)
        assert "\n\n\n\n" not in result.text
        assert "Para 1" in result.text
        assert "Para 2" in result.text

    def test_preserves_single_blank_line(self) -> None:
        text = "Para 1\n\nPara 2"
        result = normalize_markdown(text)
        assert "Para 1\n\nPara 2" in result.text

    def test_strips_trailing_whitespace(self) -> None:
        text = "Line with trailing spaces   \nNext line"
        result = normalize_markdown(text)
        assert "   \n" not in result.text

    def test_output_ends_with_single_newline(self) -> None:
        text = "Content\n\n\n"
        result = normalize_markdown(text)
        assert result.text.endswith("\n")
        assert not result.text.endswith("\n\n")


# ---------------------------------------------------------------------------
# TestIdempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Running normalize twice produces same output."""

    def test_idempotent(self) -> None:
        text = (
            "##### **Title Page**\n\n"
            "CONFIDENTIAL\n\n"
            "1.1. Synopsis ..........11\n\n"
            "|A<br>B|C|\n"
            "|---|---|\n"
            "|D|E|\n\n\n\n\n"
            "Real content here"
        )
        first = normalize_markdown(text)
        second = normalize_markdown(first.text)
        assert first.text == second.text


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline on realistic pymupdf4llm snippet."""

    SAMPLE = (
        "Protocol: I1F-IN-RHCZ (a)\n\n\n"
        "CONFIDENTIAL Protocol I1F-IN-RHCZ (a)\n\n"
        "### **Title Page**\n\n"
        "##### **Confidential Information**\n\n"
        "The information contained in this document is confidential.\n\n"
        "**1.** **Protocol Summary ........................................11**\n"
        "1.1. Synopsis ..........................................................11\n"
        "1.2. Schema ............................................................16\n\n"
        "##### **1. Protocol Summary** **1.1. Synopsis**\n\n"
        "This is a Phase 4 study.\n\n"
        "|DOCUMENT HISTORY|Col2|\n"
        "|---|---|\n"
        "|**Document**|**Date**|\n"
        "|_Original Protocol_|_05 May 2022_|\n\n"
        "Page 3 of 163\n\n\n\n\n"
        "|Section 1.1 Protocol<br>Synopsis|Updated text<br>here|"
    )

    def test_all_transforms_applied(self) -> None:
        result = normalize_markdown(self.SAMPLE, extract_toc=True)

        # TOC lines removed
        assert "........" not in result.text
        assert len(result.toc) == 3
        assert result.stats.toc_lines_removed == 3

        # Boilerplate removed — "Page 3 of 163" matches Page X of Y pattern;
        # "CONFIDENTIAL Protocol I1F-IN-RHCZ (a)" is NOT pure boilerplate
        # (contains protocol ID), so it's preserved.
        assert "Page 3 of 163" not in result.text
        assert result.stats.boilerplate_lines_removed >= 1

        # Headings cleaned
        assert "### Title Page" in result.text
        assert "##### Confidential Information" in result.text
        assert "**" not in result.text.split("\n")[
            next(
                i
                for i, line in enumerate(result.text.split("\n"))
                if "Title Page" in line
            )
        ]
        assert result.stats.headings_cleaned >= 2

        # Tables cleaned
        assert "<br>" not in result.text
        assert result.stats.br_tags_removed >= 2

        # Whitespace normalized
        assert "\n\n\n\n" not in result.text

        # Real content preserved
        assert "Phase 4 study" in result.text
        assert "DOCUMENT HISTORY" in result.text

    def test_stats_populated(self) -> None:
        result = normalize_markdown(self.SAMPLE)
        stats = result.stats
        assert stats.boilerplate_lines_removed > 0
        assert stats.headings_cleaned > 0
        assert stats.toc_lines_removed > 0
        assert stats.br_tags_removed > 0
        assert stats.blank_lines_collapsed > 0
