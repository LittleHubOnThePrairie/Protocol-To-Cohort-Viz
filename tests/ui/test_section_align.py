"""Tests for enhanced section-aligned comparison (PTCV-80).

Pure-Python tests — no Streamlit dependency.

Covers GHERKIN scenarios:
  - Synchronized scrolling between panels (HTML contains sync JS)
  - Section navigation jumps both panels (nav items with onclick)
  - Fallback for unmatched sections (no ICH mapping indicator)
"""

from __future__ import annotations

import dataclasses

from ptcv.ui.components.section_align import (
    SectionPair,
    _parse_markdown_sections,
    align_sections,
    build_comparison_html,
)


# ---------------------------------------------------------------------------
# Fake IchSection for testing (avoids importing real model)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _FakeSection:
    section_code: str
    section_name: str
    content_text: str = ""


RETEMPLATED_MD = """\
# Protocol NCT001

## B.1 Title Page

This is the retemplated title page.

## B.3 Ethics

Retemplated ethics content.

## B.5 Study Design

Retemplated study design with trial phases.

## B.7 Safety Assessments

Retemplated safety content.
"""

SECTIONS = [
    _FakeSection("B.1", "Title Page", "Original title page text"),
    _FakeSection("B.3", "Ethics", "Original ethics content"),
    _FakeSection("B.5", "Study Design", "Original study design"),
    _FakeSection("B.7", "Safety Assessments", "Original safety data"),
]


# ---------------------------------------------------------------------------
# Scenario: _parse_markdown_sections
# ---------------------------------------------------------------------------

class TestParseMarkdownSections:
    """Markdown is split into section-keyed blocks."""

    def test_finds_all_sections(self) -> None:
        result = _parse_markdown_sections(RETEMPLATED_MD)
        assert set(result.keys()) == {"B.1", "B.3", "B.5", "B.7"}

    def test_section_content_not_empty(self) -> None:
        result = _parse_markdown_sections(RETEMPLATED_MD)
        for code, text in result.items():
            assert len(text) > 0, f"{code} has empty content"

    def test_b1_contains_title(self) -> None:
        result = _parse_markdown_sections(RETEMPLATED_MD)
        assert "retemplated title page" in result["B.1"]

    def test_b7_contains_safety(self) -> None:
        result = _parse_markdown_sections(RETEMPLATED_MD)
        assert "safety" in result["B.7"].lower()

    def test_empty_markdown(self) -> None:
        assert _parse_markdown_sections("") == {}

    def test_no_section_headings(self) -> None:
        assert _parse_markdown_sections("Just some text.") == {}


# ---------------------------------------------------------------------------
# Scenario: align_sections
# ---------------------------------------------------------------------------

class TestAlignSections:
    """Sections are paired with matching retemplated content."""

    def test_returns_correct_count(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        assert len(pairs) == 4

    def test_section_codes_preserved(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        codes = [p.section_code for p in pairs]
        assert codes == ["B.1", "B.3", "B.5", "B.7"]

    def test_original_text_preserved(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        assert pairs[0].original_text == "Original title page text"

    def test_retemplated_text_matched(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        assert "retemplated title page" in pairs[0].retemplated_text

    def test_unmatched_section_gets_empty_retemplated(self) -> None:
        sections = [_FakeSection("B.99", "Unknown", "Some content")]
        pairs = align_sections(sections, RETEMPLATED_MD)
        assert pairs[0].retemplated_text == ""

    def test_empty_inputs(self) -> None:
        pairs = align_sections([], "")
        assert pairs == []

    def test_section_pair_is_named_tuple(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        assert isinstance(pairs[0], SectionPair)


# ---------------------------------------------------------------------------
# Scenario: build_comparison_html
# ---------------------------------------------------------------------------

class TestBuildComparisonHtml:
    """HTML output contains required structural elements."""

    def test_contains_nav_panel(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        assert 'class="nav-panel"' in html

    def test_contains_left_panel(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        assert 'id="left"' in html

    def test_contains_right_panel(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        assert 'id="right"' in html

    def test_contains_sync_scroll_js(self) -> None:
        """GHERKIN: synchronized scrolling JS is present."""
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        assert "syncScroll" in html

    def test_contains_jump_to_js(self) -> None:
        """GHERKIN: section navigation JS is present."""
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        assert "jumpTo" in html

    def test_nav_items_have_section_codes(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        for p in pairs:
            assert p.section_code in html

    def test_section_anchors_exist(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        assert 'id="left-B.1"' in html
        assert 'id="right-B.1"' in html

    def test_original_content_in_left_panel(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        assert "Original title page text" in html

    def test_retemplated_content_in_right_panel(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs)
        assert "retemplated title page" in html

    def test_unmatched_section_shows_indicator(self) -> None:
        """GHERKIN: fallback for unmatched sections."""
        pairs = [SectionPair("B.99", "Unknown", "Content", "")]
        html = build_comparison_html(pairs)
        assert "no retemplated content" in html

    def test_empty_pairs_returns_message(self) -> None:
        html = build_comparison_html([])
        assert "No sections available" in html

    def test_custom_height(self) -> None:
        pairs = align_sections(SECTIONS, RETEMPLATED_MD)
        html = build_comparison_html(pairs, height=800)
        assert "800px" in html

    def test_html_escapes_special_chars(self) -> None:
        pairs = [SectionPair("B.1", "Test", "<script>alert(1)</script>", "")]
        html = build_comparison_html(pairs)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
