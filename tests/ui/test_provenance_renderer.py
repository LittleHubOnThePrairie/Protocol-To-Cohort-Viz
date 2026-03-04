"""Unit tests for provenance hover renderer (PTCV-81).

Tests HTML generation, tooltip content, confidence-based styling,
and page range parsing without requiring Streamlit runtime.
"""

import json

from ptcv.ich_parser.models import IchSection
from ptcv.ui.components.provenance_renderer import (
    _LOW_CONFIDENCE_THRESHOLD,
    _parse_page_range,
    _build_tooltip,
    build_provenance_html,
    estimate_html_height,
)


def _make_section(
    code: str = "B.1",
    name: str = "General Information",
    confidence: float = 0.95,
    page_range: list | None = None,
    content_text: str = "Sample section text.",
) -> IchSection:
    """Helper to build a minimal IchSection for testing."""
    content: dict = {
        "text_excerpt": content_text[:200],
        "key_concepts": ["test"],
        "word_count": len(content_text.split()),
    }
    if page_range is not None:
        content["page_range"] = page_range
    return IchSection(
        run_id="test-run",
        source_run_id="src-run",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code=code,
        section_name=name,
        content_json=json.dumps(content),
        confidence_score=confidence,
        review_required=confidence < 0.70,
        legacy_format=False,
        content_text=content_text,
    )


class TestParsePageRange:
    """Tests for _parse_page_range()."""

    def test_multi_page_range(self) -> None:
        sec = _make_section(page_range=[12, 15])
        result = _parse_page_range(sec)
        assert result == "Pages 12\u201315"

    def test_single_page(self) -> None:
        sec = _make_section(page_range=[3, 3])
        result = _parse_page_range(sec)
        assert result == "Page 3"

    def test_empty_range(self) -> None:
        sec = _make_section(page_range=[])
        assert _parse_page_range(sec) == ""

    def test_no_page_range_key(self) -> None:
        sec = _make_section()
        # content_json has no page_range when page_range=None
        sec.content_json = json.dumps({"text_excerpt": "test"})
        assert _parse_page_range(sec) == ""

    def test_invalid_json(self) -> None:
        sec = _make_section()
        sec.content_json = "not-json"
        assert _parse_page_range(sec) == ""

    def test_single_element_list(self) -> None:
        sec = _make_section(page_range=[7])
        # Falls through to len==1 branch
        assert _parse_page_range(sec) == "Page 7"


class TestBuildTooltip:
    """Tests for _build_tooltip()."""

    def test_high_confidence_with_pages(self) -> None:
        sec = _make_section(confidence=0.94, page_range=[12, 15])
        tooltip = _build_tooltip(sec)
        assert "Source: Pages 12" in tooltip
        assert "94%" in tooltip
        assert "review recommended" not in tooltip

    def test_low_confidence_includes_warning(self) -> None:
        sec = _make_section(confidence=0.55, page_range=[1, 3])
        tooltip = _build_tooltip(sec)
        assert "55%" in tooltip
        assert "review recommended" in tooltip

    def test_no_pages_still_has_confidence(self) -> None:
        sec = _make_section(confidence=0.80)
        sec.content_json = json.dumps({"text_excerpt": "test"})
        tooltip = _build_tooltip(sec)
        assert "80%" in tooltip
        assert "Source:" not in tooltip


class TestBuildProvenanceHtml:
    """Tests for build_provenance_html()."""

    def test_returns_html_string(self) -> None:
        sections = [_make_section()]
        result = build_provenance_html(sections, "NCT00112827")
        assert isinstance(result, str)
        assert "<div" in result
        assert "</div>" in result

    def test_contains_registry_id(self) -> None:
        html = build_provenance_html(
            [_make_section()], "NCT00112827",
        )
        assert "NCT00112827" in html

    def test_high_confidence_section_blue_styling(self) -> None:
        sec = _make_section(confidence=0.95, page_range=[1, 5])
        html = build_provenance_html([sec], "NCT001")
        assert "high-confidence" in html
        assert "badge-high" in html

    def test_low_confidence_section_amber_styling(self) -> None:
        sec = _make_section(
            code="B.7",
            name="Assessment of Efficacy",
            confidence=0.55,
            page_range=[20, 22],
        )
        html = build_provenance_html([sec], "NCT001")
        assert "low-confidence" in html
        assert "badge-low" in html

    def test_missing_section_shows_placeholder(self) -> None:
        # No sections provided — all 11 should be "missing"
        html = build_provenance_html([], "NCT001")
        assert "Section not detected" in html
        assert 'class="provenance-section missing"' in html

    def test_tooltip_in_title_attribute(self) -> None:
        sec = _make_section(confidence=0.90, page_range=[5, 8])
        html = build_provenance_html([sec], "NCT001")
        assert 'title="' in html
        assert "Pages 5" in html

    def test_content_text_rendered(self) -> None:
        sec = _make_section(content_text="Study drug is administered IV")
        html = build_provenance_html([sec], "NCT001")
        assert "Study drug is administered IV" in html

    def test_html_escapes_special_chars(self) -> None:
        sec = _make_section(
            content_text='<script>alert("xss")</script>',
        )
        html = build_provenance_html([sec], "NCT001")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_long_content_truncated(self) -> None:
        long_text = "A" * 5000
        sec = _make_section(content_text=long_text)
        html = build_provenance_html([sec], "NCT001")
        assert "[...truncated...]" in html

    def test_contains_css_styles(self) -> None:
        html = build_provenance_html([], "NCT001")
        assert "<style>" in html
        assert "provenance-container" in html

    def test_provenance_meta_shows_pages(self) -> None:
        sec = _make_section(page_range=[10, 14])
        html = build_provenance_html([sec], "NCT001")
        assert "provenance-meta" in html
        assert "Pages 10" in html

    def test_low_confidence_meta_shows_warning(self) -> None:
        sec = _make_section(confidence=0.45, page_range=[1, 2])
        html = build_provenance_html([sec], "NCT001")
        assert "review recommended" in html

    def test_multiple_sections_all_rendered(self) -> None:
        sections = [
            _make_section(code="B.1", confidence=0.95, page_range=[1, 3]),
            _make_section(
                code="B.4", name="Trial Design",
                confidence=0.60, page_range=[10, 15],
            ),
        ]
        html = build_provenance_html(sections, "NCT001")
        assert "B.1" in html
        assert "B.4" in html
        assert "high-confidence" in html
        assert "low-confidence" in html


class TestEstimateHtmlHeight:
    """Tests for estimate_html_height()."""

    def test_minimum_height(self) -> None:
        assert estimate_html_height([]) >= 400

    def test_more_sections_taller(self) -> None:
        small = [_make_section(content_text="short")]
        large = [
            _make_section(content_text="x" * 2000),
            _make_section(
                code="B.2", content_text="y" * 2000,
            ),
        ]
        assert estimate_html_height(large) > estimate_html_height(small)

    def test_max_capped(self) -> None:
        huge = [
            _make_section(content_text="z" * 50000)
            for _ in range(20)
        ]
        assert estimate_html_height(huge) <= 3000

    def test_returns_int(self) -> None:
        assert isinstance(estimate_html_height([]), int)
