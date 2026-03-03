"""Unit tests for ICH E6(R3) protocol regeneration (PTCV-35).

Tests the markdown generation logic in isolation without
Streamlit runtime.
"""

from __future__ import annotations

import json

from ptcv.ich_parser.models import IchSection
from ptcv.ui.components.ich_regenerator import (
    ICH_SECTIONS,
    _extract_text,
    make_download_filename,
    regenerate_ich_markdown,
)


def _make_section(
    code: str = "B.1",
    name: str = "General Information",
    content: str = "Test content",
    confidence: float = 0.85,
    review_required: bool = False,
    legacy_format: bool = False,
) -> IchSection:
    """Helper to create a minimal IchSection for tests."""
    return IchSection(
        run_id="run-1",
        source_run_id="src-1",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code=code,
        section_name=name,
        content_json=json.dumps({"text_excerpt": content}),
        confidence_score=confidence,
        review_required=review_required,
        legacy_format=legacy_format,
        extraction_timestamp_utc="2026-03-02T00:00:00Z",
    )


class TestRegenerateIchMarkdown:
    """Tests for regenerate_ich_markdown()."""

    def test_all_11_sections_present_in_output(self) -> None:
        sections = [
            _make_section(code=code, name=name)
            for code, name in ICH_SECTIONS
        ]
        md = regenerate_ich_markdown(
            sections, "NCT001", "ICH_E6R3", 0.85
        )
        for code, name in ICH_SECTIONS:
            assert f"## {code} {name}" in md

    def test_missing_section_shows_placeholder(self) -> None:
        # Only provide B.1 — all others should be placeholders
        sections = [_make_section("B.1", "General Information")]
        md = regenerate_ich_markdown(
            sections, "NCT001", "PARTIAL_ICH", 0.40
        )
        assert "Section not detected in source protocol" in md
        assert "## B.6 Treatment of Subjects" in md

    def test_low_confidence_section_flagged(self) -> None:
        sections = [
            _make_section(
                "B.3",
                "Trial Objectives and Purpose",
                confidence=0.45,
                review_required=True,
            )
        ]
        md = regenerate_ich_markdown(
            sections, "NCT001", "PARTIAL_ICH", 0.45
        )
        assert "confidence: 0.45" in md
        assert "Low confidence" in md

    def test_high_confidence_section_no_warning(self) -> None:
        sections = [
            _make_section("B.1", "General Information", confidence=0.90)
        ]
        md = regenerate_ich_markdown(
            sections, "NCT001", "ICH_E6R3", 0.90
        )
        assert "Low confidence" not in md.split("## B.2")[0]

    def test_non_ich_verdict_shows_banner(self) -> None:
        sections = [
            _make_section("B.1", "General Information", confidence=0.15)
        ]
        md = regenerate_ich_markdown(
            sections, "NCT001", "NON_ICH", 0.10
        )
        assert "Most sections could not be mapped" in md

    def test_ich_e6r3_verdict_no_banner(self) -> None:
        sections = [
            _make_section(code=code, name=name)
            for code, name in ICH_SECTIONS
        ]
        md = regenerate_ich_markdown(
            sections, "NCT001", "ICH_E6R3", 0.85
        )
        assert "Most sections could not be mapped" not in md

    def test_title_contains_registry_id(self) -> None:
        md = regenerate_ich_markdown([], "NCT00112827", "NON_ICH", 0.0)
        assert "NCT00112827" in md.split("\n")[0]

    def test_sections_in_order(self) -> None:
        # Provide sections out of order
        sections = [
            _make_section("B.5", "Selection of Subjects"),
            _make_section("B.1", "General Information"),
            _make_section("B.9", "Statistics"),
        ]
        md = regenerate_ich_markdown(
            sections, "NCT001", "PARTIAL_ICH", 0.50
        )
        b1_pos = md.index("## B.1")
        b5_pos = md.index("## B.5")
        b9_pos = md.index("## B.9")
        assert b1_pos < b5_pos < b9_pos

    def test_duplicate_sections_highest_confidence_wins(self) -> None:
        sections = [
            _make_section("B.1", "General Information",
                          content="low", confidence=0.30),
            _make_section("B.1", "General Information",
                          content="high", confidence=0.90),
        ]
        md = regenerate_ich_markdown(
            sections, "NCT001", "PARTIAL_ICH", 0.60
        )
        # Should use the high-confidence version
        assert "high" in md
        # Low-confidence warning should not appear for B.1
        b1_chunk = md.split("## B.2")[0]
        assert "Low confidence" not in b1_chunk

    def test_content_json_text_excerpt_extracted(self) -> None:
        sec = _make_section(content="Sponsor is Acme Pharma")
        md = regenerate_ich_markdown(
            [sec], "NCT001", "ICH_E6R3", 0.85
        )
        assert "Sponsor is Acme Pharma" in md

    def test_confidence_and_verdict_in_header(self) -> None:
        md = regenerate_ich_markdown(
            [], "NCT001", "PARTIAL_ICH", 0.55
        )
        assert "PARTIAL_ICH" in md
        assert "0.55" in md


class TestExtractText:
    """Tests for _extract_text()."""

    def test_text_excerpt_key(self) -> None:
        sec = _make_section(content="hello world")
        assert "hello world" in _extract_text(sec)

    def test_key_concepts_included(self) -> None:
        sec = _make_section()
        sec.content_json = json.dumps({
            "text_excerpt": "Main text.",
            "key_concepts": ["oncology", "phase III"],
        })
        text = _extract_text(sec)
        assert "oncology" in text
        assert "phase III" in text

    def test_plain_string_json(self) -> None:
        sec = _make_section()
        sec.content_json = json.dumps("Just a plain string")
        assert "Just a plain string" in _extract_text(sec)

    def test_invalid_json_returns_raw(self) -> None:
        sec = _make_section()
        sec.content_json = "not valid json {"
        assert "not valid json {" in _extract_text(sec)


class TestMakeDownloadFilename:
    """Tests for make_download_filename()."""

    def test_standard_format(self) -> None:
        assert (
            make_download_filename("NCT00112827", "1.0")
            == "NCT00112827_1.0_ich_reformatted.md"
        )

    def test_amendment_00(self) -> None:
        assert (
            make_download_filename("NCT001", "00")
            == "NCT001_00_ich_reformatted.md"
        )
