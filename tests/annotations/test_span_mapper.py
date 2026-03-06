"""Unit tests for span mapper (PTCV-43).

Tests the mapping of classified ICH sections onto full protocol text,
including gap detection and coverage computation.
"""

from __future__ import annotations

import json

from ptcv.annotations.span_mapper import (
    TextSpan,
    compute_coverage,
    map_sections_to_spans,
)
from ptcv.ich_parser.models import IchSection


def _make_section(
    code: str = "B.3",
    name: str = "Trial Objectives and Purpose",
    text: str = "Test content for section B.3",
    confidence: float = 0.85,
) -> IchSection:
    """Create a minimal IchSection for tests."""
    return IchSection(
        run_id="run-1",
        source_run_id="src-1",
        source_sha256="abc123",
        registry_id="NCT001",
        section_code=code,
        section_name=name,
        content_json=json.dumps({"text_excerpt": text}),
        confidence_score=confidence,
        review_required=confidence < 0.70,
        legacy_format=False,
        extraction_timestamp_utc="2026-03-03T00:00:00Z",
    )


class TestMapSectionsToSpans:
    """Tests for map_sections_to_spans."""

    def test_single_section_match(self) -> None:
        text = "Introduction. Test content for section B.3. Conclusion."
        sec = _make_section(text="Test content for section B.3")
        spans = map_sections_to_spans(text, [sec])

        classified = [s for s in spans if s.classified]
        assert len(classified) == 1
        assert classified[0].section_code == "B.3"
        assert text[classified[0].start:classified[0].end] == (
            "Test content for section B.3"
        )

    def test_gaps_created(self) -> None:
        text = "AAA content for B.1 BBB content for B.3 CCC"
        sections = [
            _make_section(code="B.1", text="content for B.1"),
            _make_section(code="B.3", text="content for B.3"),
        ]
        spans = map_sections_to_spans(text, sections)

        # Should have: gap, classified, gap, classified, gap
        assert len(spans) == 5
        assert not spans[0].classified  # "AAA "
        assert spans[1].classified  # "content for B.1"
        assert spans[1].section_code == "B.1"
        assert not spans[2].classified  # " BBB "
        assert spans[3].classified  # "content for B.3"
        assert spans[3].section_code == "B.3"
        assert not spans[4].classified  # " CCC"

    def test_no_sections_single_gap(self) -> None:
        text = "This is unclassified text."
        spans = map_sections_to_spans(text, [])

        assert len(spans) == 1
        assert not spans[0].classified
        assert spans[0].start == 0
        assert spans[0].end == len(text)

    def test_case_insensitive_fallback(self) -> None:
        text = "THE OBJECTIVES OF THIS TRIAL ARE..."
        sec = _make_section(
            text="the objectives of this trial are...",
        )
        spans = map_sections_to_spans(text, [sec])

        classified = [s for s in spans if s.classified]
        assert len(classified) == 1

    def test_spans_cover_full_text(self) -> None:
        text = "Start. Content for B.1 here. Middle. Content for B.5 here. End."
        sections = [
            _make_section(code="B.1", text="Content for B.1 here"),
            _make_section(code="B.5", text="Content for B.5 here"),
        ]
        spans = map_sections_to_spans(text, sections)

        # Verify full coverage
        total = sum(s.length for s in spans)
        assert total == len(text)

        # Verify no gaps in offsets
        for i in range(1, len(spans)):
            assert spans[i].start == spans[i - 1].end

    def test_section_not_found_skipped(self) -> None:
        text = "This text has nothing matching."
        sec = _make_section(text="COMPLETELY DIFFERENT TEXT")
        spans = map_sections_to_spans(text, [sec])

        assert len(spans) == 1
        assert not spans[0].classified


class TestComputeCoverage:
    """Tests for compute_coverage."""

    def test_full_coverage(self) -> None:
        spans = [TextSpan(start=0, end=100, classified=True)]
        cov = compute_coverage(spans, 100)
        assert cov["coverage_pct"] == 100.0
        assert cov["gap_count"] == 0

    def test_partial_coverage(self) -> None:
        spans = [
            TextSpan(start=0, end=30, classified=False),
            TextSpan(
                start=30, end=70, classified=True,
                section_code="B.1",
            ),
            TextSpan(start=70, end=100, classified=False),
        ]
        cov = compute_coverage(spans, 100)
        assert cov["coverage_pct"] == 40.0
        assert cov["classified_chars"] == 40
        assert cov["unclassified_chars"] == 60
        assert cov["gap_count"] == 2
        assert cov["classified_count"] == 1

    def test_zero_length_text(self) -> None:
        cov = compute_coverage([], 0)
        assert cov["coverage_pct"] == 0.0


class TestTextSpan:
    """Tests for TextSpan dataclass."""

    def test_length_property(self) -> None:
        span = TextSpan(start=10, end=50)
        assert span.length == 40

    def test_classified_defaults(self) -> None:
        span = TextSpan(start=0, end=10)
        assert not span.classified
        assert span.section_code == ""
        assert span.confidence == 0.0
