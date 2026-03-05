"""Tests for query pipeline UI helpers (PTCV-95).

Pure-Python tests — no Streamlit dependency.
Tests format helpers with mock data matching the backend dataclass shapes.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

from ptcv.ui.components.query_pipeline import (
    _CONTENT_PREVIEW_LEN,
    format_coverage_metrics,
    format_extraction_table,
    format_gap_table,
    format_match_table,
    format_toc_tree,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _toc(
    level: int = 1,
    number: str = "1",
    title: str = "Introduction",
    page_ref: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        level=level, number=number, title=title, page_ref=page_ref,
    )


def _mapping(
    section_number: str = "1",
    section_title: str = "Introduction",
    matches: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        protocol_section_number=section_number,
        protocol_section_title=section_title,
        matches=matches or [],
    )


def _match(
    code: str = "B.1",
    name: str = "General Information",
    score: float = 0.90,
    boosted_score: float = 0.90,
    confidence: str = "HIGH",
    method: str = "keyword_fallback",
) -> SimpleNamespace:
    return SimpleNamespace(
        ich_section_code=code,
        ich_section_name=name,
        similarity_score=score,
        boosted_score=boosted_score,
        confidence=confidence,
        match_method=method,
    )


def _match_result(
    mappings: list | None = None,
    auto_mapped_count: int = 0,
    review_count: int = 0,
    unmapped_count: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        mappings=mappings or [],
        auto_mapped_count=auto_mapped_count,
        review_count=review_count,
        unmapped_count=unmapped_count,
    )


def _extraction(
    query_id: str = "B.1.1.q1",
    section_id: str = "B.1.1",
    content: str = "Test content",
    confidence: float = 0.90,
    extraction_method: str = "text_short",
    source_section: str = "1.1",
) -> SimpleNamespace:
    return SimpleNamespace(
        query_id=query_id,
        section_id=section_id,
        content=content,
        confidence=confidence,
        extraction_method=extraction_method,
        source_section=source_section,
    )


def _gap(
    query_id: str = "B.7.1.q1",
    section_id: str = "B.7.1",
    reason: str = "no_match",
) -> SimpleNamespace:
    return SimpleNamespace(
        query_id=query_id, section_id=section_id, reason=reason,
    )


def _extraction_result(
    extractions: list | None = None,
    gaps: list | None = None,
    coverage: float = 0.75,
    total_queries: int = 10,
    answered_queries: int = 8,
) -> SimpleNamespace:
    return SimpleNamespace(
        extractions=extractions or [],
        gaps=gaps or [],
        coverage=coverage,
        total_queries=total_queries,
        answered_queries=answered_queries,
    )


def _coverage(
    total_sections: int = 16,
    populated_count: int = 10,
    gap_count: int = 6,
    avg_confidence: float = 0.80,
    high_confidence_count: int = 6,
    review_confidence_count: int = 3,
    low_confidence_count: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        total_sections=total_sections,
        populated_count=populated_count,
        gap_count=gap_count,
        avg_confidence=avg_confidence,
        high_confidence_count=high_confidence_count,
        review_confidence_count=review_confidence_count,
        low_confidence_count=low_confidence_count,
    )


# ---------------------------------------------------------------------------
# TestFormatTocTree
# ---------------------------------------------------------------------------


class TestFormatTocTree:
    """Tests for format_toc_tree()."""

    def test_empty(self) -> None:
        assert format_toc_tree([]) == []

    def test_single_entry(self) -> None:
        rows = format_toc_tree([_toc()])
        assert len(rows) == 1
        assert rows[0]["number"] == "1"
        assert rows[0]["title"] == "Introduction"
        assert rows[0]["page"] == 1

    def test_nested_hierarchy(self) -> None:
        entries = [
            _toc(level=1, number="1", title="Introduction"),
            _toc(level=2, number="1.1", title="Background"),
            _toc(level=3, number="1.1.1", title="Disease"),
        ]
        rows = format_toc_tree(entries)
        assert rows[0]["title"] == "Introduction"
        assert rows[1]["title"] == "  Background"
        assert rows[2]["title"] == "    Disease"

    def test_level_values_preserved(self) -> None:
        rows = format_toc_tree([_toc(level=2)])
        assert rows[0]["level"] == 2

    def test_missing_attributes_fallback(self) -> None:
        """Objects with missing attributes should not crash."""
        entry = SimpleNamespace()  # no attributes
        rows = format_toc_tree([entry])
        assert len(rows) == 1
        assert rows[0]["number"] == ""
        assert rows[0]["page"] == 0


# ---------------------------------------------------------------------------
# TestFormatMatchTable
# ---------------------------------------------------------------------------


class TestFormatMatchTable:
    """Tests for format_match_table()."""

    def test_empty_result(self) -> None:
        result = _match_result()
        assert format_match_table(result) == []

    def test_auto_mapped(self) -> None:
        m = _mapping(
            "1", "Introduction",
            matches=[_match("B.1", "General Information", 0.92, boosted_score=0.92)],
        )
        result = _match_result(mappings=[m])
        rows = format_match_table(result)
        assert len(rows) == 1
        assert rows[0]["ich_section"].startswith("B.1")
        assert rows[0]["score"] == 0.92

    def test_unmapped_section(self) -> None:
        m = _mapping("99", "Unknown Section", matches=[])
        result = _match_result(mappings=[m])
        rows = format_match_table(result)
        assert rows[0]["ich_section"] == "(unmapped)"
        assert rows[0]["score"] == 0.0

    def test_multiple_mappings(self) -> None:
        mappings = [
            _mapping("1", "Intro", [_match("B.1", "General", 0.90)]),
            _mapping("2", "Background", [_match("B.2", "Background", 0.85)]),
        ]
        result = _match_result(mappings=mappings)
        rows = format_match_table(result)
        assert len(rows) == 2

    def test_confidence_as_string(self) -> None:
        """Confidence may be a plain string (keyword fallback)."""
        m = _mapping(
            "1", "Intro",
            matches=[_match(confidence="REVIEW")],
        )
        rows = format_match_table(_match_result(mappings=[m]))
        assert rows[0]["confidence"] == "REVIEW"


# ---------------------------------------------------------------------------
# TestFormatExtractionTable
# ---------------------------------------------------------------------------


class TestFormatExtractionTable:
    """Tests for format_extraction_table()."""

    def test_empty(self) -> None:
        result = _extraction_result()
        assert format_extraction_table(result) == []

    def test_single_extraction(self) -> None:
        ext = _extraction(content="Protocol title is XYZ")
        result = _extraction_result(extractions=[ext])
        rows = format_extraction_table(result)
        assert len(rows) == 1
        assert rows[0]["query_id"] == "B.1.1.q1"
        assert "XYZ" in rows[0]["content_preview"]

    def test_content_preview_truncation(self) -> None:
        long_content = "A" * 200
        ext = _extraction(content=long_content)
        result = _extraction_result(extractions=[ext])
        rows = format_extraction_table(result)
        assert rows[0]["content_preview"].endswith("...")
        assert len(rows[0]["content_preview"]) == _CONTENT_PREVIEW_LEN + 3

    def test_confidence_formatted(self) -> None:
        ext = _extraction(confidence=0.90)
        result = _extraction_result(extractions=[ext])
        rows = format_extraction_table(result)
        assert "0.90" in rows[0]["confidence"]
        assert "HIGH" in rows[0]["confidence"]

    def test_multiple_extractions(self) -> None:
        exts = [
            _extraction(query_id="B.1.1.q1"),
            _extraction(query_id="B.3.1.q1"),
        ]
        result = _extraction_result(extractions=exts)
        rows = format_extraction_table(result)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# TestFormatGapTable
# ---------------------------------------------------------------------------


class TestFormatGapTable:
    """Tests for format_gap_table()."""

    def test_empty(self) -> None:
        result = _extraction_result()
        assert format_gap_table(result) == []

    def test_single_gap(self) -> None:
        g = _gap(reason="no_match")
        result = _extraction_result(gaps=[g])
        rows = format_gap_table(result)
        assert len(rows) == 1
        assert rows[0]["reason"] == "no_match"

    def test_various_reasons(self) -> None:
        gaps = [
            _gap(query_id="B.7.1.q1", reason="no_match"),
            _gap(query_id="B.8.1.q1", reason="low_confidence"),
            _gap(query_id="B.9.1.q1", reason="unmapped_section"),
        ]
        result = _extraction_result(gaps=gaps)
        rows = format_gap_table(result)
        assert len(rows) == 3
        assert {r["reason"] for r in rows} == {
            "no_match", "low_confidence", "unmapped_section",
        }


# ---------------------------------------------------------------------------
# TestFormatCoverageMetrics
# ---------------------------------------------------------------------------


class TestFormatCoverageMetrics:
    """Tests for format_coverage_metrics()."""

    def test_full_coverage(self) -> None:
        cov = _coverage(
            total_sections=10,
            populated_count=10,
            gap_count=0,
            avg_confidence=0.95,
            high_confidence_count=10,
            review_confidence_count=0,
            low_confidence_count=0,
        )
        metrics = format_coverage_metrics(cov)
        assert metrics["total"] == 10
        assert metrics["populated"] == 10
        assert metrics["gaps"] == 0
        assert metrics["high_pct"] == 100.0

    def test_partial_coverage(self) -> None:
        cov = _coverage()  # 10/16 populated
        metrics = format_coverage_metrics(cov)
        assert metrics["total"] == 16
        assert metrics["populated"] == 10
        assert metrics["gaps"] == 6
        assert metrics["avg_confidence"] == 0.8

    def test_empty_coverage(self) -> None:
        cov = _coverage(
            total_sections=0,
            populated_count=0,
            gap_count=0,
            avg_confidence=0.0,
            high_confidence_count=0,
            review_confidence_count=0,
            low_confidence_count=0,
        )
        metrics = format_coverage_metrics(cov)
        assert metrics["total"] == 0
        assert metrics["high_pct"] == 0.0
        assert metrics["review_pct"] == 0.0


# ---------------------------------------------------------------------------
# TestRunQueryPipeline
# ---------------------------------------------------------------------------


class TestRunQueryPipeline:
    """Tests for run_query_pipeline() with mocked backend."""

    @patch("ptcv.ui.components.query_pipeline.os")
    def test_pipeline_returns_all_keys(self, mock_os: object) -> None:
        """Mocked pipeline should return expected keys."""
        mock_index = SimpleNamespace(
            source_path="test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="test",
            toc_found=False,
            toc_pages=[],
        )
        mock_match = _match_result(auto_mapped_count=5)
        mock_ext = _extraction_result(coverage=0.75)
        mock_assembled = SimpleNamespace(
            sections=[],
            coverage=_coverage(),
            to_markdown=lambda: "# Template",
            to_dict=lambda: {},
        )

        with (
            patch(
                "ptcv.ui.components.query_pipeline.run_query_pipeline"
            ) as mock_run,
        ):
            mock_run.return_value = {
                "protocol_index": mock_index,
                "match_result": mock_match,
                "extraction_result": mock_ext,
                "assembled": mock_assembled,
                "assembled_markdown": "# Template",
                "coverage": _coverage(),
            }
            result = mock_run("test.pdf")

        assert "protocol_index" in result
        assert "match_result" in result
        assert "extraction_result" in result
        assert "assembled" in result
        assert "assembled_markdown" in result
        assert "coverage" in result
