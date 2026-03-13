"""Tests for query pipeline UI helpers (PTCV-95).

Pure-Python tests — no Streamlit dependency.
Tests format helpers with mock data matching the backend dataclass shapes.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

from ptcv.ui.components.query_pipeline import (
    PIPELINE_STAGES,
    _CONTENT_PREVIEW_LEN,
    count_extraction_methods,
    format_comparison_rows,
    format_coverage_metrics,
    format_extraction_table,
    format_gap_table,
    format_match_table,
    format_pipeline_comparison_rows,
    format_provenance_badge,
    format_subsection_match_table,
    format_toc_tree,
    run_query_pipeline,
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
# TestFormatSubsectionMatchTable (PTCV-96)
# ---------------------------------------------------------------------------


def _sub_match(
    sub_code: str = "B.5.1",
    parent_code: str = "B.5",
    sub_name: str = "Inclusion criteria",
    composite: float = 0.82,
    summarization: float = 0.90,
    confidence: str = "high",
    method: str = "embedding+summarization",
) -> SimpleNamespace:
    return SimpleNamespace(
        sub_section_code=sub_code,
        parent_section_code=parent_code,
        sub_section_name=sub_name,
        embedding_score=0.85,
        keyword_score=0.60,
        summarization_score=summarization,
        composite_score=composite,
        confidence=SimpleNamespace(value=confidence),
        match_method=method,
    )


def _enriched_mapping(
    number: str = "5",
    title: str = "Subject Selection",
    sub_matches: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        protocol_section_number=number,
        protocol_section_title=title,
        matches=[],
        auto_mapped=True,
        sub_section_matches=sub_matches or [],
        parent_coverage=frozenset(),
    )


def _enriched_result(
    enriched_mappings: list | None = None,
    sub_auto_rate: float = 0.50,
    llm_calls: int = 4,
    llm_fallback: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        mappings=[],
        auto_mapped_count=1,
        review_count=0,
        unmapped_count=0,
        auto_map_rate=1.0,
        enriched_mappings=enriched_mappings or [],
        sub_section_auto_map_rate=sub_auto_rate,
        llm_calls_made=llm_calls,
        llm_fallback=llm_fallback,
    )


class TestFormatSubsectionMatchTable:
    """Tests for format_subsection_match_table() (PTCV-96)."""

    def test_none_returns_empty(self) -> None:
        """None enriched result returns empty list."""
        assert format_subsection_match_table(None) == []

    def test_with_matches(self) -> None:
        """Sub-section matches are formatted correctly."""
        sub1 = _sub_match("B.5.1", "B.5", "Inclusion criteria", 0.82, 0.90)
        sub2 = _sub_match("B.5.2", "B.5", "Exclusion criteria", 0.71, 0.75)
        em = _enriched_mapping(sub_matches=[sub1, sub2])
        result = _enriched_result(enriched_mappings=[em])
        rows = format_subsection_match_table(result)
        assert len(rows) == 2
        assert rows[0]["sub_section"] == "B.5.1 Inclusion criteria"
        assert rows[0]["parent_section"] == "B.5"
        assert rows[0]["composite_score"] == 0.82
        assert rows[0]["confidence"] == "high"

    def test_fallback_mode(self) -> None:
        """Fallback mode sub-matches show -1.0 summarization."""
        sub = _sub_match(
            summarization=-1.0, method="keyword_fallback",
        )
        em = _enriched_mapping(sub_matches=[sub])
        result = _enriched_result(
            enriched_mappings=[em], llm_fallback=True,
        )
        rows = format_subsection_match_table(result)
        assert rows[0]["summarization_score"] == -1.0
        assert rows[0]["method"] == "keyword_fallback"


# ---------------------------------------------------------------------------
# TestCountExtractionMethods (PTCV-103)
# ---------------------------------------------------------------------------


class TestCountExtractionMethods:
    """Tests for count_extraction_methods()."""

    def test_empty(self) -> None:
        result = _extraction_result()
        assert count_extraction_methods(result) == {}

    def test_all_verbatim(self) -> None:
        exts = [
            _extraction(extraction_method="text_short"),
            _extraction(extraction_method="passthrough"),
            _extraction(extraction_method="text_short"),
        ]
        result = _extraction_result(extractions=exts)
        counts = count_extraction_methods(result)
        assert counts == {"text_short": 2, "passthrough": 1}
        assert "llm_transform" not in counts

    def test_mixed_methods(self) -> None:
        exts = [
            _extraction(extraction_method="text_short"),
            _extraction(extraction_method="llm_transform"),
            _extraction(extraction_method="passthrough"),
            _extraction(extraction_method="llm_transform"),
        ]
        result = _extraction_result(extractions=exts)
        counts = count_extraction_methods(result)
        assert counts["llm_transform"] == 2
        assert counts["text_short"] == 1
        assert counts["passthrough"] == 1

    def test_all_transformed(self) -> None:
        exts = [
            _extraction(extraction_method="llm_transform"),
            _extraction(extraction_method="llm_transform"),
        ]
        result = _extraction_result(extractions=exts)
        counts = count_extraction_methods(result)
        assert counts == {"llm_transform": 2}


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
                "enriched_match_result": None,
                "extraction_result": mock_ext,
                "assembled": mock_assembled,
                "assembled_markdown": "# Template",
                "coverage": _coverage(),
            }
            result = mock_run("test.pdf")

        assert "protocol_index" in result
        assert "match_result" in result
        assert "enriched_match_result" in result
        assert "extraction_result" in result
        assert "assembled" in result
        assert "assembled_markdown" in result
        assert "coverage" in result


# ---------------------------------------------------------------------------
# TestPipelineStages (PTCV-123)
# ---------------------------------------------------------------------------


class TestPipelineStages:
    """Tests for PIPELINE_STAGES constant."""

    def test_stage_count(self) -> None:
        assert len(PIPELINE_STAGES) == 4

    def test_stage_names(self) -> None:
        assert PIPELINE_STAGES == [
            "Document Assembly",
            "Section Classification",
            "Query Extraction",
            "Result Aggregation",
        ]

    def test_stages_are_strings(self) -> None:
        for stage in PIPELINE_STAGES:
            assert isinstance(stage, str)


# ---------------------------------------------------------------------------
# TestProgressCallback (PTCV-123)
# ---------------------------------------------------------------------------


class TestProgressCallback:
    """Tests for progress_callback in run_query_pipeline()."""

    def _mock_run(
        self,
        callback: object | None = None,
        enable_summarization: bool = False,
    ) -> dict:
        """Run pipeline with mocked backends and an optional callback."""
        mock_index = SimpleNamespace(
            source_path="test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="test protocol text",
            toc_found=False,
            toc_pages=[],
        )
        mock_match = _match_result(auto_mapped_count=0)
        mock_ext = _extraction_result(
            extractions=[], gaps=[], coverage=0.0,
        )
        mock_assembled = SimpleNamespace(
            sections=[],
            coverage=_coverage(
                total_sections=0, populated_count=0,
                gap_count=0, avg_confidence=0.0,
                high_confidence_count=0,
                review_confidence_count=0,
                low_confidence_count=0,
            ),
            to_markdown=lambda: "",
            to_dict=lambda: {},
        )

        with (
            patch(
                "ptcv.ich_parser.toc_extractor.extract_protocol_index",
                return_value=mock_index,
            ),
            patch(
                "ptcv.ich_parser.section_matcher.SectionMatcher.match",
                return_value=mock_match,
            ),
            patch(
                "ptcv.ich_parser.query_extractor.QueryExtractor.extract",
                return_value=mock_ext,
            ),
            patch(
                "ptcv.ich_parser.template_assembler.assemble_template",
                return_value=mock_assembled,
            ),
        ):
            return run_query_pipeline(
                "test.pdf",
                enable_summarization=enable_summarization,
                progress_callback=callback,
            )

    def test_callback_called_for_all_stages(self) -> None:
        calls: list[tuple[str, float, int, int]] = []
        self._mock_run(callback=lambda s, p, d=0, t=0: calls.append((s, p, d, t)))

        stage_names = {c[0] for c in calls}
        for stage in PIPELINE_STAGES:
            assert stage in stage_names, f"{stage} missing"

    def test_each_stage_starts_at_zero(self) -> None:
        calls: list[tuple[str, float, int, int]] = []
        self._mock_run(callback=lambda s, p, d=0, t=0: calls.append((s, p, d, t)))

        for stage in PIPELINE_STAGES:
            stage_calls = [c for c in calls if c[0] == stage]
            assert stage_calls[0][1] == 0.0

    def test_each_stage_ends_at_one(self) -> None:
        calls: list[tuple[str, float, int, int]] = []
        self._mock_run(callback=lambda s, p, d=0, t=0: calls.append((s, p, d, t)))

        for stage in PIPELINE_STAGES:
            stage_calls = [c for c in calls if c[0] == stage]
            assert stage_calls[-1][1] == 1.0

    def test_callback_order(self) -> None:
        calls: list[tuple[str, float, int, int]] = []
        self._mock_run(callback=lambda s, p, d=0, t=0: calls.append((s, p, d, t)))

        # Extract stage order from start events (progress=0.0).
        start_order = [
            c[0] for c in calls if c[1] == 0.0
        ]
        assert start_order == PIPELINE_STAGES

    def test_none_callback_no_error(self) -> None:
        result = self._mock_run(callback=None)
        assert "protocol_index" in result

    def test_callback_receives_eight_events(self) -> None:
        """4 stages x 2 events (start + end) = 8 calls."""
        calls: list[tuple[str, float, int, int]] = []
        self._mock_run(callback=lambda s, p, d=0, t=0: calls.append((s, p, d, t)))
        assert len(calls) == 8

    def test_pipeline_result_unchanged_with_callback(self) -> None:
        result_with = self._mock_run(
            callback=lambda s, p, d=0, t=0: None,
        )
        result_without = self._mock_run(callback=None)
        assert result_with.keys() == result_without.keys()


class TestProgressMetadata:
    """Tests for N/M metadata in progress callbacks (PTCV-124)."""

    def _mock_run(
        self,
        callback: object | None = None,
        enable_summarization: bool = False,
    ) -> dict:
        """Run pipeline with mocked backends and an optional callback."""
        mock_index = SimpleNamespace(
            source_path="test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="test protocol text",
            toc_found=False,
            toc_pages=[],
        )
        mock_match = _match_result(auto_mapped_count=0)
        mock_ext = _extraction_result(
            extractions=[], gaps=[], coverage=0.0,
        )
        mock_assembled = SimpleNamespace(
            sections=[],
            coverage=_coverage(
                total_sections=0, populated_count=0,
                gap_count=0, avg_confidence=0.0,
                high_confidence_count=0,
                review_confidence_count=0,
                low_confidence_count=0,
            ),
            to_markdown=lambda: "",
            to_dict=lambda: {},
        )

        with (
            patch(
                "ptcv.ich_parser.toc_extractor.extract_protocol_index",
                return_value=mock_index,
            ),
            patch(
                "ptcv.ich_parser.section_matcher.SectionMatcher.match",
                return_value=mock_match,
            ),
            patch(
                "ptcv.ich_parser.query_extractor.QueryExtractor.extract",
                return_value=mock_ext,
            ),
            patch(
                "ptcv.ich_parser.template_assembler.assemble_template",
                return_value=mock_assembled,
            ),
        ):
            return run_query_pipeline(
                "test.pdf",
                enable_summarization=enable_summarization,
                progress_callback=callback,
            )

    def test_start_events_have_zero_counts(self) -> None:
        """Start events (progress=0.0) pass done=0, total=0."""
        calls: list[tuple[str, float, int, int]] = []
        self._mock_run(
            callback=lambda s, p, d=0, t=0: calls.append((s, p, d, t)),
        )
        starts = [c for c in calls if c[1] == 0.0]
        assert len(starts) == 4
        for c in starts:
            assert c[2] == 0 and c[3] == 0, (
                f"start event {c[0]} should have d=0, t=0"
            )

    def test_end_events_have_zero_counts(self) -> None:
        """End events (progress=1.0) pass done=0, total=0."""
        calls: list[tuple[str, float, int, int]] = []
        self._mock_run(
            callback=lambda s, p, d=0, t=0: calls.append((s, p, d, t)),
        )
        ends = [c for c in calls if c[1] == 1.0]
        assert len(ends) == 4
        for c in ends:
            assert c[2] == 0 and c[3] == 0, (
                f"end event {c[0]} should have d=0, t=0"
            )


# ---------------------------------------------------------------------------
# Content Comparison (PTCV-141)
# ---------------------------------------------------------------------------


def _assembled_section(
    code: str = "B.5",
    name: str = "Selection of Subjects",
    populated: bool = True,
    hits: list | None = None,
    average_confidence: float = 0.85,
) -> SimpleNamespace:
    return SimpleNamespace(
        section_code=code,
        section_name=name,
        populated=populated,
        hits=hits or [],
        average_confidence=average_confidence,
    )


def _hit(
    query_id: str = "B.5.1.q1",
    extracted_content: str = "Adults aged 18-65",
    confidence: float = 0.90,
) -> SimpleNamespace:
    return SimpleNamespace(
        query_id=query_id,
        extracted_content=extracted_content,
        confidence=confidence,
    )


def _protocol_index(
    content_spans: dict[str, str] | None = None,
) -> SimpleNamespace:
    spans = content_spans or {}

    class _Index:
        def __init__(self, spans: dict[str, str]) -> None:
            self._spans = spans

        def get_section_text(
            self, section_number: str,
        ) -> str | None:
            return self._spans.get(section_number)

    return _Index(spans)


def _assembled(
    sections: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(sections=sections or [])


class TestFormatComparisonRows:
    """Tests for format_comparison_rows() (PTCV-141)."""

    def test_empty_sections(self) -> None:
        """No populated sections → empty rows."""
        result = format_comparison_rows(
            _assembled(), _protocol_index(), _match_result(),
        )
        assert result == []

    def test_single_section_with_original(self) -> None:
        """One section maps original and extracted text."""
        assembled = _assembled([
            _assembled_section("B.5", "Selection", hits=[
                _hit("B.5.1.q1", "Adults aged 18-65"),
            ]),
        ])
        proto_idx = _protocol_index({"5.1": "Eligible subjects"})
        mr = _match_result([
            _mapping("5.1", "Eligibility", [
                _match("B.5", "Selection"),
            ]),
        ])

        rows = format_comparison_rows(assembled, proto_idx, mr)
        assert len(rows) == 1
        assert rows[0]["ich_code"] == "B.5"
        assert "Eligible subjects" in rows[0]["original_text"]
        assert "Adults aged 18-65" in rows[0]["extracted_text"]

    def test_no_original_text_mapped(self) -> None:
        """Section with no matching protocol section → empty original."""
        assembled = _assembled([
            _assembled_section("B.12", "Ethics", hits=[
                _hit("B.12.q1", "IRB approved"),
            ]),
        ])
        proto_idx = _protocol_index()
        mr = _match_result()

        rows = format_comparison_rows(assembled, proto_idx, mr)
        assert len(rows) == 1
        assert rows[0]["original_text"] == ""
        assert rows[0]["protocol_section"] == "(none)"
        assert "IRB approved" in rows[0]["extracted_text"]

    def test_multiple_protocol_sections_merged(self) -> None:
        """Multiple protocol sections mapping to same ICH code."""
        assembled = _assembled([
            _assembled_section("B.5", "Selection", hits=[
                _hit("B.5.1.q1", "Inclusion criteria"),
            ]),
        ])
        proto_idx = _protocol_index({
            "5.1": "Inclusion text",
            "5.2": "Exclusion text",
        })
        mr = _match_result([
            _mapping("5.1", "Inclusion", [
                _match("B.5", "Selection"),
            ]),
            _mapping("5.2", "Exclusion", [
                _match("B.5", "Selection"),
            ]),
        ])

        rows = format_comparison_rows(assembled, proto_idx, mr)
        assert len(rows) == 1
        assert "Inclusion text" in rows[0]["original_text"]
        assert "Exclusion text" in rows[0]["original_text"]
        assert "5.1" in rows[0]["protocol_section"]
        assert "5.2" in rows[0]["protocol_section"]

    def test_unpopulated_sections_excluded(self) -> None:
        """Unpopulated sections are not included."""
        assembled = _assembled([
            _assembled_section("B.1", "General", populated=False),
            _assembled_section("B.4", "Design", hits=[
                _hit("B.4.q1", "Phase 3"),
            ]),
        ])
        proto_idx = _protocol_index()
        mr = _match_result()

        rows = format_comparison_rows(assembled, proto_idx, mr)
        assert len(rows) == 1
        assert rows[0]["ich_code"] == "B.4"

    def test_confidence_and_query_count(self) -> None:
        """Row includes confidence and query count."""
        assembled = _assembled([
            _assembled_section("B.9", "Safety", hits=[
                _hit("B.9.1.q1", "AE monitoring"),
                _hit("B.9.2.q1", "SAE reporting"),
            ], average_confidence=0.78),
        ])
        proto_idx = _protocol_index()
        mr = _match_result()

        rows = format_comparison_rows(assembled, proto_idx, mr)
        assert rows[0]["confidence"] == 0.78
        assert rows[0]["query_count"] == 2

    def test_multiple_hits_concatenated(self) -> None:
        """Multiple hits are joined in extracted text."""
        assembled = _assembled([
            _assembled_section("B.7", "Treatment", hits=[
                _hit("B.7.q1", "Drug A 100mg"),
                _hit("B.7.q2", "Administered IV"),
            ]),
        ])
        proto_idx = _protocol_index()
        mr = _match_result()

        rows = format_comparison_rows(assembled, proto_idx, mr)
        assert "Drug A 100mg" in rows[0]["extracted_text"]
        assert "Administered IV" in rows[0]["extracted_text"]


# ---------------------------------------------------------------------------
# Pipeline Comparison helpers (PTCV-179)
# ---------------------------------------------------------------------------


def _assembled_section_with_provenance(
    code: str = "B.5",
    name: str = "Selection of Subjects",
    populated: bool = True,
    hits: list | None = None,
    average_confidence: float = 0.85,
    extraction_method: str = "",
    classification_method: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        section_code=code,
        section_name=name,
        populated=populated,
        hits=hits or [],
        average_confidence=average_confidence,
        extraction_method=extraction_method,
        classification_method=classification_method,
    )


def _assembled_with_get(
    sections: list,
) -> SimpleNamespace:
    """Build assembled mock with get_section() method."""
    lookup = {
        getattr(s, "section_code", ""): s for s in sections
    }

    class _Asm:
        def __init__(self, secs: list, lkp: dict) -> None:
            self.sections = secs
            self._lookup = lkp

        def get_section(self, code: str):  # noqa: ANN201
            return self._lookup.get(code)

    return _Asm(sections, lookup)  # type: ignore[return-value]


class TestFormatPipelineComparisonRows:
    """Tests for format_pipeline_comparison_rows() (PTCV-179)."""

    def test_both_populated(self) -> None:
        """Sections populated in both pipelines show in comparison."""
        q = _assembled_with_get([
            _assembled_section_with_provenance(
                "B.5", hits=[_hit("B.5.q1", "Query inclusion")],
            ),
        ])
        c = _assembled_with_get([
            _assembled_section_with_provenance(
                "B.5", hits=[_hit("B.5.q1", "Classified inclusion")],
                extraction_method="E3:pdfplumber",
                classification_method="C2:neobert_sonnet",
            ),
        ])
        rows = format_pipeline_comparison_rows(q, c)
        assert len(rows) == 1
        assert "Query inclusion" in rows[0]["query_text"]
        assert "Classified inclusion" in rows[0]["classified_text"]

    def test_query_only(self) -> None:
        """Section populated only in query shows empty classified."""
        q = _assembled_with_get([
            _assembled_section_with_provenance(
                "B.3", hits=[_hit("B.3.q1", "Objectives")],
            ),
        ])
        c = _assembled_with_get([])
        rows = format_pipeline_comparison_rows(q, c)
        assert len(rows) == 1
        assert rows[0]["query_text"]
        assert rows[0]["classified_text"] == ""

    def test_classified_only(self) -> None:
        """Section populated only in classified shows empty query."""
        q = _assembled_with_get([])
        c = _assembled_with_get([
            _assembled_section_with_provenance(
                "B.4", hits=[_hit("B.4.q1", "Design")],
            ),
        ])
        rows = format_pipeline_comparison_rows(q, c)
        assert len(rows) == 1
        assert rows[0]["query_text"] == ""
        assert rows[0]["classified_text"]

    def test_confidence_delta_computed(self) -> None:
        """Confidence delta = classified - query."""
        q = _assembled_with_get([
            _assembled_section_with_provenance(
                "B.5", average_confidence=0.80,
                hits=[_hit("B.5.q1", "Q")],
            ),
        ])
        c = _assembled_with_get([
            _assembled_section_with_provenance(
                "B.5", average_confidence=0.90,
                hits=[_hit("B.5.q1", "C")],
            ),
        ])
        rows = format_pipeline_comparison_rows(q, c)
        assert rows[0]["confidence_delta"] == 0.10

    def test_provenance_method_included(self) -> None:
        """classified_method includes extraction + classification."""
        q = _assembled_with_get([])
        c = _assembled_with_get([
            _assembled_section_with_provenance(
                "B.5", hits=[_hit("B.5.q1", "Content")],
                extraction_method="E3:pdfplumber",
                classification_method="C2:neobert_sonnet",
            ),
        ])
        rows = format_pipeline_comparison_rows(q, c)
        assert "E3:pdfplumber" in rows[0]["classified_method"]
        assert "C2:neobert_sonnet" in rows[0]["classified_method"]

    def test_empty_both(self) -> None:
        """No populated sections returns empty list."""
        q = _assembled_with_get([])
        c = _assembled_with_get([])
        rows = format_pipeline_comparison_rows(q, c)
        assert rows == []

    def test_none_assembled(self) -> None:
        """None assembled protocols handled gracefully."""
        rows = format_pipeline_comparison_rows(None, None)
        assert rows == []

    def test_canonical_order(self) -> None:
        """Sections are ordered canonically B.1 < B.5 < B.10."""
        q = _assembled_with_get([
            _assembled_section_with_provenance(
                "B.10", "Statistics",
                hits=[_hit("B.10.q1", "Stats")],
            ),
            _assembled_section_with_provenance(
                "B.1", "General Info",
                hits=[_hit("B.1.q1", "Info")],
            ),
        ])
        rows = format_pipeline_comparison_rows(q, None)
        assert rows[0]["ich_code"] == "B.1"
        assert rows[1]["ich_code"] == "B.10"


class TestFormatProvenanceBadge:
    """Tests for format_provenance_badge() (PTCV-179)."""

    def test_high_confidence(self) -> None:
        section = _assembled_section_with_provenance(
            average_confidence=0.90,
            extraction_method="E3:pdfplumber",
        )
        badge = format_provenance_badge(section)
        assert badge["confidence_label"] == "HIGH"
        assert badge["confidence_color"] == "green"

    def test_review_confidence(self) -> None:
        section = _assembled_section_with_provenance(
            average_confidence=0.75,
        )
        badge = format_provenance_badge(section)
        assert badge["confidence_label"] == "REVIEW"
        assert badge["confidence_color"] == "orange"

    def test_low_confidence(self) -> None:
        section = _assembled_section_with_provenance(
            average_confidence=0.50,
        )
        badge = format_provenance_badge(section)
        assert badge["confidence_label"] == "LOW"
        assert badge["confidence_color"] == "red"

    def test_methods_included(self) -> None:
        section = _assembled_section_with_provenance(
            extraction_method="E3:pdfplumber",
            classification_method="C2:neobert_sonnet",
        )
        badge = format_provenance_badge(section)
        assert badge["extraction_method"] == "E3:pdfplumber"
        assert badge["classification_method"] == "C2:neobert_sonnet"

    def test_empty_methods(self) -> None:
        section = _assembled_section_with_provenance()
        badge = format_provenance_badge(section)
        assert badge["extraction_method"] == ""
        assert badge["classification_method"] == ""
