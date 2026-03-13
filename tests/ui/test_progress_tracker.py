"""Tests for unified pipeline progress tracker (PTCV-179).

Pure-Python tests — no Streamlit dependency.
"""

from __future__ import annotations

import pytest

from ptcv.ui.components.progress_tracker import (
    UNIFIED_PIPELINE_STAGES,
    StageStatus,
    format_stage_display,
    map_classified_stages_to_unified,
    map_query_stages_to_unified,
    merge_pipeline_stages,
)


# ---------------------------------------------------------------------------
# TestUnifiedPipelineStages
# ---------------------------------------------------------------------------


class TestUnifiedPipelineStages:
    """Tests for UNIFIED_PIPELINE_STAGES constant."""

    def test_stage_count(self) -> None:
        assert len(UNIFIED_PIPELINE_STAGES) == 8

    def test_stage_keys_unique(self) -> None:
        keys = [s[0] for s in UNIFIED_PIPELINE_STAGES]
        assert len(set(keys)) == 8

    def test_stage_names_non_empty(self) -> None:
        for key, name in UNIFIED_PIPELINE_STAGES:
            assert key, "Stage key must not be empty"
            assert name, "Stage name must not be empty"

    def test_canonical_order(self) -> None:
        keys = [s[0] for s in UNIFIED_PIPELINE_STAGES]
        assert keys[0] == "pdf_extraction"
        assert keys[-1] == "soa_construction"
        assert "ml_classification" in keys
        assert "sonnet_classification" in keys


# ---------------------------------------------------------------------------
# TestMapQueryStages
# ---------------------------------------------------------------------------


class TestMapQueryStages:
    """Tests for map_query_stages_to_unified()."""

    def test_maps_all_four_stages(self) -> None:
        timings = {
            "Document Assembly": 1.5,
            "Section Classification": 3.2,
            "Query Extraction": 2.1,
            "Result Aggregation": 0.4,
        }
        result = map_query_stages_to_unified(timings)
        assert len(result) == 8
        complete = [s for s in result if s.status == "complete"]
        skipped = [s for s in result if s.status == "skipped"]
        assert len(complete) >= 4
        assert len(skipped) >= 2

    def test_ml_and_sonnet_skipped(self) -> None:
        timings = {
            "Document Assembly": 1.0,
            "Section Classification": 2.0,
            "Query Extraction": 1.5,
            "Result Aggregation": 0.3,
        }
        result = map_query_stages_to_unified(timings)
        by_key = {s.key: s for s in result}
        assert by_key["ml_classification"].status == "skipped"
        assert by_key["sonnet_classification"].status == "skipped"
        assert by_key["soa_construction"].status == "skipped"

    def test_empty_timings(self) -> None:
        result = map_query_stages_to_unified({})
        assert len(result) == 8
        assert all(
            s.status in ("pending", "skipped") for s in result
        )

    def test_source_pipeline_set(self) -> None:
        timings = {"Document Assembly": 1.0}
        result = map_query_stages_to_unified(timings)
        complete = [s for s in result if s.status == "complete"]
        for s in complete:
            assert s.source_pipeline == "query"

    def test_elapsed_seconds_captured(self) -> None:
        timings = {"Result Aggregation": 0.7}
        result = map_query_stages_to_unified(timings)
        by_key = {s.key: s for s in result}
        # Result Aggregation maps to document_assembly.
        assert by_key["document_assembly"].elapsed_seconds == 0.7


# ---------------------------------------------------------------------------
# TestMapClassifiedStages
# ---------------------------------------------------------------------------


class TestMapClassifiedStages:
    """Tests for map_classified_stages_to_unified()."""

    def test_with_cascade_stats(self) -> None:
        stats = {
            "total_sections": 10,
            "local_count": 8,
            "sonnet_count": 2,
            "local_pct": 0.8,
            "sonnet_pct": 0.2,
            "agreements": 1,
            "disagreements": 1,
        }
        result = map_classified_stages_to_unified(
            cascade_stats=stats,
        )
        assert len(result) == 8
        by_key = {s.key: s for s in result}
        ml = by_key["ml_classification"]
        assert ml.status == "complete"
        assert ml.metrics.get("local_count") == 8

    def test_sonnet_stage_complete_when_count_nonzero(self) -> None:
        stats = {"sonnet_count": 3, "agreements": 2, "disagreements": 1}
        result = map_classified_stages_to_unified(cascade_stats=stats)
        by_key = {s.key: s for s in result}
        assert by_key["sonnet_classification"].status == "complete"
        assert by_key["sonnet_classification"].metrics["sonnet_count"] == 3

    def test_sonnet_stage_skipped_when_zero(self) -> None:
        stats = {"sonnet_count": 0, "local_count": 10}
        result = map_classified_stages_to_unified(cascade_stats=stats)
        by_key = {s.key: s for s in result}
        assert by_key["sonnet_classification"].status == "skipped"

    def test_none_inputs_all_pending(self) -> None:
        result = map_classified_stages_to_unified(None, None)
        assert len(result) == 8
        assert all(
            s.status in ("pending", "skipped") for s in result
        )

    def test_query_stages_skipped(self) -> None:
        """Query-specific stages are skipped in classified path."""
        stats = {"local_count": 5}
        result = map_classified_stages_to_unified(cascade_stats=stats)
        by_key = {s.key: s for s in result}
        assert by_key["content_transforms"].status == "skipped"
        assert by_key["query_route"].status == "skipped"

    def test_with_stage_timings(self) -> None:
        timings = {
            "extraction": 2.0,
            "classification_cascade": 5.0,
            "assembly": 1.0,
        }
        result = map_classified_stages_to_unified(
            cascade_stats={"local_count": 5},
            stage_timings=timings,
        )
        by_key = {s.key: s for s in result}
        assert by_key["pdf_extraction"].elapsed_seconds == 2.0
        assert by_key["ml_classification"].elapsed_seconds == 5.0
        assert by_key["document_assembly"].elapsed_seconds == 1.0


# ---------------------------------------------------------------------------
# TestMergePipelineStages
# ---------------------------------------------------------------------------


class TestMergePipelineStages:
    """Tests for merge_pipeline_stages()."""

    def test_query_only(self) -> None:
        query = map_query_stages_to_unified({
            "Document Assembly": 1.0,
            "Section Classification": 2.0,
            "Query Extraction": 1.5,
            "Result Aggregation": 0.3,
        })
        result = merge_pipeline_stages(query, None)
        assert len(result) == 8
        assert result is query

    def test_classified_only(self) -> None:
        classified = map_classified_stages_to_unified(
            cascade_stats={"local_count": 5, "sonnet_count": 1},
        )
        result = merge_pipeline_stages(None, classified)
        assert len(result) == 8
        assert result is classified

    def test_both_none(self) -> None:
        result = merge_pipeline_stages(None, None)
        assert len(result) == 8
        assert all(s.status == "pending" for s in result)

    def test_both_prefers_complete(self) -> None:
        query = map_query_stages_to_unified({
            "Document Assembly": 1.0,
        })
        classified = map_classified_stages_to_unified(
            cascade_stats={"local_count": 5, "sonnet_count": 2},
        )
        result = merge_pipeline_stages(query, classified)
        assert len(result) == 8
        by_key = {s.key: s for s in result}
        # ML classification should come from classified (complete).
        assert by_key["ml_classification"].status == "complete"
        assert by_key["ml_classification"].source_pipeline == "classified"
        # PDF extraction should come from query (complete with timing).
        assert by_key["pdf_extraction"].status == "complete"

    def test_merged_keeps_metrics(self) -> None:
        query = map_query_stages_to_unified({})
        classified = map_classified_stages_to_unified(
            cascade_stats={
                "local_count": 8, "sonnet_count": 2,
                "agreements": 1,
            },
        )
        result = merge_pipeline_stages(query, classified)
        by_key = {s.key: s for s in result}
        assert by_key["ml_classification"].metrics.get("local_count") == 8


# ---------------------------------------------------------------------------
# TestFormatStageDisplay
# ---------------------------------------------------------------------------


class TestFormatStageDisplay:
    """Tests for format_stage_display()."""

    def test_complete_stage(self) -> None:
        stage = StageStatus(
            key="pdf_extraction", name="PDF Extraction",
            status="complete", elapsed_seconds=1.5,
            source_pipeline="query",
        )
        display = format_stage_display(stage)
        assert display["color"] == "green"
        assert display["label"] == "PDF Extraction"
        assert "1.5s" in display["detail_text"]
        assert "query" in display["detail_text"]

    def test_skipped_stage(self) -> None:
        stage = StageStatus(
            key="ml_classification", name="ML Classification",
            status="skipped",
        )
        display = format_stage_display(stage)
        assert display["color"] == "grey"
        assert display["detail_text"] == ""

    def test_error_stage(self) -> None:
        stage = StageStatus(
            key="sonnet_classification", name="Sonnet",
            status="error",
        )
        display = format_stage_display(stage)
        assert display["color"] == "red"

    def test_pending_stage(self) -> None:
        stage = StageStatus(
            key="soa_construction", name="SoA",
            status="pending",
        )
        display = format_stage_display(stage)
        assert display["color"] == "orange"

    def test_metrics_in_detail(self) -> None:
        stage = StageStatus(
            key="ml_classification", name="ML",
            status="complete",
            metrics={"local_count": 8, "sonnet_count": 2},
        )
        display = format_stage_display(stage)
        assert "local_count: 8" in display["detail_text"]
        assert "sonnet_count: 2" in display["detail_text"]

    def test_icon_present(self) -> None:
        for status in ("complete", "error", "pending", "skipped"):
            stage = StageStatus(
                key="test", name="Test", status=status,
            )
            display = format_stage_display(stage)
            assert display["icon"], f"No icon for {status}"
