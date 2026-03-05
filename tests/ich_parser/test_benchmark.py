"""Tests for benchmark query-driven vs text-first classification (PTCV-93).

Feature: Benchmark query-driven vs text-first classification

  Scenario: Content coverage comparison
  Scenario: Section accuracy measured against ground truth
  Scenario: Coherence comparison
  Scenario: Benchmark runs across full corpus
  Scenario: Results inform pipeline selection
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.benchmark import (
    APPENDIX_B_SECTION_COUNT,
    COHERENCE_RATIO_THRESHOLD,
    COVERAGE_DELTA_THRESHOLD,
    GAP_RATE_THRESHOLD,
    HIGH_CONFIDENCE,
    HIGH_CONFIDENCE_PCT_THRESHOLD,
    LOW_CONFIDENCE,
    BenchmarkReport,
    PipelineMetrics,
    ProtocolBenchmark,
    compare_protocol,
    compute_section_accuracy,
    compute_section_agreement,
    generate_recommendation,
    generate_report,
    run_benchmark,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_metrics(
    coverage_pct: float = 50.0,
    avg_confidence: float = 0.80,
    avg_span_length: float = 500.0,
    gap_rate: float = 0.5,
    section_count: int = 8,
    section_codes: set[str] | None = None,
    high_confidence_pct: float = 60.0,
    medium_confidence_pct: float = 25.0,
    low_confidence_pct: float = 15.0,
    total_chars_classified: int = 5000,
    total_chars: int = 10000,
) -> PipelineMetrics:
    if section_codes is None:
        section_codes = {
            f"B.{i}" for i in range(1, section_count + 1)
        }
    return PipelineMetrics(
        coverage_pct=coverage_pct,
        avg_confidence=avg_confidence,
        avg_span_length=avg_span_length,
        gap_rate=gap_rate,
        section_count=section_count,
        section_codes=section_codes,
        high_confidence_pct=high_confidence_pct,
        medium_confidence_pct=medium_confidence_pct,
        low_confidence_pct=low_confidence_pct,
        total_chars_classified=total_chars_classified,
        total_chars=total_chars,
    )


# -----------------------------------------------------------------------
# Scenario: Content coverage comparison
# -----------------------------------------------------------------------


class TestCoverageComparison:
    """Given the same protocol processed by text-first and query-driven,
    When coverage is measured via span mapper,
    Then query-driven coverage exceeds text-first by >= 30 pp.
    """

    def test_coverage_delta_computed(self):
        tf = _make_metrics(coverage_pct=25.0)
        qd = _make_metrics(coverage_pct=60.0)
        result = compare_protocol("NCT001", "test.pdf", tf, qd)
        assert result.coverage_delta == 35.0

    def test_coverage_delta_negative(self):
        tf = _make_metrics(coverage_pct=70.0)
        qd = _make_metrics(coverage_pct=50.0)
        result = compare_protocol("NCT001", "test.pdf", tf, qd)
        assert result.coverage_delta == -20.0

    def test_coverage_delta_threshold_constant(self):
        assert COVERAGE_DELTA_THRESHOLD == 30.0

    def test_coverage_in_report(self):
        benchmarks = [
            compare_protocol(
                "NCT001",
                "a.pdf",
                _make_metrics(coverage_pct=20.0),
                _make_metrics(coverage_pct=55.0),
            ),
            compare_protocol(
                "NCT002",
                "b.pdf",
                _make_metrics(coverage_pct=30.0),
                _make_metrics(coverage_pct=65.0),
            ),
        ]
        report = generate_report(benchmarks)
        # Avg delta = (35 + 35) / 2 = 35
        assert report.avg_coverage_delta == 35.0

    def test_coverage_with_none_pipeline(self):
        result = compare_protocol(
            "NCT001", "test.pdf",
            _make_metrics(coverage_pct=40.0), None,
        )
        assert result.coverage_delta == 0.0
        assert result.error == "Query-driven pipeline failed"


# -----------------------------------------------------------------------
# Scenario: Section accuracy measured against ground truth
# -----------------------------------------------------------------------


class TestSectionAccuracy:
    """Given a manually annotated protocol (ground truth),
    When query-driven extraction is compared to ground truth,
    Then section accuracy is >= 80%.
    """

    def test_perfect_accuracy(self):
        predicted = {"B.1", "B.2", "B.3", "B.4", "B.5"}
        truth = {"B.1", "B.2", "B.3", "B.4", "B.5"}
        assert compute_section_accuracy(predicted, truth) == 1.0

    def test_partial_accuracy(self):
        predicted = {"B.1", "B.2", "B.3", "B.4", "B.5"}
        truth = {"B.1", "B.2", "B.3", "B.6", "B.7"}
        # 3/5 correct
        assert compute_section_accuracy(predicted, truth) == 0.6

    def test_high_accuracy_meets_threshold(self):
        predicted = {"B.1", "B.2", "B.3", "B.4", "B.5"}
        truth = {
            "B.1", "B.2", "B.3", "B.4", "B.6", "B.7", "B.8",
        }
        # 4/5 correct
        accuracy = compute_section_accuracy(predicted, truth)
        assert accuracy >= 0.80

    def test_empty_predicted(self):
        assert compute_section_accuracy(set(), {"B.1"}) == 0.0

    def test_empty_truth(self):
        assert compute_section_accuracy({"B.1"}, set()) == 0.0

    def test_section_agreement_jaccard(self):
        tf = _make_metrics(
            section_codes={"B.1", "B.2", "B.3"},
        )
        qd = _make_metrics(
            section_codes={"B.2", "B.3", "B.4"},
        )
        # Jaccard = |{B.2, B.3}| / |{B.1, B.2, B.3, B.4}| = 2/4
        agreement = compute_section_agreement(tf, qd)
        assert agreement == 0.5

    def test_section_agreement_identical(self):
        codes = {"B.1", "B.2", "B.3"}
        tf = _make_metrics(section_codes=codes)
        qd = _make_metrics(section_codes=codes)
        assert compute_section_agreement(tf, qd) == 1.0

    def test_section_agreement_no_overlap(self):
        tf = _make_metrics(section_codes={"B.1", "B.2"})
        qd = _make_metrics(section_codes={"B.3", "B.4"})
        assert compute_section_agreement(tf, qd) == 0.0


# -----------------------------------------------------------------------
# Scenario: Coherence comparison
# -----------------------------------------------------------------------


class TestCoherenceComparison:
    """Given extraction results from both pipelines,
    When average span length is compared,
    Then query-driven average span is >= 2x text-first average.
    """

    def test_coherence_ratio_computed(self):
        tf = _make_metrics(avg_span_length=200.0)
        qd = _make_metrics(avg_span_length=600.0)
        result = compare_protocol("NCT001", "test.pdf", tf, qd)
        assert result.coherence_ratio == 3.0

    def test_coherence_ratio_below_threshold(self):
        tf = _make_metrics(avg_span_length=400.0)
        qd = _make_metrics(avg_span_length=500.0)
        result = compare_protocol("NCT001", "test.pdf", tf, qd)
        assert result.coherence_ratio == 1.25
        assert result.coherence_ratio < COHERENCE_RATIO_THRESHOLD

    def test_coherence_ratio_zero_text_first(self):
        tf = _make_metrics(avg_span_length=0.0)
        qd = _make_metrics(avg_span_length=500.0)
        result = compare_protocol("NCT001", "test.pdf", tf, qd)
        assert result.coherence_ratio == 0.0

    def test_coherence_threshold_constant(self):
        assert COHERENCE_RATIO_THRESHOLD == 2.0

    def test_aggregate_coherence(self):
        benchmarks = [
            compare_protocol(
                "NCT001", "a.pdf",
                _make_metrics(avg_span_length=200.0),
                _make_metrics(avg_span_length=500.0),
            ),
            compare_protocol(
                "NCT002", "b.pdf",
                _make_metrics(avg_span_length=300.0),
                _make_metrics(avg_span_length=900.0),
            ),
        ]
        report = generate_report(benchmarks)
        # Avg = (2.5 + 3.0) / 2 = 2.75
        assert report.avg_coherence_ratio == 2.75


# -----------------------------------------------------------------------
# Scenario: Benchmark runs across full corpus
# -----------------------------------------------------------------------


class TestFullCorpusBenchmark:
    """Given the qualifying protocols,
    When both pipelines process each protocol,
    Then a comparative report is produced with per-protocol
    and aggregate metrics.
    """

    def test_report_has_all_protocols(self):
        benchmarks = [
            compare_protocol(
                f"NCT{i:03d}", f"p{i}.pdf",
                _make_metrics(), _make_metrics(),
            )
            for i in range(15)
        ]
        report = generate_report(benchmarks)
        assert len(report.protocols) == 15

    def test_report_aggregates_present(self):
        benchmarks = [
            compare_protocol(
                "NCT001", "a.pdf",
                _make_metrics(), _make_metrics(),
            ),
        ]
        report = generate_report(benchmarks)
        assert report.aggregate_text_first is not None
        assert report.aggregate_query_driven is not None

    def test_report_handles_partial_failures(self):
        benchmarks = [
            compare_protocol(
                "NCT001", "a.pdf",
                _make_metrics(), _make_metrics(),
            ),
            compare_protocol(
                "NCT002", "b.pdf", None, _make_metrics(),
            ),
            compare_protocol(
                "NCT003", "c.pdf", _make_metrics(), None,
            ),
        ]
        report = generate_report(benchmarks)
        assert len(report.protocols) == 3
        # Only NCT001 contributes to avg deltas
        assert report.aggregate_text_first is not None
        assert report.aggregate_query_driven is not None

    def test_report_json_serializable(self):
        benchmarks = [
            compare_protocol(
                "NCT001", "a.pdf",
                _make_metrics(), _make_metrics(),
            ),
        ]
        report = generate_report(benchmarks)
        d = report.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_report_markdown_has_sections(self):
        benchmarks = [
            compare_protocol(
                "NCT001", "a.pdf",
                _make_metrics(coverage_pct=20.0),
                _make_metrics(coverage_pct=55.0),
            ),
        ]
        report = generate_report(benchmarks)
        md = report.to_markdown()
        assert "## Summary" in md
        assert "## Recommendation" in md
        assert "## Aggregate Metrics" in md
        assert "## Per-Protocol Results" in md
        assert "NCT001" in md

    def test_report_empty_corpus(self):
        report = generate_report([])
        assert len(report.protocols) == 0
        assert report.aggregate_text_first is None
        assert report.aggregate_query_driven is None
        assert report.recommendation == "keep_text_first"

    @patch("ptcv.ich_parser.benchmark.extract_protocol_index")
    @patch("ptcv.ich_parser.benchmark.run_text_first")
    @patch("ptcv.ich_parser.benchmark.run_query_driven")
    def test_run_benchmark_from_manifest(
        self, mock_qd, mock_tf, mock_extract,
    ):
        """Verify run_benchmark loads manifest and processes each."""
        mock_index = MagicMock()
        mock_index.full_text = "sample text"
        mock_extract.return_value = mock_index
        mock_tf.return_value = _make_metrics(coverage_pct=30.0)
        mock_qd.return_value = _make_metrics(coverage_pct=65.0)

        manifest = [
            {"nct_id": "NCT001"},
            {"nct_id": "NCT002"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            data_dir = Path(tmpdir) / "pdfs"
            data_dir.mkdir()
            (data_dir / "NCT001.pdf").write_text("fake")
            (data_dir / "NCT002.pdf").write_text("fake")

            report = run_benchmark(
                manifest_path, data_dir, max_protocols=2,
            )

        assert len(report.protocols) == 2
        assert all(
            p.text_first is not None for p in report.protocols
        )
        assert all(
            p.query_driven is not None for p in report.protocols
        )

    @patch("ptcv.ich_parser.benchmark.extract_protocol_index")
    def test_run_benchmark_missing_pdf(self, mock_extract):
        """Verify missing PDFs are handled gracefully."""
        manifest = [{"nct_id": "NCT_MISSING"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            data_dir = Path(tmpdir) / "pdfs"
            data_dir.mkdir()

            report = run_benchmark(manifest_path, data_dir)

        assert len(report.protocols) == 1
        assert "not found" in report.protocols[0].error


# -----------------------------------------------------------------------
# Scenario: Results inform pipeline selection
# -----------------------------------------------------------------------


class TestPipelineRecommendation:
    """Given the benchmark report,
    When metrics are reviewed,
    Then a recommendation is made on whether query-driven should
    replace or supplement text-first.
    """

    def test_replace_recommendation(self):
        """All 4 criteria met → replace."""
        rec, reason = generate_recommendation(
            avg_coverage_delta=35.0,
            avg_coherence_ratio=2.5,
            qd_high_confidence_pct=60.0,
            qd_gap_rate=0.20,
        )
        assert rec == "replace"
        assert "4/4 criteria" in reason

    def test_supplement_recommendation(self):
        """2 criteria met → supplement."""
        rec, reason = generate_recommendation(
            avg_coverage_delta=35.0,
            avg_coherence_ratio=1.5,
            qd_high_confidence_pct=60.0,
            qd_gap_rate=0.50,
        )
        assert rec == "supplement"
        assert "2/4 criteria" in reason

    def test_keep_text_first_recommendation(self):
        """0-1 criteria met → keep text-first."""
        rec, reason = generate_recommendation(
            avg_coverage_delta=10.0,
            avg_coherence_ratio=1.2,
            qd_high_confidence_pct=30.0,
            qd_gap_rate=0.50,
        )
        assert rec == "keep_text_first"

    def test_three_criteria_is_replace(self):
        """Exactly 3 criteria met → replace."""
        rec, _ = generate_recommendation(
            avg_coverage_delta=35.0,
            avg_coherence_ratio=2.5,
            qd_high_confidence_pct=55.0,
            qd_gap_rate=0.40,  # fails gap rate
        )
        assert rec == "replace"

    def test_recommendation_reason_lists_criteria(self):
        rec, reason = generate_recommendation(
            avg_coverage_delta=35.0,
            avg_coherence_ratio=2.5,
            qd_high_confidence_pct=60.0,
            qd_gap_rate=0.20,
        )
        assert "coverage delta" in reason
        assert "coherence ratio" in reason
        assert "high-confidence" in reason
        assert "gap rate" in reason

    def test_recommendation_in_report(self):
        benchmarks = [
            compare_protocol(
                "NCT001", "a.pdf",
                _make_metrics(
                    coverage_pct=20.0, avg_span_length=200.0,
                ),
                _make_metrics(
                    coverage_pct=60.0, avg_span_length=600.0,
                    high_confidence_pct=60.0, gap_rate=0.20,
                ),
            ),
        ]
        report = generate_report(benchmarks)
        assert report.recommendation in {
            "replace", "supplement", "keep_text_first",
        }
        assert len(report.recommendation_reason) > 0

    def test_recommendation_in_markdown(self):
        benchmarks = [
            compare_protocol(
                "NCT001", "a.pdf",
                _make_metrics(), _make_metrics(),
            ),
        ]
        report = generate_report(benchmarks)
        md = report.to_markdown()
        assert report.recommendation.upper() in md

    def test_confidence_thresholds(self):
        """Verify confidence threshold constants are correct."""
        assert HIGH_CONFIDENCE == 0.85
        assert LOW_CONFIDENCE == 0.70
        assert GAP_RATE_THRESHOLD == 0.30
        assert HIGH_CONFIDENCE_PCT_THRESHOLD == 50.0


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge case tests."""

    def test_both_pipelines_none(self):
        result = compare_protocol("NCT001", "test.pdf", None, None)
        assert result.error == "Both pipelines failed"
        assert result.coverage_delta == 0.0
        assert result.coherence_ratio == 0.0

    def test_metrics_section_codes_in_dict(self):
        m = _make_metrics(section_codes={"B.3", "B.1", "B.2"})
        benchmarks = [
            compare_protocol("NCT001", "a.pdf", m, m),
        ]
        report = generate_report(benchmarks)
        d = report.to_dict()
        # Section codes should be sorted in JSON output
        assert d["aggregate_text_first"]["section_codes"] == [
            "B.1", "B.2", "B.3",
        ]

    def test_single_protocol_report(self):
        benchmarks = [
            compare_protocol(
                "NCT001", "a.pdf",
                _make_metrics(
                    coverage_pct=10.0, section_count=2,
                    gap_rate=0.875,
                ),
                _make_metrics(
                    coverage_pct=50.0, section_count=10,
                    gap_rate=0.375,
                ),
            ),
        ]
        report = generate_report(benchmarks)
        assert report.avg_coverage_delta == 40.0

    def test_appendix_b_section_count(self):
        assert APPENDIX_B_SECTION_COUNT == 16
