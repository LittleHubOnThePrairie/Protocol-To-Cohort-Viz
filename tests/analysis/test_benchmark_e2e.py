"""Tests for end-to-end benchmark — PTCV-178.

Qualification phase: IQ/OQ
Risk tier: LOW
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.analysis.benchmark_e2e import (
    BenchmarkReport,
    DEFAULT_PROTOCOLS,
    ProtocolMetrics,
    _compute_aggregate,
    resolve_protocol_paths,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# TestResolveProtocolPaths
# ---------------------------------------------------------------------------


class TestResolveProtocolPaths:
    """Protocol path resolution from NCT IDs and file paths."""

    def test_none_returns_default_when_found(self, tmp_path: Path) -> None:
        """None protocols falls back to DEFAULT_PROTOCOLS."""
        # Create PDFs for first 2 default protocols.
        for nct in DEFAULT_PROTOCOLS[:2]:
            (tmp_path / f"{nct}_1.0.pdf").write_bytes(b"%PDF-fake")

        paths = resolve_protocol_paths(None, tmp_path)
        nct_ids = [p.stem.rsplit("_", 1)[0] for p in paths]
        for nct in DEFAULT_PROTOCOLS[:2]:
            assert nct in nct_ids

    def test_explicit_nct_id_resolved(self, tmp_path: Path) -> None:
        pdf = tmp_path / "NCT12345678_1.0.pdf"
        pdf.write_bytes(b"%PDF-fake")

        paths = resolve_protocol_paths(["NCT12345678"], tmp_path)
        assert len(paths) == 1
        assert paths[0] == pdf

    def test_explicit_pdf_path_resolved(self, tmp_path: Path) -> None:
        pdf = tmp_path / "custom.pdf"
        pdf.write_bytes(b"%PDF-fake")

        paths = resolve_protocol_paths([str(pdf)], tmp_path)
        assert len(paths) == 1
        assert paths[0] == pdf

    def test_limit_applied(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"NCT0000000{i}_1.0.pdf").write_bytes(b"%PDF")

        paths = resolve_protocol_paths(
            [f"NCT0000000{i}" for i in range(5)],
            tmp_path,
            limit=2,
        )
        assert len(paths) == 2

    def test_missing_protocol_skipped(self, tmp_path: Path) -> None:
        paths = resolve_protocol_paths(["NCT_NONEXISTENT"], tmp_path)
        assert len(paths) == 0


# ---------------------------------------------------------------------------
# TestComputeAggregate
# ---------------------------------------------------------------------------


class TestComputeAggregate:
    """Aggregate statistics computation."""

    def test_mean_median_min_max(self) -> None:
        results = [
            ProtocolMetrics(nct_id="A", file_sha="a", status="pass",
                            coverage_score=0.6),
            ProtocolMetrics(nct_id="B", file_sha="b", status="pass",
                            coverage_score=0.8),
            ProtocolMetrics(nct_id="C", file_sha="c", status="pass",
                            coverage_score=1.0),
        ]
        agg = _compute_aggregate(results, "coverage_score")
        assert agg["n"] == 3
        assert agg["min"] == 0.6
        assert agg["max"] == 1.0
        assert abs(agg["mean"] - 0.8) < 0.001
        assert agg["median"] == 0.8

    def test_empty_returns_zeros(self) -> None:
        agg = _compute_aggregate([], "coverage_score")
        assert agg == {
            "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0,
        }

    def test_skips_error_protocols(self) -> None:
        results = [
            ProtocolMetrics(nct_id="A", file_sha="a", status="pass",
                            coverage_score=0.9),
            ProtocolMetrics(nct_id="B", file_sha="b", status="error",
                            coverage_score=0.1),
        ]
        agg = _compute_aggregate(results, "coverage_score")
        assert agg["n"] == 1
        assert agg["mean"] == 0.9


# ---------------------------------------------------------------------------
# TestProtocolMetricsDataclass
# ---------------------------------------------------------------------------


class TestProtocolMetricsDataclass:
    """ProtocolMetrics dataclass behaviour."""

    def test_default_values(self) -> None:
        m = ProtocolMetrics(nct_id="NCT1", file_sha="abc", status="pass")
        assert m.text_block_count == 0
        assert m.coverage_score == 0.0
        assert m.stage_timings == {}
        assert m.missing_required_sections == []
        assert m.domain_row_counts == {}
        assert m.failed_stage is None

    def test_json_serializable(self) -> None:
        m = ProtocolMetrics(
            nct_id="NCT1",
            file_sha="abc",
            status="pass",
            stage_timings={"extraction": 1.5},
            missing_required_sections=["B.3"],
            domain_row_counts={"TS": 10, "TA": 5},
        )
        text = json.dumps(dataclasses.asdict(m))
        parsed = json.loads(text)
        assert parsed["nct_id"] == "NCT1"
        assert parsed["stage_timings"]["extraction"] == 1.5
        assert parsed["domain_row_counts"]["TS"] == 10


# ---------------------------------------------------------------------------
# TestRunBenchmarkIntegration
# ---------------------------------------------------------------------------


def _make_passing_metrics(nct_id: str = "NCT00112827") -> ProtocolMetrics:
    """Factory for a passing ProtocolMetrics with representative values."""
    return ProtocolMetrics(
        nct_id=nct_id,
        file_sha="abcd1234",
        status="pass",
        total_elapsed_seconds=5.0,
        stage_timings={
            "extraction": 1.0,
            "normalization": 0.1,
            "soa_extraction": 0.5,
            "retemplating": 2.0,
            "coverage_review": 0.2,
            "sdtm_generation": 0.8,
            "validation": 0.4,
        },
        text_block_count=100,
        table_count=5,
        format_detected="pdf",
        page_count=50,
        headings_promoted=20,
        section_count=12,
        format_verdict="ICH_E6R3",
        format_confidence=0.85,
        coverage_score=0.78,
        coverage_passed=True,
        domain_row_counts={"TS": 10, "TA": 3, "TE": 5},
        p21_error_count=2,
        tcg_passed=True,
        schedule_feasible=True,
    )


def _make_error_metrics(
    nct_id: str = "NCT_FAIL",
    stage: str = "soa_extraction",
) -> ProtocolMetrics:
    return ProtocolMetrics(
        nct_id=nct_id,
        file_sha="dead",
        status="error",
        failed_stage=stage,
        error_message="Something went wrong",
        total_elapsed_seconds=1.0,
        stage_timings={"extraction": 0.5},
    )


class TestRunBenchmarkIntegration:
    """Integration tests using mocked _run_one_protocol."""

    def test_writes_report_json(self, tmp_path: Path) -> None:
        pdf = tmp_path / "NCT00112827_1.0.pdf"
        pdf.write_bytes(b"%PDF")

        with patch(
            "ptcv.analysis.benchmark_e2e._run_one_protocol",
            return_value=_make_passing_metrics(),
        ):
            report = run_benchmark(
                protocol_paths=[pdf],
                output_dir=tmp_path / "out",
            )

        report_file = tmp_path / "out" / "benchmark_report.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert data["protocol_count"] == 1
        assert data["pass_count"] == 1

        proto_file = tmp_path / "out" / "protocols" / "NCT00112827.json"
        assert proto_file.exists()

    def test_correct_pass_error_counts(self, tmp_path: Path) -> None:
        pdfs = []
        for nct in ["NCT_A", "NCT_B"]:
            p = tmp_path / f"{nct}_1.0.pdf"
            p.write_bytes(b"%PDF")
            pdfs.append(p)

        returns = [
            _make_passing_metrics("NCT_A"),
            _make_error_metrics("NCT_B", "retemplating"),
        ]
        with patch(
            "ptcv.analysis.benchmark_e2e._run_one_protocol",
            side_effect=returns,
        ):
            report = run_benchmark(
                protocol_paths=pdfs,
                output_dir=tmp_path / "out",
            )

        assert report.pass_count == 1
        assert report.error_count == 1
        assert report.failures_by_stage == {"retemplating": 1}

    def test_output_dir_created(self, tmp_path: Path) -> None:
        pdf = tmp_path / "NCT_X_1.0.pdf"
        pdf.write_bytes(b"%PDF")
        out = tmp_path / "new" / "nested" / "dir"

        with patch(
            "ptcv.analysis.benchmark_e2e._run_one_protocol",
            return_value=_make_passing_metrics("NCT_X"),
        ):
            run_benchmark(protocol_paths=[pdf], output_dir=out)

        assert out.exists()
        assert (out / "benchmark_report.json").exists()


# ---------------------------------------------------------------------------
# TestBenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    """BenchmarkReport dataclass behaviour."""

    def test_json_serializable(self) -> None:
        report = BenchmarkReport(
            run_id="test_run",
            timestamp_utc="2026-03-09T00:00:00+00:00",
            pipeline_capabilities={"extraction_level": "E3"},
            protocol_count=1,
            pass_count=1,
            error_count=0,
            failures_by_stage={},
            total_elapsed_seconds=5.0,
            aggregate_metrics={"coverage_score": {"mean": 0.8}},
            protocol_results=[_make_passing_metrics()],
        )
        text = json.dumps(dataclasses.asdict(report))
        parsed = json.loads(text)
        assert parsed["run_id"] == "test_run"
        assert len(parsed["protocol_results"]) == 1
