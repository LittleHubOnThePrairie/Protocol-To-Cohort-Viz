"""Tests for pipeline integration harness (PTCV-150)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.analysis.batch_runner import ProtocolResult
from ptcv.analysis.data_store import AnalysisStore
from ptcv.analysis.integration import (
    AnalysisHarness,
    detect_regressions,
    generate_regression_report,
    generate_summary_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    nct_id: str = "NCT00001234",
    ich_code: str = "B.1",
    title: str = "General Information",
    boosted: float = 0.85,
    auto_mapped: bool = True,
    match_quality: str = "good",
    confidence: float = 0.85,
) -> ProtocolResult:
    return ProtocolResult(
        nct_id=nct_id,
        file_sha="abc123",
        status="pass",
        elapsed_seconds=2.5,
        toc_section_count=15,
        coverage={
            "total_sections": 14,
            "populated_count": 10,
            "gap_count": 4,
            "average_confidence": confidence,
            "total_queries": 70,
            "answered_queries": 50,
            "gap_sections": [],
        },
        section_matches=[{
            "protocol_section_number": "1",
            "protocol_section_title": title,
            "ich_section_code": ich_code,
            "ich_section_name": f"Section {ich_code}",
            "similarity_score": confidence,
            "boosted_score": boosted,
            "confidence": "HIGH" if auto_mapped else "LOW",
            "match_method": "embedding",
            "auto_mapped": auto_mapped,
        }],
        query_extractions=[{
            "query_id": "q1",
            "section_id": ich_code,
            "query_text": "Test query?",
            "answer": "Test answer",
            "confidence": confidence,
            "source_section": "1",
        }],
        comparison_pairs=[{
            "ich_section_code": ich_code,
            "protocol_section_numbers": "1",
            "original_text": "Original content",
            "extracted_text": "Extracted content",
            "match_quality": match_quality,
        }],
        stage_timings={"total": 2.5},
        error_message=None,
    )


def _populate(
    store: AnalysisStore,
    run_id: str = "run_001",
    results: list[ProtocolResult] | None = None,
) -> str:
    store.create_run(run_id, {
        "pipeline_version": "v0.1.0",
        "config_hash": "cfg",
    })
    if results is None:
        results = [
            _make_result(nct_id="NCT11111111", ich_code="B.1",
                         boosted=0.90),
            _make_result(nct_id="NCT22222222", ich_code="B.5",
                         title="Eligibility", boosted=0.45,
                         auto_mapped=False, match_quality="poor"),
            _make_result(nct_id="NCT33333333", ich_code="B.1",
                         title="Synopsis", boosted=0.75),
        ]
    for r in results:
        store.store_protocol_result(run_id, r)
    pass_count = sum(1 for r in results if r.status == "pass")
    store.finalize_run(run_id, {
        "elapsed_seconds": 30.0,
        "protocol_count": len(results),
        "pass_count": pass_count,
        "fail_count": 0,
        "error_count": 0,
    })
    return run_id


@pytest.fixture()
def store(tmp_path: Path) -> AnalysisStore:
    db = tmp_path / "test_int.db"
    s = AnalysisStore(db)
    yield s
    s.close()


@pytest.fixture()
def populated_store(store: AnalysisStore) -> AnalysisStore:
    _populate(store)
    return store


# ---------------------------------------------------------------------------
# detect_regressions
# ---------------------------------------------------------------------------

class TestDetectRegressions:
    def test_no_regressions(self) -> None:
        comparison = {
            "sections": [
                {"ich_section_code": "B.1", "run_a_avg_score": 0.80,
                 "run_b_avg_score": 0.85, "delta": 0.05},
            ],
        }
        result = detect_regressions(comparison)
        assert result == []

    def test_detects_regression(self) -> None:
        comparison = {
            "sections": [
                {"ich_section_code": "B.9", "run_a_avg_score": 0.52,
                 "run_b_avg_score": 0.48, "delta": -0.04},
            ],
        }
        result = detect_regressions(comparison)
        assert len(result) == 1
        assert result[0]["ich_section_code"] == "B.9"
        assert result[0]["delta"] == -0.04
        assert result[0]["severity"] == "warning"

    def test_critical_severity(self) -> None:
        comparison = {
            "sections": [
                {"ich_section_code": "B.3", "run_a_avg_score": 0.70,
                 "run_b_avg_score": 0.55, "delta": -0.15},
            ],
        }
        result = detect_regressions(comparison)
        assert result[0]["severity"] == "critical"

    def test_custom_threshold(self) -> None:
        comparison = {
            "sections": [
                {"ich_section_code": "B.1", "run_a_avg_score": 0.80,
                 "run_b_avg_score": 0.77, "delta": -0.03},
            ],
        }
        # With default threshold (0.02), should detect
        assert len(detect_regressions(comparison, threshold=0.02)) == 1
        # With higher threshold, should not detect
        assert len(detect_regressions(comparison, threshold=0.05)) == 0

    def test_sorted_by_delta(self) -> None:
        comparison = {
            "sections": [
                {"ich_section_code": "B.1", "run_a_avg_score": 0.80,
                 "run_b_avg_score": 0.75, "delta": -0.05},
                {"ich_section_code": "B.9", "run_a_avg_score": 0.60,
                 "run_b_avg_score": 0.40, "delta": -0.20},
            ],
        }
        result = detect_regressions(comparison)
        assert result[0]["ich_section_code"] == "B.9"
        assert result[1]["ich_section_code"] == "B.1"

    def test_empty_sections(self) -> None:
        assert detect_regressions({"sections": []}) == []
        assert detect_regressions({}) == []


# ---------------------------------------------------------------------------
# generate_summary_report
# ---------------------------------------------------------------------------

class TestGenerateSummaryReport:
    def test_basic_report(
        self, populated_store: AnalysisStore,
    ) -> None:
        report = generate_summary_report(populated_store, "run_001")
        assert "## Batch Run Summary" in report
        assert "run_001" in report
        assert "Pass rate" in report
        assert "Avg coverage" in report

    def test_contains_problem_sections(
        self, populated_store: AnalysisStore,
    ) -> None:
        report = generate_summary_report(populated_store, "run_001")
        assert "Problem Sections" in report
        assert "Best Sections" in report

    def test_coverage_distribution(
        self, populated_store: AnalysisStore,
    ) -> None:
        report = generate_summary_report(populated_store, "run_001")
        assert "Coverage Distribution" in report

    def test_nonexistent_run(self, store: AnalysisStore) -> None:
        report = generate_summary_report(store, "nope")
        assert "Error" in report

    def test_top_n_limits(
        self, populated_store: AnalysisStore,
    ) -> None:
        report_1 = generate_summary_report(
            populated_store, "run_001", top_n=1,
        )
        report_5 = generate_summary_report(
            populated_store, "run_001", top_n=5,
        )
        assert "Top-1" in report_1
        assert "Top-5" in report_5


# ---------------------------------------------------------------------------
# generate_regression_report
# ---------------------------------------------------------------------------

class TestGenerateRegressionReport:
    def test_no_regressions(self, store: AnalysisStore) -> None:
        _populate(store, run_id="run_a", results=[
            _make_result(nct_id="NCT11111111", boosted=0.80),
        ])
        _populate(store, run_id="run_b", results=[
            _make_result(nct_id="NCT11111111", boosted=0.85),
        ])
        report = generate_regression_report(
            store, "run_a", "run_b",
        )
        assert "No regressions detected" in report

    def test_with_regression(self, store: AnalysisStore) -> None:
        _populate(store, run_id="run_a", results=[
            _make_result(nct_id="NCT11111111", ich_code="B.9",
                         boosted=0.60),
        ])
        _populate(store, run_id="run_b", results=[
            _make_result(nct_id="NCT11111111", ich_code="B.9",
                         boosted=0.40),
        ])
        report = generate_regression_report(
            store, "run_a", "run_b",
        )
        assert "Regressions" in report
        assert "B.9" in report

    def test_with_improvements(self, store: AnalysisStore) -> None:
        _populate(store, run_id="run_a", results=[
            _make_result(nct_id="NCT11111111", boosted=0.60),
        ])
        _populate(store, run_id="run_b", results=[
            _make_result(nct_id="NCT11111111", boosted=0.80),
        ])
        report = generate_regression_report(
            store, "run_a", "run_b",
        )
        assert "Improvements" in report

    def test_coverage_delta_in_report(
        self, store: AnalysisStore,
    ) -> None:
        _populate(store, run_id="run_a", results=[
            _make_result(nct_id="NCT11111111"),
        ])
        _populate(store, run_id="run_b", results=[
            _make_result(nct_id="NCT11111111"),
        ])
        report = generate_regression_report(
            store, "run_a", "run_b",
        )
        assert "Coverage" in report
        assert "Confidence" in report


# ---------------------------------------------------------------------------
# AnalysisHarness
# ---------------------------------------------------------------------------

class TestAnalysisHarness:
    def test_lazy_open(self, tmp_path: Path) -> None:
        db = tmp_path / "harness.db"
        harness = AnalysisHarness(db)
        assert harness._store is None
        # Accessing store triggers creation
        _ = harness.store
        assert harness._store is not None
        harness.close()

    def test_list_runs(self, tmp_path: Path) -> None:
        db = tmp_path / "harness.db"
        store = AnalysisStore(db)
        _populate(store)
        store.close()

        harness = AnalysisHarness(db)
        runs = harness.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run_001"
        harness.close()

    def test_get_run_summary(self, tmp_path: Path) -> None:
        db = tmp_path / "harness.db"
        store = AnalysisStore(db)
        _populate(store)
        store.close()

        harness = AnalysisHarness(db)
        report = harness.get_run_summary("run_001")
        assert "Batch Run Summary" in report
        harness.close()

    def test_get_regression_report(
        self, tmp_path: Path,
    ) -> None:
        db = tmp_path / "harness.db"
        store = AnalysisStore(db)
        _populate(store, run_id="run_a", results=[
            _make_result(nct_id="NCT11111111", boosted=0.80),
        ])
        _populate(store, run_id="run_b", results=[
            _make_result(nct_id="NCT11111111", boosted=0.85),
        ])
        store.close()

        harness = AnalysisHarness(db)
        report = harness.get_regression_report("run_a", "run_b")
        assert "Regression Report" in report
        harness.close()

    def test_get_regressions(self, tmp_path: Path) -> None:
        db = tmp_path / "harness.db"
        store = AnalysisStore(db)
        _populate(store, run_id="run_a", results=[
            _make_result(nct_id="NCT11111111", ich_code="B.9",
                         boosted=0.60),
        ])
        _populate(store, run_id="run_b", results=[
            _make_result(nct_id="NCT11111111", ich_code="B.9",
                         boosted=0.40),
        ])
        store.close()

        harness = AnalysisHarness(db)
        regs = harness.get_regressions("run_a", "run_b")
        assert len(regs) == 1
        assert regs[0]["ich_section_code"] == "B.9"
        harness.close()

    def test_get_comparison_pairs(
        self, tmp_path: Path,
    ) -> None:
        db = tmp_path / "harness.db"
        store = AnalysisStore(db)
        _populate(store)
        store.close()

        harness = AnalysisHarness(db)
        pairs = harness.get_comparison_pairs(
            "run_001", "NCT11111111",
        )
        assert len(pairs) >= 1
        harness.close()

    def test_list_protocols(self, tmp_path: Path) -> None:
        db = tmp_path / "harness.db"
        store = AnalysisStore(db)
        _populate(store)
        store.close()

        harness = AnalysisHarness(db)
        protocols = harness.list_protocols("run_001")
        assert len(protocols) == 3
        harness.close()

    def test_close_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "harness.db"
        harness = AnalysisHarness(db)
        harness.close()
        harness.close()  # Should not raise
