"""Tests for SQLite analysis data store (PTCV-147)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.analysis.batch_runner import ProtocolResult
from ptcv.analysis.data_store import AnalysisStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> AnalysisStore:
    """Create a fresh in-memory-like store per test."""
    db_path = tmp_path / "test_results.db"
    s = AnalysisStore(db_path)
    yield s
    s.close()


def _make_result(
    nct_id: str = "NCT00001234",
    status: str = "pass",
    ich_code: str = "B.1",
    confidence: float = 0.85,
    boosted: float = 0.87,
    auto_mapped: bool = True,
    coverage_pct: int = 10,
    original_text: str = "Original content",
    extracted_text: str = "Extracted content",
    match_quality: str = "good",
) -> ProtocolResult:
    return ProtocolResult(
        nct_id=nct_id,
        file_sha="abc123",
        status=status,
        elapsed_seconds=2.5,
        toc_section_count=15,
        coverage={
            "total_sections": 14,
            "populated_count": coverage_pct,
            "gap_count": 14 - coverage_pct,
            "average_confidence": confidence,
            "total_queries": 70,
            "answered_queries": 50,
            "gap_sections": [],
        },
        section_matches=[{
            "protocol_section_number": "1",
            "protocol_section_title": "General Info",
            "ich_section_code": ich_code,
            "ich_section_name": "General Information",
            "similarity_score": confidence - 0.02,
            "boosted_score": boosted,
            "confidence": "HIGH" if confidence >= 0.80 else "LOW",
            "match_method": "keyword",
            "auto_mapped": auto_mapped,
        }],
        query_extractions=[{
            "query_id": f"{ich_code}.1.q1",
            "section_id": f"{ich_code}.1",
            "content": "Extracted query answer",
            "confidence": confidence,
            "extraction_method": "regex",
            "source_section": "1. General",
        }],
        comparison_pairs=[{
            "ich_section_code": ich_code,
            "ich_section_name": "General Information",
            "original_text": original_text,
            "extracted_text": extracted_text,
            "confidence": confidence,
            "match_quality": match_quality,
        }],
        stage_timings={"document_assembly": 1.0},
    )


def _populate_run(
    store: AnalysisStore,
    run_id: str = "run_001",
    protocols: list[ProtocolResult] | None = None,
) -> None:
    """Helper to create a run with protocol results."""
    store.create_run(run_id, {
        "timestamp": "2026-03-07T12:00:00Z",
        "pipeline_version": "abc123",
    })
    if protocols is None:
        protocols = [_make_result()]
    for p in protocols:
        store.store_protocol_result(run_id, p)
    store.finalize_run(run_id, {
        "protocol_count": len(protocols),
        "pass_count": sum(
            1 for p in protocols if p.status == "pass"
        ),
        "error_count": sum(
            1 for p in protocols if p.status == "error"
        ),
        "elapsed_seconds": 10.0,
    })


# ---------------------------------------------------------------------------
# Tests: Schema and initialization
# ---------------------------------------------------------------------------

class TestSchemaInit:

    def test_creates_db_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "new.db"
        s = AnalysisStore(db_path)
        assert db_path.exists()
        s.close()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "a" / "b" / "results.db"
        s = AnalysisStore(db_path)
        assert db_path.exists()
        s.close()

    def test_schema_version_set(self, store: AnalysisStore) -> None:
        cur = store._conn.execute(
            "SELECT version FROM schema_version"
        )
        assert cur.fetchone()["version"] == 2

    def test_reopening_preserves_data(
        self, tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "persist.db"
        s1 = AnalysisStore(db_path)
        _populate_run(s1, "run_001")
        s1.close()

        s2 = AnalysisStore(db_path)
        assert s2.get_run("run_001") is not None
        s2.close()


# ---------------------------------------------------------------------------
# Tests: Writer API (ResultStore interface)
# ---------------------------------------------------------------------------

class TestWriterAPI:

    def test_create_run(self, store: AnalysisStore) -> None:
        store.create_run("run_001", {
            "timestamp": "2026-03-07T12:00:00Z",
            "pipeline_version": "abc123",
        })
        run = store.get_run("run_001")
        assert run is not None
        assert run["pipeline_version"] == "abc123"

    def test_store_protocol_result(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store)
        assert store.has_result("run_001", "NCT00001234")

    def test_has_result_false(self, store: AnalysisStore) -> None:
        _populate_run(store)
        assert not store.has_result("run_001", "NCT99999999")

    def test_finalize_run_updates_counts(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00000001"),
            _make_result(nct_id="NCT00000002"),
            _make_result(nct_id="NCT00000003", status="error"),
        ])
        run = store.get_run("run_001")
        assert run["protocol_count"] == 3
        assert run["pass_count"] == 2
        assert run["error_count"] == 1

    def test_section_matches_stored(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store)
        summary = store.get_protocol_summary("run_001", "NCT00001234")
        assert len(summary["section_matches"]) == 1
        sm = summary["section_matches"][0]
        assert sm["ich_section_code"] == "B.1"

    def test_coverage_stored(self, store: AnalysisStore) -> None:
        _populate_run(store)
        summary = store.get_protocol_summary("run_001", "NCT00001234")
        assert summary["coverage"]["total_sections"] == 14

    def test_comparison_pairs_stored(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store)
        summary = store.get_protocol_summary("run_001", "NCT00001234")
        assert len(summary["comparison_pairs"]) == 1
        cp = summary["comparison_pairs"][0]
        assert cp["match_quality"] == "good"

    def test_multiple_protocols_in_run(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00000001"),
            _make_result(nct_id="NCT00000002"),
            _make_result(nct_id="NCT00000003"),
        ])
        protocols = store.list_protocols("run_001")
        assert len(protocols) == 3

    def test_error_result_stored(
        self, store: AnalysisStore,
    ) -> None:
        result = ProtocolResult(
            nct_id="NCT_ERR",
            file_sha="abc",
            status="error",
            error_message="Parse failed",
            elapsed_seconds=0.5,
        )
        _populate_run(store, protocols=[result])
        summary = store.get_protocol_summary("run_001", "NCT_ERR")
        assert summary["status"] == "error"
        assert summary["error_message"] == "Parse failed"


# ---------------------------------------------------------------------------
# Tests: get_latest_run_id
# ---------------------------------------------------------------------------

class TestGetLatestRunId:

    def test_returns_latest(self, store: AnalysisStore) -> None:
        store.create_run("run_001", {"timestamp": "2026-03-01"})
        store.create_run("run_002", {"timestamp": "2026-03-07"})
        assert store.get_latest_run_id() == "run_002"

    def test_returns_none_when_empty(
        self, store: AnalysisStore,
    ) -> None:
        assert store.get_latest_run_id() is None


# ---------------------------------------------------------------------------
# Tests: list_runs
# ---------------------------------------------------------------------------

class TestListRuns:

    def test_returns_runs_sorted_by_timestamp(
        self, store: AnalysisStore,
    ) -> None:
        store.create_run("run_001", {"timestamp": "2026-03-01"})
        store.create_run("run_002", {"timestamp": "2026-03-07"})
        store.create_run("run_003", {"timestamp": "2026-03-05"})
        runs = store.list_runs()
        assert len(runs) == 3
        assert runs[0]["run_id"] == "run_002"
        assert runs[1]["run_id"] == "run_003"
        assert runs[2]["run_id"] == "run_001"

    def test_empty_when_no_runs(
        self, store: AnalysisStore,
    ) -> None:
        assert store.list_runs() == []


# ---------------------------------------------------------------------------
# Tests: list_ich_sections
# ---------------------------------------------------------------------------

class TestListIchSections:

    def test_returns_distinct_sections(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00000001", ich_code="B.5"),
            _make_result(nct_id="NCT00000002", ich_code="B.5"),
            _make_result(nct_id="NCT00000003", ich_code="B.1"),
        ])
        sections = store.list_ich_sections("run_001")
        codes = [s["ich_section_code"] for s in sections]
        assert codes == ["B.1", "B.5"]

    def test_empty_when_no_results(
        self, store: AnalysisStore,
    ) -> None:
        store.create_run("run_empty", {"timestamp": "2026-01-01"})
        assert store.list_ich_sections("run_empty") == []


# ---------------------------------------------------------------------------
# Tests: get_low_confidence_sections
# ---------------------------------------------------------------------------

class TestLowConfidenceSections:

    def test_finds_low_confidence(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, protocols=[
            _make_result(
                nct_id="NCT00000001",
                ich_code="B.9", boosted=0.45,
            ),
            _make_result(
                nct_id="NCT00000002",
                ich_code="B.9", boosted=0.50,
            ),
            _make_result(
                nct_id="NCT00000003",
                ich_code="B.1", boosted=0.90,
            ),
        ])
        low = store.get_low_confidence_sections("run_001", 0.60)
        assert len(low) == 1
        assert low[0]["ich_section_code"] == "B.9"
        assert low[0]["protocol_count"] == 2

    def test_sorted_ascending(self, store: AnalysisStore) -> None:
        _populate_run(store, protocols=[
            _make_result(
                nct_id="NCT00000001",
                ich_code="B.9", boosted=0.50,
            ),
            _make_result(
                nct_id="NCT00000002",
                ich_code="B.5", boosted=0.40,
            ),
        ])
        low = store.get_low_confidence_sections("run_001", 0.60)
        assert low[0]["ich_section_code"] == "B.5"
        assert low[1]["ich_section_code"] == "B.9"

    def test_empty_when_all_high(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, protocols=[
            _make_result(boosted=0.90),
        ])
        low = store.get_low_confidence_sections("run_001", 0.60)
        assert low == []


# ---------------------------------------------------------------------------
# Tests: get_section_stats
# ---------------------------------------------------------------------------

class TestSectionStats:

    def test_stats_per_section(self, store: AnalysisStore) -> None:
        _populate_run(store, protocols=[
            _make_result(
                nct_id="NCT00000001", ich_code="B.1", boosted=0.90,
            ),
            _make_result(
                nct_id="NCT00000002", ich_code="B.1", boosted=0.85,
            ),
            _make_result(
                nct_id="NCT00000003", ich_code="B.5", boosted=0.70,
            ),
        ])
        stats = store.get_section_stats("run_001")
        assert len(stats) == 2
        b1 = next(s for s in stats if s["ich_section_code"] == "B.1")
        assert b1["protocol_count"] == 2
        assert b1["avg_boosted"] == pytest.approx(0.875, abs=0.01)

    def test_hit_rate_calculated(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00000001", ich_code="B.1"),
            _make_result(nct_id="NCT00000002", ich_code="B.1"),
            _make_result(nct_id="NCT00000003", ich_code="B.5"),
        ])
        stats = store.get_section_stats("run_001")
        b1 = next(s for s in stats if s["ich_section_code"] == "B.1")
        # 2 out of 3 protocols have B.1
        assert b1["hit_rate"] == pytest.approx(2 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: get_coverage_distribution
# ---------------------------------------------------------------------------

class TestCoverageDistribution:

    def test_buckets(self, store: AnalysisStore) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00000001", coverage_pct=12),
            _make_result(nct_id="NCT00000002", coverage_pct=8),
            _make_result(nct_id="NCT00000003", coverage_pct=3),
        ])
        dist = store.get_coverage_distribution("run_001")
        assert dist["protocol_count"] == 3
        # 12/14=86%, 8/14=57%, 3/14=21%
        assert dist["buckets"]["75-100%"] == 1
        assert dist["buckets"]["50-75%"] == 1
        assert dist["buckets"]["0-25%"] == 1

    def test_empty_run(self, store: AnalysisStore) -> None:
        store.create_run("run_empty", {"timestamp": "2026-01-01"})
        dist = store.get_coverage_distribution("run_empty")
        assert dist["protocol_count"] == 0


# ---------------------------------------------------------------------------
# Tests: compare_runs
# ---------------------------------------------------------------------------

class TestCompareRuns:

    def test_detects_improvement(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, run_id="run_a", protocols=[
            _make_result(
                nct_id="NCT00000001", ich_code="B.5", boosted=0.50,
            ),
        ])
        _populate_run(store, run_id="run_b", protocols=[
            _make_result(
                nct_id="NCT00000001", ich_code="B.5", boosted=0.75,
            ),
        ])
        diff = store.compare_runs("run_a", "run_b")
        b5 = next(
            s for s in diff["sections"]
            if s["ich_section_code"] == "B.5"
        )
        assert b5["status"] == "improved"
        assert b5["delta"] > 0

    def test_detects_regression(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, run_id="run_a", protocols=[
            _make_result(
                nct_id="NCT00000001", ich_code="B.9", boosted=0.80,
            ),
        ])
        _populate_run(store, run_id="run_b", protocols=[
            _make_result(
                nct_id="NCT00000001", ich_code="B.9", boosted=0.60,
            ),
        ])
        diff = store.compare_runs("run_a", "run_b")
        b9 = next(
            s for s in diff["sections"]
            if s["ich_section_code"] == "B.9"
        )
        assert b9["status"] == "regressed"
        assert b9["delta"] < 0

    def test_unchanged_within_threshold(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store, run_id="run_a", protocols=[
            _make_result(
                nct_id="NCT00000001", ich_code="B.1", boosted=0.85,
            ),
        ])
        _populate_run(store, run_id="run_b", protocols=[
            _make_result(
                nct_id="NCT00000001", ich_code="B.1", boosted=0.855,
            ),
        ])
        diff = store.compare_runs("run_a", "run_b")
        b1 = next(
            s for s in diff["sections"]
            if s["ich_section_code"] == "B.1"
        )
        assert b1["status"] == "unchanged"


# ---------------------------------------------------------------------------
# Tests: get_comparison_pairs
# ---------------------------------------------------------------------------

class TestGetComparisonPairs:

    def test_by_section(self, store: AnalysisStore) -> None:
        _populate_run(store, protocols=[
            _make_result(
                nct_id="NCT00001234",
                ich_code="B.5",
                original_text="Eligibility criteria",
                extracted_text="Inclusion: age 18+",
            ),
        ])
        pairs = store.get_comparison_pairs(
            "run_001", "NCT00001234", "B.5",
        )
        assert len(pairs) == 1
        assert "Eligibility" in pairs[0]["original_text"]
        assert "Inclusion" in pairs[0]["extracted_text"]

    def test_all_sections(self, store: AnalysisStore) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00001234", ich_code="B.1"),
        ])
        pairs = store.get_comparison_pairs(
            "run_001", "NCT00001234",
        )
        assert len(pairs) == 1

    def test_wrong_nct_returns_empty(
        self, store: AnalysisStore,
    ) -> None:
        _populate_run(store)
        pairs = store.get_comparison_pairs(
            "run_001", "NCT99999999", "B.1",
        )
        assert pairs == []


# ---------------------------------------------------------------------------
# Tests: list_protocols
# ---------------------------------------------------------------------------

class TestListProtocols:

    def test_lists_all(self, store: AnalysisStore) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00000001"),
            _make_result(nct_id="NCT00000002"),
            _make_result(nct_id="NCT00000003", status="error"),
        ])
        all_p = store.list_protocols("run_001")
        assert len(all_p) == 3

    def test_filter_by_status(self, store: AnalysisStore) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00000001"),
            _make_result(nct_id="NCT00000002", status="error"),
        ])
        errors = store.list_protocols("run_001", status="error")
        assert len(errors) == 1
        assert errors[0]["nct_id"] == "NCT00000002"


# ---------------------------------------------------------------------------
# Tests: get_protocol_summary
# ---------------------------------------------------------------------------

class TestProtocolSummary:

    def test_full_summary(self, store: AnalysisStore) -> None:
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00001234"),
        ])
        summary = store.get_protocol_summary(
            "run_001", "NCT00001234",
        )
        assert summary["nct_id"] == "NCT00001234"
        assert summary["status"] == "pass"
        assert "section_matches" in summary
        assert "coverage" in summary
        assert "comparison_pairs" in summary

    def test_missing_protocol(self, store: AnalysisStore) -> None:
        _populate_run(store)
        summary = store.get_protocol_summary(
            "run_001", "NCT99999999",
        )
        assert summary == {}


# ---------------------------------------------------------------------------
# Tests: misclassification patterns
# ---------------------------------------------------------------------------

class TestMisclassificationPatterns:

    def test_finds_repeated_mappings(
        self, store: AnalysisStore,
    ) -> None:
        # Same protocol title mapped to same ICH code across protocols
        _populate_run(store, protocols=[
            _make_result(nct_id="NCT00000001", ich_code="B.7"),
            _make_result(nct_id="NCT00000002", ich_code="B.7"),
            _make_result(nct_id="NCT00000003", ich_code="B.7"),
        ])
        patterns = store.get_misclassification_patterns("run_001")
        # "General Info" → "B.7" appears 3 times
        assert len(patterns) >= 1
        assert patterns[0]["occurrence_count"] >= 2
