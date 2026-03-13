"""Tests for analysis CLI subcommands (PTCV-149)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.analysis.batch_runner import ProtocolResult
from ptcv.analysis.data_store import AnalysisStore
from ptcv.analysis.analysis_cli import (
    cmd_compare,
    cmd_export,
    cmd_gaps,
    cmd_low_confidence,
    cmd_overview,
    cmd_patterns,
    cmd_protocol_report,
    cmd_section_report,
    main,
    _format_output,
    _to_markdown,
    _to_table,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_result(
    nct_id: str = "NCT00001234",
    status: str = "pass",
    ich_code: str = "B.1",
    title: str = "General Information",
    confidence: float = 0.85,
    boosted: float = 0.87,
    auto_mapped: bool = True,
    coverage_pct: int = 10,
    original_text: str = "Original content here",
    extracted_text: str = "Extracted content here",
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
            "original_text": original_text,
            "extracted_text": extracted_text,
            "match_quality": match_quality,
        }],
        stage_timings={"total": 2.5},
        error_message=None,
    )


def _populate_store(
    store: AnalysisStore,
    run_id: str = "run_001",
    results: list[ProtocolResult] | None = None,
) -> str:
    """Insert a run with results into the store. Returns run_id."""
    store.create_run(run_id, {
        "pipeline_version": "v0.1.0",
        "config_hash": "cfg123",
    })
    if results is None:
        results = [
            _make_result(nct_id="NCT00001111", ich_code="B.1",
                         boosted=0.90, auto_mapped=True),
            _make_result(nct_id="NCT00002222", ich_code="B.5",
                         title="Subject Eligibility",
                         boosted=0.45, auto_mapped=False,
                         match_quality="poor"),
            _make_result(nct_id="NCT00003333", ich_code="B.1",
                         title="Protocol Synopsis",
                         boosted=0.75, auto_mapped=True),
        ]
    for r in results:
        store.store_protocol_result(run_id, r)
    pass_count = sum(1 for r in results if r.status == "pass")
    err_count = sum(1 for r in results if r.status == "error")
    store.finalize_run(run_id, {
        "elapsed_seconds": 30.0,
        "protocol_count": len(results),
        "pass_count": pass_count,
        "fail_count": 0,
        "error_count": err_count,
    })
    return run_id


@pytest.fixture()
def store(tmp_path: Path) -> AnalysisStore:
    db_path = tmp_path / "test_cli.db"
    s = AnalysisStore(db_path)
    yield s
    s.close()


@pytest.fixture()
def populated_store(store: AnalysisStore) -> AnalysisStore:
    """Store with a single run of 3 protocols."""
    _populate_store(store)
    return store


# ---------------------------------------------------------------------------
# cmd_overview
# ---------------------------------------------------------------------------

class TestCmdOverview:
    def test_basic_output(self, populated_store: AnalysisStore) -> None:
        result = cmd_overview(populated_store, "run_001", top=5)
        assert result["run_id"] == "run_001"
        assert result["total_protocols"] == 3
        assert result["pass_rate"] == 1.0
        assert "worst_sections" in result
        assert "best_sections" in result

    def test_worst_best_sections(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_overview(populated_store, "run_001", top=2)
        worst = result["worst_sections"]
        best = result["best_sections"]
        assert len(worst) <= 2
        assert len(best) <= 2
        # Worst should have lower confidence than best
        if worst and best:
            assert worst[0]["avg_confidence"] <= best[0]["avg_confidence"]

    def test_nonexistent_run(self, store: AnalysisStore) -> None:
        result = cmd_overview(store, "no_such_run", top=5)
        assert "error" in result

    def test_top_limits_output(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_overview(populated_store, "run_001", top=1)
        assert len(result["worst_sections"]) == 1
        assert len(result["best_sections"]) == 1


# ---------------------------------------------------------------------------
# cmd_section_report
# ---------------------------------------------------------------------------

class TestCmdSectionReport:
    def test_valid_section(self, populated_store: AnalysisStore) -> None:
        result = cmd_section_report(
            populated_store, "run_001", "B.1", top=10,
        )
        assert result["section_code"] == "B.1"
        assert "corpus_stats" in result
        assert result["corpus_stats"]["protocol_count"] >= 1
        assert "common_protocol_headers" in result
        assert "lowest_confidence_protocols" in result

    def test_nonexistent_section(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_section_report(
            populated_store, "run_001", "B.99", top=10,
        )
        assert "error" in result

    def test_misclassification_targets(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_section_report(
            populated_store, "run_001", "B.1", top=10,
        )
        assert "misclassification_targets" in result


# ---------------------------------------------------------------------------
# cmd_protocol_report
# ---------------------------------------------------------------------------

class TestCmdProtocolReport:
    def test_valid_protocol(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_protocol_report(
            populated_store, "run_001", "NCT00001111",
        )
        assert result["nct_id"] == "NCT00001111"
        assert "section_matches" in result
        assert "comparison_pairs" in result

    def test_nonexistent_protocol(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_protocol_report(
            populated_store, "run_001", "NCT_NOPE",
        )
        assert "error" in result

    def test_long_text_trimmed(self, store: AnalysisStore) -> None:
        long_text = "x" * 1000
        _populate_store(store, results=[
            _make_result(original_text=long_text,
                         extracted_text=long_text),
        ])
        result = cmd_protocol_report(
            store, "run_001", "NCT00001234",
        )
        for cp in result.get("comparison_pairs", []):
            for field in ("original_text", "extracted_text"):
                if cp.get(field):
                    assert len(cp[field]) <= 503  # 500 + "..."


# ---------------------------------------------------------------------------
# cmd_low_confidence
# ---------------------------------------------------------------------------

class TestCmdLowConfidence:
    def test_returns_low_sections(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_low_confidence(
            populated_store, "run_001", threshold=0.60, top=20,
        )
        assert result["threshold"] == 0.60
        # B.5 has boosted=0.45 which is below 0.60
        assert len(result["sections"]) >= 1

    def test_high_threshold_captures_all(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_low_confidence(
            populated_store, "run_001", threshold=0.99, top=20,
        )
        assert len(result["sections"]) >= 1

    def test_top_limits(self, populated_store: AnalysisStore) -> None:
        result = cmd_low_confidence(
            populated_store, "run_001", threshold=0.99, top=1,
        )
        assert len(result["sections"]) <= 1


# ---------------------------------------------------------------------------
# cmd_gaps
# ---------------------------------------------------------------------------

class TestCmdGaps:
    def test_finds_poor_matches(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_gaps(populated_store, "run_001", top=20)
        # B.5 has match_quality="poor"
        assert len(result["details"]) >= 1
        assert len(result["gap_summary"]) >= 1

    def test_gap_text_trimmed(self, store: AnalysisStore) -> None:
        long_text = "y" * 500
        _populate_store(store, results=[
            _make_result(original_text=long_text,
                         match_quality="gap"),
        ])
        result = cmd_gaps(store, "run_001", top=20)
        for d in result["details"]:
            if d.get("original_text"):
                assert len(d["original_text"]) <= 203  # 200 + "..."


# ---------------------------------------------------------------------------
# cmd_patterns
# ---------------------------------------------------------------------------

class TestCmdPatterns:
    def test_header_vocabulary(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_patterns(populated_store, "run_001", top=10)
        assert "header_vocabulary" in result
        # B.1 should have entries
        assert "B.1" in result["header_vocabulary"]
        b1 = result["header_vocabulary"]["B.1"]
        assert "common" in b1

    def test_confusion_matrix(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_patterns(populated_store, "run_001", top=10)
        assert "confusion_matrix" in result
        # confusion_matrix is a dict of "title → code" → count


# ---------------------------------------------------------------------------
# cmd_compare
# ---------------------------------------------------------------------------

class TestCmdCompare:
    def test_compare_same_run(
        self, populated_store: AnalysisStore,
    ) -> None:
        result = cmd_compare(
            populated_store, "run_001", "run_001",
        )
        # Should return comparison structure (unchanged)
        assert isinstance(result, dict)

    def test_compare_two_runs(self, store: AnalysisStore) -> None:
        _populate_store(store, run_id="run_a", results=[
            _make_result(nct_id="NCT11111111", ich_code="B.1",
                         boosted=0.80),
        ])
        _populate_store(store, run_id="run_b", results=[
            _make_result(nct_id="NCT11111111", ich_code="B.1",
                         boosted=0.90),
        ])
        result = cmd_compare(store, "run_a", "run_b")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# cmd_export
# ---------------------------------------------------------------------------

class TestCmdExport:
    def test_valid_table(self, populated_store: AnalysisStore) -> None:
        csv_out = cmd_export(
            populated_store, "run_001", "section_matches",
        )
        assert isinstance(csv_out, str)
        lines = csv_out.strip().split("\n")
        assert len(lines) >= 2  # header + data row

    def test_invalid_table(
        self, populated_store: AnalysisStore,
    ) -> None:
        csv_out = cmd_export(
            populated_store, "run_001", "not_a_table",
        )
        parsed = json.loads(csv_out)
        assert "error" in parsed

    def test_protocol_results_table(
        self, populated_store: AnalysisStore,
    ) -> None:
        csv_out = cmd_export(
            populated_store, "run_001", "protocol_results",
        )
        assert isinstance(csv_out, str)
        lines = csv_out.strip().split("\n")
        assert len(lines) >= 2

    def test_comparison_pairs_table(
        self, populated_store: AnalysisStore,
    ) -> None:
        csv_out = cmd_export(
            populated_store, "run_001", "comparison_pairs",
        )
        assert isinstance(csv_out, str)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

class TestFormatOutput:
    def test_json_format(self) -> None:
        data = {"key": "value", "count": 42}
        out = _format_output(data, "json")
        parsed = json.loads(out)
        assert parsed["key"] == "value"

    def test_table_format(self) -> None:
        data = {"key": "value"}
        out = _format_output(data, "table")
        assert "key" in out
        assert "value" in out

    def test_markdown_format(self) -> None:
        data = {"key": "value"}
        out = _format_output(data, "markdown")
        assert "**key:**" in out

    def test_json_with_list(self) -> None:
        data = [{"a": 1}, {"a": 2}]
        out = _format_output(data, "json")
        parsed = json.loads(out)
        assert len(parsed) == 2


class TestToMarkdown:
    def test_string_passthrough(self) -> None:
        assert _to_markdown("hello") == "hello"

    def test_dict_formatting(self) -> None:
        result = _to_markdown({"run_id": "r1", "count": 5})
        assert "**run_id:**" in result
        assert "**count:**" in result

    def test_list_of_dicts(self) -> None:
        result = _to_markdown([{"a": 1, "b": 2}])
        assert "| a | b |" in result
        assert "---" in result

    def test_empty_list(self) -> None:
        result = _to_markdown([])
        assert "_empty_" in result

    def test_nested_dict(self) -> None:
        result = _to_markdown({"top": {"inner": 1}})
        assert "### top" in result


class TestToTable:
    def test_string_passthrough(self) -> None:
        assert _to_table("hello") == "hello"

    def test_error_dict(self) -> None:
        result = _to_table({"error": "not found"})
        assert "ERROR:" in result

    def test_list_of_dicts_aligned(self) -> None:
        data = [{"name": "a", "val": 1}, {"name": "bb", "val": 22}]
        result = _to_table(data)
        lines = result.strip().split("\n")
        assert len(lines) == 4  # header, sep, 2 rows

    def test_empty_list(self) -> None:
        result = _to_table([])
        assert "(empty)" in result


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_no_command_returns_1(self, capsys) -> None:
        ret = main(["--db", "fake.db"])
        assert ret == 1

    def test_missing_db_returns_1(self, tmp_path: Path) -> None:
        db = tmp_path / "nonexistent.db"
        ret = main(["--db", str(db), "overview"])
        assert ret == 1

    def test_overview_command(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store)
        store.close()

        ret = main(["--db", str(db), "overview"])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["total_protocols"] == 3

    def test_section_report_command(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store)
        store.close()

        ret = main([
            "--db", str(db), "section-report", "--section", "B.1",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["section_code"] == "B.1"

    def test_protocol_report_command(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store)
        store.close()

        ret = main([
            "--db", str(db), "protocol-report", "--nct", "NCT00001111",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["nct_id"] == "NCT00001111"

    def test_low_confidence_command(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store)
        store.close()

        ret = main([
            "--db", str(db), "low-confidence", "--threshold", "0.60",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["threshold"] == 0.60

    def test_gaps_command(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store)
        store.close()

        ret = main(["--db", str(db), "gaps"])
        assert ret == 0

    def test_patterns_command(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store)
        store.close()

        ret = main(["--db", str(db), "patterns"])
        assert ret == 0

    def test_export_command(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store)
        store.close()

        ret = main([
            "--db", str(db), "export", "--table", "section_matches",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        assert "ich_section_code" in captured.out  # CSV header

    def test_compare_command(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store, run_id="run_a", results=[
            _make_result(nct_id="NCT11111111"),
        ])
        _populate_store(store, run_id="run_b", results=[
            _make_result(nct_id="NCT11111111", boosted=0.95),
        ])
        store.close()

        ret = main([
            "--db", str(db), "compare",
            "--run-a", "run_a", "--run-b", "run_b",
        ])
        assert ret == 0

    def test_markdown_output_format(
        self, tmp_path: Path, capsys,
    ) -> None:
        db = tmp_path / "cli_test.db"
        store = AnalysisStore(db)
        _populate_store(store)
        store.close()

        ret = main([
            "--db", str(db), "--format", "markdown", "overview",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        assert "**" in captured.out  # markdown bold

    def test_no_runs_returns_1(
        self, tmp_path: Path,
    ) -> None:
        db = tmp_path / "empty.db"
        store = AnalysisStore(db)
        store.close()

        ret = main(["--db", str(db), "overview"])
        assert ret == 1
