"""Tests for batch protocol runner (PTCV-146)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.analysis.batch_runner import (
    BatchRunSummary,
    JsonResultStore,
    ManifestEntry,
    ProtocolResult,
    ResultStore,
    compute_file_sha,
    discover_protocols,
    process_protocol,
    run_batch,
    serialize_pipeline_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pdf(tmp_path: Path, name: str) -> Path:
    """Create a minimal fake PDF file."""
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 fake content " + name.encode())
    return p


def _make_protocol_dir(
    tmp_path: Path, count: int = 3,
) -> Path:
    """Create a directory with fake protocol PDFs."""
    proto_dir = tmp_path / "protocols"
    proto_dir.mkdir()
    for i in range(count):
        nct = f"NCT{10000000 + i:08d}"
        _make_pdf(proto_dir, f"{nct}_1.0.pdf")
    return proto_dir


@dataclass
class _FakeCoverage:
    total_sections: int = 14
    populated_count: int = 10
    gap_count: int = 4
    average_confidence: float = 0.75
    high_confidence_count: int = 5
    medium_confidence_count: int = 3
    low_confidence_count: int = 2
    total_queries: int = 70
    answered_queries: int = 50
    gap_sections: list = None  # type: ignore[assignment]
    low_confidence_sections: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gap_sections is None:
            self.gap_sections = []
        if self.low_confidence_sections is None:
            self.low_confidence_sections = []


@dataclass
class _FakeHit:
    query_id: str = "B.1.1.q1"
    section_id: str = "B.1.1"
    parent_section: str = "B.1"
    query_text: str = "What is the title?"
    extracted_content: str = "Phase 3 RCT"
    confidence: float = 0.85


@dataclass
class _FakeSection:
    section_code: str = "B.1"
    section_name: str = "General Information"
    populated: bool = True
    hits: list = None  # type: ignore[assignment]
    average_confidence: float = 0.85
    is_gap: bool = False
    has_low_confidence: bool = False

    def __post_init__(self) -> None:
        if self.hits is None:
            self.hits = [_FakeHit()]


@dataclass
class _FakeAssembled:
    sections: list = None  # type: ignore[assignment]
    coverage: _FakeCoverage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sections is None:
            self.sections = [_FakeSection()]
        if self.coverage is None:
            self.coverage = _FakeCoverage()

    def to_markdown(self) -> str:
        return "# Assembled"


@dataclass
class _FakeMatch:
    ich_section_code: str = "B.1"
    ich_section_name: str = "General Information"
    similarity_score: float = 0.90
    boosted_score: float = 0.92
    confidence: Any = None
    match_method: str = "keyword"

    def __post_init__(self) -> None:
        if self.confidence is None:
            self.confidence = MagicMock(name="HIGH")


@dataclass
class _FakeMapping:
    protocol_section_number: str = "1"
    protocol_section_title: str = "General"
    matches: list = None  # type: ignore[assignment]
    auto_mapped: bool = True

    def __post_init__(self) -> None:
        if self.matches is None:
            self.matches = [_FakeMatch()]


@dataclass
class _FakeMatchResult:
    mappings: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mappings is None:
            self.mappings = [_FakeMapping()]


@dataclass
class _FakeExtraction:
    query_id: str = "B.1.1.q1"
    section_id: str = "B.1.1"
    content: str = "Phase 3 study"
    confidence: float = 0.85
    extraction_method: str = "regex"
    source_section: str = "1. General"
    verbatim_content: str = ""


@dataclass
class _FakeExtractionResult:
    extractions: list = None  # type: ignore[assignment]
    gaps: list = None  # type: ignore[assignment]
    coverage: float = 0.75
    total_queries: int = 70
    answered_queries: int = 50

    def __post_init__(self) -> None:
        if self.extractions is None:
            self.extractions = [_FakeExtraction()]
        if self.gaps is None:
            self.gaps = []


@dataclass
class _FakeTOCEntry:
    section_number: str = "1"
    title: str = "General"


@dataclass
class _FakeProtocolIndex:
    source_path: str = "test.pdf"
    page_count: int = 50
    toc_entries: list = None  # type: ignore[assignment]
    content_spans: dict = None  # type: ignore[assignment]
    full_text: str = "Full document text"

    def __post_init__(self) -> None:
        if self.toc_entries is None:
            self.toc_entries = [_FakeTOCEntry()]
        if self.content_spans is None:
            self.content_spans = {"1": "General information content"}


def _make_pipeline_output() -> dict[str, Any]:
    """Build a fake run_query_pipeline() return dict."""
    return {
        "protocol_index": _FakeProtocolIndex(),
        "match_result": _FakeMatchResult(),
        "enriched_match_result": None,
        "extraction_result": _FakeExtractionResult(),
        "assembled": _FakeAssembled(),
        "coverage": _FakeCoverage(),
        "stage_timings": {
            "document_assembly": 1.2,
            "section_classification": 3.4,
            "query_extraction": 5.6,
            "result_aggregation": 0.1,
        },
    }


# ---------------------------------------------------------------------------
# Tests: discover_protocols
# ---------------------------------------------------------------------------

class TestDiscoverProtocols:

    def test_finds_pdfs(self, tmp_path: Path) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=5)
        entries = discover_protocols(proto_dir)
        assert len(entries) == 5
        assert all(isinstance(e, ManifestEntry) for e in entries)

    def test_extracts_nct_id(self, tmp_path: Path) -> None:
        proto_dir = tmp_path / "protocols"
        proto_dir.mkdir()
        _make_pdf(proto_dir, "NCT01234567_1.0.pdf")
        entries = discover_protocols(proto_dir)
        assert entries[0].nct_id == "NCT01234567"
        assert entries[0].version == "1.0"

    def test_skips_non_matching_files(self, tmp_path: Path) -> None:
        proto_dir = tmp_path / "protocols"
        proto_dir.mkdir()
        _make_pdf(proto_dir, "NCT01234567_1.0.pdf")
        _make_pdf(proto_dir, "README.pdf")
        _make_pdf(proto_dir, "notes.txt")
        entries = discover_protocols(proto_dir)
        assert len(entries) == 1

    def test_filter_pattern(self, tmp_path: Path) -> None:
        proto_dir = tmp_path / "protocols"
        proto_dir.mkdir()
        _make_pdf(proto_dir, "NCT00100000_1.0.pdf")
        _make_pdf(proto_dir, "NCT00200000_1.0.pdf")
        _make_pdf(proto_dir, "NCT00100001_1.0.pdf")
        entries = discover_protocols(proto_dir, filter_pattern="NCT001*")
        assert len(entries) == 2
        assert all(e.nct_id.startswith("NCT001") for e in entries)

    def test_limit(self, tmp_path: Path) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=10)
        entries = discover_protocols(proto_dir, limit=3)
        assert len(entries) == 3

    def test_empty_dir(self, tmp_path: Path) -> None:
        proto_dir = tmp_path / "empty"
        proto_dir.mkdir()
        entries = discover_protocols(proto_dir)
        assert entries == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        entries = discover_protocols(tmp_path / "nope")
        assert entries == []

    def test_sorted_by_name(self, tmp_path: Path) -> None:
        proto_dir = tmp_path / "protocols"
        proto_dir.mkdir()
        _make_pdf(proto_dir, "NCT99999999_1.0.pdf")
        _make_pdf(proto_dir, "NCT00000001_1.0.pdf")
        entries = discover_protocols(proto_dir)
        assert entries[0].nct_id == "NCT00000001"
        assert entries[1].nct_id == "NCT99999999"


# ---------------------------------------------------------------------------
# Tests: compute_file_sha
# ---------------------------------------------------------------------------

class TestComputeFileSha:

    def test_returns_hex_string(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pdf"
        p.write_bytes(b"test content")
        sha = compute_file_sha(p)
        assert len(sha) == 16
        assert all(c in "0123456789abcdef" for c in sha)

    def test_different_content_different_sha(
        self, tmp_path: Path,
    ) -> None:
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        assert compute_file_sha(a) != compute_file_sha(b)

    def test_same_content_same_sha(self, tmp_path: Path) -> None:
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"identical")
        b.write_bytes(b"identical")
        assert compute_file_sha(a) == compute_file_sha(b)


# ---------------------------------------------------------------------------
# Tests: serialize_pipeline_result
# ---------------------------------------------------------------------------

class TestSerializePipelineResult:

    def test_basic_serialization(self) -> None:
        output = _make_pipeline_output()
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 10.5,
        )
        assert result.nct_id == "NCT00001234"
        assert result.status == "pass"
        assert result.elapsed_seconds == 10.5

    def test_coverage_extracted(self) -> None:
        output = _make_pipeline_output()
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 1.0,
        )
        assert result.coverage["total_sections"] == 14
        assert result.coverage["populated_count"] == 10
        assert result.coverage["average_confidence"] == 0.75

    def test_section_matches_extracted(self) -> None:
        output = _make_pipeline_output()
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 1.0,
        )
        assert len(result.section_matches) == 1
        sm = result.section_matches[0]
        assert sm["ich_section_code"] == "B.1"
        assert sm["similarity_score"] == 0.90

    def test_query_extractions_extracted(self) -> None:
        output = _make_pipeline_output()
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 1.0,
        )
        assert len(result.query_extractions) == 1
        qe = result.query_extractions[0]
        assert qe["query_id"] == "B.1.1.q1"
        assert qe["confidence"] == 0.85

    def test_comparison_pairs_built(self) -> None:
        output = _make_pipeline_output()
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 1.0,
        )
        assert len(result.comparison_pairs) == 1
        cp = result.comparison_pairs[0]
        assert cp["ich_section_code"] == "B.1"
        assert cp["match_quality"] == "good"
        assert "Phase 3 RCT" in cp["extracted_text"]

    def test_stage_timings_preserved(self) -> None:
        output = _make_pipeline_output()
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 1.0,
        )
        assert result.stage_timings["document_assembly"] == 1.2

    def test_toc_count(self) -> None:
        output = _make_pipeline_output()
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 1.0,
        )
        assert result.toc_section_count == 1

    def test_content_preserved_full(self) -> None:
        output = _make_pipeline_output()
        ext = output["extraction_result"].extractions[0]
        ext.content = "X" * 5000
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 1.0,
        )
        assert len(result.query_extractions[0]["content"]) == 5000

    def test_empty_pipeline_output(self) -> None:
        output: dict[str, Any] = {
            "protocol_index": None,
            "match_result": None,
            "extraction_result": None,
            "assembled": None,
            "coverage": None,
            "stage_timings": {},
        }
        result = serialize_pipeline_result(
            "NCT00001234", "abc123", output, 0.5,
        )
        assert result.status == "pass"
        assert result.coverage == {}
        assert result.section_matches == []
        assert result.comparison_pairs == []


# ---------------------------------------------------------------------------
# Tests: process_protocol
# ---------------------------------------------------------------------------

class TestProcessProtocol:

    def test_success(self, tmp_path: Path) -> None:
        entry = ManifestEntry(
            nct_id="NCT00001234",
            version="1.0",
            pdf_path=tmp_path / "test.pdf",
            file_sha="abc123",
        )
        with patch(
            "ptcv.ui.components.query_pipeline.run_query_pipeline",
            return_value=_make_pipeline_output(),
        ):
            result = process_protocol(entry)
        assert result.status == "pass"
        assert result.nct_id == "NCT00001234"

    def test_error_isolation(self, tmp_path: Path) -> None:
        entry = ManifestEntry(
            nct_id="NCT00001234",
            version="1.0",
            pdf_path=tmp_path / "nonexistent.pdf",
            file_sha="abc123",
        )
        with patch(
            "ptcv.ui.components.query_pipeline.run_query_pipeline",
            side_effect=RuntimeError("PDF parse failed"),
        ):
            result = process_protocol(entry)
        assert result.status == "error"
        assert "PDF parse failed" in result.error_message


# ---------------------------------------------------------------------------
# Tests: JsonResultStore
# ---------------------------------------------------------------------------

class TestJsonResultStore:

    def test_create_run(self, tmp_path: Path) -> None:
        store = JsonResultStore(tmp_path)
        store.create_run("run_001", {"version": "abc"})
        meta = tmp_path / "run_001" / "metadata.json"
        assert meta.exists()
        data = json.loads(meta.read_text())
        assert data["version"] == "abc"

    def test_store_and_check_result(self, tmp_path: Path) -> None:
        store = JsonResultStore(tmp_path)
        store.create_run("run_001", {})
        result = ProtocolResult(
            nct_id="NCT00001234", file_sha="abc",
            status="pass", elapsed_seconds=1.0,
        )
        store.store_protocol_result("run_001", result)
        assert store.has_result("run_001", "NCT00001234")
        assert not store.has_result("run_001", "NCT99999999")

    def test_finalize_run(self, tmp_path: Path) -> None:
        store = JsonResultStore(tmp_path)
        store.create_run("run_001", {})
        store.finalize_run("run_001", {"pass_count": 5})
        summary = tmp_path / "run_001" / "summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text())
        assert data["pass_count"] == 5

    def test_result_file_content(self, tmp_path: Path) -> None:
        store = JsonResultStore(tmp_path)
        store.create_run("run_001", {})
        result = ProtocolResult(
            nct_id="NCT00001234", file_sha="abc",
            status="pass", elapsed_seconds=2.5,
            coverage={"total_sections": 14},
        )
        store.store_protocol_result("run_001", result)
        path = tmp_path / "run_001" / "protocols" / "NCT00001234.json"
        data = json.loads(path.read_text())
        assert data["nct_id"] == "NCT00001234"
        assert data["coverage"]["total_sections"] == 14


# ---------------------------------------------------------------------------
# Tests: run_batch
# ---------------------------------------------------------------------------

class TestRunBatch:

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_basic_batch(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=3)
        mock_process.return_value = ProtocolResult(
            nct_id="test", file_sha="abc", status="pass",
            elapsed_seconds=1.0,
        )
        store = JsonResultStore(tmp_path / "output")
        summary = run_batch(proto_dir, store)
        assert summary.protocol_count == 3
        assert summary.pass_count == 3
        assert summary.error_count == 0
        assert mock_process.call_count == 3

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_error_isolation(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=3)
        results = [
            ProtocolResult(
                nct_id="ok1", file_sha="a", status="pass",
                elapsed_seconds=1.0,
            ),
            ProtocolResult(
                nct_id="err", file_sha="b", status="error",
                error_message="fail", elapsed_seconds=0.5,
            ),
            ProtocolResult(
                nct_id="ok2", file_sha="c", status="pass",
                elapsed_seconds=1.0,
            ),
        ]
        mock_process.side_effect = results
        store = JsonResultStore(tmp_path / "output")
        summary = run_batch(proto_dir, store)
        assert summary.pass_count == 2
        assert summary.error_count == 1
        assert summary.protocol_count == 3

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_limit(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=10)
        mock_process.return_value = ProtocolResult(
            nct_id="test", file_sha="abc", status="pass",
            elapsed_seconds=1.0,
        )
        store = JsonResultStore(tmp_path / "output")
        summary = run_batch(proto_dir, store, limit=3)
        assert summary.protocol_count == 3
        assert mock_process.call_count == 3

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_filter_pattern(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = tmp_path / "protocols"
        proto_dir.mkdir()
        _make_pdf(proto_dir, "NCT00100000_1.0.pdf")
        _make_pdf(proto_dir, "NCT00200000_1.0.pdf")
        _make_pdf(proto_dir, "NCT00100001_1.0.pdf")
        mock_process.return_value = ProtocolResult(
            nct_id="test", file_sha="abc", status="pass",
            elapsed_seconds=1.0,
        )
        store = JsonResultStore(tmp_path / "output")
        summary = run_batch(
            proto_dir, store, filter_pattern="NCT001*",
        )
        assert summary.protocol_count == 2

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_skip_existing(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=3)
        store = JsonResultStore(tmp_path / "output")
        # Pre-populate one result — need to know run_id
        # Since skip_existing checks against the CURRENT run_id,
        # and run_id is generated inside run_batch, we mock has_result
        mock_store = MagicMock(spec=ResultStore)
        mock_store.has_result.side_effect = (
            lambda run_id, nct: nct == "NCT10000000"
        )
        mock_process.return_value = ProtocolResult(
            nct_id="test", file_sha="abc", status="pass",
            elapsed_seconds=1.0,
        )
        summary = run_batch(
            proto_dir, mock_store, skip_existing=True,
        )
        # Should skip 1 of 3
        assert summary.protocol_count == 2
        assert mock_process.call_count == 2

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_empty_dir(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = tmp_path / "empty"
        proto_dir.mkdir()
        store = JsonResultStore(tmp_path / "output")
        summary = run_batch(proto_dir, store)
        assert summary.protocol_count == 0
        assert mock_process.call_count == 0

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_parallel_workers(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=5)
        mock_process.return_value = ProtocolResult(
            nct_id="test", file_sha="abc", status="pass",
            elapsed_seconds=1.0,
        )
        store = JsonResultStore(tmp_path / "output")
        summary = run_batch(proto_dir, store, workers=2)
        assert summary.protocol_count == 5
        assert summary.pass_count == 5
        assert mock_process.call_count == 5

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_summary_stored(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=2)
        mock_process.return_value = ProtocolResult(
            nct_id="test", file_sha="abc", status="pass",
            elapsed_seconds=1.0,
        )
        output_dir = tmp_path / "output"
        store = JsonResultStore(output_dir)
        summary = run_batch(proto_dir, store)
        # Check summary file was written
        summary_path = output_dir / summary.run_id / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["pass_count"] == 2

    @patch("ptcv.analysis.batch_runner.process_protocol")
    def test_run_id_format(
        self, mock_process: MagicMock, tmp_path: Path,
    ) -> None:
        proto_dir = _make_protocol_dir(tmp_path, count=1)
        mock_process.return_value = ProtocolResult(
            nct_id="test", file_sha="abc", status="pass",
            elapsed_seconds=1.0,
        )
        store = JsonResultStore(tmp_path / "output")
        summary = run_batch(proto_dir, store)
        assert summary.run_id.startswith("run_")


# ---------------------------------------------------------------------------
# Tests: main CLI
# ---------------------------------------------------------------------------

class TestMainCli:

    @patch("ptcv.analysis.batch_runner.run_batch")
    def test_main_returns_zero_on_success(
        self, mock_batch: MagicMock,
    ) -> None:
        from ptcv.analysis.batch_runner import main

        mock_batch.return_value = BatchRunSummary(
            run_id="run_test",
            timestamp="2026-01-01T00:00:00",
            pipeline_version="abc",
            protocol_count=5,
            pass_count=5,
            error_count=0,
            elapsed_seconds=10.0,
        )
        code = main(["--limit", "5"])
        assert code == 0

    @patch("ptcv.analysis.batch_runner.run_batch")
    def test_main_returns_one_on_errors(
        self, mock_batch: MagicMock,
    ) -> None:
        from ptcv.analysis.batch_runner import main

        mock_batch.return_value = BatchRunSummary(
            run_id="run_test",
            timestamp="2026-01-01T00:00:00",
            pipeline_version="abc",
            protocol_count=5,
            pass_count=3,
            error_count=2,
            elapsed_seconds=10.0,
        )
        code = main(["--limit", "5"])
        assert code == 1
