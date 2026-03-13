"""Batch runner for query pipeline analysis (PTCV-146).

Runs all protocol PDFs through ``run_query_pipeline()`` in batch,
with error isolation, progress tracking, and structured output.

Usage::

    python -m ptcv.analysis.batch_runner [OPTIONS]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_NCT_PATTERN = re.compile(r"^(NCT\d{8})_(\d+\.\d+)\.pdf$")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    """A protocol PDF discovered for processing."""

    nct_id: str
    version: str
    pdf_path: Path
    file_sha: str


@dataclass
class ProtocolResult:
    """Result of processing one protocol through the pipeline."""

    nct_id: str
    file_sha: str
    status: str  # "pass", "error"
    error_message: str = ""
    elapsed_seconds: float = 0.0
    toc_section_count: int = 0
    coverage: dict[str, Any] = field(default_factory=dict)
    section_matches: list[dict[str, Any]] = field(default_factory=list)
    query_extractions: list[dict[str, Any]] = field(default_factory=list)
    comparison_pairs: list[dict[str, Any]] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)


@dataclass
class BatchRunSummary:
    """Summary of a complete batch run."""

    run_id: str
    timestamp: str
    pipeline_version: str
    protocol_count: int
    pass_count: int
    error_count: int
    elapsed_seconds: float
    results: list[ProtocolResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Result store abstraction
# ---------------------------------------------------------------------------

class ResultStore(ABC):
    """Abstract interface for storing batch run results.

    PTCV-147 will provide a SQLite implementation.
    """

    @abstractmethod
    def create_run(self, run_id: str, metadata: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def store_protocol_result(
        self, run_id: str, result: ProtocolResult,
    ) -> None:
        ...

    @abstractmethod
    def has_result(self, run_id: str, nct_id: str) -> bool:
        ...

    @abstractmethod
    def finalize_run(
        self, run_id: str, summary: dict[str, Any],
    ) -> None:
        ...


class JsonResultStore(ResultStore):
    """JSON file-based result store (default until PTCV-147)."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._protocols_dir: Path | None = None

    def create_run(self, run_id: str, metadata: dict[str, Any]) -> None:
        run_dir = self._output_dir / run_id
        self._protocols_dir = run_dir / "protocols"
        self._protocols_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

    def store_protocol_result(
        self, run_id: str, result: ProtocolResult,
    ) -> None:
        if self._protocols_dir is None:
            self._protocols_dir = (
                self._output_dir / run_id / "protocols"
            )
            self._protocols_dir.mkdir(parents=True, exist_ok=True)
        out = self._protocols_dir / f"{result.nct_id}.json"
        out.write_text(json.dumps(asdict(result), indent=2))

    def has_result(self, run_id: str, nct_id: str) -> bool:
        path = self._output_dir / run_id / "protocols" / f"{nct_id}.json"
        return path.exists()

    def finalize_run(
        self, run_id: str, summary: dict[str, Any],
    ) -> None:
        run_dir = self._output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Protocol discovery
# ---------------------------------------------------------------------------

def compute_file_sha(path: Path) -> str:
    """Compute truncated SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def discover_protocols(
    protocol_dir: Path,
    filter_pattern: str | None = None,
    limit: int | None = None,
) -> list[ManifestEntry]:
    """Scan directory for protocol PDFs and build manifest.

    Args:
        protocol_dir: Directory containing ``NCT*_*.pdf`` files.
        filter_pattern: Optional glob pattern for NCT IDs
            (e.g. ``NCT001*``).
        limit: Maximum number of protocols to return.

    Returns:
        Sorted list of :class:`ManifestEntry` objects.
    """
    entries: list[ManifestEntry] = []
    if not protocol_dir.is_dir():
        return entries

    for pdf_path in sorted(protocol_dir.glob("*.pdf")):
        match = _NCT_PATTERN.match(pdf_path.name)
        if not match:
            continue
        nct_id = match.group(1)
        version = match.group(2)

        if filter_pattern:
            import fnmatch

            if not fnmatch.fnmatch(nct_id, filter_pattern):
                continue

        entries.append(ManifestEntry(
            nct_id=nct_id,
            version=version,
            pdf_path=pdf_path,
            file_sha=compute_file_sha(pdf_path),
        ))

    if limit is not None and limit > 0:
        entries = entries[:limit]

    return entries


# ---------------------------------------------------------------------------
# Pipeline result serialization
# ---------------------------------------------------------------------------

def serialize_pipeline_result(
    nct_id: str,
    file_sha: str,
    pipeline_output: dict[str, Any],
    elapsed: float,
) -> ProtocolResult:
    """Convert raw pipeline output to a storable ProtocolResult.

    Extracts key metrics, match mappings, extraction results,
    and comparison pairs from the pipeline dict.
    """
    protocol_index = pipeline_output.get("protocol_index")
    match_result = pipeline_output.get("match_result")
    extraction_result = pipeline_output.get("extraction_result")
    assembled = pipeline_output.get("assembled")
    coverage_obj = pipeline_output.get("coverage")
    timings = pipeline_output.get("stage_timings", {})

    # Coverage report
    coverage_dict: dict[str, Any] = {}
    if coverage_obj is not None:
        coverage_dict = {
            "total_sections": coverage_obj.total_sections,
            "populated_count": coverage_obj.populated_count,
            "gap_count": coverage_obj.gap_count,
            "average_confidence": coverage_obj.average_confidence,
            "high_confidence_count": coverage_obj.high_confidence_count,
            "medium_confidence_count": coverage_obj.medium_confidence_count,
            "low_confidence_count": coverage_obj.low_confidence_count,
            "total_queries": coverage_obj.total_queries,
            "answered_queries": coverage_obj.answered_queries,
            "gap_sections": coverage_obj.gap_sections,
            "low_confidence_sections": (
                coverage_obj.low_confidence_sections
            ),
        }

    # Section matches
    section_matches: list[dict[str, Any]] = []
    if match_result is not None:
        for mapping in match_result.mappings:
            for sm in mapping.matches[:1]:  # Top match only
                section_matches.append({
                    "protocol_section_number": (
                        mapping.protocol_section_number
                    ),
                    "protocol_section_title": (
                        mapping.protocol_section_title
                    ),
                    "ich_section_code": sm.ich_section_code,
                    "ich_section_name": sm.ich_section_name,
                    "similarity_score": sm.similarity_score,
                    "boosted_score": sm.boosted_score,
                    "confidence": sm.confidence.name,
                    "match_method": sm.match_method,
                    "auto_mapped": mapping.auto_mapped,
                })

    # Query extractions
    query_extractions: list[dict[str, Any]] = []
    if extraction_result is not None:
        for ext in extraction_result.extractions:
            query_extractions.append({
                "query_id": ext.query_id,
                "section_id": ext.section_id,
                "content": ext.content,
                "confidence": ext.confidence,
                "extraction_method": ext.extraction_method,
                "source_section": ext.source_section,
                "verbatim_content": ext.verbatim_content,
            })

    # Comparison pairs: original text vs extracted text per ICH section
    comparison_pairs: list[dict[str, Any]] = []
    if assembled is not None and protocol_index is not None:
        content_spans = getattr(protocol_index, "content_spans", {})
        # Build mapping from ICH code → protocol section numbers
        ich_to_proto: dict[str, list[str]] = {}
        if match_result is not None:
            for mapping in match_result.mappings:
                if mapping.matches:
                    code = mapping.matches[0].ich_section_code
                    proto_num = mapping.protocol_section_number
                    ich_to_proto.setdefault(code, []).append(proto_num)

        for section in assembled.sections:
            if not section.populated:
                continue
            # Gather original text from mapped protocol sections
            proto_nums = ich_to_proto.get(section.section_code, [])
            original_parts: list[str] = []
            for pn in proto_nums:
                text = content_spans.get(pn, "")
                if text:
                    original_parts.append(text)
            original_text = "\n---\n".join(original_parts)

            # Gathered extracted text from hits
            extracted_parts = [
                h.extracted_content
                for h in section.hits
            ]
            extracted_text = "\n---\n".join(extracted_parts)

            confidence = section.average_confidence
            if not original_text and not extracted_text:
                quality = "missing"
            elif not extracted_text:
                quality = "gap"
            elif confidence >= 0.80:
                quality = "good"
            elif confidence >= 0.60:
                quality = "partial"
            else:
                quality = "poor"

            comparison_pairs.append({
                "ich_section_code": section.section_code,
                "ich_section_name": section.section_name,
                "original_text": original_text,
                "extracted_text": extracted_text,
                "confidence": confidence,
                "match_quality": quality,
            })

    toc_count = 0
    if protocol_index is not None:
        toc_count = len(
            getattr(protocol_index, "toc_entries", [])
        )

    return ProtocolResult(
        nct_id=nct_id,
        file_sha=file_sha,
        status="pass",
        elapsed_seconds=round(elapsed, 3),
        toc_section_count=toc_count,
        coverage=coverage_dict,
        section_matches=section_matches,
        query_extractions=query_extractions,
        comparison_pairs=comparison_pairs,
        stage_timings=timings,
    )


# ---------------------------------------------------------------------------
# Single-protocol processing
# ---------------------------------------------------------------------------

def process_protocol(
    entry: ManifestEntry,
    enable_summarization: bool = False,
    enable_transformation: bool = False,
    anthropic_api_key: str | None = None,
) -> ProtocolResult:
    """Process one protocol through the query pipeline.

    Errors are caught and recorded — never raises.
    """
    from ptcv.ui.components.query_pipeline import run_query_pipeline

    t0 = time.monotonic()
    try:
        result = run_query_pipeline(
            pdf_path=str(entry.pdf_path),
            anthropic_api_key=anthropic_api_key,
            enable_summarization=enable_summarization,
            enable_transformation=enable_transformation,
        )
        elapsed = time.monotonic() - t0
        return serialize_pipeline_result(
            nct_id=entry.nct_id,
            file_sha=entry.file_sha,
            pipeline_output=result,
            elapsed=elapsed,
        )
    except Exception:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc()
        logger.error("Error processing %s: %s", entry.nct_id, tb)
        return ProtocolResult(
            nct_id=entry.nct_id,
            file_sha=entry.file_sha,
            status="error",
            error_message=tb[-2000:],
            elapsed_seconds=round(elapsed, 3),
        )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def _get_pipeline_version() -> str:
    """Get current git SHA for pipeline versioning."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_batch(
    protocol_dir: Path,
    store: ResultStore,
    workers: int = 1,
    limit: int | None = None,
    filter_pattern: str | None = None,
    skip_existing: bool = False,
    enable_summarization: bool = False,
    enable_transformation: bool = False,
    anthropic_api_key: str | None = None,
    verbose: bool = False,
) -> BatchRunSummary:
    """Run all protocols through the query pipeline.

    Args:
        protocol_dir: Directory containing protocol PDFs.
        store: Result store implementation.
        workers: Number of parallel workers (1 = serial).
        limit: Maximum protocols to process.
        filter_pattern: Glob pattern for NCT ID filtering.
        skip_existing: Skip protocols already in the store.
        enable_summarization: Enable LLM sub-section enrichment.
        enable_transformation: Enable LLM content transformation.
        anthropic_api_key: Anthropic API key for LLM features.
        verbose: Enable per-protocol logging.

    Returns:
        :class:`BatchRunSummary` with aggregated results.
    """
    ts = datetime.now(timezone.utc)
    run_id = f"run_{ts.strftime('%Y%m%d_%H%M%S')}"
    pipeline_version = _get_pipeline_version()

    # Discover protocols
    entries = discover_protocols(
        protocol_dir, filter_pattern=filter_pattern, limit=limit,
    )
    if not entries:
        logger.warning("No protocols found in %s", protocol_dir)
        return BatchRunSummary(
            run_id=run_id,
            timestamp=ts.isoformat(),
            pipeline_version=pipeline_version,
            protocol_count=0,
            pass_count=0,
            error_count=0,
            elapsed_seconds=0.0,
        )

    # Initialize store
    store.create_run(run_id, {
        "pipeline_version": pipeline_version,
        "protocol_dir": str(protocol_dir),
        "protocol_count": len(entries),
        "enable_summarization": enable_summarization,
        "enable_transformation": enable_transformation,
        "llm_enabled": bool(anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")),
        "workers": workers,
        "timestamp": ts.isoformat(),
    })

    # Filter out existing results
    if skip_existing:
        entries = [
            e for e in entries
            if not store.has_result(run_id, e.nct_id)
        ]
        if not entries:
            logger.info("All protocols already processed.")

    total = len(entries)
    results: list[ProtocolResult] = []
    pass_count = 0
    error_count = 0
    batch_t0 = time.monotonic()

    def _log_progress(done: int, errors: int) -> None:
        elapsed = time.monotonic() - batch_t0
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        pct = done * 100 // total if total > 0 else 0
        msg = (
            f"[{done}/{total}] {pct}%"
            f" \u2014 {errors} errors"
            f" \u2014 elapsed {mins}m {secs:02d}s"
        )
        print(msg, file=sys.stderr, flush=True)

    def _process_one(entry: ManifestEntry) -> ProtocolResult:
        return process_protocol(
            entry,
            enable_summarization=enable_summarization,
            enable_transformation=enable_transformation,
            anthropic_api_key=anthropic_api_key,
        )

    if workers <= 1:
        # Serial execution
        for i, entry in enumerate(entries):
            if verbose:
                print(
                    f"Processing {entry.nct_id}...",
                    file=sys.stderr, flush=True,
                )
            result = _process_one(entry)
            results.append(result)
            store.store_protocol_result(run_id, result)

            if result.status == "pass":
                pass_count += 1
            else:
                error_count += 1

            if (i + 1) % 10 == 0 or (i + 1) == total:
                _log_progress(i + 1, error_count)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, entry): entry
                for entry in entries
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                results.append(result)
                store.store_protocol_result(run_id, result)

                if result.status == "pass":
                    pass_count += 1
                else:
                    error_count += 1

                if verbose:
                    entry = futures[future]
                    print(
                        f"  {entry.nct_id}: {result.status}"
                        f" ({result.elapsed_seconds:.1f}s)",
                        file=sys.stderr, flush=True,
                    )
                if done_count % 10 == 0 or done_count == total:
                    _log_progress(done_count, error_count)

    batch_elapsed = round(time.monotonic() - batch_t0, 3)

    summary = BatchRunSummary(
        run_id=run_id,
        timestamp=ts.isoformat(),
        pipeline_version=pipeline_version,
        protocol_count=total,
        pass_count=pass_count,
        error_count=error_count,
        elapsed_seconds=batch_elapsed,
        results=results,
    )

    # Finalize store
    summary_dict = {
        "run_id": run_id,
        "timestamp": ts.isoformat(),
        "pipeline_version": pipeline_version,
        "protocol_count": total,
        "pass_count": pass_count,
        "error_count": error_count,
        "elapsed_seconds": batch_elapsed,
    }
    store.finalize_run(run_id, summary_dict)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for batch runner."""
    parser = argparse.ArgumentParser(
        description="Run all protocols through the query pipeline.",
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("data/protocols/clinicaltrials"),
        help="Protocol PDF directory (default: data/protocols/clinicaltrials/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Output directory for JSON results (default: data/analysis/)",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=None,
        help="SQLite database path (uses AnalysisStore from PTCV-147)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1, serial)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip protocols already processed in current run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N protocols (for testing)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        dest="filter_pattern",
        help="Filter protocols by NCT ID pattern (e.g., NCT001*)",
    )
    parser.add_argument(
        "--enable-summarization",
        action="store_true",
        help="Enable LLM sub-section enrichment (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--enable-transformation",
        action="store_true",
        help="Enable LLM content transformation (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--anthropic-key",
        type=str,
        default=None,
        help="Anthropic API key (or ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Detailed per-protocol logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    anthropic_key = (
        args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    )

    # Warn if LLM features requested but no API key available
    if (args.enable_summarization or args.enable_transformation) and not anthropic_key:
        logger.warning(
            "LLM features requested but no ANTHROPIC_API_KEY found. "
            "Pipeline will fall back to keyword-only methods."
        )

    store: ResultStore
    if args.output_db:
        from ptcv.analysis.data_store import AnalysisStore

        store = AnalysisStore(args.output_db)
    else:
        store = JsonResultStore(args.output_dir)

    summary = run_batch(
        protocol_dir=args.protocol_dir,
        store=store,
        workers=args.workers,
        limit=args.limit,
        filter_pattern=args.filter_pattern,
        skip_existing=args.skip_existing,
        enable_summarization=args.enable_summarization,
        enable_transformation=args.enable_transformation,
        anthropic_api_key=anthropic_key,
        verbose=args.verbose,
    )

    # Print final summary
    print(
        f"\nBatch run complete: {summary.run_id}",
        file=sys.stderr,
    )
    print(
        f"  Protocols: {summary.protocol_count}"
        f" | Pass: {summary.pass_count}"
        f" | Errors: {summary.error_count}"
        f" | Elapsed: {summary.elapsed_seconds:.1f}s",
        file=sys.stderr,
    )

    # Output summary JSON to stdout
    print(json.dumps({
        "run_id": summary.run_id,
        "timestamp": summary.timestamp,
        "pipeline_version": summary.pipeline_version,
        "protocol_count": summary.protocol_count,
        "pass_count": summary.pass_count,
        "error_count": summary.error_count,
        "elapsed_seconds": summary.elapsed_seconds,
    }, indent=2))

    return 0 if summary.error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
