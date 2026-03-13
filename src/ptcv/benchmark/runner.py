"""Benchmark runner — orchestrates pipeline runs across a protocol corpus.

Runs each protocol through ``run_query_pipeline()`` without Streamlit,
collecting results, timings, and quality metrics.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ptcv.benchmark.metrics import (
    ProtocolMetrics,
    compute_protocol_metrics,
)

logger = logging.getLogger(__name__)

_DEFAULT_CORPUS_DIR = Path("data/protocols/clinicaltrials")

# Default set of diverse protocols for benchmarking.
# Selected to cover different therapeutic areas, sizes, and structures.
DEFAULT_CORPUS: list[str] = [
    "NCT00004088_1.0.pdf",
    "NCT00112827_1.0.pdf",
    "NCT00492050_1.0.pdf",
    "NCT03350035_1.0.pdf",
    "NCT03963895_1.0.pdf",
]


@dataclasses.dataclass
class ProtocolRunResult:
    """Result of running a single protocol through the pipeline."""

    protocol_file: str
    success: bool
    metrics: ProtocolMetrics | None
    error: str | None
    elapsed_seconds: float


@dataclasses.dataclass
class BenchmarkResult:
    """Aggregate result of a full benchmark run."""

    timestamp: str
    corpus_size: int
    successful_runs: int
    failed_runs: int
    enable_transformation: bool
    enable_summarization: bool
    results: list[ProtocolRunResult]
    aggregate_timings: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "timestamp": self.timestamp,
            "corpus_size": self.corpus_size,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "enable_transformation": self.enable_transformation,
            "enable_summarization": self.enable_summarization,
            "results": [
                {
                    "protocol_file": r.protocol_file,
                    "success": r.success,
                    "metrics": (
                        dataclasses.asdict(r.metrics)
                        if r.metrics
                        else None
                    ),
                    "error": r.error,
                    "elapsed_seconds": r.elapsed_seconds,
                }
                for r in self.results
            ],
            "aggregate_timings": self.aggregate_timings,
        }


def _resolve_corpus(
    corpus_file: Path | None,
    corpus_dir: Path,
) -> list[Path]:
    """Resolve the list of protocol PDFs to benchmark.

    If *corpus_file* is provided, read protocol filenames from it.
    Otherwise use ``DEFAULT_CORPUS`` against *corpus_dir*.
    """
    if corpus_file and corpus_file.exists():
        with open(corpus_file) as f:
            data = json.load(f)
        filenames = data if isinstance(data, list) else data.get(
            "protocols", [],
        )
        return [corpus_dir / name for name in filenames]

    return [corpus_dir / name for name in DEFAULT_CORPUS]


def _compute_aggregate_timings(
    results: list[ProtocolRunResult],
) -> dict[str, dict[str, float]]:
    """Compute mean/median/p95 across successful runs."""
    import statistics

    stage_keys = [
        "document_assembly",
        "section_classification",
        "query_extraction",
        "result_aggregation",
    ]

    agg: dict[str, dict[str, float]] = {}
    for key in stage_keys:
        values = [
            r.metrics.stage_timings[key]
            for r in results
            if r.success
            and r.metrics
            and key in r.metrics.stage_timings
        ]
        if not values:
            continue
        sorted_vals = sorted(values)
        p95_idx = int(len(sorted_vals) * 0.95)
        agg[key] = {
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p95": round(sorted_vals[min(p95_idx, len(sorted_vals) - 1)], 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }
    return agg


def run_benchmark(
    corpus_dir: Path | None = None,
    corpus_file: Path | None = None,
    enable_transformation: bool = False,
    enable_summarization: bool = True,
    output_dir: Path | None = None,
) -> BenchmarkResult:
    """Run the full benchmark suite.

    Args:
        corpus_dir: Directory containing protocol PDFs.
        corpus_file: Optional JSON file listing protocol filenames.
        enable_transformation: Enable LLM content transformation.
        enable_summarization: Enable LLM sub-section scoring.
        output_dir: Directory to write results. Defaults to
            ``data/benchmark/{timestamp}/``.

    Returns:
        Aggregate benchmark result.
    """
    from ptcv.ui.components.query_pipeline import run_query_pipeline

    resolved_dir = corpus_dir or _DEFAULT_CORPUS_DIR
    protocols = _resolve_corpus(corpus_file, resolved_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        output_dir = Path("data/benchmark") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ProtocolRunResult] = []

    for i, pdf_path in enumerate(protocols):
        protocol_name = pdf_path.name
        logger.info(
            "[%d/%d] Running %s ...",
            i + 1, len(protocols), protocol_name,
        )

        if not pdf_path.exists():
            logger.warning("Protocol not found: %s", pdf_path)
            results.append(ProtocolRunResult(
                protocol_file=protocol_name,
                success=False,
                metrics=None,
                error=f"File not found: {pdf_path}",
                elapsed_seconds=0.0,
            ))
            continue

        t0 = time.monotonic()
        try:
            pipeline_result = run_query_pipeline(
                pdf_path=pdf_path,
                enable_summarization=enable_summarization,
                enable_transformation=enable_transformation,
            )
            elapsed = round(time.monotonic() - t0, 3)

            metrics = compute_protocol_metrics(
                protocol_name, pipeline_result,
            )

            results.append(ProtocolRunResult(
                protocol_file=protocol_name,
                success=True,
                metrics=metrics,
                error=None,
                elapsed_seconds=elapsed,
            ))
            logger.info(
                "  -> OK (%.1fs, coverage=%.1f%%, confidence=%.3f)",
                elapsed,
                metrics.answered_required / max(metrics.required_queries, 1) * 100,
                metrics.average_confidence,
            )

        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 3)
            logger.error(
                "  -> FAILED (%s): %s", protocol_name, exc,
            )
            results.append(ProtocolRunResult(
                protocol_file=protocol_name,
                success=False,
                metrics=None,
                error=str(exc),
                elapsed_seconds=elapsed,
            ))

    aggregate_timings = _compute_aggregate_timings(results)

    benchmark = BenchmarkResult(
        timestamp=timestamp,
        corpus_size=len(protocols),
        successful_runs=sum(1 for r in results if r.success),
        failed_runs=sum(1 for r in results if not r.success),
        enable_transformation=enable_transformation,
        enable_summarization=enable_summarization,
        results=results,
        aggregate_timings=aggregate_timings,
    )

    # Write results to disk
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark.to_dict(), f, indent=2, default=str)
    logger.info("Results written to %s", results_path)

    return benchmark
