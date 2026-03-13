"""End-to-end pipeline benchmark — PTCV-178.

Runs a configurable subset of clinical trial protocol PDFs through all
7 pipeline stages, capturing per-stage wall-clock timing and quality
metrics.  Outputs per-protocol and aggregate JSON reports for regression
testing and pipeline validation.

Stages invoked individually (not via PipelineOrchestrator) to capture
per-stage timing and NormalizationStats that the orchestrator does not
surface.

Risk tier: LOW — benchmarking tooling only; no patient data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

from ptcv.extraction.extraction_service import ExtractionService
from ptcv.extraction.markdown_normalizer import normalize_markdown
from ptcv.extraction.parquet_writer import (
    parquet_to_metadata,
    parquet_to_tables,
)
from ptcv.ich_parser.coverage_reviewer import CoverageReviewer
from ptcv.ich_parser.llm_retemplater import LlmRetemplater
from ptcv.ich_parser.parquet_writer import parquet_to_sections
from ptcv.ich_parser.review_queue import ReviewQueue
from ptcv.sdtm.review_queue import CtReviewQueue
from ptcv.sdtm.sdtm_service import SdtmService
from ptcv.sdtm.validation.validation_service import ValidationService
from ptcv.soa_extractor.extractor import SoaExtractor
from ptcv.soa_extractor.writer import UsdmParquetWriter
from ptcv.storage.filesystem_adapter import FilesystemAdapter

logger = logging.getLogger(__name__)

PROTOCOL_DIR = Path("C:/Dev/PTCV/data/protocols/clinicaltrials")
DEFAULT_OUTPUT_DIR = Path("data/analysis/benchmark")

# 8 curated protocols spanning 2000–2024 for diverse coverage.
DEFAULT_PROTOCOLS: list[str] = [
    "NCT00112827",
    "NCT00783367",
    "NCT01678794",
    "NCT03291015",
    "NCT04185324",
    "NCT05097989",
    "NCT05855967",
    "NCT06011798",
]

_AGGREGATE_KEYS: list[str] = [
    "total_elapsed_seconds",
    "text_block_count",
    "table_count",
    "page_count",
    "headings_promoted",
    "timepoint_count",
    "activity_count",
    "instance_count",
    "section_count",
    "format_confidence",
    "coverage_score",
    "p21_error_count",
    "p21_warning_count",
    "retemplating_input_tokens",
    "retemplating_output_tokens",
]

_STAGE_NAMES: list[str] = [
    "extraction",
    "normalization",
    "soa_extraction",
    "retemplating",
    "coverage_review",
    "sdtm_generation",
    "validation",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProtocolMetrics:
    """Per-protocol benchmark metrics across all pipeline stages."""

    nct_id: str
    file_sha: str
    status: str  # "pass" | "error"
    failed_stage: Optional[str] = None
    error_message: str = ""
    total_elapsed_seconds: float = 0.0
    stage_timings: dict[str, float] = dataclasses.field(default_factory=dict)

    # Extraction
    text_block_count: int = 0
    table_count: int = 0
    format_detected: str = ""
    page_count: int = 0

    # Normalization
    headings_promoted: int = 0
    headings_cleaned: int = 0
    toc_lines_removed: int = 0
    boilerplate_lines_removed: int = 0
    br_tags_removed: int = 0

    # SoA
    epoch_count: int = 0
    timepoint_count: int = 0
    activity_count: int = 0
    instance_count: int = 0
    soa_review_count: int = 0

    # Retemplating
    section_count: int = 0
    format_verdict: str = ""
    format_confidence: float = 0.0
    missing_required_sections: list[str] = dataclasses.field(
        default_factory=list,
    )
    retemplating_input_tokens: int = 0
    retemplating_output_tokens: int = 0

    # Coverage
    coverage_score: float = 0.0
    coverage_passed: bool = False
    total_original_chars: int = 0
    covered_chars: int = 0

    # SDTM
    domain_row_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    ct_unmapped_count: int = 0

    # Validation
    p21_error_count: int = 0
    p21_warning_count: int = 0
    tcg_passed: bool = False
    schedule_feasible: bool = True
    schedule_issues: list[dict[str, Any]] = dataclasses.field(
        default_factory=list,
    )


@dataclasses.dataclass
class BenchmarkReport:
    """Aggregate benchmark report across all protocols."""

    run_id: str
    timestamp_utc: str
    pipeline_capabilities: dict[str, str]
    protocol_count: int
    pass_count: int
    error_count: int
    failures_by_stage: dict[str, int]
    total_elapsed_seconds: float
    aggregate_metrics: dict[str, Any]
    protocol_results: list[ProtocolMetrics]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_protocol_paths(
    protocols: Optional[list[str]],
    protocol_dir: Path,
    limit: Optional[int] = None,
) -> list[Path]:
    """Resolve protocol identifiers to PDF file paths.

    Each entry can be an NCT ID (looked up in *protocol_dir*) or an
    absolute/relative PDF file path.  Falls back to ``DEFAULT_PROTOCOLS``
    when *protocols* is ``None``.
    """
    if not protocols:
        protocols = list(DEFAULT_PROTOCOLS)

    paths: list[Path] = []
    for entry in protocols:
        p = Path(entry)
        if p.suffix.lower() == ".pdf" and p.exists():
            paths.append(p)
        else:
            matches = sorted(protocol_dir.glob(f"{entry}_*.pdf"))
            if matches:
                paths.append(matches[0])
            else:
                logger.warning("Protocol not found: %s", entry)

    if limit is not None and limit > 0:
        paths = paths[:limit]
    return paths


def _compute_aggregate(
    results: list[ProtocolMetrics],
    key: str,
) -> dict[str, float]:
    """Compute mean/median/min/max for a numeric metric across passes."""
    vals: list[float] = []
    for r in results:
        if r.status != "pass":
            continue
        v = getattr(r, key, None)
        if v is not None and isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": round(statistics.mean(vals), 3),
        "median": round(statistics.median(vals), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "n": len(vals),
    }


# ---------------------------------------------------------------------------
# Per-protocol pipeline runner
# ---------------------------------------------------------------------------


def _run_one_protocol(
    pdf_path: Path,
    gateway: FilesystemAdapter,
    review_queue: ReviewQueue,
    ct_review_queue: CtReviewQueue,
) -> ProtocolMetrics:
    """Run all 7 pipeline stages for one protocol PDF.

    Each stage is wrapped in its own try/except so that a failure at any
    point records partial metrics up to the failing stage.
    """
    nct_id = pdf_path.stem.rsplit("_", 1)[0]
    pdf_bytes = pdf_path.read_bytes()
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    run_t0 = time.perf_counter()

    metrics = ProtocolMetrics(nct_id=nct_id, file_sha=sha[:16], status="pass")

    # Shared service instances (safe to reuse per orchestrator pattern).
    extraction_svc = ExtractionService(gateway=gateway)
    soa_extractor = SoaExtractor(
        gateway=gateway, review_queue=review_queue,
    )
    retemplater = LlmRetemplater(
        gateway=gateway, review_queue=review_queue,
    )
    coverage_reviewer = CoverageReviewer()
    sdtm_svc = SdtmService(
        gateway=gateway, review_queue=ct_review_queue,
    )
    val_svc = ValidationService(gateway=gateway)

    # Intermediate results shared across stages.
    text_block_dicts: list[dict[str, Any]] = []
    extracted_tables: Optional[list[Any]] = None
    page_count = 0

    # ── Stage 1: Extraction ─────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        ext_result = extraction_svc.extract(
            protocol_data=pdf_bytes,
            registry_id=nct_id,
            amendment_number="00",
            source_sha256=sha,
            filename=pdf_path.name,
            source="ClinicalTrials.gov",
        )
        metrics.stage_timings["extraction"] = time.perf_counter() - t0
        metrics.text_block_count = ext_result.text_block_count
        metrics.table_count = ext_result.table_count
        metrics.format_detected = ext_result.format_detected

        # Load text blocks as dicts (downstream stages expect list[dict]).
        tb_bytes = gateway.get_artifact(ext_result.text_artifact_key)
        tb_df = pq.read_table(io.BytesIO(tb_bytes)).to_pandas()
        text_block_dicts = tb_df.to_dict("records")

        # Page count from metadata.
        meta_bytes = gateway.get_artifact(ext_result.metadata_artifact_key)
        meta = parquet_to_metadata(meta_bytes)
        page_count = meta.page_count
        metrics.page_count = page_count

        # Extracted tables for SoA bridge.
        if ext_result.tables_artifact_key:
            try:
                tbl_bytes = gateway.get_artifact(
                    ext_result.tables_artifact_key,
                )
                extracted_tables = parquet_to_tables(tbl_bytes)
            except Exception:
                logger.debug("Could not load tables; proceeding without")
    except Exception:
        metrics.status = "error"
        metrics.failed_stage = "extraction"
        metrics.error_message = traceback.format_exc()[-2000:]
        metrics.total_elapsed_seconds = time.perf_counter() - run_t0
        return metrics

    # ── Stage 2: Markdown Normalization ─────────────────────────────
    try:
        t0 = time.perf_counter()
        full_text = "\n".join(
            str(b.get("text", "")) for b in text_block_dicts
        )
        norm_result = normalize_markdown(full_text, extract_toc=True)
        metrics.stage_timings["normalization"] = time.perf_counter() - t0
        stats = norm_result.stats
        metrics.headings_promoted = stats.headings_promoted
        metrics.headings_cleaned = stats.headings_cleaned
        metrics.toc_lines_removed = stats.toc_lines_removed
        metrics.boilerplate_lines_removed = stats.boilerplate_lines_removed
        metrics.br_tags_removed = stats.br_tags_removed
    except Exception:
        metrics.status = "error"
        metrics.failed_stage = "normalization"
        metrics.error_message = traceback.format_exc()[-2000:]
        metrics.total_elapsed_seconds = time.perf_counter() - run_t0
        return metrics

    # ── Stage 3: SoA Extraction ─────────────────────────────────────
    try:
        t0 = time.perf_counter()
        soa_result = soa_extractor.extract(
            registry_id=nct_id,
            source_run_id=ext_result.run_id,
            source_sha256=ext_result.text_artifact_sha256,
            pdf_bytes=pdf_bytes,
            text_blocks=text_block_dicts,
            page_count=page_count,
            extracted_tables=extracted_tables,
        )
        metrics.stage_timings["soa_extraction"] = time.perf_counter() - t0
        metrics.epoch_count = soa_result.epoch_count
        metrics.timepoint_count = soa_result.timepoint_count
        metrics.activity_count = soa_result.activity_count
        metrics.instance_count = soa_result.instance_count
        metrics.soa_review_count = soa_result.review_count
    except Exception:
        metrics.status = "error"
        metrics.failed_stage = "soa_extraction"
        metrics.error_message = traceback.format_exc()[-2000:]
        metrics.total_elapsed_seconds = time.perf_counter() - run_t0
        return metrics

    # ── Stage 4: Retemplating ───────────────────────────────────────
    soa_summary: Optional[dict[str, Any]] = None
    if soa_result.timepoint_count > 0:
        soa_summary = {
            "visit_count": soa_result.timepoint_count,
            "activity_count": soa_result.activity_count,
        }

    sections: list[Any] = []
    try:
        t0 = time.perf_counter()
        retemplate_result = retemplater.retemplate(
            text_blocks=text_block_dicts,
            registry_id=nct_id,
            source_run_id=ext_result.run_id,
            source_sha256=ext_result.text_artifact_sha256,
            soa_summary=soa_summary,
        )
        metrics.stage_timings["retemplating"] = time.perf_counter() - t0
        metrics.section_count = retemplate_result.section_count
        metrics.format_verdict = retemplate_result.format_verdict
        metrics.format_confidence = retemplate_result.format_confidence
        metrics.missing_required_sections = list(
            retemplate_result.missing_required_sections,
        )
        metrics.retemplating_input_tokens = retemplate_result.input_tokens
        metrics.retemplating_output_tokens = retemplate_result.output_tokens

        # Load sections for downstream stages.
        sec_bytes = gateway.get_artifact(retemplate_result.artifact_key)
        sections = parquet_to_sections(sec_bytes)
    except Exception:
        metrics.status = "error"
        metrics.failed_stage = "retemplating"
        metrics.error_message = traceback.format_exc()[-2000:]
        metrics.total_elapsed_seconds = time.perf_counter() - run_t0
        return metrics

    # ── Stage 5: Coverage Review ────────────────────────────────────
    try:
        t0 = time.perf_counter()
        cov_result = coverage_reviewer.review(
            original_text_blocks=text_block_dicts,
            retemplated_sections=sections,
        )
        metrics.stage_timings["coverage_review"] = time.perf_counter() - t0
        metrics.coverage_score = cov_result.coverage_score
        metrics.coverage_passed = cov_result.passed
        metrics.total_original_chars = cov_result.total_original_chars
        metrics.covered_chars = cov_result.covered_chars
    except Exception:
        metrics.status = "error"
        metrics.failed_stage = "coverage_review"
        metrics.error_message = traceback.format_exc()[-2000:]
        metrics.total_elapsed_seconds = time.perf_counter() - run_t0
        return metrics

    # ── Stage 6: SDTM Generation ───────────────────────────────────
    timepoints: list[Any] = []
    tp_key = soa_result.artifact_keys.get("timepoints")
    if tp_key:
        try:
            tp_bytes = gateway.get_artifact(tp_key)
            timepoints = UsdmParquetWriter.parquet_to_timepoints(tp_bytes)
        except Exception:
            logger.debug("Could not load timepoints; proceeding without")

    try:
        t0 = time.perf_counter()
        sdtm_result = sdtm_svc.generate(
            sections=sections,
            timepoints=timepoints,
            registry_id=nct_id,
            amendment_number="00",
            source_sha256=retemplate_result.artifact_sha256,
        )
        metrics.stage_timings["sdtm_generation"] = time.perf_counter() - t0
        metrics.domain_row_counts = dict(sdtm_result.domain_row_counts)
        metrics.ct_unmapped_count = sdtm_result.ct_unmapped_count
    except Exception:
        metrics.status = "error"
        metrics.failed_stage = "sdtm_generation"
        metrics.error_message = traceback.format_exc()[-2000:]
        metrics.total_elapsed_seconds = time.perf_counter() - run_t0
        return metrics

    # ── Stage 7: Validation ─────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        val_result = val_svc.validate(
            sdtm_result=sdtm_result,
            format_verdict=retemplate_result.format_verdict,
            format_confidence=retemplate_result.format_confidence,
            sections_detected=retemplate_result.section_count,
            missing_required_sections=(
                retemplate_result.missing_required_sections
            ),
        )
        metrics.stage_timings["validation"] = time.perf_counter() - t0
        metrics.p21_error_count = val_result.p21_error_count
        metrics.p21_warning_count = val_result.p21_warning_count
        metrics.tcg_passed = val_result.tcg_passed
        metrics.schedule_feasible = val_result.schedule_feasible
        metrics.schedule_issues = [
            {
                "rule_id": si.rule_id,
                "severity": si.severity,
                "description": si.description,
            }
            for si in val_result.schedule_issues
        ]
    except Exception:
        metrics.status = "error"
        metrics.failed_stage = "validation"
        metrics.error_message = traceback.format_exc()[-2000:]
        metrics.total_elapsed_seconds = time.perf_counter() - run_t0
        return metrics

    metrics.total_elapsed_seconds = time.perf_counter() - run_t0
    return metrics


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------


def run_benchmark(
    protocol_paths: list[Path],
    output_dir: Path,
    extraction_level: Optional[str] = None,
    classification_level: Optional[str] = None,
    verbose: bool = False,
) -> BenchmarkReport:
    """Run the full benchmark across all *protocol_paths*.

    Creates an ephemeral ``FilesystemAdapter`` in a temp directory that is
    cleaned up when the benchmark finishes.  Per-protocol JSON results are
    written incrementally.
    """
    ts = datetime.now(timezone.utc)
    run_id = f"benchmark_{ts.strftime('%Y%m%d_%H%M%S')}"

    if extraction_level:
        os.environ["PTCV_FORCE_EXTRACTION_LEVEL"] = extraction_level
    if classification_level:
        os.environ["PTCV_FORCE_CLASSIFICATION_LEVEL"] = classification_level

    output_dir.mkdir(parents=True, exist_ok=True)
    protocols_dir = output_dir / "protocols"
    protocols_dir.mkdir(exist_ok=True)

    tmp = Path(tempfile.mkdtemp())
    gateway = FilesystemAdapter(root=tmp / "benchmark")
    gateway.initialise()
    review_queue = ReviewQueue(db_path=tmp / "review_queue.db")
    review_queue.initialise()
    ct_review_queue = CtReviewQueue(db_path=tmp / "ct_review_queue.db")
    ct_review_queue.initialise()

    all_metrics: list[ProtocolMetrics] = []
    batch_t0 = time.perf_counter()

    try:
        for i, pdf_path in enumerate(protocol_paths):
            nct_id = pdf_path.stem.rsplit("_", 1)[0]
            print(
                f"[{i + 1}/{len(protocol_paths)}] {nct_id} "
                f"({pdf_path.name})",
                file=sys.stderr,
                flush=True,
            )

            m = _run_one_protocol(
                pdf_path=pdf_path,
                gateway=gateway,
                review_queue=review_queue,
                ct_review_queue=ct_review_queue,
            )
            all_metrics.append(m)

            # Incremental per-protocol output.
            proto_out = protocols_dir / f"{nct_id}.json"
            proto_out.write_text(
                json.dumps(dataclasses.asdict(m), indent=2),
            )

            status_line = (
                f"  -> {m.status.upper()} "
                f"({m.total_elapsed_seconds:.1f}s)"
            )
            if m.failed_stage:
                status_line += f" FAILED at {m.failed_stage}"
            print(status_line, file=sys.stderr, flush=True)

            if verbose:
                for stage, secs in sorted(m.stage_timings.items()):
                    print(
                        f"     {stage}: {secs:.2f}s",
                        file=sys.stderr,
                    )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    total_elapsed = time.perf_counter() - batch_t0
    passing = [m for m in all_metrics if m.status == "pass"]

    failures_by_stage: dict[str, int] = {}
    for m in all_metrics:
        if m.failed_stage:
            failures_by_stage[m.failed_stage] = (
                failures_by_stage.get(m.failed_stage, 0) + 1
            )

    aggregate_metrics: dict[str, Any] = {}
    for key in _AGGREGATE_KEYS:
        aggregate_metrics[key] = _compute_aggregate(all_metrics, key)

    stage_timing_agg: dict[str, Any] = {}
    for stage in _STAGE_NAMES:
        vals = [m.stage_timings.get(stage, 0.0) for m in passing]
        if vals:
            stage_timing_agg[stage] = {
                "mean": round(statistics.mean(vals), 3),
                "median": round(statistics.median(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }
    aggregate_metrics["stage_timings"] = stage_timing_agg

    report = BenchmarkReport(
        run_id=run_id,
        timestamp_utc=ts.isoformat(),
        pipeline_capabilities={
            "extraction_level": extraction_level or "auto",
            "classification_level": classification_level or "auto",
        },
        protocol_count=len(all_metrics),
        pass_count=len(passing),
        error_count=len(all_metrics) - len(passing),
        failures_by_stage=failures_by_stage,
        total_elapsed_seconds=round(total_elapsed, 3),
        aggregate_metrics=aggregate_metrics,
        protocol_results=all_metrics,
    )

    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(
        json.dumps(dataclasses.asdict(report), indent=2),
    )
    print(f"\nReport: {report_path}", file=sys.stderr)

    return report
