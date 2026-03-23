"""PTCV Streamlit application — main entry point (PTCV-32, PTCV-33, PTCV-36, PTCV-61).

Launch with::

    streamlit run src/ptcv/ui/app.py

Provides:
- Sidebar file browser listing protocol PDFs by registry source
- "Parse PDF" action that runs extraction + retemplating (PTCV-60 order)
- ICH compliance verdict badge with confidence and section details
- Coverage review score for retemplating quality (PTCV-60)
- "Create mock SDTM file" action with lineage visualization
- Session-level result caching keyed by file SHA-256

Pipeline order (PTCV-60 document-first):
  extraction → SoA extraction → retemplating → coverage review → SDTM
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure src/ is on sys.path for non-installed package
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pyarrow.parquet as pq
import streamlit as st

# ---------------------------------------------------------------------------
# Auto-load secrets from .secrets file (PTCV-104)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/ptcv/ui -> PTCV root


def _load_secrets() -> list[str]:
    """Load secrets from .secrets file if env vars are not already set.

    Reads KEY=VALUE pairs from the project-root .secrets file and sets them
    as environment variables when they are not already present.

    Returns:
        List of warning messages for missing required secrets.
    """
    secrets_file = _PROJECT_ROOT / ".secrets"
    warnings: list[str] = []

    if not secrets_file.exists():
        # Check common required keys
        if not os.environ.get("ANTHROPIC_API_KEY"):
            warnings.append(
                "ANTHROPIC_API_KEY is not set and no .secrets file found at "
                f"`{secrets_file}`. Copy `.secrets.example` to `.secrets` "
                "and add your credentials, or run `source ./load-secrets.sh`."
            )
        return warnings

    loaded = 0
    for line in secrets_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        # Only set if not already in environment (env takes precedence)
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1

    if loaded:
        logging.getLogger(__name__).info(
            "PTCV-104: Loaded %d secret(s) from %s", loaded, secrets_file,
        )

    # Validate required secrets
    if not os.environ.get("ANTHROPIC_API_KEY"):
        warnings.append(
            "ANTHROPIC_API_KEY is not set. LLM features (retemplating, "
            "fidelity checking) will be unavailable. Add it to `.secrets` "
            "or set the environment variable."
        )

    return warnings


# Run once at import time so secrets are available before any module
# tries to read os.environ["ANTHROPIC_API_KEY"].
_SECRET_WARNINGS = _load_secrets()

from ptcv.extraction import ExtractionService, parquet_to_tables, parquet_to_text_blocks
from ptcv.ich_parser import IchParser
from ptcv.ich_parser.coverage_reviewer import CoverageReviewer
from ptcv.ich_parser.fidelity_checker import FidelityChecker
from ptcv.ich_parser.llm_retemplater import LlmRetemplater
from ptcv.ich_parser.parquet_writer import parquet_to_sections
from ptcv.sdtm import SdtmService
from ptcv.soa_extractor import SoaExtractor
from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    CoverageReport,
)
from ptcv.soa_extractor.query_bridge import (
    assembled_to_sections,
    get_assembled_protocol,
    has_query_pipeline_results,
)
from ptcv.ui.query_persistence import (
    load_assembled_protocol,
)
from ptcv.soa_extractor.writer import UsdmParquetWriter
from ptcv.storage import FilesystemAdapter
from ptcv.ui.components.file_browser import render_file_browser
from ptcv.ui.components.ich_regenerator import (
    make_download_filename,
    regenerate_ich_markdown,
)
from ptcv.ui.components.schedule_of_visits import (
    render_schedule_of_visits,
    to_plot_data,
)
from ptcv.ui.components.annotation_review import render_annotation_review
from ptcv.ui.components.protocol_diff import build_diff_label, build_original_text
from ptcv.ui.checkpoint_manager import (
    clear_checkpoints,
    get_resume_label,
    has_checkpoints,
    load_checkpoints,
    save_checkpoint,
)
from ptcv.ui.components.provenance_renderer import (
    build_provenance_html,
    estimate_html_height,
)
from ptcv.ui.components.review_queue_viewer import (
    get_pending_count,
    render_review_queue,
)
from ptcv.ui.components.benchmark_viewer import render_benchmark
from ptcv.ui.components.query_pipeline import render_query_pipeline
from ptcv.ui.components.refinement_panel import render_refinement_panel
from ptcv.ui.components.mock_data_panel import (
    MockPipelineResult,
    render_mock_data_panel,
    render_mock_data_sidebar,
)
from ptcv.ui.components.registry_panel import render_registry_panel
from ptcv.ui.components.sdtm_viewer import render_sdtm_viewer
from ptcv.ui.components.section_align import align_sections, build_comparison_html
from ptcv.ui.pipeline_stages import STAGE_BY_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag: Process tab disabled by default (PTCV-242 deprecation).
# Set PTCV_ENABLE_PROCESS_TAB=1 to re-enable for backward compatibility.
# ---------------------------------------------------------------------------
_PROCESS_TAB_DISABLED = not bool(
    os.environ.get("PTCV_ENABLE_PROCESS_TAB", "")
)


def _build_tab_names(
    review_label: str,
    include_process: bool = True,
) -> list[str]:
    """Build the ordered tab name list.

    Primary tabs follow the 7-stage pipeline flow (PTCV-268):
      Query Pipeline → SoA & Observations → SDTM & Validation

    Legacy/debug tabs are grouped under an "Advanced" tab.

    Extracted for testability (PTCV-142).
    """
    names: list[str] = [
        "Query Pipeline",
        "SoA & Observations",
        "SDTM & Validation",
        "Advanced",
    ]
    return names


# Default paths relative to PTCV project root
_DATA_ROOT = Path("C:/Dev/PTCV/data")
_PROTOCOLS_DIR = _DATA_ROOT / "protocols"

# LLM retemplating available when API key is set
def _has_anthropic_key() -> bool:
    """Check at runtime so key set after import is detected."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))

# Model tier definitions (PTCV-78)
_MODEL_TIERS: dict[str, dict[str, str | float]] = {
    "Best Quality (Opus)": {
        "model_id": "claude-opus-4-6",
        "method": "llm",
        "cost_per_page_low": 0.015,
        "cost_per_page_high": 0.020,
    },
}
_DEFAULT_TIER = "Best Quality (Opus)"


def estimate_cost(page_count: int, tier: str) -> tuple[float, float]:
    """Estimate LLM cost based on page count and model tier.

    Args:
        page_count: Number of pages in the PDF.
        tier: Key into _MODEL_TIERS.

    Returns:
        (low_estimate, high_estimate) in USD.
    """
    cfg = _MODEL_TIERS[tier]
    low = page_count * float(cfg["cost_per_page_low"])
    high = page_count * float(cfg["cost_per_page_high"])
    return (low, high)


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    """Count pages in a PDF without full extraction.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        Page count (0 on failure).
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def _get_gateway() -> FilesystemAdapter:
    """Return a shared FilesystemAdapter (initialised once)."""
    if "gateway" not in st.session_state:
        gw = FilesystemAdapter(root=_DATA_ROOT)
        gw.initialise()
        st.session_state["gateway"] = gw
    return st.session_state["gateway"]


def _extract_registry_id(filename: str) -> str:
    """Extract registry ID from a protocol filename.

    Handles patterns like ``NCT00112827_1.0.pdf`` and
    ``2024-001234-22_00.pdf``.

    Args:
        filename: PDF filename (stem + extension).

    Returns:
        Registry ID string (portion before the first underscore).
    """
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    return parts[0]


def _extract_amendment(filename: str) -> str:
    """Extract amendment/version from a protocol filename.

    Args:
        filename: PDF filename.

    Returns:
        Amendment string (portion after the first underscore, or "00").
    """
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else "00"


def _compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _run_parse(
    pdf_bytes: bytes,
    filename: str,
    gateway: FilesystemAdapter,
    model_tier: str = _DEFAULT_TIER,
) -> dict:
    """Run extraction and retemplating pipeline (PTCV-60 order).

    Chains ExtractionService.extract() → read text_blocks →
    LlmRetemplater.retemplate() (or IchParser/RAGClassifier fallback)
    → CoverageReviewer.review() and returns verdict fields for
    display.

    Args:
        pdf_bytes: Raw PDF file bytes.
        filename: Original PDF filename.
        gateway: Initialised FilesystemAdapter.
        model_tier: Model cost tier key from _MODEL_TIERS (PTCV-78).

    Returns:
        Dict with extraction info, retemplating verdict, coverage
        score, and cached data for downstream stages.
    """
    registry_id = _extract_registry_id(filename)
    amendment = _extract_amendment(filename)
    source_sha = _compute_sha256(pdf_bytes)

    # Stage 1: Extraction
    extraction_svc = ExtractionService(gateway=gateway)
    extraction_result = extraction_svc.extract(
        protocol_data=pdf_bytes,
        registry_id=registry_id,
        amendment_number=amendment,
        source_sha256=source_sha,
        filename=filename,
        source="streamlit-ui",
    )

    # Read text blocks back from storage
    text_bytes = gateway.get_artifact(extraction_result.text_artifact_key)
    text_blocks = parquet_to_text_blocks(text_bytes)
    text_blocks_sorted = sorted(
        text_blocks, key=lambda b: (b.page_number, b.block_index)
    )
    text_block_dicts = [
        {"page_number": b.page_number, "text": b.text}
        for b in text_blocks_sorted
        if b.text.strip()
    ]

    if not text_block_dicts:
        text_block_dicts = [{
            "page_number": 1,
            "text": (
                f"Protocol {registry_id} "
                "(text extraction produced no content)"
            ),
        }]

    # Load pre-extracted tables for SoA bridge (PTCV-51)
    extracted_tables = None
    tables_key = extraction_result.tables_artifact_key
    if tables_key:
        try:
            tables_bytes = gateway.get_artifact(tables_key)
            extracted_tables = parquet_to_tables(tables_bytes)
        except Exception:
            pass

    # Read page count from metadata
    meta_bytes = gateway.get_artifact(
        extraction_result.metadata_artifact_key,
    )
    meta_df = pq.read_table(io.BytesIO(meta_bytes)).to_pandas()
    page_count = (
        int(meta_df["page_count"].iloc[0])
        if "page_count" in meta_df.columns
        else 0
    )

    # Stage 3: Retemplating — routed by model tier (PTCV-78)
    tier_cfg = _MODEL_TIERS.get(model_tier, _MODEL_TIERS[_DEFAULT_TIER])
    if _has_anthropic_key() and tier_cfg["method"] == "llm":
        # Opus tier: full LLM retemplater
        retemplater = LlmRetemplater(
            gateway=gateway,
            claude_model=str(tier_cfg["model_id"]),
        )
        retemplate_result = retemplater.retemplate(
            text_blocks=text_block_dicts,
            registry_id=registry_id,
            source_run_id=extraction_result.run_id,
            source_sha256=extraction_result.text_artifact_sha256,
        )
        format_verdict = retemplate_result.format_verdict
        format_confidence = retemplate_result.format_confidence
        section_count = retemplate_result.section_count
        review_count = retemplate_result.review_count
        missing_required = retemplate_result.missing_required_sections
        artifact_key = retemplate_result.artifact_key
        retemplating_run_id = retemplate_result.run_id
        artifact_sha256 = retemplate_result.artifact_sha256
        input_tokens = retemplate_result.input_tokens
        output_tokens = retemplate_result.output_tokens
        retemplated_artifact_key = retemplate_result.retemplated_artifact_key
        retemplating_method = "llm"
    else:
        # No API key: deterministic IchParser fallback
        protocol_text = "\n".join(str(d["text"]) for d in text_block_dicts)
        parser = IchParser(gateway=gateway)
        parse_result = parser.parse(
            text=protocol_text,
            registry_id=registry_id,
            source_run_id=extraction_result.run_id,
            source_sha256=extraction_result.text_artifact_sha256,
        )
        format_verdict = parse_result.format_verdict
        format_confidence = parse_result.format_confidence
        section_count = parse_result.section_count
        review_count = parse_result.review_count
        missing_required = parse_result.missing_required_sections
        artifact_key = parse_result.artifact_key
        retemplating_run_id = parse_result.run_id
        artifact_sha256 = parse_result.artifact_sha256
        input_tokens = 0
        output_tokens = 0
        retemplated_artifact_key = ""
        retemplating_method = "ich_parser"

    # Stage 4: Coverage review
    section_bytes = gateway.get_artifact(artifact_key)
    retemplated_sections = parquet_to_sections(section_bytes)

    reviewer = CoverageReviewer()
    coverage_result = reviewer.review(
        original_text_blocks=text_block_dicts,
        retemplated_sections=retemplated_sections,
    )

    return {
        "format_verdict": format_verdict,
        "format_confidence": format_confidence,
        "section_count": section_count,
        "review_count": review_count,
        "missing_required_sections": missing_required,
        "registry_id": registry_id,
        "amendment": amendment,
        "artifact_key": artifact_key,
        "text_artifact_key": extraction_result.text_artifact_key,
        "tables_artifact_key": extraction_result.tables_artifact_key,
        "metadata_artifact_key": extraction_result.metadata_artifact_key,
        # PTCV-60: extraction data for document-first SoA
        "extraction_run_id": extraction_result.run_id,
        "text_artifact_sha256": extraction_result.text_artifact_sha256,
        "text_block_dicts": text_block_dicts,
        "extracted_tables": extracted_tables,
        "page_count": page_count,
        # PTCV-60: retemplating metadata
        "retemplating_run_id": retemplating_run_id,
        "artifact_sha256": artifact_sha256,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retemplating_method": retemplating_method,
        "retemplated_artifact_key": retemplated_artifact_key,
        "model_tier": model_tier,
        # PTCV-60: coverage review
        "coverage_score": coverage_result.coverage_score,
        "coverage_passed": coverage_result.passed,
        "uncovered_block_count": len(coverage_result.uncovered_blocks),
    }


def _run_soa_extraction(
    cached: dict,
    pdf_bytes: bytes,
    gateway: FilesystemAdapter,
) -> dict:
    """Run document-first SoA extraction (PTCV-60).

    Uses pre-extracted tables and PDF table discovery — no ICH
    sections required. Reads back USDM Parquet artifacts and
    returns them as lists suitable for the schedule of visits
    component.

    Args:
        cached: Extraction result dict from _run_parse().
        pdf_bytes: Raw PDF file bytes for table discovery.
        gateway: Initialised FilesystemAdapter.

    Returns:
        Dict with keys: timepoints, activities, instances,
        registry_id, format_verdict, soa_artifact_keys.
    """
    extractor = SoaExtractor(gateway=gateway)
    soa_result = extractor.extract(
        registry_id=cached["registry_id"],
        source_run_id=cached["extraction_run_id"],
        source_sha256=cached["text_artifact_sha256"],
        pdf_bytes=pdf_bytes,
        text_blocks=cached["text_block_dicts"],
        page_count=cached.get("page_count", 0),
        extracted_tables=cached.get("extracted_tables"),
        # No sections= arg — document-first (PTCV-60)
    )

    writer = UsdmParquetWriter()
    timepoints = []
    activities = []
    instances = []

    if "timepoints" in soa_result.artifact_keys:
        tp_bytes = gateway.get_artifact(soa_result.artifact_keys["timepoints"])
        timepoints = writer.parquet_to_timepoints(tp_bytes)
    if "activities" in soa_result.artifact_keys:
        act_bytes = gateway.get_artifact(soa_result.artifact_keys["activities"])
        activities = writer.parquet_to_activities(act_bytes)
    if "scheduled_instances" in soa_result.artifact_keys:
        inst_bytes = gateway.get_artifact(
            soa_result.artifact_keys["scheduled_instances"]
        )
        instances = writer.parquet_to_instances(inst_bytes)

    return {
        "timepoints": timepoints,
        "activities": activities,
        "instances": instances,
        "registry_id": cached["registry_id"],
        "format_verdict": cached["format_verdict"],
        "soa_artifact_keys": soa_result.artifact_keys,
        "soa_run_id": soa_result.run_id,
        "timepoint_count": soa_result.timepoint_count,
        "activity_count": soa_result.activity_count,
    }


def _run_soa_from_query_pipeline(
    cached: dict,
    pdf_bytes: bytes,
    gateway: FilesystemAdapter,
    assembled: "AssembledProtocol",
) -> dict:
    """Run SoA extraction using query pipeline assembled content (PTCV-112).

    Converts assembled Appendix B sections to IchSection objects and
    passes them as a supplemental input to the document-first SoA
    pipeline. The sections from the query pipeline are used as an
    additional fallback — pre-extracted tables and PDF discovery still
    take priority per the document-first strategy (PTCV-60).

    Args:
        cached: Extraction result dict from _run_parse().
        pdf_bytes: Raw PDF file bytes for table discovery.
        gateway: Initialised FilesystemAdapter.
        assembled: AssembledProtocol from query pipeline.

    Returns:
        Dict with keys: timepoints, activities, instances, etc.
    """
    sections = assembled_to_sections(
        assembled,
        registry_id=cached["registry_id"],
        source_sha256=cached["text_artifact_sha256"],
    )

    extractor = SoaExtractor(gateway=gateway)
    soa_result = extractor.extract(
        registry_id=cached["registry_id"],
        source_run_id=cached["extraction_run_id"],
        source_sha256=cached["text_artifact_sha256"],
        pdf_bytes=pdf_bytes,
        text_blocks=cached["text_block_dicts"],
        page_count=cached.get("page_count", 0),
        extracted_tables=cached.get("extracted_tables"),
        sections=sections if sections else None,
    )

    writer = UsdmParquetWriter()
    timepoints = []
    activities = []
    instances = []

    if "timepoints" in soa_result.artifact_keys:
        tp_bytes = gateway.get_artifact(
            soa_result.artifact_keys["timepoints"],
        )
        timepoints = writer.parquet_to_timepoints(tp_bytes)
    if "activities" in soa_result.artifact_keys:
        act_bytes = gateway.get_artifact(
            soa_result.artifact_keys["activities"],
        )
        activities = writer.parquet_to_activities(act_bytes)
    if "scheduled_instances" in soa_result.artifact_keys:
        inst_bytes = gateway.get_artifact(
            soa_result.artifact_keys["scheduled_instances"],
        )
        instances = writer.parquet_to_instances(inst_bytes)

    return {
        "timepoints": timepoints,
        "activities": activities,
        "instances": instances,
        "registry_id": cached["registry_id"],
        "format_verdict": cached["format_verdict"],
        "soa_artifact_keys": soa_result.artifact_keys,
        "soa_run_id": soa_result.run_id,
        "timepoint_count": soa_result.timepoint_count,
        "activity_count": soa_result.activity_count,
    }


def _run_soa_query_pipeline_only(
    assembled: "AssembledProtocol",
    registry_id: str,
    pdf_bytes: bytes,
    gateway: FilesystemAdapter,
) -> dict:
    """Run SoA extraction using only query pipeline content (PTCV-112).

    Used when the Process tab has NOT been run but the Query Pipeline
    tab has results. Converts assembled sections to IchSection objects
    and uses them as the sole input — no pre-extracted tables or PDF
    table discovery.

    Args:
        assembled: AssembledProtocol from query pipeline.
        registry_id: Trial identifier from filename.
        pdf_bytes: Raw PDF bytes (for text_blocks extraction).
        gateway: Initialised FilesystemAdapter.

    Returns:
        Dict with keys: timepoints, activities, instances, etc.
    """
    sections = assembled_to_sections(
        assembled,
        registry_id=registry_id,
    )

    if not sections:
        return {
            "timepoints": [],
            "activities": [],
            "instances": [],
            "registry_id": registry_id,
            "format_verdict": "query_pipeline",
            "soa_artifact_keys": {},
            "soa_run_id": "",
            "timepoint_count": 0,
            "activity_count": 0,
        }

    extractor = SoaExtractor(gateway=gateway)
    soa_result = extractor.extract(
        registry_id=registry_id,
        sections=sections,
    )

    writer = UsdmParquetWriter()
    timepoints = []
    activities = []
    instances = []

    if "timepoints" in soa_result.artifact_keys:
        tp_bytes = gateway.get_artifact(
            soa_result.artifact_keys["timepoints"],
        )
        timepoints = writer.parquet_to_timepoints(tp_bytes)
    if "activities" in soa_result.artifact_keys:
        act_bytes = gateway.get_artifact(
            soa_result.artifact_keys["activities"],
        )
        activities = writer.parquet_to_activities(act_bytes)
    if "scheduled_instances" in soa_result.artifact_keys:
        inst_bytes = gateway.get_artifact(
            soa_result.artifact_keys["scheduled_instances"],
        )
        instances = writer.parquet_to_instances(inst_bytes)

    return {
        "timepoints": timepoints,
        "activities": activities,
        "instances": instances,
        "registry_id": registry_id,
        "format_verdict": "query_pipeline",
        "soa_artifact_keys": soa_result.artifact_keys,
        "soa_run_id": soa_result.run_id,
        "timepoint_count": soa_result.timepoint_count,
        "activity_count": soa_result.activity_count,
    }


def _run_sdtm_generation(
    cached: dict,
    soa_cached: dict | None,
    gateway: FilesystemAdapter,
) -> dict:
    """Run SDTM generation from retemplated sections and SoA data (PTCV-60).

    Reads retemplated ICH sections from storage, collects timepoints
    from SoA cache (if available), and runs SdtmService.generate().

    Args:
        cached: Parse result dict from _run_parse() (artifact_key
            points to retemplated sections).
        soa_cached: SoA extraction result dict (may be None).
        gateway: Initialised FilesystemAdapter.

    Returns:
        Dict with keys: sdtm_result, sections, has_soa,
        registry_id, format_verdict.
    """
    section_bytes = gateway.get_artifact(cached["artifact_key"])
    sections = parquet_to_sections(section_bytes)

    timepoints = []
    if soa_cached and soa_cached.get("timepoints"):
        timepoints = soa_cached["timepoints"]

    sdtm_svc = SdtmService(gateway=gateway)
    sdtm_result = sdtm_svc.generate(
        sections=sections,
        timepoints=timepoints,
        registry_id=cached["registry_id"],
        amendment_number=cached.get("amendment", "00"),
    )

    return {
        "sdtm_result": sdtm_result,
        "sections": sections,
        "has_soa": len(timepoints) > 0,
        "registry_id": cached["registry_id"],
        "format_verdict": cached["format_verdict"],
    }


def _run_sdtm_from_assembled(
    assembled: "AssembledProtocol",
    soa_cached: dict | None,
    gateway: FilesystemAdapter,
    registry_id: str,
) -> dict:
    """Run SDTM generation from query pipeline AssembledProtocol (PTCV-140).

    Args:
        assembled: Completed AssembledProtocol from the query pipeline.
        soa_cached: SoA extraction result dict (may be None).
        gateway: Initialised FilesystemAdapter.
        registry_id: Trial registry identifier.

    Returns:
        Dict with keys: sdtm_result, sections, has_soa,
        registry_id, format_verdict.
    """
    timepoints = []
    if soa_cached and soa_cached.get("timepoints"):
        timepoints = soa_cached["timepoints"]

    sdtm_svc = SdtmService(gateway=gateway)
    sdtm_result = sdtm_svc.generate_from_assembled(
        assembled=assembled,
        timepoints=timepoints,
        registry_id=registry_id,
    )

    # Build IchSections for the SDTM viewer (display purposes).
    sections = assembled_to_sections(assembled, registry_id=registry_id)

    return {
        "sdtm_result": sdtm_result,
        "sections": sections,
        "has_soa": len(timepoints) > 0,
        "registry_id": registry_id,
        "format_verdict": "query_pipeline",
    }


def _run_fidelity_check(
    cached: dict,
    gateway: FilesystemAdapter,
    enable_llm: bool = True,
) -> dict:
    """Run fidelity check on retemplated protocol (PTCV-65).

    Args:
        cached: Parse result dict from _run_parse().
        gateway: Initialised FilesystemAdapter.
        enable_llm: Whether to enable LLM semantic check.

    Returns:
        Dict with fidelity check results for display.
    """
    section_bytes = gateway.get_artifact(cached["artifact_key"])
    sections = parquet_to_sections(section_bytes)
    text_block_dicts = cached["text_block_dicts"]

    checker = FidelityChecker(enable_llm=enable_llm)
    result = checker.check(
        original_text_blocks=text_block_dicts,
        retemplated_sections=sections,
        registry_id=cached["registry_id"],
        source_sha256=cached.get("artifact_sha256", ""),
    )

    return {
        "fidelity_passed": result.fidelity_passed,
        "overall_score": result.overall_score,
        "section_results": result.section_results,
        "total_hallucinations": result.total_hallucinations,
        "total_omissions": result.total_omissions,
        "total_drift_flags": result.total_drift_flags,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "method": result.method,
    }


def _display_fidelity(fidelity_cached: dict) -> None:
    """Render fidelity check results (PTCV-65).

    Args:
        fidelity_cached: Result dict from _run_fidelity_check().
    """
    st.subheader("Fidelity Check Results")

    if fidelity_cached["fidelity_passed"]:
        st.success("Fidelity check: PASS")
    else:
        st.error("Fidelity check: FAIL")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Overall score",
            f"{fidelity_cached['overall_score']:.2f}",
        )
    with col2:
        st.metric(
            "Hallucinations",
            fidelity_cached["total_hallucinations"],
        )
    with col3:
        st.metric(
            "Omissions",
            fidelity_cached["total_omissions"],
        )
    with col4:
        st.metric(
            "Drift flags",
            fidelity_cached["total_drift_flags"],
        )

    method = fidelity_cached["method"]
    if method == "llm":
        st.caption(
            f"LLM fidelity check: "
            f"{fidelity_cached['input_tokens']:,} in / "
            f"{fidelity_cached['output_tokens']:,} out tokens "
            f"(~${fidelity_cached['estimated_cost_usd']:.2f})"
        )
    else:
        st.caption("Deterministic-only fidelity check")

    # Per-section expandable details
    for sr in fidelity_cached["section_results"]:
        label = (
            f"{sr.section_code}: "
            f"{sr.fidelity_score:.2f}"
        )

        with st.expander(label):
            if sr.hallucinations:
                st.markdown("**Hallucinations:**")
                for h in sr.hallucinations:
                    st.markdown(f"- {h}")
            if sr.omissions:
                st.markdown("**Omissions:**")
                for o in sr.omissions:
                    st.markdown(f"- {o}")
            if sr.drift_flags:
                st.markdown("**Drift:**")
                for d in sr.drift_flags:
                    st.markdown(f"- {d}")
            if not (
                sr.hallucinations
                or sr.omissions
                or sr.drift_flags
            ):
                st.markdown("No issues detected.")


def compute_quality_verdict(
    coverage: CoverageReport,
) -> dict:
    """Compute quality verdict from a CoverageReport (PTCV-138).

    Pure function — no Streamlit dependency. Returns a dict
    with verdict, coverage_pct, and query_pct for testability.

    Args:
        coverage: CoverageReport from an AssembledProtocol.

    Returns:
        Dict with keys: verdict ("PASS", "REVIEW", "FAIL"),
        coverage_pct (float 0-100), query_pct (float 0-100).
    """
    coverage_pct = (
        coverage.populated_count / coverage.total_sections * 100
        if coverage.total_sections > 0
        else 0.0
    )
    query_pct = (
        coverage.answered_queries / coverage.total_queries * 100
        if coverage.total_queries > 0
        else 0.0
    )

    if coverage_pct >= 75 and coverage.average_confidence >= 0.70:
        verdict = "PASS"
    elif coverage_pct >= 50:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "coverage_pct": coverage_pct,
        "query_pct": query_pct,
    }


def _display_query_quality(
    assembled: AssembledProtocol,
) -> None:
    """Render quality metrics from query pipeline results (PTCV-138).

    Shows section coverage, confidence distribution, query answer
    rate, and per-section details derived from the AssembledProtocol
    coverage report — replacing the Process-tab-only fidelity check
    when query pipeline results are available.

    Args:
        assembled: Completed AssembledProtocol from the query pipeline.
    """
    cov = assembled.coverage
    st.subheader("Query Pipeline Quality")

    qv = compute_quality_verdict(cov)
    coverage_pct = qv["coverage_pct"]

    if qv["verdict"] == "PASS":
        st.success(
            f"Quality: PASS — {coverage_pct:.0f}% section coverage, "
            f"{cov.average_confidence:.0%} avg confidence"
        )
    elif qv["verdict"] == "REVIEW":
        st.warning(
            f"Quality: REVIEW — {coverage_pct:.0f}% section coverage, "
            f"{cov.average_confidence:.0%} avg confidence"
        )
    else:
        st.error(
            f"Quality: FAIL — {coverage_pct:.0f}% section coverage, "
            f"{cov.average_confidence:.0%} avg confidence"
        )

    # Top-level metrics.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Sections Populated",
            f"{cov.populated_count}/{cov.total_sections}",
        )
    with col2:
        st.metric("Gap Sections", cov.gap_count)
    with col3:
        st.metric(
            "Queries Answered",
            f"{cov.answered_queries}/{cov.total_queries} "
            f"({qv['query_pct']:.0f}%)",
        )
    with col4:
        st.metric(
            "Avg Confidence",
            f"{cov.average_confidence:.2f}",
        )

    # Confidence distribution bar.
    col_h, col_m, col_l = st.columns(3)
    with col_h:
        st.metric("High Confidence", cov.high_confidence_count)
    with col_m:
        st.metric("Medium Confidence", cov.medium_confidence_count)
    with col_l:
        st.metric("Low Confidence", cov.low_confidence_count)

    # Gap sections.
    if cov.gap_sections:
        st.markdown(
            "**Gap sections (no content):** "
            + ", ".join(cov.gap_sections)
        )

    # Low confidence sections.
    if cov.low_confidence_sections:
        st.markdown(
            "**Low confidence sections:** "
            + ", ".join(cov.low_confidence_sections)
        )

    # Per-section expandable details.
    st.divider()
    st.subheader("Per-Section Detail")
    for section in assembled.sections:
        if section.is_gap:
            icon = "gap"
        elif section.has_low_confidence:
            icon = "low"
        else:
            icon = "ok"

        label = (
            f"{section.section_code} — "
            f"{section.section_name} "
            f"[{icon}] "
            f"(conf: {section.average_confidence:.2f}, "
            f"queries: {section.answered_required_count}"
            f"/{section.required_query_count})"
        )

        with st.expander(label):
            if section.is_gap:
                st.warning("No content extracted for this section.")
            elif section.has_low_confidence:
                st.warning(
                    "One or more queries returned low-confidence "
                    "results. Manual review recommended."
                )
            else:
                st.info("Section populated with adequate confidence.")

            if section.hits:
                for hit in section.hits:
                    conf_str = f"{hit.confidence:.2f}"
                    preview = hit.extracted_content[:200]
                    if len(hit.extracted_content) > 200:
                        preview += "..."
                    st.markdown(
                        f"- **{hit.query_id}** "
                        f"(conf {conf_str}): {preview}"
                    )


def _display_query_results(
    assembled: AssembledProtocol,
    key_prefix: str = "",
) -> None:
    """Render extraction results from query pipeline (PTCV-137).

    Shows quality metrics (via :func:`_display_query_quality`),
    the assembled Appendix B template markdown preview, and a
    download button.

    Args:
        assembled: Completed AssembledProtocol from the query pipeline.
        key_prefix: Optional prefix for Streamlit widget keys to avoid
            duplicates when rendered in multiple tabs (PTCV-278).
    """
    _display_query_quality(assembled)

    st.divider()
    st.subheader("Assembled Appendix B Template")
    md = assembled.to_markdown()
    with st.container(height=500):
        st.markdown(md)

    st.download_button(
        label="Download Assembled Template (.md)",
        data=md,
        file_name="assembled_appendix_b.md",
        mime="text/markdown",
        key=f"{key_prefix}btn_dl_results_assembled",
    )


def _display_verdict(result: dict) -> None:
    """Render retemplating verdict and coverage score in the main panel.

    Args:
        result: Dict from _run_parse() containing verdict,
            coverage, and retemplating metadata.
    """
    verdict = result["format_verdict"]
    confidence = result["format_confidence"]
    section_count = result["section_count"]
    missing = result["missing_required_sections"]

    st.subheader("ICH Compliance")

    # Verdict badge
    if verdict == "ICH_E6R3":
        st.success(f"**{verdict}**")
    elif verdict == "PARTIAL_ICH":
        st.warning(f"**{verdict}**")
    else:
        st.error(f"**{verdict}**")
        st.warning(
            "SDTM output from this protocol may be incomplete "
            "— manual review recommended"
        )

    # Details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Format confidence", f"{confidence:.2f}")
    with col2:
        st.metric("Sections detected", f"{section_count} / 11")
    with col3:
        coverage = result.get("coverage_score")
        if coverage is not None:
            st.metric("Coverage", f"{coverage:.0%}")

    if missing:
        st.caption(f"Missing required sections: {', '.join(missing)}")

    if result["review_count"] > 0:
        st.caption(
            f"{result['review_count']} section(s) flagged for human review"
        )

    # Coverage review detail (PTCV-60)
    if "coverage_passed" in result:
        if result["coverage_passed"]:
            st.success("Coverage review: PASS")
        else:
            st.warning(
                f"Coverage review: FAIL "
                f"({result.get('uncovered_block_count', 0)} "
                f"uncovered block(s))"
            )

    # Retemplating method and token usage (PTCV-60, PTCV-78)
    method = result.get("retemplating_method", "ich_parser")
    tier = result.get("model_tier", "")
    if method == "llm" and result.get("input_tokens", 0) > 0:
        tier_label = f" [{tier}]" if tier else ""
        st.caption(
            f"LLM retemplating{tier_label}: "
            f"{result['input_tokens']:,} in / "
            f"{result['output_tokens']:,} out tokens"
        )
    elif method == "ich_parser":
        st.caption(
            "Using IchParser fallback "
            "(set ANTHROPIC_API_KEY for LLM retemplating)"
        )


def _render_regeneration(cached: dict) -> None:
    """Render the ICH E6(R3) regenerated protocol view (PTCV-35, PTCV-79, PTCV-80).

    Shows three tabs: "Retemplated" (markdown view), "Diff" (word-level
    diff via streamlit-code-diff), and "Section Compare" (synchronized
    scrolling with section alignment).

    Args:
        cached: Parse result dict with artifact_key, registry_id, etc.
    """
    gateway = _get_gateway()

    # Load retemplated sections for alignment (PTCV-80)
    section_bytes = gateway.get_artifact(cached["artifact_key"])
    sections = parquet_to_sections(section_bytes)

    # PTCV-64: Use pre-generated retemplated markdown if available
    retemplated_key = cached.get("retemplated_artifact_key", "")
    if retemplated_key:
        md_bytes = gateway.get_artifact(retemplated_key)
        md = md_bytes.decode("utf-8")
    else:
        md = regenerate_ich_markdown(
            sections=sections,
            registry_id=cached["registry_id"],
            format_verdict=cached["format_verdict"],
            format_confidence=cached["format_confidence"],
        )

    st.subheader("ICH E6(R3) Reformatted Protocol")

    tab_retemplated, tab_provenance, tab_diff, tab_section = st.tabs(
        ["Retemplated", "Provenance", "Diff", "Section Compare"],
    )

    with tab_retemplated:
        with st.container(height=500):
            st.markdown(md)

        dl_filename = make_download_filename(
            cached["registry_id"],
            cached.get("amendment", "00"),
        )
        st.download_button(
            label="Download as .md",
            data=md,
            file_name=dl_filename,
            mime="text/markdown",
        )

    # PTCV-81: Provenance annotation view
    with tab_provenance:
        show_prov = st.toggle(
            "Show annotations",
            value=True,
            key="provenance_toggle",
        )
        if show_prov and sections:
            import streamlit.components.v1 as components
            prov_html = build_provenance_html(
                sections=sections,
                registry_id=cached["registry_id"],
            )
            components.html(
                prov_html,
                height=estimate_html_height(sections),
                scrolling=True,
            )
        elif not sections:
            st.info(
                "Run ICH Retemplating first to enable "
                "provenance view"
            )
        else:
            st.caption("Annotations hidden. Toggle above to show.")

    with tab_diff:
        text_block_dicts = cached.get("text_block_dicts")
        if not text_block_dicts:
            st.info(
                "Run ICH Retemplating first to enable comparison"
            )
        else:
            original = build_original_text(text_block_dicts)
            left_label, right_label = build_diff_label(
                cached["registry_id"],
                cached.get("retemplating_method", ""),
            )

            mode = st.radio(
                "Display mode",
                ["Side by side", "Line by line"],
                horizontal=True,
                key="diff_mode",
            )

            try:
                from streamlit_code_diff import st_code_diff
                st_code_diff(
                    old_string=original,
                    new_string=md,
                    filename=left_label,
                    new_filename=right_label,
                    output_format=(
                        "side-by-side"
                        if mode == "Side by side"
                        else "line-by-line"
                    ),
                    language="markdown",
                )
            except ImportError:
                st.warning(
                    "Install `streamlit-code-diff` for diff view: "
                    "`pip install streamlit-code-diff`"
                )

    # PTCV-80: Section-aligned comparison with sync scrolling
    with tab_section:
        if not sections:
            st.info(
                "Run ICH Retemplating first to enable comparison"
            )
        else:
            pairs = align_sections(sections, md)
            if pairs:
                import streamlit.components.v1 as components
                html_content = build_comparison_html(pairs)
                components.html(html_content, height=660, scrolling=False)
            else:
                st.info("No sections available for alignment.")


def _is_cached(file_sha: str, stage_key: str) -> bool:
    """Check if a stage's result is already cached for this file."""
    cache_key = STAGE_BY_KEY[stage_key].cache_key
    if not cache_key:
        return False
    return file_sha in st.session_state.get(cache_key, {})


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="PTCV — Protocol Viewer",
        page_icon="🧬",
        layout="wide",
    )
    st.title("Protocol-To-Cohort-Viz")

    # Show secrets warnings (PTCV-104)
    for _warn in _SECRET_WARNINGS:
        st.warning(_warn)

    # Sidebar: mode selector + file browser
    with st.sidebar:
        app_mode = st.radio(
            "Mode",
            ["Protocol Viewer", "Batch Analysis"],
            key="app_mode",
            horizontal=True,
        )

    if app_mode == "Batch Analysis":
        from ptcv.ui.pages.analysis_page import render_analysis_page
        render_analysis_page()
        return

    source, file_path = render_file_browser(_PROTOCOLS_DIR)

    if file_path is None:
        st.info("Select a protocol file from the sidebar to get started.")
        return

    # Read PDF and compute SHA for caching
    pdf_bytes = file_path.read_bytes()
    file_sha = _compute_sha256(pdf_bytes)
    page_count = _count_pdf_pages(pdf_bytes)
    registry_id = _extract_registry_id(file_path.name)

    st.caption(f"{source}/{file_path.name}")

    # Initialise caches
    for ck in (
        "parse_cache", "soa_cache", "sdtm_cache", "fidelity_cache",
        "query_cache", "benchmark_cache", "classified_cache",
    ):
        if ck not in st.session_state:
            st.session_state[ck] = {}

    # Read cached results
    cached = st.session_state["parse_cache"].get(file_sha)
    soa_cached = st.session_state["soa_cache"].get(file_sha)
    fidelity_cached = st.session_state["fidelity_cache"].get(file_sha)
    sdtm_cached = st.session_state["sdtm_cache"].get(file_sha)

    # ------------------------------------------------------------------
    # Tabbed wizard
    # ------------------------------------------------------------------
    pending = get_pending_count(registry_id=registry_id)
    review_label = (
        f"Review ({pending})" if pending > 0 else "Review"
    )

    # ------------------------------------------------------------------
    # PTCV-268: Pipeline-aligned tab layout
    # Primary: Query Pipeline → SoA & Observations → SDTM & Validation
    # Legacy tabs grouped under Advanced.
    # ------------------------------------------------------------------
    _tab_names = _build_tab_names(review_label)
    _tabs = dict(zip(_tab_names, st.tabs(_tab_names)))
    tab_query = _tabs["Query Pipeline"]
    tab_soa = _tabs["SoA & Observations"]
    tab_sdtm = _tabs["SDTM & Validation"]
    tab_advanced = _tabs["Advanced"]

    # ------------------------------------------------------------------
    # Query pipeline assembled protocol — shared by Results, Quality,
    # & SoA tabs.  Fetched once here (PTCV-138, PTCV-137).
    # ------------------------------------------------------------------
    _qp_assembled = get_assembled_protocol(
        st.session_state, file_sha,
    )
    if _qp_assembled is None:
        _qp_assembled = load_assembled_protocol(
            _get_gateway(), file_sha,
        )

    # === Tab 1: Query Pipeline (PTCV-268) ===
    with tab_query:
        render_query_pipeline(file_path, file_sha)

        # Show results inline if available
        if _qp_assembled is not None:
            st.divider()
            _display_query_results(_qp_assembled, key_prefix="qp_")

    # === Tab 2: SoA & Observations (PTCV-268) ===
    _soa_available = bool(cached) or _qp_assembled is not None

    with tab_soa:
        if not _soa_available:
            st.info(
                "Run the Process tab or Query Pipeline first "
                "to enable SoA extraction and SDTM generation."
            )
        else:
            # SoA extraction
            if soa_cached:
                # PTCV-250: Indicate source of SoA results
                _soa_source = soa_cached.get(
                    "_source", "manual"
                )
                if _soa_source == "query_pipeline":
                    st.caption(
                        "SoA results auto-populated from "
                        "Query Pipeline (Stage 5)."
                    )

                plot_data = to_plot_data(
                    soa_cached["timepoints"],
                    soa_cached["activities"],
                    soa_cached["instances"],
                )
                render_schedule_of_visits(
                    timepoints=plot_data.timepoints,
                    activities=plot_data.activities,
                    instances=plot_data.instances,
                    registry_id=soa_cached["registry_id"],
                    format_verdict=soa_cached["format_verdict"],
                )
            else:
                # PTCV-112: offer query pipeline integration
                use_qp = False
                if _qp_assembled is not None and cached:
                    use_qp = st.checkbox(
                        "Use Query Pipeline output",
                        value=True,
                        help=(
                            "Feed assembled Appendix B content "
                            "(B.4/B.7) from the Query Pipeline "
                            "as additional input to SoA extraction."
                        ),
                        key="chk_soa_use_qp",
                    )
                elif _qp_assembled is not None and not cached:
                    # Query pipeline only — always use it
                    use_qp = True
                    st.caption(
                        "Using Query Pipeline assembled content "
                        "for SoA extraction (Process tab not run)."
                    )

                if st.button(
                    "Run SoA Extraction",
                    type="primary",
                    key="btn_soa",
                ):
                    gateway = _get_gateway()
                    with st.status(
                        "Running SoA Extraction...",
                        expanded=True,
                    ) as status:
                        t0 = time.monotonic()
                        try:
                            if use_qp and _qp_assembled and cached:
                                st.write(
                                    "Discovering tables with "
                                    "query pipeline content...",
                                )
                                soa_result = _run_soa_from_query_pipeline(
                                    cached, pdf_bytes, gateway,
                                    _qp_assembled,
                                )
                            elif use_qp and _qp_assembled:
                                st.write(
                                    "Extracting from query "
                                    "pipeline content...",
                                )
                                soa_result = _run_soa_query_pipeline_only(
                                    _qp_assembled,
                                    registry_id,
                                    pdf_bytes,
                                    gateway,
                                )
                            else:
                                st.write(
                                    "Discovering tables and "
                                    "timepoints...",
                                )
                                soa_result = _run_soa_extraction(
                                    cached, pdf_bytes, gateway,
                                )
                            elapsed = time.monotonic() - t0
                            st.session_state["soa_cache"][
                                file_sha
                            ] = soa_result
                            soa_cached = soa_result
                            save_checkpoint(
                                _DATA_ROOT, file_sha,
                                "soa_cache", soa_result,
                            )
                            status.update(
                                label=(
                                    "SoA Extraction: "
                                    f"Complete ({elapsed:.1f}s)"
                                ),
                                state="complete",
                            )
                            st.rerun()
                        except Exception:
                            elapsed = time.monotonic() - t0
                            status.update(
                                label=(
                                    "SoA Extraction: "
                                    f"Error ({elapsed:.1f}s)"
                                ),
                                state="error",
                            )
                            st.code(
                                traceback.format_exc(),
                                language="text",
                            )

        # --- Observation Domain Specs & EX (PTCV-271) ---
        if sdtm_cached and sdtm_cached.get("sdtm_result") is not None:
            from ptcv.ui.components.observation_specs_panel import (
                render_ex_domain_spec,
                render_observation_specs,
            )

            _sdtm_result = sdtm_cached["sdtm_result"]
            st.divider()
            render_observation_specs(
                getattr(_sdtm_result, "domain_specs", None),
            )
            render_ex_domain_spec(
                getattr(_sdtm_result, "ex_domain_spec", None),
            )

    # === Tab 3: SDTM & Validation (PTCV-268, PTCV-272) ===
    with tab_sdtm:
        if sdtm_cached and sdtm_cached.get("sdtm_result") is not None:
            render_sdtm_viewer(
                sdtm_result=sdtm_cached.get("sdtm_result"),
                sections=sdtm_cached.get("sections", []),
                has_soa=sdtm_cached.get("has_soa", False),
                gateway=_get_gateway(),
                registry_id=sdtm_cached.get("registry_id", ""),
                format_verdict=sdtm_cached.get("format_verdict", ""),
            )

            # --- Domain checklist & validation (PTCV-272) ---
            from ptcv.ui.components.validation_results_panel import (
                render_domain_checklist,
                render_validation_results,
            )

            _sdtm_res = sdtm_cached["sdtm_result"]

            # Validation results (from pipeline or cached)
            _val_result = st.session_state.get(
                "validation_cache", {},
            ).get(file_sha)
            if _val_result is not None:
                st.divider()
                # _val_result may be dict (query pipeline) or
                # ValidationResult (PTCV-289)
                _domain_check = (
                    _val_result.get("domain_check")
                    if isinstance(_val_result, dict)
                    else getattr(_val_result, "domain_check", None)
                )
                render_domain_checklist(_domain_check)
                st.divider()
                render_validation_results(_val_result)
        elif soa_cached and (cached or _qp_assembled is not None):
            # Prefer query pipeline when assembled protocol exists
            _use_assembled = _qp_assembled is not None
            _source_label = (
                "Source: Query Pipeline"
                if _use_assembled
                else "Source: Process Tab"
            )
            st.caption(_source_label)
            if st.button(
                "Generate SDTM",
                type="primary",
                key="btn_sdtm",
            ):
                gateway = _get_gateway()
                with st.status(
                    "Running SDTM Generation...",
                    expanded=True,
                ) as status:
                    t0 = time.monotonic()
                    st.write("Generating SDTM domains...")
                    try:
                        if _use_assembled:
                            sdtm_result = _run_sdtm_from_assembled(
                                _qp_assembled,
                                soa_cached,
                                gateway,
                                registry_id=(
                                    cached["registry_id"]
                                    if cached
                                    else file_sha[:20]
                                ),
                            )
                        else:
                            sdtm_result = _run_sdtm_generation(
                                cached, soa_cached, gateway,
                            )
                        elapsed = time.monotonic() - t0
                        st.session_state["sdtm_cache"][
                            file_sha
                        ] = sdtm_result
                        sdtm_cached = sdtm_result
                        save_checkpoint(
                            _DATA_ROOT, file_sha,
                            "sdtm_cache", sdtm_result,
                        )
                        status.update(
                            label=(
                                "SDTM Generation: "
                                f"Complete ({elapsed:.1f}s)"
                            ),
                            state="complete",
                        )
                        st.rerun()
                    except Exception:
                        elapsed = time.monotonic() - t0
                        status.update(
                            label=(
                                "SDTM Generation: "
                                f"Error ({elapsed:.1f}s)"
                            ),
                            state="error",
                        )
                        st.code(
                            traceback.format_exc(),
                            language="text",
                        )
        elif soa_cached and not cached and _qp_assembled is None:
            st.caption(
                "SDTM generation requires the Query Pipeline. "
                "Run it first."
            )
        else:
            st.info(
                "Run the **Query Pipeline** and **SoA Extraction** "
                "first to enable SDTM generation."
            )

    # === Tab 4: Advanced (PTCV-268) ===
    # Legacy/debug tabs consolidated into sub-tabs within Advanced.
    with tab_advanced:
        _adv_names = [
            "Results", "Quality", "Benchmark",
            "Refinement", "Mock Data", review_label,
        ]
        _adv_tabs = dict(zip(_adv_names, st.tabs(_adv_names)))

        # --- Results sub-tab ---
        with _adv_tabs["Results"]:
            _registry_cache_dir = (
                _PROTOCOLS_DIR / "clinicaltrials" / "registry_cache"
            )
            render_registry_panel(
                _registry_cache_dir, registry_id,
            )
            if _qp_assembled is not None:
                _display_query_results(_qp_assembled, key_prefix="adv_")
            elif cached:
                _display_verdict(cached)
                st.divider()
                _render_regeneration(cached)
            else:
                st.info(
                    "Run the **Query Pipeline** tab to "
                    "generate results."
                )

        # --- Quality sub-tab ---
        with _adv_tabs["Quality"]:
            if _qp_assembled is not None:
                _display_query_quality(_qp_assembled)
            elif cached:
                if fidelity_cached:
                    _display_fidelity(fidelity_cached)
                else:
                    if st.button(
                        "Run Fidelity Check",
                        type="primary",
                        key="btn_fidelity",
                    ):
                        gateway = _get_gateway()
                        with st.status(
                            "Running Fidelity Check...",
                            expanded=True,
                        ) as status:
                            t0 = time.monotonic()
                            try:
                                fidelity_result = (
                                    _run_fidelity_check(
                                        cached, gateway,
                                        enable_llm=_has_anthropic_key(),
                                    )
                                )
                                elapsed = time.monotonic() - t0
                                st.session_state[
                                    "fidelity_cache"
                                ][file_sha] = fidelity_result
                                status.update(
                                    label=f"Fidelity Check: Complete ({elapsed:.1f}s)",
                                    state="complete",
                                )
                                st.rerun()
                            except Exception:
                                elapsed = time.monotonic() - t0
                                status.update(
                                    label=f"Fidelity Check: Error ({elapsed:.1f}s)",
                                    state="error",
                                )
                                st.code(traceback.format_exc(), language="text")
            else:
                st.info("Run the **Query Pipeline** first to enable quality checks.")

        # --- Benchmark sub-tab ---
        with _adv_tabs["Benchmark"]:
            query_cached = st.session_state["query_cache"].get(file_sha)
            render_benchmark(file_path, file_sha, query_cached)

        # --- Refinement sub-tab ---
        with _adv_tabs["Refinement"]:
            query_cached = st.session_state["query_cache"].get(file_sha)
            render_refinement_panel(registry_id, query_cached)

        # --- Mock Data sub-tab ---
        with _adv_tabs["Mock Data"]:
            if not soa_cached:
                st.info("Run SoA Extraction first to enable mock SDTM data generation.")
            else:
                mock_cfg = render_mock_data_sidebar()
                mock_cache_key = (
                    f"mock_{file_sha}_"
                    f"{mock_cfg['num_subjects']}_"
                    f"{mock_cfg['synthesizer_type']}"
                )
                if "mock_data_cache" not in st.session_state:
                    st.session_state["mock_data_cache"] = {}
                mock_cached = st.session_state["mock_data_cache"].get(mock_cache_key)
                if mock_cached is not None:
                    render_mock_data_panel(mock_cached)
                else:
                    if st.button("Generate Mock Data", type="primary", key="btn_mock_gen"):
                        with st.status("Generating mock SDTM data...", expanded=True) as status:
                            t0 = time.monotonic()
                            try:
                                from ptcv.mock_data.sdtm_metadata import get_all_domain_specs
                                from ptcv.mock_data.sdv_synthesizer import SdvConfig, SdvSynthesizerService
                                specs = get_all_domain_specs()
                                cfg = SdvConfig(
                                    synthesizer_type=mock_cfg["synthesizer_type"],
                                    num_subjects=mock_cfg["num_subjects"],
                                )
                                svc = SdvSynthesizerService(cfg)
                                gen_result = svc.generate_all(specs, num_subjects=mock_cfg["num_subjects"])
                                pipeline_result = MockPipelineResult(
                                    domain_dataframes=gen_result.domain_dataframes,
                                    validation_results={},
                                    run_id=gen_result.run_id,
                                    num_subjects=mock_cfg["num_subjects"],
                                    synthesizer_type=mock_cfg["synthesizer_type"],
                                )
                                st.session_state["mock_data_cache"][mock_cache_key] = pipeline_result
                                elapsed = time.monotonic() - t0
                                status.update(label=f"Mock Data: Complete ({elapsed:.1f}s)", state="complete")
                                st.rerun()
                            except ImportError:
                                elapsed = time.monotonic() - t0
                                status.update(label=f"Mock Data: Missing dependency ({elapsed:.1f}s)", state="error")
                                st.error("SDV is required. Install with: `pip install sdv`")
                            except Exception:
                                elapsed = time.monotonic() - t0
                                status.update(label=f"Mock Data: Error ({elapsed:.1f}s)", state="error")
                                st.code(traceback.format_exc(), language="text")

        # --- Review sub-tab ---
        with _adv_tabs[review_label]:
            _review_sections: list | None = None
            if _qp_assembled is not None:
                _review_sections = assembled_to_sections(
                    _qp_assembled, registry_id=registry_id,
                )
            if not _review_sections and cached:
                gateway = _get_gateway()
                section_bytes = gateway.get_artifact(cached["artifact_key"])
                _review_sections = parquet_to_sections(section_bytes)
            if _review_sections:
                gateway = _get_gateway()
                protocol_text = ""
                if cached:
                    text_key = cached.get("text_artifact_key")
                    if text_key:
                        text_bytes = gateway.get_artifact(text_key)
                        text_blocks = parquet_to_text_blocks(text_bytes)
                        text_blocks_sorted = sorted(
                            text_blocks,
                            key=lambda b: (b.page_number, b.block_index),
                        )
                        protocol_text = "\n".join(
                            b.text for b in text_blocks_sorted if b.text.strip()
                        )
                render_annotation_review(
                    sections=_review_sections,
                    registry_id=registry_id,
                    gateway=gateway,
                    protocol_text=protocol_text,
                )
                st.divider()
            elif not cached and _qp_assembled is None:
                st.info("Run the **Query Pipeline** first to enable review.")
            render_review_queue(registry_id=registry_id)


if __name__ == "__main__":
    main()
