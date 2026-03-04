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

from ptcv.extraction import ExtractionService, parquet_to_tables, parquet_to_text_blocks
from ptcv.ich_parser import IchParser, RAGClassifier
from ptcv.ich_parser.coverage_reviewer import CoverageReviewer
from ptcv.ich_parser.fidelity_checker import FidelityChecker
from ptcv.ich_parser.llm_retemplater import LlmRetemplater
from ptcv.ich_parser.parquet_writer import parquet_to_sections
from ptcv.sdtm import SdtmService
from ptcv.soa_extractor import SoaExtractor
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
from ptcv.ui.components.sdtm_viewer import render_sdtm_viewer
from ptcv.ui.components.section_align import align_sections, build_comparison_html
from ptcv.ui.pipeline_stages import (
    PIPELINE_STAGES,
    STAGE_BY_KEY,
    compute_active_stages,
    get_all_dependents,
    get_all_prerequisites,
    get_execution_order,
)

logger = logging.getLogger(__name__)

# Default paths relative to PTCV project root
_DATA_ROOT = Path("C:/Dev/PTCV/data")
_PROTOCOLS_DIR = _DATA_ROOT / "protocols"

# LLM retemplating available when API key is set
_HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Model tier definitions (PTCV-78)
_MODEL_TIERS: dict[str, dict[str, str | float]] = {
    "Best Quality (Opus)": {
        "model_id": "claude-opus-4-6",
        "method": "llm",
        "cost_per_page_low": 0.015,
        "cost_per_page_high": 0.020,
    },
    "Balanced (Sonnet)": {
        "model_id": "claude-sonnet-4-6",
        "method": "rag",
        "cost_per_page_low": 0.0008,
        "cost_per_page_high": 0.002,
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
    if _HAS_ANTHROPIC_KEY and tier_cfg["method"] == "llm":
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
    elif _HAS_ANTHROPIC_KEY and tier_cfg["method"] == "rag":
        # Sonnet tier: RAGClassifier (Cohere + Sonnet) via IchParser
        protocol_text = "\n".join(str(d["text"]) for d in text_block_dicts)
        classifier = RAGClassifier(claude_model=str(tier_cfg["model_id"]))
        parser = IchParser(gateway=gateway, classifier=classifier)
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
        retemplating_method = "rag"
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
    elif method == "rag":
        st.caption(
            f"RAG retemplating [{tier}]: "
            "Cohere embeddings + Claude Sonnet"
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
                    old_code=original,
                    new_code=md,
                    old_title=left_label,
                    new_title=right_label,
                    split_view=(mode == "Side by side"),
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


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="PTCV — Protocol Viewer",
        page_icon="🧬",
        layout="wide",
    )
    st.title("Protocol-To-Cohort-Viz")

    # Sidebar: file browser
    source, file_path = render_file_browser(_PROTOCOLS_DIR)

    if file_path is None:
        st.info("Select a protocol file from the sidebar to get started.")
        return

    st.markdown(f"**Selected:** `{source}/{file_path.name}`")

    # Read PDF and compute SHA for caching
    pdf_bytes = file_path.read_bytes()
    file_sha = _compute_sha256(pdf_bytes)
    page_count = _count_pdf_pages(pdf_bytes)

    # Initialise caches
    if "parse_cache" not in st.session_state:
        st.session_state["parse_cache"] = {}
    if "soa_cache" not in st.session_state:
        st.session_state["soa_cache"] = {}
    if "sdtm_cache" not in st.session_state:
        st.session_state["sdtm_cache"] = {}
    if "fidelity_cache" not in st.session_state:
        st.session_state["fidelity_cache"] = {}

    # ------------------------------------------------------------------
    # Checkpoint resume (PTCV-83)
    # ------------------------------------------------------------------

    if has_checkpoints(_DATA_ROOT, file_sha):
        resume_label = get_resume_label(_DATA_ROOT, file_sha)
        col_resume, col_clear = st.columns([3, 1])
        with col_resume:
            if st.button(
                resume_label or "Resume pipeline",
                key="btn_resume",
            ):
                restored = load_checkpoints(_DATA_ROOT, file_sha)
                for cache_key, data in restored.items():
                    st.session_state[cache_key][file_sha] = data
                st.toast(
                    f"Restored {len(restored)} stage(s) "
                    "from checkpoint"
                )
                st.rerun()
        with col_clear:
            if st.button(
                "Start fresh",
                key="btn_clear_cp",
            ):
                clear_checkpoints(_DATA_ROOT, file_sha)
                for ck in (
                    "parse_cache", "soa_cache",
                    "fidelity_cache", "sdtm_cache",
                ):
                    st.session_state[ck].pop(file_sha, None)
                st.toast("Checkpoints cleared")
                st.rerun()

    # ------------------------------------------------------------------
    # Pipeline stage checkboxes (PTCV-77)
    # ------------------------------------------------------------------

    if "user_stages" not in st.session_state:
        st.session_state["user_stages"] = set()

    def _on_stage_toggle(stage_key: str) -> None:
        """Handle checkbox toggle with auto-enable/disable logic."""
        new_val = st.session_state[f"cb_{stage_key}"]
        user_stages: set[str] = st.session_state["user_stages"]

        if new_val:
            user_stages.add(stage_key)
            # Auto-enable all transitive prerequisites
            for prereq in get_all_prerequisites(stage_key):
                user_stages.add(prereq)
                st.session_state[f"cb_{prereq}"] = True
        else:
            user_stages.discard(stage_key)
            # Cascade disable all transitive dependents
            for dep in get_all_dependents(stage_key):
                user_stages.discard(dep)
                st.session_state[f"cb_{dep}"] = False

    def _is_cached(stage_key: str) -> bool:
        """Check if a stage's result is already cached for this file."""
        cache_key = STAGE_BY_KEY[stage_key].cache_key
        if not cache_key:
            return False
        return file_sha in st.session_state.get(cache_key, {})

    # Sidebar: model cost tier (PTCV-78)
    with st.sidebar:
        st.subheader("Model Tier")
        tier_options = list(_MODEL_TIERS.keys())
        selected_tier = st.radio(
            "Retemplating model",
            tier_options,
            index=tier_options.index(_DEFAULT_TIER),
            key="model_tier",
            label_visibility="collapsed",
        )

        if page_count > 0 and _HAS_ANTHROPIC_KEY:
            low, high = estimate_cost(page_count, selected_tier)
            st.caption(
                f"Est. cost ({page_count} pages): "
                f"${low:.2f}\u2013${high:.2f}"
            )
        elif not _HAS_ANTHROPIC_KEY:
            st.caption("No API key — deterministic fallback")

    # Sidebar: pipeline stage checkboxes
    with st.sidebar:
        st.subheader("Pipeline Stages")

        user_stages: set[str] = st.session_state["user_stages"]
        active = compute_active_stages(user_stages)

        for stage in PIPELINE_STAGES:
            is_auto = stage.key in active and stage.key not in user_stages
            is_cached = _is_cached(stage.key)

            label = stage.label
            if is_auto:
                label += " *(auto)*"
            if is_cached:
                label += " [cached]"

            st.checkbox(
                label,
                key=f"cb_{stage.key}",
                on_change=_on_stage_toggle,
                args=(stage.key,),
            )

    active = compute_active_stages(st.session_state["user_stages"])

    # ------------------------------------------------------------------
    # Pipeline execution (PTCV-77)
    # ------------------------------------------------------------------

    cached = st.session_state.get("parse_cache", {}).get(file_sha)
    soa_cached = st.session_state.get("soa_cache", {}).get(file_sha)
    fidelity_cached = st.session_state.get(
        "fidelity_cache", {},
    ).get(file_sha)
    sdtm_cached = st.session_state.get("sdtm_cache", {}).get(file_sha)

    if st.button("Run Pipeline", disabled=not active):
        gateway = _get_gateway()
        execution_order = get_execution_order(active)
        pipeline_error = False

        for stage_key in execution_order:
            stage = STAGE_BY_KEY[stage_key]

            # Skip display-only stages (no execution needed)
            if not stage.cache_key:
                continue

            # Stop pipeline on prior error (PTCV-82)
            if pipeline_error:
                st.status(
                    f"{stage.label}: Skipped (prior error)",
                    state="error",
                )
                continue

            # Cached stages: show status instead of toast (PTCV-82)
            if _is_cached(stage_key):
                st.status(
                    f"{stage.label}: Cached",
                    state="complete",
                )
                continue

            # Execute stage with timing and error handling
            if stage_key in ("extraction", "retemplating"):
                # Both resolve via _run_parse()
                if cached is None:
                    with st.status(
                        f"Running {stage.label}...",
                        expanded=True,
                    ) as status:
                        t0 = time.monotonic()
                        tier = st.session_state.get(
                            "model_tier", _DEFAULT_TIER,
                        )
                        st.write(
                            f"Extracting and retemplating "
                            f"{page_count} pages "
                            f"({tier})..."
                        )
                        try:
                            result = _run_parse(
                                pdf_bytes, file_path.name,
                                gateway, model_tier=tier,
                            )
                            elapsed = time.monotonic() - t0
                            st.session_state["parse_cache"][
                                file_sha
                            ] = result
                            cached = result
                            save_checkpoint(
                                _DATA_ROOT, file_sha,
                                "parse_cache", result,
                            )
                            status.update(
                                label=(
                                    f"{stage.label}: "
                                    f"Complete ({elapsed:.1f}s)"
                                ),
                                state="complete",
                            )
                        except Exception:
                            elapsed = time.monotonic() - t0
                            status.update(
                                label=(
                                    f"{stage.label}: "
                                    f"Error ({elapsed:.1f}s)"
                                ),
                                state="error",
                            )
                            st.code(
                                traceback.format_exc(),
                                language="text",
                            )
                            pipeline_error = True
                else:
                    st.status(
                        f"{stage.label}: Cached",
                        state="complete",
                    )

            elif stage_key == "soa":
                with st.status(
                    "Running SoA Extraction...", expanded=True,
                ) as status:
                    t0 = time.monotonic()
                    st.write("Discovering tables and timepoints...")
                    try:
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
                        pipeline_error = True

            elif stage_key == "fidelity":
                with st.status(
                    "Running Fidelity Check...", expanded=True,
                ) as status:
                    t0 = time.monotonic()
                    st.write("Comparing retemplated vs original...")
                    try:
                        fidelity_result = _run_fidelity_check(
                            cached, gateway,
                            enable_llm=_HAS_ANTHROPIC_KEY,
                        )
                        elapsed = time.monotonic() - t0
                        st.session_state["fidelity_cache"][
                            file_sha
                        ] = fidelity_result
                        fidelity_cached = fidelity_result
                        save_checkpoint(
                            _DATA_ROOT, file_sha,
                            "fidelity_cache", fidelity_result,
                        )
                        status.update(
                            label=(
                                "Fidelity Check: "
                                f"Complete ({elapsed:.1f}s)"
                            ),
                            state="complete",
                        )
                    except Exception:
                        elapsed = time.monotonic() - t0
                        status.update(
                            label=(
                                "Fidelity Check: "
                                f"Error ({elapsed:.1f}s)"
                            ),
                            state="error",
                        )
                        st.code(
                            traceback.format_exc(),
                            language="text",
                        )
                        pipeline_error = True

            elif stage_key == "sdtm":
                with st.status(
                    "Running SDTM Generation...", expanded=True,
                ) as status:
                    t0 = time.monotonic()
                    st.write("Generating SDTM domains...")
                    try:
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
                        pipeline_error = True

    # ------------------------------------------------------------------
    # Display results for all active stages (PTCV-77)
    # ------------------------------------------------------------------

    # Extraction: ICH compliance verdict
    if "extraction" in active and cached:
        _display_verdict(cached)

    # ICH Retemplating: regenerated protocol view
    if "retemplating" in active and cached:
        st.divider()
        _render_regeneration(cached)

    # Fidelity Check
    if "fidelity" in active and fidelity_cached:
        st.divider()
        _display_fidelity(fidelity_cached)

    # Schedule of Visits (PTCV-76: plot-only view)
    if "sov" in active and soa_cached:
        st.divider()
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

    # SDTM Generation
    if "sdtm" in active and sdtm_cached:
        st.divider()
        render_sdtm_viewer(
            sdtm_result=sdtm_cached["sdtm_result"],
            sections=sdtm_cached["sections"],
            has_soa=sdtm_cached["has_soa"],
            gateway=_get_gateway(),
            registry_id=sdtm_cached["registry_id"],
            format_verdict=sdtm_cached["format_verdict"],
        )

    # Annotation Review
    if "annotations" in active and cached:
        st.divider()
        gateway = _get_gateway()
        section_bytes = gateway.get_artifact(cached["artifact_key"])
        review_sections = parquet_to_sections(section_bytes)
        protocol_text = ""
        text_key = cached.get("text_artifact_key")
        if text_key:
            text_bytes = gateway.get_artifact(text_key)
            text_blocks = parquet_to_text_blocks(text_bytes)
            text_blocks_sorted = sorted(
                text_blocks,
                key=lambda b: (b.page_number, b.block_index),
            )
            protocol_text = "\n".join(
                b.text
                for b in text_blocks_sorted
                if b.text.strip()
            )
        render_annotation_review(
            sections=review_sections,
            registry_id=cached["registry_id"],
            gateway=gateway,
            protocol_text=protocol_text,
        )

    # Empty state
    if not active:
        st.caption(
            "Select pipeline stages from the sidebar and "
            "click **Run Pipeline** to start."
        )


if __name__ == "__main__":
    main()
