"""PTCV Streamlit application — main entry point (PTCV-32, PTCV-33, PTCV-36).

Launch with::

    streamlit run src/ptcv/ui/app.py

Provides:
- Sidebar file browser listing protocol PDFs by registry source
- "Parse PDF" action that runs extraction + ICH classification
- ICH compliance verdict badge with confidence and section details
- "Create mock SDTM file" action with lineage visualization
- Session-level result caching keyed by file SHA-256
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path for non-installed package
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st

from ptcv.extraction import ExtractionService, parquet_to_tables, parquet_to_text_blocks
from ptcv.ich_parser import IchParser
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
from ptcv.ui.components.schedule_of_visits import render_schedule_of_visits
from ptcv.ui.components.annotation_review import render_annotation_review
from ptcv.ui.components.sdtm_viewer import render_sdtm_viewer

logger = logging.getLogger(__name__)

# Default paths relative to PTCV project root
_DATA_ROOT = Path("C:/Dev/PTCV/data")
_PROTOCOLS_DIR = _DATA_ROOT / "protocols"


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
) -> dict:
    """Run extraction + ICH parse and return verdict dict.

    Chains ExtractionService.extract() → read text_blocks →
    IchParser.parse() and returns a dict with all verdict fields
    needed for display.

    Args:
        pdf_bytes: Raw PDF file bytes.
        filename: Original PDF filename.
        gateway: Initialised FilesystemAdapter.

    Returns:
        Dict with keys: format_verdict, format_confidence,
        section_count, review_count, missing_required_sections,
        registry_id.
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
    protocol_text = "\n".join(
        b.text for b in text_blocks_sorted if b.text.strip()
    )

    if not protocol_text.strip():
        protocol_text = (
            f"Protocol {registry_id} "
            "(text extraction produced no content)"
        )

    # Stage 2: ICH Parse
    parser = IchParser(gateway=gateway)
    parse_result = parser.parse(
        text=protocol_text,
        registry_id=registry_id,
        source_run_id=extraction_result.run_id,
        source_sha256=extraction_result.text_artifact_sha256,
    )

    return {
        "format_verdict": parse_result.format_verdict,
        "format_confidence": parse_result.format_confidence,
        "section_count": parse_result.section_count,
        "review_count": parse_result.review_count,
        "missing_required_sections": parse_result.missing_required_sections,
        "registry_id": registry_id,
        "amendment": amendment,
        "artifact_key": parse_result.artifact_key,
        "text_artifact_key": extraction_result.text_artifact_key,
        "tables_artifact_key": extraction_result.tables_artifact_key,
    }


def _run_soa_extraction(
    cached: dict,
    gateway: FilesystemAdapter,
) -> dict:
    """Run SoA extraction on already-parsed ICH sections.

    Reads classified sections from storage, runs SoaExtractor, then
    reads back the USDM Parquet artifacts and returns them as lists
    suitable for the schedule of visits component.

    Args:
        cached: Parse result dict from _run_parse() (needs artifact_key).
        gateway: Initialised FilesystemAdapter.

    Returns:
        Dict with keys: timepoints, activities, instances,
        registry_id, format_verdict, soa_artifact_keys.
    """
    section_bytes = gateway.get_artifact(cached["artifact_key"])
    sections = parquet_to_sections(section_bytes)

    # Load pre-extracted tables for SoA bridge (PTCV-51)
    pre_extracted_tables = None
    tables_key = cached.get("tables_artifact_key")
    if tables_key:
        try:
            tables_bytes = gateway.get_artifact(tables_key)
            pre_extracted_tables = parquet_to_tables(tables_bytes)
        except Exception:
            pass  # Fall back to text-based parsing

    extractor = SoaExtractor(gateway=gateway)
    soa_result = extractor.extract(
        sections=sections,
        registry_id=cached["registry_id"],
        source_run_id="",
        source_sha256="",
        extracted_tables=pre_extracted_tables,
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
    }


def _run_sdtm_generation(
    cached: dict,
    soa_cached: dict | None,
    gateway: FilesystemAdapter,
) -> dict:
    """Run SDTM generation from parsed ICH sections and SoA data.

    Reads classified sections from storage, collects timepoints from
    SoA cache (if available), and runs SdtmService.generate().

    Args:
        cached: Parse result dict from _run_parse().
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


def _display_verdict(result: dict) -> None:
    """Render ICH compliance verdict in the main panel.

    Args:
        result: Dict from _run_parse().
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
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Format confidence", f"{confidence:.2f}")
    with col2:
        st.metric("Sections detected", f"{section_count} / 11")

    if missing:
        st.caption(f"Missing required sections: {', '.join(missing)}")

    if result["review_count"] > 0:
        st.caption(
            f"{result['review_count']} section(s) flagged for human review"
        )


def _render_regeneration(cached: dict) -> None:
    """Render the ICH E6(R3) regenerated protocol view (PTCV-35).

    Retrieves classified sections from storage, builds an ICH-ordered
    markdown document, displays it in a scrollable container, and
    provides a download button.

    Args:
        cached: Parse result dict with artifact_key, registry_id, etc.
    """
    gateway = _get_gateway()
    section_bytes = gateway.get_artifact(cached["artifact_key"])
    sections = parquet_to_sections(section_bytes)

    md = regenerate_ich_markdown(
        sections=sections,
        registry_id=cached["registry_id"],
        format_verdict=cached["format_verdict"],
        format_confidence=cached["format_confidence"],
    )

    st.subheader("ICH E6(R3) Reformatted Protocol")

    # Scrollable container with fixed height
    with st.container(height=500):
        st.markdown(md)

    # Download button
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

    # Initialise caches
    if "parse_cache" not in st.session_state:
        st.session_state["parse_cache"] = {}
    if "soa_cache" not in st.session_state:
        st.session_state["soa_cache"] = {}
    if "sdtm_cache" not in st.session_state:
        st.session_state["sdtm_cache"] = {}

    # Actions
    action = st.selectbox(
        "Action",
        [
            "Parse PDF",
            "Regenerate protocol in ICH format",
            "Generate schedule of visits",
            "Create mock SDTM file",
            "Review annotations",
        ],
        label_visibility="collapsed",
    )

    cached = st.session_state.get("parse_cache", {}).get(file_sha)
    soa_cached = st.session_state.get("soa_cache", {}).get(file_sha)
    sdtm_cached = st.session_state.get("sdtm_cache", {}).get(file_sha)

    if st.button("Run"):
        if action == "Parse PDF":
            gateway = _get_gateway()
            with st.spinner("Parsing PDF..."):
                result = _run_parse(pdf_bytes, file_path.name, gateway)
            st.session_state["parse_cache"][file_sha] = result
            cached = result
        elif action == "Regenerate protocol in ICH format":
            if cached is None:
                st.warning(
                    "Parse the PDF first before regenerating "
                    "in ICH format."
                )
        elif action == "Generate schedule of visits":
            gateway = _get_gateway()
            if cached is None:
                with st.spinner("Parsing PDF..."):
                    result = _run_parse(pdf_bytes, file_path.name, gateway)
                st.session_state["parse_cache"][file_sha] = result
                cached = result
            with st.spinner("Extracting schedule of visits..."):
                soa_result = _run_soa_extraction(cached, gateway)
            st.session_state["soa_cache"][file_sha] = soa_result
            soa_cached = soa_result
        elif action == "Create mock SDTM file":
            gateway = _get_gateway()
            # Auto-parse if needed
            if cached is None:
                with st.spinner("Parsing PDF..."):
                    result = _run_parse(pdf_bytes, file_path.name, gateway)
                st.session_state["parse_cache"][file_sha] = result
                cached = result
            # Auto-run SoA extraction if needed
            if soa_cached is None:
                with st.spinner("Extracting schedule of visits..."):
                    soa_result = _run_soa_extraction(cached, gateway)
                st.session_state["soa_cache"][file_sha] = soa_result
                soa_cached = soa_result
            # Run SDTM generation
            with st.spinner("Generating mock SDTM domains..."):
                sdtm_result = _run_sdtm_generation(
                    cached, soa_cached, gateway
                )
            st.session_state["sdtm_cache"][file_sha] = sdtm_result
            sdtm_cached = sdtm_result
        elif action == "Review annotations":
            gateway = _get_gateway()
            if cached is None:
                with st.spinner("Parsing PDF..."):
                    result = _run_parse(pdf_bytes, file_path.name, gateway)
                st.session_state["parse_cache"][file_sha] = result
                cached = result

    # Display results
    if cached:
        _display_verdict(cached)

        if action == "Regenerate protocol in ICH format":
            st.divider()
            _render_regeneration(cached)

        if action == "Generate schedule of visits" and soa_cached:
            st.divider()
            render_schedule_of_visits(
                timepoints=soa_cached["timepoints"],
                activities=soa_cached["activities"],
                instances=soa_cached["instances"],
                registry_id=soa_cached["registry_id"],
                format_verdict=soa_cached["format_verdict"],
            )

        if action == "Create mock SDTM file" and sdtm_cached:
            st.divider()
            render_sdtm_viewer(
                sdtm_result=sdtm_cached["sdtm_result"],
                sections=sdtm_cached["sections"],
                has_soa=sdtm_cached["has_soa"],
                gateway=_get_gateway(),
                registry_id=sdtm_cached["registry_id"],
                format_verdict=sdtm_cached["format_verdict"],
            )

        if action == "Review annotations":
            st.divider()
            gateway = _get_gateway()
            section_bytes = gateway.get_artifact(cached["artifact_key"])
            review_sections = parquet_to_sections(section_bytes)
            # Reconstruct full protocol text for PTCV-43
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
    elif cached is None:
        st.caption(
            "Select **Parse PDF** and click **Run** to start."
        )


if __name__ == "__main__":
    main()
