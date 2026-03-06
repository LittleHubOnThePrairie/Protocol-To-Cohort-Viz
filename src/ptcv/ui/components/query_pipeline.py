"""Query-driven pipeline component for Streamlit UI (PTCV-95).

Orchestrates the full query-driven pipeline:
  extract_protocol_index -> SectionMatcher.match -> QueryExtractor.extract
  -> assemble_template

Pure-Python helpers are separated from Streamlit rendering for testability.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from ptcv.ui.components.confidence_badge import (
    confidence_color,
    confidence_label,
    format_confidence,
)

logger = logging.getLogger(__name__)

# Maximum chars to show in content preview columns.
_CONTENT_PREVIEW_LEN = 120

# Bump when pipeline output schema changes to invalidate stale caches.
_PIPELINE_CACHE_VERSION = 3  # v3: enable_transformation toggle (PTCV-103)


# ---------------------------------------------------------------------------
# Pure-Python helpers (testable, no Streamlit imports)
# ---------------------------------------------------------------------------


def run_query_pipeline(
    pdf_path: str | Path,
    anthropic_api_key: str | None = None,
    enable_summarization: bool = True,
    enable_transformation: bool = False,
) -> dict[str, Any]:
    """Orchestrate the full query-driven pipeline.

    Args:
        pdf_path: Path to the protocol PDF.
        anthropic_api_key: Optional Anthropic API key for LLM-driven
            sub-section scoring. Falls back to keyword-only if absent.
        enable_summarization: Whether to run the sub-section enrichment
            stage (PTCV-96). Defaults to *True*.
        enable_transformation: Whether to apply LLM-driven content
            transformation for high-confidence extractions
            (PTCV-103). Defaults to *False*.

    Returns:
        Dict with keys: ``protocol_index``, ``match_result``,
        ``enriched_match_result``, ``extraction_result``, ``assembled``,
        ``assembled_markdown``, ``coverage``.

    Raises:
        Exception: Re-raised from any pipeline stage.
    """
    from ptcv.ich_parser.toc_extractor import extract_protocol_index
    from ptcv.ich_parser.section_matcher import SectionMatcher
    from ptcv.ich_parser.summarization_matcher import SummarizationMatcher
    from ptcv.ich_parser.query_extractor import QueryExtractor
    from ptcv.ich_parser.template_assembler import (
        QueryExtractionHit,
        SourceReference,
        assemble_template,
    )

    # Stage 1: Extract protocol index
    protocol_index = extract_protocol_index(pdf_path)

    # Stage 2: Section matching (keyword-only, PTCV-102)
    matcher = SectionMatcher()
    match_result = matcher.match(protocol_index)

    # Stage 2.5: Sub-section enrichment (PTCV-96)
    enriched_result = None
    if enable_summarization:
        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        summarizer = SummarizationMatcher(anthropic_api_key=api_key)
        enriched_result = summarizer.refine(match_result, protocol_index)

    # Stage 3: Query extraction (PTCV-103: optional LLM transform)
    api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    extractor = QueryExtractor(
        enable_transformation=enable_transformation,
        anthropic_api_key=api_key,
    )
    extraction_result = extractor.extract(protocol_index, match_result)

    # Stage 4: Template assembly
    hits = []
    for ext in extraction_result.extractions:
        parent = ext.section_id.rsplit(".", 1)[0] if "." in ext.section_id else ext.section_id
        hits.append(QueryExtractionHit(
            query_id=ext.query_id,
            section_id=ext.section_id,
            parent_section=parent,
            query_text="",
            extracted_content=ext.content,
            confidence=ext.confidence,
            source=SourceReference(
                section_header=ext.source_section,
            ),
        ))

    assembled = assemble_template(hits)

    return {
        "protocol_index": protocol_index,
        "match_result": match_result,
        "enriched_match_result": enriched_result,
        "extraction_result": extraction_result,
        "assembled": assembled,
        "assembled_markdown": assembled.to_markdown(),
        "coverage": assembled.coverage,
    }


def format_toc_tree(toc_entries: list[Any]) -> list[dict[str, Any]]:
    """Flatten TOC entries for tabular display.

    Args:
        toc_entries: List of ``TOCEntry`` dataclass instances.

    Returns:
        List of dicts with keys: ``level``, ``number``, ``title``,
        ``page``.
    """
    rows: list[dict[str, Any]] = []
    for entry in toc_entries:
        indent = "  " * (getattr(entry, "level", 1) - 1)
        rows.append({
            "level": getattr(entry, "level", 1),
            "number": getattr(entry, "number", ""),
            "title": f"{indent}{getattr(entry, 'title', '')}",
            "page": getattr(entry, "page_ref", 0),
        })
    return rows


def format_match_table(match_result: Any) -> list[dict[str, Any]]:
    """Format section mappings for tabular display.

    Args:
        match_result: ``MatchResult`` from ``SectionMatcher.match()``.

    Returns:
        List of dicts with keys: ``protocol_section``, ``ich_section``,
        ``score``, ``confidence``, ``method``.
    """
    rows: list[dict[str, Any]] = []
    for mapping in getattr(match_result, "mappings", []):
        matches = getattr(mapping, "matches", [])
        if matches:
            best = matches[0]
            rows.append({
                "protocol_section": (
                    f"{getattr(mapping, 'protocol_section_number', '')} "
                    f"{getattr(mapping, 'protocol_section_title', '')}"
                ).strip(),
                "ich_section": (
                    f"{getattr(best, 'ich_section_code', '')} "
                    f"{getattr(best, 'ich_section_name', '')}"
                ).strip(),
                "score": round(
                    getattr(best, "boosted_score", 0.0)
                    or getattr(best, "similarity_score", 0.0),
                    3,
                ),
                "confidence": getattr(
                    getattr(best, "confidence", None), "value", "LOW"
                ) if hasattr(getattr(best, "confidence", None), "value") else str(
                    getattr(best, "confidence", "LOW")
                ),
                "method": getattr(best, "match_method", ""),
            })
        else:
            rows.append({
                "protocol_section": (
                    f"{getattr(mapping, 'protocol_section_number', '')} "
                    f"{getattr(mapping, 'protocol_section_title', '')}"
                ).strip(),
                "ich_section": "(unmapped)",
                "score": 0.0,
                "confidence": "LOW",
                "method": "",
            })
    return rows


def format_subsection_match_table(
    enriched_result: Any,
) -> list[dict[str, Any]]:
    """Format sub-section matches for tabular display (PTCV-96).

    Args:
        enriched_result: ``EnrichedMatchResult`` from
            ``SummarizationMatcher.refine()``, or *None*.

    Returns:
        List of dicts with keys: ``protocol_section``,
        ``parent_section``, ``sub_section``, ``composite_score``,
        ``summarization_score``, ``confidence``, ``method``.
    """
    if enriched_result is None:
        return []

    rows: list[dict[str, Any]] = []
    for em in getattr(enriched_result, "enriched_mappings", []):
        proto = (
            f"{getattr(em, 'protocol_section_number', '')} "
            f"{getattr(em, 'protocol_section_title', '')}"
        ).strip()
        for sub in getattr(em, "sub_section_matches", []):
            conf = getattr(
                getattr(sub, "confidence", None), "value", "low"
            ) if hasattr(getattr(sub, "confidence", None), "value") else str(
                getattr(sub, "confidence", "low")
            )
            rows.append({
                "protocol_section": proto,
                "parent_section": getattr(
                    sub, "parent_section_code", ""
                ),
                "sub_section": (
                    f"{getattr(sub, 'sub_section_code', '')} "
                    f"{getattr(sub, 'sub_section_name', '')}"
                ).strip(),
                "composite_score": round(
                    getattr(sub, "composite_score", 0.0), 3
                ),
                "summarization_score": round(
                    getattr(sub, "summarization_score", -1.0), 3
                ),
                "confidence": conf,
                "method": getattr(sub, "match_method", ""),
            })
    return rows


def format_extraction_table(
    extraction_result: Any,
) -> list[dict[str, Any]]:
    """Format query extractions for tabular display.

    Args:
        extraction_result: ``ExtractionResult`` from
            ``QueryExtractor.extract()``.

    Returns:
        List of dicts with keys: ``query_id``, ``section``,
        ``content_preview``, ``confidence``, ``method``.
    """
    rows: list[dict[str, Any]] = []
    for ext in getattr(extraction_result, "extractions", []):
        content = getattr(ext, "content", "")
        preview = (
            content[:_CONTENT_PREVIEW_LEN] + "..."
            if len(content) > _CONTENT_PREVIEW_LEN
            else content
        )
        rows.append({
            "query_id": getattr(ext, "query_id", ""),
            "section": getattr(ext, "section_id", ""),
            "content_preview": preview,
            "confidence": format_confidence(
                getattr(ext, "confidence", 0.0)
            ),
            "method": getattr(ext, "extraction_method", ""),
        })
    return rows


def format_gap_table(extraction_result: Any) -> list[dict[str, Any]]:
    """Format extraction gaps for tabular display.

    Args:
        extraction_result: ``ExtractionResult`` from
            ``QueryExtractor.extract()``.

    Returns:
        List of dicts with keys: ``query_id``, ``section``, ``reason``.
    """
    rows: list[dict[str, Any]] = []
    for gap in getattr(extraction_result, "gaps", []):
        rows.append({
            "query_id": getattr(gap, "query_id", ""),
            "section": getattr(gap, "section_id", ""),
            "reason": getattr(gap, "reason", "unknown"),
        })
    return rows


def count_extraction_methods(
    extraction_result: Any,
) -> dict[str, int]:
    """Count extractions by method type (PTCV-103).

    Args:
        extraction_result: ``ExtractionResult`` from
            ``QueryExtractor.extract()``.

    Returns:
        Dict mapping method name to count, e.g.
        ``{"text_short": 5, "llm_transform": 3}``.
    """
    counts: dict[str, int] = {}
    for ext in getattr(extraction_result, "extractions", []):
        method = getattr(ext, "extraction_method", "unknown")
        counts[method] = counts.get(method, 0) + 1
    return counts


def format_coverage_metrics(coverage: Any) -> dict[str, Any]:
    """Extract key coverage metrics from a CoverageReport.

    Args:
        coverage: ``CoverageReport`` from ``AssembledProtocol.coverage``.

    Returns:
        Dict with keys: ``total``, ``populated``, ``gaps``,
        ``avg_confidence``, ``high_pct``, ``review_pct``, ``low_pct``.
    """
    total = getattr(coverage, "total_sections", 0)
    populated = getattr(coverage, "populated_count", 0)
    gaps = getattr(coverage, "gap_count", 0)
    avg_conf = getattr(coverage, "avg_confidence", 0.0)
    high = getattr(coverage, "high_confidence_count", 0)
    review = getattr(coverage, "review_confidence_count", 0)
    low = getattr(coverage, "low_confidence_count", 0)

    return {
        "total": total,
        "populated": populated,
        "gaps": gaps,
        "avg_confidence": round(avg_conf, 3),
        "high_pct": round(high / total * 100, 1) if total else 0.0,
        "review_pct": round(review / total * 100, 1) if total else 0.0,
        "low_pct": round(low / total * 100, 1) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def render_query_pipeline(file_path: Path, file_sha: str) -> None:
    """Render the Query Pipeline tab.

    Runs the full query-driven pipeline on the selected PDF and
    displays: Protocol Index, Section Matching, Extraction Results,
    and Assembled Template.

    Args:
        file_path: Absolute path to the selected PDF.
        file_sha: SHA-256 hex digest for session caching.
    """
    import streamlit as st

    # --- Options (PTCV-103, PTCV-111) ---
    _has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    transform_on = st.checkbox(
        "Enable LLM Transformation",
        value=_has_api_key,
        help=(
            "Reframe scoped extractions using Claude Sonnet to "
            "directly address each Appendix B query. "
            "Requires ANTHROPIC_API_KEY."
        ),
        key="chk_enable_transform",
    )

    if transform_on and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning(
            "LLM Transformation requires an Anthropic API key. "
            "Set **ANTHROPIC_API_KEY** in your environment or "
            "`.secrets` file. Pipeline will run in verbatim mode."
        )

    # Cache init — version-stamped key invalidates stale entries.
    # Include transform toggle in cache key so toggling re-runs.
    cache_key = (
        f"{file_sha}_v{_PIPELINE_CACHE_VERSION}"
        f"_t{int(transform_on)}"
    )
    if "query_cache" not in st.session_state:
        st.session_state["query_cache"] = {}

    cached = st.session_state["query_cache"].get(cache_key)

    if cached:
        st.success("Query pipeline complete. Results below.")
    else:
        if st.button(
            "Run Query Pipeline",
            type="primary",
            key="btn_query_pipeline",
        ):
            with st.status(
                "Running query-driven pipeline...",
                expanded=True,
            ) as status:
                t0 = time.monotonic()
                try:
                    st.write("Extracting protocol index...")
                    result = run_query_pipeline(
                        str(file_path),
                        enable_transformation=transform_on,
                        anthropic_api_key=os.environ.get(
                            "ANTHROPIC_API_KEY"
                        ),
                    )
                    elapsed = time.monotonic() - t0
                    st.session_state["query_cache"][cache_key] = result
                    cached = result
                    status.update(
                        label=(
                            "Query Pipeline: "
                            f"Complete ({elapsed:.1f}s)"
                        ),
                        state="complete",
                    )
                except Exception:
                    elapsed = time.monotonic() - t0
                    status.update(
                        label=(
                            "Query Pipeline: "
                            f"Error ({elapsed:.1f}s)"
                        ),
                        state="error",
                    )
                    st.code(traceback.format_exc(), language="text")
                    return

    if not cached:
        st.info(
            "Click **Run Query Pipeline** to extract and classify "
            "this protocol using the query-driven approach."
        )
        return

    # --- Protocol Index ---
    protocol_index = cached["protocol_index"]
    st.subheader("Protocol Index")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages", getattr(protocol_index, "page_count", 0))
    with col2:
        st.metric(
            "TOC entries",
            len(getattr(protocol_index, "toc_entries", [])),
        )
    with col3:
        st.metric(
            "Section headers",
            len(getattr(protocol_index, "section_headers", [])),
        )

    toc_rows = format_toc_tree(
        getattr(protocol_index, "toc_entries", [])
    )
    if toc_rows:
        with st.expander("Table of Contents", expanded=False):
            st.dataframe(toc_rows, use_container_width=True)

    st.divider()

    # --- Section Matching ---
    match_result = cached["match_result"]
    st.subheader("Section Matching")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Auto-mapped",
            getattr(match_result, "auto_mapped_count", 0),
        )
    with col2:
        st.metric("Review", getattr(match_result, "review_count", 0))
    with col3:
        st.metric(
            "Unmapped", getattr(match_result, "unmapped_count", 0),
        )

    match_rows = format_match_table(match_result)
    if match_rows:
        with st.expander("Section Mappings", expanded=False):
            st.dataframe(match_rows, use_container_width=True)

    st.divider()

    # --- Sub-Section Matching (PTCV-96) ---
    enriched_result = cached.get("enriched_match_result")
    if enriched_result is not None:
        st.subheader("Sub-Section Matching")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Sub-section auto-map",
                f"{getattr(enriched_result, 'sub_section_auto_map_rate', 0.0):.0%}",
            )
        with col2:
            st.metric(
                "LLM calls",
                getattr(enriched_result, "llm_calls_made", 0),
            )
        with col3:
            mode = (
                "Fallback" if getattr(enriched_result, "llm_fallback", True)
                else "Active"
            )
            st.metric("LLM mode", mode)

        sub_rows = format_subsection_match_table(enriched_result)
        if sub_rows:
            with st.expander(
                "Sub-Section Mappings", expanded=False
            ):
                st.dataframe(sub_rows, use_container_width=True)

        st.divider()

    # --- Extraction Results ---
    extraction_result = cached["extraction_result"]
    st.subheader("Query Extraction")
    method_counts = count_extraction_methods(extraction_result)
    transformed_count = method_counts.get("llm_transform", 0)
    verbatim_count = sum(
        v for k, v in method_counts.items() if k != "llm_transform"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Answered",
            getattr(extraction_result, "answered_queries", 0),
        )
    with col2:
        st.metric(
            "Total queries",
            getattr(extraction_result, "total_queries", 0),
        )
    with col3:
        coverage = getattr(extraction_result, "coverage", 0.0)
        st.metric("Coverage", f"{coverage:.0%}")

    # Transformation metrics (PTCV-103)
    if transformed_count > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LLM Transformed", transformed_count)
        with col2:
            st.metric("Verbatim", verbatim_count)

    ext_rows = format_extraction_table(extraction_result)
    if ext_rows:
        with st.expander("Extractions", expanded=False):
            st.dataframe(ext_rows, use_container_width=True)

    gap_rows = format_gap_table(extraction_result)
    if gap_rows:
        with st.expander(f"Gaps ({len(gap_rows)})", expanded=False):
            st.dataframe(gap_rows, use_container_width=True)

    st.divider()

    # --- Assembled Template ---
    assembled = cached["assembled"]
    st.subheader("Assembled Appendix B Template")

    cov = cached["coverage"]
    cov_metrics = format_coverage_metrics(cov)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sections", cov_metrics["total"])
    with col2:
        st.metric("Populated", cov_metrics["populated"])
    with col3:
        st.metric("Gaps", cov_metrics["gaps"])
    with col4:
        st.metric(
            "Avg confidence",
            f"{cov_metrics['avg_confidence']:.2f}",
        )

    md = cached["assembled_markdown"]
    with st.expander("Template Preview", expanded=True):
        with st.container(height=500):
            st.markdown(md)

    st.download_button(
        label="Download as .md",
        data=md,
        file_name="assembled_appendix_b.md",
        mime="text/markdown",
        key="btn_dl_assembled",
    )
