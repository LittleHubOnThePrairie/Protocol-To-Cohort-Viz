"""Benchmark comparison component for Streamlit UI (PTCV-95).

Runs text-first pipeline on a single protocol and compares metrics
against cached query-driven results from the Query Pipeline tab.

Pure-Python helpers are separated from Streamlit rendering for testability.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from ptcv.ui.components.confidence_badge import format_confidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python helpers (testable, no Streamlit imports)
# ---------------------------------------------------------------------------


def run_single_benchmark(
    full_text: str,
    nct_id: str,
    query_cached: dict[str, Any],
) -> dict[str, Any]:
    """Run text-first pipeline and compare with query-driven results.

    Args:
        full_text: Full protocol text for text-first classification.
        nct_id: Protocol registry ID.
        query_cached: Cached result dict from ``run_query_pipeline()``.

    Returns:
        Dict with keys: ``text_first``, ``query_driven``,
        ``coverage_delta``, ``coherence_ratio``, ``section_agreement``,
        ``recommendation``, ``recommendation_reason``.
    """
    from ptcv.ich_parser.benchmark import (
        compare_protocol,
        generate_recommendation,
        run_query_driven,
        run_text_first,
    )

    # Run text-first
    tf_metrics = run_text_first(full_text, nct_id)

    # Run query-driven from cached protocol_index
    protocol_index = query_cached["protocol_index"]
    qd_metrics = run_query_driven(protocol_index)

    # Compute deltas
    coverage_delta = qd_metrics.coverage_pct - tf_metrics.coverage_pct
    coherence_ratio = (
        qd_metrics.avg_span_length / tf_metrics.avg_span_length
        if tf_metrics.avg_span_length > 0
        else 0.0
    )

    # Section agreement (Jaccard similarity)
    tf_codes = tf_metrics.section_codes
    qd_codes = qd_metrics.section_codes
    intersection = len(tf_codes & qd_codes)
    union = len(tf_codes | qd_codes)
    section_agreement = intersection / union if union > 0 else 0.0

    # Recommendation
    recommendation, reason = generate_recommendation(
        avg_coverage_delta=coverage_delta,
        avg_coherence_ratio=coherence_ratio,
        qd_high_confidence_pct=qd_metrics.high_confidence_pct,
        qd_gap_rate=qd_metrics.gap_rate,
    )

    return {
        "text_first": tf_metrics,
        "query_driven": qd_metrics,
        "coverage_delta": round(coverage_delta, 2),
        "coherence_ratio": round(coherence_ratio, 2),
        "section_agreement": round(section_agreement, 3),
        "recommendation": recommendation,
        "recommendation_reason": reason,
    }


def format_metrics_comparison(
    text_first: Any,
    query_driven: Any,
) -> list[dict[str, Any]]:
    """Build a side-by-side metrics comparison table.

    Args:
        text_first: ``PipelineMetrics`` from text-first pipeline.
        query_driven: ``PipelineMetrics`` from query-driven pipeline.

    Returns:
        List of dicts with keys: ``metric``, ``text_first``,
        ``query_driven``, ``delta``.
    """
    if text_first is None or query_driven is None:
        return []

    def _get(obj: Any, attr: str, default: float = 0.0) -> float:
        return getattr(obj, attr, default)

    rows = [
        {
            "metric": "Coverage (%)",
            "text_first": f"{_get(text_first, 'coverage_pct'):.1f}",
            "query_driven": f"{_get(query_driven, 'coverage_pct'):.1f}",
            "delta": f"{_get(query_driven, 'coverage_pct') - _get(text_first, 'coverage_pct'):+.1f}",
        },
        {
            "metric": "Avg Confidence",
            "text_first": f"{_get(text_first, 'avg_confidence'):.3f}",
            "query_driven": f"{_get(query_driven, 'avg_confidence'):.3f}",
            "delta": f"{_get(query_driven, 'avg_confidence') - _get(text_first, 'avg_confidence'):+.3f}",
        },
        {
            "metric": "Avg Span Length",
            "text_first": f"{_get(text_first, 'avg_span_length'):.0f}",
            "query_driven": f"{_get(query_driven, 'avg_span_length'):.0f}",
            "delta": f"{_get(query_driven, 'avg_span_length') - _get(text_first, 'avg_span_length'):+.0f}",
        },
        {
            "metric": "Gap Rate",
            "text_first": f"{_get(text_first, 'gap_rate'):.2f}",
            "query_driven": f"{_get(query_driven, 'gap_rate'):.2f}",
            "delta": f"{_get(query_driven, 'gap_rate') - _get(text_first, 'gap_rate'):+.2f}",
        },
        {
            "metric": "Sections Detected",
            "text_first": str(int(_get(text_first, "section_count"))),
            "query_driven": str(int(_get(query_driven, "section_count"))),
            "delta": f"{int(_get(query_driven, 'section_count') - _get(text_first, 'section_count')):+d}",
        },
        {
            "metric": "High Confidence (%)",
            "text_first": f"{_get(text_first, 'high_confidence_pct'):.1f}",
            "query_driven": f"{_get(query_driven, 'high_confidence_pct'):.1f}",
            "delta": f"{_get(query_driven, 'high_confidence_pct') - _get(text_first, 'high_confidence_pct'):+.1f}",
        },
    ]
    return rows


def format_recommendation(
    recommendation: str,
    reason: str,
) -> tuple[str, str]:
    """Return Streamlit badge type and display text for the recommendation.

    Args:
        recommendation: One of ``'replace'``, ``'supplement'``,
            ``'keep_text_first'``.
        reason: Human-readable reason string.

    Returns:
        Tuple of ``(badge_type, display_text)`` where badge_type is
        ``'success'``, ``'warning'``, or ``'info'``.
    """
    if recommendation == "replace":
        return ("success", f"Replace text-first pipeline: {reason}")
    if recommendation == "supplement":
        return ("warning", f"Supplement with query-driven: {reason}")
    return ("info", f"Keep text-first pipeline: {reason}")


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def render_benchmark(
    file_path: Path,
    file_sha: str,
    query_cached: dict[str, Any] | None,
) -> None:
    """Render the Benchmark tab.

    Runs text-first pipeline and compares against cached query-driven
    results.  Shows side-by-side metrics, recommendation, and
    confidence distribution.

    Args:
        file_path: Absolute path to the selected PDF.
        file_sha: SHA-256 hex digest for session caching.
        query_cached: Result from Query Pipeline tab (may be None).
    """
    import streamlit as st

    if not query_cached:
        st.info(
            "Run the **Query Pipeline** first, then return here "
            "to compare pipelines."
        )
        return

    # Cache init
    if "benchmark_cache" not in st.session_state:
        st.session_state["benchmark_cache"] = {}

    cached = st.session_state["benchmark_cache"].get(file_sha)

    if cached:
        st.success("Benchmark complete. Results below.")
    else:
        if st.button(
            "Run Benchmark",
            type="primary",
            key="btn_benchmark",
        ):
            with st.status(
                "Running benchmark comparison...",
                expanded=True,
            ) as status:
                t0 = time.monotonic()
                try:
                    st.write("Running text-first pipeline...")
                    protocol_index = query_cached["protocol_index"]
                    full_text = getattr(
                        protocol_index, "full_text", ""
                    )
                    nct_id = str(
                        getattr(protocol_index, "source_path", "")
                    )
                    result = run_single_benchmark(
                        full_text, nct_id, query_cached,
                    )
                    elapsed = time.monotonic() - t0
                    st.session_state["benchmark_cache"][
                        file_sha
                    ] = result
                    cached = result
                    status.update(
                        label=f"Benchmark: Complete ({elapsed:.1f}s)",
                        state="complete",
                    )
                except Exception:
                    elapsed = time.monotonic() - t0
                    status.update(
                        label=f"Benchmark: Error ({elapsed:.1f}s)",
                        state="error",
                    )
                    st.code(traceback.format_exc(), language="text")
                    return

    if not cached:
        st.info(
            "Click **Run Benchmark** to compare text-first vs "
            "query-driven pipelines."
        )
        return

    # --- Recommendation ---
    rec = cached["recommendation"]
    reason = cached["recommendation_reason"]
    badge_type, display_text = format_recommendation(rec, reason)
    if badge_type == "success":
        st.success(display_text)
    elif badge_type == "warning":
        st.warning(display_text)
    else:
        st.info(display_text)

    # --- Key metrics ---
    st.subheader("Comparison Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Coverage delta",
            f"{cached['coverage_delta']:+.1f}%",
        )
    with col2:
        st.metric(
            "Coherence ratio",
            f"{cached['coherence_ratio']:.2f}x",
        )
    with col3:
        st.metric(
            "Section agreement",
            f"{cached['section_agreement']:.1%}",
        )

    # --- Detailed comparison table ---
    st.subheader("Metrics Comparison")
    rows = format_metrics_comparison(
        cached["text_first"], cached["query_driven"],
    )
    if rows:
        st.dataframe(rows, use_container_width=True)
