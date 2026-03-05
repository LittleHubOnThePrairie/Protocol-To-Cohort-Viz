"""Refinement feedback panel for Streamlit UI (PTCV-95).

Exposes the RefinementStore (PTCV-94) through a UI for:
  - Submitting header mapping and extraction corrections
  - Viewing calibration reports with bias detection
  - Viewing accuracy trends and promoted synonyms

Pure-Python helpers are separated from Streamlit rendering for testability.
"""

from __future__ import annotations

import logging
from typing import Any

from ptcv.ui.components.confidence_badge import format_confidence

logger = logging.getLogger(__name__)

# Bias threshold for flagging calibration bands (matches refinement_store.py).
BIAS_FLAG_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Pure-Python helpers (testable, no Streamlit imports)
# ---------------------------------------------------------------------------


def format_calibration_table(
    report: Any,
) -> list[dict[str, Any]]:
    """Format calibration bands for tabular display.

    Args:
        report: ``CalibrationReport`` from
            ``RefinementStore.get_calibration_report()``.

    Returns:
        List of dicts with keys: ``range``, ``count``,
        ``predicted_avg``, ``actual_accuracy``, ``bias``, ``flagged``.
    """
    rows: list[dict[str, Any]] = []
    for band in getattr(report, "bands", []):
        bias = getattr(band, "bias", 0.0)
        rows.append({
            "range": (
                f"{getattr(band, 'range_low', 0.0):.2f}"
                f"-{getattr(band, 'range_high', 0.0):.2f}"
            ),
            "count": getattr(band, "count", 0),
            "predicted_avg": f"{getattr(band, 'predicted_avg', 0.0):.3f}",
            "actual_accuracy": f"{getattr(band, 'actual_accuracy', 0.0):.3f}",
            "bias": f"{bias:+.4f}",
            "flagged": abs(bias) > BIAS_FLAG_THRESHOLD,
        })
    return rows


def format_trend_summary(report: Any) -> dict[str, Any]:
    """Extract trend data for display.

    Args:
        report: ``TrendReport`` from
            ``RefinementStore.get_trend_report()``.

    Returns:
        Dict with keys: ``total_corrections``,
        ``total_calibration``, ``common_corrections``,
        ``synonyms``, ``accuracy_by_month``.
    """
    common = []
    for item in getattr(report, "common_corrections", []):
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            common.append({
                "header": item[0],
                "corrected_section": item[1],
                "count": item[2],
            })

    synonyms = []
    for item in getattr(report, "new_synonyms", []):
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            synonyms.append({
                "header": item[0],
                "section_code": item[1],
            })

    monthly = []
    for item in getattr(report, "accuracy_by_month", []):
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            monthly.append({
                "month": item[0],
                "accuracy": f"{item[1]:.1%}",
            })

    return {
        "total_corrections": getattr(
            report, "total_corrections", 0
        ),
        "total_calibration": getattr(
            report, "total_calibration_entries", 0
        ),
        "common_corrections": common,
        "synonyms": synonyms,
        "accuracy_by_month": monthly,
    }


def format_synonym_table(
    boosts: dict[str, str],
) -> list[dict[str, str]]:
    """Format synonym boosts for tabular display.

    Args:
        boosts: Dict mapping header text to section code from
            ``RefinementStore.get_synonym_boosts()``.

    Returns:
        List of dicts with keys: ``header``, ``section_code``.
    """
    return [
        {"header": header, "section_code": code}
        for header, code in sorted(boosts.items())
    ]


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def render_refinement_panel(
    registry_id: str,
    query_cached: dict[str, Any] | None,
) -> None:
    """Render the Refinement tab.

    Shows four sub-tabs: Corrections, Calibration, Trends,
    and Synonym Boosts.

    Args:
        registry_id: Current protocol's registry ID.
        query_cached: Result from Query Pipeline tab (may be None).
    """
    import streamlit as st
    from ptcv.ich_parser.refinement_store import RefinementStore

    # Shared store across the session
    if "refinement_store" not in st.session_state:
        st.session_state["refinement_store"] = RefinementStore()
    store: RefinementStore = st.session_state["refinement_store"]

    tab_correct, tab_calib, tab_trends, tab_syn = st.tabs(
        ["Corrections", "Calibration", "Trends", "Synonyms"],
    )

    # === Corrections tab ===
    with tab_correct:
        st.subheader("Submit Header Correction")
        with st.form("header_correction_form", clear_on_submit=True):
            header_text = st.text_input(
                "Protocol header text",
                key="hc_header",
            )
            original = st.text_input(
                "Original mapping (e.g. B.7)",
                key="hc_original",
            )
            corrected = st.text_input(
                "Corrected mapping (e.g. B.9)",
                key="hc_corrected",
            )
            submitted = st.form_submit_button("Submit Correction")
            if submitted and header_text and original and corrected:
                store.record_header_correction(
                    protocol_id=registry_id,
                    protocol_header=header_text,
                    original_mapping=original,
                    corrected_mapping=corrected,
                )
                st.success(
                    f"Correction recorded: {header_text} "
                    f"{original} -> {corrected}"
                )

        st.divider()

        st.subheader("Submit Extraction Correction")
        with st.form("extraction_correction_form", clear_on_submit=True):
            query_id = st.text_input(
                "Query ID (e.g. B.6.1.q1)",
                key="ec_query_id",
            )
            orig_content = st.text_area(
                "Original content",
                key="ec_original",
            )
            corr_content = st.text_area(
                "Corrected content",
                key="ec_corrected",
            )
            corr_section = st.text_input(
                "Corrected section (if section was wrong)",
                key="ec_section",
            )
            ec_submitted = st.form_submit_button(
                "Submit Extraction Correction"
            )
            if ec_submitted and query_id and corr_content:
                store.record_extraction_correction(
                    protocol_id=registry_id,
                    query_id=query_id,
                    original_content=orig_content,
                    corrected_content=corr_content,
                    corrected_section=corr_section,
                )
                st.success(
                    f"Extraction correction recorded for {query_id}"
                )

        st.caption(
            f"Total corrections: "
            f"{store.header_correction_count} header, "
            f"{store.extraction_correction_count} extraction"
        )

    # === Calibration tab ===
    with tab_calib:
        st.subheader("Confidence Calibration")
        cal_report = store.get_calibration_report()

        if cal_report.total_entries == 0:
            st.info(
                "No calibration data yet. Calibration entries are "
                "recorded when extractions are reviewed."
            )
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entries", cal_report.total_entries)
            with col2:
                st.metric(
                    "Overall accuracy",
                    f"{cal_report.overall_accuracy:.1%}",
                )
            with col3:
                adj = cal_report.suggested_threshold_adjustment
                st.metric(
                    "Threshold adjustment",
                    f"{adj:+.3f}",
                )

            if cal_report.bias_detected:
                st.warning(
                    "Bias detected: one or more confidence bands "
                    "have |bias| > 0.15"
                )

            bands = format_calibration_table(cal_report)
            if bands:
                st.dataframe(bands, use_container_width=True)

    # === Trends tab ===
    with tab_trends:
        st.subheader("Accuracy Trends")
        trend_report = store.get_trend_report()
        summary = format_trend_summary(trend_report)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total corrections",
                summary["total_corrections"],
            )
        with col2:
            st.metric(
                "Calibration entries",
                summary["total_calibration"],
            )

        if summary["common_corrections"]:
            st.subheader("Most Common Corrections")
            st.dataframe(
                summary["common_corrections"],
                use_container_width=True,
            )

        if summary["accuracy_by_month"]:
            st.subheader("Monthly Accuracy")
            st.dataframe(
                summary["accuracy_by_month"],
                use_container_width=True,
            )

    # === Synonyms tab ===
    with tab_syn:
        st.subheader("Promoted Synonym Boosts")
        boosts = store.get_synonym_boosts()

        if not boosts:
            st.info(
                "No synonyms promoted yet. Synonyms are promoted "
                "when a header correction is seen 2+ times."
            )
        else:
            rows = format_synonym_table(boosts)
            st.dataframe(rows, use_container_width=True)
            st.caption(
                f"{len(boosts)} synonym(s) promoted to "
                "SectionMatcher boost table"
            )
