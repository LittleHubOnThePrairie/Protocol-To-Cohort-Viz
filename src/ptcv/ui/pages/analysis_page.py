"""Batch Analysis page for the PTCV Streamlit app (PTCV-150).

Provides a cross-protocol analysis view:
- Batch run browser (list, select, compare runs)
- Run summary with per-section stats
- Comparison pair viewer (original vs extracted)
- Regression detection between runs

This page operates independently of the per-protocol tabs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

_DEFAULT_DB = Path("data/analysis/results.db")


def _load_harness(db_path: Path):  # noqa: ANN202 — lazy import
    """Lazy-load AnalysisHarness to avoid import errors."""
    from ptcv.analysis.integration import AnalysisHarness
    return AnalysisHarness(db_path)


def render_analysis_page(db_path: Path | None = None) -> None:
    """Render the batch analysis page.

    Args:
        db_path: Path to SQLite analysis database.
            Defaults to ``data/analysis/results.db``.
    """
    db = db_path or _DEFAULT_DB

    if not db.exists():
        st.info(
            "No analysis database found. Run the batch runner first:\n\n"
            "```\npython -m ptcv.analysis.batch_runner "
            "--output-db data/analysis/results.db\n```",
        )
        return

    harness = _load_harness(db)

    try:
        runs = harness.list_runs()
    except Exception as exc:
        st.error(f"Failed to load runs: {exc}")
        return

    if not runs:
        st.info("No batch runs found in the database.")
        return

    # --- Run selector ---
    run_options = {
        f"{r['run_id']} ({r.get('protocol_count', '?')} protocols, "
        f"{r.get('timestamp', '')[:19]})": r["run_id"]
        for r in runs
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_label = st.selectbox(
            "Batch Run",
            options=list(run_options.keys()),
            key="analysis_run_selector",
        )
    selected_run = run_options[selected_label]

    with col2:
        view_mode = st.radio(
            "View",
            ["Summary", "Sections", "Protocols", "Compare"],
            horizontal=True,
            key="analysis_view_mode",
        )

    if view_mode == "Summary":
        _render_summary(harness, selected_run)
    elif view_mode == "Sections":
        _render_sections(harness, selected_run)
    elif view_mode == "Protocols":
        _render_protocols(harness, selected_run)
    elif view_mode == "Compare":
        _render_compare(harness, selected_run, runs)


def _render_summary(harness: Any, run_id: str) -> None:
    """Display run summary report."""
    report = harness.get_run_summary(run_id)
    st.markdown(report)


def _render_sections(harness: Any, run_id: str) -> None:
    """Display per-section statistics with drilldown."""
    stats = harness.get_section_stats(run_id)
    if not stats:
        st.info("No section data for this run.")
        return

    # Summary table
    table_data = []
    for s in sorted(stats, key=lambda x: x["avg_boosted"]):
        table_data.append({
            "ICH Code": s["ich_section_code"],
            "Section": s["ich_section_name"],
            "Avg Confidence": round(s["avg_boosted"], 4),
            "Auto-Map Rate": f"{s['auto_map_rate']:.1%}",
            "Hit Rate": f"{s['hit_rate']:.1%}",
            "Protocols": s["protocol_count"],
        })

    st.dataframe(table_data, use_container_width=True)

    # Section drilldown
    section_codes = [s["ich_section_code"] for s in stats]
    selected_section = st.selectbox(
        "Drill into section",
        options=section_codes,
        key="analysis_section_drill",
    )

    if selected_section:
        protocols = harness.list_protocols(run_id)
        matching = []
        for p in protocols:
            pairs = harness.get_comparison_pairs(
                run_id, p["nct_id"], selected_section,
            )
            for pair in pairs:
                matching.append({
                    "NCT ID": p["nct_id"],
                    "Quality": pair.get("match_quality", ""),
                    "Confidence": pair.get("confidence", 0.0),
                    "Original": (pair.get("original_text") or "")[:200],
                    "Extracted": (pair.get("extracted_text") or "")[:200],
                })

        if matching:
            st.dataframe(matching, use_container_width=True)
        else:
            st.info(
                f"No comparison pairs for {selected_section}.",
            )


def _render_protocols(harness: Any, run_id: str) -> None:
    """Display protocol list with comparison pair viewer."""
    protocols = harness.list_protocols(run_id)
    if not protocols:
        st.info("No protocols in this run.")
        return

    nct_options = [p["nct_id"] for p in protocols]
    selected_nct = st.selectbox(
        "Protocol",
        options=nct_options,
        key="analysis_protocol_selector",
    )

    if selected_nct:
        pairs = harness.get_comparison_pairs(run_id, selected_nct)
        if not pairs:
            st.info(f"No comparison pairs for {selected_nct}.")
            return

        for pair in pairs:
            quality = pair.get("match_quality", "unknown")
            code = pair.get("ich_section_code", "")
            conf = pair.get("confidence", 0.0)

            color = {
                "good": "green", "partial": "orange",
                "poor": "red", "gap": "red", "missing": "gray",
            }.get(quality, "gray")

            st.markdown(
                f"#### {code} "
                f"— :{color}[{quality}] "
                f"(confidence: {conf:.2f})",
            )

            col_orig, col_ext = st.columns(2)
            with col_orig:
                st.text_area(
                    "Original",
                    value=pair.get("original_text") or "(none)",
                    height=150,
                    key=f"orig_{code}_{selected_nct}",
                    disabled=True,
                )
            with col_ext:
                st.text_area(
                    "Extracted",
                    value=pair.get("extracted_text") or "(none)",
                    height=150,
                    key=f"ext_{code}_{selected_nct}",
                    disabled=True,
                )
            st.divider()


def _render_compare(
    harness: Any, current_run: str, runs: list[dict],
) -> None:
    """Compare two batch runs with regression detection."""
    if len(runs) < 2:
        st.info("Need at least two runs to compare.")
        return

    run_ids = [r["run_id"] for r in runs]
    other_runs = [r for r in run_ids if r != current_run]

    if not other_runs:
        st.info("No other runs to compare against.")
        return

    compare_to = st.selectbox(
        "Compare against (baseline)",
        options=other_runs,
        key="analysis_compare_run",
    )

    threshold = st.slider(
        "Regression threshold",
        min_value=0.01,
        max_value=0.10,
        value=0.02,
        step=0.01,
        key="analysis_regression_threshold",
    )

    report = harness.get_regression_report(
        compare_to, current_run, threshold,
    )
    st.markdown(report)

    # Detailed regressions
    regressions = harness.get_regressions(
        compare_to, current_run, threshold,
    )
    if regressions:
        st.subheader("Regressed Sections")
        st.dataframe(regressions, use_container_width=True)
