"""SDTM viewer with lineage visualization component (PTCV-36).

Renders tabbed SDTM domain tables (TS, TA, TE, TV, TI) with download
buttons, and a Plotly Sankey diagram showing the protocol-to-SDTM
lineage flow.

Pure-Python helpers (build_lineage_records, build_lineage_figure) are
testable without Streamlit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ptcv.ich_parser.models import IchSection
    from ptcv.sdtm.models import SdtmGenerationResult


# ---------------------------------------------------------------------------
# Lineage mapping: ICH section → SDTM domain
# ---------------------------------------------------------------------------

# Each tuple: (source_label, target_domain_key)
LINEAGE_LINKS: list[tuple[str, str]] = [
    ("B.1 General Information", "TS"),
    ("B.4 Trial Design", "TS"),
    ("B.4 Trial Design", "TA"),
    ("B.4 Trial Design", "TE"),
    ("B.4 Trial Design", "DM"),
    ("B.4 Trial Design", "EX"),
    ("B.5 Selection of Subjects", "TI"),
    ("B.5 Selection of Subjects", "MH"),
    ("SoA Visit Schedule", "TV"),
    ("SoA Visit Schedule", "SV"),
    ("SoA Visit Schedule", "LB"),
    ("SoA Visit Schedule", "VS"),
    ("SoA Visit Schedule", "AE"),
    ("SoA Visit Schedule", "DS"),
    ("SoA Visit Schedule", "CM"),
]

DOMAIN_LABELS: dict[str, str] = {
    "TS": "TS \u2014 Trial Summary",
    "TA": "TA \u2014 Trial Arms",
    "TE": "TE \u2014 Trial Elements",
    "TV": "TV \u2014 Trial Visits",
    "TI": "TI \u2014 Inclusion/Exclusion",
    "DM": "DM \u2014 Demographics",
    "SV": "SV \u2014 Subject Visits",
    "LB": "LB \u2014 Laboratory",
    "AE": "AE \u2014 Adverse Events",
    "VS": "VS \u2014 Vital Signs",
    "CM": "CM \u2014 Concomitant Meds",
    "MH": "MH \u2014 Medical History",
    "DS": "DS \u2014 Disposition",
    "EX": "EX \u2014 Exposure",
}

# Ordered domain list for tab display
# Trial design domains first, then clinical domains (PTCV-284)
DOMAIN_ORDER: list[str] = [
    "TV", "TA", "TE", "TS", "TI",
    "DM", "SV", "LB", "AE", "VS", "CM", "MH", "DS", "EX",
]

# Map section codes to source labels for matching
_CODE_TO_SOURCE: dict[str, str] = {
    "B.1": "B.1 General Information",
    "B.4": "B.4 Trial Design",
    "B.5": "B.5 Selection of Subjects",
}


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------

def build_lineage_records(
    section_codes: list[str],
    domain_row_counts: dict[str, int],
    has_soa: bool = False,
) -> list[dict[str, Any]]:
    """Build lineage flow records for the Sankey diagram.

    Only includes links where the source section was actually detected
    and the target domain has rows.

    Args:
        section_codes: ICH section codes present (e.g. ["B.1", "B.4"]).
        domain_row_counts: Domain name → row count (e.g. {"TS": 12}).
        has_soa: True if SoA extraction produced timepoints.

    Returns:
        List of dicts with keys: source, target, value, target_domain.
    """
    available_sources: set[str] = set()
    for code in section_codes:
        label = _CODE_TO_SOURCE.get(code)
        if label:
            available_sources.add(label)
    if has_soa:
        available_sources.add("SoA Visit Schedule")

    records: list[dict[str, Any]] = []
    for source_label, domain_key in LINEAGE_LINKS:
        if source_label not in available_sources:
            continue
        row_count = domain_row_counts.get(domain_key, 0)
        if row_count == 0:
            continue
        records.append({
            "source": source_label,
            "target": DOMAIN_LABELS.get(domain_key, domain_key),
            "value": max(row_count, 1),
            "target_domain": domain_key,
        })

    return records


def build_lineage_figure(
    records: list[dict[str, Any]],
    registry_id: str = "",
) -> Any:
    """Build a Plotly Sankey diagram from lineage records.

    Args:
        records: List from build_lineage_records().
        registry_id: Trial ID for chart title.

    Returns:
        plotly.graph_objects.Figure instance.
    """
    import plotly.graph_objects as go

    if not records:
        fig = go.Figure()
        fig.add_annotation(
            text="No lineage data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )
        title = "Protocol \u2192 SDTM Lineage"
        if registry_id:
            title = f"Protocol \u2192 SDTM Lineage \u2014 {registry_id}"
        fig.update_layout(title=title)
        return fig

    # Collect unique node labels (sources first, then targets)
    source_labels: list[str] = []
    target_labels: list[str] = []
    for rec in records:
        if rec["source"] not in source_labels:
            source_labels.append(rec["source"])
        if rec["target"] not in target_labels:
            target_labels.append(rec["target"])

    all_labels = source_labels + target_labels
    label_idx = {label: i for i, label in enumerate(all_labels)}

    # Build link arrays
    link_source: list[int] = []
    link_target: list[int] = []
    link_value: list[int] = []
    link_label: list[str] = []

    for rec in records:
        link_source.append(label_idx[rec["source"]])
        link_target.append(label_idx[rec["target"]])
        link_value.append(rec["value"])
        link_label.append(f"{rec['value']} rows")

    # Node colours: blue for sources, green for targets
    node_colors = (
        ["#636EFA"] * len(source_labels)
        + ["#00CC96"] * len(target_labels)
    )

    # Scale height based on number of nodes to avoid compression
    n_nodes = len(all_labels)
    fig_height = max(400, 30 * n_nodes + 100)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=22,
            line=dict(color="white", width=0.5),
            label=all_labels,
            color=node_colors,
        ),
        link=dict(
            source=link_source,
            target=link_target,
            value=link_value,
            label=link_label,
            color="rgba(99, 110, 250, 0.3)",
        ),
        textfont=dict(size=13, family="Arial, sans-serif"),
    )])

    title = "Protocol \u2192 SDTM Lineage"
    if registry_id:
        title = f"Protocol \u2192 SDTM Lineage \u2014 {registry_id}"
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=fig_height,
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(size=13),
    )

    return fig


# ---------------------------------------------------------------------------
# XPT deserialization helper
# ---------------------------------------------------------------------------

def _xpt_bytes_to_dataframe(xpt_bytes: bytes) -> Any:
    """Read SAS XPT bytes into a pandas DataFrame.

    Args:
        xpt_bytes: Raw XPT file content.

    Returns:
        pandas.DataFrame.
    """
    import os
    import tempfile

    import pandas as pd
    import pyreadstat  # type: ignore[import-untyped]

    fd, tmp_path = tempfile.mkstemp(suffix=".xpt")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(xpt_bytes)
        df, _ = pyreadstat.read_xport(tmp_path)
        return df
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Streamlit render function
# ---------------------------------------------------------------------------

def render_sdtm_viewer(
    sdtm_result: "SdtmGenerationResult",
    sections: list["IchSection"],
    has_soa: bool,
    gateway: Any,
    registry_id: str = "",
    format_verdict: str = "",
) -> None:
    """Render the SDTM viewer in Streamlit.

    Displays:
    - NON_ICH warning banner (if applicable)
    - Tabbed domain tables with row counts
    - XPT download buttons per domain
    - Sankey lineage diagram
    - Unmapped CT terms (collapsible)

    Args:
        sdtm_result: SdtmGenerationResult from SdtmService.generate().
        sections: IchSection list for lineage mapping.
        has_soa: Whether SoA data was available.
        gateway: StorageGateway to read XPT artifacts.
        registry_id: Trial ID for titles.
        format_verdict: ICH format verdict string.
    """
    import streamlit as st

    st.subheader("Mock SDTM Generation")

    # NON_ICH warning banner
    if format_verdict == "NON_ICH":
        st.warning(
            "SDTM generated from non-ICH protocol "
            "\u2014 output is likely incomplete"
        )

    # --- Sankey lineage diagram ---
    section_codes = list({s.section_code for s in sections})
    lineage = build_lineage_records(
        section_codes=section_codes,
        domain_row_counts=sdtm_result.domain_row_counts,
        has_soa=has_soa,
    )
    fig = build_lineage_figure(lineage, registry_id=registry_id)
    st.plotly_chart(fig, use_container_width=True)

    # --- Domain tables in tabs ---
    domain_tabs = []
    for domain in DOMAIN_ORDER:
        row_count = sdtm_result.domain_row_counts.get(domain, 0)
        domain_tabs.append(f"{domain} \u2014 {row_count} rows")

    tabs = st.tabs(domain_tabs)
    for tab, domain in zip(tabs, DOMAIN_ORDER):
        with tab:
            row_count = sdtm_result.domain_row_counts.get(domain, 0)
            domain_lower = domain.lower()

            if row_count == 0:
                st.info(
                    f"No {DOMAIN_LABELS.get(domain, domain)} data "
                    f"was generated from this protocol."
                )
                continue

            # Read XPT from storage and display as DataFrame
            if domain_lower in sdtm_result.artifact_keys:
                try:
                    xpt_bytes = gateway.get_artifact(
                        sdtm_result.artifact_keys[domain_lower]
                    )
                    df = _xpt_bytes_to_dataframe(xpt_bytes)
                    st.dataframe(df, use_container_width=True)

                    # Download button for XPT
                    st.download_button(
                        label=f"Download {domain}.xpt",
                        data=xpt_bytes,
                        file_name=f"{domain.lower()}_{registry_id}.xpt",
                        mime="application/octet-stream",
                        key=f"dl_{domain}",
                    )
                except Exception as exc:
                    st.error(f"Error reading {domain}.xpt: {exc}")

    # --- Unmapped CT terms ---
    if sdtm_result.ct_unmapped_count > 0:
        with st.expander(
            f"Unmapped CT Terms ({sdtm_result.ct_unmapped_count})"
        ):
            st.warning(
                f"{sdtm_result.ct_unmapped_count} term(s) could not be "
                f"mapped to CDISC Controlled Terminology. "
                f"These require manual coding before submission."
            )

    # --- Generation metadata ---
    col1, col2, col3 = st.columns(3)
    with col1:
        total_rows = sum(sdtm_result.domain_row_counts.values())
        st.metric("Total SDTM rows", total_rows)
    with col2:
        st.metric("Domains generated", len(sdtm_result.domain_row_counts))
    with col3:
        st.metric("Unmapped CT terms", sdtm_result.ct_unmapped_count)
