"""Schedule of Visits swimlane timeline component (PTCV-34).

Renders a multi-swimlane Plotly chart mapping UsdmActivity instances
to four category rows across protocol visits (UsdmTimepoint).

Swimlane rows:
  Clinical Encounter  - assessments, vitals, ECG, safety, consent
  Specimens           - labs, pharmacokinetics
  Imaging             - imaging procedures
  Other               - procedures, unclassified activities

Pure-Python helpers (classify_swimlane, build_sov_grid, build_sov_csv)
are testable without Streamlit.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from typing import Any

from ptcv.soa_extractor.models import (
    UsdmActivity,
    UsdmScheduledInstance,
    UsdmTimepoint,
)


# ---------------------------------------------------------------------------
# Swimlane classification
# ---------------------------------------------------------------------------

# Ordered list: first match wins
SWIMLANE_ROWS = [
    "Clinical Encounter",
    "Specimens",
    "Imaging",
    "Other",
]

_TYPE_TO_SWIMLANE: dict[str, str] = {
    "Assessment": "Clinical Encounter",
    "Vital Signs": "Clinical Encounter",
    "ECG": "Clinical Encounter",
    "Safety": "Clinical Encounter",
    "Consent": "Clinical Encounter",
    "Lab": "Specimens",
    "Pharmacokinetics": "Specimens",
    "Imaging": "Imaging",
    "Procedure": "Other",
    "Other": "Other",
}


def classify_swimlane(activity_type: str) -> str:
    """Map a UsdmActivity.activity_type to one of 4 swimlane rows.

    Args:
        activity_type: Activity type string from UsdmMapper
            (e.g. "Assessment", "Lab", "Imaging").

    Returns:
        One of SWIMLANE_ROWS.
    """
    return _TYPE_TO_SWIMLANE.get(activity_type, "Other")


# ---------------------------------------------------------------------------
# Grid builder (pure Python)
# ---------------------------------------------------------------------------

def build_sov_grid(
    timepoints: list[UsdmTimepoint],
    activities: list[UsdmActivity],
    instances: list[UsdmScheduledInstance],
) -> list[dict[str, Any]]:
    """Build a flat list of records for the swimlane chart.

    Each record represents one filled cell (visit x swimlane row)
    with a tooltip listing the specific activities at that intersection.

    Args:
        timepoints: USDM timepoints (visits) in protocol order.
        activities: USDM activities extracted from the SoA.
        instances: Scheduled instances linking activities to timepoints.

    Returns:
        List of dicts with keys: visit_name, day_offset, swimlane,
        activities (comma-joined names), activity_count, visit_index.
    """
    # Index lookups
    tp_by_id = {tp.timepoint_id: tp for tp in timepoints}
    act_by_id = {a.activity_id: a for a in activities}

    # Sort timepoints by day_offset for x-axis ordering
    sorted_tps = sorted(timepoints, key=lambda t: t.day_offset)
    tp_order = {tp.timepoint_id: idx for idx, tp in enumerate(sorted_tps)}

    # Group: (timepoint_id, swimlane) -> list[activity_name]
    cell_activities: dict[tuple[str, str], list[str]] = defaultdict(list)

    for inst in instances:
        if not inst.scheduled:
            continue
        act = act_by_id.get(inst.activity_id)
        tp = tp_by_id.get(inst.timepoint_id)
        if act is None or tp is None:
            continue
        swimlane = classify_swimlane(act.activity_type)
        cell_activities[(inst.timepoint_id, swimlane)].append(
            act.activity_name
        )

    # Build flat records
    records: list[dict[str, Any]] = []
    for (tp_id, swimlane), names in cell_activities.items():
        tp = tp_by_id[tp_id]
        records.append({
            "visit_name": tp.visit_name,
            "day_offset": tp.day_offset,
            "swimlane": swimlane,
            "activities": ", ".join(sorted(set(names))),
            "activity_count": len(set(names)),
            "visit_index": tp_order.get(tp_id, 0),
        })

    # Sort by visit order then swimlane order
    lane_order = {lane: i for i, lane in enumerate(SWIMLANE_ROWS)}
    records.sort(key=lambda r: (r["visit_index"], lane_order.get(r["swimlane"], 99)))
    return records


def build_sov_csv(
    timepoints: list[UsdmTimepoint],
    activities: list[UsdmActivity],
    instances: list[UsdmScheduledInstance],
) -> str:
    """Build a CSV string from the SoA grid data.

    Columns: Visit, Day, Category, Activities

    Args:
        timepoints: USDM timepoints.
        activities: USDM activities.
        instances: Scheduled instances.

    Returns:
        CSV string with header row.
    """
    grid = build_sov_grid(timepoints, activities, instances)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Visit", "Day", "Category", "Activities"])
    for row in grid:
        writer.writerow([
            row["visit_name"],
            row["day_offset"],
            row["swimlane"],
            row["activities"],
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Plotly chart builder (pure Python — no Streamlit dependency)
# ---------------------------------------------------------------------------

def build_sov_figure(
    timepoints: list[UsdmTimepoint],
    activities: list[UsdmActivity],
    instances: list[UsdmScheduledInstance],
    registry_id: str = "",
    low_confidence: bool = False,
) -> Any:
    """Build a Plotly Figure for the schedule of visits swimlane chart.

    Uses go.Scatter with markers to create a heatmap-like grid.
    Each filled cell is a marker whose hover text lists the activities.

    Args:
        timepoints: USDM timepoints.
        activities: USDM activities.
        instances: Scheduled instances.
        registry_id: Trial ID for chart title.
        low_confidence: If True, markers use a muted palette to
            indicate low-confidence extraction.

    Returns:
        plotly.graph_objects.Figure instance.
    """
    import plotly.graph_objects as go

    grid = build_sov_grid(timepoints, activities, instances)
    if not grid:
        fig = go.Figure()
        fig.add_annotation(
            text="No schedule of visits data extracted",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title=f"Schedule of Visits — {registry_id}" if registry_id else "Schedule of Visits",
        )
        return fig

    # Build ordered visit labels for x-axis
    sorted_tps = sorted(timepoints, key=lambda t: t.day_offset)
    visit_labels = [tp.visit_name for tp in sorted_tps]
    visit_indices = list(range(len(visit_labels)))

    # Swimlane row positions (bottom to top: Other=0, Imaging=1, Specimens=2, Clinical=3)
    lane_y = {lane: i for i, lane in enumerate(reversed(SWIMLANE_ROWS))}

    # Colour per swimlane
    if low_confidence:
        lane_colors = {
            "Clinical Encounter": "rgba(180,180,200,0.6)",
            "Specimens": "rgba(180,200,180,0.6)",
            "Imaging": "rgba(200,180,180,0.6)",
            "Other": "rgba(190,190,190,0.6)",
        }
    else:
        lane_colors = {
            "Clinical Encounter": "#636EFA",
            "Specimens": "#EF553B",
            "Imaging": "#00CC96",
            "Other": "#AB63FA",
        }

    # Build traces per swimlane for legend grouping
    traces_by_lane: dict[str, dict[str, list]] = {
        lane: {"x": [], "y": [], "text": [], "size": []}
        for lane in SWIMLANE_ROWS
    }

    for rec in grid:
        lane = rec["swimlane"]
        traces_by_lane[lane]["x"].append(rec["visit_index"])
        traces_by_lane[lane]["y"].append(lane_y[lane])
        traces_by_lane[lane]["text"].append(rec["activities"])
        traces_by_lane[lane]["size"].append(
            min(12 + rec["activity_count"] * 4, 30)
        )

    fig = go.Figure()
    for lane in SWIMLANE_ROWS:
        data = traces_by_lane[lane]
        if not data["x"]:
            continue
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name=lane,
            marker=dict(
                size=data["size"],
                color=lane_colors.get(lane, "#888"),
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            hovertext=data["text"],
            hoverinfo="text",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "<i>%{hovertext}</i>"
                "<extra></extra>"
            ),
            customdata=[
                [visit_labels[x] if x < len(visit_labels) else "", lane]
                for x in data["x"]
            ],
        ))

    title = "Schedule of Visits"
    if registry_id:
        title = f"Schedule of Visits — {registry_id}"

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=visit_indices,
            ticktext=visit_labels,
            title="Visit",
            tickangle=-45,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(SWIMLANE_ROWS))),
            ticktext=list(reversed(SWIMLANE_ROWS)),
            title="",
        ),
        height=350,
        margin=dict(l=140, r=20, t=50, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )

    # Add gridlines
    for i in range(len(SWIMLANE_ROWS)):
        fig.add_hline(
            y=i - 0.5, line_dash="dot",
            line_color="lightgrey", line_width=0.5,
        )
    for i in range(len(visit_labels)):
        fig.add_vline(
            x=i - 0.5, line_dash="dot",
            line_color="lightgrey", line_width=0.5,
        )

    return fig


# ---------------------------------------------------------------------------
# Streamlit render function
# ---------------------------------------------------------------------------

def render_schedule_of_visits(
    timepoints: list[UsdmTimepoint],
    activities: list[UsdmActivity],
    instances: list[UsdmScheduledInstance],
    registry_id: str = "",
    format_verdict: str = "",
) -> None:
    """Render the schedule of visits swimlane chart in Streamlit.

    Displays a warning banner for NON_ICH protocols, renders the
    Plotly chart, and provides CSV and PNG download buttons.

    Args:
        timepoints: USDM timepoints from the SoA extraction.
        activities: USDM activities from the SoA extraction.
        instances: Scheduled instances linking activities to timepoints.
        registry_id: Trial ID for the chart title.
        format_verdict: ICH format verdict (ICH_E6R3/PARTIAL_ICH/NON_ICH).
    """
    import streamlit as st

    st.subheader("Schedule of Visits")

    low_confidence = format_verdict == "NON_ICH"
    if low_confidence:
        st.warning(
            "Protocol structure is non-ICH — "
            "visit schedule may be incomplete"
        )

    if not timepoints or not activities:
        st.info("No schedule of visits data was extracted from this protocol.")
        return

    fig = build_sov_figure(
        timepoints, activities, instances,
        registry_id=registry_id,
        low_confidence=low_confidence,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv_data = build_sov_csv(timepoints, activities, instances)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"schedule_of_visits_{registry_id}.csv",
            mime="text/csv",
        )
    with col2:
        png_bytes = fig.to_image(format="png", width=1200, height=400)
        st.download_button(
            label="Download as PNG",
            data=png_bytes,
            file_name=f"schedule_of_visits_{registry_id}.png",
            mime="image/png",
        )
