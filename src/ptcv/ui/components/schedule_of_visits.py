"""Schedule of Visits swimlane timeline component (PTCV-34, PTCV-71).

Renders a multi-swimlane Plotly chart mapping UsdmActivity instances
to five category rows across protocol visits (UsdmTimepoint).

Swimlane rows:
  Intervention        - study drug/placebo administration, dosing
  Clinical Encounter  - assessments, vitals, ECG, safety, consent
  Specimens           - labs, pharmacokinetics, serology, biomarkers
  Imaging             - imaging procedures
  Other               - procedures, unclassified activities

Pure-Python helpers (classify_swimlane, build_sov_grid, build_sov_csv)
are testable without Streamlit.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import re
from collections import defaultdict
from typing import Any, NamedTuple, Sequence, Union

from ptcv.soa_extractor.models import (
    UsdmActivity,
    UsdmScheduledInstance,
    UsdmTimepoint,
)


# ---------------------------------------------------------------------------
# Lightweight plot-only types (PTCV-76)
# ---------------------------------------------------------------------------

class PlotTimepoint(NamedTuple):
    """Timepoint fields needed for visualization only."""

    timepoint_id: str
    visit_name: str
    day_offset: int


class PlotActivity(NamedTuple):
    """Activity fields needed for visualization only."""

    activity_id: str
    activity_name: str
    activity_type: str


class PlotInstance(NamedTuple):
    """Scheduled-instance fields needed for visualization only."""

    activity_id: str
    timepoint_id: str
    scheduled: bool


@dataclasses.dataclass
class SoVPlotData:
    """Filtered view of SoA data carrying only the 9 plotting fields.

    Strips ALCOA+ traceability fields (run_id, source_run_id,
    source_sha256, etc.) that are irrelevant to visualization.
    """

    timepoints: list[PlotTimepoint]
    activities: list[PlotActivity]
    instances: list[PlotInstance]


def to_plot_data(
    timepoints: Sequence[UsdmTimepoint],
    activities: Sequence[UsdmActivity],
    instances: Sequence[UsdmScheduledInstance],
) -> SoVPlotData:
    """Convert full USDM objects to a lightweight plot-only view.

    Call this at the UI boundary (app.py) so that visualization
    functions receive only the 9 fields they actually use.

    Args:
        timepoints: Full USDM timepoints from pipeline.
        activities: Full USDM activities from pipeline.
        instances: Full USDM scheduled instances from pipeline.

    Returns:
        SoVPlotData with only plotting-relevant fields.
    """
    return SoVPlotData(
        timepoints=[
            PlotTimepoint(tp.timepoint_id, tp.visit_name, tp.day_offset)
            for tp in timepoints
        ],
        activities=[
            PlotActivity(a.activity_id, a.activity_name, a.activity_type)
            for a in activities
        ],
        instances=[
            PlotInstance(i.activity_id, i.timepoint_id, i.scheduled)
            for i in instances
        ],
    )


# Type aliases for functions that accept either full or plot types.
_TP = Union[UsdmTimepoint, PlotTimepoint]
_ACT = Union[UsdmActivity, PlotActivity]
_INST = Union[UsdmScheduledInstance, PlotInstance]


# ---------------------------------------------------------------------------
# Swimlane classification
# ---------------------------------------------------------------------------

# Ordered list: first match wins (PTCV-71: added Intervention)
SWIMLANE_ROWS = [
    "Intervention",
    "Clinical Encounter",
    "Specimens",
    "Imaging",
    "Other",
]

_TYPE_TO_SWIMLANE: dict[str, str] = {
    "Drug Administration": "Intervention",
    "Treatment": "Intervention",
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

# Keyword patterns for name-based reclassification (PTCV-71).
# Checked when the coarse activity_type mapping is ambiguous.
_INTERVENTION_RE = re.compile(
    r"\bstudy\s+drug\b|\bplacebo\b|\b[Ii]MP\b|\binfusion\b"
    r"|\binjection\b|\bdosing\b|\bdose\s+admin"
    r"|\bdrug\s+admin|\btreatment\s+admin",
    re.IGNORECASE,
)
_SPECIMEN_RE = re.compile(
    r"\bserology\b|\bCBC\b|\bpregnancy\s+test\b|\burine\b"
    r"|\bblood\s+draw\b|\bbiomarker\b|\bantibod"
    r"|\bPCR\b|\bculture\b|\bhaematology\b|\bhematology\b"
    r"|\bchemistry\b|\bcoagulation\b|\burinalysis\b"
    r"|\bHep\s*[A-C]\b|\bhepatitis\b|\bblood\s+sample"
    r"|\bspecimen\b|\bserum\b|\bplasma\b",
    re.IGNORECASE,
)


def classify_swimlane(
    activity_type: str, activity_name: str = "",
) -> str:
    """Map a UsdmActivity to one of 5 swimlane rows.

    Uses the coarse ``activity_type`` first, then refines with
    keyword matching on ``activity_name`` to fix common
    misclassifications (PTCV-71).

    Args:
        activity_type: Activity type string from UsdmMapper
            (e.g. "Assessment", "Lab", "Imaging").
        activity_name: Optional activity name for keyword-based
            reclassification.

    Returns:
        One of SWIMLANE_ROWS.
    """
    lane = _TYPE_TO_SWIMLANE.get(activity_type, "Other")

    # Keyword-based reclassification when name is provided
    if activity_name:
        # Intervention keywords override everything except Imaging
        if lane != "Imaging" and _INTERVENTION_RE.search(activity_name):
            return "Intervention"
        # Specimen keywords override Clinical Encounter and Other
        if lane in ("Clinical Encounter", "Other") and _SPECIMEN_RE.search(
            activity_name
        ):
            return "Specimens"

    return lane


# ---------------------------------------------------------------------------
# Timepoint deduplication (PTCV-73)
# ---------------------------------------------------------------------------

def _dedup_timepoints(
    timepoints: Sequence[_TP],
) -> tuple[list[_TP], dict[str, str]]:
    """Merge timepoints that share the same (visit_name, day_offset).

    Returns:
        Tuple of (unique_timepoints, id_map) where id_map maps every
        original timepoint_id to the canonical (first-seen) timepoint_id
        for that (visit_name, day_offset) group.
    """
    seen: dict[tuple[str, int], _TP] = {}
    id_map: dict[str, str] = {}
    for tp in timepoints:
        key = (tp.visit_name, tp.day_offset)
        if key not in seen:
            seen[key] = tp
        id_map[tp.timepoint_id] = seen[key].timepoint_id
    return list(seen.values()), id_map


# ---------------------------------------------------------------------------
# Grid builder (pure Python)
# ---------------------------------------------------------------------------

def build_sov_grid(
    timepoints: Sequence[_TP],
    activities: Sequence[_ACT],
    instances: Sequence[_INST],
) -> list[dict[str, Any]]:
    """Build a flat list of records for the swimlane chart.

    Each record represents one filled cell (visit x swimlane row)
    with a tooltip listing the specific activities at that intersection.

    Duplicate timepoints (same visit_name + day_offset) are merged so
    their activities appear in a single column (PTCV-73).

    Args:
        timepoints: USDM timepoints (visits) in protocol order.
        activities: USDM activities extracted from the SoA.
        instances: Scheduled instances linking activities to timepoints.

    Returns:
        List of dicts with keys: visit_name, day_offset, swimlane,
        activities (comma-joined names), activity_count, visit_index.
    """
    # Deduplicate timepoints (PTCV-73)
    unique_tps, tp_id_map = _dedup_timepoints(timepoints)

    # Index lookups — use unique timepoints only
    tp_by_id = {tp.timepoint_id: tp for tp in unique_tps}
    act_by_id = {a.activity_id: a for a in activities}

    # Sort unique timepoints by day_offset for x-axis ordering
    sorted_tps = sorted(unique_tps, key=lambda t: t.day_offset)
    tp_order = {tp.timepoint_id: idx for idx, tp in enumerate(sorted_tps)}

    # Group: (canonical_timepoint_id, swimlane) -> list[activity_name]
    cell_activities: dict[tuple[str, str], list[str]] = defaultdict(list)

    for inst in instances:
        if not inst.scheduled:
            continue
        act = act_by_id.get(inst.activity_id)
        # Map to canonical timepoint id
        canonical_tp_id = tp_id_map.get(inst.timepoint_id)
        if act is None or canonical_tp_id is None:
            continue
        tp = tp_by_id.get(canonical_tp_id)
        if tp is None:
            continue
        swimlane = classify_swimlane(act.activity_type, act.activity_name)
        cell_activities[(canonical_tp_id, swimlane)].append(
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
    timepoints: Sequence[_TP],
    activities: Sequence[_ACT],
    instances: Sequence[_INST],
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
    timepoints: Sequence[_TP],
    activities: Sequence[_ACT],
    instances: Sequence[_INST],
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

    # Deduplicate timepoints for x-axis (PTCV-73) and add day offset (PTCV-72)
    unique_tps, _ = _dedup_timepoints(timepoints)
    sorted_tps = sorted(unique_tps, key=lambda t: t.day_offset)
    visit_labels = [
        f"{tp.visit_name}<br><sub>Day {tp.day_offset}</sub>"
        for tp in sorted_tps
    ]
    visit_indices = list(range(len(visit_labels)))

    # Swimlane row positions (bottom to top: Other=0, Imaging=1, Specimens=2, Clinical=3)
    lane_y = {lane: i for i, lane in enumerate(reversed(SWIMLANE_ROWS))}

    # Colour per swimlane
    if low_confidence:
        lane_colors = {
            "Intervention": "rgba(200,180,160,0.6)",
            "Clinical Encounter": "rgba(180,180,200,0.6)",
            "Specimens": "rgba(180,200,180,0.6)",
            "Imaging": "rgba(200,180,180,0.6)",
            "Other": "rgba(190,190,190,0.6)",
        }
    else:
        lane_colors = {
            "Intervention": "#FFA15A",
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
    timepoints: Sequence[_TP],
    activities: Sequence[_ACT],
    instances: Sequence[_INST],
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
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=400)
            st.download_button(
                label="Download as PNG",
                data=png_bytes,
                file_name=f"schedule_of_visits_{registry_id}.png",
                mime="image/png",
            )
        except (ValueError, ImportError, OSError):
            st.button(
                "Download as PNG",
                disabled=True,
                help="Install kaleido for PNG export: pip install kaleido",
            )
