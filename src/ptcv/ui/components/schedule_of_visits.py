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
    visit_type: str = ""


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
            PlotTimepoint(
                tp.timepoint_id, tp.visit_name, tp.day_offset,
                getattr(tp, "visit_type", ""),
            )
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
    "ECG": "Other",
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
# PTCV-131: eDiary and ECG should be "Other", not "Clinical Encounter".
_OTHER_RE = re.compile(
    r"\be[-.]?diary\b|\belectronic\s+diary\b|\bpatient\s+diary\b"
    r"|\b[Ee][Cc][Gg]\b|\belectrocardiogram\b",
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
        # eDiary/ECG keywords override Clinical Encounter (PTCV-131)
        if lane == "Clinical Encounter" and _OTHER_RE.search(
            activity_name
        ):
            return "Other"

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
# Visit sort key (PTCV-131)
# ---------------------------------------------------------------------------

_VISIT_NUM_SORT_RE = re.compile(r"[Vv](?:isit\s*)?(\d+)")


def _visit_sort_key(tp: Any) -> tuple[int, int, str]:
    """Sort key: (day_offset, visit_number, visit_name).

    Primary sort by day_offset; visit number tiebreaks when
    multiple visits share the same day_offset.
    """
    m = _VISIT_NUM_SORT_RE.search(tp.visit_name)
    num = int(m.group(1)) if m else 0
    return (tp.day_offset, num, tp.visit_name)


# ---------------------------------------------------------------------------
# Screening-anchored day rebasing (PTCV-151)
# ---------------------------------------------------------------------------

def _rebase_to_screening(
    sorted_tps: Sequence[_TP],
) -> dict[str, int]:
    """Map timepoint_id to display day with screening anchored at day 0.

    Finds the minimum day_offset (the screening visit) and subtracts it
    from all offsets so that screening becomes day 0 and all subsequent
    visits have positive offsets.

    Args:
        sorted_tps: Timepoints (need not be sorted; min is computed).

    Returns:
        Dict mapping timepoint_id to rebased display day.
    """
    if not sorted_tps:
        return {}
    min_day = min(tp.day_offset for tp in sorted_tps)
    return {tp.timepoint_id: tp.day_offset - min_day for tp in sorted_tps}


# ---------------------------------------------------------------------------
# Grid builder (pure Python) — PTCV-151: assessment × visit matrix
# ---------------------------------------------------------------------------

def build_sov_grid(
    timepoints: Sequence[_TP],
    activities: Sequence[_ACT],
    instances: Sequence[_INST],
) -> list[dict[str, Any]]:
    """Build a flat assessment × visit matrix.

    Each record represents one scheduled (assessment, visit) pair.
    No assessments are concatenated — each gets its own row.

    Duplicate timepoints (same visit_name + day_offset) are merged so
    their activities appear in a single column (PTCV-73).

    Day offsets are rebased so Screening = day 0 (PTCV-151).

    Args:
        timepoints: USDM timepoints (visits) in protocol order.
        activities: USDM activities extracted from the SoA.
        instances: Scheduled instances linking activities to timepoints.

    Returns:
        List of dicts with keys: assessment, assessment_type, visit_label,
        day, visit_index, activity_id.
    """
    # Deduplicate timepoints (PTCV-73)
    unique_tps, tp_id_map = _dedup_timepoints(timepoints)

    # Index lookups — use unique timepoints only
    tp_by_id = {tp.timepoint_id: tp for tp in unique_tps}
    act_by_id = {a.activity_id: a for a in activities}

    # PTCV-254: Resolve visit labels to real study days for
    # chronological ordering.  Uses resolver for labels that parse
    # to a confident day; keeps existing day_offset otherwise.
    resolved_days: dict[str, int] = {}
    try:
        from ptcv.soa_extractor.visit_day_resolver import (
            VisitDayResolver,
        )
        resolver = VisitDayResolver()
        for tp in unique_tps:
            rv = resolver.resolve(tp.visit_name)
            if rv.confidence >= 0.8:
                resolved_days[tp.timepoint_id] = rv.real_day
            # else: keep day_offset (handled by fallback below)
    except ImportError:
        pass  # Fallback to day_offset if resolver not available

    def _resolved_sort_key(tp: Any) -> tuple[int, int, str]:
        """Sort by resolved real_day, falling back to day_offset."""
        day = resolved_days.get(tp.timepoint_id, tp.day_offset)
        m = _VISIT_NUM_SORT_RE.search(tp.visit_name)
        num = int(m.group(1)) if m else 0
        return (day, num, tp.visit_name)

    # Sort unique timepoints by resolved day for x-axis ordering
    sorted_tps = sorted(unique_tps, key=_resolved_sort_key)
    tp_order = {tp.timepoint_id: idx for idx, tp in enumerate(sorted_tps)}

    # Use resolved days for display; fall back to rebased day_offset
    rebased_days = _rebase_to_screening(unique_tps)
    # Override with resolved real days where available
    for tp_id, real_day in resolved_days.items():
        rebased_days[tp_id] = real_day

    # Build one record per scheduled (assessment, visit) pair
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()  # (activity_id, canonical_tp_id)

    for inst in instances:
        if not inst.scheduled:
            continue
        act = act_by_id.get(inst.activity_id)
        canonical_tp_id = tp_id_map.get(inst.timepoint_id)
        if act is None or canonical_tp_id is None:
            continue
        tp = tp_by_id.get(canonical_tp_id)
        if tp is None:
            continue
        # Deduplicate: same activity at same canonical timepoint
        key = (act.activity_id, canonical_tp_id)
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "assessment": act.activity_name,
            "assessment_type": classify_swimlane(
                act.activity_type, act.activity_name,
            ),
            "visit_label": tp.visit_name,
            "day": rebased_days.get(canonical_tp_id, tp.day_offset),
            "visit_index": tp_order.get(canonical_tp_id, 0),
            "activity_id": act.activity_id,
        })

    # Sort by visit order, then assessment type, then assessment name
    lane_order = {lane: i for i, lane in enumerate(SWIMLANE_ROWS)}
    records.sort(key=lambda r: (
        r["visit_index"],
        lane_order.get(r["assessment_type"], 99),
        r["assessment"],
    ))
    return records


def build_sov_csv(
    timepoints: Sequence[_TP],
    activities: Sequence[_ACT],
    instances: Sequence[_INST],
) -> str:
    """Build a CSV string from the SoA assessment × visit matrix.

    Columns: Assessment, Assessment Type, Visit, Day

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
    writer.writerow(["Assessment", "Assessment Type", "Visit", "Day"])
    for row in grid:
        writer.writerow([
            row["assessment"],
            row["assessment_type"],
            row["visit_label"],
            row["day"],
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
    """Build a Plotly Figure for the assessment × visit schedule chart.

    Y-axis shows individual assessments grouped by assessment type.
    X-axis shows visits ordered by day offset (screening = day 0).
    Markers indicate which assessments are scheduled at which visits.

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

    # Build x-axis: unique visits ordered by visit_index
    visit_map: dict[int, tuple[str, int]] = {}  # index -> (label, day)
    for rec in grid:
        vi = rec["visit_index"]
        if vi not in visit_map:
            visit_map[vi] = (rec["visit_label"], rec["day"])
    sorted_visits = sorted(visit_map.items())
    visit_indices = [vi for vi, _ in sorted_visits]
    visit_labels = [
        f"{label}<br><sub>Day {day}</sub>"
        for _, (label, day) in sorted_visits
    ]

    # Build y-axis: individual assessments grouped by assessment_type
    type_assessments: dict[str, list[str]] = {}
    for rec in grid:
        atype = rec["assessment_type"]
        aname = rec["assessment"]
        if atype not in type_assessments:
            type_assessments[atype] = []
        if aname not in type_assessments[atype]:
            type_assessments[atype].append(aname)

    for atype in type_assessments:
        type_assessments[atype].sort()

    # Assign y-positions with gaps between type groups
    assessment_y: dict[str, float] = {}
    y_labels: list[str] = []
    y_vals: list[float] = []
    y_pos = 0.0
    separator_ys: list[float] = []

    for lane in SWIMLANE_ROWS:
        names = type_assessments.get(lane, [])
        if not names:
            continue
        if y_pos > 0:
            separator_ys.append(y_pos - 0.5)
        for name in names:
            assessment_y[name] = y_pos
            y_labels.append(name)
            y_vals.append(y_pos)
            y_pos += 1
        y_pos += 0.5

    # Colour per assessment type
    if low_confidence:
        lane_colors: dict[str, str] = {
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

    # Build traces per assessment_type for legend grouping
    traces_by_type: dict[str, dict[str, list]] = {
        lane: {"x": [], "y": [], "text": []}
        for lane in SWIMLANE_ROWS
    }

    for rec in grid:
        atype = rec["assessment_type"]
        y = assessment_y.get(rec["assessment"])
        if y is None:
            continue
        traces_by_type[atype]["x"].append(rec["visit_index"])
        traces_by_type[atype]["y"].append(y)
        traces_by_type[atype]["text"].append(
            f"{rec['assessment']}<br>{rec['visit_label']} (Day {rec['day']})"
        )

    fig = go.Figure()
    for lane in SWIMLANE_ROWS:
        data = traces_by_type[lane]
        if not data["x"]:
            continue
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name=lane,
            marker=dict(
                size=12,
                color=lane_colors.get(lane, "#888"),
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            hovertext=data["text"],
            hoverinfo="text",
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    title = "Schedule of Visits"
    if registry_id:
        title = f"Schedule of Visits — {registry_id}"

    num_assessments = len(y_labels)
    chart_height = max(400, 30 * num_assessments)

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
            tickvals=y_vals,
            ticktext=y_labels,
            title="",
            autorange="reversed",
        ),
        height=chart_height,
        margin=dict(l=200, r=20, t=50, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )

    # Separator lines between assessment type groups
    for sy in separator_ys:
        fig.add_hline(
            y=sy, line_dash="dot",
            line_color="lightgrey", line_width=1,
        )

    # Vertical gridlines
    for vi in visit_indices:
        fig.add_vline(
            x=vi - 0.5, line_dash="dot",
            line_color="lightgrey", line_width=0.5,
        )

    return fig


# ---------------------------------------------------------------------------
# Matrix-based chart builder (PTCV-151)
# ---------------------------------------------------------------------------

def build_sov_figure_from_matrix(
    timepoints: Sequence[_TP],
    activities: Sequence[_ACT],
    instances: Sequence[_INST],
    registry_id: str = "",
    low_confidence: bool = False,
) -> Any:
    """Build assessment x visit Plotly chart via matrix_builder.

    Uses matrix_builder to produce assessment-level rows, then renders
    each (assessment, visit) as a scatter marker.  Falls back to
    :func:`build_sov_figure` if the matrix is empty.
    """
    import plotly.graph_objects as go

    # Lazy import to avoid circular dependency
    from ptcv.soa_extractor.matrix_builder import (
        build_assessment_matrix_from_usdm,
    )

    df = build_assessment_matrix_from_usdm(
        timepoints,  # type: ignore[arg-type]
        activities,  # type: ignore[arg-type]
        instances,  # type: ignore[arg-type]
        anchor_screening=True,
    )

    if df.empty:
        return build_sov_figure(
            timepoints, activities, instances,
            registry_id=registry_id,
            low_confidence=low_confidence,
        )

    # Determine type ordering: SWIMLANE_ROWS first, extras after
    all_types = list(df["assessment_type"].unique())
    ordered_types: list[str] = [
        t for t in SWIMLANE_ROWS if t in all_types
    ]
    for t in all_types:
        if t not in ordered_types:
            ordered_types.append(t)

    # Y-axis: assessments grouped by type
    assessment_y: dict[str, float] = {}
    y_labels: list[str] = []
    y_vals: list[float] = []
    separator_ys: list[float] = []
    y_pos = 0.0

    for atype in ordered_types:
        names = sorted(
            df.loc[
                df["assessment_type"] == atype, "assessment_name"
            ].unique(),
        )
        if not names:
            continue
        if y_pos > 0:
            separator_ys.append(y_pos - 0.5)
        for name in names:
            assessment_y[name] = y_pos
            y_labels.append(name)
            y_vals.append(y_pos)
            y_pos += 1
        y_pos += 0.5

    # X-axis: visits sorted by day_offset
    visit_info = (
        df[["visit_label", "day_offset"]]
        .drop_duplicates()
        .sort_values("day_offset")
    )
    visit_labels_list: list[str] = []
    visit_x_map: dict[str, int] = {}
    for i, (_, row) in enumerate(visit_info.iterrows()):
        label = row["visit_label"]
        day = int(row["day_offset"])
        visit_labels_list.append(f"{label}<br><sub>Day {day}</sub>")
        visit_x_map[label] = i
    visit_x_vals = list(range(len(visit_labels_list)))

    # Colour palette
    if low_confidence:
        lane_colors: dict[str, str] = {
            "Intervention": "rgba(200,180,160,0.6)",
            "Clinical Encounter": "rgba(180,180,200,0.6)",
            "Specimens": "rgba(180,200,180,0.6)",
            "Imaging": "rgba(200,180,180,0.6)",
            "At-home": "rgba(160,200,200,0.6)",
            "Other": "rgba(190,190,190,0.6)",
        }
    else:
        lane_colors = {
            "Intervention": "#FFA15A",
            "Clinical Encounter": "#636EFA",
            "Specimens": "#EF553B",
            "Imaging": "#00CC96",
            "At-home": "#19D3F3",
            "Other": "#AB63FA",
        }

    # Build traces per type
    traces_by_type: dict[str, dict[str, list]] = {
        t: {"x": [], "y": [], "text": []} for t in ordered_types
    }

    scheduled = df[df["scheduled"]] if "scheduled" in df.columns else df
    for _, row in scheduled.iterrows():
        atype = row["assessment_type"]
        aname = row["assessment_name"]
        vlabel = row["visit_label"]
        day = int(row["day_offset"])
        y = assessment_y.get(aname)
        x = visit_x_map.get(vlabel)
        if y is None or x is None:
            continue
        traces_by_type[atype]["x"].append(x)
        traces_by_type[atype]["y"].append(y)
        traces_by_type[atype]["text"].append(
            f"{aname}<br>{vlabel} (Day {day})",
        )

    fig = go.Figure()
    for atype in ordered_types:
        data = traces_by_type[atype]
        if not data["x"]:
            continue
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name=atype,
            marker=dict(
                size=14,
                color=lane_colors.get(atype, "#888"),
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            hovertext=data["text"],
            hoverinfo="text",
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    title = "Schedule of Visits"
    if registry_id:
        title = f"Schedule of Visits — {registry_id}"

    n_assessments = len(y_labels)
    chart_height = max(350, 50 + n_assessments * 22)

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=visit_x_vals,
            ticktext=visit_labels_list,
            title="Visit",
            tickangle=-45,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_vals,
            ticktext=y_labels,
            title="",
            autorange="reversed",
        ),
        height=chart_height,
        margin=dict(l=200, r=20, t=50, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )

    for sy in separator_ys:
        fig.add_hline(
            y=sy, line_dash="dot",
            line_color="lightgrey", line_width=1,
        )
    for vi in visit_x_vals:
        fig.add_vline(
            x=vi - 0.5, line_dash="dot",
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

    fig = build_sov_figure_from_matrix(
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
