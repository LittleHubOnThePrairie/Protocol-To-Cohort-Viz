"""Tests for SoVPlotData filtered view (PTCV-76).

Verifies:
  - to_plot_data() strips ALCOA+ traceability fields
  - Plot types carry only the 9 plotting-relevant fields
  - build_sov_grid() produces identical output with plot types
  - Full USDM objects are not mutated by conversion
"""

from __future__ import annotations

import pytest

try:
    import plotly  # noqa: F401
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ptcv.soa_extractor.models import (
    UsdmActivity,
    UsdmScheduledInstance,
    UsdmTimepoint,
)
from ptcv.ui.components.schedule_of_visits import (
    PlotActivity,
    PlotInstance,
    PlotTimepoint,
    SoVPlotData,
    build_sov_csv,
    build_sov_figure,
    build_sov_grid,
    to_plot_data,
)


# ---------------------------------------------------------------------------
# Fixtures — full USDM entities with traceability fields
# ---------------------------------------------------------------------------

def _tp(
    tp_id: str, visit_name: str, day_offset: int = 0,
) -> UsdmTimepoint:
    return UsdmTimepoint(
        run_id="r1", source_run_id="sr1", source_sha256="sha256abc",
        registry_id="NCT001", timepoint_id=tp_id, epoch_id="ep-1",
        visit_name=visit_name, visit_type="Treatment",
        day_offset=day_offset, window_early=0, window_late=0,
        mandatory=True, extraction_timestamp_utc="2026-01-01T00:00:00Z",
    )


def _act(act_id: str, name: str, act_type: str) -> UsdmActivity:
    return UsdmActivity(
        run_id="r1", source_run_id="sr1", source_sha256="sha256abc",
        registry_id="NCT001", activity_id=act_id,
        activity_name=name, activity_type=act_type,
        extraction_timestamp_utc="2026-01-01T00:00:00Z",
    )


def _inst(act_id: str, tp_id: str) -> UsdmScheduledInstance:
    return UsdmScheduledInstance(
        run_id="r1", source_run_id="sr1", source_sha256="sha256abc",
        registry_id="NCT001", instance_id=f"si-{act_id}-{tp_id}",
        activity_id=act_id, timepoint_id=tp_id, scheduled=True,
        extraction_timestamp_utc="2026-01-01T00:00:00Z",
    )


TIMEPOINTS = [
    _tp("v1", "Screening", day_offset=-14),
    _tp("v2", "C1D1", day_offset=1),
    _tp("v3", "EOT", day_offset=84),
]

ACTIVITIES = [
    _act("a1", "Vital Signs", "Vital Signs"),
    _act("a2", "CBC", "Lab"),
    _act("a3", "MRI Brain", "Imaging"),
    _act("a4", "ECG Holter", "Procedure"),
]

INSTANCES = [
    _inst("a1", "v1"), _inst("a2", "v1"), _inst("a3", "v1"),
    _inst("a1", "v2"), _inst("a2", "v2"), _inst("a4", "v2"),
    _inst("a1", "v3"), _inst("a3", "v3"),
]

# ALCOA+ fields that must NOT appear on plot types
ALCOA_FIELDS = {"run_id", "source_run_id", "source_sha256"}


# ---------------------------------------------------------------------------
# Scenario: Plotting receives only relevant fields
# ---------------------------------------------------------------------------

class TestPlotTypesHaveOnlyRelevantFields:
    """Plot NamedTuples carry exactly the fields needed for visualization."""

    def test_plot_timepoint_fields(self) -> None:
        pt = PlotTimepoint("v1", "Screening", -14)
        assert set(pt._fields) == {"timepoint_id", "visit_name", "day_offset"}

    def test_plot_activity_fields(self) -> None:
        pa = PlotActivity("a1", "Vital Signs", "Vital Signs")
        assert set(pa._fields) == {
            "activity_id", "activity_name", "activity_type",
        }

    def test_plot_instance_fields(self) -> None:
        pi = PlotInstance("a1", "v1", True)
        assert set(pi._fields) == {
            "activity_id", "timepoint_id", "scheduled",
        }

    def test_no_alcoa_fields_on_plot_timepoint(self) -> None:
        pt = PlotTimepoint("v1", "Screening", -14)
        for field in ALCOA_FIELDS:
            assert not hasattr(pt, field), (
                f"PlotTimepoint should not have {field}"
            )

    def test_no_alcoa_fields_on_plot_activity(self) -> None:
        pa = PlotActivity("a1", "Vital Signs", "Vital Signs")
        for field in ALCOA_FIELDS:
            assert not hasattr(pa, field), (
                f"PlotActivity should not have {field}"
            )

    def test_no_alcoa_fields_on_plot_instance(self) -> None:
        pi = PlotInstance("a1", "v1", True)
        for field in ALCOA_FIELDS:
            assert not hasattr(pi, field), (
                f"PlotInstance should not have {field}"
            )


# ---------------------------------------------------------------------------
# Scenario: to_plot_data conversion
# ---------------------------------------------------------------------------

class TestToPlotData:
    """to_plot_data() extracts only plotting fields from full USDM objects."""

    def test_returns_sov_plot_data(self) -> None:
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert isinstance(pd, SoVPlotData)

    def test_timepoint_count_preserved(self) -> None:
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert len(pd.timepoints) == len(TIMEPOINTS)

    def test_activity_count_preserved(self) -> None:
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert len(pd.activities) == len(ACTIVITIES)

    def test_instance_count_preserved(self) -> None:
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert len(pd.instances) == len(INSTANCES)

    def test_timepoint_values_match(self) -> None:
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        for orig, plot in zip(TIMEPOINTS, pd.timepoints):
            assert plot.timepoint_id == orig.timepoint_id
            assert plot.visit_name == orig.visit_name
            assert plot.day_offset == orig.day_offset

    def test_activity_values_match(self) -> None:
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        for orig, plot in zip(ACTIVITIES, pd.activities):
            assert plot.activity_id == orig.activity_id
            assert plot.activity_name == orig.activity_name
            assert plot.activity_type == orig.activity_type

    def test_instance_values_match(self) -> None:
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        for orig, plot in zip(INSTANCES, pd.instances):
            assert plot.activity_id == orig.activity_id
            assert plot.timepoint_id == orig.timepoint_id
            assert plot.scheduled == orig.scheduled

    def test_empty_inputs(self) -> None:
        pd = to_plot_data([], [], [])
        assert pd.timepoints == []
        assert pd.activities == []
        assert pd.instances == []


# ---------------------------------------------------------------------------
# Scenario: Full USDM objects remain available downstream
# ---------------------------------------------------------------------------

class TestUsdmObjectsUnmutated:
    """Conversion does not modify the original USDM objects."""

    def test_timepoints_unchanged_after_conversion(self) -> None:
        original_ids = [tp.run_id for tp in TIMEPOINTS]
        to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert [tp.run_id for tp in TIMEPOINTS] == original_ids

    def test_activities_unchanged_after_conversion(self) -> None:
        original_ids = [a.run_id for a in ACTIVITIES]
        to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert [a.run_id for a in ACTIVITIES] == original_ids

    def test_full_objects_still_have_traceability(self) -> None:
        to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        for tp in TIMEPOINTS:
            assert tp.run_id == "r1"
            assert tp.source_run_id == "sr1"
            assert tp.source_sha256 == "sha256abc"


# ---------------------------------------------------------------------------
# Scenario: Existing SoV output unchanged
# ---------------------------------------------------------------------------

class TestPlotDataProducesSameOutput:
    """build_sov_grid() output is identical with full vs plot types."""

    def test_grid_identical(self) -> None:
        grid_full = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        grid_plot = build_sov_grid(
            pd.timepoints, pd.activities, pd.instances,
        )
        assert grid_full == grid_plot

    def test_csv_identical(self) -> None:
        csv_full = build_sov_csv(TIMEPOINTS, ACTIVITIES, INSTANCES)
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        csv_plot = build_sov_csv(
            pd.timepoints, pd.activities, pd.instances,
        )
        assert csv_full == csv_plot

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_figure_traces_identical(self) -> None:
        fig_full = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001",
        )
        pd = to_plot_data(TIMEPOINTS, ACTIVITIES, INSTANCES)
        fig_plot = build_sov_figure(
            pd.timepoints, pd.activities, pd.instances,
            registry_id="NCT001",
        )
        # Same number of traces
        assert len(fig_full.data) == len(fig_plot.data)
        # Same data points per trace
        for t_full, t_plot in zip(fig_full.data, fig_plot.data):
            assert list(t_full.x) == list(t_plot.x)
            assert list(t_full.y) == list(t_plot.y)
            assert list(t_full.hovertext) == list(t_plot.hovertext)

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_all_five_swimlanes_render(self) -> None:
        """12 timepoints, 25 activities across all swimlanes."""
        tps = [
            _tp(f"t{i}", f"Visit {i}", day_offset=i * 7)
            for i in range(12)
        ]
        acts = []
        insts = []
        types = [
            ("Drug Administration", "Intervention"),
            ("Assessment", "Clinical Encounter"),
            ("Lab", "Specimens"),
            ("Imaging", "Imaging"),
            ("Procedure", "Other"),
        ]
        act_idx = 0
        for i in range(25):
            atype, _ = types[i % len(types)]
            aid = f"act{act_idx}"
            acts.append(_act(aid, f"Activity {aid}", atype))
            insts.append(_inst(aid, f"t{i % 12}"))
            act_idx += 1

        pd = to_plot_data(tps, acts, insts)
        fig = build_sov_figure(
            pd.timepoints, pd.activities, pd.instances,
            registry_id="NCT-12x25",
        )
        trace_names = {t.name for t in fig.data}
        assert trace_names == {
            "Intervention", "Clinical Encounter",
            "Specimens", "Imaging", "Other",
        }
