"""Unit tests for schedule of visits swimlane component (PTCV-34, PTCV-71).

Tests the pure-Python helpers (classify_swimlane, build_sov_grid,
build_sov_csv, build_sov_figure) without Streamlit runtime.

Covers GHERKIN scenarios:
  - Clinical encounter row lists clinician-collected variables
  - Specimen row lists specimen types and derived assays
  - Imaging row lists imaging modality
  - Non-ICH protocol produces empty/minimal schedule
  - CSV download has correct structure
  - Intervention swimlane displays drug administration (PTCV-71)
  - Specimen-based tests classified correctly (PTCV-71)
  - Existing Clinical Encounter activities unchanged (PTCV-71)
  - Backward-compatible classify_swimlane signature (PTCV-71)
"""

from __future__ import annotations

import csv
import io

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
    SWIMLANE_ROWS,
    _dedup_timepoints,
    build_sov_csv,
    build_sov_figure,
    build_sov_grid,
    classify_swimlane,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal USDM entities
# ---------------------------------------------------------------------------

def _tp(
    tp_id: str, visit_name: str, day_offset: int = 0,
) -> UsdmTimepoint:
    return UsdmTimepoint(
        run_id="r1", source_run_id="", source_sha256="",
        registry_id="NCT001", timepoint_id=tp_id, epoch_id="ep-1",
        visit_name=visit_name, visit_type="Treatment",
        day_offset=day_offset, window_early=0, window_late=0,
        mandatory=True, extraction_timestamp_utc="2026-01-01T00:00:00Z",
    )


def _act(act_id: str, name: str, act_type: str) -> UsdmActivity:
    return UsdmActivity(
        run_id="r1", source_run_id="", source_sha256="",
        registry_id="NCT001", activity_id=act_id,
        activity_name=name, activity_type=act_type,
        extraction_timestamp_utc="2026-01-01T00:00:00Z",
    )


def _inst(act_id: str, tp_id: str) -> UsdmScheduledInstance:
    return UsdmScheduledInstance(
        run_id="r1", source_run_id="", source_sha256="",
        registry_id="NCT001", instance_id=f"si-{act_id}-{tp_id}",
        activity_id=act_id, timepoint_id=tp_id, scheduled=True,
        extraction_timestamp_utc="2026-01-01T00:00:00Z",
    )


# Standard test data: 3 visits, 4 activities across all swimlanes
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
    # Screening: vitals + CBC + MRI
    _inst("a1", "v1"), _inst("a2", "v1"), _inst("a3", "v1"),
    # C1D1: vitals + CBC + ECG Holter
    _inst("a1", "v2"), _inst("a2", "v2"), _inst("a4", "v2"),
    # EOT: vitals + MRI
    _inst("a1", "v3"), _inst("a3", "v3"),
]


# ---------------------------------------------------------------------------
# classify_swimlane
# ---------------------------------------------------------------------------

class TestClassifySwimlane:
    """Tests for classify_swimlane()."""

    @pytest.mark.parametrize("activity_type,expected", [
        ("Drug Administration", "Intervention"),
        ("Treatment", "Intervention"),
        ("Assessment", "Clinical Encounter"),
        ("Vital Signs", "Clinical Encounter"),
        ("ECG", "Clinical Encounter"),
        ("Safety", "Clinical Encounter"),
        ("Consent", "Clinical Encounter"),
        ("Lab", "Specimens"),
        ("Pharmacokinetics", "Specimens"),
        ("Imaging", "Imaging"),
        ("Procedure", "Other"),
        ("Other", "Other"),
    ])
    def test_known_types(self, activity_type: str, expected: str) -> None:
        assert classify_swimlane(activity_type) == expected

    def test_unknown_type_defaults_to_other(self) -> None:
        assert classify_swimlane("SomethingNew") == "Other"

    def test_swimlane_rows_has_five_entries(self) -> None:
        """PTCV-71: 5 swimlane rows including Intervention."""
        assert len(SWIMLANE_ROWS) == 5
        assert set(SWIMLANE_ROWS) == {
            "Intervention", "Clinical Encounter", "Specimens",
            "Imaging", "Other",
        }

    def test_intervention_is_first_row(self) -> None:
        """PTCV-71: Intervention swimlane is the topmost row."""
        assert SWIMLANE_ROWS[0] == "Intervention"

    def test_backward_compatible_no_name(self) -> None:
        """PTCV-71: classify_swimlane(type) still works without name."""
        assert classify_swimlane("Assessment") == "Clinical Encounter"
        assert classify_swimlane("Lab") == "Specimens"


class TestKeywordReclassification:
    """PTCV-71: Keyword-based reclassification via activity_name."""

    @pytest.mark.parametrize("name", [
        "Study drug administration",
        "Placebo infusion",
        "IMP dosing",
        "IV injection of drug",
        "Drug administration Day 1",
        "Treatment administration",
    ])
    def test_intervention_keywords(self, name: str) -> None:
        """Activities with intervention keywords → Intervention."""
        assert classify_swimlane("Other", name) == "Intervention"
        assert classify_swimlane("Procedure", name) == "Intervention"

    @pytest.mark.parametrize("name,expected", [
        ("Urine pregnancy test", "Specimens"),
        ("HepB/HepC serology", "Specimens"),
        ("CBC", "Specimens"),
        ("Blood draw", "Specimens"),
        ("Biomarker assay", "Specimens"),
        ("Anti-drug antibody", "Specimens"),
        ("PCR test", "Specimens"),
        ("Throat culture", "Specimens"),
        ("Haematology panel", "Specimens"),
        ("Chemistry panel", "Specimens"),
        ("Coagulation tests", "Specimens"),
        ("Urinalysis", "Specimens"),
        ("Blood sample collection", "Specimens"),
        ("Serum creatinine", "Specimens"),
        ("Plasma concentration", "Specimens"),
        ("Hepatitis B screening", "Specimens"),
    ])
    def test_specimen_keywords(self, name: str, expected: str) -> None:
        """Activities with specimen keywords → Specimens."""
        assert classify_swimlane("Assessment", name) == expected

    def test_specimen_keyword_does_not_override_lab(self) -> None:
        """Lab type already maps to Specimens — no change needed."""
        assert classify_swimlane("Lab", "CBC") == "Specimens"

    def test_specimen_keyword_does_not_override_imaging(self) -> None:
        """Imaging type is not overridden by specimen keywords."""
        assert classify_swimlane("Imaging", "Blood sample MRI") == "Imaging"

    def test_intervention_does_not_override_imaging(self) -> None:
        """Imaging type is not overridden by intervention keywords."""
        assert classify_swimlane("Imaging", "Study drug imaging") == "Imaging"

    def test_clinical_encounter_unchanged_without_keywords(self) -> None:
        """Vital Signs 'Blood pressure' stays Clinical Encounter."""
        assert classify_swimlane("Vital Signs", "Blood pressure") == "Clinical Encounter"

    def test_intervention_overrides_clinical_encounter(self) -> None:
        """An Assessment named 'Study drug' → Intervention."""
        assert classify_swimlane("Assessment", "Study drug dose") == "Intervention"


# ---------------------------------------------------------------------------
# build_sov_grid
# ---------------------------------------------------------------------------

class TestBuildSovGrid:
    """Tests for build_sov_grid()."""

    def test_grid_has_expected_cells(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        # Should have cells for each (visit, swimlane) combo that has data
        assert len(grid) > 0
        # All records have required keys
        for rec in grid:
            assert "visit_name" in rec
            assert "swimlane" in rec
            assert "activities" in rec
            assert "activity_count" in rec

    def test_clinical_encounter_row_for_vitals(self) -> None:
        """[PTCV-34 Scenario: Clinical encounter row lists variables]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        clinical_cells = [
            r for r in grid if r["swimlane"] == "Clinical Encounter"
        ]
        # Vitals scheduled at all 3 visits
        assert len(clinical_cells) == 3
        for cell in clinical_cells:
            assert "Vital Signs" in cell["activities"]

    def test_specimen_row_for_labs(self) -> None:
        """[PTCV-34 Scenario: Specimen row lists specimen types]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        specimen_cells = [
            r for r in grid if r["swimlane"] == "Specimens"
        ]
        # CBC scheduled at Screening + C1D1
        assert len(specimen_cells) == 2
        for cell in specimen_cells:
            assert "CBC" in cell["activities"]

    def test_imaging_row_for_mri(self) -> None:
        """[PTCV-34 Scenario: Imaging row lists imaging modality]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        imaging_cells = [
            r for r in grid if r["swimlane"] == "Imaging"
        ]
        # MRI Brain at Screening + EOT
        assert len(imaging_cells) == 2
        for cell in imaging_cells:
            assert "MRI Brain" in cell["activities"]

    def test_other_row_for_procedure(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        other_cells = [
            r for r in grid if r["swimlane"] == "Other"
        ]
        # ECG Holter at C1D1
        assert len(other_cells) == 1
        assert "ECG Holter" in other_cells[0]["activities"]

    def test_grid_sorted_by_visit_then_swimlane(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        visit_indices = [r["visit_index"] for r in grid]
        assert visit_indices == sorted(visit_indices)

    def test_empty_instances_produces_empty_grid(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, [])
        assert grid == []

    def test_empty_timepoints_produces_empty_grid(self) -> None:
        grid = build_sov_grid([], ACTIVITIES, INSTANCES)
        assert grid == []

    def test_multiple_activities_same_cell(self) -> None:
        """Two activities in the same swimlane at the same visit."""
        acts = [
            _act("a10", "Physical Exam", "Assessment"),
            _act("a11", "ECOG Score", "Assessment"),
        ]
        insts = [_inst("a10", "v1"), _inst("a11", "v1")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        clinical_v1 = [
            r for r in grid
            if r["swimlane"] == "Clinical Encounter"
            and r["visit_name"] == "Screening"
        ]
        assert len(clinical_v1) == 1
        assert clinical_v1[0]["activity_count"] == 2
        assert "ECOG Score" in clinical_v1[0]["activities"]
        assert "Physical Exam" in clinical_v1[0]["activities"]


# ---------------------------------------------------------------------------
# build_sov_csv
# ---------------------------------------------------------------------------

class TestBuildSovCsv:
    """Tests for build_sov_csv()."""

    def test_csv_has_header(self) -> None:
        """[PTCV-34 Scenario: CSV download has correct structure]"""
        csv_str = build_sov_csv(TIMEPOINTS, ACTIVITIES, INSTANCES)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header == ["Visit", "Day", "Category", "Activities"]

    def test_csv_row_count_matches_grid(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        csv_str = build_sov_csv(TIMEPOINTS, ACTIVITIES, INSTANCES)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        # header + data rows
        assert len(rows) == 1 + len(grid)

    def test_csv_contains_visit_names(self) -> None:
        csv_str = build_sov_csv(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert "Screening" in csv_str
        assert "C1D1" in csv_str
        assert "EOT" in csv_str

    def test_empty_data_returns_header_only(self) -> None:
        csv_str = build_sov_csv([], [], [])
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0] == ["Visit", "Day", "Category", "Activities"]


# ---------------------------------------------------------------------------
# build_sov_figure
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestBuildSovFigure:
    """Tests for build_sov_figure() — Plotly chart builder."""

    def test_figure_has_traces(self) -> None:
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001"
        )
        assert len(fig.data) > 0

    def test_figure_title_contains_registry_id(self) -> None:
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT00112827"
        )
        assert "NCT00112827" in fig.layout.title.text

    def test_empty_grid_shows_annotation(self) -> None:
        """[PTCV-34 Scenario: Non-ICH produces empty schedule]"""
        fig = build_sov_figure([], [], [])
        assert len(fig.layout.annotations) > 0
        assert "No schedule" in fig.layout.annotations[0].text

    def test_low_confidence_flag_changes_colours(self) -> None:
        fig_normal = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, low_confidence=False,
        )
        fig_low = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, low_confidence=True,
        )
        # Colours should differ between the two
        normal_colors = {t.marker.color for t in fig_normal.data}
        low_colors = {t.marker.color for t in fig_low.data}
        assert normal_colors != low_colors

    def test_figure_xaxis_has_visit_labels(self) -> None:
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001"
        )
        ticktext = fig.layout.xaxis.ticktext
        joined = " ".join(ticktext)
        assert "Screening" in joined
        assert "C1D1" in joined
        assert "EOT" in joined

    def test_figure_xaxis_has_day_offsets(self) -> None:
        """PTCV-72: X-axis labels include day offsets."""
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001"
        )
        ticktext = fig.layout.xaxis.ticktext
        joined = " ".join(ticktext)
        assert "Day -14" in joined  # Screening
        assert "Day 1" in joined    # C1D1
        assert "Day 84" in joined   # EOT

    def test_figure_yaxis_has_swimlane_labels(self) -> None:
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001"
        )
        ticktext = fig.layout.yaxis.ticktext
        assert "Intervention" in ticktext
        assert "Clinical Encounter" in ticktext
        assert "Specimens" in ticktext
        assert "Imaging" in ticktext
        assert "Other" in ticktext


# ---------------------------------------------------------------------------
# PTCV-71: Intervention swimlane grid integration
# ---------------------------------------------------------------------------

class TestInterventionSwimlaneGrid:
    """PTCV-71: Drug administration appears in Intervention swimlane."""

    def test_drug_admin_type_in_intervention(self) -> None:
        """Activity type 'Drug Administration' → Intervention row."""
        acts = [_act("a20", "Study Drug", "Drug Administration")]
        insts = [_inst("a20", "v2")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        intervention = [r for r in grid if r["swimlane"] == "Intervention"]
        assert len(intervention) == 1
        assert "Study Drug" in intervention[0]["activities"]

    def test_keyword_reclassification_in_grid(self) -> None:
        """Activity with name 'Placebo infusion' reclassified in grid."""
        acts = [_act("a21", "Placebo infusion", "Procedure")]
        insts = [_inst("a21", "v1")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        intervention = [r for r in grid if r["swimlane"] == "Intervention"]
        assert len(intervention) == 1
        assert "Placebo infusion" in intervention[0]["activities"]

    def test_specimen_reclassification_in_grid(self) -> None:
        """Activity 'Urine pregnancy test' (Assessment) → Specimens."""
        acts = [_act("a22", "Urine pregnancy test", "Assessment")]
        insts = [_inst("a22", "v1")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        specimens = [r for r in grid if r["swimlane"] == "Specimens"]
        assert len(specimens) == 1
        assert "Urine pregnancy test" in specimens[0]["activities"]


# ---------------------------------------------------------------------------
# PTCV-73: Timepoint deduplication
# ---------------------------------------------------------------------------

class TestTimepointDedup:
    """PTCV-73: Duplicate timepoints collapsed on x-axis."""

    def test_dedup_merges_same_name_and_offset(self) -> None:
        """25 timepoints with same name+offset → 1 unique."""
        tps = [
            _tp(f"dup-{i}", "Screening", day_offset=7)
            for i in range(25)
        ]
        unique, id_map = _dedup_timepoints(tps)
        assert len(unique) == 1
        assert unique[0].visit_name == "Screening"
        # All map to the same canonical id
        assert len(set(id_map.values())) == 1

    def test_dedup_preserves_distinct_visits(self) -> None:
        """Distinct (name, offset) pairs remain separate."""
        unique, id_map = _dedup_timepoints(TIMEPOINTS)
        assert len(unique) == 3  # Screening, C1D1, EOT

    def test_dedup_same_name_different_offset(self) -> None:
        """Same name but different day offsets stay separate."""
        tps = [
            _tp("t1", "Visit", day_offset=1),
            _tp("t2", "Visit", day_offset=14),
        ]
        unique, _ = _dedup_timepoints(tps)
        assert len(unique) == 2

    def test_grid_merges_duplicate_timepoint_activities(self) -> None:
        """Activities from duplicate timepoints merge into one column."""
        # 3 duplicate Screening timepoints, each with a different activity
        tps = [
            _tp("dup-1", "Screening", day_offset=7),
            _tp("dup-2", "Screening", day_offset=7),
            _tp("dup-3", "Screening", day_offset=7),
        ]
        acts = [
            _act("a30", "Vital Signs", "Vital Signs"),
            _act("a31", "CBC", "Lab"),
            _act("a32", "MRI Brain", "Imaging"),
        ]
        insts = [
            _inst("a30", "dup-1"),
            _inst("a31", "dup-2"),
            _inst("a32", "dup-3"),
        ]
        grid = build_sov_grid(tps, acts, insts)
        # All activities collapsed under one "Screening" visit
        visit_names = {r["visit_name"] for r in grid}
        assert visit_names == {"Screening"}
        # 3 swimlane rows (Clinical Encounter, Specimens, Imaging)
        swimlanes = {r["swimlane"] for r in grid}
        assert swimlanes == {"Clinical Encounter", "Specimens", "Imaging"}

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_figure_xaxis_deduped(self) -> None:
        """Figure x-axis shows 1 column, not 25, for duplicate timepoints."""
        tps = [
            _tp(f"dup-{i}", "Screening", day_offset=7)
            for i in range(25)
        ]
        acts = [_act("a40", "Vital Signs", "Vital Signs")]
        insts = [_inst("a40", "dup-0")]
        fig = build_sov_figure(tps, acts, insts)
        assert len(fig.layout.xaxis.ticktext) == 1
        assert "Screening" in fig.layout.xaxis.ticktext[0]
