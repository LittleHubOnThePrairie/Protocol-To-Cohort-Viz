"""Unit tests for schedule of visits component (PTCV-34, PTCV-71, PTCV-151).

Tests the pure-Python helpers (classify_swimlane, build_sov_grid,
build_sov_csv, build_sov_figure) without Streamlit runtime.

PTCV-151: Tests updated for assessment × visit matrix output.

Covers GHERKIN scenarios:
  - Builder outputs assessment-level rows (PTCV-151)
  - Screening visit anchored at day 0 (PTCV-151)
  - No assessments are concatenated (PTCV-151)
  - Clinical encounter row lists clinician-collected variables
  - Specimen row lists specimen types and derived assays
  - Imaging row lists imaging modality
  - Non-ICH protocol produces empty/minimal schedule
  - CSV download has correct structure
  - Intervention swimlane displays drug administration (PTCV-71)
  - Specimen-based tests classified correctly (PTCV-71)
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
    _rebase_to_screening,
    build_sov_csv,
    build_sov_figure,
    build_sov_figure_from_matrix,
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
        ("ECG", "Other"),
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
        """Activities with intervention keywords -> Intervention."""
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
        """Activities with specimen keywords -> Specimens."""
        assert classify_swimlane("Assessment", name) == expected

    def test_specimen_keyword_does_not_override_lab(self) -> None:
        """Lab type already maps to Specimens -- no change needed."""
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
        """An Assessment named 'Study drug' -> Intervention."""
        assert classify_swimlane("Assessment", "Study drug dose") == "Intervention"


# ---------------------------------------------------------------------------
# _rebase_to_screening (PTCV-151)
# ---------------------------------------------------------------------------

class TestRebaseToScreening:
    """PTCV-151: Screening day anchored at day 0."""

    def test_screening_becomes_day_zero(self) -> None:
        tps = [
            _tp("v1", "Screening", day_offset=-14),
            _tp("v2", "Baseline", day_offset=1),
            _tp("v3", "EOT", day_offset=84),
        ]
        rebased = _rebase_to_screening(tps)
        assert rebased["v1"] == 0
        assert rebased["v2"] == 15
        assert rebased["v3"] == 98

    def test_all_positive_offsets(self) -> None:
        """When minimum offset is 0, nothing shifts."""
        tps = [_tp("t1", "V1", 0), _tp("t2", "V2", 7)]
        rebased = _rebase_to_screening(tps)
        assert rebased["t1"] == 0
        assert rebased["t2"] == 7

    def test_empty_returns_empty(self) -> None:
        assert _rebase_to_screening([]) == {}


# ---------------------------------------------------------------------------
# build_sov_grid -- PTCV-151: assessment x visit matrix
# ---------------------------------------------------------------------------

class TestBuildSovGrid:
    """Tests for build_sov_grid() -- assessment-level output (PTCV-151)."""

    def test_grid_has_expected_keys(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert len(grid) > 0
        for rec in grid:
            assert "assessment" in rec
            assert "assessment_type" in rec
            assert "visit_label" in rec
            assert "day" in rec
            assert "visit_index" in rec
            assert "activity_id" in rec

    def test_one_record_per_scheduled_instance(self) -> None:
        """[PTCV-151 Scenario: Builder outputs assessment-level rows]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        # 8 scheduled instances in INSTANCES
        assert len(grid) == 8

    def test_no_concatenated_activity_names(self) -> None:
        """[PTCV-151 Scenario: No assessments concatenated]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        for rec in grid:
            assert "," not in rec["assessment"]

    def test_screening_anchored_at_day_zero(self) -> None:
        """[PTCV-151 Scenario: Screening visit anchored at day 0]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        screening = [r for r in grid if r["visit_label"] == "Screening"]
        assert len(screening) > 0
        assert all(r["day"] == 0 for r in screening)
        # Baseline (C1D1) should have positive day offset
        c1d1 = [r for r in grid if r["visit_label"] == "C1D1"]
        assert all(r["day"] > 0 for r in c1d1)

    def test_clinical_encounter_for_vitals(self) -> None:
        """[PTCV-34 Scenario: Clinical encounter row lists variables]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        vitals = [
            r for r in grid
            if r["assessment"] == "Vital Signs"
        ]
        # Vitals scheduled at all 3 visits
        assert len(vitals) == 3
        assert all(r["assessment_type"] == "Clinical Encounter" for r in vitals)

    def test_specimen_for_labs(self) -> None:
        """[PTCV-34 Scenario: Specimen row lists specimen types]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        cbc = [r for r in grid if r["assessment"] == "CBC"]
        # CBC scheduled at Screening + C1D1
        assert len(cbc) == 2
        assert all(r["assessment_type"] == "Specimens" for r in cbc)

    def test_imaging_for_mri(self) -> None:
        """[PTCV-34 Scenario: Imaging row lists imaging modality]"""
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        mri = [r for r in grid if r["assessment"] == "MRI Brain"]
        # MRI Brain at Screening + EOT
        assert len(mri) == 2
        assert all(r["assessment_type"] == "Imaging" for r in mri)

    def test_other_for_procedure(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        ecg = [r for r in grid if r["assessment"] == "ECG Holter"]
        # ECG Holter at C1D1
        assert len(ecg) == 1
        assert ecg[0]["assessment_type"] == "Other"

    def test_grid_sorted_by_visit_then_type(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, INSTANCES)
        visit_indices = [r["visit_index"] for r in grid]
        assert visit_indices == sorted(visit_indices)

    def test_empty_instances_produces_empty_grid(self) -> None:
        grid = build_sov_grid(TIMEPOINTS, ACTIVITIES, [])
        assert grid == []

    def test_empty_timepoints_produces_empty_grid(self) -> None:
        grid = build_sov_grid([], ACTIVITIES, INSTANCES)
        assert grid == []

    def test_two_activities_same_visit_produce_two_records(self) -> None:
        """Two activities at the same visit produce separate records."""
        acts = [
            _act("a10", "Physical Exam", "Assessment"),
            _act("a11", "ECOG Score", "Assessment"),
        ]
        insts = [_inst("a10", "v1"), _inst("a11", "v1")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        clinical_v1 = [
            r for r in grid
            if r["assessment_type"] == "Clinical Encounter"
            and r["visit_label"] == "Screening"
        ]
        assert len(clinical_v1) == 2
        names = {r["assessment"] for r in clinical_v1}
        assert names == {"Physical Exam", "ECOG Score"}


# ---------------------------------------------------------------------------
# build_sov_csv -- PTCV-151
# ---------------------------------------------------------------------------

class TestBuildSovCsv:
    """Tests for build_sov_csv()."""

    def test_csv_has_header(self) -> None:
        """[PTCV-34 Scenario: CSV download has correct structure]"""
        csv_str = build_sov_csv(TIMEPOINTS, ACTIVITIES, INSTANCES)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header == ["Assessment", "Assessment Type", "Visit", "Day"]

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

    def test_csv_contains_assessment_names(self) -> None:
        csv_str = build_sov_csv(TIMEPOINTS, ACTIVITIES, INSTANCES)
        assert "Vital Signs" in csv_str
        assert "CBC" in csv_str
        assert "MRI Brain" in csv_str

    def test_empty_data_returns_header_only(self) -> None:
        csv_str = build_sov_csv([], [], [])
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0] == ["Assessment", "Assessment Type", "Visit", "Day"]


# ---------------------------------------------------------------------------
# build_sov_figure -- PTCV-151: assessment x visit chart
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestBuildSovFigure:
    """Tests for build_sov_figure() -- assessment x visit chart."""

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
        normal_colors = {t.marker.color for t in fig_normal.data}
        low_colors = {t.marker.color for t in fig_low.data}
        assert normal_colors != low_colors

    def test_figure_yaxis_has_assessment_names(self) -> None:
        """PTCV-151: Y-axis shows individual assessment names."""
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001"
        )
        ticktext = list(fig.layout.yaxis.ticktext)
        assert "Vital Signs" in ticktext
        assert "CBC" in ticktext
        assert "MRI Brain" in ticktext
        assert "ECG Holter" in ticktext

    def test_figure_xaxis_has_visit_labels(self) -> None:
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001"
        )
        ticktext = fig.layout.xaxis.ticktext
        joined = " ".join(ticktext)
        assert "Screening" in joined
        assert "C1D1" in joined
        assert "EOT" in joined

    def test_figure_xaxis_has_rebased_day_offsets(self) -> None:
        """PTCV-151: X-axis labels use rebased days (screening = day 0)."""
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001"
        )
        ticktext = fig.layout.xaxis.ticktext
        joined = " ".join(ticktext)
        assert "Day 0" in joined   # Screening (was -14)
        assert "Day 15" in joined  # C1D1 (was 1, rebased: 1 - (-14) = 15)
        assert "Day 98" in joined  # EOT (was 84, rebased: 84 - (-14) = 98)

    def test_figure_height_scales_with_assessments(self) -> None:
        """Chart height increases with more assessments."""
        fig = build_sov_figure(
            TIMEPOINTS, ACTIVITIES, INSTANCES, registry_id="NCT001"
        )
        # 4 assessments: max(400, 30*4) = 400
        assert fig.layout.height >= 400


# ---------------------------------------------------------------------------
# PTCV-71: Intervention swimlane grid integration
# ---------------------------------------------------------------------------

class TestInterventionSwimlaneGrid:
    """PTCV-71: Drug administration appears in Intervention type."""

    def test_drug_admin_type_in_intervention(self) -> None:
        """Activity type 'Drug Administration' -> Intervention."""
        acts = [_act("a20", "Study Drug", "Drug Administration")]
        insts = [_inst("a20", "v2")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        intervention = [r for r in grid if r["assessment_type"] == "Intervention"]
        assert len(intervention) == 1
        assert intervention[0]["assessment"] == "Study Drug"

    def test_keyword_reclassification_in_grid(self) -> None:
        """Activity with name 'Placebo infusion' reclassified in grid."""
        acts = [_act("a21", "Placebo infusion", "Procedure")]
        insts = [_inst("a21", "v1")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        intervention = [r for r in grid if r["assessment_type"] == "Intervention"]
        assert len(intervention) == 1
        assert intervention[0]["assessment"] == "Placebo infusion"

    def test_specimen_reclassification_in_grid(self) -> None:
        """Activity 'Urine pregnancy test' (Assessment) -> Specimens."""
        acts = [_act("a22", "Urine pregnancy test", "Assessment")]
        insts = [_inst("a22", "v1")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        specimens = [r for r in grid if r["assessment_type"] == "Specimens"]
        assert len(specimens) == 1
        assert specimens[0]["assessment"] == "Urine pregnancy test"


# ---------------------------------------------------------------------------
# PTCV-73: Timepoint deduplication
# ---------------------------------------------------------------------------

class TestTimepointDedup:
    """PTCV-73: Duplicate timepoints collapsed on x-axis."""

    def test_dedup_merges_same_name_and_offset(self) -> None:
        """25 timepoints with same name+offset -> 1 unique."""
        tps = [
            _tp(f"dup-{i}", "Screening", day_offset=7)
            for i in range(25)
        ]
        unique, id_map = _dedup_timepoints(tps)
        assert len(unique) == 1
        assert unique[0].visit_name == "Screening"
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
        """Activities from duplicate timepoints merge into one visit column."""
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
        # All activities under one "Screening" visit
        visit_labels = {r["visit_label"] for r in grid}
        assert visit_labels == {"Screening"}
        # 3 separate assessment records
        assert len(grid) == 3
        assessments = {r["assessment"] for r in grid}
        assert assessments == {"Vital Signs", "CBC", "MRI Brain"}

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


# ---------------------------------------------------------------------------
# PTCV-131: eDiary and ECG classification fixes
# ---------------------------------------------------------------------------

class TestEdiarySwimlane:
    """PTCV-131: eDiary should be 'Other', not 'Clinical Encounter'."""

    def test_ediary_assessment_reclassified(self) -> None:
        assert classify_swimlane("Assessment", "eDiary") == "Other"

    def test_electronic_diary_reclassified(self) -> None:
        assert classify_swimlane("Assessment", "Electronic diary") == "Other"

    def test_patient_diary_reclassified(self) -> None:
        assert classify_swimlane("Assessment", "Patient diary") == "Other"

    def test_ediary_in_grid(self) -> None:
        """eDiary activity lands in Other in full grid."""
        acts = [_act("a50", "eDiary", "Assessment")]
        insts = [_inst("a50", "v1")]
        grid = build_sov_grid(TIMEPOINTS, acts, insts)
        other = [r for r in grid if r["assessment_type"] == "Other"]
        assert len(other) == 1
        assert other[0]["assessment"] == "eDiary"


class TestEcgSwimlane:
    """PTCV-131: ECG type should map to 'Other'."""

    def test_ecg_direct_type(self) -> None:
        assert classify_swimlane("ECG") == "Other"

    def test_ecg_name_reclassifies_assessment(self) -> None:
        assert classify_swimlane("Assessment", "ECG") == "Other"

    def test_electrocardiogram_reclassifies(self) -> None:
        assert classify_swimlane("Assessment", "Electrocardiogram") == "Other"


# ---------------------------------------------------------------------------
# PTCV-131: Visit sort order
# ---------------------------------------------------------------------------

class TestVisitSortOrder:
    """PTCV-131: Visits sorted chronologically by day_offset then number."""

    def test_v1_v2_v3_v4_order(self) -> None:
        """V1 through V4 should appear in order."""
        tps = [
            _tp("t4", "V4", day_offset=4),
            _tp("t2", "V2", day_offset=2),
            _tp("t1", "V1", day_offset=1),
            _tp("t3", "V3", day_offset=3),
        ]
        acts = [_act("a60", "Vitals", "Vital Signs")]
        insts = [
            _inst("a60", "t1"), _inst("a60", "t2"),
            _inst("a60", "t3"), _inst("a60", "t4"),
        ]
        grid = build_sov_grid(tps, acts, insts)
        visit_labels = [r["visit_label"] for r in grid]
        assert visit_labels == ["V1", "V2", "V3", "V4"]

    def test_mixed_day_offsets_sorted(self) -> None:
        """Visits with real day offsets sort correctly."""
        tps = [
            _tp("t1", "EOT", day_offset=84),
            _tp("t2", "Screening", day_offset=-14),
            _tp("t3", "C1D1", day_offset=1),
        ]
        acts = [_act("a61", "CBC", "Lab")]
        insts = [
            _inst("a61", "t1"),
            _inst("a61", "t2"),
            _inst("a61", "t3"),
        ]
        grid = build_sov_grid(tps, acts, insts)
        visit_labels = [r["visit_label"] for r in grid]
        assert visit_labels == ["Screening", "C1D1", "EOT"]


# ---------------------------------------------------------------------------
# PTCV-151: build_sov_figure_from_matrix
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestBuildSovFigureFromMatrix:
    """Tests for the assessment x visit matrix chart (PTCV-151)."""

    def test_basic_matrix_chart(self) -> None:
        """3 assessments x 3 visits produces a figure with traces."""
        fig = build_sov_figure_from_matrix(
            TIMEPOINTS, ACTIVITIES, INSTANCES,
        )
        assert fig is not None
        assert len(fig.data) >= 1

    def test_assessments_on_y_axis(self) -> None:
        """Y-axis tick labels include individual assessment names."""
        fig = build_sov_figure_from_matrix(
            TIMEPOINTS, ACTIVITIES, INSTANCES,
        )
        y_labels = fig.layout.yaxis.ticktext
        assert y_labels is not None
        label_set = set(y_labels)
        for act in ACTIVITIES:
            assert act.activity_name in label_set

    def test_visits_on_x_axis(self) -> None:
        """X-axis labels contain visit names."""
        fig = build_sov_figure_from_matrix(
            TIMEPOINTS, ACTIVITIES, INSTANCES,
        )
        x_labels = fig.layout.xaxis.ticktext
        assert x_labels is not None
        joined = " ".join(x_labels)
        assert "Screening" in joined
        assert "C1D1" in joined
        assert "EOT" in joined

    def test_screening_anchored_day_0(self) -> None:
        """X-axis shows Day 0 for screening visit."""
        fig = build_sov_figure_from_matrix(
            TIMEPOINTS, ACTIVITIES, INSTANCES,
        )
        x_labels = fig.layout.xaxis.ticktext
        assert x_labels is not None
        assert "Day 0" in x_labels[0]

    def test_type_grouping(self) -> None:
        """Assessments are grouped by type on the Y-axis."""
        fig = build_sov_figure_from_matrix(
            TIMEPOINTS, ACTIVITIES, INSTANCES,
        )
        y_labels = list(fig.layout.yaxis.ticktext)
        specimens = [
            i for i, l in enumerate(y_labels) if l == "CBC"
        ]
        others = [
            i for i, l in enumerate(y_labels) if l == "ECG Holter"
        ]
        if specimens and others:
            assert min(specimens) < min(others)

    def test_at_home_detected(self) -> None:
        """eDiary assessment gets At-home type."""
        tps = [
            _tp("v1", "Screening", -14),
            _tp("v2", "V1", 1),
            _tp("v3", "V2", 8),
        ]
        acts = [
            _act("a1", "Vital Signs", "Vital Signs"),
            _act("a2", "CBC", "Lab"),
            _act("a3", "eDiary", "Other"),
        ]
        insts = [
            _inst("a1", "v1"), _inst("a1", "v2"),
            _inst("a2", "v1"), _inst("a2", "v3"),
            _inst("a3", "v2"), _inst("a3", "v3"),
        ]
        fig = build_sov_figure_from_matrix(tps, acts, insts)
        trace_names = [t.name for t in fig.data]
        assert "At-home" in trace_names

    def test_empty_matrix_fallback(self) -> None:
        """Empty instances fall back to swimlane chart."""
        fig = build_sov_figure_from_matrix(
            TIMEPOINTS, ACTIVITIES, [],
        )
        assert fig is not None

    def test_low_confidence_muted_colors(self) -> None:
        """Muted palette used when low_confidence=True."""
        fig = build_sov_figure_from_matrix(
            TIMEPOINTS, ACTIVITIES, INSTANCES,
            low_confidence=True,
        )
        for trace in fig.data:
            color = trace.marker.color
            assert "rgba" in str(color)
