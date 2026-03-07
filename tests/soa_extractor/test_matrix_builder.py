"""Tests for build_assessment_matrix (PTCV-152).

Covers 5 GHERKIN scenarios:
  1. Matrix populated from query pipeline output
  2. Assessments classified by type
  3. Visit dates extracted and anchored at screening
  4. At-home assessments captured with frequency
  5. Fallback when SoA table not parseable
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.soa_extractor.matrix_builder import (
    _validate_matrix,
    build_assessment_matrix_from_usdm,
)
from ptcv.soa_extractor.models import (
    UsdmActivity,
    UsdmScheduledInstance,
    UsdmTimepoint,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_TS = "2024-01-15T10:00:00+00:00"
_RUN = "run-matrix-001"
_SHA = "a" * 64
_REG = "NCT0001"


def _tp(
    tid: str,
    name: str,
    vtype: str = "Treatment",
    day: int = 0,
) -> UsdmTimepoint:
    return UsdmTimepoint(
        run_id=_RUN,
        source_run_id="",
        source_sha256=_SHA,
        registry_id=_REG,
        timepoint_id=tid,
        epoch_id="ep-1",
        visit_name=name,
        visit_type=vtype,
        day_offset=day,
        window_early=0,
        window_late=0,
        mandatory=True,
        extraction_timestamp_utc=_TS,
    )


def _act(aid: str, name: str, atype: str = "Assessment") -> UsdmActivity:
    return UsdmActivity(
        run_id=_RUN,
        source_run_id="",
        source_sha256=_SHA,
        registry_id=_REG,
        activity_id=aid,
        activity_name=name,
        activity_type=atype,
        extraction_timestamp_utc=_TS,
    )


def _inst(aid: str, tid: str) -> UsdmScheduledInstance:
    return UsdmScheduledInstance(
        run_id=_RUN,
        source_run_id="",
        source_sha256=_SHA,
        registry_id=_REG,
        instance_id=f"inst-{aid}-{tid}",
        activity_id=aid,
        timepoint_id=tid,
        scheduled=True,
        extraction_timestamp_utc=_TS,
    )


# Standard test fixtures: 3 visits, 4 assessments.
@pytest.fixture
def sample_timepoints():
    return [
        _tp("tp-scr", "Screening", "Screening", day=-14),
        _tp("tp-bl", "Baseline", "Baseline", day=1),
        _tp("tp-w2", "Week 2", "Treatment", day=15),
    ]


@pytest.fixture
def sample_activities():
    return [
        _act("act-vs", "Vital Signs", "Vital Signs"),
        _act("act-cbc", "CBC", "Lab"),
        _act("act-ecg", "ECG", "ECG"),
        _act("act-pe", "Physical Exam", "Assessment"),
    ]


@pytest.fixture
def sample_instances():
    return [
        _inst("act-vs", "tp-scr"),
        _inst("act-vs", "tp-bl"),
        _inst("act-vs", "tp-w2"),
        _inst("act-cbc", "tp-scr"),
        _inst("act-cbc", "tp-w2"),
        _inst("act-ecg", "tp-scr"),
        _inst("act-ecg", "tp-bl"),
        _inst("act-pe", "tp-scr"),
        _inst("act-pe", "tp-bl"),
        _inst("act-pe", "tp-w2"),
    ]


# ---------------------------------------------------------------------------
# Scenario 1: Matrix populated from query pipeline output
# ---------------------------------------------------------------------------


class TestMatrixFromUsdm:
    """GHERKIN: Matrix populated from query pipeline output."""

    def test_output_has_required_columns(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        required = {
            "assessment_name", "assessment_type",
            "visit_label", "day_offset", "scheduled",
        }
        assert required.issubset(set(df.columns))

    def test_each_row_is_assessment_visit_pair(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        pairs = df[["assessment_name", "visit_label"]].values.tolist()
        assert len(pairs) == len(set(map(tuple, pairs)))

    def test_row_count_matches_instances(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        assert len(df) == len(sample_instances)

    def test_empty_inputs_return_empty_df(self):
        df = build_assessment_matrix_from_usdm([], [], [])
        assert df.empty
        required = {
            "assessment_name", "assessment_type",
            "visit_label", "day_offset", "scheduled",
        }
        assert required.issubset(set(df.columns))

    def test_scheduled_always_true(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        assert (df["scheduled"] == True).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Scenario 2: Assessments classified by type
# ---------------------------------------------------------------------------


class TestAssessmentTypeClassification:
    """GHERKIN: Assessments classified by type."""

    def test_lab_classified_as_specimens(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [_act("a1", "CBC", "Lab")]
        insts = [_inst("a1", "t1"), _inst("a1", "t2")]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert (df["assessment_type"] == "Specimens").all()

    def test_vitals_classified_as_clinical_encounter(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [_act("a1", "Vital Signs", "Vital Signs")]
        insts = [_inst("a1", "t1"), _inst("a1", "t2")]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert (df["assessment_type"] == "Clinical Encounter").all()

    def test_ecg_classified_as_other(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [_act("a1", "ECG", "ECG")]
        insts = [_inst("a1", "t1"), _inst("a1", "t2")]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert (df["assessment_type"] == "Other").all()

    def test_imaging_classified_as_imaging(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [_act("a1", "MRI Brain", "Imaging")]
        insts = [_inst("a1", "t1"), _inst("a1", "t2")]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert (df["assessment_type"] == "Imaging").all()

    def test_drug_admin_classified_as_intervention(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [_act("a1", "Study Drug", "Drug Administration")]
        insts = [_inst("a1", "t1"), _inst("a1", "t2")]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert (df["assessment_type"] == "Intervention").all()


# ---------------------------------------------------------------------------
# Scenario 3: Visit dates extracted and anchored at screening
# ---------------------------------------------------------------------------


class TestScreeningAnchor:
    """GHERKIN: Visit dates extracted and anchored at screening."""

    def test_screening_anchored_to_day_zero(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        scr_rows = df[df["visit_label"] == "Screening"]
        assert (scr_rows["day_offset"] == 0).all()

    def test_baseline_offset_relative_to_screening(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        # Screening at -14, Baseline at 1 → 1 - (-14) = 15
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        bl_rows = df[df["visit_label"] == "Baseline"]
        assert (bl_rows["day_offset"] == 15).all()

    def test_visit_labels_preserved(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        labels = set(df["visit_label"].unique())
        assert labels == {"Screening", "Baseline", "Week 2"}

    def test_no_screening_uses_minimum_offset(self):
        tps = [
            _tp("t1", "Day 1", "Baseline", 1),
            _tp("t2", "Week 2", "Treatment", 15),
        ]
        acts = [
            _act("a1", "Vital Signs", "Vital Signs"),
            _act("a2", "CBC", "Lab"),
        ]
        insts = [
            _inst("a1", "t1"), _inst("a1", "t2"),
            _inst("a2", "t1"), _inst("a2", "t2"),
        ]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        # Minimum offset is 1 → Day 1 becomes 0, Week 2 becomes 14.
        d1_rows = df[df["visit_label"] == "Day 1"]
        assert (d1_rows["day_offset"] == 0).all()
        w2_rows = df[df["visit_label"] == "Week 2"]
        assert (w2_rows["day_offset"] == 14).all()

    def test_anchor_false_preserves_offsets(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [
            _act("a1", "Vital Signs", "Vital Signs"),
            _act("a2", "CBC", "Lab"),
        ]
        insts = [
            _inst("a1", "t1"), _inst("a1", "t2"),
            _inst("a2", "t1"), _inst("a2", "t2"),
        ]
        df = build_assessment_matrix_from_usdm(
            tps, acts, insts, anchor_screening=False,
        )
        scr_rows = df[df["visit_label"] == "Screening"]
        assert (scr_rows["day_offset"] == -14).all()


# ---------------------------------------------------------------------------
# Scenario 4: At-home assessments captured with frequency
# ---------------------------------------------------------------------------


class TestAtHomeAssessments:
    """GHERKIN: At-home assessments captured with frequency."""

    def test_ediary_type_is_at_home(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [
            _act("a1", "eDiary", "Other"),
            _act("a2", "CBC", "Lab"),
        ]
        insts = [
            _inst("a1", "t1"), _inst("a1", "t2"),
            _inst("a2", "t1"), _inst("a2", "t2"),
        ]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        ediary_rows = df[df["assessment_name"] == "eDiary"]
        assert (ediary_rows["assessment_type"] == "At-home").all()

    def test_patient_diary_type_is_at_home(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [
            _act("a1", "Patient diary", "Other"),
            _act("a2", "CBC", "Lab"),
        ]
        insts = [
            _inst("a1", "t1"), _inst("a1", "t2"),
            _inst("a2", "t1"), _inst("a2", "t2"),
        ]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        diary_rows = df[df["assessment_name"] == "Patient diary"]
        assert (diary_rows["assessment_type"] == "At-home").all()

    def test_daily_frequency_extracted(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [
            _act("a1", "eDiary (daily)", "Other"),
            _act("a2", "CBC", "Lab"),
        ]
        insts = [
            _inst("a1", "t1"), _inst("a1", "t2"),
            _inst("a2", "t1"), _inst("a2", "t2"),
        ]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        ediary_rows = df[df["assessment_name"] == "eDiary (daily)"]
        assert "frequency" in df.columns
        assert (ediary_rows["frequency"] == "daily").all()

    def test_no_frequency_column_when_not_needed(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        assert "frequency" not in df.columns


# ---------------------------------------------------------------------------
# Scenario 5: Fallback when SoA table not parseable
# ---------------------------------------------------------------------------


class TestFallbackPath:
    """GHERKIN: Fallback when SoA table not parseable."""

    def test_garbage_visit_headers_rejected(self):
        """Prose-word visit names fail validation."""
        tps = [
            _tp("t1", "randomized,", "Treatment", 0),
            _tp("t2", "double-blind,", "Treatment", 0),
            _tp("t3", "placebo-controlled,", "Treatment", 0),
        ]
        acts = [
            _act("a1", "Vital Signs", "Vital Signs"),
            _act("a2", "CBC", "Lab"),
        ]
        insts = [
            _inst("a1", "t1"), _inst("a1", "t2"),
            _inst("a2", "t1"), _inst("a2", "t3"),
        ]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert df.empty

    def test_fewer_than_two_visits_rejected(self):
        tps = [_tp("t1", "Screening", "Screening", -14)]
        acts = [
            _act("a1", "Vital Signs", "Vital Signs"),
            _act("a2", "CBC", "Lab"),
        ]
        insts = [_inst("a1", "t1"), _inst("a2", "t1")]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert df.empty

    def test_fewer_than_two_assessments_rejected(self):
        tps = [
            _tp("t1", "Screening", "Screening", -14),
            _tp("t2", "V1", "Treatment", 1),
        ]
        acts = [_act("a1", "Vital Signs", "Vital Signs")]
        insts = [_inst("a1", "t1"), _inst("a1", "t2")]
        df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert df.empty

    def test_fallback_logs_warning(self, caplog):
        """Garbage data logs a validation warning."""
        tps = [
            _tp("t1", "randomized,", "Treatment", 0),
            _tp("t2", "double-blind,", "Treatment", 0),
        ]
        acts = [
            _act("a1", "Vital Signs", "Vital Signs"),
            _act("a2", "CBC", "Lab"),
        ]
        insts = [
            _inst("a1", "t1"), _inst("a1", "t2"),
            _inst("a2", "t1"), _inst("a2", "t2"),
        ]
        with caplog.at_level(logging.WARNING):
            df = build_assessment_matrix_from_usdm(tps, acts, insts)
        assert df.empty
        assert "validation failed" in caplog.text.lower()

    def test_valid_matrix_passes_validation(
        self, sample_timepoints, sample_activities, sample_instances,
    ):
        df = build_assessment_matrix_from_usdm(
            sample_timepoints, sample_activities, sample_instances,
        )
        assert not df.empty
        assert len(df) == len(sample_instances)
