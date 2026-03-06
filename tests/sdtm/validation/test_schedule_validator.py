"""Unit tests for visit schedule feasibility validator (PTCV-57).

Tests the 12 validation rules (VS-001 through VS-012) covering all
5 GHERKIN acceptance criteria plus edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import pandas as pd
import pytest

from ptcv.sdtm.models import SoaCellMetadata
from ptcv.sdtm.validation.schedule_validator import (
    ScheduleIssue,
    ScheduleValidationReport,
    VisitScheduleValidator,
)


# -----------------------------------------------------------------------
# Helper to build TV DataFrames
# -----------------------------------------------------------------------

def _make_tv(
    visits: list[dict],
) -> pd.DataFrame:
    """Build a TV DataFrame from a list of visit dicts.

    Each dict needs at minimum: VISIT, VISITDY.
    STUDYID, DOMAIN, VISITNUM, TVSTRL, TVENRL default if absent.
    """
    rows = []
    for i, v in enumerate(visits):
        rows.append({
            "STUDYID": v.get("STUDYID", "NCT_TEST"),
            "DOMAIN": "TV",
            "VISITNUM": v.get("VISITNUM", float(i + 1)),
            "VISIT": v["VISIT"],
            "VISITDY": float(v["VISITDY"]),
            "TVSTRL": float(v.get("TVSTRL", 0)),
            "TVENRL": float(v.get("TVENRL", 0)),
        })
    return pd.DataFrame(rows)


def _make_te(
    elements: list[dict] | None = None,
) -> pd.DataFrame:
    """Build a TE DataFrame. Defaults to SCRN + TRT + FUP."""
    if elements is None:
        elements = [
            {"ETCD": "SCRN", "ELEMENT": "Screening"},
            {"ETCD": "TRT", "ELEMENT": "Treatment"},
            {"ETCD": "FUP", "ELEMENT": "Follow-up"},
        ]
    rows = []
    for e in elements:
        rows.append({
            "STUDYID": e.get("STUDYID", "NCT_TEST"),
            "DOMAIN": "TE",
            "ETCD": e["ETCD"],
            "ELEMENT": e["ELEMENT"],
            "TESTRL": e.get("TESTRL", ""),
            "TEENRL": e.get("TEENRL", ""),
            "TEDUR": e.get("TEDUR", ""),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Correct schedule (reused across clean-schedule tests)
# -----------------------------------------------------------------------

CLEAN_TV = _make_tv([
    {"VISIT": "Screening", "VISITDY": -14, "TVSTRL": -3, "TVENRL": 0},
    {"VISIT": "Baseline", "VISITDY": 1, "TVSTRL": -1, "TVENRL": 1},
    {"VISIT": "Week 4", "VISITDY": 29, "TVSTRL": -3, "TVENRL": 3},
    {"VISIT": "Week 8", "VISITDY": 57, "TVSTRL": -3, "TVENRL": 3},
    {"VISIT": "Follow-up", "VISITDY": 85, "TVSTRL": -7, "TVENRL": 7},
])

CLEAN_TE = _make_te()


# -----------------------------------------------------------------------
# Scenario 1: Overlapping visit windows detected
# -----------------------------------------------------------------------

class TestOverlappingWindows:
    """Scenario: Overlapping visit windows detected."""

    def test_vs004_overlap_detected(self) -> None:
        """Given Visit 3 (Day 14 +/- 7) and Visit 4 (Day 18 +/- 3),
        when the schedule validator runs,
        then a VS-004 warning is raised."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14, "TVSTRL": -3, "TVENRL": 0},
            {"VISIT": "Baseline", "VISITDY": 1, "TVSTRL": -1, "TVENRL": 1},
            {"VISIT": "Week 2", "VISITDY": 14, "TVSTRL": -7, "TVENRL": 7},
            {"VISIT": "Week 3", "VISITDY": 18, "TVSTRL": -3, "TVENRL": 3},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs004 = [i for i in report.issues if i.rule_id == "VS-004"]
        assert len(vs004) >= 1
        assert vs004[0].severity == "Warning"

    def test_vs004_overlap_range_reported(self) -> None:
        """The overlap amount is mentioned in the description."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Baseline", "VISITDY": 1},
            {"VISIT": "Week 2", "VISITDY": 14, "TVSTRL": -7, "TVENRL": 7},
            {"VISIT": "Week 3", "VISITDY": 18, "TVSTRL": -3, "TVENRL": 3},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs004 = [i for i in report.issues if i.rule_id == "VS-004"]
        assert len(vs004) >= 1
        # Description mentions the overlap
        assert "overlap" in vs004[0].description.lower() or "day" in vs004[0].description.lower()

    def test_vs004_no_overlap_no_issue(self) -> None:
        """Non-overlapping windows produce no VS-004."""
        validator = VisitScheduleValidator(CLEAN_TV, CLEAN_TE)
        report = validator.validate()

        vs004 = [i for i in report.issues if i.rule_id == "VS-004"]
        assert len(vs004) == 0


# -----------------------------------------------------------------------
# Scenario 2: Logical ordering violation detected
# -----------------------------------------------------------------------

class TestLogicalOrderingViolation:
    """Scenario: Logical ordering violation detected."""

    def test_vs002_followup_before_treatment(self) -> None:
        """Given Follow-up at Day 1 and Screening at Day 14,
        when the schedule validator runs,
        then a VS-002 error is raised."""
        tv = _make_tv([
            {"VISIT": "Follow-up", "VISITDY": 1},
            {"VISIT": "Screening", "VISITDY": 14},
            {"VISIT": "Baseline", "VISITDY": 28},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs002 = [i for i in report.issues if i.rule_id == "VS-002"]
        assert len(vs002) >= 1
        assert vs002[0].severity == "Error"

    def test_vs001_screening_after_treatment(self) -> None:
        """Screening day > baseline day triggers VS-001."""
        tv = _make_tv([
            {"VISIT": "Baseline", "VISITDY": 1},
            {"VISIT": "Screening", "VISITDY": 14},
            {"VISIT": "Follow-up", "VISITDY": 85},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs001 = [i for i in report.issues if i.rule_id == "VS-001"]
        assert len(vs001) >= 1
        assert vs001[0].severity == "Error"

    def test_vs003_early_term_before_screening(self) -> None:
        """Early Termination before Screening triggers VS-003."""
        tv = _make_tv([
            {"VISIT": "Early Termination", "VISITDY": -30},
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Baseline", "VISITDY": 1},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs003 = [i for i in report.issues if i.rule_id == "VS-003"]
        assert len(vs003) >= 1
        assert vs003[0].severity == "Error"

    def test_correct_ordering_no_errors(self) -> None:
        """Correctly ordered visits produce no ordering errors."""
        validator = VisitScheduleValidator(CLEAN_TV, CLEAN_TE)
        report = validator.validate()

        ordering_errors = [
            i for i in report.issues
            if i.rule_id in ("VS-001", "VS-002", "VS-003")
        ]
        assert len(ordering_errors) == 0


# -----------------------------------------------------------------------
# Scenario 3: Duplicate visits detected
# -----------------------------------------------------------------------

class TestDuplicateVisits:
    """Scenario: Duplicate visits detected."""

    def test_vs006_duplicate_visit_name(self) -> None:
        """Two visits named 'Week 4' triggers VS-006."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14, "VISITNUM": 1.0},
            {"VISIT": "Week 4", "VISITDY": 29, "VISITNUM": 2.0},
            {"VISIT": "Week 4", "VISITDY": 57, "VISITNUM": 3.0},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs006 = [
            i for i in report.issues
            if i.rule_id == "VS-006" and i.variable == "VISIT"
        ]
        assert len(vs006) >= 1
        assert vs006[0].severity == "Error"

    def test_vs006_duplicate_visitnum(self) -> None:
        """Two visits with VISITNUM=2 triggers VS-006."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14, "VISITNUM": 1.0},
            {"VISIT": "Week 2", "VISITDY": 15, "VISITNUM": 2.0},
            {"VISIT": "Week 4", "VISITDY": 29, "VISITNUM": 2.0},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs006 = [
            i for i in report.issues
            if i.rule_id == "VS-006" and i.variable == "VISITNUM"
        ]
        assert len(vs006) >= 1
        assert vs006[0].severity == "Error"

    def test_unique_visits_no_vs006(self) -> None:
        """Unique visits produce no VS-006."""
        validator = VisitScheduleValidator(CLEAN_TV, CLEAN_TE)
        report = validator.validate()

        vs006 = [i for i in report.issues if i.rule_id == "VS-006"]
        assert len(vs006) == 0


# -----------------------------------------------------------------------
# Scenario 4: Clean schedule passes validation
# -----------------------------------------------------------------------

class TestCleanSchedule:
    """Scenario: Clean schedule passes validation."""

    def test_no_errors_reported(self) -> None:
        """Correctly ordered, non-overlapping visits produce no errors."""
        validator = VisitScheduleValidator(CLEAN_TV, CLEAN_TE)
        report = validator.validate()

        assert report.error_count == 0

    def test_schedule_marked_feasible(self) -> None:
        """Clean schedule is marked as feasible."""
        validator = VisitScheduleValidator(CLEAN_TV, CLEAN_TE)
        report = validator.validate()

        assert report.feasible is True

    def test_report_type(self) -> None:
        """Validate returns ScheduleValidationReport."""
        validator = VisitScheduleValidator(CLEAN_TV, CLEAN_TE)
        report = validator.validate()
        assert isinstance(report, ScheduleValidationReport)


# -----------------------------------------------------------------------
# Scenario 5: Validation report includes remediation guidance
# -----------------------------------------------------------------------

class TestReportRemediation:
    """Scenario: Report includes remediation guidance."""

    def test_each_issue_has_required_fields(self) -> None:
        """Each issue includes rule_id, severity, message, suggestion."""
        # Use overlapping schedule to generate issues
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Baseline", "VISITDY": 1},
            {"VISIT": "Week 2", "VISITDY": 14, "TVSTRL": -7, "TVENRL": 7},
            {"VISIT": "Week 3", "VISITDY": 18, "TVSTRL": -3, "TVENRL": 3},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        assert len(report.issues) > 0
        for issue in report.issues:
            assert issue.rule_id
            assert issue.severity in ("Error", "Warning")
            assert issue.description
            assert issue.remediation_guidance

    def test_issues_sorted_by_severity(self) -> None:
        """Errors appear before warnings in the report."""
        tv = _make_tv([
            {"VISIT": "Follow-up", "VISITDY": 1},
            {"VISIT": "Screening", "VISITDY": 14},
            {"VISIT": "Baseline", "VISITDY": 28},
            {"VISIT": "Week 2", "VISITDY": 14, "TVSTRL": -7, "TVENRL": 7},
            {"VISIT": "Week 3", "VISITDY": 18, "TVSTRL": -3, "TVENRL": 3},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        severities = [i.severity for i in report.issues]
        # All Errors should come before all Warnings
        error_done = False
        for s in severities:
            if s == "Warning":
                error_done = True
            if s == "Error" and error_done:
                pytest.fail("Error after Warning — issues not sorted")

    def test_error_count_matches(self) -> None:
        """error_count matches number of Error severity issues."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14, "VISITNUM": 1.0},
            {"VISIT": "Week 4", "VISITDY": 29, "VISITNUM": 2.0},
            {"VISIT": "Week 4", "VISITDY": 57, "VISITNUM": 3.0},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        actual = sum(1 for i in report.issues if i.severity == "Error")
        assert report.error_count == actual

    def test_feasible_false_when_errors(self) -> None:
        """feasible is False when errors exist."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14, "VISITNUM": 1.0},
            {"VISIT": "Week 4", "VISITDY": 29, "VISITNUM": 2.0},
            {"VISIT": "Week 4", "VISITDY": 57, "VISITNUM": 2.0},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        assert report.feasible is False


# -----------------------------------------------------------------------
# Additional rule coverage
# -----------------------------------------------------------------------

class TestAdditionalRules:
    """Coverage for VS-005 through VS-012."""

    def test_vs005_window_proportion(self) -> None:
        """Window > 50% of interval triggers VS-005."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Baseline", "VISITDY": 1, "TVSTRL": -5, "TVENRL": 5},
            {"VISIT": "Week 1", "VISITDY": 8},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs005 = [i for i in report.issues if i.rule_id == "VS-005"]
        assert len(vs005) >= 1
        assert vs005[0].severity == "Warning"

    def test_vs008_epoch_no_visits(self) -> None:
        """TE epoch with no matching TV visits triggers VS-008."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Baseline", "VISITDY": 1},
        ])
        # TE has FUP but TV has no Follow-up visit
        te = _make_te([
            {"ETCD": "SCRN", "ELEMENT": "Screening"},
            {"ETCD": "TRT", "ELEMENT": "Treatment"},
            {"ETCD": "FUP", "ELEMENT": "Follow-up"},
        ])
        validator = VisitScheduleValidator(tv, te)
        report = validator.validate()

        vs008 = [i for i in report.issues if i.rule_id == "VS-008"]
        assert len(vs008) >= 1
        assert vs008[0].severity == "Warning"

    def test_vs009_epoch_order_violation(self) -> None:
        """Follow-up visit between screening and treatment triggers VS-009."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Follow-up", "VISITDY": -7},
            {"VISIT": "Baseline", "VISITDY": 1},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs009 = [i for i in report.issues if i.rule_id == "VS-009"]
        assert len(vs009) >= 1
        assert vs009[0].severity == "Error"

    def test_vs010_missing_visitdy(self) -> None:
        """Visit with null VISITDY triggers VS-010."""
        tv = pd.DataFrame([
            {"STUDYID": "NCT_TEST", "DOMAIN": "TV", "VISITNUM": 1.0,
             "VISIT": "Screening", "VISITDY": -14.0,
             "TVSTRL": 0.0, "TVENRL": 0.0},
            {"STUDYID": "NCT_TEST", "DOMAIN": "TV", "VISITNUM": 2.0,
             "VISIT": "Missing Day", "VISITDY": float("nan"),
             "TVSTRL": 0.0, "TVENRL": 0.0},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs010 = [i for i in report.issues if i.rule_id == "VS-010"]
        assert len(vs010) >= 1
        assert vs010[0].severity == "Warning"

    def test_vs011_long_study(self) -> None:
        """Study duration > 10 years triggers VS-011."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Baseline", "VISITDY": 1},
            {"VISIT": "Follow-up", "VISITDY": 4000},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs011 = [i for i in report.issues if i.rule_id == "VS-011"]
        assert len(vs011) >= 1
        assert vs011[0].severity == "Warning"

    def test_vs012_empty_visit(self) -> None:
        """Visit with zero activities triggers VS-012 when soa_matrix given."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Baseline", "VISITDY": 1},
            {"VISIT": "Week 4", "VISITDY": 29},
        ])
        # Only visits 1 and 2 have activities
        soa_matrix: dict[tuple[int, str], SoaCellMetadata] = {
            (1, "Physical Exam"): SoaCellMetadata(
                visitnum=1, assessment="Physical Exam",
                status="required", condition="", category="safety",
                cdash_domain="PE", timing_window_days=(0, 0),
            ),
            (2, "Vital Signs"): SoaCellMetadata(
                visitnum=2, assessment="Vital Signs",
                status="required", condition="", category="safety",
                cdash_domain="VS", timing_window_days=(0, 0),
            ),
        }
        validator = VisitScheduleValidator(tv, CLEAN_TE, soa_matrix=soa_matrix)
        report = validator.validate()

        vs012 = [i for i in report.issues if i.rule_id == "VS-012"]
        assert len(vs012) >= 1
        assert vs012[0].severity == "Warning"
        assert "Week 4" in vs012[0].description

    def test_vs012_skipped_without_soa_matrix(self) -> None:
        """VS-012 is skipped when no soa_matrix is provided."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
            {"VISIT": "Baseline", "VISITDY": 1},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs012 = [i for i in report.issues if i.rule_id == "VS-012"]
        assert len(vs012) == 0


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_empty_tv_no_crash(self) -> None:
        """Empty TV DataFrame produces empty report."""
        tv = pd.DataFrame(columns=[
            "STUDYID", "DOMAIN", "VISITNUM", "VISIT", "VISITDY",
            "TVSTRL", "TVENRL",
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()
        assert report.feasible is True
        assert report.error_count == 0

    def test_single_visit(self) -> None:
        """Single visit produces no ordering errors."""
        tv = _make_tv([
            {"VISIT": "Screening", "VISITDY": -14},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        ordering = [
            i for i in report.issues
            if i.rule_id in ("VS-001", "VS-002", "VS-003",
                             "VS-004", "VS-005", "VS-009")
        ]
        assert len(ordering) == 0

    def test_treatment_only_no_screening_no_error(self) -> None:
        """Schedule with no screening visits skips VS-001."""
        tv = _make_tv([
            {"VISIT": "Baseline", "VISITDY": 1},
            {"VISIT": "Week 4", "VISITDY": 29},
        ])
        validator = VisitScheduleValidator(tv, CLEAN_TE)
        report = validator.validate()

        vs001 = [i for i in report.issues if i.rule_id == "VS-001"]
        assert len(vs001) == 0
