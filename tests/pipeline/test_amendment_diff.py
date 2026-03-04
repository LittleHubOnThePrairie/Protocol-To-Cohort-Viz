"""Tests for protocol amendment diff engine (PTCV-59).

Covers all 5 GHERKIN acceptance criteria scenarios plus edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pandas as pd
import pytest

from ptcv.pipeline.amendment_diff import (
    AmendmentChange,
    AmendmentDiff,
    AmendmentDiffEngine,
    ProtocolVersion,
    VersionAncestry,
)
from ptcv.sdtm.models import SoaCellMetadata
from ptcv.sdtm.soa_mapper import MockSdtmDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tv(rows: list[dict]) -> pd.DataFrame:
    """Build a TV DataFrame from row dicts."""
    cols = ["STUDYID", "DOMAIN", "VISITNUM", "VISIT", "VISITDY",
            "TVSTRL", "TVENRL"]
    for r in rows:
        r.setdefault("STUDYID", "NCT001")
        r.setdefault("DOMAIN", "TV")
    return pd.DataFrame(rows, columns=cols)


def _make_te(rows: list[dict]) -> pd.DataFrame:
    """Build a TE DataFrame from row dicts."""
    cols = ["STUDYID", "DOMAIN", "ETCD", "ELEMENT", "TESTRL",
            "TEENRL", "TEDUR"]
    for r in rows:
        r.setdefault("STUDYID", "NCT001")
        r.setdefault("DOMAIN", "TE")
    return pd.DataFrame(rows, columns=cols)


def _make_dataset(
    tv_rows: list[dict],
    te_rows: list[dict] | None = None,
    soa_matrix: dict[tuple[int, str], SoaCellMetadata] | None = None,
    studyid: str = "NCT001",
) -> MockSdtmDataset:
    """Build a MockSdtmDataset for testing."""
    tv = _make_tv(tv_rows)
    te = _make_te(te_rows or [])
    ta = pd.DataFrame(columns=["STUDYID", "DOMAIN", "ARMCD", "ARM"])
    se = pd.DataFrame(columns=["STUDYID", "DOMAIN", "USUBJID"])
    dm = pd.DataFrame(columns=["ASSESSMENT", "SDTM_DOMAIN"])
    return MockSdtmDataset(
        tv=tv, ta=ta, te=te, se=se,
        domain_mapping=dm, studyid=studyid, run_id="test-run",
        soa_matrix=soa_matrix or {},
    )


def _make_cell(
    visitnum: int,
    assessment: str,
    status: str = "required",
    condition: str = "",
    category: str = "safety",
    cdash_domain: str = "VS",
    timing_window_days: tuple[int, int] = (0, 0),
) -> SoaCellMetadata:
    """Build a SoaCellMetadata for testing."""
    return SoaCellMetadata(
        visitnum=visitnum,
        assessment=assessment,
        status=status,
        condition=condition,
        category=category,
        cdash_domain=cdash_domain,
        timing_window_days=timing_window_days,
    )


# ---------------------------------------------------------------------------
# Baseline fixtures
# ---------------------------------------------------------------------------

_BASE_TV_ROWS = [
    {"VISITNUM": 1, "VISIT": "Screening", "VISITDY": -14.0,
     "TVSTRL": -3.0, "TVENRL": 0.0},
    {"VISITNUM": 2, "VISIT": "Baseline", "VISITDY": 1.0,
     "TVSTRL": -1.0, "TVENRL": 1.0},
    {"VISITNUM": 3, "VISIT": "Week 4", "VISITDY": 29.0,
     "TVSTRL": -3.0, "TVENRL": 3.0},
    {"VISITNUM": 4, "VISIT": "Week 8", "VISITDY": 57.0,
     "TVSTRL": -3.0, "TVENRL": 3.0},
    {"VISITNUM": 5, "VISIT": "Week 12", "VISITDY": 85.0,
     "TVSTRL": -3.0, "TVENRL": 3.0},
    {"VISITNUM": 6, "VISIT": "Follow-up", "VISITDY": 113.0,
     "TVSTRL": -7.0, "TVENRL": 7.0},
    {"VISITNUM": 7, "VISIT": "End of Study", "VISITDY": 141.0,
     "TVSTRL": -7.0, "TVENRL": 7.0},
    {"VISITNUM": 8, "VISIT": "Early Termination", "VISITDY": 142.0,
     "TVSTRL": 0.0, "TVENRL": 0.0},
]


# ---------------------------------------------------------------------------
# Scenario 1: Added visit detected between versions
# ---------------------------------------------------------------------------

class TestAddedVisit:
    """GHERKIN: Given v1.0 has 8 visits and v1.1 has 9 visits."""

    def test_added_visit_detected(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS)
        amended_rows = list(_BASE_TV_ROWS) + [
            {"VISITNUM": 9, "VISIT": "Follow-up 2", "VISITDY": 169.0,
             "TVSTRL": -7.0, "TVENRL": 7.0},
        ]
        ds_b = _make_dataset(amended_rows)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        added = [c for c in diff.changes if c.category == "visit_added"]
        assert len(added) == 1
        assert added[0].visitnum == 9
        assert added[0].visit_name == "Follow-up 2"
        assert "Day 169" in added[0].description

    def test_removed_visit_detected(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS)
        # Remove visit 8 (Early Termination)
        ds_b = _make_dataset(_BASE_TV_ROWS[:7])

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        removed = [c for c in diff.changes if c.category == "visit_removed"]
        assert len(removed) == 1
        assert removed[0].visitnum == 8
        assert removed[0].visit_name == "Early Termination"

    def test_added_visit_domain_is_tv(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS[:3])
        ds_b = _make_dataset(_BASE_TV_ROWS[:4])

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        added = [c for c in diff.changes if c.category == "visit_added"]
        assert all(c.domain == "TV" for c in added)


# ---------------------------------------------------------------------------
# Scenario 2: Changed visit window detected
# ---------------------------------------------------------------------------

class TestChangedWindow:
    """GHERKIN: Given Visit 4 has window +/-3 in v1.0 and +/-5 in v1.1."""

    def test_window_change_detected(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS)

        amended_rows = [dict(r) for r in _BASE_TV_ROWS]
        # Visit 4 (index 3): change window from +/-3 to +/-5
        amended_rows[3]["TVSTRL"] = -5.0
        amended_rows[3]["TVENRL"] = 5.0
        ds_b = _make_dataset(amended_rows)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        window_changes = [
            c for c in diff.changes if c.category == "window_changed"
        ]
        assert len(window_changes) == 2  # TVSTRL + TVENRL
        vnums = {c.visitnum for c in window_changes}
        assert vnums == {4}
        fields = {c.field for c in window_changes}
        assert fields == {"TVSTRL", "TVENRL"}

    def test_window_change_old_new_values(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS)

        amended_rows = [dict(r) for r in _BASE_TV_ROWS]
        amended_rows[3]["TVENRL"] = 5.0
        ds_b = _make_dataset(amended_rows)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        window_changes = [
            c for c in diff.changes if c.category == "window_changed"
        ]
        assert len(window_changes) == 1
        assert window_changes[0].old_value == "3.0"
        assert window_changes[0].new_value == "5.0"

    def test_no_window_change_when_same(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS)
        ds_b = _make_dataset([dict(r) for r in _BASE_TV_ROWS])

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        window_changes = [
            c for c in diff.changes if c.category == "window_changed"
        ]
        assert len(window_changes) == 0


# ---------------------------------------------------------------------------
# Scenario 3: Assessment status change detected
# ---------------------------------------------------------------------------

class TestStatusChanged:
    """GHERKIN: Given 'ECG' at Visit 3 is optional in v1.0 and required
    in v2.0."""

    def test_status_change_detected(self) -> None:
        matrix_a = {
            (3, "ECG"): _make_cell(3, "ECG", status="optional",
                                   cdash_domain="EG"),
        }
        matrix_b = {
            (3, "ECG"): _make_cell(3, "ECG", status="required",
                                   cdash_domain="EG"),
        }
        ds_a = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_a)
        ds_b = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_b)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "2.0")

        status_changes = [
            c for c in diff.changes if c.category == "status_changed"
        ]
        assert len(status_changes) == 1
        assert status_changes[0].assessment == "ECG"
        assert status_changes[0].visitnum == 3
        assert status_changes[0].old_value == "optional"
        assert status_changes[0].new_value == "required"

    def test_status_change_domain_is_soa_matrix(self) -> None:
        matrix_a = {
            (1, "Vital Signs"): _make_cell(1, "Vital Signs",
                                           status="required"),
        }
        matrix_b = {
            (1, "Vital Signs"): _make_cell(1, "Vital Signs",
                                           status="optional"),
        }
        ds_a = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_a)
        ds_b = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_b)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        status_changes = [
            c for c in diff.changes if c.category == "status_changed"
        ]
        assert all(c.domain == "SOA_MATRIX" for c in status_changes)

    def test_assessment_added_detected(self) -> None:
        matrix_a: dict[tuple[int, str], SoaCellMetadata] = {}
        matrix_b = {
            (3, "ECG"): _make_cell(3, "ECG", status="required",
                                   cdash_domain="EG"),
        }
        ds_a = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_a)
        ds_b = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_b)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        added = [
            c for c in diff.changes if c.category == "assessment_added"
        ]
        assert len(added) == 1
        assert added[0].assessment == "ECG"
        assert added[0].visitnum == 3

    def test_assessment_removed_detected(self) -> None:
        matrix_a = {
            (7, "PK Sampling"): _make_cell(7, "PK Sampling",
                                           status="required",
                                           cdash_domain="PC",
                                           category="efficacy"),
        }
        matrix_b: dict[tuple[int, str], SoaCellMetadata] = {}
        ds_a = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_a)
        ds_b = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_b)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        removed = [
            c for c in diff.changes if c.category == "assessment_removed"
        ]
        assert len(removed) == 1
        assert removed[0].assessment == "PK Sampling"


# ---------------------------------------------------------------------------
# Scenario 4: No changes produces empty diff
# ---------------------------------------------------------------------------

class TestNoChanges:
    """GHERKIN: Given two identical protocol versions, diff shows zero
    changes."""

    def test_identical_versions_no_changes(self) -> None:
        matrix = {
            (1, "Vital Signs"): _make_cell(1, "Vital Signs"),
            (2, "Lab Panel"): _make_cell(2, "Lab Panel", cdash_domain="LB"),
        }
        ds_a = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix)
        ds_b = _make_dataset(
            [dict(r) for r in _BASE_TV_ROWS],
            soa_matrix=dict(matrix),
        )

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.0")

        assert diff.change_count == 0
        assert not diff.has_changes
        assert diff.changes_by_category() == {}

    def test_identical_without_soa_matrix(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS)
        ds_b = _make_dataset([dict(r) for r in _BASE_TV_ROWS])

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.0")

        assert diff.change_count == 0

    def test_diff_metadata_populated(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS, studyid="NCT001")
        ds_b = _make_dataset([dict(r) for r in _BASE_TV_ROWS])

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1", registry_id="NCT001")

        assert diff.registry_id == "NCT001"
        assert diff.version_a == "1.0"
        assert diff.version_b == "1.1"
        assert diff.diff_timestamp_utc != ""


# ---------------------------------------------------------------------------
# Scenario 5: Version ancestry tracked
# ---------------------------------------------------------------------------

class TestVersionAncestry:
    """GHERKIN: Given v1.0, v1.1, v2.0 — ancestry is chronologically
    ordered with SHA-256 linkage."""

    def test_versions_ordered_chronologically(self) -> None:
        versions = [
            ProtocolVersion("NCT001", "2.0", "02",
                            "sha256_v2", "2025-06-01T00:00:00Z"),
            ProtocolVersion("NCT001", "1.0", "00",
                            "sha256_v1", "2025-01-01T00:00:00Z"),
            ProtocolVersion("NCT001", "1.1", "01",
                            "sha256_v1_1", "2025-03-15T00:00:00Z"),
        ]

        engine = AmendmentDiffEngine()
        ancestry = engine.build_ancestry(versions)

        assert ancestry.chain == ["1.0", "1.1", "2.0"]
        assert ancestry.registry_id == "NCT001"

    def test_current_is_latest_version(self) -> None:
        versions = [
            ProtocolVersion("NCT001", "1.0", "00", "sha_a"),
            ProtocolVersion("NCT001", "1.1", "01", "sha_b"),
            ProtocolVersion("NCT001", "2.0", "02", "sha_c"),
        ]

        engine = AmendmentDiffEngine()
        ancestry = engine.build_ancestry(versions)

        assert ancestry.current is not None
        assert ancestry.current.version == "2.0"
        assert ancestry.current.source_sha256 == "sha_c"

    def test_each_version_has_sha256(self) -> None:
        versions = [
            ProtocolVersion("NCT001", "1.0", "00", "abc123"),
            ProtocolVersion("NCT001", "1.1", "01", "def456"),
        ]

        engine = AmendmentDiffEngine()
        ancestry = engine.build_ancestry(versions)

        for v in ancestry.versions:
            assert v.source_sha256 != ""

    def test_empty_ancestry(self) -> None:
        engine = AmendmentDiffEngine()
        ancestry = engine.build_ancestry([])

        assert ancestry.current is None
        assert ancestry.chain == []

    def test_single_version_ancestry(self) -> None:
        versions = [
            ProtocolVersion("NCT001", "1.0", "00", "sha_only"),
        ]

        engine = AmendmentDiffEngine()
        ancestry = engine.build_ancestry(versions)

        assert ancestry.chain == ["1.0"]
        assert ancestry.current is not None
        assert ancestry.current.version == "1.0"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty DataFrames, timing changes, changes_by_category."""

    def test_empty_dataframes(self) -> None:
        ds_a = _make_dataset([])
        ds_b = _make_dataset([])

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        assert diff.change_count == 0

    def test_timing_change_detected(self) -> None:
        rows_a = [
            {"VISITNUM": 1, "VISIT": "Week 4", "VISITDY": 29.0,
             "TVSTRL": -3.0, "TVENRL": 3.0},
        ]
        rows_b = [
            {"VISITNUM": 1, "VISIT": "Week 4", "VISITDY": 42.0,
             "TVSTRL": -3.0, "TVENRL": 3.0},
        ]
        ds_a = _make_dataset(rows_a)
        ds_b = _make_dataset(rows_b)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        timing = [c for c in diff.changes if c.category == "timing_changed"]
        assert len(timing) == 1
        assert timing[0].old_value == "29.0"
        assert timing[0].new_value == "42.0"
        assert "moved from Day 29" in timing[0].description

    def test_changes_by_category_grouping(self) -> None:
        ds_a = _make_dataset(_BASE_TV_ROWS[:5])
        amended_rows = [dict(r) for r in _BASE_TV_ROWS[:5]]
        amended_rows.append(
            {"VISITNUM": 6, "VISIT": "New Visit", "VISITDY": 100.0,
             "TVSTRL": -3.0, "TVENRL": 3.0}
        )
        amended_rows[2]["VISITDY"] = 35.0  # timing change at visit 3
        ds_b = _make_dataset(amended_rows)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        grouped = diff.changes_by_category()
        assert "visit_added" in grouped
        assert "timing_changed" in grouped
        assert len(grouped["visit_added"]) == 1
        assert len(grouped["timing_changed"]) == 1

    def test_registry_id_from_studyid(self) -> None:
        ds_a = _make_dataset(
            _BASE_TV_ROWS[:2], studyid="NCT99999"
        )
        ds_b = _make_dataset(
            [dict(r) for r in _BASE_TV_ROWS[:2]], studyid="NCT99999"
        )

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "1.1")

        assert diff.registry_id == "NCT99999"

    def test_multiple_simultaneous_changes(self) -> None:
        """Multiple change categories detected in single diff."""
        matrix_a = {
            (1, "ECG"): _make_cell(1, "ECG", status="optional",
                                   cdash_domain="EG"),
        }
        matrix_b = {
            (1, "ECG"): _make_cell(1, "ECG", status="required",
                                   cdash_domain="EG"),
            (2, "Lab Panel"): _make_cell(2, "Lab Panel",
                                         cdash_domain="LB"),
        }
        amended_rows = [dict(r) for r in _BASE_TV_ROWS]
        amended_rows.append(
            {"VISITNUM": 9, "VISIT": "Extra Visit", "VISITDY": 200.0,
             "TVSTRL": -3.0, "TVENRL": 3.0}
        )
        ds_a = _make_dataset(_BASE_TV_ROWS, soa_matrix=matrix_a)
        ds_b = _make_dataset(amended_rows, soa_matrix=matrix_b)

        engine = AmendmentDiffEngine()
        diff = engine.diff(ds_a, ds_b, "1.0", "2.0")

        categories = {c.category for c in diff.changes}
        assert "visit_added" in categories
        assert "status_changed" in categories
        assert "assessment_added" in categories
        assert diff.change_count >= 3
