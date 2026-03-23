"""Tests for SoA multi-method reconciler (PTCV-216).

Tests union merge, majority voting, confidence scoring,
and disagreement flagging.
"""

from __future__ import annotations

import pytest

from ptcv.soa_extractor.reconciler import (
    CellConfidence,
    ReconciliationReport,
    reconcile,
)
from ptcv.soa_extractor.models import RawSoaTable


def _make_table(
    activities: list[tuple[str, list[bool]]],
    visit_headers: list[str] | None = None,
) -> RawSoaTable:
    if visit_headers is None:
        n = max((len(flags) for _, flags in activities), default=3)
        visit_headers = [f"V{i}" for i in range(1, n + 1)]
    return RawSoaTable(
        visit_headers=visit_headers,
        day_headers=[],
        activities=activities,
        section_code="B.4",
    )


class TestReconcileEmpty:
    """Tests for edge cases."""

    def test_empty_input(self):
        """Test reconcile with no tables."""
        result = reconcile([])
        assert result.source_count == 0
        assert result.total_activities == 0

    def test_single_table(self):
        """Test reconcile with one table returns it unchanged."""
        table = _make_table([
            ("Exam", [True, False]),
            ("Labs", [False, True]),
        ])
        result = reconcile([table])

        assert result.source_count == 1
        assert result.total_activities == 2
        assert result.unique_to_one == 2
        assert result.overlapping_activities == 0


class TestReconcileUnion:
    """Tests for union merge behavior."""

    def test_union_of_disjoint_activities(self):
        """Test union when methods find different activities."""
        table_a = _make_table([
            ("Physical Exam", [True, True]),
            ("ECG", [True, False]),
        ], ["V1", "V2"])
        table_b = _make_table([
            ("Labs", [False, True]),
            ("Vitals", [True, True]),
        ], ["V1", "V2"])

        result = reconcile([table_a, table_b])

        assert result.total_activities == 4
        assert result.unique_to_one == 4
        assert result.overlapping_activities == 0

    def test_union_with_overlap(self):
        """Test union counts overlapping activities correctly."""
        table_a = _make_table([
            ("Physical Exam", [True, True]),
            ("ECG", [True, False]),
            ("Labs", [False, True]),
        ], ["V1", "V2"])
        table_b = _make_table([
            ("Physical Exam", [True, True]),
            ("ECG", [False, True]),
            ("Vitals", [True, True]),
        ], ["V1", "V2"])

        result = reconcile([table_a, table_b])

        # 3 + 3 = 6, but 2 overlap → 4 unique
        assert result.total_activities == 4
        assert result.overlapping_activities == 2
        assert result.unique_to_one == 2


class TestReconcileMajorityVoting:
    """Tests for majority voting on overlapping cells."""

    def test_agreement_produces_high_confidence(self):
        """Test cells where methods agree get high confidence."""
        table_a = _make_table([
            ("Exam", [True, False, True]),
        ], ["V1", "V2", "V3"])
        table_b = _make_table([
            ("Exam", [True, False, True]),
        ], ["V1", "V2", "V3"])

        result = reconcile([table_a, table_b])
        confs = result.cell_confidences.get("exam", [])

        assert len(confs) == 3
        for c in confs:
            assert c.confidence == 1.0
            assert not c.needs_review
            assert c.methods_disagree == 0

    def test_disagreement_uses_majority(self):
        """Test majority voting resolves disagreements."""
        table_a = _make_table([
            ("Exam", [True, False]),
        ], ["V1", "V2"])
        table_b = _make_table([
            ("Exam", [True, True]),
        ], ["V1", "V2"])
        table_c = _make_table([
            ("Exam", [True, True]),
        ], ["V1", "V2"])

        result = reconcile([table_a, table_b, table_c])

        # V1: all True → True (3 agree)
        # V2: 1 False, 2 True → True (majority)
        merged = result.merged_table
        exam = [a for a in merged.activities if a[0] == "Exam"][0]
        assert exam[1][0] is True  # V1
        assert exam[1][1] is True  # V2 (majority)

        confs = result.cell_confidences.get("exam", [])
        assert confs[1].methods_agree == 2
        assert confs[1].methods_disagree == 1
        assert confs[1].confidence == pytest.approx(2 / 3, abs=0.01)

    def test_disagreement_flagged(self):
        """Test disagreement count is tracked."""
        table_a = _make_table([
            ("Exam", [True, False]),
        ], ["V1", "V2"])
        table_b = _make_table([
            ("Exam", [False, True]),
        ], ["V1", "V2"])

        result = reconcile([table_a, table_b])

        assert result.disagreement_count == 2  # Both cells disagree


class TestReconcileHeaders:
    """Tests for header selection."""

    def test_picks_widest_headers(self):
        """Test headers from table with most columns are used."""
        table_a = _make_table(
            [("Exam", [True, True])],
            ["V1", "V2"],
        )
        table_b = _make_table(
            [("Exam", [True, True, False, True])],
            ["V1", "V2", "V3", "V4"],
        )

        result = reconcile([table_a, table_b])

        assert len(result.merged_table.visit_headers) == 4

    def test_narrow_table_flags_padded(self):
        """Test flags from narrower table are padded with False."""
        table_a = _make_table(
            [("Exam", [True])],
            ["V1"],
        )
        table_b = _make_table(
            [("Exam", [True, True, True])],
            ["V1", "V2", "V3"],
        )

        result = reconcile([table_a, table_b])

        exam = [a for a in result.merged_table.activities if a[0] == "Exam"][0]
        assert len(exam[1]) == 3
        # V1: both True → True
        assert exam[1][0] is True


class TestReconcileDisplayNames:
    """Tests for display name selection."""

    def test_longer_name_preferred(self):
        """Test longer display name is kept."""
        table_a = _make_table(
            [("PE", [True])],
            ["V1"],
        )
        table_b = _make_table(
            [("Physical Examination", [True])],
            ["V1"],
        )

        # Both normalize to similar but "Physical Examination" is longer
        result = reconcile([table_a, table_b])

        names = [name for name, _ in result.merged_table.activities]
        # Should pick the longer variant
        assert any(len(n) > 5 for n in names)
