"""Tests for PTCV-253: Visit Day Resolver.

Tests verify visit label parsing, cycle-based resolution, milestone
handling, batch resolution, and cycle length detection.
"""

import pytest

from ptcv.soa_extractor.visit_day_resolver import (
    VisitDayResolver,
    ResolvedVisit,
    resolve_timepoint_days,
)


class TestExplicitDayParsing:
    """Tests for plain day number extraction."""

    def test_day_1(self):
        r = VisitDayResolver().resolve("Day 1")
        assert r.real_day == 1
        assert r.resolution_method == "explicit_day"

    def test_day_15(self):
        r = VisitDayResolver().resolve("Day 15")
        assert r.real_day == 15

    def test_day_with_d_prefix(self):
        r = VisitDayResolver().resolve("D8")
        assert r.real_day == 8

    def test_day_100(self):
        r = VisitDayResolver().resolve("Day 100")
        assert r.real_day == 100


class TestCycleDayParsing:
    """Tests for Cycle N Day M resolution."""

    def test_cycle_1_day_1(self):
        r = VisitDayResolver(cycle_length=21).resolve("Cycle 1 Day 1")
        assert r.real_day == 1
        assert r.resolution_method == "cycle_day"

    def test_cycle_2_day_1_21day(self):
        r = VisitDayResolver(cycle_length=21).resolve("Cycle 2 Day 1")
        assert r.real_day == 22

    def test_cycle_3_day_1_21day(self):
        r = VisitDayResolver(cycle_length=21).resolve("Cycle 3 Day 1")
        assert r.real_day == 43

    def test_cycle_3_day_1_28day(self):
        r = VisitDayResolver(cycle_length=28).resolve("Cycle 3 Day 1")
        assert r.real_day == 57

    def test_c3d1_shorthand(self):
        r = VisitDayResolver(cycle_length=21).resolve("C3D1")
        assert r.real_day == 43

    def test_cycle_2_day_8(self):
        r = VisitDayResolver(cycle_length=21).resolve("Cycle 2, Day 8")
        assert r.real_day == 29  # (2-1)*21 + 8

    def test_cycle_comma_separated(self):
        r = VisitDayResolver(cycle_length=21).resolve("Cycle 4, Day 1")
        assert r.real_day == 64  # (4-1)*21 + 1

    def test_cycle_only(self):
        r = VisitDayResolver(cycle_length=21).resolve("Cycle 5")
        assert r.real_day == 85  # (5-1)*21 + 1
        assert r.resolution_method == "cycle_only"


class TestWeekMonthParsing:
    """Tests for week and month conversion."""

    def test_week_1(self):
        r = VisitDayResolver().resolve("Week 1")
        assert r.real_day == 1

    def test_week_4(self):
        r = VisitDayResolver().resolve("Week 4")
        assert r.real_day == 22  # (4-1)*7 + 1

    def test_week_12(self):
        r = VisitDayResolver().resolve("Week 12")
        assert r.real_day == 78  # (12-1)*7 + 1

    def test_wk_shorthand(self):
        r = VisitDayResolver().resolve("Wk 8")
        assert r.real_day == 50

    def test_month_3(self):
        r = VisitDayResolver().resolve("Month 3")
        assert r.real_day == 61  # (3-1)*30 + 1

    def test_month_6(self):
        r = VisitDayResolver().resolve("Month 6")
        assert r.real_day == 151


class TestMilestones:
    """Tests for milestone visit resolution."""

    def test_screening(self):
        r = VisitDayResolver().resolve("Screening")
        assert r.real_day == -14
        assert r.resolution_method == "screening"

    def test_screening_visit(self):
        r = VisitDayResolver().resolve("Screening Visit")
        assert r.real_day == -14

    def test_baseline(self):
        r = VisitDayResolver().resolve("Baseline")
        assert r.real_day == 1
        assert r.resolution_method == "baseline"

    def test_baseline_day_1(self):
        r = VisitDayResolver().resolve("Baseline / Day 1")
        assert r.real_day == 1
        assert r.resolution_method == "baseline_day1"

    def test_end_of_treatment(self):
        r = VisitDayResolver().resolve("End of Treatment")
        assert r.resolution_method == "eot"

    def test_eot_abbreviation(self):
        r = VisitDayResolver().resolve("EOT")
        assert r.resolution_method == "eot"

    def test_follow_up(self):
        r = VisitDayResolver().resolve("Follow-Up")
        assert r.resolution_method == "follow_up"

    def test_post_treatment(self):
        r = VisitDayResolver().resolve("Post-treatment Visit")
        assert r.resolution_method == "follow_up"

    def test_custom_screening_day(self):
        r = VisitDayResolver(screening_day=-28).resolve("Screening")
        assert r.real_day == -28


class TestBatchResolution:
    """Tests for resolve_batch with EOT/follow-up fixup."""

    def test_chronological_order(self):
        labels = ["Cycle 2 Day 1", "Screening", "Cycle 1 Day 1", "Day 8"]
        resolved = VisitDayResolver(cycle_length=21).resolve_batch(labels)

        days = [r.real_day for r in resolved]
        assert days == sorted(days)
        assert resolved[0].real_day == -14  # Screening first

    def test_eot_resolved_after_max(self):
        labels = ["Day 1", "Day 22", "Day 43", "End of Treatment"]
        resolved = VisitDayResolver().resolve_batch(labels)

        eot = next(r for r in resolved if r.resolution_method == "eot")
        assert eot.real_day == 44  # max(43) + 1

    def test_follow_up_after_eot(self):
        labels = ["Day 1", "Day 15", "EOT", "Follow-Up"]
        resolved = VisitDayResolver().resolve_batch(labels)

        eot = next(r for r in resolved if r.resolution_method == "eot")
        fu = next(r for r in resolved if r.resolution_method == "follow_up")
        assert fu.real_day > eot.real_day

    def test_nct02853318_scenario(self):
        """Baseline should appear first, not last."""
        labels = [
            "Cycle 2 Day 1", "Cycle 3 Day 1", "Cycle 4 Day 1",
            "Cycle 5 Day 1", "Cycle 6 Day 1",
            "Every 6 Cycles Thereafter", "End of Treatment",
            "Follow-Up", "Baseline / Day 1 (Cycle 1)",
        ]
        resolved = VisitDayResolver(cycle_length=21).resolve_batch(labels)

        # Baseline should be first (Day 1)
        assert resolved[0].real_day == 1
        assert "baseline" in resolved[0].resolution_method.lower() or resolved[0].real_day == 1

    def test_empty_list(self):
        assert VisitDayResolver().resolve_batch([]) == []


class TestCycleLengthDetection:
    """Tests for detect_cycle_length from protocol text."""

    def test_21_day_cycle(self):
        assert VisitDayResolver.detect_cycle_length("Treatment in 21-day cycles") == 21

    def test_28_day_cycle(self):
        assert VisitDayResolver.detect_cycle_length("Each 28-day treatment cycle") == 28

    def test_q3w(self):
        assert VisitDayResolver.detect_cycle_length("administered q3w") == 21

    def test_q4w(self):
        assert VisitDayResolver.detect_cycle_length("dosing q4w") == 28

    def test_every_14_days(self):
        assert VisitDayResolver.detect_cycle_length("given every 14 days") == 14

    def test_no_cycle(self):
        assert VisitDayResolver.detect_cycle_length("open-label study") is None

    def test_empty_text(self):
        assert VisitDayResolver.detect_cycle_length("") is None


class TestUnresolved:
    """Tests for labels that can't be resolved."""

    def test_unknown_label(self):
        r = VisitDayResolver().resolve("Randomization")
        assert r.real_day == 0
        assert r.resolution_method == "unresolved"
        assert r.confidence == 0.0

    def test_empty_label(self):
        r = VisitDayResolver().resolve("")
        assert r.resolution_method == "unresolved"


class TestResolveTimepointDays:
    """Tests for convenience function."""

    def test_with_objects(self):
        from unittest.mock import MagicMock

        tps = [
            MagicMock(visit_name="Screening"),
            MagicMock(visit_name="Day 1"),
            MagicMock(visit_name="Day 15"),
        ]

        resolved = resolve_timepoint_days(tps)
        assert len(resolved) == 3
        assert resolved[0].real_day == -14
        assert resolved[1].real_day == 1
        assert resolved[2].real_day == 15

    def test_with_dicts(self):
        tps = [
            {"visit_name": "Week 1"},
            {"visit_name": "Week 4"},
        ]
        resolved = resolve_timepoint_days(tps)
        assert resolved[0].real_day == 1
        assert resolved[1].real_day == 22

    def test_with_protocol_text(self):
        tps = [
            {"visit_name": "Cycle 2 Day 1"},
        ]
        resolved = resolve_timepoint_days(
            tps, protocol_text="28-day cycles",
        )
        assert resolved[0].real_day == 29  # (2-1)*28 + 1

    def test_cycle_length_override(self):
        tps = [{"visit_name": "Cycle 3 Day 1"}]
        resolved = resolve_timepoint_days(tps, cycle_length=14)
        assert resolved[0].real_day == 29  # (3-1)*14 + 1


class TestCrossProtocolComparison:
    """Tests for the cross-protocol comparison scenario."""

    def test_21_vs_28_day_cycles(self):
        labels = ["Cycle 1 Day 1", "Cycle 2 Day 1", "Cycle 3 Day 1"]

        r21 = VisitDayResolver(cycle_length=21).resolve_batch(labels)
        r28 = VisitDayResolver(cycle_length=28).resolve_batch(labels)

        # Same cycle visit, different real days
        assert r21[2].real_day == 43   # Cycle 3 Day 1 @ 21-day
        assert r28[2].real_day == 57   # Cycle 3 Day 1 @ 28-day
        assert r28[2].real_day > r21[2].real_day
