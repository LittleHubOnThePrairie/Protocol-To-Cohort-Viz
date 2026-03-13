"""Tests for SynonymResolver."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.soa_extractor.resolver import SynonymResolver, VISIT_TYPES


@pytest.fixture
def resolver():
    return SynonymResolver(use_spacy=False)


class TestDirectLookup:
    def test_screening_direct(self, resolver):
        r = resolver.resolve("Screening")
        assert r.visit_type == "Screening"
        assert r.method == "lookup"
        assert r.confidence >= 0.90

    def test_visit_zero_maps_to_screening(self, resolver):
        """GHERKIN: 'Visit 0' → canonical label 'Screening'"""
        r = resolver.resolve("Visit 0")
        assert r.visit_type == "Screening"

    def test_pre_treatment_maps_to_screening(self, resolver):
        """GHERKIN: 'Pre-treatment' → canonical label 'Screening'"""
        r = resolver.resolve("Pre-treatment")
        assert r.visit_type == "Screening"

    def test_baseline_direct(self, resolver):
        r = resolver.resolve("Baseline")
        assert r.visit_type == "Baseline"
        assert r.day_offset == 1

    def test_day1_maps_to_baseline(self, resolver):
        r = resolver.resolve("Day 1")
        assert r.visit_type == "Baseline"

    def test_unscheduled_visit(self, resolver):
        """GHERKIN: Unscheduled visit has conditional_rule='PRN'"""
        r = resolver.resolve("Unscheduled Visit")
        assert r.visit_type == "Unscheduled"
        assert r.conditional_rule == "PRN"
        assert r.mandatory is False

    def test_early_termination(self, resolver):
        """GHERKIN: Early termination has conditional_rule='EARLY_TERM'"""
        r = resolver.resolve("Early Termination")
        assert r.visit_type == "Early Termination"
        assert r.conditional_rule == "EARLY_TERM"
        assert r.mandatory is False

    def test_end_of_study(self, resolver):
        r = resolver.resolve("End of Study")
        assert r.visit_type == "End of Study"

    def test_eot_abbreviation(self, resolver):
        r = resolver.resolve("eot")
        assert r.visit_type == "End of Study"

    def test_ltfu_abbreviation(self, resolver):
        r = resolver.resolve("ltfu")
        assert r.visit_type == "Long-term Follow-up"


class TestRegexTemporal:
    def test_week_2_maps_to_day_8(self, resolver):
        r = resolver.resolve("Week 2")
        assert r.visit_type == "Treatment"
        assert r.day_offset == 8
        assert r.method == "regex"

    def test_week_4_day_offset(self, resolver):
        r = resolver.resolve("Week 4")
        assert r.day_offset == 22  # (4-1)*7 + 1

    def test_day_15_with_window(self, resolver):
        r = resolver.resolve("Day 15 ± 3")
        assert r.day_offset == 15
        assert r.window_early == 3
        assert r.window_late == 3

    def test_day_range_screening(self, resolver):
        r = resolver.resolve("Days -14 to -1")
        assert r.visit_type == "Screening"
        assert r.day_offset == -14
        assert r.window_late == 13

    def test_cycle_day(self, resolver):
        r = resolver.resolve("Cycle 2 Day 1")
        assert r.visit_type == "Treatment"
        assert r.repeat_cycle == "C2"
        assert r.day_offset == 1

    def test_cycle_1_day_1_is_baseline(self, resolver):
        r = resolver.resolve("Cycle 1 Day 1")
        assert r.visit_type == "Baseline"
        assert r.repeat_cycle == "C1"

    def test_year_1_long_term_fu(self, resolver):
        r = resolver.resolve("Year 1")
        assert r.visit_type == "Long-term Follow-up"
        assert r.day_offset == 365

    def test_month_regex(self, resolver):
        r = resolver.resolve("Month 3")
        assert r.day_offset == 90


class TestResolveToMapping:
    def test_mapping_fields_populated(self, resolver):
        resolved, mapping = resolver.resolve_to_mapping(
            "Visit 0", run_id="run-1", timestamp="2024-01-15T00:00:00+00:00"
        )
        assert mapping.original_text == "Visit 0"
        assert mapping.canonical_label == "Screening"
        assert mapping.method == "lookup"
        assert 0.0 <= mapping.confidence <= 1.0
        assert mapping.run_id == "run-1"

    def test_low_confidence_sets_review_required(self, resolver):
        """Default fallback produces low confidence → review required."""
        resolved, mapping = resolver.resolve_to_mapping(
            "ZZZ unknown header", run_id="r", timestamp="t"
        )
        if mapping.confidence < 0.80:
            assert mapping.review_required is True

    def test_high_confidence_no_review(self, resolver):
        _, mapping = resolver.resolve_to_mapping(
            "Screening", run_id="r", timestamp="t"
        )
        assert mapping.review_required is False


class TestVisitNumberRegex:
    """PTCV-131: Visit number patterns (V2, Visit 3, V4/ET)."""

    def test_v2_maps_to_day_2(self, resolver):
        r = resolver.resolve("V2")
        assert r.visit_type == "Treatment"
        assert r.day_offset == 2
        assert r.method == "regex"

    def test_v3_maps_to_day_3(self, resolver):
        r = resolver.resolve("V3")
        assert r.day_offset == 3

    def test_visit_4_full_word(self, resolver):
        r = resolver.resolve("Visit 4")
        assert r.day_offset == 4
        assert r.visit_type == "Treatment"

    def test_v1_still_baseline_via_lookup(self, resolver):
        """V1 should match lookup first (Baseline, day 1)."""
        r = resolver.resolve("V1")
        assert r.visit_type == "Baseline"
        assert r.day_offset == 1

    def test_v0_screening_via_lookup(self, resolver):
        r = resolver.resolve("V0")
        assert r.visit_type == "Screening"

    def test_v4_et_compound_header(self, resolver):
        """'V4/ET' should extract visit number 4."""
        r = resolver.resolve("V4/ET")
        assert r.day_offset == 4

    def test_visit_5_day_5(self, resolver):
        r = resolver.resolve("Visit 5")
        assert r.day_offset == 5

    def test_confidence_lower_than_full_regex(self, resolver):
        """Visit number regex has lower confidence (synthetic offset)."""
        r = resolver.resolve("V3")
        assert r.confidence < 0.85


class TestDefaultFallback:
    def test_unknown_header_returns_treatment(self, resolver):
        r = resolver.resolve("ZZZ completely unknown")
        assert r.visit_type == "Treatment"
        assert r.method == "default"
        assert r.confidence < 0.80

    def test_all_visit_types_in_taxonomy(self):
        assert len(VISIT_TYPES) == 9
        assert "Unscheduled" in VISIT_TYPES
        assert "Early Termination" in VISIT_TYPES
        assert "Long-term Follow-up" in VISIT_TYPES
