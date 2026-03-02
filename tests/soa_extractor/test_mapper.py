"""Tests for UsdmMapper — USDM entity building from RawSoaTable."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.soa_extractor.mapper import UsdmMapper, _EPOCH_FOR_TYPE
from ptcv.soa_extractor.models import RawSoaTable
from ptcv.soa_extractor.resolver import SynonymResolver, VISIT_TYPES


TS = "2024-01-15T10:00:00+00:00"
RUN = "run-map-001"
SRC_SHA = "a" * 64
REG = "NCT0001"


def make_table(
    visit_headers: list[str],
    activities: list[tuple[str, list[bool]]] | None = None,
    day_headers: list[str] | None = None,
) -> RawSoaTable:
    if activities is None:
        activities = [("ECG", [True] * len(visit_headers))]
    return RawSoaTable(
        visit_headers=visit_headers,
        day_headers=day_headers or [],
        activities=activities,
        section_code="B.4",
    )


@pytest.fixture
def mapper():
    return UsdmMapper(resolver=SynonymResolver(use_spacy=False))


class TestEpochBuilding:
    def test_screening_epoch_created(self, mapper):
        table = make_table(["Screening"])
        epochs, tps, acts, insts, syns = mapper.map(
            [table], RUN, "", SRC_SHA, REG, TS
        )
        epoch_names = [e.epoch_name for e in epochs]
        assert "Screening" in epoch_names

    def test_epoch_order_is_sequential(self, mapper):
        table = make_table(["Screening", "Baseline", "Week 2"])
        epochs, *_ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        orders = [e.order for e in epochs]
        assert orders == sorted(orders)

    def test_no_duplicate_epochs(self, mapper):
        # Multiple treatment visits → only one Treatment epoch
        table = make_table(["Week 2", "Week 4", "Week 8"])
        epochs, *_ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        epoch_names = [e.epoch_name for e in epochs]
        assert len(epoch_names) == len(set(epoch_names))

    def test_all_epochs_have_run_id(self, mapper):
        table = make_table(["Screening", "Baseline"])
        epochs, *_ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        for e in epochs:
            assert e.run_id == RUN


class TestTimepointBuilding:
    def test_one_timepoint_per_visit_header(self, mapper):
        table = make_table(["Screening", "Baseline", "Week 2"])
        _, tps, _, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert len(tps) == 3

    def test_seven_required_attributes_populated(self, mapper):
        """GHERKIN: All 7 required SoA attributes present."""
        table = make_table(["Screening", "Day 1", "Week 4"])
        _, tps, _, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        for tp in tps:
            assert tp.visit_name
            assert tp.visit_type in VISIT_TYPES
            assert tp.day_offset is not None
            assert tp.window_early is not None
            assert tp.window_late is not None
            assert tp.mandatory is not None
            assert tp.extraction_timestamp_utc == TS

    def test_epoch_id_assigned(self, mapper):
        table = make_table(["Screening", "Baseline"])
        epochs, tps, _, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        epoch_ids = {e.epoch_id for e in epochs}
        for tp in tps:
            assert tp.epoch_id in epoch_ids

    def test_unscheduled_conditional_rule(self, mapper):
        """GHERKIN: Unscheduled visit has conditional_rule='PRN'."""
        table = make_table(["Unscheduled"])
        _, tps, _, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert tps[0].conditional_rule == "PRN"
        assert tps[0].mandatory is False

    def test_early_termination_conditional_rule(self, mapper):
        """GHERKIN: Early termination has conditional_rule='EARLY_TERM'."""
        table = make_table(["Early Termination"])
        _, tps, _, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert tps[0].conditional_rule == "EARLY_TERM"
        assert tps[0].mandatory is False

    def test_source_sha256_propagated(self, mapper):
        table = make_table(["Screening"])
        _, tps, _, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        for tp in tps:
            assert tp.source_sha256 == SRC_SHA

    def test_visit_types_in_9_type_taxonomy(self, mapper):
        headers = [
            "Screening", "Baseline", "Week 2", "Week 4",
            "Unscheduled", "Early Termination", "End of Study",
        ]
        table = make_table(headers)
        _, tps, _, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        for tp in tps:
            assert tp.visit_type in VISIT_TYPES


class TestActivityBuilding:
    def test_one_activity_per_row(self, mapper):
        table = make_table(
            ["Baseline"],
            activities=[
                ("ECG", [True]),
                ("Labs", [True]),
                ("Vitals", [True]),
            ],
        )
        _, _, acts, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert len(acts) == 3

    def test_activity_names_preserved(self, mapper):
        table = make_table(
            ["Baseline"],
            activities=[("Informed Consent", [True]), ("ECG", [True])],
        )
        _, _, acts, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        names = {a.activity_name for a in acts}
        assert "Informed Consent" in names
        assert "ECG" in names

    def test_ecg_type_classified(self, mapper):
        table = make_table(["Baseline"], activities=[("ECG", [True])])
        _, _, acts, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert acts[0].activity_type == "ECG"

    def test_vital_signs_classified(self, mapper):
        table = make_table(["Baseline"], activities=[("Vital Signs", [True])])
        _, _, acts, _, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert acts[0].activity_type == "Vital Signs"

    def test_no_duplicate_activities_across_tables(self, mapper):
        table1 = make_table(["Baseline"], activities=[("ECG", [True])])
        table2 = make_table(["Week 2"], activities=[("ECG", [True])])
        _, _, acts, _, _ = mapper.map(
            [table1, table2], RUN, "", SRC_SHA, REG, TS
        )
        # ECG should only appear once
        ecg_acts = [a for a in acts if a.activity_name == "ECG"]
        assert len(ecg_acts) == 1


class TestScheduledInstanceBuilding:
    def test_scheduled_instances_created(self, mapper):
        table = make_table(
            ["Baseline", "Week 2"],
            activities=[("ECG", [True, True])],
        )
        _, _, _, insts, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert len(insts) == 2

    def test_unscheduled_flag_not_in_instances(self, mapper):
        table = make_table(
            ["Baseline", "Week 2"],
            activities=[("ECG", [True, False])],
        )
        _, _, _, insts, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert len(insts) == 1  # Only the True entry

    def test_instance_links_activity_and_timepoint(self, mapper):
        table = make_table(["Baseline"], activities=[("ECG", [True])])
        _, tps, acts, insts, _ = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert insts[0].activity_id == acts[0].activity_id
        assert insts[0].timepoint_id == tps[0].timepoint_id


class TestSynonymMappings:
    def test_synonym_per_visit_header(self, mapper):
        table = make_table(["Screening", "Baseline", "Week 2"])
        _, _, _, _, syns = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        assert len(syns) == 3

    def test_synonym_original_text_matches_header(self, mapper):
        table = make_table(["Visit 0", "Pre-treatment"])
        _, _, _, _, syns = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        originals = {s.original_text for s in syns}
        # day_headers empty so headers are used directly
        assert "Visit 0" in originals or "Pre-treatment" in originals

    def test_synonym_has_run_id(self, mapper):
        table = make_table(["Screening"])
        _, _, _, _, syns = mapper.map([table], RUN, "", SRC_SHA, REG, TS)
        for s in syns:
            assert s.run_id == RUN

    def test_epoch_for_type_covers_all_types(self):
        for vtype in VISIT_TYPES:
            assert vtype in _EPOCH_FOR_TYPE, f"{vtype} missing from _EPOCH_FOR_TYPE"
