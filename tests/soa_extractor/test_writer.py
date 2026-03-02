"""Tests for UsdmParquetWriter — serialisation and round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.soa_extractor.models import (
    SynonymMapping,
    UsdmActivity,
    UsdmEpoch,
    UsdmScheduledInstance,
    UsdmTimepoint,
)
from ptcv.soa_extractor.writer import UsdmParquetWriter


TS = "2024-01-15T10:00:00+00:00"
RUN = "run-001"
SHA = "a" * 64
REG = "NCT0001"


def make_epoch(order: int = 1) -> UsdmEpoch:
    return UsdmEpoch(
        run_id=RUN, source_run_id="", source_sha256=SHA,
        registry_id=REG, epoch_id=f"ep-{order}",
        epoch_name="Screening", epoch_type="Screening",
        order=order, extraction_timestamp_utc=TS,
    )


def make_timepoint(tp_id: str = "tp-1", day: int = 1) -> UsdmTimepoint:
    return UsdmTimepoint(
        run_id=RUN, source_run_id="", source_sha256=SHA,
        registry_id=REG, timepoint_id=tp_id, epoch_id="ep-1",
        visit_name="Baseline", visit_type="Baseline",
        day_offset=day, window_early=0, window_late=0, mandatory=True,
        extraction_timestamp_utc=TS,
    )


def make_activity(act_id: str = "act-1") -> UsdmActivity:
    return UsdmActivity(
        run_id=RUN, source_run_id="", source_sha256=SHA,
        registry_id=REG, activity_id=act_id,
        activity_name="ECG", activity_type="ECG",
        extraction_timestamp_utc=TS,
    )


def make_instance(inst_id: str = "si-1") -> UsdmScheduledInstance:
    return UsdmScheduledInstance(
        run_id=RUN, source_run_id="", source_sha256=SHA,
        registry_id=REG, instance_id=inst_id,
        activity_id="act-1", timepoint_id="tp-1",
        scheduled=True, extraction_timestamp_utc=TS,
    )


def make_synonym(text: str = "Visit 0", label: str = "Screening") -> SynonymMapping:
    return SynonymMapping(
        run_id=RUN, original_text=text,
        canonical_label=label, method="lookup",
        confidence=0.92, review_required=False,
        extraction_timestamp_utc=TS,
    )


@pytest.fixture
def writer():
    return UsdmParquetWriter()


class TestEpochsParquet:
    def test_produces_bytes(self, writer):
        data = writer.epochs_to_parquet([make_epoch()])
        assert isinstance(data, bytes)
        assert data[:4] == b"PAR1"

    def test_empty_raises(self, writer):
        with pytest.raises(ValueError):
            writer.epochs_to_parquet([])


class TestTimepointsParquet:
    def test_round_trip(self, writer):
        tp = make_timepoint()
        data = writer.timepoints_to_parquet([tp])
        restored = writer.parquet_to_timepoints(data)
        assert len(restored) == 1
        assert restored[0].visit_name == "Baseline"
        assert restored[0].day_offset == 1
        assert restored[0].mandatory is True

    def test_seven_required_fields_non_null(self, writer):
        tp = make_timepoint()
        data = writer.timepoints_to_parquet([tp])
        restored = writer.parquet_to_timepoints(data)
        r = restored[0]
        # Seven required attributes
        assert r.visit_name
        assert r.visit_type
        assert r.day_offset is not None
        assert r.window_early is not None
        assert r.window_late is not None
        assert r.mandatory is not None
        assert r.extraction_timestamp_utc

    def test_missing_visit_name_raises(self, writer):
        tp = make_timepoint()
        tp.visit_name = ""
        with pytest.raises(ValueError, match="visit_name"):
            writer.timepoints_to_parquet([tp])

    def test_missing_timestamp_raises(self, writer):
        tp = make_timepoint()
        tp.extraction_timestamp_utc = ""
        with pytest.raises(ValueError, match="extraction_timestamp_utc"):
            writer.timepoints_to_parquet([tp])

    def test_review_required_preserved(self, writer):
        tp = make_timepoint()
        tp.review_required = True
        data = writer.timepoints_to_parquet([tp])
        restored = writer.parquet_to_timepoints(data)
        assert restored[0].review_required is True

    def test_conditional_rule_preserved(self, writer):
        tp = make_timepoint()
        tp.conditional_rule = "PRN"
        data = writer.timepoints_to_parquet([tp])
        restored = writer.parquet_to_timepoints(data)
        assert restored[0].conditional_rule == "PRN"

    def test_empty_raises(self, writer):
        with pytest.raises(ValueError):
            writer.timepoints_to_parquet([])


class TestActivitiesParquet:
    def test_produces_bytes(self, writer):
        data = writer.activities_to_parquet([make_activity()])
        assert isinstance(data, bytes)

    def test_empty_raises(self, writer):
        with pytest.raises(ValueError):
            writer.activities_to_parquet([])


class TestInstancesParquet:
    def test_produces_bytes(self, writer):
        data = writer.instances_to_parquet([make_instance()])
        assert isinstance(data, bytes)

    def test_empty_raises(self, writer):
        with pytest.raises(ValueError):
            writer.instances_to_parquet([])


class TestSynonymsParquet:
    def test_round_trip(self, writer):
        sm = make_synonym()
        data = writer.synonyms_to_parquet([sm])
        restored = writer.parquet_to_synonyms(data)
        assert len(restored) == 1
        assert restored[0].original_text == "Visit 0"
        assert restored[0].canonical_label == "Screening"
        assert restored[0].method == "lookup"

    def test_confidence_float32_precision(self, writer):
        sm = make_synonym()
        sm.confidence = 0.92
        data = writer.synonyms_to_parquet([sm])
        restored = writer.parquet_to_synonyms(data)
        assert abs(restored[0].confidence - 0.92) < 0.01

    def test_empty_raises(self, writer):
        with pytest.raises(ValueError):
            writer.synonyms_to_parquet([])
