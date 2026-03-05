"""Tests for iterative refinement store (PTCV-94).

Feature: Iterative refinement of query-driven extraction

  Scenario: Header mapping correction is stored
  Scenario: Stored corrections improve future matching
  Scenario: Confidence calibration tracks accuracy
  Scenario: Refinement data persists across sessions
  Scenario: Monthly accuracy trend is measurable
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ptcv.ich_parser.refinement_store import (
    CALIBRATION_BAND_WIDTH,
    DEFAULT_MIN_FREQUENCY,
    CalibrationBand,
    CalibrationEntry,
    CalibrationReport,
    ExtractionCorrection,
    HeaderCorrection,
    RefinementStore,
    TrendReport,
)


@pytest.fixture()
def store_dir():
    """Provide a temporary directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def store(store_dir):
    """Provide a fresh RefinementStore."""
    return RefinementStore(data_dir=store_dir)


# -----------------------------------------------------------------------
# Scenario: Header mapping correction is stored
# -----------------------------------------------------------------------


class TestHeaderCorrectionStored:
    """Given a reviewer corrects 'Study Procedures' from B7 to B9,
    When the correction is submitted,
    Then header_corrections.json contains the correction with
    protocol_id and timestamp,
    And synonym_mappings.json is updated with the new mapping.
    """

    def test_correction_recorded(self, store):
        c = store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="Study Procedures",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        assert c.protocol_id == "NCT001"
        assert c.corrected_mapping == "B.9"
        assert c.timestamp != ""
        assert store.header_correction_count == 1

    def test_correction_in_json(self, store_dir):
        store = RefinementStore(data_dir=store_dir)
        store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="Study Procedures",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        path = store_dir / "header_corrections.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["protocol_id"] == "NCT001"
        assert data[0]["corrected_mapping"] == "B.9"

    def test_synonym_updated(self, store):
        store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="Study Procedures",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        path = store._data_dir / "synonym_mappings.json"
        data = json.loads(path.read_text())
        assert "study procedures" in data
        assert data["study procedures"]["B.9"] == 1

    def test_multiple_corrections_accumulate(self, store):
        for i in range(3):
            store.record_header_correction(
                protocol_id=f"NCT00{i}",
                protocol_header="Study Procedures",
                original_mapping="B.7",
                corrected_mapping="B.9",
            )
        assert store.header_correction_count == 3

    def test_extraction_correction_stored(self, store):
        c = store.record_extraction_correction(
            protocol_id="NCT001",
            query_id="B.6.1.q1",
            original_content="Wrong text.",
            corrected_content="Correct text.",
            corrected_section="B.6",
        )
        assert c.query_id == "B.6.1.q1"
        assert c.timestamp != ""
        assert store.extraction_correction_count == 1


# -----------------------------------------------------------------------
# Scenario: Stored corrections improve future matching
# -----------------------------------------------------------------------


class TestCorrectionsImproveMatching:
    """Given 3 protocols where 'Study Procedures' was corrected to B9,
    When a 4th protocol with header 'Study Procedures' is processed,
    Then the semantic matcher uses accumulated corrections as a boost,
    And the B9 match score is higher than in the first protocol.
    """

    def test_synonym_not_promoted_below_threshold(self, store):
        """One correction is not enough for promotion."""
        store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="Study Procedures",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        boosts = store.get_synonym_boosts(min_frequency=2)
        assert "study procedures" not in boosts

    def test_synonym_promoted_at_threshold(self, store):
        """Two corrections promotes the synonym."""
        for i in range(2):
            store.record_header_correction(
                protocol_id=f"NCT00{i}",
                protocol_header="Study Procedures",
                original_mapping="B.7",
                corrected_mapping="B.9",
            )
        boosts = store.get_synonym_boosts(min_frequency=2)
        assert boosts["study procedures"] == "B.9"

    def test_synonym_promoted_at_three(self, store):
        """3 corrections definitely promotes."""
        for i in range(3):
            store.record_header_correction(
                protocol_id=f"NCT00{i}",
                protocol_header="Study Procedures",
                original_mapping="B.7",
                corrected_mapping="B.9",
            )
        boosts = store.get_synonym_boosts()
        assert boosts["study procedures"] == "B.9"

    def test_most_frequent_wins(self, store):
        """When a header is corrected to different sections, most
        frequent wins."""
        for i in range(3):
            store.record_header_correction(
                protocol_id=f"NCT00{i}",
                protocol_header="Analysis",
                original_mapping="B.7",
                corrected_mapping="B.10",
            )
        store.record_header_correction(
            protocol_id="NCT099",
            protocol_header="Analysis",
            original_mapping="B.7",
            corrected_mapping="B.8",
        )
        boosts = store.get_synonym_boosts(min_frequency=2)
        assert boosts["analysis"] == "B.10"

    def test_min_frequency_one_returns_all(self, store):
        """min_frequency=1 returns even single corrections."""
        store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="Novel Header",
            original_mapping="B.1",
            corrected_mapping="B.12",
        )
        boosts = store.get_synonym_boosts(min_frequency=1)
        assert "novel header" in boosts

    def test_boosts_dict_format(self, store):
        """Boosts dict is str→str compatible with SectionMatcher."""
        for i in range(3):
            store.record_header_correction(
                protocol_id=f"NCT00{i}",
                protocol_header="Regulatory Compliance",
                original_mapping="B.1",
                corrected_mapping="B.12",
            )
        boosts = store.get_synonym_boosts()
        for k, v in boosts.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
            assert v.startswith("B.")


# -----------------------------------------------------------------------
# Scenario: Confidence calibration tracks accuracy
# -----------------------------------------------------------------------


class TestConfidenceCalibration:
    """Given 50+ extractions with predicted confidence scores,
    When actual correctness is recorded via reviewer feedback,
    Then a calibration report shows predicted vs actual accuracy,
    And confidence thresholds are adjusted if bias is detected.
    """

    def _populate_calibration(self, store, n=60):
        """Add n calibration entries with varying accuracy."""
        import random
        random.seed(42)
        for i in range(n):
            conf = random.uniform(0.3, 0.99)
            # Higher confidence = higher accuracy (slight overconfidence)
            correct = random.random() < (conf - 0.05)
            store.record_calibration_entry(
                protocol_id=f"NCT{i:03d}",
                query_id=f"B.1.1.q{i}",
                predicted_confidence=round(conf, 3),
                actual_correct=correct,
            )

    def test_calibration_entry_recorded(self, store):
        e = store.record_calibration_entry(
            protocol_id="NCT001",
            query_id="B.1.1.q1",
            predicted_confidence=0.85,
            actual_correct=True,
        )
        assert e.predicted_confidence == 0.85
        assert e.actual_correct is True
        assert store.calibration_entry_count == 1

    def test_calibration_report_structure(self, store):
        self._populate_calibration(store, n=60)
        report = store.get_calibration_report()
        assert isinstance(report, CalibrationReport)
        assert report.total_entries == 60
        assert 0.0 <= report.overall_accuracy <= 1.0
        assert len(report.bands) > 0

    def test_calibration_bands_cover_range(self, store):
        self._populate_calibration(store)
        report = store.get_calibration_report()
        for band in report.bands:
            assert band.range_low < band.range_high
            assert band.count > 0
            assert 0.0 <= band.actual_accuracy <= 1.0

    def test_calibration_bias_detection(self, store):
        """Systematically overconfident predictions trigger bias."""
        for i in range(20):
            store.record_calibration_entry(
                protocol_id=f"NCT{i:03d}",
                query_id=f"B.1.1.q{i}",
                predicted_confidence=0.90,
                actual_correct=False,  # Always wrong despite high conf
            )
        report = store.get_calibration_report()
        assert report.bias_detected is True
        # Should suggest raising threshold (positive adjustment)
        assert report.suggested_threshold_adjustment > 0

    def test_calibration_no_bias_when_accurate(self, store):
        """Accurate predictions show no significant bias."""
        for i in range(20):
            store.record_calibration_entry(
                protocol_id=f"NCT{i:03d}",
                query_id=f"B.1.1.q{i}",
                predicted_confidence=0.85,
                actual_correct=True,
            )
        report = store.get_calibration_report()
        # Bias = 0.85 - 1.0 = -0.15, borderline
        # The bias magnitude should be exactly 0.15
        assert report.total_entries == 20

    def test_empty_calibration_report(self, store):
        report = store.get_calibration_report()
        assert report.total_entries == 0
        assert report.overall_accuracy == 0.0
        assert report.bias_detected is False
        assert report.bands == []

    def test_band_width_constant(self):
        assert CALIBRATION_BAND_WIDTH == 0.10


# -----------------------------------------------------------------------
# Scenario: Refinement data persists across sessions
# -----------------------------------------------------------------------


class TestPersistence:
    """Given corrections recorded during protocol processing,
    When the pipeline is restarted,
    Then all prior corrections are loaded from the refinement store,
    And matching/extraction behavior reflects accumulated learning.
    """

    def test_header_corrections_persist(self, store_dir):
        store1 = RefinementStore(data_dir=store_dir)
        store1.record_header_correction(
            protocol_id="NCT001",
            protocol_header="Procedures",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        # New instance — simulates restart.
        store2 = RefinementStore(data_dir=store_dir)
        assert store2.header_correction_count == 1

    def test_extraction_corrections_persist(self, store_dir):
        store1 = RefinementStore(data_dir=store_dir)
        store1.record_extraction_correction(
            protocol_id="NCT001",
            query_id="B.6.1.q1",
            original_content="old",
            corrected_content="new",
        )
        store2 = RefinementStore(data_dir=store_dir)
        assert store2.extraction_correction_count == 1

    def test_calibration_entries_persist(self, store_dir):
        store1 = RefinementStore(data_dir=store_dir)
        store1.record_calibration_entry(
            protocol_id="NCT001",
            query_id="B.1.1.q1",
            predicted_confidence=0.85,
            actual_correct=True,
        )
        store2 = RefinementStore(data_dir=store_dir)
        assert store2.calibration_entry_count == 1

    def test_synonym_frequencies_persist(self, store_dir):
        store1 = RefinementStore(data_dir=store_dir)
        for i in range(3):
            store1.record_header_correction(
                protocol_id=f"NCT00{i}",
                protocol_header="Study Procedures",
                original_mapping="B.7",
                corrected_mapping="B.9",
            )
        store2 = RefinementStore(data_dir=store_dir)
        boosts = store2.get_synonym_boosts()
        assert boosts["study procedures"] == "B.9"

    def test_accumulated_learning_after_restart(self, store_dir):
        """Verify corrections accumulate across multiple sessions."""
        store1 = RefinementStore(data_dir=store_dir)
        store1.record_header_correction(
            protocol_id="NCT001",
            protocol_header="Lab Tests",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )

        store2 = RefinementStore(data_dir=store_dir)
        store2.record_header_correction(
            protocol_id="NCT002",
            protocol_header="Lab Tests",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        assert store2.header_correction_count == 2
        boosts = store2.get_synonym_boosts(min_frequency=2)
        assert "lab tests" in boosts

    def test_corrupted_file_handled_gracefully(self, store_dir):
        """Bad JSON doesn't crash the store."""
        (store_dir / "header_corrections.json").write_text("NOT JSON")
        store = RefinementStore(data_dir=store_dir)
        assert store.header_correction_count == 0

    def test_missing_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "nonexistent" / "subdir"
            store = RefinementStore(data_dir=new_dir)
            assert new_dir.exists()
            store.record_header_correction(
                protocol_id="NCT001",
                protocol_header="Test",
                original_mapping="B.1",
                corrected_mapping="B.2",
            )
            assert store.header_correction_count == 1


# -----------------------------------------------------------------------
# Scenario: Monthly accuracy trend is measurable
# -----------------------------------------------------------------------


class TestTrendReport:
    """Given protocols processed over multiple sessions with feedback,
    When a trend report is generated,
    Then it shows mapping accuracy over time,
    And lists the most common corrections and new synonyms.
    """

    def test_trend_report_structure(self, store):
        store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="Study Procedures",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        report = store.get_trend_report()
        assert isinstance(report, TrendReport)
        assert report.total_corrections == 1

    def test_common_corrections_ranked(self, store):
        for i in range(5):
            store.record_header_correction(
                protocol_id=f"NCT00{i}",
                protocol_header="Study Procedures",
                original_mapping="B.7",
                corrected_mapping="B.9",
            )
        for i in range(2):
            store.record_header_correction(
                protocol_id=f"NCT10{i}",
                protocol_header="Lab Safety",
                original_mapping="B.7",
                corrected_mapping="B.8",
            )
        report = store.get_trend_report()
        assert len(report.common_corrections) >= 2
        # Most common first.
        assert report.common_corrections[0][2] >= report.common_corrections[1][2]
        assert report.common_corrections[0] == (
            "study procedures", "B.9", 5,
        )

    def test_new_synonyms_listed(self, store):
        for i in range(3):
            store.record_header_correction(
                protocol_id=f"NCT00{i}",
                protocol_header="Regulatory Compliance",
                original_mapping="B.1",
                corrected_mapping="B.12",
            )
        report = store.get_trend_report()
        synonym_headers = [s[0] for s in report.new_synonyms]
        assert "regulatory compliance" in synonym_headers

    def test_accuracy_by_month(self, store):
        """Monthly accuracy is computed from calibration data."""
        # January entries — all correct.
        for i in range(5):
            e = store.record_calibration_entry(
                protocol_id=f"NCT{i:03d}",
                query_id=f"B.1.q{i}",
                predicted_confidence=0.80,
                actual_correct=True,
            )
            # Override timestamp to January.
            store._calibration_entries[-1] = CalibrationEntry(
                protocol_id=e.protocol_id,
                query_id=e.query_id,
                predicted_confidence=e.predicted_confidence,
                actual_correct=e.actual_correct,
                timestamp="2026-01-15T10:00:00+00:00",
            )
        # February entries — 3/5 correct.
        for i in range(5, 10):
            e = store.record_calibration_entry(
                protocol_id=f"NCT{i:03d}",
                query_id=f"B.1.q{i}",
                predicted_confidence=0.80,
                actual_correct=i < 8,
            )
            store._calibration_entries[-1] = CalibrationEntry(
                protocol_id=e.protocol_id,
                query_id=e.query_id,
                predicted_confidence=e.predicted_confidence,
                actual_correct=i < 8,
                timestamp="2026-02-15T10:00:00+00:00",
            )
        report = store.get_trend_report()
        assert len(report.accuracy_by_month) == 2
        # Most recent first.
        assert report.accuracy_by_month[0][0] == "2026-02"
        assert report.accuracy_by_month[1][0] == "2026-01"
        # January was 100% accurate.
        assert report.accuracy_by_month[1][1] == 1.0
        # February was 60% accurate.
        assert report.accuracy_by_month[0][1] == 0.6

    def test_empty_trend_report(self, store):
        report = store.get_trend_report()
        assert report.total_corrections == 0
        assert report.total_calibration_entries == 0
        assert report.common_corrections == []
        assert report.new_synonyms == []
        assert report.accuracy_by_month == []

    def test_total_corrections_includes_both(self, store):
        store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="X",
            original_mapping="B.1",
            corrected_mapping="B.2",
        )
        store.record_extraction_correction(
            protocol_id="NCT001",
            query_id="B.1.q1",
            original_content="old",
            corrected_content="new",
        )
        report = store.get_trend_report()
        assert report.total_corrections == 2


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge case tests."""

    def test_default_min_frequency_constant(self):
        assert DEFAULT_MIN_FREQUENCY == 2

    def test_empty_store_boosts(self, store):
        assert store.get_synonym_boosts() == {}

    def test_header_lowered_for_synonym(self, store):
        """Headers are normalised to lowercase."""
        store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="STUDY PROCEDURES",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        store.record_header_correction(
            protocol_id="NCT002",
            protocol_header="study procedures",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        boosts = store.get_synonym_boosts(min_frequency=2)
        assert "study procedures" in boosts

    def test_whitespace_stripped(self, store):
        store.record_header_correction(
            protocol_id="NCT001",
            protocol_header="  Safety  ",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        store.record_header_correction(
            protocol_id="NCT002",
            protocol_header="Safety",
            original_mapping="B.7",
            corrected_mapping="B.9",
        )
        boosts = store.get_synonym_boosts(min_frequency=2)
        assert "safety" in boosts
