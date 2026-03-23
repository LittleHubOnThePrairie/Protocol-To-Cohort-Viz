"""Tests for multi-source SoA extraction (PTCV-260).

GHERKIN Scenarios:
  Feature: Multi-source SoA extraction

    Scenario: Narrative assessment schedule extracted
    Scenario: Table and narrative results merged
    Scenario: Diagram provides additional assessment-visit pairs
    Scenario: Multi-stream agreement boosts confidence
    Scenario: Deduplicated across streams
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.soa_extractor.narrative_extractor import (
    AssessmentVisitPair,
    extract_from_narrative,
)
from ptcv.soa_extractor.diagram_extractor import extract_from_diagrams
from ptcv.soa_extractor.soa_merger import (
    MergedAssessment,
    merge_streams,
    pairs_to_raw_table,
)


# ---------------------------------------------------------------------------
# Scenario: Narrative assessment schedule extracted
# ---------------------------------------------------------------------------


class TestNarrativeExtraction:
    """Extract assessment-visit pairs from prose text."""

    def test_vital_signs_at_four_timepoints(self):
        """'Vital signs at Screening, Day 1, Day 15, and Day 29'."""
        texts = [
            "Vital signs at Screening, Day 1, Day 15, and Day 29."
        ]
        pairs = extract_from_narrative(texts)

        names = [(p.assessment_name, p.visit_label) for p in pairs]
        assert ("Vital signs", "Screening") in names
        assert ("Vital signs", "Day 1") in names
        assert ("Vital signs", "Day 15") in names
        assert ("Vital signs", "Day 29") in names
        assert len(pairs) == 4

    def test_source_tagged_as_narrative(self):
        texts = ["ECG at Screening and Day 1."]
        pairs = extract_from_narrative(texts)
        assert all(p.source == "narrative" for p in pairs)

    def test_performed_at_pattern(self):
        """'X will be performed at ...' pattern."""
        texts = [
            "Physical Exam will be performed at Screening, "
            "Baseline, and End of Treatment."
        ]
        pairs = extract_from_narrative(texts)
        assert len(pairs) == 3
        assert any(p.visit_label == "End of Treatment" for p in pairs)

    def test_every_n_cycles(self):
        """'X every 3 cycles' pattern."""
        texts = ["Tumor Assessment every 3 cycles."]
        pairs = extract_from_narrative(texts)
        assert len(pairs) >= 1
        assert pairs[0].visit_label == "Every 3 cycles"

    def test_at_each_visit(self):
        """'X at each visit' pattern."""
        texts = ["Vital Signs at each visit."]
        pairs = extract_from_narrative(texts)
        assert len(pairs) >= 1
        assert pairs[0].visit_label == "Each Visit"

    def test_empty_text(self):
        assert extract_from_narrative([]) == []
        assert extract_from_narrative(["", "  "]) == []

    def test_dedup_within_text(self):
        """Same pair mentioned twice is only counted once."""
        texts = [
            "ECG at Screening. ECG at Screening for safety."
        ]
        pairs = extract_from_narrative(texts)
        ecg_screening = [
            p for p in pairs
            if "ecg" in p.assessment_name.lower()
            and "screening" in p.visit_label.lower()
        ]
        assert len(ecg_screening) == 1


# ---------------------------------------------------------------------------
# Scenario: Diagram provides additional assessment-visit pairs
# ---------------------------------------------------------------------------


class TestDiagramExtraction:
    """Extract pairs from diagram node labels."""

    def test_assessment_with_visit_in_label(self):
        """Node label with both assessment and visit."""
        diagrams = [
            {
                "diagram_type": "study_design",
                "page_number": 3,
                "nodes": [
                    {"node_id": 1, "label": "Tumor Assessment at Cycle 3"},
                    {"node_id": 2, "label": "Screening"},
                ],
                "edges": [],
            }
        ]
        pairs = extract_from_diagrams(diagrams)
        assert len(pairs) >= 1
        assert any(
            "tumor" in p.assessment_name.lower()
            and "cycle 3" in p.visit_label.lower()
            for p in pairs
        )

    def test_assessment_node_connected_to_visit_node(self):
        """Assessment node connected to visit node via edge."""
        diagrams = [
            {
                "diagram_type": "study_design",
                "page_number": 5,
                "nodes": [
                    {"node_id": 1, "label": "ECG"},
                    {"node_id": 2, "label": "Screening"},
                    {"node_id": 3, "label": "Day 1"},
                ],
                "edges": [
                    {"source_id": 1, "target_id": 2, "label": "", "edge_type": "arrow"},
                    {"source_id": 1, "target_id": 3, "label": "", "edge_type": "arrow"},
                ],
            }
        ]
        pairs = extract_from_diagrams(diagrams)
        visits = {p.visit_label.lower() for p in pairs}
        assert "screening" in visits
        assert "day 1" in visits

    def test_source_tagged_as_diagram(self):
        diagrams = [
            {
                "nodes": [{"node_id": 1, "label": "Labs at Day 1"}],
                "edges": [],
            }
        ]
        pairs = extract_from_diagrams(diagrams)
        assert all(p.source == "diagram" for p in pairs)

    def test_empty_diagrams(self):
        assert extract_from_diagrams([]) == []


# ---------------------------------------------------------------------------
# Scenario: Table and narrative results merged
# ---------------------------------------------------------------------------


class TestTableNarrativeMerge:
    """Merge table and narrative streams."""

    def test_narrative_adds_missing_assessment(self):
        """ECG from narrative added when not in table."""
        table_pairs = [
            AssessmentVisitPair("Physical Exam", "Screening", "table", 0.80),
            AssessmentVisitPair("Physical Exam", "Day 1", "table", 0.80),
        ]
        narrative_pairs = [
            AssessmentVisitPair("ECG", "Screening", "narrative", 0.70),
            AssessmentVisitPair("ECG", "End of Treatment", "narrative", 0.70),
        ]
        result = merge_streams(table_pairs, narrative_pairs)

        names = {m.assessment_name for m in result.assessments}
        assert "Physical Exam" in names
        assert "ECG" in names
        assert result.total_merged == 4

    def test_table_assessments_preserved(self):
        table_pairs = [
            AssessmentVisitPair("Vital Signs", "Day 1", "table", 0.80),
        ]
        narrative_pairs = [
            AssessmentVisitPair("ECG", "Day 1", "narrative", 0.70),
        ]
        result = merge_streams(table_pairs, narrative_pairs)

        table_sourced = [
            m for m in result.assessments if "table" in m.sources
        ]
        assert len(table_sourced) >= 1

    def test_source_tags_correct(self):
        table_pairs = [
            AssessmentVisitPair("Labs", "Screening", "table", 0.80),
        ]
        narrative_pairs = [
            AssessmentVisitPair("ECG", "Screening", "narrative", 0.70),
        ]
        result = merge_streams(table_pairs, narrative_pairs)

        ecg = [m for m in result.assessments if "ECG" in m.assessment_name]
        assert len(ecg) == 1
        assert ecg[0].source_tag == "narrative"


# ---------------------------------------------------------------------------
# Scenario: Multi-stream agreement boosts confidence
# ---------------------------------------------------------------------------


class TestMultiStreamConfidenceBoost:
    """Pairs found in 2+ streams get boosted confidence."""

    def test_table_and_narrative_agree(self):
        """CBC at Day 1 found in both → boosted."""
        table_pairs = [
            AssessmentVisitPair("CBC", "Day 1", "table", 0.80),
        ]
        narrative_pairs = [
            AssessmentVisitPair("CBC", "Day 1", "narrative", 0.70),
        ]
        result = merge_streams(table_pairs, narrative_pairs)

        cbc = [m for m in result.assessments if "CBC" in m.assessment_name]
        assert len(cbc) == 1
        assert cbc[0].confidence > 0.80  # Boosted
        assert cbc[0].source_tag == "narrative+table"

    def test_multi_stream_count(self):
        table_pairs = [
            AssessmentVisitPair("CBC", "Day 1", "table", 0.80),
        ]
        narrative_pairs = [
            AssessmentVisitPair("CBC", "Day 1", "narrative", 0.70),
        ]
        result = merge_streams(table_pairs, narrative_pairs)
        assert result.multi_stream_count == 1


# ---------------------------------------------------------------------------
# Scenario: Deduplicated across streams
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Fuzzy dedup across streams."""

    def test_physical_exam_deduped(self):
        """'Physical Exam' and 'Physical Examination' merge."""
        table_pairs = [
            AssessmentVisitPair(
                "Physical Exam", "Screening", "table", 0.80,
            ),
        ]
        narrative_pairs = [
            AssessmentVisitPair(
                "Physical Examination", "Screening", "narrative", 0.70,
            ),
        ]
        result = merge_streams(table_pairs, narrative_pairs)

        screening_pe = [
            m for m in result.assessments
            if "physical" in m.assessment_name.lower()
            and "screening" in m.visit_label.lower()
        ]
        assert len(screening_pe) == 1
        assert len(screening_pe[0].sources) == 2

    def test_screening_visit_variants_deduped(self):
        """'Screening' and 'Screening visit' merge."""
        a = [AssessmentVisitPair("ECG", "Screening", "table", 0.80)]
        b = [AssessmentVisitPair("ECG", "Screening visit", "narrative", 0.70)]
        result = merge_streams(a, b)

        ecg = [m for m in result.assessments if "ECG" in m.assessment_name]
        assert len(ecg) == 1


# ---------------------------------------------------------------------------
# pairs_to_raw_table tests
# ---------------------------------------------------------------------------


class TestPairsToRawTable:
    """Convert merged pairs into a RawSoaTable."""

    def test_basic_conversion(self):
        pairs = [
            MergedAssessment("ECG", "Screening", 0.70, {"narrative"}),
            MergedAssessment("ECG", "Day 1", 0.70, {"narrative"}),
            MergedAssessment("Labs", "Screening", 0.80, {"table"}),
        ]
        table = pairs_to_raw_table(pairs)

        assert table is not None
        assert table.construction_method == "multi_source"
        assert len(table.visit_headers) == 2  # Screening, Day 1
        assert len(table.activities) == 2  # ECG, Labs

    def test_empty_pairs(self):
        assert pairs_to_raw_table([]) is None

    def test_scheduling_matrix(self):
        """Each pair should mark the correct cell as True."""
        pairs = [
            MergedAssessment("ECG", "Screening", 0.70, {"narrative"}),
            MergedAssessment("Labs", "Day 1", 0.80, {"table"}),
        ]
        table = pairs_to_raw_table(pairs)

        # ECG at Screening=True, Day 1=False
        ecg = [a for a in table.activities if a[0] == "ECG"][0]
        screening_idx = table.visit_headers.index("Screening")
        assert ecg[1][screening_idx] is True

        # Labs at Day 1=True
        labs = [a for a in table.activities if a[0] == "Labs"][0]
        day1_idx = table.visit_headers.index("Day 1")
        assert labs[1][day1_idx] is True


# ---------------------------------------------------------------------------
# Integration: SoaExtractor with narrative input
# ---------------------------------------------------------------------------


class TestExtractorNarrativeIntegration:
    """SoaExtractor accepts narrative_texts and produces results."""

    def test_narrative_texts_accepted(
        self,
        sample_ich_sections,
        tmp_gateway,
        tmp_review_queue,
    ):
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
        )
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
            narrative_texts=[
                "CBC collected at Screening, Day 1, and Day 15."
            ],
        )
        assert result is not None
        assert result.activity_count > 0

    def test_diagrams_accepted(
        self,
        sample_ich_sections,
        tmp_gateway,
        tmp_review_queue,
    ):
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
        )
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
            diagrams=[
                {
                    "diagram_type": "study_design",
                    "nodes": [
                        {"node_id": 1, "label": "ECG at Screening"},
                    ],
                    "edges": [],
                }
            ],
        )
        assert result is not None
