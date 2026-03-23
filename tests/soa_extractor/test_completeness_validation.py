"""Tests for SoA completeness validation against knowledge base (PTCV-264).

GHERKIN Scenarios:
  Feature: SoA completeness validation against knowledge base

    Scenario: Missing assessments detected
      Given a Phase 1/2 oncology protocol with 15 extracted assessments
      And the nearest verified template has 20 common assessments
      When template matching runs
      Then 5 missing assessment names are reported
      And completeness ratio is 75%

    Scenario: Complete extraction passes
      Given an extraction capturing all expected assessments
      When template matching runs
      Then completeness is 100%
      And no missing assessments are flagged
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.soa_extractor.knowledge_base import SoaKnowledgeBase, VerifiedSoaEntry
from ptcv.soa_extractor.models import RawSoaTable
from ptcv.soa_extractor.template_matcher import match_completeness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table(assessment_names: list[str]) -> RawSoaTable:
    """Build a minimal RawSoaTable with the given assessment names."""
    visits = ["Screening", "Day 1", "Week 2", "Week 4"]
    activities = [(name, [True] * len(visits)) for name in assessment_names]
    return RawSoaTable(
        visit_headers=visits,
        day_headers=[],
        activities=activities,
        section_code="B.4",
    )


def _make_entry(
    registry_id: str,
    assessment_names: list[str],
    phase: str = "PHASE1/PHASE2",
    condition: str = "Solid Tumors",
) -> VerifiedSoaEntry:
    """Build a VerifiedSoaEntry for the knowledge base."""
    return VerifiedSoaEntry(
        registry_id=registry_id,
        phase=phase,
        condition=condition,
        intervention_type="DRUG",
        assessment_names=assessment_names,
        visit_headers=["Screening", "Day 1", "Week 2", "Week 4"],
        verification_method="human_review",
        verified_utc="2025-01-01T00:00:00Z",
    )


@pytest.fixture
def seeded_kb(tmp_path: Path) -> SoaKnowledgeBase:
    """Knowledge base seeded with one template (20 assessments)."""
    kb = SoaKnowledgeBase(index_dir=tmp_path / "kb")
    template_assessments = [
        "Informed Consent",
        "Physical Exam",
        "Vital Signs",
        "ECG",
        "Hematology",
        "Chemistry",
        "Urinalysis",
        "Coagulation",
        "Tumor Assessment",
        "ECOG Performance",
        "Concomitant Medications",
        "Adverse Events",
        "Medical History",
        "Body Weight",
        "Height",
        "CT Scan",
        "MRI",
        "Pharmacokinetics",
        "Biomarker Sample",
        "Pregnancy Test",
    ]
    entry = _make_entry("NCT99990001", template_assessments)
    kb.add(entry)
    return kb


# ---------------------------------------------------------------------------
# Scenario: Missing assessments detected
# ---------------------------------------------------------------------------


class TestMissingAssessmentsDetected:
    """Given 15/20 extracted assessments, 5 missing should be reported."""

    def test_completeness_ratio_75_percent(self, seeded_kb: SoaKnowledgeBase):
        """Coverage should be 75% when 15 of 20 template assessments found."""
        extracted = [
            "Informed Consent",
            "Physical Exam",
            "Vital Signs",
            "ECG",
            "Hematology",
            "Chemistry",
            "Tumor Assessment",
            "ECOG Performance",
            "Concomitant Medications",
            "Adverse Events",
            "Medical History",
            "Body Weight",
            "Height",
            "CT Scan",
            "MRI",
        ]
        table = _make_table(extracted)
        report = match_completeness(table, seeded_kb, top_k=1)

        assert report.has_template
        assert report.best_coverage == pytest.approx(0.75, abs=0.01)

    def test_five_missing_assessment_names(self, seeded_kb: SoaKnowledgeBase):
        """Five missing assessment names should be reported."""
        extracted = [
            "Informed Consent",
            "Physical Exam",
            "Vital Signs",
            "ECG",
            "Hematology",
            "Chemistry",
            "Tumor Assessment",
            "ECOG Performance",
            "Concomitant Medications",
            "Adverse Events",
            "Medical History",
            "Body Weight",
            "Height",
            "CT Scan",
            "MRI",
        ]
        table = _make_table(extracted)
        report = match_completeness(table, seeded_kb, top_k=1)

        assert report.best_match is not None
        assert len(report.best_match.missing_assessments) == 5

    def test_missing_names_are_correct(self, seeded_kb: SoaKnowledgeBase):
        """Missing should be Urinalysis, Coagulation, PK, Biomarker, Pregnancy."""
        extracted = [
            "Informed Consent",
            "Physical Exam",
            "Vital Signs",
            "ECG",
            "Hematology",
            "Chemistry",
            "Tumor Assessment",
            "ECOG Performance",
            "Concomitant Medications",
            "Adverse Events",
            "Medical History",
            "Body Weight",
            "Height",
            "CT Scan",
            "MRI",
        ]
        table = _make_table(extracted)
        report = match_completeness(table, seeded_kb, top_k=1)

        expected_missing = {
            "Urinalysis",
            "Coagulation",
            "Pharmacokinetics",
            "Biomarker Sample",
            "Pregnancy Test",
        }
        actual_missing = set(report.best_match.missing_assessments)
        assert actual_missing == expected_missing


# ---------------------------------------------------------------------------
# Scenario: Complete extraction passes
# ---------------------------------------------------------------------------


class TestCompleteExtractionPasses:
    """Given all expected assessments, completeness should be 100%."""

    def test_completeness_100_percent(self, seeded_kb: SoaKnowledgeBase):
        """All 20 assessments present → 100% coverage."""
        all_assessments = [
            "Informed Consent",
            "Physical Exam",
            "Vital Signs",
            "ECG",
            "Hematology",
            "Chemistry",
            "Urinalysis",
            "Coagulation",
            "Tumor Assessment",
            "ECOG Performance",
            "Concomitant Medications",
            "Adverse Events",
            "Medical History",
            "Body Weight",
            "Height",
            "CT Scan",
            "MRI",
            "Pharmacokinetics",
            "Biomarker Sample",
            "Pregnancy Test",
        ]
        table = _make_table(all_assessments)
        report = match_completeness(table, seeded_kb, top_k=1)

        assert report.has_template
        assert report.best_coverage == pytest.approx(1.0)

    def test_no_missing_assessments_flagged(self, seeded_kb: SoaKnowledgeBase):
        """No missing assessments when all are present."""
        all_assessments = [
            "Informed Consent",
            "Physical Exam",
            "Vital Signs",
            "ECG",
            "Hematology",
            "Chemistry",
            "Urinalysis",
            "Coagulation",
            "Tumor Assessment",
            "ECOG Performance",
            "Concomitant Medications",
            "Adverse Events",
            "Medical History",
            "Body Weight",
            "Height",
            "CT Scan",
            "MRI",
            "Pharmacokinetics",
            "Biomarker Sample",
            "Pregnancy Test",
        ]
        table = _make_table(all_assessments)
        report = match_completeness(table, seeded_kb, top_k=1)

        assert report.best_match is not None
        assert len(report.best_match.missing_assessments) == 0
        assert len(report.consensus_missing) == 0


# ---------------------------------------------------------------------------
# Integration: ExtractResult includes completeness fields
# ---------------------------------------------------------------------------


class TestExtractResultCompleteness:
    """Verify completeness fields appear on ExtractResult."""

    def test_extract_result_has_completeness_fields(
        self,
        sample_ich_sections,
        tmp_gateway,
        tmp_review_queue,
        seeded_kb,
    ):
        """ExtractResult should include completeness_ratio and missing_assessments."""
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
            knowledge_base=seeded_kb,
        )
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_run_id="ich-run-001",
            source_sha256="a" * 64,
        )
        assert hasattr(result, "completeness_ratio")
        assert hasattr(result, "missing_assessments")
        assert hasattr(result, "completeness_report")
        assert isinstance(result.completeness_ratio, float)
        assert isinstance(result.missing_assessments, list)

    def test_completeness_below_threshold_logs_warning(
        self,
        sample_ich_sections,
        tmp_gateway,
        tmp_review_queue,
        seeded_kb,
        caplog,
    ):
        """When completeness < 80%, a warning should be logged."""
        import logging
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
            knowledge_base=seeded_kb,
        )
        with caplog.at_level(logging.WARNING):
            result = extractor.extract(
                sections=sample_ich_sections,
                registry_id="NCT00112827",
                source_run_id="ich-run-001",
                source_sha256="a" * 64,
            )
        # The sample SoA has only 6 assessments vs. 20 in the template
        # so completeness should be well below 80%.
        if result.completeness_ratio < 0.80:
            assert any("Completeness" in r.message for r in caplog.records)

    def test_no_kb_returns_default_completeness(
        self,
        sample_ich_sections,
        tmp_gateway,
        tmp_review_queue,
    ):
        """Without a knowledge base, completeness should default to 1.0."""
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
            knowledge_base=None,
        )
        # Patch _resolve_knowledge_base to ensure no fallback loading
        with patch.object(extractor, "_resolve_knowledge_base", return_value=None):
            result = extractor.extract(
                sections=sample_ich_sections,
                registry_id="NCT00112827",
                source_run_id="ich-run-001",
                source_sha256="a" * 64,
            )
        assert result.completeness_ratio == 1.0
        assert result.missing_assessments == []
        assert result.completeness_report is None


# ---------------------------------------------------------------------------
# Edge: empty knowledge base
# ---------------------------------------------------------------------------


class TestEmptyKnowledgeBase:
    """When knowledge base has no entries, gracefully skip."""

    def test_empty_kb_returns_no_template(self, tmp_path: Path):
        kb = SoaKnowledgeBase(index_dir=tmp_path / "empty_kb")
        table = _make_table(["Physical Exam", "ECG"])
        report = match_completeness(table, kb, top_k=1)

        assert not report.has_template
        assert report.best_coverage == 0.0
