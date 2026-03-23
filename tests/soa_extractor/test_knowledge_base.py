"""Tests for FAISS SoA knowledge base (PTCV-219).

Tests data models, serialization, and template matching logic.
FAISS/sentence-transformers tests are skipped if not installed.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ptcv.soa_extractor.knowledge_base import (
    SoaKnowledgeBase,
    SimilarProtocol,
    VerifiedSoaEntry,
)
from ptcv.soa_extractor.template_matcher import (
    CompletenessMatch,
    TemplateMatchReport,
    match_against_template,
    match_completeness,
)
from ptcv.soa_extractor.models import RawSoaTable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(
    registry_id: str = "NCT01512251",
    phase: str = "PHASE1/PHASE2",
    condition: str = "Melanoma",
    assessments: list[str] | None = None,
) -> VerifiedSoaEntry:
    if assessments is None:
        assessments = [
            "Physical Exam", "Vital Signs", "ECG", "Hematology",
            "Chemistry", "Urinalysis", "Coagulation", "Tumor Assessment",
        ]
    return VerifiedSoaEntry(
        registry_id=registry_id,
        phase=phase,
        condition=condition,
        intervention_type="DRUG",
        assessment_names=assessments,
        visit_headers=["Screening", "Day 1", "Week 2", "EOT"],
        verification_method="human_review",
        verified_utc="2026-03-20T12:00:00Z",
    )


def _make_table(
    activities: list[tuple[str, list[bool]]],
    visit_headers: list[str] | None = None,
) -> RawSoaTable:
    if visit_headers is None:
        n = max((len(f) for _, f in activities), default=3)
        visit_headers = [f"V{i}" for i in range(1, n + 1)]
    return RawSoaTable(
        visit_headers=visit_headers,
        day_headers=[],
        activities=activities,
        section_code="B.4",
    )


# ---------------------------------------------------------------------------
# VerifiedSoaEntry tests
# ---------------------------------------------------------------------------


class TestVerifiedSoaEntry:
    """Tests for VerifiedSoaEntry dataclass."""

    def test_creation(self):
        entry = _make_entry()
        assert entry.registry_id == "NCT01512251"
        assert entry.activity_count == 8
        assert entry.visit_count == 4

    def test_auto_counts(self):
        entry = VerifiedSoaEntry(
            registry_id="NCT00001",
            assessment_names=["A", "B", "C"],
            visit_headers=["V1", "V2"],
        )
        assert entry.activity_count == 3
        assert entry.visit_count == 2

    def test_embed_text(self):
        entry = _make_entry()
        text = entry.embed_text
        assert "PHASE1/PHASE2" in text
        assert "Melanoma" in text
        assert "Physical Exam" in text

    def test_to_dict_roundtrip(self):
        entry = _make_entry()
        d = entry.to_dict()
        restored = VerifiedSoaEntry.from_dict(d)
        assert restored.registry_id == entry.registry_id
        assert restored.assessment_names == entry.assessment_names
        assert restored.phase == entry.phase

    def test_from_dict_ignores_extra_keys(self):
        d = {"registry_id": "NCT00001", "extra_key": "ignored"}
        entry = VerifiedSoaEntry.from_dict(d)
        assert entry.registry_id == "NCT00001"


class TestSimilarProtocol:
    """Tests for SimilarProtocol dataclass."""

    def test_creation(self):
        entry = _make_entry()
        sp = SimilarProtocol(entry=entry, similarity=0.85, rank=1)
        assert sp.similarity == 0.85
        assert sp.rank == 1


# ---------------------------------------------------------------------------
# SoaKnowledgeBase tests (no FAISS dependency)
# ---------------------------------------------------------------------------


class TestKnowledgeBaseNoFaiss:
    """Tests that work without FAISS installed."""

    def test_initial_size(self):
        kb = SoaKnowledgeBase(index_dir="/tmp/test_kb")
        assert kb.size == 0

    def test_contains_empty(self):
        kb = SoaKnowledgeBase(index_dir="/tmp/test_kb")
        assert not kb.contains("NCT00001")

    def test_query_empty_returns_empty(self):
        kb = SoaKnowledgeBase(index_dir="/tmp/test_kb")
        assert kb.query("test") == []

    def test_get_entry_by_id_empty(self):
        kb = SoaKnowledgeBase(index_dir="/tmp/test_kb")
        assert kb.get_entry_by_id("NCT00001") is None

    def test_load_nonexistent_returns_false(self):
        kb = SoaKnowledgeBase(index_dir="/tmp/nonexistent_kb_path")
        assert kb.load() is False


# ---------------------------------------------------------------------------
# SoaKnowledgeBase integration (requires FAISS + sentence-transformers)
# ---------------------------------------------------------------------------


_faiss_available = False
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _faiss_available = True
except ImportError:
    pass


@pytest.mark.skipif(
    not _faiss_available,
    reason="faiss-cpu or sentence-transformers not installed",
)
class TestKnowledgeBaseWithFaiss:
    """Integration tests requiring FAISS + sentence-transformers."""

    def test_add_and_query(self, tmp_path: Path):
        kb = SoaKnowledgeBase(index_dir=tmp_path / "kb")
        entry = _make_entry()
        kb.add(entry)

        assert kb.size == 1
        assert kb.contains("NCT01512251")

        results = kb.query("Phase 1/2 melanoma oncology", top_k=1)
        assert len(results) == 1
        assert results[0].entry.registry_id == "NCT01512251"
        assert results[0].similarity > 0.0

    def test_add_batch(self, tmp_path: Path):
        kb = SoaKnowledgeBase(index_dir=tmp_path / "kb")
        entries = [
            _make_entry("NCT001"),
            _make_entry("NCT002", condition="Diabetes"),
            _make_entry("NCT003", condition="Heart Failure"),
        ]
        count = kb.add_batch(entries)
        assert count == 3
        assert kb.size == 3

    def test_save_and_load(self, tmp_path: Path):
        kb = SoaKnowledgeBase(index_dir=tmp_path / "kb")
        kb.add(_make_entry("NCT001"))
        kb.add(_make_entry("NCT002", condition="Diabetes"))
        kb.save()

        kb2 = SoaKnowledgeBase(index_dir=tmp_path / "kb")
        assert kb2.load()
        assert kb2.size == 2
        assert kb2.contains("NCT001")
        assert kb2.contains("NCT002")

    def test_query_returns_ranked(self, tmp_path: Path):
        kb = SoaKnowledgeBase(index_dir=tmp_path / "kb")
        kb.add(_make_entry("NCT001", condition="Melanoma"))
        kb.add(_make_entry("NCT002", condition="Diabetes"))
        kb.add(_make_entry("NCT003", condition="Lung Cancer"))

        results = kb.query(
            "Phase 1/2 melanoma BRAF inhibitor oncology",
            top_k=3,
        )
        assert len(results) == 3
        # Ranks should be 1, 2, 3
        assert results[0].rank == 1
        assert results[1].rank == 2
        # Melanoma should be most similar to melanoma query
        assert results[0].entry.condition == "Melanoma"


# ---------------------------------------------------------------------------
# Template matcher tests (no FAISS dependency)
# ---------------------------------------------------------------------------


class TestMatchAgainstTemplate:
    """Tests for match_against_template (no FAISS needed)."""

    def test_full_coverage(self):
        template = _make_entry(assessments=[
            "Physical Exam", "ECG", "Vital Signs",
        ])
        table = _make_table([
            ("Physical Exam", [True, True]),
            ("ECG", [True, False]),
            ("Vital Signs Assessment", [True, True]),  # substring match
        ])
        result = match_against_template(table, template)

        assert result.coverage_ratio == pytest.approx(1.0)
        assert result.missing_assessments == []
        assert len(result.found_assessments) == 3

    def test_partial_coverage(self):
        template = _make_entry(assessments=[
            "Physical Exam", "ECG", "Hematology", "Chemistry",
        ])
        table = _make_table([
            ("Physical Exam", [True]),
            ("ECG", [True]),
        ])
        result = match_against_template(table, template)

        assert result.coverage_ratio == 0.5
        assert "Hematology" in result.missing_assessments
        assert "Chemistry" in result.missing_assessments

    def test_zero_coverage(self):
        template = _make_entry(assessments=["Alpha", "Beta", "Gamma"])
        table = _make_table([("Delta", [True])])
        result = match_against_template(table, template)

        assert result.coverage_ratio == 0.0
        assert len(result.missing_assessments) == 3


class TestMatchCompleteness:
    """Tests for match_completeness (uses mocked KB)."""

    def test_empty_kb(self):
        kb = SoaKnowledgeBase(index_dir="/tmp/empty")
        table = _make_table([("Exam", [True])])
        report = match_completeness(table, kb)

        assert not report.has_template
        assert report.templates_checked == 0

    @pytest.mark.skipif(
        not _faiss_available,
        reason="faiss-cpu or sentence-transformers not installed",
    )
    def test_completeness_with_kb(self, tmp_path: Path):
        kb = SoaKnowledgeBase(index_dir=tmp_path / "kb")
        kb.add(_make_entry("NCT001", assessments=[
            "Physical Exam", "ECG", "Hematology", "Chemistry",
        ]))
        kb.add(_make_entry("NCT002", assessments=[
            "Physical Exam", "ECG", "Hematology", "Urinalysis",
        ]))

        table = _make_table([
            ("Physical Exam", [True]),
            ("ECG", [True]),
        ])

        report = match_completeness(table, kb, top_k=2)

        assert report.has_template
        assert report.templates_checked == 2
        assert report.best_match is not None
        assert report.best_coverage > 0

    @pytest.mark.skipif(
        not _faiss_available,
        reason="faiss-cpu or sentence-transformers not installed",
    )
    def test_consensus_missing(self, tmp_path: Path):
        """Test consensus missing: assessments missing from most templates."""
        kb = SoaKnowledgeBase(index_dir=tmp_path / "kb")
        # Both templates have Hematology
        kb.add(_make_entry("NCT001", assessments=[
            "Physical Exam", "ECG", "Hematology",
        ]))
        kb.add(_make_entry("NCT002", assessments=[
            "Physical Exam", "ECG", "Hematology",
        ]))

        # Table is missing Hematology
        table = _make_table([
            ("Physical Exam", [True]),
            ("ECG", [True]),
        ])

        report = match_completeness(
            table, kb, top_k=2, consensus_threshold=0.5,
        )

        # "hematology" should appear in consensus_missing
        assert any(
            "hematology" in m.lower()
            for m in report.consensus_missing
        )
