"""Unit tests for annotation review component (PTCV-40, PTCV-41).

Tests the confidence classification, rendering helpers, and
annotation data model without Streamlit runtime.
"""

from __future__ import annotations

import json

from ptcv.annotations.models import AnnotationRecord, AnnotationSession
from ptcv.annotations.service import AnnotationService
from ptcv.ich_parser.models import IchSection
from ptcv.ui.components.annotation_review import (
    _confidence_bg,
    _confidence_colour,
    _SECTION_LABELS,
)


def _make_section(
    code: str = "B.3",
    name: str = "Trial Objectives and Purpose",
    confidence: float = 0.45,
    text: str = "Test content",
) -> IchSection:
    """Create a minimal IchSection for tests."""
    return IchSection(
        run_id="run-1",
        source_run_id="src-1",
        source_sha256="abc123",
        registry_id="NCT001",
        section_code=code,
        section_name=name,
        content_json=json.dumps({"text_excerpt": text}),
        confidence_score=confidence,
        review_required=confidence < 0.70,
        legacy_format=False,
        extraction_timestamp_utc="2026-03-03T00:00:00Z",
    )


class TestConfidenceColour:
    """Tests for _confidence_colour and _confidence_bg."""

    def test_high_confidence_green(self) -> None:
        assert "#2e7d32" in _confidence_colour("high")

    def test_low_confidence_orange(self) -> None:
        assert "#e65100" in _confidence_colour("low")

    def test_high_bg_green_tint(self) -> None:
        assert "#e8f5e9" in _confidence_bg("high")

    def test_low_bg_amber_tint(self) -> None:
        assert "#fff3e0" in _confidence_bg("low")


class TestSectionLabels:
    """Tests for the section labels constant."""

    def test_11_labels_defined(self) -> None:
        assert len(_SECTION_LABELS) == 11

    def test_labels_start_with_section_code(self) -> None:
        for label in _SECTION_LABELS:
            assert label.startswith("B.")

    def test_b1_label(self) -> None:
        assert "B.1 General Information" in _SECTION_LABELS

    def test_b11_label(self) -> None:
        assert "B.11 Quality Control and Quality Assurance" in _SECTION_LABELS


class TestAnnotationWithSections:
    """Integration tests for annotation flow without Streamlit."""

    def test_low_confidence_section_classified(self) -> None:
        from tests.annotations.test_service import StubGateway

        svc = AnnotationService(gateway=StubGateway())
        sec = _make_section(confidence=0.45)
        assert svc.classify_confidence(sec.confidence_score) == "low"

    def test_high_confidence_section_classified(self) -> None:
        from tests.annotations.test_service import StubGateway

        svc = AnnotationService(gateway=StubGateway())
        sec = _make_section(confidence=0.90)
        assert svc.classify_confidence(sec.confidence_score) == "high"

    def test_session_tracks_section_ids(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=2,
        )
        record = AnnotationRecord(
            section_id="run-1:B.3",
            section_code="B.3",
            original_label="Trial Objectives and Purpose",
            confidence=0.45,
            reviewer_label="B.3 Trial Objectives and Purpose",
            reviewer_action="accept",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Sample text",
        )
        session.add(record)
        assert "run-1:B.3" in session.annotations
        assert session.annotated_count == 1


class TestAnnotationRecordNotes:
    """Tests for reviewer_notes field (PTCV-41)."""

    def test_reviewer_notes_default_empty(self) -> None:
        record = AnnotationRecord(
            section_id="run-1:B.1",
            section_code="B.1",
            original_label="General Information",
            confidence=0.90,
            reviewer_label="B.1 General Information",
            reviewer_action="accept",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Some text",
        )
        assert record.reviewer_notes == ""

    def test_reviewer_notes_persisted(self) -> None:
        record = AnnotationRecord(
            section_id="run-1:B.1",
            section_code="B.1",
            original_label="General Information",
            confidence=0.90,
            reviewer_label="B.1 General Information",
            reviewer_action="accept",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Some text",
            reviewer_notes="Looks correct, matches protocol header",
        )
        d = record.to_dict()
        assert d["reviewer_notes"] == "Looks correct, matches protocol header"

    def test_reviewer_notes_roundtrip(self) -> None:
        original = AnnotationRecord(
            section_id="run-1:B.5",
            section_code="B.5",
            original_label="Selection of Subjects",
            confidence=0.55,
            reviewer_label="B.5 Selection of Subjects",
            reviewer_action="reject",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Eligibility criteria text",
            reviewer_notes="Wrong section — this is exclusion criteria",
        )
        restored = AnnotationRecord.from_dict(original.to_dict())
        assert restored.reviewer_notes == original.reviewer_notes

    def test_reviewer_notes_in_jsonl(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=1,
        )
        record = AnnotationRecord(
            section_id="run-1:B.3",
            section_code="B.3",
            original_label="Trial Objectives",
            confidence=0.70,
            reviewer_label="B.3 Trial Objectives and Purpose",
            reviewer_action="override",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Objective text",
            reviewer_notes="Reclassified from B.2",
        )
        session.add(record)
        jsonl = session.to_jsonl()
        parsed = json.loads(jsonl.strip())
        assert parsed["reviewer_notes"] == "Reclassified from B.2"


class TestNoTruncation:
    """Tests verifying section text is not truncated (PTCV-41)."""

    def test_long_text_not_truncated_in_record(self) -> None:
        long_text = "A" * 2000
        record = AnnotationRecord(
            section_id="run-1:B.4",
            section_code="B.4",
            original_label="Trial Design",
            confidence=0.85,
            reviewer_label="B.4 Trial Design",
            reviewer_action="accept",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span=long_text,
        )
        assert len(record.text_span) == 2000
        assert record.text_span == long_text

    def test_long_text_survives_jsonl_roundtrip(self) -> None:
        long_text = "B" * 1500
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=1,
        )
        record = AnnotationRecord(
            section_id="run-1:B.6",
            section_code="B.6",
            original_label="Treatment of Subjects",
            confidence=0.75,
            reviewer_label="B.6 Treatment of Subjects",
            reviewer_action="accept",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span=long_text,
        )
        session.add(record)
        loaded = AnnotationSession.from_jsonl(
            session.to_jsonl(),
            registry_id="NCT001",
            run_id="run-1",
        )
        restored = loaded.annotations["run-1:B.6"]
        assert len(restored.text_span) == 1500


class TestAnnotationSource:
    """Tests for source field (PTCV-43)."""

    def test_source_defaults_to_classifier(self) -> None:
        record = AnnotationRecord(
            section_id="run-1:B.1",
            section_code="B.1",
            original_label="General Information",
            confidence=0.90,
            reviewer_label="B.1 General Information",
            reviewer_action="accept",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Some text",
        )
        assert record.source == "classifier"

    def test_manual_span_source(self) -> None:
        record = AnnotationRecord(
            section_id="run-1:manual:5",
            section_code="B.4",
            original_label="",
            confidence=0.0,
            reviewer_label="B.4 Trial Design",
            reviewer_action="manual_label",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Manually selected text",
            reviewer_notes="Found this in the raw protocol",
            source="manual_span",
        )
        assert record.source == "manual_span"
        assert record.reviewer_action == "manual_label"

    def test_source_roundtrip(self) -> None:
        original = AnnotationRecord(
            section_id="run-1:manual:0",
            section_code="B.7",
            original_label="",
            confidence=0.0,
            reviewer_label="B.7 Assessment of Efficacy",
            reviewer_action="manual_label",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Efficacy text",
            source="manual_span",
        )
        restored = AnnotationRecord.from_dict(original.to_dict())
        assert restored.source == "manual_span"

    def test_source_in_jsonl(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=1,
        )
        record = AnnotationRecord(
            section_id="run-1:manual:0",
            section_code="B.2",
            original_label="",
            confidence=0.0,
            reviewer_label="B.2 Background Information",
            reviewer_action="manual_label",
            timestamp="2026-03-03T12:00:00+00:00",
            text_span="Background text",
            source="manual_span",
        )
        session.add(record)
        jsonl = session.to_jsonl()
        parsed = json.loads(jsonl.strip())
        assert parsed["source"] == "manual_span"
