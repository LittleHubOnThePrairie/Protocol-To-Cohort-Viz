"""Unit tests for annotation data models (PTCV-40).

Tests AnnotationRecord and AnnotationSession serialisation,
JSONL round-trip, and session state management.
"""

from __future__ import annotations

import json

from ptcv.annotations.models import AnnotationRecord, AnnotationSession


def _make_record(
    section_code: str = "B.3",
    confidence: float = 0.45,
    action: str = "accept",
    label: str = "B.3 Trial Objectives and Purpose",
    run_id: str = "run-1",
) -> AnnotationRecord:
    """Create a minimal AnnotationRecord for tests."""
    return AnnotationRecord(
        section_id=f"{run_id}:{section_code}",
        section_code=section_code,
        original_label=label,
        confidence=confidence,
        reviewer_label=label,
        reviewer_action=action,
        timestamp="2026-03-03T12:00:00+00:00",
        text_span="Sample section text for testing purposes.",
    )


class TestAnnotationRecord:
    """Tests for AnnotationRecord."""

    def test_to_dict_roundtrip(self) -> None:
        record = _make_record()
        d = record.to_dict()
        restored = AnnotationRecord.from_dict(d)
        assert restored.section_id == record.section_id
        assert restored.confidence == record.confidence
        assert restored.reviewer_action == record.reviewer_action

    def test_to_dict_all_fields_present(self) -> None:
        d = _make_record().to_dict()
        expected_keys = {
            "section_id", "section_code", "original_label",
            "confidence", "reviewer_label", "reviewer_action",
            "timestamp", "text_span", "reviewer_notes", "source",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_json_serialisable(self) -> None:
        d = _make_record().to_dict()
        # Should not raise
        json.dumps(d)

    def test_from_dict_ignores_extra_keys(self) -> None:
        d = _make_record().to_dict()
        d["extra_field"] = "should be ignored"
        record = AnnotationRecord.from_dict(d)
        assert record.section_code == "B.3"


class TestAnnotationSession:
    """Tests for AnnotationSession."""

    def test_empty_session_counts(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=5,
        )
        assert session.annotated_count == 0
        assert session.is_complete is False

    def test_add_record_updates_count(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=2,
        )
        session.add(_make_record("B.1"))
        assert session.annotated_count == 1
        assert session.is_complete is False

    def test_session_complete_when_all_annotated(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=2,
        )
        session.add(_make_record("B.1", run_id="run-1"))
        session.add(_make_record("B.2", run_id="run-1"))
        assert session.annotated_count == 2
        assert session.is_complete is True

    def test_add_same_section_overwrites(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=5,
        )
        session.add(_make_record("B.3", action="accept"))
        session.add(_make_record("B.3", action="reject"))
        assert session.annotated_count == 1
        assert session.annotations["run-1:B.3"].reviewer_action == "reject"

    def test_jsonl_roundtrip(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=3,
        )
        session.add(_make_record("B.1"))
        session.add(_make_record("B.3"))

        jsonl = session.to_jsonl()
        lines = jsonl.strip().splitlines()
        assert len(lines) == 2

        restored = AnnotationSession.from_jsonl(
            jsonl,
            registry_id="NCT001",
            run_id="run-1",
            total_sections=3,
        )
        assert restored.annotated_count == 2
        assert "run-1:B.1" in restored.annotations
        assert "run-1:B.3" in restored.annotations

    def test_jsonl_empty_session(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
        )
        assert session.to_jsonl() == ""

    def test_from_jsonl_handles_empty_lines(self) -> None:
        data = "\n\n"
        session = AnnotationSession.from_jsonl(
            data, registry_id="NCT001", run_id="run-1",
        )
        assert session.annotated_count == 0

    def test_created_at_auto_populated(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
        )
        assert session.created_at != ""
        assert "T" in session.created_at  # ISO 8601

    def test_updated_at_changes_on_add(self) -> None:
        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
        )
        original = session.updated_at
        session.add(_make_record("B.1"))
        # updated_at should be >= original
        assert session.updated_at >= original
