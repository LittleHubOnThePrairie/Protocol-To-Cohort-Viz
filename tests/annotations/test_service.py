"""Unit tests for AnnotationService (PTCV-40).

Tests persistence via a stub StorageGateway and confidence
classification logic.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from ptcv.annotations.models import AnnotationRecord, AnnotationSession
from ptcv.annotations.service import AnnotationService, _storage_key
from ptcv.storage.models import ArtifactRecord, LineageRecord


class StubGateway:
    """Minimal in-memory StorageGateway for testing."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def initialise(self) -> None:
        pass

    def put_artifact(
        self,
        key: str,
        data: bytes,
        content_type: str = "",
        run_id: str = "",
        source_hash: str = "",
        user: str = "",
        immutable: bool = False,
        stage: str = "download",
        registry_id: Optional[str] = None,
        amendment_number: Optional[str] = None,
        source: Optional[str] = None,
    ) -> ArtifactRecord:
        self._store[key] = data
        return ArtifactRecord(
            key=key,
            version_id="",
            sha256="stub",
            content_type=content_type,
            timestamp_utc="2026-03-03T00:00:00Z",
            run_id=run_id,
            user=user,
        )

    def get_artifact(self, key: str, version_id: str = "") -> bytes:
        if key not in self._store:
            raise FileNotFoundError(key)
        return self._store[key]

    def list_versions(self, key: str) -> list[ArtifactRecord]:
        return []

    def get_lineage(self, run_id: str) -> list[LineageRecord]:
        return []


def _make_record(
    section_code: str = "B.3",
    confidence: float = 0.45,
) -> AnnotationRecord:
    return AnnotationRecord(
        section_id=f"run-1:{section_code}",
        section_code=section_code,
        original_label="Trial Objectives",
        confidence=confidence,
        reviewer_label="B.3 Trial Objectives and Purpose",
        reviewer_action="accept",
        timestamp="2026-03-03T12:00:00+00:00",
        text_span="Sample text.",
    )


class TestStorageKey:
    """Tests for _storage_key helper."""

    def test_format(self) -> None:
        key = _storage_key("NCT001", "run-1")
        assert key == "annotations/NCT001/run-1.jsonl"


class TestAnnotationService:
    """Tests for AnnotationService."""

    def test_save_and_load_roundtrip(self) -> None:
        gw = StubGateway()
        svc = AnnotationService(gateway=gw)

        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=3,
        )
        session.add(_make_record("B.1"))
        session.add(_make_record("B.3"))

        key = svc.save(session)
        assert key == "annotations/NCT001/run-1.jsonl"

        loaded = svc.load("NCT001", "run-1", total_sections=3)
        assert loaded is not None
        assert loaded.annotated_count == 2
        assert "run-1:B.1" in loaded.annotations
        assert "run-1:B.3" in loaded.annotations

    def test_load_returns_none_when_not_found(self) -> None:
        gw = StubGateway()
        svc = AnnotationService(gateway=gw)
        result = svc.load("NCT999", "nonexistent")
        assert result is None

    def test_save_overwrites_on_update(self) -> None:
        gw = StubGateway()
        svc = AnnotationService(gateway=gw)

        session = AnnotationSession(
            registry_id="NCT001",
            run_id="run-1",
            total_sections=3,
        )
        session.add(_make_record("B.1"))
        svc.save(session)

        # Add another annotation and save again
        session.add(_make_record("B.5"))
        svc.save(session)

        loaded = svc.load("NCT001", "run-1")
        assert loaded is not None
        assert loaded.annotated_count == 2

    def test_classify_confidence_high(self) -> None:
        svc = AnnotationService(gateway=StubGateway())
        assert svc.classify_confidence(0.85) == "high"
        assert svc.classify_confidence(0.80) == "high"

    def test_classify_confidence_low(self) -> None:
        svc = AnnotationService(gateway=StubGateway())
        assert svc.classify_confidence(0.79) == "low"
        assert svc.classify_confidence(0.0) == "low"

    def test_custom_threshold(self) -> None:
        svc = AnnotationService(
            gateway=StubGateway(),
            confidence_threshold=0.50,
        )
        assert svc.classify_confidence(0.50) == "high"
        assert svc.classify_confidence(0.49) == "low"
