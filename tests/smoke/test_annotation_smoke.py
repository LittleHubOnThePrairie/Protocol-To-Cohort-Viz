"""Smoke tests for annotation package (PTCV-40, PTCV-41).

Verifies that the annotation package can be imported and that
the core data flow works end-to-end with in-memory storage.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path for non-installed package
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


class TestAnnotationSmoke:
    """Smoke tests for the annotation package."""

    def test_import_package(self) -> None:
        from ptcv.annotations import (
            AnnotationRecord,
            AnnotationService,
            AnnotationSession,
        )
        assert AnnotationRecord is not None
        assert AnnotationSession is not None
        assert AnnotationService is not None

    def test_import_ui_component(self) -> None:
        from ptcv.ui.components.annotation_review import (
            render_annotation_review,
        )
        assert render_annotation_review is not None

    def test_end_to_end_annotation_flow(self) -> None:
        """Full annotation lifecycle: create, save, load, verify."""
        from ptcv.annotations.models import (
            AnnotationRecord,
            AnnotationSession,
        )
        from ptcv.annotations.service import AnnotationService
        from tests.annotations.test_service import StubGateway

        gateway = StubGateway()
        svc = AnnotationService(gateway=gateway)

        # Create session
        session = AnnotationSession(
            registry_id="NCT00112827",
            run_id="run-smoke",
            total_sections=3,
        )

        # Add annotations (with reviewer_notes on one for PTCV-41)
        for i, code in enumerate(["B.1", "B.3", "B.5"]):
            record = AnnotationRecord(
                section_id=f"run-smoke:{code}",
                section_code=code,
                original_label=f"Section {code}",
                confidence=0.65,
                reviewer_label=f"Section {code}",
                reviewer_action="accept",
                timestamp="2026-03-03T00:00:00+00:00",
                text_span="Smoke test content.",
                reviewer_notes="Reviewed" if i == 0 else "",
            )
            session.add(record)

        assert session.is_complete is True

        # Save
        key = svc.save(session)
        assert "NCT00112827" in key
        assert key.endswith(".jsonl")

        # Load and verify
        loaded = svc.load(
            "NCT00112827", "run-smoke", total_sections=3,
        )
        assert loaded is not None
        assert loaded.annotated_count == 3
        assert loaded.is_complete is True

        # Verify JSONL content
        jsonl = loaded.to_jsonl()
        lines = jsonl.strip().splitlines()
        assert len(lines) == 3

    def test_confidence_classification(self) -> None:
        """Verify threshold-based classification."""
        from ptcv.annotations.service import AnnotationService
        from tests.annotations.test_service import StubGateway

        svc = AnnotationService(
            gateway=StubGateway(),
            confidence_threshold=0.80,
        )
        assert svc.classify_confidence(0.79) == "low"
        assert svc.classify_confidence(0.80) == "high"
        assert svc.classify_confidence(0.95) == "high"
        assert svc.classify_confidence(0.10) == "low"
