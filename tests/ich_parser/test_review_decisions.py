"""Tests for ReviewQueue resolve, pending_count, and decision filtering (PTCV-84)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ptcv.ich_parser.models import ReviewQueueEntry
from ptcv.ich_parser.review_queue import ReviewQueue


def _entry(
    section_code: str = "B.3",
    confidence: float = 0.45,
    registry_id: str = "EUCT-001",
) -> ReviewQueueEntry:
    return ReviewQueueEntry(
        run_id="run-1",
        registry_id=registry_id,
        section_code=section_code,
        confidence_score=confidence,
        content_json='{"text_excerpt": "Objectives..."}',
        queue_timestamp_utc="2026-01-01T00:00:00Z",
    )


@pytest.fixture()
def rq(tmp_path: Path) -> ReviewQueue:
    """Provide an initialised ReviewQueue."""
    queue = ReviewQueue(db_path=tmp_path / "rq.db")
    queue.initialise()
    return queue


class TestResolve:
    """Tests for ReviewQueue.resolve()."""

    def test_resolve_marks_entry_decided(self, rq: ReviewQueue) -> None:
        entry = rq.enqueue(_entry())
        rq.resolve(entry.id, "approved")
        assert rq.pending() == []

    def test_resolve_with_corrected_value(self, rq: ReviewQueue) -> None:
        entry = rq.enqueue(_entry())
        rq.resolve(
            entry.id, "edited",
            corrected_value='{"canonical_label": "fixed"}',
        )
        assert rq.pending_count() == 0

    def test_resolve_with_notes(self, rq: ReviewQueue) -> None:
        entry = rq.enqueue(_entry())
        rq.resolve(
            entry.id, "rejected",
            reviewer_notes="Incorrect classification",
        )
        assert rq.pending_count() == 0

    def test_resolve_idempotent(self, rq: ReviewQueue) -> None:
        entry = rq.enqueue(_entry())
        rq.resolve(entry.id, "approved")
        rq.resolve(entry.id, "rejected")  # second resolve replaces
        assert rq.pending_count() == 0

    def test_resolved_entry_excluded_from_pending(
        self, rq: ReviewQueue,
    ) -> None:
        e1 = rq.enqueue(_entry("B.3"))
        e2 = rq.enqueue(_entry("B.7"))
        rq.resolve(e1.id, "approved")
        pending = rq.pending()
        assert len(pending) == 1
        assert pending[0].section_code == "B.7"


class TestPendingCount:
    """Tests for ReviewQueue.pending_count()."""

    def test_zero_when_empty(self, rq: ReviewQueue) -> None:
        assert rq.pending_count() == 0

    def test_counts_enqueued(self, rq: ReviewQueue) -> None:
        rq.enqueue(_entry("B.3"))
        rq.enqueue(_entry("B.7"))
        assert rq.pending_count() == 2

    def test_excludes_resolved(self, rq: ReviewQueue) -> None:
        e1 = rq.enqueue(_entry("B.3"))
        rq.enqueue(_entry("B.7"))
        rq.resolve(e1.id, "approved")
        assert rq.pending_count() == 1

    def test_filter_by_registry(self, rq: ReviewQueue) -> None:
        rq.enqueue(_entry("B.3", registry_id="EUCT-001"))
        rq.enqueue(_entry("B.5", registry_id="NCT-999"))
        assert rq.pending_count(registry_id="EUCT-001") == 1
        assert rq.pending_count(registry_id="NCT-999") == 1
        assert rq.pending_count() == 2

    def test_all_resolved_returns_zero(self, rq: ReviewQueue) -> None:
        e1 = rq.enqueue(_entry("B.3"))
        e2 = rq.enqueue(_entry("B.7"))
        rq.resolve(e1.id, "approved")
        rq.resolve(e2.id, "rejected")
        assert rq.pending_count() == 0


class TestPendingFiltersDecisions:
    """Verify pending() correctly filters resolved entries."""

    def test_pending_excludes_all_resolved(
        self, rq: ReviewQueue,
    ) -> None:
        entries = [rq.enqueue(_entry(f"B.{i}")) for i in range(1, 4)]
        for e in entries:
            rq.resolve(e.id, "approved")
        assert rq.pending() == []

    def test_pending_with_registry_filter_excludes_resolved(
        self, rq: ReviewQueue,
    ) -> None:
        e1 = rq.enqueue(_entry("B.3", registry_id="EUCT-001"))
        e2 = rq.enqueue(_entry("B.5", registry_id="EUCT-001"))
        rq.resolve(e1.id, "approved")
        pending = rq.pending(registry_id="EUCT-001")
        assert len(pending) == 1
        assert pending[0].id == e2.id
