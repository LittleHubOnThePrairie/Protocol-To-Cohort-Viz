"""Tests for ReviewQueue (PTCV-20)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ptcv.ich_parser.models import ReviewQueueEntry
from ptcv.ich_parser.review_queue import ReviewQueue


def _entry(
    run_id: str = "run-1",
    section_code: str = "B.3",
    confidence: float = 0.45,
) -> ReviewQueueEntry:
    return ReviewQueueEntry(
        run_id=run_id,
        registry_id="EUCT-001",
        section_code=section_code,
        confidence_score=confidence,
        content_json='{"text_excerpt": "Objectives..."}',
        queue_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


class TestReviewQueue:
    def test_initialise_creates_db(self, tmp_path: Path) -> None:
        rq = ReviewQueue(db_path=tmp_path / "sub" / "rq.db")
        rq.initialise()
        assert (tmp_path / "sub" / "rq.db").exists()

    def test_initialise_is_idempotent(self, tmp_path: Path) -> None:
        rq = ReviewQueue(db_path=tmp_path / "rq.db")
        rq.initialise()
        rq.initialise()  # second call must not raise

    def test_enqueue_returns_entry_with_id(self, tmp_path: Path) -> None:
        rq = ReviewQueue(db_path=tmp_path / "rq.db")
        rq.initialise()
        entry = _entry()
        result = rq.enqueue(entry)
        assert result.id is not None
        assert result.id >= 1

    def test_pending_returns_enqueued_entries(self, tmp_path: Path) -> None:
        rq = ReviewQueue(db_path=tmp_path / "rq.db")
        rq.initialise()
        rq.enqueue(_entry("run-1", "B.3", 0.45))
        rq.enqueue(_entry("run-1", "B.7", 0.60))
        all_pending = rq.pending()
        assert len(all_pending) == 2

    def test_pending_filter_by_registry_id(self, tmp_path: Path) -> None:
        rq = ReviewQueue(db_path=tmp_path / "rq.db")
        rq.initialise()

        e1 = ReviewQueueEntry(
            run_id="r1", registry_id="EUCT-001", section_code="B.3",
            confidence_score=0.4, content_json="{}", queue_timestamp_utc="2026-01-01T00:00:00Z"
        )
        e2 = ReviewQueueEntry(
            run_id="r2", registry_id="NCT-999", section_code="B.5",
            confidence_score=0.5, content_json="{}", queue_timestamp_utc="2026-01-01T00:00:00Z"
        )
        rq.enqueue(e1)
        rq.enqueue(e2)

        euct_pending = rq.pending(registry_id="EUCT-001")
        assert len(euct_pending) == 1
        assert euct_pending[0].registry_id == "EUCT-001"

    def test_enqueue_preserves_all_fields(self, tmp_path: Path) -> None:
        rq = ReviewQueue(db_path=tmp_path / "rq.db")
        rq.initialise()
        entry = ReviewQueueEntry(
            run_id="run-xyz",
            registry_id="NCT-12345678",
            section_code="B.9",
            confidence_score=0.35,
            content_json='{"key": "val"}',
            queue_timestamp_utc="2026-03-01T12:00:00+00:00",
        )
        rq.enqueue(entry)
        results = rq.pending()
        assert len(results) == 1
        r = results[0]
        assert r.run_id == "run-xyz"
        assert r.registry_id == "NCT-12345678"
        assert r.section_code == "B.9"
        assert abs(r.confidence_score - 0.35) < 1e-6
        assert r.content_json == '{"key": "val"}'
        assert r.queue_timestamp_utc == "2026-03-01T12:00:00+00:00"

    def test_multiple_runs_same_registry(self, tmp_path: Path) -> None:
        rq = ReviewQueue(db_path=tmp_path / "rq.db")
        rq.initialise()
        for i in range(3):
            rq.enqueue(_entry(run_id=f"run-{i}"))
        results = rq.pending(registry_id="EUCT-001")
        assert len(results) == 3
        # Ordered by insertion sequence
        run_ids = [r.run_id for r in results]
        assert run_ids == ["run-0", "run-1", "run-2"]
