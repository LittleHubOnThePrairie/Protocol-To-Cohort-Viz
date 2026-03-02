"""SQLite review_queue.db for low-confidence ICH section classifications.

Sections where confidence_score < 0.70 are enqueued here for human
review before downstream pipeline stages consume them. The schema is
intentionally simple (no UPDATE/DELETE triggers — entries are resolved
by downstream status updates via a separate resolved_at column).

Risk tier: MEDIUM — data pipeline quality gate (no patient data).

Regulatory references:
- ALCOA+ Accurate: low-confidence sections flagged for manual review
- ALCOA+ Contemporaneous: queue_timestamp_utc captured at enqueue time
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import ReviewQueueEntry


_DDL = """
CREATE TABLE IF NOT EXISTS review_queue (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT NOT NULL,
    registry_id          TEXT NOT NULL,
    section_code         TEXT NOT NULL,
    confidence_score     REAL NOT NULL,
    content_json         TEXT NOT NULL,
    queue_timestamp_utc  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_queue_run_id
    ON review_queue (run_id);

CREATE INDEX IF NOT EXISTS idx_review_queue_registry
    ON review_queue (registry_id);
"""


class ReviewQueue:
    """Append-only SQLite queue for human review of low-confidence sections.

    Args:
        db_path: Path to the SQLite database file. Parent directories are
            created automatically on initialise().
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Create the review_queue table and indexes.

        Idempotent — safe to call on an existing database.
        [PTCV-20 Scenario: Low-confidence sections routed to review queue]
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_DDL)
        finally:
            conn.close()

    def enqueue(self, entry: ReviewQueueEntry) -> ReviewQueueEntry:
        """Insert one ReviewQueueEntry and return it with id populated.

        Args:
            entry: Entry to persist. The id field is ignored on insert.

        Returns:
            A copy of the entry with the assigned AUTOINCREMENT id.
        [PTCV-20 Scenario: Low-confidence sections routed to review queue]
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO review_queue (
                    run_id, registry_id, section_code,
                    confidence_score, content_json, queue_timestamp_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.run_id,
                    entry.registry_id,
                    entry.section_code,
                    entry.confidence_score,
                    entry.content_json,
                    entry.queue_timestamp_utc,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
        finally:
            conn.close()

        import dataclasses
        return dataclasses.replace(entry, id=row_id)

    def pending(self, registry_id: str | None = None) -> list[ReviewQueueEntry]:
        """Return all pending review entries, optionally filtered by trial.

        Args:
            registry_id: Optional filter. If None, returns all entries.

        Returns:
            List of ReviewQueueEntry ordered by insertion sequence.
        """
        conn = self._connect()
        try:
            if registry_id is not None:
                rows = conn.execute(
                    "SELECT id, run_id, registry_id, section_code, "
                    "confidence_score, content_json, queue_timestamp_utc "
                    "FROM review_queue WHERE registry_id = ? ORDER BY id",
                    (registry_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, run_id, registry_id, section_code, "
                    "confidence_score, content_json, queue_timestamp_utc "
                    "FROM review_queue ORDER BY id"
                ).fetchall()
        finally:
            conn.close()

        return [
            ReviewQueueEntry(
                id=row[0],
                run_id=row[1],
                registry_id=row[2],
                section_code=row[3],
                confidence_score=row[4],
                content_json=row[5],
                queue_timestamp_utc=row[6],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))
