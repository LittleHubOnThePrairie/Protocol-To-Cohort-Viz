"""CT Review Queue — SQLite storage for unmapped CDISC CT terms.

Appends rows to ct_review_queue.db. Triggers prevent UPDATE and DELETE
(ALCOA+ Original). Distinct from the ICH parser review queue.

Risk tier: MEDIUM — regulatory metadata; no patient data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .models import CtReviewQueueEntry


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ct_review_queue (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT    NOT NULL,
    registry_id          TEXT    NOT NULL,
    domain               TEXT    NOT NULL,
    variable             TEXT    NOT NULL,
    original_value       TEXT    NOT NULL,
    ct_lookup_attempted  INTEGER NOT NULL DEFAULT 1,
    queue_timestamp_utc  TEXT    NOT NULL
);
CREATE TRIGGER IF NOT EXISTS prevent_update_ct
    BEFORE UPDATE ON ct_review_queue
    BEGIN SELECT RAISE(ABORT, 'Updates not allowed on ct_review_queue'); END;
CREATE TRIGGER IF NOT EXISTS prevent_delete_ct
    BEFORE DELETE ON ct_review_queue
    BEGIN SELECT RAISE(ABORT, 'Deletes not allowed on ct_review_queue'); END;
"""


class CtReviewQueue:
    """Append-only SQLite queue for unmapped CDISC CT terms.

    Args:
        db_path: Path to the SQLite database file. Created if absent.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def initialise(self) -> None:
        """Create the database file and schema (idempotent).

        Safe to call on every service start-up.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def enqueue(self, entry: CtReviewQueueEntry) -> CtReviewQueueEntry:
        """Insert one unmapped CT term into the review queue.

        Args:
            entry: CtReviewQueueEntry to insert (id is ignored).

        Returns:
            CtReviewQueueEntry with id populated from the new row.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO ct_review_queue
                    (run_id, registry_id, domain, variable,
                     original_value, ct_lookup_attempted, queue_timestamp_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.run_id,
                    entry.registry_id,
                    entry.domain,
                    entry.variable,
                    entry.original_value,
                    int(entry.ct_lookup_attempted),
                    entry.queue_timestamp_utc,
                ),
            )
            conn.commit()
            return dataclasses.replace(entry, id=cursor.lastrowid or 0)

    def query_by_run(self, run_id: str) -> list[CtReviewQueueEntry]:
        """Return all entries for a given run_id.

        Args:
            run_id: UUID4 run identifier.

        Returns:
            List of CtReviewQueueEntry ordered by insertion time.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT id, run_id, registry_id, domain, variable, "
                "original_value, ct_lookup_attempted, queue_timestamp_utc "
                "FROM ct_review_queue WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        result: list[CtReviewQueueEntry] = []
        for row in rows:
            result.append(
                CtReviewQueueEntry(
                    id=row[0],
                    run_id=row[1],
                    registry_id=row[2],
                    domain=row[3],
                    variable=row[4],
                    original_value=row[5],
                    ct_lookup_attempted=bool(row[6]),
                    queue_timestamp_utc=row[7],
                )
            )
        return result


import dataclasses
