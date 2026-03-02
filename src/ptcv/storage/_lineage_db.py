"""Shared SQLite helpers for the PTCV lineage chain.

Both FilesystemAdapter and LocalStorageAdapter use these helpers to read
and write the append-only lineage_records table. The schema includes
trigger-level guards that prevent any UPDATE or DELETE operation,
enforcing the ALCOA+ Traceable / 21 CFR 11.10(e) requirements.

Risk tier: MEDIUM — data pipeline audit trail.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import LineageRecord


_DDL = """
CREATE TABLE IF NOT EXISTS lineage_records (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    stage            TEXT NOT NULL,
    artifact_key     TEXT NOT NULL,
    version_id       TEXT NOT NULL,
    sha256           TEXT NOT NULL,
    source_hash      TEXT NOT NULL,
    user             TEXT NOT NULL,
    timestamp_utc    TEXT NOT NULL,
    registry_id      TEXT,
    amendment_number TEXT,
    source           TEXT
);

CREATE TRIGGER IF NOT EXISTS prevent_update
BEFORE UPDATE ON lineage_records
BEGIN
    SELECT RAISE(ABORT, 'Updates not allowed on lineage_records');
END;

CREATE TRIGGER IF NOT EXISTS prevent_delete
BEFORE DELETE ON lineage_records
BEGIN
    SELECT RAISE(ABORT, 'Deletes not allowed on lineage_records');
END;
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create lineage_records table and immutability triggers.

    Safe to call on an existing database — all statements use IF NOT
    EXISTS. Does NOT commit; caller is responsible for the transaction.

    Args:
        conn: Open SQLite connection.
    """
    conn.executescript(_DDL)


def insert_record(conn: sqlite3.Connection, record: "LineageRecord") -> int:
    """Insert one LineageRecord and return the assigned row id.

    Args:
        conn: Open SQLite connection (autocommit or within a transaction).
        record: LineageRecord to persist. The ``id`` field is ignored.

    Returns:
        The AUTOINCREMENT row id assigned by SQLite.
    """
    cursor = conn.execute(
        """
        INSERT INTO lineage_records (
            run_id, stage, artifact_key, version_id, sha256,
            source_hash, user, timestamp_utc,
            registry_id, amendment_number, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.run_id,
            record.stage,
            record.artifact_key,
            record.version_id,
            record.sha256,
            record.source_hash,
            record.user,
            record.timestamp_utc,
            record.registry_id,
            record.amendment_number,
            record.source,
        ),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def query_by_run_id(
    conn: sqlite3.Connection, run_id: str
) -> list["LineageRecord"]:
    """Return all records for a given run_id, ordered by id.

    Args:
        conn: Open SQLite connection.
        run_id: UUID4 run identifier to filter on.

    Returns:
        List of LineageRecord instances, ordered by insertion sequence.
    """
    from .models import LineageRecord  # local import to avoid circular dep

    rows = conn.execute(
        "SELECT id, run_id, stage, artifact_key, version_id, sha256, "
        "source_hash, user, timestamp_utc, registry_id, amendment_number, "
        "source FROM lineage_records WHERE run_id = ? ORDER BY id",
        (run_id,),
    ).fetchall()

    return [
        LineageRecord(
            id=row[0],
            run_id=row[1],
            stage=row[2],
            artifact_key=row[3],
            version_id=row[4],
            sha256=row[5],
            source_hash=row[6],
            user=row[7],
            timestamp_utc=row[8],
            registry_id=row[9],
            amendment_number=row[10],
            source=row[11],
        )
        for row in rows
    ]
