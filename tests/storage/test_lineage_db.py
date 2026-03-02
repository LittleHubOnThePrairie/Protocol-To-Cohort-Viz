"""Unit tests for ptcv.storage._lineage_db SQLite helpers.

IQ: create_schema, insert_record, query_by_run_id
Risk tier: MEDIUM — data pipeline audit trail.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.storage._lineage_db import create_schema, insert_record, query_by_run_id
from ptcv.storage.models import LineageRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    return conn


def _make_record(**kwargs) -> LineageRecord:
    defaults = dict(
        id=0,
        run_id="run-001",
        stage="download",
        artifact_key="eu-ctr/EUCT-001_01.pdf",
        version_id="",
        sha256="a" * 64,
        source_hash="",
        user="test",
        timestamp_utc="2026-03-02T14:00:00Z",
        registry_id=None,
        amendment_number=None,
        source=None,
    )
    defaults.update(kwargs)
    return LineageRecord(**defaults)


# ---------------------------------------------------------------------------
# TestCreateSchema
# ---------------------------------------------------------------------------

class TestCreateSchema:
    """IQ: create_schema creates table and triggers."""

    def test_table_exists_after_create_schema(self):
        """lineage_records table is created."""
        conn = sqlite3.connect(":memory:")
        create_schema(conn)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lineage_records'"
        ).fetchone()
        assert result is not None

    def test_prevent_update_trigger_exists(self):
        """prevent_update trigger is created."""
        conn = sqlite3.connect(":memory:")
        create_schema(conn)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='prevent_update'"
        ).fetchone()
        assert result is not None

    def test_prevent_delete_trigger_exists(self):
        """prevent_delete trigger is created."""
        conn = sqlite3.connect(":memory:")
        create_schema(conn)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='prevent_delete'"
        ).fetchone()
        assert result is not None

    def test_create_schema_is_idempotent(self):
        """Calling create_schema twice does not raise."""
        conn = sqlite3.connect(":memory:")
        create_schema(conn)
        create_schema(conn)  # should not raise


# ---------------------------------------------------------------------------
# TestInsertRecord
# ---------------------------------------------------------------------------

class TestInsertRecord:
    """IQ: insert_record persists data and returns row id."""

    def test_insert_returns_positive_row_id(self):
        """insert_record returns an integer row id >= 1."""
        conn = _open_memory_db()
        row_id = insert_record(conn, _make_record())
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_inserted_row_readable(self):
        """Inserted record can be read back from SQLite."""
        conn = _open_memory_db()
        rec = _make_record(run_id="run-xyz", stage="download", sha256="b" * 64)
        insert_record(conn, rec)
        row = conn.execute(
            "SELECT run_id, stage, sha256 FROM lineage_records WHERE run_id=?",
            ("run-xyz",),
        ).fetchone()
        assert row is not None
        assert row[0] == "run-xyz"
        assert row[2] == "b" * 64

    def test_update_trigger_blocks_update(self):
        """UPDATE on lineage_records is blocked by trigger."""
        conn = _open_memory_db()
        insert_record(conn, _make_record())
        with pytest.raises(sqlite3.DatabaseError, match="Updates not allowed"):
            conn.execute("UPDATE lineage_records SET stage='x' WHERE id=1")

    def test_delete_trigger_blocks_delete(self):
        """DELETE on lineage_records is blocked by trigger."""
        conn = _open_memory_db()
        insert_record(conn, _make_record())
        with pytest.raises(sqlite3.DatabaseError, match="Deletes not allowed"):
            conn.execute("DELETE FROM lineage_records WHERE id=1")


# ---------------------------------------------------------------------------
# TestQueryByRunId
# ---------------------------------------------------------------------------

class TestQueryByRunId:
    """IQ: query_by_run_id returns correct records."""

    def test_returns_empty_for_unknown_run(self):
        """query_by_run_id returns [] for a run_id that was never inserted."""
        conn = _open_memory_db()
        result = query_by_run_id(conn, "unknown-run")
        assert result == []

    def test_returns_all_records_for_run(self):
        """All records for a run_id are returned in insertion order."""
        conn = _open_memory_db()
        insert_record(conn, _make_record(run_id="run-A", artifact_key="key/a.pdf"))
        insert_record(conn, _make_record(run_id="run-A", artifact_key="key/b.pdf"))
        insert_record(conn, _make_record(run_id="run-B", artifact_key="key/c.pdf"))

        records = query_by_run_id(conn, "run-A")
        assert len(records) == 2
        assert records[0].artifact_key == "key/a.pdf"
        assert records[1].artifact_key == "key/b.pdf"

    def test_does_not_return_other_run_records(self):
        """Records for other run_ids are excluded."""
        conn = _open_memory_db()
        insert_record(conn, _make_record(run_id="run-X"))
        insert_record(conn, _make_record(run_id="run-Y"))

        assert len(query_by_run_id(conn, "run-X")) == 1
        assert len(query_by_run_id(conn, "run-Y")) == 1

    def test_returned_record_fields_match_inserted(self):
        """LineageRecord fields match what was inserted."""
        conn = _open_memory_db()
        rec = _make_record(
            run_id="run-check",
            stage="download",
            artifact_key="eu-ctr/TEST.pdf",
            sha256="c" * 64,
            user="ptcv-system",
            registry_id="2024-000001-10-00",
            source="EU-CTR",
        )
        insert_record(conn, rec)
        result = query_by_run_id(conn, "run-check")

        assert len(result) == 1
        r = result[0]
        assert r.run_id == "run-check"
        assert r.sha256 == "c" * 64
        assert r.user == "ptcv-system"
        assert r.registry_id == "2024-000001-10-00"
        assert r.source == "EU-CTR"
        assert r.id >= 1  # auto-assigned by SQLite
