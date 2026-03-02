"""Tests for PTCV-29 LocalStorageAdapter (MinIO WORM + SQLite lineage).

All MinIO calls are mocked via unittest.mock.patch to avoid requiring a
live MinIO server. Tests verify that the adapter calls the MinIO client
with the expected parameters and that lineage records are written to SQLite.

Qualification phase: OQ (operational qualification)
Regulatory requirement: ALCOA+ Original (COMPLIANCE retention lock applied
  for immutable=True), ALCOA+ Traceable (lineage chain).
Risk tier: MEDIUM

GHERKIN Scenarios covered:
- Scenario 1 (initialise WORM + schema): test_initialise_creates_bucket_and_schema
- Scenario 2 (put_artifact records lock + lineage): test_put_artifact_immutable_locks_object
- Scenario 3 (WORM prevents overwrite): test_put_artifact_immutable_raises_on_duplicate
- Scenario 4 (amendment = new object): test_different_amendments_stored_as_separate_objects
- Scenario 5 (get_lineage run records): test_get_lineage_returns_records_from_sqlite
"""

import hashlib
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


def _make_adapter(tmp_path):
    """Build a LocalStorageAdapter with a mocked Minio client."""
    with patch("ptcv.storage.local_adapter.Minio") as MockMinio:
        mock_client = MagicMock()
        MockMinio.return_value = mock_client
        from ptcv.storage.local_adapter import LocalStorageAdapter
        adapter = LocalStorageAdapter(
            endpoint="localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket="protocols-test",
            db_path=tmp_path / "lineage.db",
            secure=False,
        )
        adapter._client = mock_client
        return adapter, mock_client


class TestLocalStorageAdapterInitialise:
    """IQ: initialise() creates bucket with Object Lock and SQLite schema."""

    def test_initialise_creates_bucket_and_schema(self, tmp_path):
        """PTCV-29 Scenario 1: make_bucket called when bucket does not exist."""
        adapter, mock_client = _make_adapter(tmp_path)
        mock_client.bucket_exists.return_value = False

        adapter.initialise()

        mock_client.bucket_exists.assert_called_once_with("protocols-test")
        mock_client.make_bucket.assert_called_once()
        mock_client.set_object_lock_config.assert_called_once()

    def test_initialise_skips_make_bucket_if_exists(self, tmp_path):
        """initialise() is idempotent: make_bucket not called if bucket exists."""
        adapter, mock_client = _make_adapter(tmp_path)
        mock_client.bucket_exists.return_value = True

        adapter.initialise()

        mock_client.make_bucket.assert_not_called()

    def test_initialise_creates_sqlite_schema(self, tmp_path):
        """SQLite lineage.db is created with the lineage_records table."""
        import sqlite3
        adapter, mock_client = _make_adapter(tmp_path)
        mock_client.bucket_exists.return_value = False

        adapter.initialise()

        conn = sqlite3.connect(str(tmp_path / "lineage.db"))
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "lineage_records" in tables
        conn.close()


class TestLocalStorageAdapterPutArtifact:
    """OQ: put_artifact() uploads to MinIO and writes lineage."""

    def _put_result(self, version_id="v001"):
        """Build a mock PutObjectResult."""
        result = MagicMock()
        result.version_id = version_id
        return result

    def test_put_artifact_calls_put_object(self, tmp_path):
        """PTCV-29 Scenario 2: put_object is called with correct bucket/key."""
        adapter, mock_client = _make_adapter(tmp_path)
        adapter.initialise()
        mock_client.bucket_exists.return_value = True
        # stat_object raises → object does not yet exist
        mock_client.stat_object.side_effect = Exception("not found")
        mock_client.put_object.return_value = self._put_result()

        data = b"pdf content"
        artifact = adapter.put_artifact(
            key="clinicaltrials/NCT00112827_00.pdf",
            data=data,
            content_type="application/pdf",
            run_id="run-001",
            source_hash="",
            user="tester",
            immutable=True,
        )

        mock_client.put_object.assert_called_once()
        call_args = mock_client.put_object.call_args
        assert call_args.args[0] == "protocols-test"  # bucket
        assert call_args.args[1] == "clinicaltrials/NCT00112827_00.pdf"  # key

    def test_put_artifact_immutable_sets_retention(self, tmp_path):
        """PTCV-29 Scenario 2: immutable=True calls set_object_retention."""
        adapter, mock_client = _make_adapter(tmp_path)
        adapter.initialise()
        mock_client.bucket_exists.return_value = True
        # stat_object raises → object does not yet exist
        mock_client.stat_object.side_effect = Exception("not found")
        mock_client.put_object.return_value = self._put_result(version_id="v001")

        adapter.put_artifact(
            key="clinicaltrials/NCT00112827_00.pdf",
            data=b"content",
            content_type="application/pdf",
            run_id="run-002",
            source_hash="",
            user="tester",
            immutable=True,
        )

        mock_client.set_object_retention.assert_called_once()

    def test_put_artifact_mutable_skips_retention(self, tmp_path):
        """immutable=False: put_object called but set_object_retention is not."""
        adapter, mock_client = _make_adapter(tmp_path)
        adapter.initialise()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.return_value = self._put_result()

        adapter.put_artifact(
            key="metadata/NCT00112827_00.json",
            data=b'{"v": 1}',
            content_type="application/json",
            run_id="run-003",
            source_hash="",
            user="tester",
            immutable=False,
        )

        mock_client.put_object.assert_called_once()
        mock_client.set_object_retention.assert_not_called()

    def test_put_artifact_immutable_raises_on_duplicate(self, tmp_path):
        """PTCV-29 Scenario 3: FileExistsError when object already exists."""
        adapter, mock_client = _make_adapter(tmp_path)
        adapter.initialise()
        mock_client.bucket_exists.return_value = True
        # stat_object succeeds → object exists
        mock_client.stat_object.return_value = MagicMock(version_id="existing")

        with pytest.raises(FileExistsError, match="already exists"):
            adapter.put_artifact(
                key="clinicaltrials/NCT99999_00.pdf",
                data=b"duplicate",
                content_type="application/pdf",
                run_id="run-004",
                source_hash="",
                user="tester",
                immutable=True,
            )

    def test_put_artifact_returns_artifact_record(self, tmp_path):
        """put_artifact returns ArtifactRecord with correct sha256 and version_id."""
        adapter, mock_client = _make_adapter(tmp_path)
        adapter.initialise()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.return_value = self._put_result(version_id="v002")

        data = b"test bytes"
        artifact = adapter.put_artifact(
            key="clinicaltrials/NCT11111_00.pdf",
            data=data,
            content_type="application/pdf",
            run_id="run-005",
            source_hash="",
            user="tester",
        )

        assert artifact.sha256 == hashlib.sha256(data).hexdigest()
        assert artifact.version_id == "v002"
        assert artifact.key == "clinicaltrials/NCT11111_00.pdf"

    def test_different_amendments_stored_as_separate_objects(self, tmp_path):
        """PTCV-29 Scenario 4: different amendments produce separate put_object calls."""
        adapter, mock_client = _make_adapter(tmp_path)
        adapter.initialise()
        mock_client.bucket_exists.return_value = True
        mock_client.stat_object.side_effect = Exception("not found")
        mock_client.put_object.return_value = self._put_result()

        adapter.put_artifact(
            key="clinicaltrials/NCT11111_00.pdf",
            data=b"v00",
            content_type="application/pdf",
            run_id="run-006",
            source_hash="",
            user="tester",
            immutable=True,
        )
        adapter.put_artifact(
            key="clinicaltrials/NCT11111_01.pdf",
            data=b"v01",
            content_type="application/pdf",
            run_id="run-007",
            source_hash="",
            user="tester",
            immutable=True,
        )

        assert mock_client.put_object.call_count == 2
        keys = [
            call.args[1] for call in mock_client.put_object.call_args_list
        ]
        assert "clinicaltrials/NCT11111_00.pdf" in keys
        assert "clinicaltrials/NCT11111_01.pdf" in keys


class TestLocalStorageAdapterGetLineage:
    """OQ: get_lineage() returns SQLite records."""

    def test_get_lineage_returns_records_from_sqlite(self, tmp_path):
        """PTCV-29 Scenario 5: get_lineage returns lineage for a run_id."""
        adapter, mock_client = _make_adapter(tmp_path)
        adapter.initialise()
        mock_client.bucket_exists.return_value = True
        mock_client.stat_object.side_effect = Exception("not found")
        mock_client.put_object.return_value = MagicMock(version_id="v001")

        run_id = "run-lineage-001"
        adapter.put_artifact(
            key="clinicaltrials/NCT55555_00.pdf",
            data=b"data",
            content_type="application/pdf",
            run_id=run_id,
            source_hash="",
            user="ptcv-service",
            registry_id="NCT55555",
            amendment_number="00",
            source="ClinicalTrials.gov",
        )

        records = adapter.get_lineage(run_id)
        assert len(records) == 1
        rec = records[0]
        assert rec.run_id == run_id
        assert rec.artifact_key == "clinicaltrials/NCT55555_00.pdf"
        assert rec.registry_id == "NCT55555"
        assert rec.source == "ClinicalTrials.gov"

    def test_get_lineage_empty_for_unknown_run(self, tmp_path):
        """get_lineage returns empty list for an unknown run_id."""
        adapter, _ = _make_adapter(tmp_path)
        adapter.initialise()
        records = adapter.get_lineage("no-such-run")
        assert records == []


class TestLocalStorageAdapterGetArtifact:
    """OQ: get_artifact() downloads from MinIO."""

    def test_get_artifact_calls_get_object(self, tmp_path):
        """get_artifact calls get_object with the correct bucket and key."""
        adapter, mock_client = _make_adapter(tmp_path)
        response_mock = MagicMock()
        response_mock.read.return_value = b"downloaded bytes"
        mock_client.get_object.return_value = response_mock

        result = adapter.get_artifact("clinicaltrials/NCT00001_00.pdf")

        mock_client.get_object.assert_called_once_with(
            "protocols-test", "clinicaltrials/NCT00001_00.pdf"
        )
        assert result == b"downloaded bytes"

    def test_get_artifact_with_version_id(self, tmp_path):
        """get_artifact passes version_id to get_object when provided."""
        adapter, mock_client = _make_adapter(tmp_path)
        response_mock = MagicMock()
        response_mock.read.return_value = b"versioned bytes"
        mock_client.get_object.return_value = response_mock

        adapter.get_artifact(
            "clinicaltrials/NCT00001_00.pdf", version_id="v001"
        )

        call_kwargs = mock_client.get_object.call_args.kwargs
        assert call_kwargs.get("version_id") == "v001"
