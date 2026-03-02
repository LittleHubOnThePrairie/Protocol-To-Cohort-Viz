"""Tests for PTCV-29 FilesystemAdapter (local filesystem + SQLite lineage).

Qualification phase: OQ (operational qualification)
Regulatory requirement: ALCOA+ Original (immutable=True raises FileExistsError),
  ALCOA+ Traceable (every put writes lineage record), 21 CFR 11.10(e).
Risk tier: MEDIUM

GHERKIN Scenarios covered:
- Scenario 1 (initialise creates schema): test_initialise_creates_dirs_and_schema
- Scenario 2 (put_artifact records lineage): test_put_artifact_writes_file_and_lineage
- Scenario 3 (WORM prevents overwrite): test_put_artifact_immutable_raises_on_duplicate
- Scenario 4 (amendment = new object): test_different_amendments_stored_separately
- Scenario 5 (get_lineage run records): test_get_lineage_returns_records_for_run
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.storage import FilesystemAdapter
from ptcv.storage.models import ArtifactRecord, LineageRecord


@pytest.fixture()
def adapter(tmp_path: Path) -> FilesystemAdapter:
    """FilesystemAdapter rooted at a temporary directory."""
    a = FilesystemAdapter(root=tmp_path / "protocols")
    a.initialise()
    return a


class TestFilesystemAdapterInitialise:
    """IQ: initialise() creates directories and SQLite schema."""

    def test_initialise_creates_dirs_and_schema(self, tmp_path):
        """PTCV-29 Scenario 1: initialise creates directories and lineage schema."""
        root = tmp_path / "protocols"
        a = FilesystemAdapter(root=root)
        a.initialise()

        assert (root / "eu-ctr").is_dir()
        assert (root / "clinicaltrials").is_dir()
        assert (root / "metadata").is_dir()
        assert (root / "lineage.db").exists()

    def test_initialise_is_idempotent(self, tmp_path):
        """Calling initialise() twice does not raise."""
        a = FilesystemAdapter(root=tmp_path / "protocols")
        a.initialise()
        a.initialise()  # Should not raise

    def test_lineage_schema_has_append_only_triggers(self, tmp_path):
        """lineage_records table has prevent_update and prevent_delete triggers."""
        import sqlite3
        a = FilesystemAdapter(root=tmp_path / "protocols")
        a.initialise()
        conn = sqlite3.connect(str(a._db_path))
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "prevent_update" in names
        assert "prevent_delete" in names
        conn.close()


class TestFilesystemAdapterPutArtifact:
    """OQ: put_artifact() stores file, computes SHA-256, writes lineage."""

    def test_put_artifact_writes_file(self, adapter):
        """PTCV-29 Scenario 2: artifact bytes written to disk."""
        data = b"%PDF-1.4 test content"
        artifact = adapter.put_artifact(
            key="clinicaltrials/NCT00112827_00.pdf",
            data=data,
            content_type="application/pdf",
            run_id="run-001",
            source_hash="",
            user="tester",
            immutable=True,
        )
        dest = adapter.root / "clinicaltrials" / "NCT00112827_00.pdf"
        assert dest.exists()
        assert dest.read_bytes() == data

    def test_put_artifact_returns_correct_sha256(self, adapter):
        """SHA-256 in ArtifactRecord matches content."""
        data = b"hello world"
        artifact = adapter.put_artifact(
            key="clinicaltrials/NCT00001_00.pdf",
            data=data,
            content_type="application/pdf",
            run_id="run-002",
            source_hash="",
            user="tester",
        )
        assert artifact.sha256 == hashlib.sha256(data).hexdigest()

    def test_put_artifact_returns_artifact_record(self, adapter):
        """put_artifact returns a populated ArtifactRecord."""
        artifact = adapter.put_artifact(
            key="eu-ctr/2024-123456-10-00_00.pdf",
            data=b"content",
            content_type="application/pdf",
            run_id="run-003",
            source_hash="",
            user="ptcv-service",
        )
        assert isinstance(artifact, ArtifactRecord)
        assert artifact.key == "eu-ctr/2024-123456-10-00_00.pdf"
        assert artifact.version_id == ""  # no versioning on filesystem
        assert artifact.run_id == "run-003"
        assert artifact.user == "ptcv-service"

    def test_put_artifact_immutable_raises_on_duplicate(self, adapter):
        """PTCV-29 Scenario 3: immutable=True raises FileExistsError on second write."""
        data = b"original"
        adapter.put_artifact(
            key="clinicaltrials/NCT99999_00.pdf",
            data=data,
            content_type="application/pdf",
            run_id="run-004",
            source_hash="",
            user="tester",
            immutable=True,
        )
        with pytest.raises(FileExistsError, match="already exists"):
            adapter.put_artifact(
                key="clinicaltrials/NCT99999_00.pdf",
                data=b"different content",
                content_type="application/pdf",
                run_id="run-005",
                source_hash="",
                user="tester",
                immutable=True,
            )

    def test_put_artifact_mutable_overwrites(self, adapter):
        """immutable=False allows overwriting (metadata artifacts)."""
        adapter.put_artifact(
            key="metadata/NCT99999_00.json",
            data=b'{"version": 1}',
            content_type="application/json",
            run_id="run-006",
            source_hash="",
            user="tester",
            immutable=False,
        )
        adapter.put_artifact(
            key="metadata/NCT99999_00.json",
            data=b'{"version": 2}',
            content_type="application/json",
            run_id="run-007",
            source_hash="",
            user="tester",
            immutable=False,
        )
        content = (adapter.root / "metadata" / "NCT99999_00.json").read_bytes()
        assert b'"version": 2' in content

    def test_different_amendments_stored_separately(self, adapter):
        """PTCV-29 Scenario 4: different amendment numbers create separate objects."""
        adapter.put_artifact(
            key="clinicaltrials/NCT11111_00.pdf",
            data=b"amendment 00",
            content_type="application/pdf",
            run_id="run-008",
            source_hash="",
            user="tester",
            immutable=True,
        )
        adapter.put_artifact(
            key="clinicaltrials/NCT11111_01.pdf",
            data=b"amendment 01",
            content_type="application/pdf",
            run_id="run-009",
            source_hash="",
            user="tester",
            immutable=True,
        )
        assert (adapter.root / "clinicaltrials" / "NCT11111_00.pdf").exists()
        assert (adapter.root / "clinicaltrials" / "NCT11111_01.pdf").exists()


class TestFilesystemAdapterGetArtifact:
    """OQ: get_artifact() round-trip."""

    def test_get_artifact_returns_stored_bytes(self, adapter):
        """Bytes retrieved match bytes stored."""
        data = b"round-trip test content"
        adapter.put_artifact(
            key="clinicaltrials/NCT22222_00.pdf",
            data=data,
            content_type="application/pdf",
            run_id="run-010",
            source_hash="",
            user="tester",
        )
        retrieved = adapter.get_artifact("clinicaltrials/NCT22222_00.pdf")
        assert retrieved == data

    def test_get_artifact_missing_raises_file_not_found(self, adapter):
        """FileNotFoundError raised for non-existent key."""
        with pytest.raises(FileNotFoundError):
            adapter.get_artifact("clinicaltrials/DOESNOTEXIST.pdf")

    def test_get_artifact_version_id_ignored(self, adapter):
        """version_id argument is accepted and ignored on filesystem."""
        data = b"versioned data"
        adapter.put_artifact(
            key="clinicaltrials/NCT33333_00.pdf",
            data=data,
            content_type="application/pdf",
            run_id="run-011",
            source_hash="",
            user="tester",
        )
        retrieved = adapter.get_artifact(
            "clinicaltrials/NCT33333_00.pdf", version_id="ignored"
        )
        assert retrieved == data


class TestFilesystemAdapterListVersions:
    """OQ: list_versions() returns single-element list for filesystem."""

    def test_list_versions_returns_one_entry(self, adapter):
        """PTCV-29: filesystem has no versioning — list_versions returns one."""
        adapter.put_artifact(
            key="clinicaltrials/NCT44444_00.pdf",
            data=b"one version",
            content_type="application/pdf",
            run_id="run-012",
            source_hash="",
            user="tester",
        )
        versions = adapter.list_versions("clinicaltrials/NCT44444_00.pdf")
        assert len(versions) == 1
        assert versions[0].key == "clinicaltrials/NCT44444_00.pdf"
        assert versions[0].version_id == ""

    def test_list_versions_missing_returns_empty(self, adapter):
        """list_versions for non-existent key returns empty list."""
        versions = adapter.list_versions("clinicaltrials/MISSING.pdf")
        assert versions == []


class TestFilesystemAdapterGetLineage:
    """OQ: get_lineage() returns correct records from SQLite."""

    def test_get_lineage_returns_records_for_run(self, adapter):
        """PTCV-29 Scenario 5: get_lineage returns lineage records for a run_id."""
        run_id = "run-lineage-001"
        adapter.put_artifact(
            key="clinicaltrials/NCT55555_00.pdf",
            data=b"protocol content",
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
        assert isinstance(rec, LineageRecord)
        assert rec.run_id == run_id
        assert rec.artifact_key == "clinicaltrials/NCT55555_00.pdf"
        assert rec.registry_id == "NCT55555"
        assert rec.amendment_number == "00"
        assert rec.source == "ClinicalTrials.gov"
        assert rec.user == "ptcv-service"

    def test_get_lineage_empty_for_unknown_run(self, adapter):
        """get_lineage returns empty list for unknown run_id."""
        records = adapter.get_lineage("nonexistent-run-id")
        assert records == []

    def test_get_lineage_multiple_artifacts_same_run(self, adapter):
        """Both protocol and metadata lineage records are returned for a run."""
        run_id = "run-lineage-002"
        adapter.put_artifact(
            key="clinicaltrials/NCT66666_00.pdf",
            data=b"protocol",
            content_type="application/pdf",
            run_id=run_id,
            source_hash="",
            user="tester",
            immutable=True,
        )
        adapter.put_artifact(
            key="metadata/NCT66666_00.json",
            data=b'{"registry_id": "NCT66666"}',
            content_type="application/json",
            run_id=run_id,
            source_hash="abc",
            user="tester",
            immutable=False,
        )
        records = adapter.get_lineage(run_id)
        assert len(records) == 2
        keys = [r.artifact_key for r in records]
        assert "clinicaltrials/NCT66666_00.pdf" in keys
        assert "metadata/NCT66666_00.json" in keys

    def test_get_lineage_different_runs_isolated(self, adapter):
        """Records for different run_ids are not mixed together."""
        for i in range(3):
            adapter.put_artifact(
                key=f"clinicaltrials/NCT{i:05d}_00.pdf",
                data=b"content",
                content_type="application/pdf",
                run_id=f"run-iso-{i}",
                source_hash="",
                user="tester",
            )
        assert len(adapter.get_lineage("run-iso-0")) == 1
        assert len(adapter.get_lineage("run-iso-1")) == 1
        assert len(adapter.get_lineage("run-iso-2")) == 1
