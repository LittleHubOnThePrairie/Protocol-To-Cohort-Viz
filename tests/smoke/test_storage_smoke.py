"""Smoke tests for PTCV storage module.

OQ: Verifies the storage stack is importable and the filesystem adapter
can initialise and round-trip an artifact without errors.
Risk tier: MEDIUM — pipeline storage smoke.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


def test_storage_imports():
    """All public exports from ptcv.storage are importable."""
    from ptcv.storage import (
        ArtifactRecord,
        FilesystemAdapter,
        LineageRecord,
        LocalStorageAdapter,
        StorageGateway,
    )
    assert StorageGateway is not None
    assert FilesystemAdapter is not None
    assert LocalStorageAdapter is not None
    assert ArtifactRecord is not None
    assert LineageRecord is not None


def test_filesystem_adapter_round_trip(tmp_path):
    """FilesystemAdapter can write and read back an artifact."""
    from ptcv.storage import FilesystemAdapter

    adapter = FilesystemAdapter(root=tmp_path / "protocols")
    adapter.initialise()

    data = b"smoke test content"
    artifact = adapter.put_artifact(
        key="clinicaltrials/NCT00000001_1.0.pdf",
        data=data,
        content_type="application/pdf",
        run_id="smoke-run-001",
        source_hash="",
        user="smoke",
        immutable=True,
    )

    assert artifact.sha256 != ""
    assert len(artifact.sha256) == 64

    retrieved = adapter.get_artifact("clinicaltrials/NCT00000001_1.0.pdf")
    assert retrieved == data


def test_filesystem_adapter_lineage_appended(tmp_path):
    """FilesystemAdapter appends a lineage record on put_artifact."""
    from ptcv.storage import FilesystemAdapter

    adapter = FilesystemAdapter(root=tmp_path / "protocols")
    adapter.initialise()

    adapter.put_artifact(
        key="eu-ctr/EUCT-001_01.pdf",
        data=b"protocol content",
        content_type="application/pdf",
        run_id="smoke-lineage-run",
        source_hash="",
        user="smoke",
        immutable=True,
    )

    records = adapter.get_lineage("smoke-lineage-run")
    assert len(records) == 1
    assert records[0].run_id == "smoke-lineage-run"
    assert records[0].artifact_key == "eu-ctr/EUCT-001_01.pdf"
