"""Tests for gateway_factory.create_gateway() — PTCV-24."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.pipeline.gateway_factory import create_gateway
from ptcv.storage import FilesystemAdapter, StorageGateway


class TestCreateGateway:
    def test_filesystem_backend_returns_filesystem_adapter(self, tmp_path: Path):
        gw = create_gateway(backend="filesystem", root=tmp_path)
        assert isinstance(gw, FilesystemAdapter)

    def test_local_fs_alias_returns_filesystem_adapter(self, tmp_path: Path):
        gw = create_gateway(backend="local_fs", root=tmp_path)
        assert isinstance(gw, FilesystemAdapter)

    def test_default_backend_is_filesystem(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("PTCV_STORAGE_BACKEND", raising=False)
        gw = create_gateway(backend="", root=tmp_path)
        assert isinstance(gw, FilesystemAdapter)

    def test_env_var_selects_backend(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PTCV_STORAGE_BACKEND", "filesystem")
        gw = create_gateway(backend="", root=tmp_path)
        assert isinstance(gw, FilesystemAdapter)

    def test_returns_storage_gateway_interface(self, tmp_path: Path):
        gw = create_gateway(backend="filesystem", root=tmp_path)
        assert isinstance(gw, StorageGateway)

    def test_gateway_is_initialised(self, tmp_path: Path):
        """Gateway is initialised on creation (directories exist)."""
        gw = create_gateway(backend="filesystem", root=tmp_path)
        # FilesystemAdapter creates subdirectories on initialise()
        assert tmp_path.exists()

    def test_unknown_backend_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown PTCV_STORAGE_BACKEND"):
            create_gateway(backend="nonexistent_backend", root=tmp_path)

    def test_gateway_can_put_and_get_artifact(self, tmp_path: Path):
        """Smoke-test: gateway returned by factory supports put/get."""
        gw = create_gateway(backend="filesystem", root=tmp_path)
        import uuid
        run_id = str(uuid.uuid4())
        gw.put_artifact(
            key=f"test/{run_id}/hello.txt",
            data=b"hello world",
            content_type="text/plain",
            run_id=run_id,
            source_hash="",
            user="test",
            immutable=False,
            stage="test",
        )
        data = gw.get_artifact(f"test/{run_id}/hello.txt")
        assert data == b"hello world"

    def test_explicit_root_is_used(self, tmp_path: Path):
        """Factory uses the provided root for FilesystemAdapter."""
        custom_root = tmp_path / "custom_root"
        custom_root.mkdir()
        gw = create_gateway(backend="filesystem", root=custom_root)
        # Should be able to put artifact under the custom root
        import uuid
        run_id = str(uuid.uuid4())
        gw.put_artifact(
            key=f"test/{run_id}/data.bin",
            data=b"data",
            content_type="application/octet-stream",
            run_id=run_id,
            source_hash="",
            user="test",
            immutable=False,
            stage="test",
        )
        data = gw.get_artifact(f"test/{run_id}/data.bin")
        assert data == b"data"
