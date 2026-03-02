"""StorageGateway factory for the PTCV pipeline.

Reads PTCV_STORAGE_BACKEND from the environment (default: "filesystem")
and returns a configured StorageGateway instance.

Supported backends:
- "filesystem": FilesystemAdapter backed by local directories (dev/test)
- "local":      LocalStorageAdapter backed by MinIO WORM (production)

Risk tier: LOW — factory / configuration module.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..storage import FilesystemAdapter, StorageGateway

_DEFAULT_ROOT = Path("C:/Dev/PTCV/data")

# Env var that controls the storage backend
_BACKEND_ENV = "PTCV_STORAGE_BACKEND"


def create_gateway(
    backend: str = "",
    root: Path = _DEFAULT_ROOT,
) -> StorageGateway:
    """Create and initialise a StorageGateway for the given backend.

    Args:
        backend: Backend identifier. If empty, reads PTCV_STORAGE_BACKEND
            env var. Defaults to "filesystem" when env var is also absent.
        root: Filesystem root for FilesystemAdapter (ignored for other
            backends).

    Returns:
        Initialised StorageGateway implementation.

    Raises:
        ValueError: If the backend identifier is unrecognised.
    [PTCV-24 Scenario: StorageGateway initialises on first run]
    """
    resolved = backend or os.environ.get(_BACKEND_ENV, "filesystem")

    if resolved in ("filesystem", "local_fs"):
        gw: StorageGateway = FilesystemAdapter(root=root)
        gw.initialise()
        return gw

    if resolved == "local":
        # LocalStorageAdapter requires MinIO credentials in environment
        from ..storage import LocalStorageAdapter  # type: ignore[attr-defined]

        gw = LocalStorageAdapter(
            endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            db_path=Path(
                os.environ.get("PTCV_SQLITE_PATH", str(root / "sqlite/lineage.db"))
            ),
        )
        gw.initialise()
        return gw

    raise ValueError(
        f"Unknown PTCV_STORAGE_BACKEND: {resolved!r}. "
        "Supported values: 'filesystem', 'local'."
    )
