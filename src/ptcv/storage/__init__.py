"""PTCV Storage Gateway package.

Public API:

    StorageGateway          — Abstract base class for all adapters
    FilesystemAdapter       — Local filesystem + SQLite (dev/test default)
    LocalStorageAdapter     — MinIO WORM + SQLite (production)
    ArtifactRecord          — Data model returned by put_artifact()
    LineageRecord           — Data model returned by get_lineage()
"""

from .gateway import StorageGateway
from .filesystem_adapter import FilesystemAdapter
from .local_adapter import LocalStorageAdapter
from .models import ArtifactRecord, LineageRecord

__all__ = [
    "StorageGateway",
    "FilesystemAdapter",
    "LocalStorageAdapter",
    "ArtifactRecord",
    "LineageRecord",
]
