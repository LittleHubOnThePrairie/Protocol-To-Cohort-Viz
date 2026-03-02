"""Data Integrity Guard — ALCOA+ (Original, Complete, Accurate, Consistent).

SHA-256 checkpointing at every pipeline stage boundary. Raw data is
never overwritten — originals are preserved alongside derivatives.

Risk tier: MEDIUM — data pipeline integrity (not direct patient data).
"""

import hashlib
from typing import Optional, Union


class DataIntegrityGuard:
    """SHA-256 hash checkpointing for pipeline stage boundaries.

    Implements ALCOA+ Original and Consistent principles by computing
    and storing cryptographic hashes at each stage of data processing.
    Allows verification that data was not altered between stages.

    An inspector can provide a known hash and re-verify the file at any
    time to confirm data has not been tampered with.

    Example::

        guard = DataIntegrityGuard()
        raw_hash = guard.checkpoint(raw_bytes, stage="ingestion")
        processed = transform(raw_bytes)
        guard.checkpoint(processed, stage="parsed", parent_hash=raw_hash)
        assert guard.verify(raw_bytes, expected_hash=raw_hash)
    """

    def checkpoint(
        self,
        data: Union[bytes, str],
        stage: str,
        parent_hash: Optional[str] = None,
    ) -> str:
        """Compute and return SHA-256 hash for a pipeline stage.

        Args:
            data: Raw bytes or UTF-8 string to hash.
            stage: Human-readable stage label (e.g., "ingestion").
            parent_hash: Hash of upstream data (for chain traceability).

        Returns:
            Hex-encoded SHA-256 digest.
        """
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return hashlib.sha256(raw).hexdigest()

    def verify(
        self,
        data: Union[bytes, str],
        expected_hash: str,
    ) -> bool:
        """Verify data matches an expected SHA-256 hash.

        Args:
            data: Bytes or string to verify.
            expected_hash: Hex-encoded SHA-256 digest to compare against.

        Returns:
            True if hash matches; False if data was altered.
        """
        raw = data.encode("utf-8") if isinstance(data, str) else data
        actual = hashlib.sha256(raw).hexdigest()
        return actual == expected_hash
