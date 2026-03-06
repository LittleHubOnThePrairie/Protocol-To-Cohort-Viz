"""Annotation persistence service (PTCV-40).

Stores and retrieves annotation sessions via StorageGateway, using
JSONL format under ``annotations/{registry_id}/{run_id}.jsonl``.

Risk tier: LOW — reviewer metadata only (no patient data).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from .models import AnnotationRecord, AnnotationSession

if TYPE_CHECKING:
    from ptcv.storage.gateway import StorageGateway


# Storage key pattern
_KEY_TEMPLATE = "annotations/{registry_id}/{run_id}.jsonl"


def _storage_key(registry_id: str, run_id: str) -> str:
    """Build the storage key for an annotation session."""
    return _KEY_TEMPLATE.format(
        registry_id=registry_id,
        run_id=run_id,
    )


class AnnotationService:
    """CRUD service for annotation sessions.

    Args:
        gateway: StorageGateway instance for persistence.
        confidence_threshold: Sections below this score are flagged
            as low confidence. Default 0.80.
    """

    def __init__(
        self,
        gateway: "StorageGateway",
        confidence_threshold: float = 0.80,
    ) -> None:
        self._gateway = gateway
        self.confidence_threshold = confidence_threshold

    def save(self, session: AnnotationSession) -> str:
        """Persist an annotation session as JSONL.

        Args:
            session: The annotation session to save.

        Returns:
            The storage key where the session was written.
        """
        key = _storage_key(session.registry_id, session.run_id)
        data = session.to_jsonl().encode("utf-8")
        self._gateway.put_artifact(
            key=key,
            data=data,
            content_type="application/jsonl",
            run_id=str(uuid.uuid4()),
            source_hash="",
            user="streamlit-reviewer",
            stage="annotation",
            registry_id=session.registry_id,
        )
        return key

    def load(
        self,
        registry_id: str,
        run_id: str,
        total_sections: int = 0,
    ) -> Optional[AnnotationSession]:
        """Load an existing annotation session if one exists.

        Args:
            registry_id: Trial registry identifier.
            run_id: ICH parser run_id.
            total_sections: Total section count for completeness tracking.

        Returns:
            AnnotationSession if found, None otherwise.
        """
        key = _storage_key(registry_id, run_id)
        try:
            data = self._gateway.get_artifact(key)
        except FileNotFoundError:
            return None
        return AnnotationSession.from_jsonl(
            data.decode("utf-8"),
            registry_id=registry_id,
            run_id=run_id,
            total_sections=total_sections,
        )

    def classify_confidence(
        self,
        score: float,
    ) -> str:
        """Classify a confidence score as 'high' or 'low'.

        Args:
            score: Confidence score 0.0-1.0.

        Returns:
            "high" if >= threshold, "low" otherwise.
        """
        return "high" if score >= self.confidence_threshold else "low"
