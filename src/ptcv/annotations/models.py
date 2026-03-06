"""Annotation data models (PTCV-40, PTCV-41, PTCV-43).

AnnotationRecord holds a single reviewer decision for one classified
section or a manual span annotation. AnnotationSession groups all
annotations for a protocol run.

Risk tier: LOW — reviewer metadata only (no patient data).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from typing import Optional


@dataclasses.dataclass
class AnnotationRecord:
    """One reviewer annotation for a classified ICH section or manual span.

    Attributes:
        section_id: Unique identifier ``{run_id}:{section_code}``.
            For manual spans: ``{run_id}:manual:{index}``.
        section_code: ICH section code (e.g. "B.3").
        original_label: Classifier-assigned section name.
            Empty string for manual span annotations.
        confidence: Classifier confidence score 0.0-1.0.
            0.0 for manual span annotations.
        reviewer_label: Reviewer-assigned label (may match original).
        reviewer_action: One of "accept", "reject", "override",
            "manual_label".
        timestamp: ISO 8601 UTC timestamp of the annotation.
        text_span: Full section text for context.
        reviewer_notes: Optional free-text notes from the reviewer.
        source: Origin of the annotation: "classifier" (default)
            or "manual_span" (reviewer-selected text).
    """

    section_id: str
    section_code: str
    original_label: str
    confidence: float
    reviewer_label: str
    reviewer_action: str
    timestamp: str
    text_span: str
    reviewer_notes: str = ""
    source: str = "classifier"

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AnnotationRecord:
        """Deserialise from a dict."""
        return cls(**{
            f.name: data[f.name]
            for f in dataclasses.fields(cls)
            if f.name in data
        })


@dataclasses.dataclass
class AnnotationSession:
    """All annotations for one protocol parse run.

    Attributes:
        registry_id: Trial registry identifier.
        run_id: ICH parser run_id for the sections being annotated.
        annotations: Map of section_id -> AnnotationRecord.
        created_at: ISO 8601 UTC timestamp of session creation.
        updated_at: ISO 8601 UTC timestamp of last update.
        total_sections: Total number of sections in the run.
    """

    registry_id: str
    run_id: str
    annotations: dict[str, AnnotationRecord] = dataclasses.field(
        default_factory=dict,
    )
    created_at: str = ""
    updated_at: str = ""
    total_sections: int = 0

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def annotated_count(self) -> int:
        """Number of sections that have been annotated."""
        return len(self.annotations)

    @property
    def is_complete(self) -> bool:
        """True when all sections have been annotated."""
        return (
            self.total_sections > 0
            and self.annotated_count >= self.total_sections
        )

    def add(self, record: AnnotationRecord) -> None:
        """Add or update an annotation record."""
        self.annotations[record.section_id] = record
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_jsonl(self) -> str:
        """Serialise all annotations as JSONL (one record per line)."""
        lines = [
            json.dumps(rec.to_dict(), ensure_ascii=False)
            for rec in self.annotations.values()
        ]
        return "\n".join(lines) + "\n" if lines else ""

    @classmethod
    def from_jsonl(
        cls,
        data: str,
        registry_id: str,
        run_id: str,
        total_sections: int = 0,
    ) -> AnnotationSession:
        """Load annotations from JSONL string."""
        session = cls(
            registry_id=registry_id,
            run_id=run_id,
            total_sections=total_sections,
        )
        for line in data.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            record = AnnotationRecord.from_dict(json.loads(line))
            session.annotations[record.section_id] = record
        return session
