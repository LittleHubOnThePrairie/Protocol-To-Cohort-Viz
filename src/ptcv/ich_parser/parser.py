"""IchParser — main ICH E6(R3) section parsing service.

Orchestrates the full parse pipeline for one protocol:
  1. Classify text → list[IchSection] via SectionClassifier
  2. Stamp extraction_timestamp_utc on every section
  3. Serialise to Parquet via sections_to_parquet()
  4. Write Parquet bytes to StorageGateway with stage="ich_parse"
  5. Enqueue low-confidence sections to ReviewQueue
  6. Return the artifact SHA-256 for downstream lineage chaining

Risk tier: MEDIUM — data pipeline ML component (no patient data).

Regulatory references:
- ALCOA+ Accurate: confidence_score required non-null; low-confidence
  sections flagged for human review
- ALCOA+ Traceable: source_sha256 links to PTCV-19 extraction artifact
- ALCOA+ Contemporaneous: extraction_timestamp_utc at write boundary
- ALCOA+ Original: re-run creates new run_id, prior Parquet preserved
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..storage import FilesystemAdapter, StorageGateway
from .classifier import RuleBasedClassifier, SectionClassifier
from .format_detector import FormatDetector, ProtocolFormat
from .models import IchSection, ReviewQueueEntry
from .parquet_writer import sections_to_parquet
from .query_schema import section_sort_key
from .review_queue import ReviewQueue


_DEFAULT_REVIEW_DB = Path("C:/Dev/PTCV/data/sqlite/review_queue.db")
_USER = "ptcv-ich-parser"

# ICH E6(R3) Appendix B sections required for a complete protocol
_REQUIRED_SECTIONS: frozenset[str] = frozenset({"B.3", "B.4", "B.5"})


def _compute_format_verdict(
    sections: list[IchSection],
) -> tuple[str, float, list[str]]:
    """Compute protocol format verdict from classified ICH sections.

    Args:
        sections: IchSection list produced by the classifier.

    Returns:
        Tuple of (format_verdict, format_confidence, missing_required_sections).
        format_verdict is one of "ICH_E6R3", "PARTIAL_ICH", or "NON_ICH".
        format_confidence is a float in [0.0, 1.0].
        missing_required_sections lists required codes absent from sections.
    [PTCV-30 Scenario: Format verdict computed from section distribution]
    """
    from statistics import mean

    found_codes = {s.section_code for s in sections}
    avg_conf = mean(s.confidence_score for s in sections) if sections else 0.0
    missing = sorted(_REQUIRED_SECTIONS - found_codes, key=section_sort_key)

    n_required_found = len(found_codes & _REQUIRED_SECTIONS)
    format_confidence = avg_conf * (n_required_found / len(_REQUIRED_SECTIONS))

    if format_confidence >= 0.60 and _REQUIRED_SECTIONS <= found_codes:
        verdict = "ICH_E6R3"
    elif format_confidence >= 0.30 or len(sections) >= 3:
        verdict = "PARTIAL_ICH"
    else:
        verdict = "NON_ICH"

    return verdict, round(format_confidence, 4), missing


class ParseResult:
    """Result returned by IchParser.parse().

    Attributes:
        run_id: UUID4 for this parse run.
        registry_id: Trial identifier.
        artifact_key: Storage key of the sections.parquet artifact.
        artifact_sha256: SHA-256 of the stored Parquet bytes.
        section_count: Number of sections classified and stored.
        review_count: Number of sections routed to review queue.
        source_sha256: SHA-256 of the upstream artifact (from input).
    """

    def __init__(
        self,
        run_id: str,
        registry_id: str,
        artifact_key: str,
        artifact_sha256: str,
        section_count: int,
        review_count: int,
        source_sha256: str,
        format_verdict: str = "NON_ICH",
        format_confidence: float = 0.0,
        missing_required_sections: Optional[list[str]] = None,
        detected_format: str = "UNKNOWN",
    ) -> None:
        self.run_id = run_id
        self.registry_id = registry_id
        self.artifact_key = artifact_key
        self.artifact_sha256 = artifact_sha256
        self.section_count = section_count
        self.review_count = review_count
        self.source_sha256 = source_sha256
        self.format_verdict = format_verdict
        self.format_confidence = format_confidence
        self.missing_required_sections: list[str] = (
            missing_required_sections if missing_required_sections is not None else []
        )
        self.detected_format = detected_format

    def __repr__(self) -> str:
        return (
            f"ParseResult(run_id={self.run_id!r}, "
            f"registry_id={self.registry_id!r}, "
            f"sections={self.section_count}, "
            f"review={self.review_count}, "
            f"format_verdict={self.format_verdict!r})"
        )


class IchParser:
    """ICH E6(R3) section parsing service.

    Accepts raw protocol text (or pre-extracted text bytes from PTCV-19),
    classifies it into ICH Appendix B sections, stores the result as
    DuckDB-compatible Parquet via StorageGateway, and routes low-confidence
    sections to the human review queue.

    Args:
        gateway: StorageGateway instance. Uses FilesystemAdapter with
            default PTCV data root if None.
        classifier: SectionClassifier backend. Uses RuleBasedClassifier
            if None. Pass RAGClassifier() for production use.
        review_queue: ReviewQueue instance. Uses default SQLite path if None.
    [PTCV-20 Scenario: Classify protocol sections and write to Parquet]
    """

    def __init__(
        self,
        gateway: Optional[StorageGateway] = None,
        classifier: Optional[SectionClassifier] = None,
        review_queue: Optional[ReviewQueue] = None,
    ) -> None:
        if gateway is None:
            gateway = FilesystemAdapter(root=Path("C:/Dev/PTCV/data"))
        if classifier is None:
            classifier = RuleBasedClassifier()
        if review_queue is None:
            review_queue = ReviewQueue(db_path=_DEFAULT_REVIEW_DB)

        self._gateway = gateway
        self._classifier = classifier
        self._review_queue = review_queue

        self._gateway.initialise()
        self._review_queue.initialise()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse(
        self,
        text: str,
        registry_id: str,
        source_run_id: str = "",
        source_sha256: str = "",
        who: str = _USER,
    ) -> ParseResult:
        """Classify protocol text and write ICH sections to Parquet.

        Creates a new run_id for this parse run. Prior Parquet files
        under different run_ids are never touched (ALCOA+ Original).

        Args:
            text: Full protocol text to classify (UTF-8 string).
            registry_id: Trial identifier (EUCT-Code or NCT-ID).
            source_run_id: run_id from the upstream PTCV-19 extraction.
                Pass "" when providing text directly.
            source_sha256: SHA-256 of the upstream extraction artifact.
                Pass "" when providing text directly.
            who: Actor identifier for audit trail.

        Returns:
            ParseResult with run_id, artifact key/sha256, and counts.

        Raises:
            ValueError: If text is empty.
        [PTCV-20 Scenario: Classify protocol sections and write to Parquet]
        [PTCV-20 Scenario: Re-run creates new run_id, prior parse preserved]
        """
        if not text.strip():
            raise ValueError("text must not be empty")

        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # 0. Pre-parser format detection (PTCV-31)
        # Runs before classification to detect the protocol format family.
        # Non-ICH protocols are flagged via a review queue entry but
        # classification still proceeds (legacy fallback) so downstream
        # pipeline stages always receive a valid sections artifact.
        detector = FormatDetector()
        fmt_result = detector.detect(text)
        detected_format = fmt_result.format.value

        if (
            fmt_result.format == ProtocolFormat.UNKNOWN
            and fmt_result.confidence < 0.20
        ):
            self._review_queue.enqueue(
                ReviewQueueEntry(
                    run_id=run_id,
                    registry_id=registry_id,
                    section_code="FORMAT",
                    confidence_score=fmt_result.confidence,
                    content_json="{}",
                    queue_timestamp_utc=timestamp,
                )
            )

        # 1. Classify
        sections = self._classifier.classify(
            text=text,
            registry_id=registry_id,
            run_id=run_id,
            source_run_id=source_run_id,
            source_sha256=source_sha256,
        )

        if not sections:
            # No sections detected — treat entire text as a legacy single block
            sections = self._legacy_fallback(
                text, registry_id, run_id, source_run_id, source_sha256
            )

        # 2a. Compute format verdict from section distribution (PTCV-30)
        format_verdict, format_confidence, missing_required = (
            _compute_format_verdict(sections)
        )

        # 2b. Stamp timestamp on every section (ALCOA+ Contemporaneous)
        for sec in sections:
            sec.extraction_timestamp_utc = timestamp


        # 3. Serialise to Parquet
        parquet_bytes = sections_to_parquet(sections)

        # 4. Write to StorageGateway
        artifact_key = f"ich-json/{run_id}/sections.parquet"
        artifact = self._gateway.put_artifact(
            key=artifact_key,
            data=parquet_bytes,
            content_type="application/vnd.apache.parquet",
            run_id=run_id,
            source_hash=source_sha256,
            user=who,
            immutable=False,
            stage="ich_parse",
            registry_id=registry_id,
        )

        # 5. Enqueue low-confidence sections for human review
        review_count = 0
        for sec in sections:
            if sec.review_required:
                self._review_queue.enqueue(
                    ReviewQueueEntry(
                        run_id=run_id,
                        registry_id=registry_id,
                        section_code=sec.section_code,
                        confidence_score=sec.confidence_score,
                        content_json=sec.content_json,
                        queue_timestamp_utc=timestamp,
                    )
                )
                review_count += 1

        return ParseResult(
            run_id=run_id,
            registry_id=registry_id,
            artifact_key=artifact_key,
            artifact_sha256=artifact.sha256,
            section_count=len(sections),
            review_count=review_count,
            source_sha256=source_sha256,
            format_verdict=format_verdict,
            format_confidence=format_confidence,
            missing_required_sections=missing_required,
            detected_format=detected_format,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _legacy_fallback(
        self,
        text: str,
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        """Return a single catch-all section for unclassifiable protocols.

        When no sections are detected by the primary classifier, the entire
        text is stored as a single B.1 entry with legacy_format=True and
        confidence_score=0.10 (below review threshold).
        [PTCV-20 Scenario: Legacy protocol handled with flag]
        """
        import json

        return [
            IchSection(
                run_id=run_id,
                source_run_id=source_run_id,
                source_sha256=source_sha256,
                registry_id=registry_id,
                section_code="B.1",
                section_name="General Information",
                content_json=json.dumps(
                    {
                        "text_excerpt": text[:2000].strip(),
                        "word_count": len(text.split()),
                        "note": "Legacy fallback — no ICH sections detected",
                    },
                    ensure_ascii=False,
                ),
                confidence_score=0.10,
                review_required=True,
                legacy_format=True,
                content_text=text,
            )
        ]
