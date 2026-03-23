"""Query Pipeline Lineage: ALCOA+ Compliant SHA-256 Chain Tracking.

PTCV-241: Adds lineage tracking to the Query Pipeline, mirroring
the Process Pipeline's ``PipelineResult.verify_lineage_chain()`` pattern
from ``pipeline/models.py``.

Each Query Pipeline stage records a ``StageCheckpoint`` with:
- stage name
- run_id (UUID4 per stage)
- artifact SHA-256 (output hash)
- source SHA-256 (input hash — chains to previous stage's output)
- timestamp

The SHA-256 chain can be verified end-to-end via
``QueryPipelineResult.verify_lineage_chain()``.

Query Pipeline Stages (in order)::

    Stage 1: pdf_extraction     — Text + tables from PDF
    Stage 2: classification     — ICH section routing
    Stage 3: query_extraction   — Appendix B query answers
    Stage 4: assembly           — AssembledProtocol
    Stage 5: soa_extraction     — Schedule of Activities
    Stage 6: sdtm_generation    — SDTM domains + define.xml
    Stage 7: validation         — P21/TCG validation reports

Risk tier: LOW — data lineage metadata only (no patient data).

Regulatory references:
- ALCOA+ Traceable: SHA-256 chain from PDF through all stages
- ALCOA+ Contemporaneous: ISO 8601 timestamps per stage
- 21 CFR 11.10(e): audit trail via lineage chain verification

Usage::

    from ptcv.pipeline.query_pipeline_lineage import (
        QueryPipelineResult,
        record_checkpoint,
    )

    result = QueryPipelineResult(pipeline_run_id="...", registry_id="NCT...")
    record_checkpoint(result, "pdf_extraction", run_id, artifact_sha, source_sha)
    record_checkpoint(result, "classification", run_id, artifact_sha, source_sha)
    verification = result.verify_lineage_chain()
"""

import dataclasses
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

from .models import LineageChainVerification, StageCheckpoint


# Query Pipeline stage order
QUERY_PIPELINE_STAGES = [
    "pdf_extraction",
    "classification",
    "query_extraction",
    "assembly",
    "soa_extraction",
    "sdtm_generation",
    "validation",
]


@dataclasses.dataclass
class QueryPipelineResult:
    """End-to-end Query Pipeline execution result with lineage chain.

    Mirrors ``PipelineResult`` from the Process Pipeline but tailored
    for the 7-stage Query Pipeline.

    Attributes:
        pipeline_run_id: UUID4 for this pipeline execution.
        registry_id: Trial registry identifier (e.g. "NCT05376319").
        protocol_sha256: SHA-256 of the input PDF file.
        stage_checkpoints: Ordered list of stage checkpoints.
        pipeline_timestamp_utc: ISO 8601 UTC timestamp of run start.
        metadata: Optional key-value metadata.
    """

    pipeline_run_id: str
    registry_id: str
    protocol_sha256: str = ""
    stage_checkpoints: list[StageCheckpoint] = dataclasses.field(
        default_factory=list
    )
    pipeline_timestamp_utc: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def verify_lineage_chain(self) -> LineageChainVerification:
        """Verify that SHA-256 values form an unbroken chain.

        Each stage's ``source_sha256`` must match the previous stage's
        ``artifact_sha256``.  The first stage's ``source_sha256`` must
        match ``protocol_sha256`` (the input PDF hash).

        Returns:
            LineageChainVerification with is_valid=True when the
            chain is intact.
        """
        if not self.stage_checkpoints:
            return LineageChainVerification(
                is_valid=False,
                stages_verified=0,
                broken_at_stage=None,
                details=["No stage checkpoints recorded"],
            )

        details: list[str] = []
        prior_sha = self.protocol_sha256

        for i, cp in enumerate(self.stage_checkpoints):
            if prior_sha and cp.source_sha256 != prior_sha:
                details.append(
                    f"  BREAK {cp.stage}: source_sha256 mismatch — "
                    f"expected {prior_sha[:16]}... "
                    f"got {cp.source_sha256[:16]}..."
                )
                return LineageChainVerification(
                    is_valid=False,
                    stages_verified=i,
                    broken_at_stage=cp.stage,
                    details=details,
                )

            details.append(
                f"  OK {cp.stage}: "
                f"sha256={cp.artifact_sha256[:16]}... "
                f"<- source={cp.source_sha256[:16]}..."
            )
            prior_sha = cp.artifact_sha256

        return LineageChainVerification(
            is_valid=True,
            stages_verified=len(self.stage_checkpoints),
            broken_at_stage=None,
            details=details,
        )

    @property
    def stages_completed(self) -> list[str]:
        """List of completed stage names in order."""
        return [cp.stage for cp in self.stage_checkpoints]

    @property
    def is_complete(self) -> bool:
        """True when all 7 Query Pipeline stages have checkpoints."""
        return set(self.stages_completed) >= set(QUERY_PIPELINE_STAGES)

    @property
    def last_stage(self) -> Optional[str]:
        """Name of the last completed stage, or None."""
        return (
            self.stage_checkpoints[-1].stage
            if self.stage_checkpoints
            else None
        )

    @property
    def last_artifact_sha256(self) -> str:
        """SHA-256 of the most recent stage's artifact."""
        return (
            self.stage_checkpoints[-1].artifact_sha256
            if self.stage_checkpoints
            else ""
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for storage."""
        return {
            "pipeline_run_id": self.pipeline_run_id,
            "registry_id": self.registry_id,
            "protocol_sha256": self.protocol_sha256,
            "pipeline_timestamp_utc": self.pipeline_timestamp_utc,
            "stages_completed": self.stages_completed,
            "is_complete": self.is_complete,
            "checkpoints": [
                {
                    "stage": cp.stage,
                    "stage_run_id": cp.stage_run_id,
                    "artifact_key": cp.artifact_key,
                    "artifact_sha256": cp.artifact_sha256,
                    "source_sha256": cp.source_sha256,
                }
                for cp in self.stage_checkpoints
            ],
            "metadata": self.metadata,
        }


def record_checkpoint(
    result: QueryPipelineResult,
    stage: str,
    stage_run_id: str,
    artifact_sha256: str,
    source_sha256: str = "",
    artifact_key: str = "",
) -> StageCheckpoint:
    """Record a stage completion checkpoint.

    If ``source_sha256`` is not provided, automatically chains from
    the previous checkpoint's ``artifact_sha256``, or from
    ``protocol_sha256`` if this is the first stage.

    Args:
        result: The pipeline result to append to.
        stage: Stage name (must be one of QUERY_PIPELINE_STAGES).
        stage_run_id: UUID4 for this stage execution.
        artifact_sha256: SHA-256 of the stage's primary output.
        source_sha256: SHA-256 of the input (auto-chained if empty).
        artifact_key: Storage key for the output artifact.

    Returns:
        The created StageCheckpoint.
    """
    if not source_sha256:
        if result.stage_checkpoints:
            source_sha256 = result.stage_checkpoints[-1].artifact_sha256
        else:
            source_sha256 = result.protocol_sha256

    checkpoint = StageCheckpoint(
        stage=stage,
        stage_run_id=stage_run_id,
        artifact_key=artifact_key,
        artifact_sha256=artifact_sha256,
        source_sha256=source_sha256,
    )

    result.stage_checkpoints.append(checkpoint)
    return checkpoint


def compute_sha256(data: bytes) -> str:
    """Compute hex-encoded SHA-256 digest.

    Args:
        data: Bytes to hash.

    Returns:
        64-character hex string.
    """
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "QueryPipelineResult",
    "QUERY_PIPELINE_STAGES",
    "record_checkpoint",
    "compute_sha256",
]
