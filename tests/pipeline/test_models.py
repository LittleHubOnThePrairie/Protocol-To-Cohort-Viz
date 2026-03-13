"""Tests for PipelineResult and LineageChainVerification models — PTCV-24/60."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.pipeline.models import (
    LineageChainVerification,
    PipelineResult,
    StageCheckpoint,
    PIPELINE_STAGES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sha(n: int = 0) -> str:
    """Return a deterministic 64-char hex string for testing."""
    return format(n, "064x")


def make_checkpoints(chain: bool = True) -> list[StageCheckpoint]:
    """Build 8 sequential stage checkpoints, optionally with a broken chain."""
    sha0 = make_sha(0)  # protocol_sha256
    sha1 = make_sha(1)
    sha2 = make_sha(2)
    sha2a = make_sha(20)  # classification_cascade
    sha3 = make_sha(3)
    sha4 = make_sha(4)
    sha5 = make_sha(5)
    sha6 = make_sha(6)

    # chain: each artifact_sha256 becomes the next stage's source_sha256
    return [
        StageCheckpoint("download", "run-0", "key-0", sha0, ""),
        StageCheckpoint("extraction", "run-1", "key-1", sha1, sha0),
        StageCheckpoint("soa_extraction", "run-2", "key-2", sha2, sha1),
        StageCheckpoint(
            "classification_cascade", "run-2a", "key-2a", sha2a, sha2,
        ),
        StageCheckpoint(
            "retemplating",
            "run-3",
            "key-3",
            sha3,
            sha2a if chain else make_sha(99),  # break chain here
        ),
        StageCheckpoint("coverage_review", "run-4", "key-4", sha4, sha3),
        StageCheckpoint("sdtm_generation", "run-5", "key-5", sha5, sha4),
        StageCheckpoint("validation", "run-6", "key-6", sha6, sha5),
    ]


def make_result(checkpoints: list[StageCheckpoint] | None = None) -> PipelineResult:
    if checkpoints is None:
        checkpoints = make_checkpoints()
    return PipelineResult(
        pipeline_run_id=str(uuid.uuid4()),
        registry_id="NCT00000001",
        amendment_number="00",
        extraction_result=None,
        soa_result=None,
        retemplating_result=None,
        coverage_result=None,
        sdtm_result=None,
        validation_result=None,
        protocol_sha256=make_sha(0),
        stage_checkpoints=checkpoints,
        pipeline_timestamp_utc="2024-01-15T10:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# PIPELINE_STAGES constant
# ---------------------------------------------------------------------------


class TestPipelineStages:
    def test_eight_stages_defined(self):
        assert len(PIPELINE_STAGES) == 8

    def test_stages_in_order(self):
        assert PIPELINE_STAGES == [
            "download",
            "extraction",
            "soa_extraction",
            "classification_cascade",
            "retemplating",
            "coverage_review",
            "sdtm_generation",
            "validation",
        ]


# ---------------------------------------------------------------------------
# StageCheckpoint
# ---------------------------------------------------------------------------


class TestStageCheckpoint:
    def test_fields_accessible(self):
        cp = StageCheckpoint(
            stage="extraction",
            stage_run_id="run-001",
            artifact_key="extracted/run-001/text_blocks.parquet",
            artifact_sha256=make_sha(1),
            source_sha256=make_sha(0),
        )
        assert cp.stage == "extraction"
        assert cp.stage_run_id == "run-001"
        assert cp.artifact_sha256 == make_sha(1)
        assert cp.source_sha256 == make_sha(0)


# ---------------------------------------------------------------------------
# PipelineResult.verify_lineage_chain
# ---------------------------------------------------------------------------


class TestVerifyLineageChain:
    def test_unbroken_chain_is_valid(self):
        result = make_result(make_checkpoints(chain=True))
        verification = result.verify_lineage_chain()
        assert verification.is_valid

    def test_unbroken_chain_stages_verified(self):
        result = make_result(make_checkpoints(chain=True))
        verification = result.verify_lineage_chain()
        assert verification.stages_verified == 8

    def test_unbroken_chain_broken_at_stage_is_none(self):
        result = make_result(make_checkpoints(chain=True))
        verification = result.verify_lineage_chain()
        assert verification.broken_at_stage is None

    def test_broken_chain_is_invalid(self):
        result = make_result(make_checkpoints(chain=False))
        verification = result.verify_lineage_chain()
        assert not verification.is_valid

    def test_broken_chain_identifies_stage(self):
        result = make_result(make_checkpoints(chain=False))
        verification = result.verify_lineage_chain()
        # The break is at retemplating (stage index 3)
        assert verification.broken_at_stage == "retemplating"

    def test_empty_checkpoints_is_invalid(self):
        result = make_result(checkpoints=[])
        verification = result.verify_lineage_chain()
        assert not verification.is_valid
        assert verification.stages_verified == 0

    def test_details_populated(self):
        result = make_result(make_checkpoints(chain=True))
        verification = result.verify_lineage_chain()
        assert len(verification.details) >= 8

    def test_verification_is_lineage_chain_verification_type(self):
        result = make_result()
        verification = result.verify_lineage_chain()
        assert isinstance(verification, LineageChainVerification)


# ---------------------------------------------------------------------------
# PipelineResult properties
# ---------------------------------------------------------------------------


class TestPipelineResultProperties:
    def test_all_stages_complete_true_when_all_results_present(self):
        # Mock result objects (just need non-None)
        result = PipelineResult(
            pipeline_run_id=str(uuid.uuid4()),
            registry_id="NCT0001",
            amendment_number="00",
            extraction_result=object(),       # type: ignore[arg-type]
            soa_result=object(),              # type: ignore[arg-type]
            retemplating_result=object(),     # type: ignore[arg-type]
            coverage_result=object(),         # type: ignore[arg-type]
            sdtm_result=object(),             # type: ignore[arg-type]
            validation_result=object(),       # type: ignore[arg-type]
            protocol_sha256=make_sha(0),
            stage_checkpoints=[],
            pipeline_timestamp_utc="2024-01-15T10:00:00+00:00",
        )
        assert result.all_stages_complete is True

    def test_all_stages_complete_false_when_any_missing(self):
        result = make_result()  # all stage results are None
        assert result.all_stages_complete is False

    def test_stage_names_present_returns_set(self):
        result = make_result(make_checkpoints())
        names = result.stage_names_present
        assert isinstance(names, set)
        assert names == set(PIPELINE_STAGES)

    def test_stage_names_present_empty_when_no_checkpoints(self):
        result = make_result(checkpoints=[])
        assert result.stage_names_present == set()

    def test_pipeline_run_id_is_string(self):
        result = make_result()
        assert isinstance(result.pipeline_run_id, str)

    def test_registry_id_preserved(self):
        result = make_result()
        assert result.registry_id == "NCT00000001"

    def test_deprecated_parse_result_defaults_to_none(self):
        result = make_result()
        assert result.parse_result is None
