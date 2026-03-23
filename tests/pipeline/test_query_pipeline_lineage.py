"""Tests for PTCV-241: Query Pipeline Lineage Tracking.

Tests verify checkpoint recording, SHA-256 chain verification,
auto-chaining, completeness detection, and serialization.
"""

import pytest

from ptcv.pipeline.query_pipeline_lineage import (
    QueryPipelineResult,
    QUERY_PIPELINE_STAGES,
    record_checkpoint,
    compute_sha256,
)
from ptcv.pipeline.models import StageCheckpoint


class TestQueryPipelineStages:
    """Tests for stage definitions."""

    def test_seven_stages(self):
        assert len(QUERY_PIPELINE_STAGES) == 7

    def test_stage_order(self):
        assert QUERY_PIPELINE_STAGES[0] == "pdf_extraction"
        assert QUERY_PIPELINE_STAGES[-1] == "validation"

    def test_expected_stages(self):
        expected = {
            "pdf_extraction", "classification", "query_extraction",
            "assembly", "soa_extraction", "sdtm_generation", "validation",
        }
        assert set(QUERY_PIPELINE_STAGES) == expected


class TestQueryPipelineResult:
    """Tests for QueryPipelineResult dataclass."""

    def test_creation(self):
        result = QueryPipelineResult(
            pipeline_run_id="run-001",
            registry_id="NCT12345678",
            protocol_sha256="abc123",
        )
        assert result.pipeline_run_id == "run-001"
        assert result.registry_id == "NCT12345678"
        assert result.protocol_sha256 == "abc123"

    def test_defaults(self):
        result = QueryPipelineResult(
            pipeline_run_id="run-001",
            registry_id="NCT12345678",
        )
        assert result.stage_checkpoints == []
        assert result.metadata == {}
        assert result.protocol_sha256 == ""

    def test_stages_completed_empty(self):
        result = QueryPipelineResult("run-001", "NCT12345678")
        assert result.stages_completed == []
        assert result.last_stage is None
        assert result.last_artifact_sha256 == ""

    def test_is_complete_false(self):
        result = QueryPipelineResult("run-001", "NCT12345678")
        assert result.is_complete is False


class TestRecordCheckpoint:
    """Tests for record_checkpoint function."""

    def test_basic_recording(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        cp = record_checkpoint(
            result, "pdf_extraction", "stage-001",
            artifact_sha256="extract_hash",
            source_sha256="pdf_hash",
        )

        assert len(result.stage_checkpoints) == 1
        assert cp.stage == "pdf_extraction"
        assert cp.artifact_sha256 == "extract_hash"
        assert cp.source_sha256 == "pdf_hash"

    def test_auto_chains_from_previous(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        record_checkpoint(result, "pdf_extraction", "s1", "hash_1", "pdf_hash")
        record_checkpoint(result, "classification", "s2", "hash_2")  # No source

        assert result.stage_checkpoints[1].source_sha256 == "hash_1"

    def test_auto_chains_from_protocol(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        record_checkpoint(result, "pdf_extraction", "s1", "hash_1")  # No source, first stage

        assert result.stage_checkpoints[0].source_sha256 == "pdf_hash"

    def test_multiple_stages(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        record_checkpoint(result, "pdf_extraction", "s1", "h1", "pdf_hash")
        record_checkpoint(result, "classification", "s2", "h2")
        record_checkpoint(result, "query_extraction", "s3", "h3")

        assert len(result.stage_checkpoints) == 3
        assert result.stages_completed == [
            "pdf_extraction", "classification", "query_extraction",
        ]
        assert result.last_stage == "query_extraction"
        assert result.last_artifact_sha256 == "h3"


class TestVerifyLineageChain:
    """Tests for SHA-256 chain verification."""

    def test_valid_chain(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        record_checkpoint(result, "pdf_extraction", "s1", "h1", "pdf_hash")
        record_checkpoint(result, "classification", "s2", "h2", "h1")
        record_checkpoint(result, "query_extraction", "s3", "h3", "h2")

        verification = result.verify_lineage_chain()

        assert verification.is_valid is True
        assert verification.stages_verified == 3
        assert verification.broken_at_stage is None
        assert len(verification.details) == 3

    def test_broken_chain(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        record_checkpoint(result, "pdf_extraction", "s1", "h1", "pdf_hash")
        record_checkpoint(result, "classification", "s2", "h2", "WRONG_HASH")

        verification = result.verify_lineage_chain()

        assert verification.is_valid is False
        assert verification.broken_at_stage == "classification"
        assert verification.stages_verified == 1

    def test_empty_chain(self):
        result = QueryPipelineResult("run-001", "NCT12345678")

        verification = result.verify_lineage_chain()

        assert verification.is_valid is False
        assert verification.stages_verified == 0
        assert "No stage checkpoints" in verification.details[0]

    def test_single_stage_valid(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        record_checkpoint(result, "pdf_extraction", "s1", "h1", "pdf_hash")

        verification = result.verify_lineage_chain()

        assert verification.is_valid is True
        assert verification.stages_verified == 1

    def test_full_7_stage_chain(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        prev = "pdf_hash"
        for i, stage in enumerate(QUERY_PIPELINE_STAGES):
            new_hash = f"hash_{i}"
            record_checkpoint(result, stage, f"s{i}", new_hash, prev)
            prev = new_hash

        verification = result.verify_lineage_chain()

        assert verification.is_valid is True
        assert verification.stages_verified == 7
        assert result.is_complete is True


class TestIsComplete:
    """Tests for completeness detection."""

    def test_complete_with_all_stages(self):
        result = QueryPipelineResult("run-001", "NCT12345678")

        prev = "pdf_hash"
        for i, stage in enumerate(QUERY_PIPELINE_STAGES):
            h = f"h{i}"
            record_checkpoint(result, stage, f"s{i}", h, prev)
            prev = h

        assert result.is_complete is True

    def test_incomplete_missing_stages(self):
        result = QueryPipelineResult("run-001", "NCT12345678")

        record_checkpoint(result, "pdf_extraction", "s1", "h1", "pdf")
        record_checkpoint(result, "classification", "s2", "h2", "h1")

        assert result.is_complete is False


class TestToDict:
    """Tests for serialization."""

    def test_serializes_correctly(self):
        result = QueryPipelineResult("run-001", "NCT12345678", "pdf_hash")

        record_checkpoint(result, "pdf_extraction", "s1", "h1", "pdf_hash")
        record_checkpoint(result, "classification", "s2", "h2", "h1")

        data = result.to_dict()

        assert data["pipeline_run_id"] == "run-001"
        assert data["registry_id"] == "NCT12345678"
        assert data["protocol_sha256"] == "pdf_hash"
        assert len(data["checkpoints"]) == 2
        assert data["checkpoints"][0]["stage"] == "pdf_extraction"
        assert data["checkpoints"][0]["artifact_sha256"] == "h1"
        assert data["stages_completed"] == ["pdf_extraction", "classification"]
        assert data["is_complete"] is False

    def test_empty_serialization(self):
        result = QueryPipelineResult("run-001", "NCT12345678")
        data = result.to_dict()

        assert data["checkpoints"] == []
        assert data["stages_completed"] == []
        assert data["is_complete"] is False


class TestComputeSha256:
    """Tests for compute_sha256 utility."""

    def test_known_hash(self):
        h = compute_sha256(b"hello")
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_empty_bytes(self):
        h = compute_sha256(b"")
        assert len(h) == 64

    def test_deterministic(self):
        assert compute_sha256(b"test") == compute_sha256(b"test")
