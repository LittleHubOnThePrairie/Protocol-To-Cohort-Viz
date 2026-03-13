"""Integration tests for PipelineOrchestrator — PTCV-24/60.

Tests verify end-to-end pipeline execution, artifact production,
and ALCOA++ lineage chain integrity across all seven stages.

All tests use FilesystemAdapter (tmp_gateway), a MockLlmRetemplater
(no Anthropic API calls), and CTR-XML input to avoid PDF library
dependencies and real network access.
"""

from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.pipeline import (
    PIPELINE_STAGES,
    PipelineOrchestrator,
    PipelineResult,
)
from ptcv.pipeline.models import LineageChainVerification

from tests.pipeline.conftest import SYNTHETIC_CTR_XML


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def run_pipeline(
    orchestrator: PipelineOrchestrator,
    ctr_xml_bytes: bytes,
    registry_id: str,
    amendment_number: str = "00",
    pipeline_run_id: str | None = None,
) -> PipelineResult:
    """Execute the full pipeline and return the result."""
    sha = hashlib.sha256(ctr_xml_bytes).hexdigest()
    return orchestrator.run(
        protocol_data=ctr_xml_bytes,
        registry_id=registry_id,
        amendment_number=amendment_number,
        source_sha256=sha,
        filename="protocol.xml",
        source="ClinicalTrials.gov",
        pipeline_run_id=pipeline_run_id,
    )


# -----------------------------------------------------------------------
# Scenario: Complete pipeline run
# -----------------------------------------------------------------------


class TestCompletePipelineRun:
    """GHERKIN: Complete pipeline run for EU-CTR PDF protocol."""

    def test_pipeline_returns_result(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result is not None
        assert isinstance(result, PipelineResult)

    def test_pipeline_run_id_is_uuid4(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        parsed = uuid.UUID(result.pipeline_run_id)
        assert str(parsed) == result.pipeline_run_id

    def test_all_stages_complete(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.all_stages_complete, (
            "Not all stages completed — check extraction, soa_extraction, "
            "retemplating, coverage_review, sdtm_generation, and validation results"
        )

    def test_extraction_result_present(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.extraction_result is not None
        assert result.extraction_result.registry_id == registry_id

    def test_retemplating_result_present(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.retemplating_result is not None
        assert result.retemplating_result.section_count >= 1

    def test_coverage_result_present(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.coverage_result is not None
        assert 0.0 <= result.coverage_result.coverage_score <= 1.0

    def test_soa_result_present(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.soa_result is not None
        assert result.soa_result.registry_id == registry_id

    def test_sdtm_result_present(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.sdtm_result is not None
        assert result.sdtm_result.registry_id == registry_id

    def test_validation_result_present(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.validation_result is not None
        assert result.validation_result.registry_id == registry_id

    def test_empty_protocol_raises(self, orchestrator, registry_id):
        with pytest.raises(ValueError):
            orchestrator.run(protocol_data=b"", registry_id=registry_id)

    def test_rerun_creates_new_pipeline_run_id(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """ALCOA++ Original: re-run must produce a different pipeline_run_id."""
        r1 = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        r2 = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert r1.pipeline_run_id != r2.pipeline_run_id

    def test_deprecated_parse_result_is_none(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """parse_result is deprecated — should be None in new pipeline."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.parse_result is None


# -----------------------------------------------------------------------
# Scenario: StorageGateway initialises and produces artifacts
# -----------------------------------------------------------------------


class TestStorageGatewayArtifacts:
    """GHERKIN: StorageGateway initialises on first run."""

    def test_extraction_artifacts_written(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        # Text artifact must be readable
        data = tmp_gateway.get_artifact(result.extraction_result.text_artifact_key)
        assert data[:4] == b"PAR1"  # Parquet magic bytes

    def test_retemplating_artifact_written(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        data = tmp_gateway.get_artifact(result.retemplating_result.artifact_key)
        assert data[:4] == b"PAR1"

    def test_sdtm_ts_xpt_artifact_written(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        ts_key = result.sdtm_result.artifact_keys.get("ts")
        assert ts_key, "TS XPT artifact key missing"
        data = tmp_gateway.get_artifact(ts_key)
        assert len(data) > 0

    def test_define_xml_artifact_written(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        define_key = result.sdtm_result.artifact_keys.get("define")
        assert define_key, "define.xml artifact key missing"
        data = tmp_gateway.get_artifact(define_key)
        assert b"<ODM" in data or b"<?xml" in data

    def test_validation_reports_written(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        p21_key = result.validation_result.artifact_keys.get("p21")
        assert p21_key, "p21_report.json key missing"
        data = tmp_gateway.get_artifact(p21_key)
        import json
        report = json.loads(data)
        assert report["report_type"] == "p21_validation"


# -----------------------------------------------------------------------
# Scenario: Unbroken ALCOA++ lineage chain
# -----------------------------------------------------------------------


class TestLineageChain:
    """GHERKIN: Unbroken ALCOA++ lineage chain from download to validation report."""

    def test_eight_stage_checkpoints_recorded(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """Exactly 8 stage checkpoints (PTCV-60/161 pipeline)."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert len(result.stage_checkpoints) == 8

    def test_all_stage_names_present(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """All 7 expected stage names appear in checkpoints."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.stage_names_present == set(PIPELINE_STAGES)

    def test_lineage_chain_is_valid(self, orchestrator, ctr_xml_bytes, registry_id):
        """verify_lineage_chain() returns is_valid=True."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        verification = result.verify_lineage_chain()
        assert verification.is_valid, (
            f"Lineage chain broken at stage {verification.broken_at_stage}.\n"
            + "\n".join(verification.details)
        )

    def test_lineage_chain_all_stages_verified(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        verification = result.verify_lineage_chain()
        assert verification.stages_verified == 8

    def test_get_lineage_returns_eight_records(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        """get_lineage(pipeline_run_id) returns at least 8 records."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        lineage = tmp_gateway.get_lineage(result.pipeline_run_id)
        # One checkpoint artifact per stage was written under pipeline_run_id
        assert len(lineage) >= 8

    def test_lineage_records_have_expected_stages(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        lineage = tmp_gateway.get_lineage(result.pipeline_run_id)
        stage_names = {rec.stage for rec in lineage}
        for expected_stage in PIPELINE_STAGES:
            assert expected_stage in stage_names, (
                f"Stage {expected_stage!r} missing from lineage records"
            )

    def test_each_record_has_required_lineage_fields(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        """Each LineageRecord has run_id, stage, artifact_key, sha256, source_hash."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        lineage = tmp_gateway.get_lineage(result.pipeline_run_id)
        for rec in lineage:
            assert rec.run_id == result.pipeline_run_id
            assert rec.stage
            assert rec.artifact_key
            assert rec.sha256
            assert rec.user == "ptcv-pipeline-orchestrator"

    def test_protocol_sha256_matches_input(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """protocol_sha256 in PipelineResult matches sha256 of input bytes."""
        expected_sha = hashlib.sha256(ctr_xml_bytes).hexdigest()
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id, "00")
        assert result.protocol_sha256 == expected_sha


# -----------------------------------------------------------------------
# Scenario: Migration smoke test (FilesystemAdapter)
# -----------------------------------------------------------------------


class TestMigrationSmokeTest:
    """GHERKIN: Migration smoke test passes on LocalStorageAdapter (using FilesystemAdapter)."""

    def test_smoke_test_all_stages_pass(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """2-page synthetic fixture runs all 7 stages without error."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.all_stages_complete

    def test_smoke_test_lineage_unbroken(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        verification = result.verify_lineage_chain()
        assert verification.is_valid

    def test_smoke_test_explicit_pipeline_run_id(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """Explicit pipeline_run_id is preserved in PipelineResult."""
        explicit_id = str(uuid.uuid4())
        result = run_pipeline(
            orchestrator, ctr_xml_bytes, registry_id, pipeline_run_id=explicit_id
        )
        assert result.pipeline_run_id == explicit_id

    def test_smoke_test_sdtm_domains_non_empty(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.sdtm_result.domain_row_counts.get("TS", 0) >= 1

    def test_smoke_test_validation_report_parseable(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        import json
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        summary_key = result.validation_result.artifact_keys.get("summary")
        assert summary_key
        data = tmp_gateway.get_artifact(summary_key)
        report = json.loads(data)
        assert report["report_type"] == "compliance_summary"
        assert "total_issues" in report


# -----------------------------------------------------------------------
# Scenario: Amendment re-run with delta detection (basic)
# -----------------------------------------------------------------------


class TestAmendmentRerun:
    """GHERKIN: Amendment re-run creates new pipeline_run_id."""

    def test_amendment_rerun_creates_new_run_id(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """ALCOA++ Original: re-run with same input must produce new pipeline_run_id."""
        r1 = run_pipeline(orchestrator, ctr_xml_bytes, registry_id, "00")
        r2 = run_pipeline(orchestrator, ctr_xml_bytes, registry_id, "01")
        assert r1.pipeline_run_id != r2.pipeline_run_id

    def test_amendment_rerun_different_sha_has_different_extraction(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """Different protocol bytes (different amendment) produce different extraction sha256."""
        amended_bytes = ctr_xml_bytes + b"\n<!-- Amendment 01 -->"
        r1 = run_pipeline(orchestrator, ctr_xml_bytes, registry_id, "00")
        r2 = run_pipeline(orchestrator, amended_bytes, registry_id, "01")
        assert r1.protocol_sha256 != r2.protocol_sha256

    def test_amendment_rerun_prior_artifacts_preserved(
        self, orchestrator, ctr_xml_bytes, registry_id, tmp_gateway
    ):
        """Prior run's artifacts remain readable after a re-run (ALCOA+ Original)."""
        r1 = run_pipeline(orchestrator, ctr_xml_bytes, registry_id, "00")
        prior_key = r1.extraction_result.text_artifact_key

        # Run again — prior artifacts must still be readable
        run_pipeline(orchestrator, ctr_xml_bytes, registry_id, "01")
        data = tmp_gateway.get_artifact(prior_key)
        assert len(data) > 0


# -----------------------------------------------------------------------
# Scenario: SoA and SDTM outputs are coherent
# -----------------------------------------------------------------------


class TestPipelineCoherence:
    def test_timepoint_count_non_negative(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.soa_result.timepoint_count >= 0

    def test_activity_count_non_negative(self, orchestrator, ctr_xml_bytes, registry_id):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.soa_result.activity_count >= 0

    def test_sdtm_all_five_domains_written(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        for domain in ("ts", "ta", "te", "tv", "ti"):
            assert domain in result.sdtm_result.artifact_keys, (
                f"SDTM domain {domain.upper()} not written"
            )

    def test_validation_tcg_report_present(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert "tcg" in result.validation_result.artifact_keys

    def test_registry_id_consistent_across_stages(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.extraction_result.registry_id == registry_id
        assert result.retemplating_result.registry_id == registry_id
        assert result.soa_result.registry_id == registry_id
        assert result.sdtm_result.registry_id == registry_id
        assert result.validation_result.registry_id == registry_id


# -----------------------------------------------------------------------
# Scenario: PTCV-60 document-first pipeline specifics
# -----------------------------------------------------------------------


class TestDocumentFirstPipeline:
    """PTCV-60: SoA extraction runs before ICH retemplating."""

    def test_soa_stage_before_retemplating_in_checkpoints(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """SoA extraction checkpoint appears before retemplating."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        stage_names = [cp.stage for cp in result.stage_checkpoints]
        soa_idx = stage_names.index("soa_extraction")
        ret_idx = stage_names.index("retemplating")
        assert soa_idx < ret_idx

    def test_coverage_review_stage_present(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        """Coverage review appears in stage checkpoints."""
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        stage_names = [cp.stage for cp in result.stage_checkpoints]
        assert "coverage_review" in stage_names

    def test_coverage_result_has_score(
        self, orchestrator, ctr_xml_bytes, registry_id
    ):
        result = run_pipeline(orchestrator, ctr_xml_bytes, registry_id)
        assert result.coverage_result is not None
        assert isinstance(result.coverage_result.coverage_score, float)
