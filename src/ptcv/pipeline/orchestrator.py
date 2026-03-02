"""PipelineOrchestrator — end-to-end PTCV pipeline (PTCV-24).

Sequences all six pipeline stages from protocol bytes through validated
SDTM package, using a single shared StorageGateway instance injected into
every stage service.

Pipeline sequence:
  PTCV-18  Protocol download  (input: protocol bytes + sha256)
  PTCV-19  ExtractionService  (PDF/XML → text_blocks + tables Parquet)
  PTCV-20  IchParser          (text → ICH sections Parquet)
  PTCV-21  SoaExtractor       (ICH sections → USDM Parquet)
  PTCV-22  SdtmService        (ICH sections + USDM timepoints → XPT + define.xml)
  PTCV-23  ValidationService  (XPT + define.xml → P21/TCG/compliance reports)

After each stage, the orchestrator writes a lightweight JSON checkpoint
artifact under the orchestrator's pipeline_run_id via StorageGateway, so
that get_lineage(pipeline_run_id) returns exactly one LineageRecord per
stage (six total).

Risk tier: MEDIUM — data pipeline orchestration (no patient data).

Regulatory references:
- ALCOA++ Contemporaneous: single timestamp per pipeline run
- ALCOA++ Traceable: source_hash chain across all six stages
- ALCOA++ Original: each stage service generates its own run_id
- 21 CFR 11.10(e): stage-checkpoint lineage records for audit trail
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..extraction.models import ExtractionResult
from ..extraction.extraction_service import ExtractionService
from ..extraction.parquet_writer import parquet_to_text_blocks
from ..ich_parser.parser import IchParser, ParseResult
from ..ich_parser.parquet_writer import parquet_to_sections
from ..ich_parser.review_queue import ReviewQueue
from ..soa_extractor.extractor import SoaExtractor
from ..soa_extractor.models import ExtractResult
from ..soa_extractor.writer import UsdmParquetWriter
from ..sdtm.sdtm_service import SdtmService
from ..sdtm.models import SdtmGenerationResult
from ..sdtm.review_queue import CtReviewQueue
from ..sdtm.validation.validation_service import ValidationService
from ..sdtm.validation.models import ValidationResult
from ..storage import FilesystemAdapter, StorageGateway
from .models import LineageChainVerification, PipelineResult, StageCheckpoint

logger = logging.getLogger(__name__)

_USER = "ptcv-pipeline-orchestrator"
_DEFAULT_ROOT = Path("C:/Dev/PTCV/data")


class PipelineOrchestrator:
    """End-to-end pipeline from protocol bytes to validated SDTM package.

    One StorageGateway instance is created at construction and shared
    across all six stage services (PTCV-18 design requirement).

    Args:
        gateway: StorageGateway to use for all artifact writes. If None,
            uses FilesystemAdapter with the default PTCV data root.
        review_queue: ReviewQueue for ICH parser low-confidence sections.
            Shared with SoaExtractor. If None, uses default SQLite path.
        ct_review_queue: CtReviewQueue for SDTM unmapped CT terms. If
            None, uses default SQLite path.
    [PTCV-24 Scenario: Complete pipeline run for EU-CTR PDF protocol]
    [PTCV-24 Scenario: StorageGateway initialises on first run]
    """

    def __init__(
        self,
        gateway: Optional[StorageGateway] = None,
        review_queue: Optional[ReviewQueue] = None,
        ct_review_queue: Optional[CtReviewQueue] = None,
    ) -> None:
        if gateway is None:
            gateway = FilesystemAdapter(root=_DEFAULT_ROOT)
            gateway.initialise()

        self._gateway = gateway

        # Shared review queues — instantiated here so stage services can
        # share them (avoids double-initialisation of SQLite tables).
        self._review_queue = review_queue
        self._ct_review_queue = ct_review_queue

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        protocol_data: bytes,
        registry_id: str,
        amendment_number: str = "00",
        source_sha256: str = "",
        filename: str = "",
        source: str = "",
        pipeline_run_id: Optional[str] = None,
    ) -> PipelineResult:
        """Run the full pipeline for one protocol amendment.

        Args:
            protocol_data: Raw bytes of the protocol file (PDF or CTR-XML).
            registry_id: Trial registry identifier (e.g. "NCT00112827").
            amendment_number: Protocol amendment version (e.g. "00").
            source_sha256: SHA-256 of the protocol_data bytes (PTCV-18
                artifact hash). If empty, computed from protocol_data.
            filename: Original filename for format detection hint.
            source: Registry source name for lineage records (e.g.
                "ClinicalTrials.gov").
            pipeline_run_id: Optional explicit UUID4 for the orchestrator
                run. Defaults to a fresh UUID4. Intended for testing only.

        Returns:
            PipelineResult with all stage results and lineage chain.

        Raises:
            ValueError: If protocol_data is empty.
        [PTCV-24 Scenario: Complete pipeline run for EU-CTR PDF protocol]
        [PTCV-24 Scenario: Unbroken ALCOA++ lineage chain]
        [PTCV-24 Scenario: Migration smoke test passes on LocalStorageAdapter]
        """
        if not protocol_data:
            raise ValueError("protocol_data must not be empty")

        if not pipeline_run_id:
            pipeline_run_id = str(uuid.uuid4())

        timestamp = datetime.now(timezone.utc).isoformat()

        # Compute source_sha256 from bytes if not supplied (covers the
        # case where the orchestrator drives directly from a file path
        # without going through PTCV-18 download service).
        if not source_sha256:
            import hashlib
            source_sha256 = hashlib.sha256(protocol_data).hexdigest()

        logger.info(
            "Pipeline started: pipeline_run_id=%s registry_id=%s",
            pipeline_run_id,
            registry_id,
        )

        checkpoints: list[StageCheckpoint] = []

        # ----------------------------------------------------------------
        # Stage 0: Record download checkpoint (PTCV-18)
        # The protocol bytes were already downloaded; we just record the
        # pipeline-level lineage entry for the download artifact.
        # ----------------------------------------------------------------
        download_cp = self._write_checkpoint(
            pipeline_run_id=pipeline_run_id,
            stage="download",
            stage_run_id=pipeline_run_id,
            artifact_key=f"pipeline/{pipeline_run_id}/stage-01-download.json",
            payload={
                "stage": "download",
                "registry_id": registry_id,
                "amendment_number": amendment_number,
                "protocol_sha256": source_sha256,
                "source": source or "direct",
                "filename": filename,
            },
            artifact_sha256=source_sha256,
            source_sha256="",
        )
        checkpoints.append(download_cp)

        # ----------------------------------------------------------------
        # Stage 1: Extraction (PTCV-19)
        # ----------------------------------------------------------------
        logger.info("Stage extraction: running ExtractionService")
        extraction_svc = ExtractionService(gateway=self._gateway)
        extraction_result: ExtractionResult = extraction_svc.extract(
            protocol_data=protocol_data,
            registry_id=registry_id,
            amendment_number=amendment_number,
            source_sha256=source_sha256,
            filename=filename,
            source=source,
        )

        extraction_cp = self._write_checkpoint(
            pipeline_run_id=pipeline_run_id,
            stage="extraction",
            stage_run_id=extraction_result.run_id,
            artifact_key=f"pipeline/{pipeline_run_id}/stage-02-extraction.json",
            payload={
                "stage": "extraction",
                "stage_run_id": extraction_result.run_id,
                "text_artifact_key": extraction_result.text_artifact_key,
                "text_artifact_sha256": extraction_result.text_artifact_sha256,
                "format_detected": extraction_result.format_detected,
                "table_count": extraction_result.table_count,
                "text_block_count": extraction_result.text_block_count,
            },
            artifact_sha256=extraction_result.text_artifact_sha256,
            source_sha256=source_sha256,
        )
        checkpoints.append(extraction_cp)

        # ----------------------------------------------------------------
        # Stage 2: ICH Parsing (PTCV-20)
        # Reconstruct protocol text from text_blocks.parquet for IchParser.
        # ----------------------------------------------------------------
        logger.info("Stage ich_parse: reading text_blocks.parquet")
        text_bytes = self._gateway.get_artifact(extraction_result.text_artifact_key)
        text_blocks = parquet_to_text_blocks(text_bytes)
        # Join all text blocks ordered by page then block_index
        text_blocks_sorted = sorted(
            text_blocks, key=lambda b: (b.page_number, b.block_index)
        )
        protocol_text = "\n".join(b.text for b in text_blocks_sorted if b.text.strip())

        if not protocol_text.strip():
            # Fallback: use a minimal placeholder so IchParser produces a
            # legacy section rather than raising ValueError.
            protocol_text = f"Protocol {registry_id} (text extraction produced no content)"

        logger.info("Stage ich_parse: running IchParser")
        ich_parser = IchParser(
            gateway=self._gateway,
            review_queue=self._review_queue,
        )
        parse_result: ParseResult = ich_parser.parse(
            text=protocol_text,
            registry_id=registry_id,
            source_run_id=extraction_result.run_id,
            source_sha256=extraction_result.text_artifact_sha256,
        )

        ich_cp = self._write_checkpoint(
            pipeline_run_id=pipeline_run_id,
            stage="ich_parse",
            stage_run_id=parse_result.run_id,
            artifact_key=f"pipeline/{pipeline_run_id}/stage-03-ich-parse.json",
            payload={
                "stage": "ich_parse",
                "stage_run_id": parse_result.run_id,
                "artifact_key": parse_result.artifact_key,
                "artifact_sha256": parse_result.artifact_sha256,
                "section_count": parse_result.section_count,
                "review_count": parse_result.review_count,
                "format_verdict": parse_result.format_verdict,
                "format_confidence": parse_result.format_confidence,
                "missing_required_sections": parse_result.missing_required_sections,
            },
            artifact_sha256=parse_result.artifact_sha256,
            source_sha256=extraction_result.text_artifact_sha256,
        )
        checkpoints.append(ich_cp)

        # ----------------------------------------------------------------
        # Stage 3: SoA Extraction / USDM mapping (PTCV-21)
        # Read ICH sections back from storage, pass to SoaExtractor.
        # ----------------------------------------------------------------
        logger.info("Stage soa_extraction: reading sections.parquet")
        sections_bytes = self._gateway.get_artifact(parse_result.artifact_key)
        sections = parquet_to_sections(sections_bytes)

        logger.info("Stage soa_extraction: running SoaExtractor")
        soa_extractor = SoaExtractor(
            gateway=self._gateway,
            review_queue=self._review_queue,
        )
        soa_result: ExtractResult = soa_extractor.extract(
            sections=sections,
            registry_id=registry_id,
            source_run_id=parse_result.run_id,
            source_sha256=parse_result.artifact_sha256,
        )

        # Primary artifact for lineage: timepoints.parquet (if available)
        soa_primary_key = soa_result.artifact_keys.get(
            "timepoints",
            next(iter(soa_result.artifact_keys.values()), ""),
        )
        # sha256 of the primary artifact — retrieve from gateway lineage
        soa_primary_sha256 = self._resolve_artifact_sha256(soa_primary_key)

        soa_cp = self._write_checkpoint(
            pipeline_run_id=pipeline_run_id,
            stage="soa_extraction",
            stage_run_id=soa_result.run_id,
            artifact_key=f"pipeline/{pipeline_run_id}/stage-04-soa-extraction.json",
            payload={
                "stage": "soa_extraction",
                "stage_run_id": soa_result.run_id,
                "timepoint_count": soa_result.timepoint_count,
                "activity_count": soa_result.activity_count,
                "primary_artifact_key": soa_primary_key,
                "primary_artifact_sha256": soa_primary_sha256,
            },
            artifact_sha256=soa_primary_sha256 or parse_result.artifact_sha256,
            source_sha256=parse_result.artifact_sha256,
        )
        checkpoints.append(soa_cp)

        # ----------------------------------------------------------------
        # Stage 4: SDTM Generation (PTCV-22)
        # Deserialise USDM timepoints for TV domain generation.
        # ----------------------------------------------------------------
        logger.info("Stage sdtm_generation: reading timepoints.parquet")
        timepoints = []
        timepoints_key = soa_result.artifact_keys.get("timepoints")
        if timepoints_key:
            tp_bytes = self._gateway.get_artifact(timepoints_key)
            writer = UsdmParquetWriter()
            timepoints = writer.parquet_to_timepoints(tp_bytes)

        logger.info("Stage sdtm_generation: running SdtmService")
        sdtm_svc = SdtmService(
            gateway=self._gateway,
            review_queue=self._ct_review_queue,
        )
        sdtm_result: SdtmGenerationResult = sdtm_svc.generate(
            sections=sections,
            timepoints=timepoints,
            registry_id=registry_id,
            amendment_number=amendment_number,
            source_sha256=soa_cp.artifact_sha256,
        )

        sdtm_ts_sha256 = sdtm_result.artifact_sha256s.get(
            "ts", sdtm_result.source_sha256
        )

        sdtm_cp = self._write_checkpoint(
            pipeline_run_id=pipeline_run_id,
            stage="sdtm_generation",
            stage_run_id=sdtm_result.run_id,
            artifact_key=f"pipeline/{pipeline_run_id}/stage-05-sdtm-generation.json",
            payload={
                "stage": "sdtm_generation",
                "stage_run_id": sdtm_result.run_id,
                "ts_sha256": sdtm_ts_sha256,
                "domain_row_counts": sdtm_result.domain_row_counts,
                "ct_unmapped_count": sdtm_result.ct_unmapped_count,
            },
            artifact_sha256=sdtm_ts_sha256,
            source_sha256=soa_cp.artifact_sha256,
        )
        checkpoints.append(sdtm_cp)

        # ----------------------------------------------------------------
        # Stage 5: Validation (PTCV-23)
        # ----------------------------------------------------------------
        logger.info("Stage validation: running ValidationService")
        val_svc = ValidationService(gateway=self._gateway)
        validation_result: ValidationResult = val_svc.validate(
            sdtm_result=sdtm_result,
            format_verdict=parse_result.format_verdict,
            format_confidence=parse_result.format_confidence,
            sections_detected=parse_result.section_count,
            missing_required_sections=parse_result.missing_required_sections,
        )

        val_primary_sha256 = validation_result.artifact_sha256s.get(
            "p21",
            validation_result.sdtm_sha256,
        )

        val_cp = self._write_checkpoint(
            pipeline_run_id=pipeline_run_id,
            stage="validation",
            stage_run_id=validation_result.run_id,
            artifact_key=f"pipeline/{pipeline_run_id}/stage-06-validation.json",
            payload={
                "stage": "validation",
                "stage_run_id": validation_result.run_id,
                "p21_error_count": validation_result.p21_error_count,
                "p21_warning_count": validation_result.p21_warning_count,
                "tcg_passed": validation_result.tcg_passed,
                "p21_report_sha256": val_primary_sha256,
            },
            artifact_sha256=val_primary_sha256,
            source_sha256=sdtm_ts_sha256,
        )
        checkpoints.append(val_cp)

        logger.info(
            "Pipeline complete: pipeline_run_id=%s registry_id=%s stages=%d",
            pipeline_run_id,
            registry_id,
            len(checkpoints),
        )

        return PipelineResult(
            pipeline_run_id=pipeline_run_id,
            registry_id=registry_id,
            amendment_number=amendment_number,
            extraction_result=extraction_result,
            parse_result=parse_result,
            soa_result=soa_result,
            sdtm_result=sdtm_result,
            validation_result=validation_result,
            protocol_sha256=source_sha256,
            stage_checkpoints=checkpoints,
            pipeline_timestamp_utc=timestamp,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_checkpoint(
        self,
        pipeline_run_id: str,
        stage: str,
        stage_run_id: str,
        artifact_key: str,
        payload: dict,
        artifact_sha256: str,
        source_sha256: str,
    ) -> StageCheckpoint:
        """Write a pipeline-level stage checkpoint artifact to storage.

        Uses pipeline_run_id as the run_id so all six checkpoint records
        appear in get_lineage(pipeline_run_id).

        Args:
            pipeline_run_id: Orchestrator's shared run identifier.
            stage: Stage name (one of PIPELINE_STAGES).
            stage_run_id: The stage service's own run_id.
            artifact_key: Storage key for the checkpoint JSON artifact.
            payload: JSON-serialisable content of the checkpoint.
            artifact_sha256: sha256 of the stage's primary artifact.
            source_sha256: sha256 of the prior stage's primary artifact.

        Returns:
            StageCheckpoint with resolved sha256 values.
        """
        import hashlib

        checkpoint_data = json.dumps(
            {
                "pipeline_run_id": pipeline_run_id,
                "stage_run_id": stage_run_id,
                **payload,
            },
            indent=2,
        ).encode("utf-8")

        self._gateway.put_artifact(
            key=artifact_key,
            data=checkpoint_data,
            content_type="application/json",
            run_id=pipeline_run_id,
            source_hash=source_sha256,
            user=_USER,
            immutable=False,
            stage=stage,
        )

        return StageCheckpoint(
            stage=stage,
            stage_run_id=stage_run_id,
            artifact_key=artifact_key,
            artifact_sha256=artifact_sha256,
            source_sha256=source_sha256,
        )

    def _resolve_artifact_sha256(self, artifact_key: str) -> str:
        """Retrieve sha256 of an artifact written by a stage service.

        Reads the artifact bytes from storage and computes sha256. This
        is used for artifacts whose sha256 is not surfaced in the stage
        service's result object.

        Args:
            artifact_key: Storage key of the artifact.

        Returns:
            Hex-encoded sha256 string, or empty string if key is empty
            or the artifact cannot be read.
        """
        import hashlib

        if not artifact_key:
            return ""
        try:
            data = self._gateway.get_artifact(artifact_key)
            return hashlib.sha256(data).hexdigest()
        except Exception:
            return ""
