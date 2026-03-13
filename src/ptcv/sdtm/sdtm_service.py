"""SdtmService — CDISC SDTM Trial Design domain generation orchestrator.

Orchestrates the full SDTM generation pipeline for one protocol amendment:
  1. Generate TS, TA, TE, TV, TI DataFrames from ICH sections + USDM timepoints
  2. Serialize each domain to SAS XPT bytes via pyreadstat
  3. Write each XPT artifact to StorageGateway (immutable=True — WORM lock)
  4. Verify SHA-256 of each written artifact matches the returned ArtifactRecord
  5. Generate Define-XML v2.1 referencing all 5 XPT datasets
  6. Write Define-XML to StorageGateway (immutable=True)
  7. Route CT-unmapped terms to the CT review queue (SQLite)
  8. Return SdtmGenerationResult with artifact keys, sha256s, and counts

Storage layout (under gateway root):
  sdtm/{registry_id}/{run_id}/ts.xpt
  sdtm/{registry_id}/{run_id}/ta.xpt
  sdtm/{registry_id}/{run_id}/te.xpt
  sdtm/{registry_id}/{run_id}/tv.xpt
  sdtm/{registry_id}/{run_id}/ti.xpt
  sdtm/{registry_id}/{run_id}/define.xml

Risk tier: MEDIUM — produces regulatory submission artefacts.

Regulatory references:
- ALCOA+ Original: new run_id per amendment; prior XPT WORM-locked
- ALCOA+ Traceable: source_hash links XPT to upstream timepoints.parquet
- ALCOA+ Contemporaneous: generation_timestamp_utc at write boundary
- 21 CFR 11.10(e): audit trail via StorageGateway LineageRecord chain
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from ..storage import FilesystemAdapter, StorageGateway
from .ct_normalizer import CtNormalizer
from .define_xml import DefineXmlGenerator
from .domain_generators import TaGenerator, TeGenerator, TiGenerator, TsGenerator, TvGenerator
from .models import CtReviewQueueEntry, SdtmGenerationResult
from .review_queue import CtReviewQueue

if TYPE_CHECKING:
    from ..ich_parser.models import IchSection
    from ..ich_parser.template_assembler import AssembledProtocol, QueryExtractionHit
    from ..soa_extractor.models import UsdmTimepoint


logger = logging.getLogger(__name__)

_USER = "ptcv-sdtm-generator"
_DEFAULT_REVIEW_DB = Path("C:/Dev/PTCV/data/sqlite/ct_review_queue.db")


def _df_to_xpt_bytes(df: pd.DataFrame, file_label: str = "") -> bytes:
    """Serialize a DataFrame to SAS XPT v5 bytes via pyreadstat.

    Uses a temporary file as pyreadstat requires a file path, not a buffer.

    Args:
        df: DataFrame to serialize.
        file_label: SAS dataset label (max 40 chars).

    Returns:
        XPT file bytes.
    """
    import pyreadstat  # type: ignore[import-untyped]

    fd, tmp_path = tempfile.mkstemp(suffix=".xpt")
    os.close(fd)
    try:
        pyreadstat.write_xport(
            df,
            tmp_path,
            file_label=file_label[:40],
        )
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class SdtmService:
    """Orchestrates SDTM trial design domain generation for one protocol.

    Args:
        gateway: StorageGateway to write artifacts to. Uses FilesystemAdapter
            with default PTCV data root if None.
        review_queue: CtReviewQueue for unmapped CT terms. Uses default
            SQLite path if None.
    """

    def __init__(
        self,
        gateway: Optional[StorageGateway] = None,
        review_queue: Optional[CtReviewQueue] = None,
    ) -> None:
        if gateway is None:
            gateway = FilesystemAdapter(root=Path("C:/Dev/PTCV/data"))
        if review_queue is None:
            review_queue = CtReviewQueue(db_path=_DEFAULT_REVIEW_DB)

        self._gateway = gateway
        self._review_queue = review_queue
        self._ct = CtNormalizer()

        self._ts_gen = TsGenerator(ct_normalizer=self._ct)
        self._ta_gen = TaGenerator()
        self._te_gen = TeGenerator()
        self._tv_gen = TvGenerator()
        self._ti_gen = TiGenerator()
        self._define_gen = DefineXmlGenerator()

        self._gateway.initialise()
        self._review_queue.initialise()

    def generate(
        self,
        sections: list["IchSection"],
        timepoints: list["UsdmTimepoint"],
        registry_id: str,
        amendment_number: str = "00",
        source_sha256: str = "",
        run_id: Optional[str] = None,
    ) -> SdtmGenerationResult:
        """Generate all SDTM trial design domains and write to storage.

        Creates new run_id so prior WORM artifacts are never touched
        (ALCOA+ Original). SHA-256 is verified after every write.

        Args:
            sections: IchSection records from PTCV-20 ICH parser.
            timepoints: UsdmTimepoint records from PTCV-21 SoA extractor.
            registry_id: Trial registry identifier (e.g. "NCT00112827").
            amendment_number: Protocol amendment version (e.g. "00").
            source_sha256: SHA-256 of the upstream timepoints.parquet.
                Pass "" when testing without an upstream artifact.
            run_id: Optional explicit run_id for deterministic testing.

        Returns:
            SdtmGenerationResult with artifact keys, sha256s, and counts.

        Raises:
            ValueError: If both sections and timepoints are empty.
        [PTCV-22 Scenario: Generate TS domain and write to WORM with hash]
        [PTCV-22 Scenario: WORM lock confirmed for all submission artifacts]
        [PTCV-22 Scenario: Generate TV domain from USDM Timepoints]
        [PTCV-22 Scenario: CT normalization writes unmapped terms to queue]
        [PTCV-22 Scenario: Define-XML written to WORM with structural integrity]
        """
        if not sections and not timepoints:
            raise ValueError(
                "At least one of sections or timepoints must be non-empty."
            )

        if run_id is None:
            run_id = str(uuid.uuid4())

        studyid = registry_id[:20]

        # --- Generate DataFrames ----------------------------------------
        ts_df, unmapped_results = self._ts_gen.generate(
            sections, studyid, run_id,
            timepoints=timepoints,
            registry_id=registry_id,
        )
        ta_df = self._ta_gen.generate(sections, studyid)
        te_df = self._te_gen.generate(sections, studyid)
        tv_df = self._tv_gen.generate(timepoints, studyid)
        ti_df = self._ti_gen.generate(sections, studyid)

        datasets: dict[str, pd.DataFrame] = {
            "ts": ts_df,
            "ta": ta_df,
            "te": te_df,
            "tv": tv_df,
            "ti": ti_df,
        }

        logger.info(
            "SDTM generation: registry_id=%s run_id=%s "
            "ts=%d ta=%d te=%d tv=%d ti=%d",
            registry_id,
            run_id,
            len(ts_df),
            len(ta_df),
            len(te_df),
            len(tv_df),
            len(ti_df),
        )

        return self._write_and_finalize(
            datasets=datasets,
            unmapped_results=unmapped_results,
            registry_id=registry_id,
            run_id=run_id,
            source_sha256=source_sha256,
            amendment_number=amendment_number,
            source_type="ich_section",
        )

    def generate_from_assembled(
        self,
        assembled: "AssembledProtocol",
        timepoints: list["UsdmTimepoint"],
        registry_id: str,
        amendment_number: str = "00",
        source_sha256: str = "",
        run_id: Optional[str] = None,
    ) -> SdtmGenerationResult:
        """Generate SDTM domains from query pipeline AssembledProtocol.

        Uses typed QueryExtractionHit records for direct field mapping,
        avoiding lossy IchSection bridge conversion (PTCV-140).

        Args:
            assembled: Completed AssembledProtocol from the query pipeline.
            timepoints: UsdmTimepoint records from PTCV-21 SoA extractor.
            registry_id: Trial registry identifier.
            amendment_number: Protocol amendment version.
            source_sha256: SHA-256 of the upstream artifact.
            run_id: Optional explicit run_id for deterministic testing.

        Returns:
            SdtmGenerationResult with source_type="query_pipeline".
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        studyid = registry_id[:20]

        # Collect all hits from assembled sections.
        all_hits: list["QueryExtractionHit"] = []
        for section in assembled.sections:
            all_hits.extend(section.hits)

        # --- Generate DataFrames ----------------------------------------
        ts_df, unmapped_results = self._ts_gen.generate_from_hits(
            all_hits, studyid, run_id,
            timepoints=timepoints,
            registry_id=registry_id,
        )
        ta_df = self._ta_gen.generate_from_hits(all_hits, studyid)
        te_df = self._te_gen.generate_from_hits(all_hits, studyid)
        tv_df = self._tv_gen.generate(timepoints, studyid)
        ti_df = self._ti_gen.generate_from_hits(all_hits, studyid)

        datasets: dict[str, pd.DataFrame] = {
            "ts": ts_df,
            "ta": ta_df,
            "te": te_df,
            "tv": tv_df,
            "ti": ti_df,
        }

        logger.info(
            "SDTM generation (query_pipeline): registry_id=%s run_id=%s "
            "ts=%d ta=%d te=%d tv=%d ti=%d",
            registry_id,
            run_id,
            len(ts_df),
            len(ta_df),
            len(te_df),
            len(tv_df),
            len(ti_df),
        )

        return self._write_and_finalize(
            datasets=datasets,
            unmapped_results=unmapped_results,
            registry_id=registry_id,
            run_id=run_id,
            source_sha256=source_sha256,
            amendment_number=amendment_number,
            source_type="query_pipeline",
        )

    # -------------------------------------------------------------------
    # Shared write + finalize tail (PTCV-140 refactor)
    # -------------------------------------------------------------------

    def _write_and_finalize(
        self,
        datasets: dict[str, pd.DataFrame],
        unmapped_results: list,
        registry_id: str,
        run_id: str,
        source_sha256: str,
        amendment_number: str,
        source_type: str = "ich_section",
    ) -> SdtmGenerationResult:
        """Write XPT artifacts, Define-XML, and route CT-unmapped terms.

        Shared tail for both ``generate()`` and ``generate_from_assembled()``.
        """
        studyid = registry_id[:20]
        prefix = f"sdtm/{registry_id}/{run_id}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # --- Route CT-unmapped terms to review queue --------------------
        ct_unmapped_count = 0
        for ct_result in unmapped_results:
            self._review_queue.enqueue(
                CtReviewQueueEntry(
                    run_id=run_id,
                    registry_id=registry_id,
                    domain="TS",
                    variable="TSVAL",
                    original_value=ct_result.original_value,
                    ct_lookup_attempted=True,
                    queue_timestamp_utc=timestamp,
                )
            )
            ct_unmapped_count += 1

        # --- Write XPT artifacts ----------------------------------------
        artifact_keys: dict[str, str] = {}
        artifact_sha256s: dict[str, str] = {}
        domain_row_counts: dict[str, int] = {}
        xpt_sha256s: dict[str, str] = {}

        for domain, df in datasets.items():
            domain_upper = domain.upper()
            xpt_bytes = _df_to_xpt_bytes(df, file_label=domain_upper)
            key = f"{prefix}/{domain}.xpt"

            # Pre-compute sha256 for post-write integrity check
            expected_sha256 = hashlib.sha256(xpt_bytes).hexdigest()

            artifact = self._gateway.put_artifact(
                key=key,
                data=xpt_bytes,
                content_type="application/octet-stream",
                run_id=run_id,
                source_hash=source_sha256,
                user=_USER,
                immutable=True,
                stage="sdtm_generation",
                registry_id=registry_id,
                amendment_number=amendment_number,
            )

            # Post-write SHA-256 integrity verification (GHERKIN Scenario 1)
            assert artifact.sha256 == expected_sha256, (
                f"Write integrity check failed for {key}: "
                f"expected {expected_sha256} got {artifact.sha256}"
            )

            artifact_keys[domain] = key
            artifact_sha256s[domain] = artifact.sha256
            domain_row_counts[domain_upper] = len(df)
            xpt_sha256s[domain_upper] = artifact.sha256

        # --- Generate and write Define-XML ------------------------------
        define_bytes = self._define_gen.generate(
            datasets={k.upper(): v for k, v in datasets.items()},
            studyid=studyid,
            source_xpt_sha256s=xpt_sha256s,
        )
        define_key = f"{prefix}/define.xml"
        expected_define_sha256 = hashlib.sha256(define_bytes).hexdigest()

        # Use ts sha256 as the primary source_hash for define.xml lineage
        define_source_hash = artifact_sha256s.get("ts", source_sha256)

        define_artifact = self._gateway.put_artifact(
            key=define_key,
            data=define_bytes,
            content_type="application/xml",
            run_id=run_id,
            source_hash=define_source_hash,
            user=_USER,
            immutable=True,
            stage="sdtm_generation",
            registry_id=registry_id,
            amendment_number=amendment_number,
        )

        # Post-write SHA-256 integrity check
        assert define_artifact.sha256 == expected_define_sha256, (
            f"Write integrity check failed for {define_key}"
        )

        artifact_keys["define"] = define_key
        artifact_sha256s["define"] = define_artifact.sha256

        logger.info(
            "SDTM generation complete: registry_id=%s run_id=%s ct_unmapped=%d",
            registry_id,
            run_id,
            ct_unmapped_count,
        )

        return SdtmGenerationResult(
            run_id=run_id,
            registry_id=registry_id,
            artifact_keys=artifact_keys,
            artifact_sha256s=artifact_sha256s,
            source_sha256=source_sha256,
            domain_row_counts=domain_row_counts,
            ct_unmapped_count=ct_unmapped_count,
            generation_timestamp_utc=timestamp,
            source_type=source_type,
        )
