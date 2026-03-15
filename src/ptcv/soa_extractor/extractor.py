"""SoaExtractor — main Schedule of Activities extraction service.

Orchestrates the full SoA extraction pipeline for one protocol:
  1. Find SoA tables via document-first search (PTCV-60):
     a. Pre-extracted tables from tables.parquet (PTCV-51 bridge)
     b. PDF table discovery via Camelot/TATR (PTCV-38)
     c. Text parsing from ICH sections (fallback when sections provided)
  2. Map tables to USDM entities via UsdmMapper
  3. Stamp extraction_timestamp_utc on every entity
  4. Serialise each entity type to Parquet via UsdmParquetWriter
  5. Write five Parquet artifacts to StorageGateway with stage="soa_extraction"
  6. Enqueue uncertain synonym mappings (confidence < 0.80) to ReviewQueue
  7. Return ExtractResult with run_id and artifact metadata

Risk tier: MEDIUM — data pipeline service (no patient data).

Regulatory references:
- ALCOA+ Accurate: seven required timepoint attributes validated at write
- ALCOA+ Contemporaneous: extraction_timestamp_utc stamped immediately
  before StorageGateway.put_artifact() call
- ALCOA+ Traceable: source_sha256 and stage="soa_extraction" in LineageRecord
- ALCOA+ Original: each run produces a new run_id; prior Parquet preserved
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from ..ich_parser.models import IchSection, ReviewQueueEntry
from ..ich_parser.review_queue import ReviewQueue
from ..storage import FilesystemAdapter, StorageGateway
from .llm_soa_builder import LlmSoaBuilder, MIN_ACTIVITIES_THRESHOLD
from .mapper import UsdmMapper
from .models import ExtractResult, SynonymMapping
from .parser import SoaTableParser
from .resolver import SynonymResolver
from .table_bridge import filter_soa_tables
from .table_discovery import TableDiscovery
from .writer import UsdmParquetWriter


_DEFAULT_REVIEW_DB = Path("C:/Dev/PTCV/data/sqlite/review_queue.db")
_DEFAULT_REGISTRY_CACHE = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials/registry_cache"
)
_USER = "ptcv-soa-extractor"

# Synonym mappings below this threshold are routed to the review queue
_REVIEW_THRESHOLD = 0.80


class SoaExtractor:
    """Schedule of Activities extraction service.

    Extracts SoA tables from protocol documents using a document-first
    approach (PTCV-60): pre-extracted tables and PDF discovery are tried
    before ICH section text parsing. Maps results to CDISC USDM v4.0
    entities and writes five Parquet artifacts under usdm/{run_id}/ via
    StorageGateway.

    Args:
        gateway: StorageGateway instance. Uses FilesystemAdapter with
            default PTCV data root if None.
        review_queue: ReviewQueue for uncertain synonym mappings. Uses
            default SQLite path if None.
    [PTCV-21 Scenario: Extract SoA to USDM Parquet with lineage]
    """

    def __init__(
        self,
        gateway: Optional[StorageGateway] = None,
        review_queue: Optional[ReviewQueue] = None,
    ) -> None:
        if gateway is None:
            gateway = FilesystemAdapter(root=Path("C:/Dev/PTCV/data"))
        if review_queue is None:
            review_queue = ReviewQueue(db_path=_DEFAULT_REVIEW_DB)

        self._gateway = gateway
        self._review_queue = review_queue
        self._parser = SoaTableParser()
        self._discovery = TableDiscovery()
        self._llm_builder = LlmSoaBuilder()
        self._mapper = UsdmMapper(resolver=SynonymResolver())
        self._writer = UsdmParquetWriter()

        self._gateway.initialise()
        self._review_queue.initialise()

    # ------------------------------------------------------------------
    # Registry metadata (PTCV-198)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_registry_metadata(registry_id: str) -> dict[str, Any] | None:
        """Load cached ClinicalTrials.gov metadata for NCT IDs.

        Reads from the same cache directory that
        ``RegistryMetadataFetcher`` (PTCV-194) writes to.  Returns
        None when the registry_id is not an NCT ID or the cache file
        does not exist — in that case Level 3 proceeds without
        registry context.
        """
        if not registry_id.startswith("NCT"):
            return None
        cache_path = _DEFAULT_REGISTRY_CACHE / f"{registry_id}.json"
        if not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.debug(
                "Could not load registry cache for %s", registry_id,
            )
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        registry_id: str,
        source_run_id: str = "",
        source_sha256: str = "",
        who: str = _USER,
        sections: Optional[list[IchSection]] = None,
        pdf_bytes: Optional[bytes] = None,
        text_blocks: Optional[list[dict]] = None,
        page_count: int = 0,
        extracted_tables: Optional[list] = None,
    ) -> ExtractResult:
        """Extract SoA tables and write USDM Parquet artifacts.

        Uses a document-first approach (PTCV-60): pre-extracted tables
        and PDF table discovery are tried before ICH section text
        parsing. ICH sections are optional — the extractor can run
        independently of ICH classification.

        Creates a new run_id for this extraction run. Prior Parquet files
        under different run_ids are never touched (ALCOA+ Original).

        Args:
            registry_id: Trial identifier (EUCT-Code or NCT-ID).
            source_run_id: run_id from upstream stage.
            source_sha256: SHA-256 of the upstream artifact.
                Used as source_hash in lineage records.
            who: Actor identifier for audit trail.
            sections: Optional IchSection list from ICH parser.
                When provided, text-based SoA parsing is attempted
                as a final fallback after table bridge and PDF
                discovery.
            pdf_bytes: Optional raw PDF bytes for table discovery.
                Enables the blended Camelot Stream + Table Transformer
                discovery (PTCV-38).
            text_blocks: Optional list of dicts with 'page_number' and
                'text' keys, used by table discovery to locate candidate
                pages. Required when pdf_bytes is provided.
            page_count: Total page count for the PDF. Required when
                pdf_bytes is provided.
            extracted_tables: Optional list of ExtractedTable objects from
                tables.parquet. Checked first for SoA candidates
                (PTCV-51).

        Returns:
            ExtractResult with run_id, artifact keys/counts, and
            review count.

        Raises:
            ValueError: If no input sources are provided (sections,
                extracted_tables, and pdf_bytes are all None/empty).
        [PTCV-21 Scenario: Extract SoA to USDM Parquet with lineage]
        [PTCV-21 Scenario: Lineage chain verifiable from USDM back to download]
        """
        if (
            not sections
            and not extracted_tables
            and pdf_bytes is None
            and not text_blocks
        ):
            raise ValueError(
                "At least one input source required: sections, "
                "extracted_tables, pdf_bytes, or text_blocks"
            )

        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        tables: list = []

        # 1a. Primary: check pre-extracted tables.parquet (PTCV-51)
        if extracted_tables:
            tables = filter_soa_tables(extracted_tables)

        # 1b. Secondary: blended table discovery from PDF (PTCV-38)
        if not tables and pdf_bytes is not None and text_blocks:
            discovered = self._discovery.discover(
                pdf_bytes=pdf_bytes,
                text_blocks=text_blocks,
                page_count=page_count,
            )
            tables.extend(discovered)

        # 1c. Tertiary: parse SoA from ICH section text (fallback)
        if not tables and sections:
            table = self._parser.parse(sections)
            tables = [table] if table is not None else []

        # Level 3 (PTCV-166): LLM SoA construction from text
        activity_count = sum(len(t.activities) for t in tables)
        if activity_count < MIN_ACTIVITIES_THRESHOLD:
            partial_tables = tables if tables else None
            reg_meta = self._load_registry_metadata(registry_id)
            try:
                llm_tables = self._llm_builder.build(
                    sections=sections,
                    text_blocks=text_blocks,
                    partial_tables=partial_tables,
                    registry_metadata=reg_meta,
                )
                if llm_tables:
                    tables = llm_tables
            except Exception:
                logger.warning(
                    "Level 3 LLM SoA construction failed for %s; "
                    "continuing with Level 1/2 results",
                    registry_id,
                    exc_info=True,
                )

        # 2. Map to USDM entities (empty table list → zero entities)
        epochs, timepoints, activities, instances, synonyms = self._mapper.map(
            tables,
            run_id=run_id,
            source_run_id=source_run_id,
            source_sha256=source_sha256,
            registry_id=registry_id,
            timestamp=timestamp,
        )

        artifact_keys: dict[str, str] = {}

        # 3. Write Parquet artifacts (only when entities exist)
        if epochs:
            key = f"usdm/{run_id}/epochs.parquet"
            self._gateway.put_artifact(
                key=key,
                data=self._writer.epochs_to_parquet(epochs),
                content_type="application/vnd.apache.parquet",
                run_id=run_id,
                source_hash=source_sha256,
                user=who,
                immutable=False,
                stage="soa_extraction",
                registry_id=registry_id,
            )
            artifact_keys["epochs"] = key

        if timepoints:
            key = f"usdm/{run_id}/timepoints.parquet"
            self._gateway.put_artifact(
                key=key,
                data=self._writer.timepoints_to_parquet(timepoints),
                content_type="application/vnd.apache.parquet",
                run_id=run_id,
                source_hash=source_sha256,
                user=who,
                immutable=False,
                stage="soa_extraction",
                registry_id=registry_id,
            )
            artifact_keys["timepoints"] = key

        if activities:
            key = f"usdm/{run_id}/activities.parquet"
            self._gateway.put_artifact(
                key=key,
                data=self._writer.activities_to_parquet(activities),
                content_type="application/vnd.apache.parquet",
                run_id=run_id,
                source_hash=source_sha256,
                user=who,
                immutable=False,
                stage="soa_extraction",
                registry_id=registry_id,
            )
            artifact_keys["activities"] = key

        if instances:
            key = f"usdm/{run_id}/scheduled_instances.parquet"
            self._gateway.put_artifact(
                key=key,
                data=self._writer.instances_to_parquet(instances),
                content_type="application/vnd.apache.parquet",
                run_id=run_id,
                source_hash=source_sha256,
                user=who,
                immutable=False,
                stage="soa_extraction",
                registry_id=registry_id,
            )
            artifact_keys["scheduled_instances"] = key

        if synonyms:
            key = f"usdm/{run_id}/synonym_mappings.parquet"
            self._gateway.put_artifact(
                key=key,
                data=self._writer.synonyms_to_parquet(synonyms),
                content_type="application/vnd.apache.parquet",
                run_id=run_id,
                source_hash=source_sha256,
                user=who,
                immutable=False,
                stage="soa_extraction",
                registry_id=registry_id,
            )
            artifact_keys["synonym_mappings"] = key

        # 4. Enqueue uncertain synonym mappings for human review
        review_count = 0
        for sm in synonyms:
            if sm.review_required:
                self._review_queue.enqueue(
                    ReviewQueueEntry(
                        run_id=run_id,
                        registry_id=registry_id,
                        section_code="soa_synonym",
                        confidence_score=sm.confidence,
                        content_json=self._synonym_to_json(sm),
                        queue_timestamp_utc=timestamp,
                    )
                )
                review_count += 1

        return ExtractResult(
            run_id=run_id,
            registry_id=registry_id,
            epoch_count=len(epochs),
            timepoint_count=len(timepoints),
            activity_count=len(activities),
            instance_count=len(instances),
            synonym_mapping_count=len(synonyms),
            review_count=review_count,
            artifact_keys=artifact_keys,
            source_sha256=source_sha256,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _synonym_to_json(sm: SynonymMapping) -> str:
        """Serialise a SynonymMapping to a JSON string for review_queue."""
        import json
        return json.dumps(
            {
                "original_text": sm.original_text,
                "canonical_label": sm.canonical_label,
                "method": sm.method,
                "confidence": sm.confidence,
            },
            ensure_ascii=False,
        )
