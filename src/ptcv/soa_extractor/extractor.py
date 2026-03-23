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
from .diagram_extractor import extract_from_diagrams
from .llm_soa_builder import LlmSoaBuilder, MIN_ACTIVITIES_THRESHOLD
from .knowledge_base import SoaKnowledgeBase
from .mapper import UsdmMapper
from .models import ExtractResult, SynonymMapping
from .narrative_extractor import AssessmentVisitPair, extract_from_narrative
from .parser import SoaTableParser
from .resolver import SynonymResolver
from .column_validator import validate_columns
from .cross_validator import cross_validate
from .soa_merger import merge_streams, pairs_to_raw_table
from .table_bridge import filter_soa_tables
from .template_matcher import TemplateMatchReport, match_completeness
from .vision_verifier import VisionVerifier
from .table_discovery import TableDiscovery
from .writer import UsdmParquetWriter


_DEFAULT_REVIEW_DB = Path("C:/Dev/PTCV/data/sqlite/review_queue.db")
_DEFAULT_REGISTRY_CACHE = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials/registry_cache"
)
_DEFAULT_KB_DIR = Path("C:/Dev/PTCV/data/soa_knowledge_base")
_USER = "ptcv-soa-extractor"

# Synonym mappings below this threshold are routed to the review queue
_REVIEW_THRESHOLD = 0.80

# Completeness below this threshold triggers a warning (PTCV-264)
_COMPLETENESS_THRESHOLD = 0.80


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
        knowledge_base: Optional[SoaKnowledgeBase] = None,
    ) -> None:
        if gateway is None:
            gateway = FilesystemAdapter(root=Path("C:/Dev/PTCV/data"))
        if review_queue is None:
            review_queue = ReviewQueue(db_path=_DEFAULT_REVIEW_DB)

        self._gateway = gateway
        self._review_queue = review_queue
        self._knowledge_base = knowledge_base
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
    # Vision verifier helpers (PTCV-221)
    # ------------------------------------------------------------------

    @staticmethod
    def _render_soa_pages(
        pdf_bytes: bytes,
        text_blocks: Optional[list[dict]] = None,
    ) -> list[bytes]:
        """Render SoA candidate pages as PNG images.

        Identifies pages with schedule-like headings and renders
        them for Vision API verification.

        Args:
            pdf_bytes: Raw PDF bytes.
            text_blocks: Optional text blocks with page_number.

        Returns:
            List of PNG bytes for SoA candidate pages (max 3).
        """
        import re

        _SOA_RE = re.compile(
            r"schedule\s+of\s+(?:activities|assessments|visits)"
            r"|visit\s+schedule"
            r"|assessment\s+schedule",
            re.IGNORECASE,
        )

        # Find candidate pages from text blocks
        candidate_pages: set[int] = set()
        if text_blocks:
            for block in text_blocks:
                text = block.get("text", "")
                page = block.get("page_number", 0)
                if _SOA_RE.search(text):
                    candidate_pages.add(page)
                    candidate_pages.add(page + 1)

        if not candidate_pages:
            return []

        # Render pages as PNG
        try:
            import fitz
        except ImportError:
            return []

        images: list[bytes] = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in sorted(candidate_pages)[:3]:
                idx = page_num - 1
                if 0 <= idx < len(doc):
                    mat = fitz.Matrix(2.0, 2.0)  # 144 DPI
                    pix = doc[idx].get_pixmap(matrix=mat)
                    images.append(pix.tobytes("png"))
            doc.close()
        except Exception:
            pass

        return images

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
        narrative_texts: Optional[list[str]] = None,
        diagrams: Optional[list[dict]] = None,
    ) -> ExtractResult:
        """Extract SoA tables and write USDM Parquet artifacts.

        Uses a document-first approach (PTCV-60): pre-extracted tables
        and PDF table discovery are tried before ICH section text
        parsing. ICH sections are optional — the extractor can run
        independently of ICH classification.

        PTCV-260: Multi-source extraction from tables (Stream A),
        narrative prose (Stream B), and diagrams (Stream C).

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
            narrative_texts: Optional list of prose strings from B.7/B.9
                query hits (PTCV-260 Stream B). Assessment-visit pairs
                are extracted and merged with table results.
            diagrams: Optional list of diagram dicts from DiagramFinder
                (PTCV-260 Stream C). Assessment-visit pairs are
                extracted from node labels and merged.

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
        # PTCV-228: When Stage 1 runs universal table extraction
        # (PTCV-223), extracted_tables already contains ALL tables
        # including those Camelot/pdfplumber would find, so Level 1b
        # (PDF re-discovery) is skipped when Level 1a has candidates.
        if extracted_tables:
            tables = filter_soa_tables(extracted_tables)

        # 1b. Secondary: blended table discovery from PDF (PTCV-38)
        # Skipped when Level 1a produced results (PTCV-228).
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

        # Column validation (PTCV-215): check header alignment
        for tbl in tables:
            validation = validate_columns(
                header=tbl.visit_headers,
                rows=[
                    [name] + [str(s) for s in sched]
                    for name, sched in tbl.activities
                ],
            )
            if validation.needs_escalation:
                logger.warning(
                    "Column validation escalation for %s: %s",
                    registry_id,
                    validation.escalation_reason,
                )
            if validation.multi_column_headers:
                logger.info(
                    "Multi-column headers detected for %s: %s",
                    registry_id,
                    validation.multi_column_headers,
                )

        # Cross-validation (PTCV-216): check assessment completeness
        # + Vision verification (PTCV-221): selective diff-based
        # correction when cross-validation detects issues.
        for i, tbl in enumerate(tables):
            cv_result = cross_validate(tbl)
            if cv_result.needs_escalation:
                for reason in cv_result.escalation_reasons:
                    logger.warning(
                        "Cross-validation escalation for %s: %s",
                        registry_id, reason,
                    )

                # PTCV-221: Invoke Vision verifier for diff-based
                # correction instead of full Level 2 re-extraction.
                if pdf_bytes is not None:
                    coverage = (
                        cv_result.completeness.coverage_ratio
                        if cv_result.completeness
                        else 0.0
                    )
                    try:
                        verifier = VisionVerifier()
                        page_images = self._render_soa_pages(
                            pdf_bytes, text_blocks,
                        )
                        vr = verifier.verify(
                            tbl, page_images,
                            coverage_ratio=coverage,
                        )
                        if vr.verified and vr.corrected_table is not None:
                            tables[i] = vr.corrected_table
                            logger.info(
                                "Vision verifier corrected %s: "
                                "%d corrections, %d added, "
                                "cost=%d tokens",
                                registry_id,
                                len(vr.corrections),
                                len(vr.added_activities),
                                vr.token_cost,
                            )
                    except Exception:
                        logger.debug(
                            "Vision verifier failed for %s; "
                            "continuing with Level 1 results",
                            registry_id,
                            exc_info=True,
                        )

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

        # 1d. PTCV-260: Multi-source merge — narrative + diagram streams.
        # Extract assessment-visit pairs from narrative prose and
        # diagram node labels, merge with table results, and add
        # the merged table to the table list if it contributes new
        # assessments.
        narrative_pairs: list[AssessmentVisitPair] = []
        if narrative_texts:
            try:
                narrative_pairs = extract_from_narrative(narrative_texts)
            except Exception:
                logger.debug(
                    "Narrative extraction failed for %s; "
                    "continuing without narrative stream",
                    registry_id,
                    exc_info=True,
                )

        diagram_pairs: list[AssessmentVisitPair] = []
        if diagrams:
            try:
                diagram_pairs = extract_from_diagrams(diagrams)
            except Exception:
                logger.debug(
                    "Diagram extraction failed for %s; "
                    "continuing without diagram stream",
                    registry_id,
                    exc_info=True,
                )

        # Convert table activities to AssessmentVisitPair for merge
        table_pairs: list[AssessmentVisitPair] = []
        for tbl in tables:
            for name, sched in tbl.activities:
                for i, is_scheduled in enumerate(sched):
                    if is_scheduled and i < len(tbl.visit_headers):
                        table_pairs.append(AssessmentVisitPair(
                            assessment_name=name,
                            visit_label=tbl.visit_headers[i],
                            source="table",
                            confidence=0.80,
                        ))

        if narrative_pairs or diagram_pairs:
            merge_result = merge_streams(
                table_pairs, narrative_pairs, diagram_pairs,
            )
            # If merge found new assessments beyond tables, build
            # a supplementary RawSoaTable from the merged set.
            if merge_result.total_merged > len(table_pairs):
                merged_table = pairs_to_raw_table(
                    merge_result.assessments,
                )
                if merged_table is not None:
                    tables.append(merged_table)
                    logger.info(
                        "PTCV-260: Multi-source merge added %d new "
                        "pairs for %s (%d multi-stream confirmed)",
                        merge_result.total_merged - len(table_pairs),
                        registry_id,
                        merge_result.multi_stream_count,
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

        # 2b. Completeness validation against knowledge base (PTCV-264)
        completeness_report: Optional[TemplateMatchReport] = None
        completeness_ratio = 1.0
        missing_assessments: list[str] = []

        kb = self._resolve_knowledge_base()
        if kb is not None and tables:
            reg_meta = self._load_registry_metadata(registry_id)
            phase = ""
            condition = ""
            if reg_meta:
                phase = reg_meta.get("phase", "")
                condition = reg_meta.get("condition", "")

            # Run template matching on each table and keep the
            # best (highest-coverage) report across all tables.
            best_report: Optional[TemplateMatchReport] = None
            for tbl in tables:
                report = match_completeness(
                    tbl, kb, phase=phase, condition=condition,
                )
                if not report.has_template:
                    continue
                if (
                    best_report is None
                    or report.best_coverage > best_report.best_coverage
                ):
                    best_report = report

            if best_report is not None:
                completeness_report = best_report
                completeness_ratio = best_report.best_coverage
                missing_assessments = list(best_report.consensus_missing)

                if completeness_ratio < _COMPLETENESS_THRESHOLD:
                    logger.warning(
                        "Completeness %.0f%% below threshold for %s. "
                        "Expected assessments not found: %s",
                        completeness_ratio * 100,
                        registry_id,
                        ", ".join(missing_assessments)
                        if missing_assessments
                        else "(see report)",
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
            completeness_ratio=completeness_ratio,
            missing_assessments=missing_assessments,
            completeness_report=completeness_report,
        )

    # ------------------------------------------------------------------
    # Knowledge base resolution (PTCV-264)
    # ------------------------------------------------------------------

    def _resolve_knowledge_base(self) -> Optional[SoaKnowledgeBase]:
        """Return a loaded knowledge base, or None if unavailable.

        Uses the instance injected at construction time, or attempts
        to load from the default on-disk path.
        """
        if self._knowledge_base is not None:
            return self._knowledge_base

        kb = SoaKnowledgeBase(index_dir=_DEFAULT_KB_DIR)
        if kb.load():
            return kb

        logger.debug("SoA knowledge base not available; skipping completeness check")
        return None

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
