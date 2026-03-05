"""ExtractionService — main orchestrator for PTCV-19 extraction.

Detects protocol format and dispatches to the appropriate extractor.
All output is written through StorageGateway as DuckDB-ready Parquet
files, partitioned by run_id, with lineage records linking each
artifact back to the source protocol SHA-256.

Storage layout::

    {gateway_root}/
        extracted/
            {run_id}/
                text_blocks.parquet
                tables.parquet
                metadata.parquet

Re-extraction creates a new run_id and new Parquet files; prior
extractions under their original run_id are preserved (ALCOA+ Original).

Risk tier: MEDIUM — data pipeline orchestrator.

Regulatory references:
- ALCOA+ Contemporaneous: timestamp captured once, stamped on all rows
- ALCOA+ Traceable: source_hash chain from source PDF → each Parquet
- ALCOA+ Original: new run_id per extraction; prior artifacts immutable
  only when written with immutable=True (protocol PDFs); Parquet
  artifacts are written with immutable=False (versioned by run_id)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from ..storage import StorageGateway
from .ctr_xml_extractor import CtrXmlExtractor
from .format_detector import FormatDetector, ProtocolFormat
from .models import ExtractionMetadata, ExtractionResult
from .parquet_writer import (
    metadata_to_parquet,
    tables_to_parquet,
    text_blocks_to_parquet,
)
from .pdf_extractor import PdfExtractor

logger = logging.getLogger(__name__)

_USER = "ptcv-extraction-service"


class ExtractionService:
    """Orchestrates protocol format detection and extraction.

    Supports:
    - PDF → PdfExtractor cascade (pdfplumber → Camelot → tabula)
    - CTR-XML → CtrXmlExtractor (CDISC ODM; no PDF fallback)
    - Word (.docx) → not yet implemented; raises NotImplementedError

    Args:
        gateway: StorageGateway implementation to use for writes.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway
        self._detector = FormatDetector()
        self._pdf_extractor = PdfExtractor()
        self._xml_extractor = CtrXmlExtractor()

    def extract(
        self,
        protocol_data: bytes,
        registry_id: str,
        amendment_number: str,
        source_sha256: str,
        filename: str = "",
        source: str = "",
        run_id: Optional[str] = None,
    ) -> ExtractionResult:
        """Detect format and extract text/tables from a protocol file.

        Args:
            protocol_data: Raw bytes of the protocol file (PDF, XML…).
            registry_id: Trial identifier (e.g. "NCT00112827").
            amendment_number: Amendment version string (e.g. "1.0").
            source_sha256: SHA-256 hex of the upstream source artifact
                (i.e. the protocol file bytes as stored by PTCV-18).
            filename: Original filename — used as format detection hint.
                May be empty.
            source: Registry source name for lineage records (e.g.
                "ClinicalTrials.gov"). May be empty.
            run_id: Optional explicit run_id. If None, a UUID4 is
                generated. Intended for deterministic testing only.

        Returns:
            ExtractionResult with artifact keys, SHA-256 hashes, and
            counts.

        Raises:
            NotImplementedError: If the format is Word (.docx).
            ValueError: If the format cannot be determined.
        [PTCV-19 Scenario: Detect PDF and extract with lineage record]
        [PTCV-19 Scenario: Parse CTR-XML natively with lineage]
        [PTCV-19 Scenario: Re-extraction creates new run_id]
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        fmt = self._detector.detect_from_bytes(
            protocol_data, filename=filename
        )

        if fmt == ProtocolFormat.WORD:
            raise NotImplementedError(
                "Word (.docx) extraction is not yet implemented."
            )

        logger.info(
            "Extraction started: registry_id=%s run_id=%s format=%s",
            registry_id,
            run_id,
            fmt.value,
        )

        # Capture a single contemporaneous timestamp for all artifacts
        timestamp = datetime.now(timezone.utc).isoformat()

        landscape_pages: list[int] = []

        if fmt == ProtocolFormat.PDF:
            text_blocks, tables, page_count, landscape_pages = (
                self._pdf_extractor.extract(
                    pdf_bytes=protocol_data,
                    run_id=run_id,
                    registry_id=registry_id,
                    source_sha256=source_sha256,
                )
            )
            # Determine the predominant extractor for metadata
            extractor_used = _predominant_extractor(tables) or "pdfplumber"

        elif fmt == ProtocolFormat.CTR_XML:
            text_blocks, page_count = self._xml_extractor.extract(
                xml_bytes=protocol_data,
                run_id=run_id,
                registry_id=registry_id,
                source_sha256=source_sha256,
            )
            tables = []
            extractor_used = "ctr-xml"

        else:
            # Unknown format — attempt PDF extraction as best-effort
            logger.warning(
                "Unknown format for %s; attempting PDF extraction.",
                registry_id,
            )
            text_blocks, tables, page_count, landscape_pages = (
                self._pdf_extractor.extract(
                    pdf_bytes=protocol_data,
                    run_id=run_id,
                    registry_id=registry_id,
                    source_sha256=source_sha256,
                )
            )
            extractor_used = _predominant_extractor(tables) or "pdfplumber"

        # Stamp contemporaneous timestamp on all rows
        for blk in text_blocks:
            blk.extraction_timestamp_utc = timestamp
        for tbl in tables:
            tbl.extraction_timestamp_utc = timestamp

        # Build metadata record
        import json as _json

        meta = ExtractionMetadata(
            run_id=run_id,
            source_registry_id=registry_id,
            source_sha256=source_sha256,
            format_detected=fmt.value,
            extractor_used=extractor_used,
            page_count=page_count,
            table_count=len(tables),
            text_block_count=len(text_blocks),
            extraction_timestamp_utc=timestamp,
            landscape_pages=_json.dumps(landscape_pages),
        )

        # Serialise to Parquet
        text_bytes = text_blocks_to_parquet(text_blocks)
        tables_bytes = tables_to_parquet(tables)
        meta_bytes = metadata_to_parquet(meta)

        # Storage keys follow the spec layout
        text_key = f"extracted/{run_id}/text_blocks.parquet"
        tables_key = f"extracted/{run_id}/tables.parquet"
        meta_key = f"extracted/{run_id}/metadata.parquet"

        # Write via gateway — immutable=False (new run_id per re-run)
        _source: Optional[str] = source or None

        text_artifact = self._gateway.put_artifact(
            key=text_key,
            data=text_bytes,
            content_type="application/parquet",
            run_id=run_id,
            source_hash=source_sha256,
            user=_USER,
            immutable=False,
            stage="extraction",
            registry_id=registry_id,
            amendment_number=amendment_number,
            source=_source,
        )
        tables_artifact = self._gateway.put_artifact(
            key=tables_key,
            data=tables_bytes,
            content_type="application/parquet",
            run_id=run_id,
            source_hash=source_sha256,
            user=_USER,
            immutable=False,
            stage="extraction",
            registry_id=registry_id,
            amendment_number=amendment_number,
            source=_source,
        )
        meta_artifact = self._gateway.put_artifact(
            key=meta_key,
            data=meta_bytes,
            content_type="application/parquet",
            run_id=run_id,
            source_hash=source_sha256,
            user=_USER,
            immutable=False,
            stage="extraction",
            registry_id=registry_id,
            amendment_number=amendment_number,
            source=_source,
        )

        logger.info(
            "Extraction complete: registry_id=%s run_id=%s "
            "tables=%d text_blocks=%d",
            registry_id,
            run_id,
            len(tables),
            len(text_blocks),
        )

        return ExtractionResult(
            run_id=run_id,
            registry_id=registry_id,
            format_detected=fmt.value,
            text_artifact_key=text_key,
            tables_artifact_key=tables_key,
            metadata_artifact_key=meta_key,
            text_artifact_sha256=text_artifact.sha256,
            tables_artifact_sha256=tables_artifact.sha256,
            metadata_artifact_sha256=meta_artifact.sha256,
            source_sha256=source_sha256,
            table_count=len(tables),
            text_block_count=len(text_blocks),
        )


# -----------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------


def _predominant_extractor(tables: list) -> Optional[str]:
    """Return the most-used extractor name among extracted tables.

    Args:
        tables: List of ExtractedTable instances.

    Returns:
        Most common extractor_used value, or None if list is empty.
    """
    if not tables:
        return None
    counts: dict[str, int] = {}
    for t in tables:
        counts[t.extractor_used] = counts.get(t.extractor_used, 0) + 1
    return max(counts, key=lambda k: counts[k])
