"""PTCV Extraction package — PTCV-19.

Protocol Format Detection and Text/Table Extraction.

Public API:

    ProtocolFormat          — Enum of supported formats (pdf, ctr-xml, word)
    FormatDetector          — Detects format from bytes or file path
    PdfExtractor            — PDF cascade extractor (pdfplumber→camelot→tabula)
    CtrXmlExtractor         — CDISC ODM / CTR-XML extractor
    ExtractionService       — Main orchestrator; writes via StorageGateway
    ExtractionResult        — Return type from ExtractionService.extract()
    TextBlock               — Data model for text_blocks.parquet rows
    ExtractedTable          — Data model for tables.parquet rows
    ExtractionMetadata      — Data model for metadata.parquet rows
"""

from .ctr_xml_extractor import CtrXmlExtractor
from .extraction_service import ExtractionService
from .format_detector import FormatDetector, ProtocolFormat
from .markdown_normalizer import (
    NormalizedResult,
    NormalizationStats,
    TOCItem,
    normalize_markdown,
)
from .models import ExtractionMetadata, ExtractionResult, ExtractedTable, TextBlock
from .parquet_writer import (
    metadata_to_parquet,
    parquet_to_metadata,
    parquet_to_tables,
    parquet_to_text_blocks,
    tables_to_parquet,
    text_blocks_to_parquet,
)
from .pdf_extractor import PdfExtractor

# Optional vision enhancement (requires anthropic + fitz)
try:
    from .vision_enhancer import VisionEnhancer, VisionEnhancementResult
except ImportError:
    pass

__all__ = [
    "ProtocolFormat",
    "FormatDetector",
    "PdfExtractor",
    "CtrXmlExtractor",
    "ExtractionService",
    "ExtractionResult",
    "TextBlock",
    "ExtractedTable",
    "ExtractionMetadata",
    "NormalizedResult",
    "NormalizationStats",
    "TOCItem",
    "normalize_markdown",
    "text_blocks_to_parquet",
    "tables_to_parquet",
    "metadata_to_parquet",
    "parquet_to_text_blocks",
    "parquet_to_tables",
    "parquet_to_metadata",
    "VisionEnhancer",
    "VisionEnhancementResult",
]
