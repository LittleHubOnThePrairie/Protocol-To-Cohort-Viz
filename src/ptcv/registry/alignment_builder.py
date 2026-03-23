"""Ground-Truth Alignment Corpus Builder.

PTCV-232: Aligns ClinicalTrials.gov registry metadata against protocol
PDF text to build a labeled corpus for downstream classifier training.

For each registry-linked protocol, locates where registry fields (official
title, sponsor, endpoints, eligibility criteria) appear in the PDF text
and records the alignment as structured records.

Risk tier: LOW -- read-only analysis of cached data (no API calls, no PHI).

Usage::

    from ptcv.registry.alignment_builder import (
        AlignmentBuilder,
        AlignmentRecord,
    )

    builder = AlignmentBuilder()
    corpus = builder.build_corpus()
    builder.save_parquet(corpus, "data/alignment_corpus/alignments.parquet")
"""

import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_CACHE = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials/registry_cache"
)
_DEFAULT_PDF_DIR = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials"
)
_NCT_RE = re.compile(r"NCT\d{8}")

# Fuzzy matching threshold (0-100). 80 = allow ~20% character differences.
_MATCH_THRESHOLD = 80

# Maximum characters to search in PDF text per field alignment.
_MAX_PDF_SCAN_CHARS = 50_000

# For long fields, split into chunks of this size for matching.
_CHUNK_SIZE = 300


@dataclasses.dataclass
class AlignmentRecord:
    """A single alignment between registry content and PDF text.

    Attributes:
        nct_id: ClinicalTrials.gov trial identifier.
        section_code: ICH E6(R3) section code (e.g. "B.4").
        section_name: Human-readable section name.
        registry_field: Which registry module/field this came from.
        registry_text: The registry content that was searched for.
        pdf_text_span: The matched text span from the PDF.
        char_offset: Character offset of the match in the PDF text.
        span_length: Length of the matched span.
        quality_rating: MetadataToIchMapper quality (0.3/0.6/0.9).
        similarity_score: Fuzzy match score (0-100, normalised to 0.0-1.0).
    """

    nct_id: str
    section_code: str
    section_name: str
    registry_field: str
    registry_text: str
    pdf_text_span: str
    char_offset: int
    span_length: int
    quality_rating: float
    similarity_score: float


class AlignmentBuilder:
    """Build a ground-truth alignment corpus from registry-linked protocols.

    For each protocol with both a cached registry JSON and a PDF file,
    maps registry metadata to ICH sections and locates each field's
    content within the PDF text using fuzzy matching.

    Args:
        registry_cache_dir: Directory containing ``{nct_id}.json`` cache files.
        pdf_dir: Directory containing ``{nct_id}_*.pdf`` files.
        match_threshold: Minimum fuzzy match score (0-100).
    """

    def __init__(
        self,
        registry_cache_dir: Optional[Path] = None,
        pdf_dir: Optional[Path] = None,
        match_threshold: int = _MATCH_THRESHOLD,
    ) -> None:
        self._cache_dir = registry_cache_dir or _DEFAULT_REGISTRY_CACHE
        self._pdf_dir = pdf_dir or _DEFAULT_PDF_DIR
        self._threshold = match_threshold

    def build_corpus(
        self,
        nct_ids: Optional[list[str]] = None,
    ) -> list[AlignmentRecord]:
        """Build alignment corpus from all available registry-linked protocols.

        Args:
            nct_ids: Specific NCT IDs to process. If None, processes
                all protocols with both cache and PDF files.

        Returns:
            List of AlignmentRecord objects.
        """
        if nct_ids is None:
            nct_ids = self._discover_linkable_protocols()

        logger.info(
            "AlignmentBuilder: processing %d protocols", len(nct_ids)
        )

        corpus: list[AlignmentRecord] = []
        for nct_id in nct_ids:
            records = self._align_protocol(nct_id)
            corpus.extend(records)

        logger.info(
            "AlignmentBuilder: produced %d alignment records "
            "from %d protocols",
            len(corpus),
            len(nct_ids),
        )
        return corpus

    def save_parquet(
        self,
        corpus: list[AlignmentRecord],
        output_path: str | Path,
    ) -> Path:
        """Save alignment corpus to a Parquet file.

        Args:
            corpus: List of AlignmentRecord objects.
            output_path: Destination Parquet file path.

        Returns:
            Path to the written file.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "pandas required for Parquet output. "
                "Install with: pip install pandas pyarrow"
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [dataclasses.asdict(r) for r in corpus]
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False, engine="pyarrow")

        logger.info(
            "AlignmentBuilder: saved %d records to %s",
            len(rows),
            path,
        )
        return path

    def _discover_linkable_protocols(self) -> list[str]:
        """Find NCT IDs that have both a registry cache and a PDF file.

        Returns:
            Sorted list of NCT IDs.
        """
        cached_ids: set[str] = set()
        for f in self._cache_dir.glob("NCT*.json"):
            m = _NCT_RE.search(f.stem)
            if m:
                cached_ids.add(m.group(0))

        pdf_ids: set[str] = set()
        for f in self._pdf_dir.glob("NCT*.pdf"):
            m = _NCT_RE.search(f.stem)
            if m:
                pdf_ids.add(m.group(0))

        linkable = sorted(cached_ids & pdf_ids)
        logger.info(
            "AlignmentBuilder: %d cached, %d PDFs, %d linkable",
            len(cached_ids),
            len(pdf_ids),
            len(linkable),
        )
        return linkable

    def _align_protocol(self, nct_id: str) -> list[AlignmentRecord]:
        """Align registry metadata to PDF text for one protocol.

        Args:
            nct_id: ClinicalTrials.gov identifier.

        Returns:
            List of AlignmentRecord for this protocol.
        """
        # Load registry metadata
        cache_path = self._cache_dir / f"{nct_id}.json"
        try:
            metadata = json.loads(
                cache_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cannot read cache for %s: %s", nct_id, e)
            return []

        # Map to ICH sections
        from .ich_mapper import MetadataToIchMapper

        mapper = MetadataToIchMapper()
        mapped_sections = mapper.map(metadata)
        if not mapped_sections:
            return []

        # Extract PDF text
        pdf_text = self._extract_pdf_text(nct_id)
        if not pdf_text:
            return []

        # Align each mapped section against PDF text
        records: list[AlignmentRecord] = []
        for section in mapped_sections:
            matches = self._fuzzy_align(
                registry_text=section.content_text,
                pdf_text=pdf_text,
                nct_id=nct_id,
                section_code=section.section_code,
                section_name=section.section_name,
                registry_field=section.ct_gov_module,
                quality_rating=section.quality_rating,
            )
            records.extend(matches)

        return records

    def _extract_pdf_text(self, nct_id: str) -> str:
        """Extract full text from the protocol PDF.

        Args:
            nct_id: NCT ID to find the PDF for.

        Returns:
            Full text string, or empty string on failure.
        """
        # Find PDF file (pattern: NCT{id}_*.pdf)
        candidates = list(self._pdf_dir.glob(f"{nct_id}_*.pdf"))
        if not candidates:
            candidates = list(self._pdf_dir.glob(f"{nct_id}.pdf"))
        if not candidates:
            logger.debug("No PDF found for %s", nct_id)
            return ""

        pdf_path = candidates[0]

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning(
                "PyMuPDF (fitz) not available for PDF text extraction"
            )
            return ""

        try:
            doc = fitz.open(str(pdf_path))
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(pages)
        except Exception as e:
            logger.warning(
                "PDF extraction failed for %s: %s", nct_id, e
            )
            return ""

    def _fuzzy_align(
        self,
        registry_text: str,
        pdf_text: str,
        nct_id: str,
        section_code: str,
        section_name: str,
        registry_field: str,
        quality_rating: float,
    ) -> list[AlignmentRecord]:
        """Fuzzy-match registry content against PDF text.

        For short fields (< _CHUNK_SIZE chars), matches the full field.
        For long fields, splits into paragraph-level chunks and matches
        each independently.

        Args:
            registry_text: Registry content to search for.
            pdf_text: Full PDF text to search within.
            nct_id: Trial identifier.
            section_code: ICH section code.
            section_name: Section name.
            registry_field: Source CT.gov module name.
            quality_rating: Mapping quality (0.3/0.6/0.9).

        Returns:
            List of AlignmentRecord for successful matches.
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("rapidfuzz not installed; skipping alignment")
            return []

        # Truncate PDF text for performance
        scan_text = pdf_text[:_MAX_PDF_SCAN_CHARS]

        # Split registry text into matchable chunks
        chunks = self._split_into_chunks(registry_text)

        records: list[AlignmentRecord] = []
        for chunk in chunks:
            if len(chunk.strip()) < 10:
                continue

            # Find best matching window in PDF text
            best_score = 0.0
            best_offset = 0
            best_span = ""

            # Sliding window approach for partial_ratio context
            window_size = min(len(chunk) * 2, len(scan_text))
            step = max(window_size // 4, 50)

            for start in range(0, len(scan_text) - len(chunk), step):
                end = min(start + window_size, len(scan_text))
                window = scan_text[start:end]

                score = fuzz.partial_ratio(
                    chunk.lower(), window.lower()
                )
                if score > best_score:
                    best_score = score
                    best_offset = start
                    # Extract the most relevant portion
                    best_span = window[:len(chunk) + 100]

            if best_score >= self._threshold:
                records.append(AlignmentRecord(
                    nct_id=nct_id,
                    section_code=section_code,
                    section_name=section_name,
                    registry_field=registry_field,
                    registry_text=chunk[:500],
                    pdf_text_span=best_span[:500],
                    char_offset=best_offset,
                    span_length=len(best_span),
                    quality_rating=quality_rating,
                    similarity_score=best_score / 100.0,
                ))

        return records

    @staticmethod
    def _split_into_chunks(text: str) -> list[str]:
        """Split text into matchable chunks.

        Short texts (< _CHUNK_SIZE) are returned as-is.
        Long texts are split by paragraph boundaries (double newline)
        or by _CHUNK_SIZE if no paragraph boundaries exist.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        if len(text) <= _CHUNK_SIZE:
            return [text]

        # Split by paragraph boundaries
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue
            if len(para) <= _CHUNK_SIZE:
                chunks.append(para)
            else:
                # Further split by sentences for very long paragraphs
                for i in range(0, len(para), _CHUNK_SIZE):
                    chunk = para[i : i + _CHUNK_SIZE]
                    if len(chunk.strip()) >= 20:
                        chunks.append(chunk)

        return chunks if chunks else [text[:_CHUNK_SIZE]]


__all__ = [
    "AlignmentBuilder",
    "AlignmentRecord",
]
