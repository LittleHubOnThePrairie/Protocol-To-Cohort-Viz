"""Hybrid Claude vision extraction for sparse protocol pages (PTCV-172).

Post-extraction enhancement that identifies pages where text extraction
yielded poor results and sends page images to Claude's vision API for
structured extraction.

Gated by ``ExtractionLevel.E1`` (``PTCV_VISION_API_KEY`` env var).
The actual API client uses ``ANTHROPIC_API_KEY``.

Cost guardrails:
    - ``PTCV_VISION_MAX_PAGES`` env var (default 5)
    - Pages sorted by priority: cover first, empty, sparse
    - Total pages sent and estimated cost logged

Risk tier: MEDIUM — data pipeline component.

Regulatory references:
- ALCOA+ Contemporaneous: extraction_timestamp_utc set by caller
- ALCOA+ Traceable: extractor_used="claude_vision" on all table records
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import os
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Sparsity thresholds
_MIN_BLOCKS_THRESHOLD = 2
_MIN_CHARS_THRESHOLD = 50
_COVER_PAGE_LIMIT = 3  # First N pages are cover candidates

# Cost estimation constants (Sonnet vision pricing)
_EST_INPUT_TOKENS_PER_PAGE = 1800
_EST_OUTPUT_TOKENS_PER_PAGE = 500
_EST_COST_PER_INPUT_TOKEN = 3.0e-6   # $3/M input tokens
_EST_COST_PER_OUTPUT_TOKEN = 15.0e-6  # $15/M output tokens

_DEFAULT_MAX_PAGES = 5
_DEFAULT_VISION_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_DPI = 150
_MAX_TOKENS_PER_CALL = 2048

# Vision block index offset to avoid collisions with text extraction
_VISION_BLOCK_INDEX_OFFSET = 1000


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SparsePageCandidate:
    """A page identified as needing vision extraction.

    Attributes:
        page_number: 1-based page number.
        priority: Extraction priority (0=cover, 1=empty, 2=sparse).
            Lower is higher priority.
        text_char_count: Total characters of text on this page.
        block_count: Number of text blocks on this page.
    """

    page_number: int
    priority: int
    text_char_count: int
    block_count: int


@dataclasses.dataclass
class VisionExtractionResult:
    """Result from extracting a single page via vision API.

    Attributes:
        text_blocks: TextBlock instances extracted from the image.
        tables: ExtractedTable instances extracted from the image.
        is_cover_page: Whether the page was identified as a cover page.
        cover_fields: Extracted cover page fields (protocol_title,
            sponsor, principal_investigator, protocol_number).
        input_tokens: API input token count.
        output_tokens: API output token count.
    """

    text_blocks: list
    tables: list
    is_cover_page: bool
    cover_fields: dict[str, str]
    input_tokens: int
    output_tokens: int


@dataclasses.dataclass
class VisionEnhancementResult:
    """Aggregate result from vision enhancement across all pages.

    Attributes:
        text_blocks: All vision-extracted text blocks.
        tables: All vision-extracted tables.
        pages_processed: Number of pages sent to vision API.
        total_input_tokens: Total API input tokens.
        total_output_tokens: Total API output tokens.
        cover_fields: Merged cover page fields from all pages.
        estimated_cost_usd: Estimated API cost in USD.
    """

    text_blocks: list
    tables: list
    pages_processed: int
    total_input_tokens: int
    total_output_tokens: int
    cover_fields: dict[str, str]
    estimated_cost_usd: float


# ------------------------------------------------------------------
# Vision extraction prompt
# ------------------------------------------------------------------

_VISION_EXTRACTION_PROMPT = """\
You are analyzing a page from a clinical trial protocol PDF. \
Extract ALL text and tabular content visible on this page.

Return ONLY a valid JSON object (no markdown fences) with this structure:
{
  "text_blocks": [
    {"text": "...", "block_type": "heading|paragraph|list_item"}
  ],
  "tables": [
    {
      "header": ["col1", "col2"],
      "data_rows": [["val1", "val2"]]
    }
  ],
  "is_cover_page": false,
  "cover_fields": {
    "protocol_title": "",
    "sponsor": "",
    "principal_investigator": "",
    "protocol_number": "",
    "amendment_number": ""
  }
}

Guidelines:
- Extract text in reading order (top-to-bottom, left-to-right)
- For tables (especially Schedule of Activities / SoA), preserve \
all column headers and data cells
- Use "X", checkmark, or the actual cell content for activity markers
- If this is a cover/title page, set is_cover_page to true and \
populate cover_fields
- Set block_type to "heading" for titles/headers, "list_item" for \
bulleted/numbered items, "paragraph" for body text
- Omit empty text blocks
"""


# ------------------------------------------------------------------
# VisionEnhancer
# ------------------------------------------------------------------

class VisionEnhancer:
    """Post-extraction vision enhancement for sparse pages.

    Identifies pages where text extraction yielded poor results and
    sends page images to Claude's vision API for structured extraction.

    Args:
        max_pages: Maximum pages to send to vision API per protocol.
            0 means read from ``PTCV_VISION_MAX_PAGES`` env var
            (default 5).
        claude_model: Model identifier for vision API calls.
    """

    def __init__(
        self,
        max_pages: int = 0,
        claude_model: str = _DEFAULT_VISION_MODEL,
    ) -> None:
        if max_pages <= 0:
            max_pages = int(
                os.environ.get("PTCV_VISION_MAX_PAGES",
                               str(_DEFAULT_MAX_PAGES))
            )
        self._max_pages = max_pages
        self._claude_model = claude_model
        self._client: Any = None

    @property
    def max_pages(self) -> int:
        """Maximum pages to process per protocol."""
        return self._max_pages

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazy-initialize Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                logger.warning(
                    "VisionEnhancer: 'anthropic' package not installed "
                    "— vision extraction unavailable. "
                    "Install with: pip install anthropic"
                )
                raise
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
        return self._client

    # ------------------------------------------------------------------
    # Sparse page identification (pure logic)
    # ------------------------------------------------------------------

    def identify_sparse_pages(
        self,
        text_blocks: list,
        tables: list,
        page_count: int,
    ) -> list[SparsePageCandidate]:
        """Identify pages that may benefit from vision extraction.

        Pure logic method — no API calls, no side effects.

        Args:
            text_blocks: TextBlock instances from text extraction.
            tables: ExtractedTable instances from text extraction.
            page_count: Total page count in the document.

        Returns:
            List of SparsePageCandidate sorted by priority, capped
            at max_pages.
        """
        # Build per-page stats
        page_block_counts: dict[int, int] = defaultdict(int)
        page_char_counts: dict[int, int] = defaultdict(int)
        for blk in text_blocks:
            pn = blk.page_number
            page_block_counts[pn] += 1
            page_char_counts[pn] += len(blk.text)

        pages_with_tables: set[int] = set()
        for tbl in tables:
            pages_with_tables.add(tbl.page_number)

        candidates: dict[int, SparsePageCandidate] = {}

        # Cover page candidates (first N pages)
        for pn in range(1, min(_COVER_PAGE_LIMIT + 1, page_count + 1)):
            if pn not in candidates:
                candidates[pn] = SparsePageCandidate(
                    page_number=pn,
                    priority=0,
                    text_char_count=page_char_counts.get(pn, 0),
                    block_count=page_block_counts.get(pn, 0),
                )

        # Empty / sparse pages
        for pn in range(1, page_count + 1):
            if pn in candidates:
                continue
            bc = page_block_counts.get(pn, 0)
            cc = page_char_counts.get(pn, 0)
            has_table = pn in pages_with_tables

            if bc < _MIN_BLOCKS_THRESHOLD and not has_table:
                candidates[pn] = SparsePageCandidate(
                    page_number=pn,
                    priority=1,
                    text_char_count=cc,
                    block_count=bc,
                )
            elif cc < _MIN_CHARS_THRESHOLD and not has_table:
                candidates[pn] = SparsePageCandidate(
                    page_number=pn,
                    priority=2,
                    text_char_count=cc,
                    block_count=bc,
                )

        # Sort by priority then page number, cap at max_pages
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: (c.priority, c.page_number),
        )
        return sorted_candidates[: self._max_pages]

    # ------------------------------------------------------------------
    # Page rendering
    # ------------------------------------------------------------------

    @staticmethod
    def render_page_to_png(
        pdf_bytes: bytes,
        page_number: int,
        dpi: int = _DEFAULT_DPI,
    ) -> bytes:
        """Render a PDF page to PNG bytes.

        Args:
            pdf_bytes: Raw PDF file contents.
            page_number: 1-based page number.
            dpi: Render resolution (default 150).

        Returns:
            PNG image bytes.

        Raises:
            ImportError: If PyMuPDF (fitz) is not installed.
            IndexError: If page_number is out of range.
        """
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                raise IndexError(
                    f"Page {page_number} out of range "
                    f"(document has {len(doc)} pages)"
                )
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            return pix.tobytes("png")
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Single-page vision extraction
    # ------------------------------------------------------------------

    def extract_from_image(
        self,
        image_png: bytes,
        page_number: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
        table_index_start: int = 0,
    ) -> VisionExtractionResult:
        """Extract content from a page image via Claude vision API.

        Args:
            image_png: PNG image bytes of the page.
            page_number: 1-based page number.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.
            table_index_start: Starting table_index for numbering.

        Returns:
            VisionExtractionResult with extracted blocks and tables.
        """
        from .models import ExtractedTable, TextBlock

        client = self._get_client()
        b64_data = base64.b64encode(image_png).decode("ascii")

        response = client.messages.create(
            model=self._claude_model,
            max_tokens=_MAX_TOKENS_PER_CALL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": _VISION_EXTRACTION_PROMPT,
                    },
                ],
            }],
        )

        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)

        # Parse response
        raw_text = response.content[0].text if response.content else ""
        try:
            parsed = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Vision extraction returned unparseable JSON for "
                "page %d; skipping",
                page_number,
            )
            return VisionExtractionResult(
                text_blocks=[],
                tables=[],
                is_cover_page=False,
                cover_fields={},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Build TextBlock instances
        blocks: list[TextBlock] = []
        for idx, item in enumerate(parsed.get("text_blocks", [])):
            text = item.get("text", "").strip()
            if not text:
                continue
            blocks.append(TextBlock(
                run_id=run_id,
                source_registry_id=registry_id,
                source_sha256=source_sha256,
                page_number=page_number,
                block_index=_VISION_BLOCK_INDEX_OFFSET + idx,
                text=text,
                block_type=item.get("block_type", "paragraph"),
            ))

        # Build ExtractedTable instances
        extracted_tables: list[ExtractedTable] = []
        for idx, tbl in enumerate(parsed.get("tables", [])):
            header = tbl.get("header", [])
            data_rows = tbl.get("data_rows", [])
            if not header or not data_rows:
                continue
            extracted_tables.append(ExtractedTable(
                run_id=run_id,
                source_registry_id=registry_id,
                source_sha256=source_sha256,
                page_number=page_number,
                extractor_used="claude_vision",
                table_index=table_index_start + idx,
                header_row=json.dumps(header),
                data_rows=json.dumps(data_rows),
            ))

        is_cover = bool(parsed.get("is_cover_page", False))
        cover_fields = parsed.get("cover_fields", {}) if is_cover else {}

        return VisionExtractionResult(
            text_blocks=blocks,
            tables=extracted_tables,
            is_cover_page=is_cover,
            cover_fields=cover_fields,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ------------------------------------------------------------------
    # Main enhancement orchestrator
    # ------------------------------------------------------------------

    def enhance(
        self,
        pdf_bytes: bytes,
        text_blocks: list,
        tables: list,
        page_count: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
    ) -> VisionEnhancementResult:
        """Run vision enhancement on sparse pages of a protocol.

        Args:
            pdf_bytes: Raw PDF file contents.
            text_blocks: TextBlock instances from text extraction.
            tables: ExtractedTable instances from text extraction.
            page_count: Total page count.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.

        Returns:
            VisionEnhancementResult with all vision-extracted content.
        """
        candidates = self.identify_sparse_pages(
            text_blocks, tables, page_count,
        )

        if not candidates:
            return VisionEnhancementResult(
                text_blocks=[],
                tables=[],
                pages_processed=0,
                total_input_tokens=0,
                total_output_tokens=0,
                cover_fields={},
                estimated_cost_usd=0.0,
            )

        # Compute starting table_index after existing tables
        max_existing_table_idx = max(
            (t.table_index for t in tables), default=-1,
        )
        next_table_index = max_existing_table_idx + 1

        all_blocks: list = []
        all_tables: list = []
        total_in = 0
        total_out = 0
        merged_cover: dict[str, str] = {}
        pages_processed = 0

        for candidate in candidates:
            try:
                png_bytes = self.render_page_to_png(
                    pdf_bytes, candidate.page_number,
                )
            except Exception:
                logger.debug(
                    "Failed to render page %d; skipping",
                    candidate.page_number,
                    exc_info=True,
                )
                continue

            try:
                result = self.extract_from_image(
                    image_png=png_bytes,
                    page_number=candidate.page_number,
                    run_id=run_id,
                    registry_id=registry_id,
                    source_sha256=source_sha256,
                    table_index_start=next_table_index,
                )
            except Exception:
                logger.warning(
                    "Vision extraction failed for page %d; skipping",
                    candidate.page_number,
                    exc_info=True,
                )
                continue

            all_blocks.extend(result.text_blocks)
            all_tables.extend(result.tables)
            next_table_index += len(result.tables)
            total_in += result.input_tokens
            total_out += result.output_tokens
            pages_processed += 1

            if result.is_cover_page and result.cover_fields:
                for k, v in result.cover_fields.items():
                    if v and k not in merged_cover:
                        merged_cover[k] = v

        estimated_cost = (
            total_in * _EST_COST_PER_INPUT_TOKEN
            + total_out * _EST_COST_PER_OUTPUT_TOKEN
        )

        logger.info(
            "Vision enhancement complete: %d/%d candidate pages "
            "processed, %d text blocks, %d tables, "
            "%d input tokens, %d output tokens, cost ~$%.4f "
            "for %s",
            pages_processed,
            len(candidates),
            len(all_blocks),
            len(all_tables),
            total_in,
            total_out,
            estimated_cost,
            registry_id,
        )

        return VisionEnhancementResult(
            text_blocks=all_blocks,
            tables=all_tables,
            pages_processed=pages_processed,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            cover_fields=merged_cover,
            estimated_cost_usd=round(estimated_cost, 6),
        )
