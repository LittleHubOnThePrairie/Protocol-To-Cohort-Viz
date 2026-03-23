"""PDF extraction cascade for PTCV-19.

Applies a three-level cascade to extract text blocks and tables from
clinical trial protocol PDFs:

  Level 1 — pdfplumber (primary): Best for complex SoA tables with
      merged cells.
  Level 2 — Camelot Lattice (fallback): Standard bordered tables where
      pdfplumber returns zero rows.
  Level 3 — tabula-py (final fallback): Lattice + stream modes.
  Level 4 — LLM vision (placeholder, not yet implemented): For
      fully-scanned pages where all text-based extractors fail.

Multi-page table reconstruction:
  When a SoA table spans multiple pages the column header row repeats
  at the top of each continuation page. The extractor detects repeated
  headers and merges the continuation rows into the first table record.

Risk tier: MEDIUM — data pipeline component.

Regulatory references:
- ALCOA+ Contemporaneous: extraction_timestamp_utc is set by caller
- ALCOA+ Traceable: extractor_used recorded per table row
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import ExtractedTable, TextBlock

logger = logging.getLogger(__name__)

_USER = "ptcv-extraction-service"

# Minimum fraction of sampled lines that must have rotated direction
# vectors before we classify a page as landscape-content.
_ROTATION_THRESHOLD = 0.5

# Maximum number of text blocks to sample for rotation detection.
_ROTATION_SAMPLE_LIMIT = 20


def _collect_table_cell_texts(page: Any) -> set[str]:
    """Collect normalized text fragments from table cells on a page.

    Uses pdfplumber's ``extract_tables()`` to find table regions,
    then collects all non-empty cell texts. Used to detect which
    ``extract_text()`` lines come from table regions (PTCV-224).

    Args:
        page: A pdfplumber Page object.

    Returns:
        Set of normalized cell text fragments (lowercased, stripped).
    """
    cell_texts: set[str] = set()
    try:
        tables = page.extract_tables()  # type: ignore[union-attr]
        if not tables:
            return cell_texts
        for tbl in tables:
            for row in tbl:
                for cell in row:
                    if cell:
                        text = str(cell).strip()
                        if len(text) >= 2:
                            cell_texts.add(text.lower())
    except Exception:
        pass
    return cell_texts


def _line_overlaps_table(
    line: str,
    table_cell_texts: set[str],
) -> bool:
    """Check if a text line overlaps with table cell content.

    A line is considered part of a table if it matches a table
    cell exactly, or if >50% of its words appear in table cells.

    Args:
        line: Stripped text line from ``extract_text()``.
        table_cell_texts: Set of normalized table cell texts.

    Returns:
        True if the line likely comes from a table region.
    """
    if not table_cell_texts:
        return False

    normalized = line.strip().lower()

    # Exact match
    if normalized in table_cell_texts:
        return True

    # Word-level overlap: if most words in this line appear in
    # table cells, it's likely garbled table content
    words = normalized.split()
    if len(words) < 2:
        return False

    matches = sum(
        1 for w in words
        if any(w in cell for cell in table_cell_texts)
    )
    return matches / len(words) > 0.5


class PdfExtractor:
    """Extracts text blocks and tables from PDF bytes.

    Attributes:
        min_words_for_text_block: Minimum word count to keep a text
            block (filters noise). Default 3.
    """

    def __init__(
        self,
        min_words_for_text_block: int = 3,
        enable_universal_tables: bool = False,
    ) -> None:
        self._min_words = min_words_for_text_block
        self._enable_universal_tables = enable_universal_tables

    # ------------------------------------------------------------------
    # Page rotation normalisation (PTCV-63)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_page_rotation(
        pdf_bytes: bytes,
    ) -> tuple[bytes, list[int]]:
        """Detect and fix pages with rotated content.

        Some PDFs render landscape tables by applying a 90-degree
        rotation transform in the content stream rather than setting
        the ``/Rotate`` page attribute.  pdfplumber does not handle
        this case, producing reversed text.

        This method uses PyMuPDF (fitz) to detect such pages via their
        text-line direction vectors, then sets ``/Rotate = 90`` so that
        pdfplumber reads them correctly.

        Args:
            pdf_bytes: Raw PDF file contents.

        Returns:
            Tuple of (normalized_pdf_bytes, landscape_page_numbers)
            where landscape_page_numbers is a list of 1-based page
            numbers that were rotated.
        """
        try:
            import fitz
        except ImportError:
            logger.warning(
                "PyMuPDF (fitz) not installed — landscape page "
                "detection disabled."
            )
            return pdf_bytes, []

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        landscape_pages: list[int] = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                # Skip pages that already have explicit rotation.
                if page.rotation != 0:
                    continue

                blocks = page.get_text("dict").get("blocks", [])
                rotated = 0
                total = 0
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        d = line.get("dir", (1.0, 0.0))
                        total += 1
                        # Normal text: dir ≈ (1, 0).
                        # Rotated text: dir ≈ (0, ±1).
                        if abs(d[0]) < 0.1 and abs(d[1]) > 0.1:
                            rotated += 1
                        if total >= _ROTATION_SAMPLE_LIMIT:
                            break
                    if total >= _ROTATION_SAMPLE_LIMIT:
                        break

                if total > 0 and rotated / total >= _ROTATION_THRESHOLD:
                    page.set_rotation(90)
                    landscape_pages.append(page_idx + 1)  # 1-based

            if landscape_pages:
                logger.info(
                    "Landscape content detected on pages %s — "
                    "rotation applied.",
                    landscape_pages,
                )
                pdf_bytes = doc.tobytes()
        finally:
            doc.close()

        return pdf_bytes, landscape_pages

    def extract(
        self,
        pdf_bytes: bytes,
        run_id: str,
        registry_id: str,
        source_sha256: str,
    ) -> tuple[list["TextBlock"], list["ExtractedTable"], int, list[int]]:
        """Extract text blocks and tables from a PDF.

        Uses pymupdf4llm as the primary extractor with PTCV-176
        normalisation; falls back to the pdfplumber cascade on failure.

        Args:
            pdf_bytes: Raw PDF file contents.
            run_id: UUID4 for this extraction run (used in all rows).
            registry_id: Trial identifier (e.g. "NCT00112827").
            source_sha256: SHA-256 of the source PDF file.

        Returns:
            Tuple of (text_blocks, tables, page_count, landscape_pages).
            landscape_pages is a list of 1-based page numbers where
            rotated content was detected and normalised.
        """
        # Pre-process: detect and fix landscape-content pages (PTCV-63).
        pdf_bytes, landscape_pages = self._normalize_page_rotation(
            pdf_bytes,
        )

        # PTCV-170: Try pymupdf4llm first, fall back to pdfplumber
        try:
            return self._extract_pymupdf4llm(
                pdf_bytes, run_id, registry_id, source_sha256,
                landscape_pages,
            )
        except Exception as exc:
            logger.warning(
                "pymupdf4llm extraction failed (%s) — "
                "falling back to pdfplumber cascade",
                exc,
            )
            return self._extract_pdfplumber(
                pdf_bytes, run_id, registry_id, source_sha256,
                landscape_pages,
            )

    # ------------------------------------------------------------------
    # pymupdf4llm extraction (PTCV-170)
    # ------------------------------------------------------------------

    def _extract_pymupdf4llm(
        self,
        pdf_bytes: bytes,
        run_id: str,
        registry_id: str,
        source_sha256: str,
        landscape_pages: list[int],
    ) -> tuple[list["TextBlock"], list["ExtractedTable"], int, list[int]]:
        """Extract text and tables using pymupdf4llm + PTCV-176 normaliser.

        Args:
            pdf_bytes: Rotation-normalised PDF bytes.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.
            landscape_pages: Pages with detected landscape rotation.

        Returns:
            Same tuple as :meth:`extract`.
        """
        import fitz
        import pymupdf4llm

        from .markdown_normalizer import normalize_markdown

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_count = len(doc)
            chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
        finally:
            doc.close()

        text_blocks: list["TextBlock"] = []
        tables: list["ExtractedTable"] = []
        table_index = 0

        for page_num, chunk in enumerate(chunks, start=1):
            raw_md = chunk["text"]
            normalized = normalize_markdown(raw_md)
            md_text = normalized.text

            # Parse TextBlocks from normalised markdown
            page_blocks = self._parse_markdown_blocks(
                md_text, page_num, run_id, registry_id, source_sha256,
            )
            text_blocks.extend(page_blocks)

            # Parse markdown tables into ExtractedTable objects
            page_tables = self._parse_markdown_tables(
                md_text, page_num, run_id, registry_id,
                source_sha256, table_index,
            )
            tables.extend(page_tables)
            table_index += len(page_tables)

        # PTCV-223: Universal table extraction — supplement pymupdf4llm
        # tables with Camelot + pdfplumber discoveries.
        if self._enable_universal_tables:
            from .table_extractor import extract_all_tables

            tables = extract_all_tables(
                pdf_bytes=pdf_bytes,
                page_count=page_count,
                run_id=run_id,
                registry_id=registry_id,
                source_sha256=source_sha256,
                existing_tables=tables,
            )

        # Quality gate: if pymupdf4llm produced very few text blocks
        # relative to page count, the PDF likely has unusual structure
        # (scanned, form-overlaid, etc.) and pdfplumber may do better.
        pages_with_content = len({
            b.page_number for b in text_blocks
        } | {t.page_number for t in tables})
        if page_count > 3 and pages_with_content < page_count * 0.3:
            raise RuntimeError(
                f"pymupdf4llm quality gate: only {pages_with_content}/"
                f"{page_count} pages produced content"
            )

        logger.info(
            "pymupdf4llm extraction: %d pages, %d text blocks, "
            "%d tables for %s",
            page_count, len(text_blocks), len(tables), registry_id,
        )
        return text_blocks, tables, page_count, landscape_pages

    def _parse_markdown_blocks(
        self,
        md_text: str,
        page_num: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
    ) -> list["TextBlock"]:
        """Parse normalised markdown into TextBlock instances.

        Strips markdown syntax so TextBlock.text contains plain text,
        preserving backward compatibility with existing consumers.
        Block type is inferred from markdown syntax (more accurate than
        the heuristic classifier used by pdfplumber).

        Args:
            md_text: Normalised markdown for one page.
            page_num: 1-based page number.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.

        Returns:
            List of TextBlock instances.
        """
        import re

        from .models import TextBlock

        blocks: list[TextBlock] = []
        block_index = 0

        for line in md_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Skip table rows (handled by _parse_markdown_tables)
            if stripped.startswith("|"):
                continue
            # Skip horizontal rules / page separators
            if re.match(r"^-{3,}$", stripped):
                continue

            # Detect block type from markdown syntax
            if stripped.startswith("#"):
                block_type = "heading"
                text = re.sub(r"^#+\s*", "", stripped)
            elif re.match(r"^[-*+]\s", stripped):
                block_type = "list_item"
                text = stripped
            elif re.match(r"^\d+[.)]\s", stripped):
                block_type = "list_item"
                text = stripped
            else:
                block_type = "paragraph"
                text = stripped

            # Strip markdown formatting for plain-text compatibility
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = re.sub(r"\*(.+?)\*", r"\1", text)
            text = re.sub(r"_(.+?)_", r"\1", text)

            word_count = len(text.split())
            if word_count < self._min_words:
                continue

            blocks.append(
                TextBlock(
                    run_id=run_id,
                    source_registry_id=registry_id,
                    source_sha256=source_sha256,
                    page_number=page_num,
                    block_index=block_index,
                    text=text,
                    block_type=block_type,
                )
            )
            block_index += 1

        return blocks

    def _parse_markdown_tables(
        self,
        md_text: str,
        page_num: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
        table_index_start: int,
    ) -> list["ExtractedTable"]:
        """Parse markdown tables into ExtractedTable objects.

        Detects contiguous blocks of ``|``-delimited rows, separates
        the header row from the separator and data rows, and builds
        ExtractedTable instances compatible with the pdfplumber schema.

        Args:
            md_text: Normalised markdown for one page.
            page_num: 1-based page number.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.
            table_index_start: Starting table_index for numbering.

        Returns:
            List of ExtractedTable instances.
        """
        tables: list["ExtractedTable"] = []
        table_index = table_index_start

        lines = md_text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line.startswith("|"):
                i += 1
                continue

            # Collect contiguous table rows
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            # Need header + separator + at least 1 data row
            if len(table_lines) < 3:
                continue

            # First row = header, second = separator (skip), rest = data
            header = self._parse_table_row(table_lines[0])
            data_rows = [
                self._parse_table_row(r)
                for r in table_lines[2:]
                if not all(
                    c.strip() in ("", "---", "---:")
                    for c in r.strip("|").split("|")
                )
            ]

            if header and data_rows:
                extracted = self._build_table(
                    header=header,
                    rows=data_rows,
                    page_num=page_num,
                    table_index=table_index,
                    run_id=run_id,
                    registry_id=registry_id,
                    source_sha256=source_sha256,
                    extractor="pymupdf4llm",
                )
                tables.append(extracted)
                table_index += 1

        return tables

    @staticmethod
    def _parse_table_row(line: str) -> list[str]:
        """Parse a markdown table row ``|a|b|c|`` into cell strings."""
        cells = line.strip("|").split("|")
        return [cell.strip() for cell in cells]

    # ------------------------------------------------------------------
    # pdfplumber fallback extraction
    # ------------------------------------------------------------------

    def _extract_pdfplumber(
        self,
        pdf_bytes: bytes,
        run_id: str,
        registry_id: str,
        source_sha256: str,
        landscape_pages: list[int],
    ) -> tuple[list["TextBlock"], list["ExtractedTable"], int, list[int]]:
        """Fallback extraction using pdfplumber + Camelot + tabula.

        This is the original PTCV-19 cascade, retained as a fallback
        when pymupdf4llm is unavailable or fails.
        """
        import io

        import pdfplumber

        text_blocks: list[TextBlock] = []
        raw_tables_by_page: list[list[list[list[str]]]] = []
        page_count = 0

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                blocks = self._extract_text_blocks(
                    page, page_num, run_id, registry_id, source_sha256,
                )
                text_blocks.extend(blocks)
                page_tables = self._extract_tables_pdfplumber(page)
                raw_tables_by_page.append(page_tables)

        merged_pages = self._reconstruct_multi_page_tables(
            raw_tables_by_page,
        )

        tables: list[ExtractedTable] = []
        table_index = 0
        for page_num, page_tables in enumerate(merged_pages, start=1):
            for table_data in page_tables:
                if not table_data or len(table_data) < 2:
                    continue
                header = table_data[0]
                rows = table_data[1:]
                extracted = self._build_table(
                    header=header,
                    rows=rows,
                    page_num=page_num,
                    table_index=table_index,
                    run_id=run_id,
                    registry_id=registry_id,
                    source_sha256=source_sha256,
                    extractor="pdfplumber",
                )
                tables.append(extracted)
                table_index += 1

        tables_from_fallback = self._fallback_extract(
            pdf_bytes=pdf_bytes,
            pages_with_tables={t.page_number for t in tables},
            page_count=page_count,
            run_id=run_id,
            registry_id=registry_id,
            source_sha256=source_sha256,
            table_index_start=table_index,
        )
        tables.extend(tables_from_fallback)

        logger.info(
            "pdfplumber fallback extraction: %d pages, %d text blocks, "
            "%d tables for %s",
            page_count, len(text_blocks), len(tables), registry_id,
        )
        return text_blocks, tables, page_count, landscape_pages

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text_blocks(
        self,
        page: Any,
        page_num: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
    ) -> list["TextBlock"]:
        """Extract text blocks from a pdfplumber page.

        PTCV-224: Detects lines that overlap with table regions
        and marks them with ``in_table=True``. Downstream consumers
        can filter ``in_table=False`` for clean prose.

        Args:
            page: pdfplumber Page object.
            page_num: 1-based page number.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source file SHA-256.

        Returns:
            List of TextBlock instances (empty string blocks excluded).
        """
        from .models import TextBlock

        raw_text = page.extract_text() or ""

        # PTCV-224: Build set of table-cell text fragments
        # to detect which text lines come from table regions
        table_cell_texts = _collect_table_cell_texts(page)

        blocks: list[TextBlock] = []
        block_index = 0
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            word_count = len(stripped.split())
            if word_count < self._min_words:
                continue
            block_type = self._classify_block_type(stripped)

            # Check if this line overlaps with table content
            in_table = _line_overlaps_table(stripped, table_cell_texts)

            blocks.append(
                TextBlock(
                    run_id=run_id,
                    source_registry_id=registry_id,
                    source_sha256=source_sha256,
                    page_number=page_num,
                    block_index=block_index,
                    text=stripped,
                    block_type=block_type,
                    in_table=in_table,
                )
            )
            block_index += 1
        return blocks

    @staticmethod
    def _classify_block_type(text: str) -> str:
        """Infer block type from text content heuristics.

        Args:
            text: Text line (already stripped).

        Returns:
            One of "heading", "list_item", or "paragraph".
        """
        stripped = text.strip()
        # Short ALL-CAPS or title-case lines ≤ 8 words → heading
        words = stripped.split()
        if len(words) <= 8 and (
            stripped.isupper()
            or stripped.istitle()
            or (words[0].isupper() and len(words) <= 4)
        ):
            return "heading"
        # Lines starting with bullet markers → list_item
        if stripped[:2] in ("• ", "- ", "* ", "· ") or (
            len(stripped) > 2
            and stripped[0].isdigit()
            and stripped[1] in (".", ")")
        ):
            return "list_item"
        return "paragraph"

    # ------------------------------------------------------------------
    # Table extraction — pdfplumber
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tables_pdfplumber(page: Any) -> list[list[list[str]]]:
        """Extract tables from a pdfplumber page.

        Args:
            page: pdfplumber Page object.

        Returns:
            List of tables; each table is a list of rows; each row is a
            list of cell strings (None cells coerced to "").
        """
        raw = page.extract_tables() or []
        result = []
        for table in raw:
            cleaned = [
                [str(cell) if cell is not None else "" for cell in row]
                for row in table
            ]
            result.append(cleaned)
        return result

    # ------------------------------------------------------------------
    # Multi-page table reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_multi_page_tables(
        self,
        pages_tables: list[list[list[list[str]]]],
    ) -> list[list[list[list[str]]]]:
        """Merge tables that span multiple pages.

        Detects continuation pages by comparing the first row of a
        page's first table against the running column header signature.
        If they match, the data rows are appended to the merged table
        and the continuation entry is removed from the page.

        Args:
            pages_tables: Outer list is pages (0-indexed); inner list is
                tables on that page; innermost is rows×cells.

        Returns:
            List of per-page tables with multi-page tables merged.
        """
        if not pages_tables:
            return pages_tables

        # State: (header_sig, merged_table_page_idx, merged_table_idx)
        active_header: list[str] | None = None
        active_page: int = -1
        active_idx: int = -1

        result: list[list[list[list[str]]]] = [
            list(page) for page in pages_tables
        ]

        for page_idx, page_tables in enumerate(result):
            if not page_tables:
                active_header = None
                continue

            new_page_tables: list[list[list[str]]] = []
            for tbl_idx, table in enumerate(page_tables):
                if not table:
                    new_page_tables.append(table)
                    continue

                first_row = [c.strip() for c in table[0]]

                if (
                    active_header is not None
                    and first_row == active_header
                    and tbl_idx == 0
                    and len(table) > 1
                ):
                    # Continuation: append data rows to the merged table
                    result[active_page][active_idx].extend(table[1:])
                    # Skip this table on the current page
                else:
                    # Start a new merged table
                    active_header = first_row if first_row else None
                    active_page = page_idx
                    active_idx = len(new_page_tables)
                    new_page_tables.append(table)

            result[page_idx] = new_page_tables

        return result

    # ------------------------------------------------------------------
    # Fallback extraction — Camelot + tabula
    # ------------------------------------------------------------------

    def _fallback_extract(
        self,
        pdf_bytes: bytes,
        pages_with_tables: set[int],
        page_count: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
        table_index_start: int,
    ) -> list["ExtractedTable"]:
        """Run Camelot then tabula on pages missed by pdfplumber.

        Args:
            pdf_bytes: Raw PDF bytes.
            pages_with_tables: 1-based page numbers already covered.
            page_count: Total page count.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.
            table_index_start: Start index for table_index numbering.

        Returns:
            List of ExtractedTable from fallback extractors.
        """
        import io
        import tempfile

        tables: list[ExtractedTable] = []
        table_index = table_index_start

        missing_pages = [
            p
            for p in range(1, page_count + 1)
            if p not in pages_with_tables
        ]
        if not missing_pages:
            return tables

        # Write to temp file (Camelot/tabula need a filesystem path)
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            for page_num in missing_pages:
                new_tables = self._camelot_extract(
                    tmp_path, page_num, run_id, registry_id,
                    source_sha256, table_index
                )
                if new_tables:
                    tables.extend(new_tables)
                    table_index += len(new_tables)
                    continue

                # Final fallback: tabula
                new_tables = self._tabula_extract(
                    tmp_path, page_num, run_id, registry_id,
                    source_sha256, table_index
                )
                tables.extend(new_tables)
                table_index += len(new_tables)
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return tables

    def _camelot_extract(
        self,
        pdf_path: str,
        page_num: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
        table_index_start: int,
    ) -> list["ExtractedTable"]:
        """Extract tables from one page using Camelot Lattice.

        Args:
            pdf_path: Filesystem path to the PDF.
            page_num: 1-based page number.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.
            table_index_start: First table_index value to assign.

        Returns:
            List of ExtractedTable (empty if no tables found).
        """
        try:
            import camelot

            tables_camelot = camelot.read_pdf(
                pdf_path,
                pages=str(page_num),
                flavor="lattice",
            )
        except Exception as exc:
            logger.debug(
                "Camelot failed on page %d: %s", page_num, exc
            )
            return []

        result: list[ExtractedTable] = []
        for i, tbl in enumerate(tables_camelot):
            df = tbl.df
            if df.empty or len(df) < 2:
                continue
            header = df.iloc[0].tolist()
            rows = df.iloc[1:].values.tolist()
            extracted = self._build_table(
                header=header,
                rows=rows,
                page_num=page_num,
                table_index=table_index_start + i,
                run_id=run_id,
                registry_id=registry_id,
                source_sha256=source_sha256,
                extractor="camelot",
            )
            result.append(extracted)
        return result

    def _tabula_extract(
        self,
        pdf_path: str,
        page_num: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
        table_index_start: int,
    ) -> list["ExtractedTable"]:
        """Extract tables from one page using tabula-py.

        Tries lattice mode first, then stream mode.

        Args:
            pdf_path: Filesystem path to the PDF.
            page_num: 1-based page number.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.
            table_index_start: First table_index value to assign.

        Returns:
            List of ExtractedTable (empty if no tables found).
        """
        try:
            import tabula

            dfs = tabula.read_pdf(
                pdf_path,
                pages=page_num,
                lattice=True,
                pandas_options={"header": None},
                silent=True,
            )
            if not dfs:
                dfs = tabula.read_pdf(
                    pdf_path,
                    pages=page_num,
                    stream=True,
                    pandas_options={"header": None},
                    silent=True,
                )
        except Exception as exc:
            logger.debug(
                "tabula failed on page %d: %s", page_num, exc
            )
            return []

        result: list[ExtractedTable] = []
        for i, df in enumerate(dfs or []):
            if df is None or df.empty or len(df) < 2:
                continue
            rows_list = df.fillna("").values.tolist()
            header = [str(c) for c in rows_list[0]]
            rows = rows_list[1:]
            extracted = self._build_table(
                header=header,
                rows=rows,
                page_num=page_num,
                table_index=table_index_start + i,
                run_id=run_id,
                registry_id=registry_id,
                source_sha256=source_sha256,
                extractor="tabula",
            )
            result.append(extracted)
        return result

    # ------------------------------------------------------------------
    # Shared builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_table(
        header: list[str],
        rows: list[list[str]],
        page_num: int,
        table_index: int,
        run_id: str,
        registry_id: str,
        source_sha256: str,
        extractor: str,
    ) -> "ExtractedTable":
        """Build an ExtractedTable from parsed header and rows.

        Args:
            header: List of column header strings.
            rows: List of data rows (each row is a list of strings).
            page_num: 1-based page number.
            table_index: 0-based table position in document.
            run_id: Extraction run UUID4.
            registry_id: Trial identifier.
            source_sha256: Source SHA-256.
            extractor: Extractor name.

        Returns:
            ExtractedTable instance (timestamp empty — set by caller).
        """
        from .models import ExtractedTable

        clean_header = [str(c) if c is not None else "" for c in header]
        clean_rows = [
            [str(c) if c is not None else "" for c in row]
            for row in rows
        ]
        return ExtractedTable(
            run_id=run_id,
            source_registry_id=registry_id,
            source_sha256=source_sha256,
            page_number=page_num,
            extractor_used=extractor,
            table_index=table_index,
            header_row=json.dumps(clean_header),
            data_rows=json.dumps(clean_rows),
        )
