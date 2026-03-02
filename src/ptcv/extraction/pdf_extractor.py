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


class PdfExtractor:
    """Extracts text blocks and tables from PDF bytes.

    Attributes:
        min_words_for_text_block: Minimum word count to keep a text
            block (filters noise). Default 3.
    """

    def __init__(self, min_words_for_text_block: int = 3) -> None:
        self._min_words = min_words_for_text_block

    def extract(
        self,
        pdf_bytes: bytes,
        run_id: str,
        registry_id: str,
        source_sha256: str,
    ) -> tuple[list["TextBlock"], list["ExtractedTable"], int]:
        """Extract text blocks and tables from a PDF.

        Uses pdfplumber as the primary extractor; falls back to Camelot
        Lattice and then tabula-py on a per-page basis for tables.

        Args:
            pdf_bytes: Raw PDF file contents.
            run_id: UUID4 for this extraction run (used in all rows).
            registry_id: Trial identifier (e.g. "NCT00112827").
            source_sha256: SHA-256 of the source PDF file.

        Returns:
            Tuple of (text_blocks, tables, page_count).
        """
        import io

        import pdfplumber

        text_blocks: list[TextBlock] = []
        raw_tables_by_page: list[list[list[list[str]]]] = []
        page_count = 0

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                # ---- Text extraction --------------------------------
                blocks = self._extract_text_blocks(
                    page, page_num, run_id, registry_id, source_sha256
                )
                text_blocks.extend(blocks)

                # ---- Table extraction per page ----------------------
                page_tables = self._extract_tables_pdfplumber(page)
                raw_tables_by_page.append(page_tables)

        # Reconstruct multi-page tables
        merged_pages = self._reconstruct_multi_page_tables(raw_tables_by_page)

        tables: list[ExtractedTable] = []
        table_index = 0
        for page_num, page_tables in enumerate(merged_pages, start=1):
            for table_data in page_tables:
                if not table_data or len(table_data) < 2:
                    continue
                header = table_data[0]
                rows = table_data[1:]
                # Fallback to Camelot / tabula if pdfplumber gave 0 data rows
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

        # For pages where pdfplumber found no tables, try Camelot then tabula
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

        return text_blocks, tables, page_count

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
            blocks.append(
                TextBlock(
                    run_id=run_id,
                    source_registry_id=registry_id,
                    source_sha256=source_sha256,
                    page_number=page_num,
                    block_index=block_index,
                    text=stripped,
                    block_type=block_type,
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
