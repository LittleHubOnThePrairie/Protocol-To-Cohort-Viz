"""Blended table discovery for non-standard SoA formats (PTCV-38).

Scans extracted text blocks for schedule-like headings, identifies
candidate pages, and runs a two-phase extraction cascade:

  Phase 1 — Camelot Stream: fast first pass for bordered/semi-bordered
      tables.
  Phase 2 — Table Transformer (TATR): high-accuracy fallback for pages
      where Camelot finds zero tables but schedule headings exist.

Multi-page table merging detects continuation pages by fuzzy header
similarity and concatenates rows across page boundaries.

Risk tier: MEDIUM — data pipeline component (no patient data).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .models import RawSoaTable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule heading detection
# ---------------------------------------------------------------------------

_SCHEDULE_HEADING_RE = re.compile(
    r"schedule\s+of\s+(?:activities|assessments|visits)"
    r"|visit\s+schedule"
    r"|assessment\s+schedule"
    r"|(?:study|trial)\s+(?:procedures|calendar|timeline)"
    r"|treatment\s+calendar"
    r"|assessment\s+calendar"
    r"|study\s+calendar",
    re.IGNORECASE,
)

# Activity cell markers — same as parser.py
_MARKED_RE = re.compile(r"^[Xx✓•Y1]$|^[Yy]es$", re.IGNORECASE)

# Minimum cells for a credible table
_MIN_TABLE_CELLS = 6


class TableDiscovery:
    """Blended Camelot Stream + Table Transformer table discovery.

    Args:
        min_header_similarity: Threshold (0–1) for fuzzy header match
            when merging multi-page tables. Default 0.80.
    """

    def __init__(self, min_header_similarity: float = 0.80) -> None:
        self._min_header_similarity = min_header_similarity

    def discover(
        self,
        pdf_bytes: bytes,
        text_blocks: list[dict],
        page_count: int,
    ) -> list[RawSoaTable]:
        """Discover SoA tables from PDF using blended extraction.

        Args:
            pdf_bytes: Raw PDF file bytes.
            text_blocks: List of dicts with keys 'page_number' and
                'text' from the extraction stage.
            page_count: Total number of pages in the PDF.

        Returns:
            List of RawSoaTable objects discovered. May be empty.
        """
        candidate_pages = self._find_candidate_pages(text_blocks)
        if not candidate_pages:
            return []

        logger.info(
            "Table discovery: %d candidate pages: %s",
            len(candidate_pages),
            candidate_pages,
        )

        # Phase 1: Camelot Stream on candidate pages
        raw_tables = self._camelot_stream_extract(
            pdf_bytes, candidate_pages
        )

        # Phase 2: Table Transformer fallback for pages with no results
        pages_with_tables = {t["page"] for t in raw_tables}
        missing_pages = [
            p for p in candidate_pages if p not in pages_with_tables
        ]

        if missing_pages:
            tatr_tables = self._tatr_extract(pdf_bytes, missing_pages)
            raw_tables.extend(tatr_tables)

        if not raw_tables:
            return []

        # Sort by page number for merge
        raw_tables.sort(key=lambda t: t["page"])

        # Phase 3: Multi-page table merging
        merged = self._merge_multi_page(raw_tables)

        # Phase 4: Select best result per page group
        return self._to_raw_soa_tables(merged)

    # ------------------------------------------------------------------
    # Candidate page detection
    # ------------------------------------------------------------------

    def _find_candidate_pages(
        self, text_blocks: list[dict]
    ) -> list[int]:
        """Find pages containing schedule-like headings.

        Args:
            text_blocks: Extracted text blocks with page_number and text.

        Returns:
            Sorted list of unique candidate page numbers.
        """
        candidate_pages: set[int] = set()
        for block in text_blocks:
            text = block.get("text", "")
            page = block.get("page_number", 0)
            if _SCHEDULE_HEADING_RE.search(text):
                candidate_pages.add(page)
                # Also include next two pages (table may span pages)
                candidate_pages.add(page + 1)
                candidate_pages.add(page + 2)
        return sorted(candidate_pages)

    # ------------------------------------------------------------------
    # Phase 1: Camelot Stream extraction
    # ------------------------------------------------------------------

    def _camelot_stream_extract(
        self, pdf_bytes: bytes, pages: list[int]
    ) -> list[dict]:
        """Run Camelot Stream mode on specified pages.

        Args:
            pdf_bytes: Raw PDF bytes.
            pages: 1-based page numbers to scan.

        Returns:
            List of dicts with keys: page, header, rows, extractor.
        """
        import io
        import tempfile

        results: list[dict] = []

        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            import camelot

            for page_num in pages:
                try:
                    tables = camelot.read_pdf(
                        tmp_path,
                        pages=str(page_num),
                        flavor="stream",
                    )
                except Exception as exc:
                    logger.debug(
                        "Camelot stream failed on page %d: %s",
                        page_num, exc,
                    )
                    continue

                for tbl in tables:
                    df = tbl.df
                    if df.empty or len(df) < 2:
                        continue
                    cell_count = df.size
                    if cell_count < _MIN_TABLE_CELLS:
                        continue
                    header = df.iloc[0].tolist()
                    rows = df.iloc[1:].values.tolist()
                    results.append({
                        "page": page_num,
                        "header": [str(c) for c in header],
                        "rows": [
                            [str(c) for c in row] for row in rows
                        ],
                        "extractor": "camelot_stream",
                        "cell_count": cell_count,
                    })
        except ImportError:
            logger.warning("camelot-py not installed; skipping stream extraction")
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return results

    # ------------------------------------------------------------------
    # Phase 2: Table Transformer fallback
    # ------------------------------------------------------------------

    def _tatr_extract(
        self, pdf_bytes: bytes, pages: list[int]
    ) -> list[dict]:
        """Run Table Transformer on pages where Camelot found nothing.

        Table Transformer is an optional dependency. If not installed,
        this method returns an empty list without raising an error.

        Args:
            pdf_bytes: Raw PDF bytes.
            pages: 1-based page numbers to scan.

        Returns:
            List of dicts with keys: page, header, rows, extractor.
        """
        try:
            from table_transformer import TableTransformerDetector
        except ImportError:
            logger.info(
                "table-transformer not installed; skipping TATR fallback"
            )
            return []

        results: list[dict] = []
        try:
            detector = TableTransformerDetector()
            for page_num in pages:
                try:
                    detected = detector.detect_tables(
                        pdf_bytes, page_num
                    )
                    for tbl in detected:
                        header = tbl.get("header", [])
                        rows = tbl.get("rows", [])
                        if len(header) < 2 or len(rows) < 1:
                            continue
                        cell_count = len(header) * (len(rows) + 1)
                        results.append({
                            "page": page_num,
                            "header": [str(c) for c in header],
                            "rows": [
                                [str(c) for c in row]
                                for row in rows
                            ],
                            "extractor": "table_transformer",
                            "cell_count": cell_count,
                        })
                except Exception as exc:
                    logger.debug(
                        "TATR failed on page %d: %s", page_num, exc
                    )
        except Exception as exc:
            logger.warning("TATR initialisation failed: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Phase 3: Multi-page table merging
    # ------------------------------------------------------------------

    def _merge_multi_page(
        self, tables: list[dict]
    ) -> list[dict]:
        """Merge tables spanning consecutive pages.

        Detects continuation pages by fuzzy header similarity.
        When the first row of a subsequent page's table matches
        the header of an earlier table (>= threshold), the rows
        are appended to the earlier table.

        Args:
            tables: Sorted list of extracted table dicts.

        Returns:
            List of merged table dicts.
        """
        if not tables:
            return []

        merged: list[dict] = [tables[0]]

        for tbl in tables[1:]:
            prev = merged[-1]
            similarity = self._header_similarity(
                prev["header"], tbl["header"]
            )
            # Merge if: consecutive page, similar headers, same extractor
            if (
                similarity >= self._min_header_similarity
                and tbl["page"] <= prev["page"] + 2
            ):
                prev["rows"].extend(tbl["rows"])
                prev["cell_count"] = (
                    len(prev["header"])
                    * (len(prev["rows"]) + 1)
                )
                logger.info(
                    "Merged page %d into table starting at page %d "
                    "(header similarity=%.2f)",
                    tbl["page"], prev["page"], similarity,
                )
            else:
                merged.append(tbl)

        return merged

    @staticmethod
    def _header_similarity(
        header_a: list[str], header_b: list[str]
    ) -> float:
        """Compute Jaccard similarity between two header rows.

        Args:
            header_a: First header row (list of cell strings).
            header_b: Second header row (list of cell strings).

        Returns:
            Similarity score 0.0–1.0.
        """
        set_a = {c.strip().lower() for c in header_a if c.strip()}
        set_b = {c.strip().lower() for c in header_b if c.strip()}
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Phase 4: Convert to RawSoaTable
    # ------------------------------------------------------------------

    def _to_raw_soa_tables(
        self, tables: list[dict]
    ) -> list[RawSoaTable]:
        """Convert extracted table dicts to RawSoaTable objects.

        Applies activity marker detection and day-row heuristics
        consistent with SoaTableParser.

        Args:
            tables: Merged table dicts.

        Returns:
            List of RawSoaTable objects.
        """
        results: list[RawSoaTable] = []

        for tbl in tables:
            header = tbl["header"]
            rows = tbl["rows"]

            if len(header) < 2 or not rows:
                continue

            # Strip the first column header (Activity / Assessment)
            visit_headers = header[1:]

            # Check if first data row is a day/window row
            day_headers: list[str] = []
            activity_start = 0
            if rows:
                first_data = rows[0]
                is_day_row = any(
                    bool(re.search(
                        r"[Dd]ay\s*-?\d+|[Ww]eek\s*\d+|[-]\d|\d+\s*[±]",
                        str(cell),
                    ))
                    for cell in first_data[1:]
                    if cell
                )
                if is_day_row:
                    day_headers = [str(c) for c in first_data[1:]]
                    activity_start = 1

            activities: list[tuple[str, list[bool]]] = []
            for row in rows[activity_start:]:
                if not row or not str(row[0]).strip():
                    continue
                activity_name = str(row[0]).strip()
                scheduled = [
                    _MARKED_RE.match(str(cell).strip()) is not None
                    for cell in row[1:]
                ]
                # Pad to match visit header count
                n = len(visit_headers)
                while len(scheduled) < n:
                    scheduled.append(False)
                activities.append((activity_name, scheduled[:n]))

            if not activities:
                continue

            results.append(RawSoaTable(
                visit_headers=visit_headers,
                day_headers=day_headers,
                activities=activities,
                section_code="B.4",
            ))

        return results
