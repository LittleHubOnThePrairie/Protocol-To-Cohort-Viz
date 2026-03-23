"""Blended table discovery for non-standard SoA formats (PTCV-38, PTCV-215).

Scans extracted text blocks for schedule-like headings, identifies
candidate pages, and runs a multi-phase extraction cascade:

  Phase 1a — Camelot Stream: fast first pass for borderless/semi-bordered
      tables.
  Phase 1b — Camelot Lattice: captures well-bordered tables that Stream
      may miss due to ruled-line detection (PTCV-215).
  Phase 2  — Table Transformer (TATR): high-accuracy fallback for pages
      where Camelot finds zero tables but schedule headings exist.

When both Stream and Lattice produce results for the same page, the
result with more cells is kept (higher coverage wins).

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
    """Blended table discovery for SoA tables.

    Uses pdfplumber as primary extractor (fast). Camelot Stream+Lattice
    available as opt-in for bordered tables (PTCV-244).

    Args:
        min_header_similarity: Threshold (0–1) for fuzzy header match
            when merging multi-page tables. Default 0.80.
        enable_lattice: Enable Camelot Lattice mode alongside Stream.
            Default False (PTCV-244: Camelot opt-in).
    """

    def __init__(
        self,
        min_header_similarity: float = 0.80,
        enable_lattice: bool = False,
        enable_camelot: bool = False,
    ) -> None:
        self._min_header_similarity = min_header_similarity
        self._enable_lattice = enable_lattice
        self._enable_camelot = enable_camelot

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

        # Phase 1a: pdfplumber extract_tables() — primary (fast)
        raw_tables = self._pdfplumber_table_extract(
            pdf_bytes, candidate_pages
        )

        # Phase 1b: Camelot Stream+Lattice — opt-in (slow, PTCV-244)
        if self._enable_camelot:
            stream_tables = self._camelot_stream_extract(
                pdf_bytes, candidate_pages
            )

            lattice_tables: list[dict] = []
            if self._enable_lattice:
                lattice_tables = self._camelot_lattice_extract(
                    pdf_bytes, candidate_pages
                )

            camelot_tables = self._merge_camelot_modes(
                stream_tables, lattice_tables,
            )

            # Merge Camelot with pdfplumber: keep best per page
            raw_tables = self._merge_camelot_modes(
                raw_tables, camelot_tables,
            )

        # Phase 2: Table Transformer fallback for pages still with no results
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

        # Phase 4: Convert to RawSoaTable and reconcile
        raw_soa = self._to_raw_soa_tables(merged)

        # PTCV-265: Reconcile when multiple RawSoaTable objects exist.
        # Multiple tables arise from multi-page merging or overlapping
        # extraction methods. The reconciler merges by assessment name
        # union + majority voting per cell.
        if len(raw_soa) > 1:
            try:
                from .reconciler import reconcile

                report = reconcile(raw_soa)
                if report.merged_table and report.total_activities > 0:
                    logger.info(
                        "PTCV-265: Reconciled %d tables → %d activities "
                        "(%d overlapping, %d unique, %d disagreements)",
                        report.source_count,
                        report.total_activities,
                        report.overlapping_activities,
                        report.unique_to_one,
                        report.disagreement_count,
                    )
                    return [report.merged_table]
            except Exception:
                logger.debug(
                    "Reconciler failed; returning unreconciled tables",
                    exc_info=True,
                )

        return raw_soa

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
    # Phase 1b: Camelot Lattice extraction (PTCV-215)
    # ------------------------------------------------------------------

    def _camelot_lattice_extract(
        self, pdf_bytes: bytes, pages: list[int]
    ) -> list[dict]:
        """Run Camelot Lattice mode on specified pages.

        Lattice mode detects tables using ruled lines (borders).
        Works best for well-bordered tables but may miss tables
        with partial or no gridlines. Complements Stream mode.

        Args:
            pdf_bytes: Raw PDF bytes.
            pages: 1-based page numbers to scan.

        Returns:
            List of dicts with keys: page, header, rows, extractor.
        """
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
                        flavor="lattice",
                    )
                except Exception as exc:
                    logger.debug(
                        "Camelot lattice failed on page %d: %s",
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
                        "extractor": "camelot_lattice",
                        "cell_count": cell_count,
                    })
        except ImportError:
            logger.warning(
                "camelot-py not installed; skipping lattice extraction"
            )
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return results

    # ------------------------------------------------------------------
    # Merge Stream + Lattice results (PTCV-215)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_camelot_modes(
        stream_tables: list[dict],
        lattice_tables: list[dict],
    ) -> list[dict]:
        """Merge Stream and Lattice results, keeping best per page.

        When both modes find tables on the same page, the result
        with higher cell count is kept (more cells = higher coverage).
        Tables from pages where only one mode produced results are
        included unconditionally.

        Args:
            stream_tables: Results from Camelot Stream.
            lattice_tables: Results from Camelot Lattice.

        Returns:
            Merged list of table dicts.
        """
        if not lattice_tables:
            return stream_tables
        if not stream_tables:
            return lattice_tables

        # Group by page, pick best per page
        by_page: dict[int, list[dict]] = {}
        for t in stream_tables + lattice_tables:
            by_page.setdefault(t["page"], []).append(t)

        merged: list[dict] = []
        for page in sorted(by_page):
            candidates = by_page[page]
            best = max(candidates, key=lambda t: t["cell_count"])
            merged.append(best)
            if len(candidates) > 1:
                logger.info(
                    "Page %d: kept %s (%d cells) over %s (%d cells)",
                    page,
                    best["extractor"],
                    best["cell_count"],
                    [c for c in candidates if c is not best][0]["extractor"],
                    [c for c in candidates if c is not best][0]["cell_count"],
                )

        return merged

    # ------------------------------------------------------------------
    # Phase 1c: pdfplumber extract_table() (PTCV-215)
    # ------------------------------------------------------------------

    def _pdfplumber_table_extract(
        self, pdf_bytes: bytes, pages: list[int]
    ) -> list[dict]:
        """Run pdfplumber's cell-boundary-aware table extraction.

        Uses ``page.extract_table()`` with text-based strategies
        instead of ``page.extract_text()``, preserving cell boundaries
        even for tables without full gridlines.

        Args:
            pdf_bytes: Raw PDF bytes.
            pages: 1-based page numbers to scan.

        Returns:
            List of dicts with keys: page, header, rows, extractor.
        """
        import io

        results: list[dict] = []

        try:
            import pdfplumber
        except ImportError:
            logger.warning(
                "pdfplumber not installed; skipping extract_table path"
            )
            return results

        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        except Exception as exc:
            logger.debug("pdfplumber failed to open PDF: %s", exc)
            return results

        table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 5,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
        }

        try:
            for page_num in pages:
                idx = page_num - 1  # pdfplumber uses 0-based
                if idx < 0 or idx >= len(pdf.pages):
                    continue

                page = pdf.pages[idx]

                try:
                    extracted = page.extract_tables(table_settings)
                except Exception as exc:
                    logger.debug(
                        "pdfplumber extract_tables failed on page %d: %s",
                        page_num, exc,
                    )
                    continue

                for tbl in extracted:
                    if not tbl or len(tbl) < 2:
                        continue

                    # Filter empty rows
                    non_empty = [
                        row for row in tbl
                        if any(cell and str(cell).strip() for cell in row)
                    ]
                    if len(non_empty) < 2:
                        continue

                    header = [str(c or "").strip() for c in non_empty[0]]
                    rows = [
                        [str(c or "").strip() for c in row]
                        for row in non_empty[1:]
                    ]
                    cell_count = len(header) * (len(rows) + 1)

                    if cell_count < _MIN_TABLE_CELLS:
                        continue

                    results.append({
                        "page": page_num,
                        "header": header,
                        "rows": rows,
                        "extractor": "pdfplumber_table",
                        "cell_count": cell_count,
                    })
        finally:
            pdf.close()

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

        Detects continuation pages by:
        1. Fuzzy header similarity (Jaccard >= threshold)
        2. Continuation text signals (PTCV-218): "continued",
           "(cont'd)", repeated header row, same column count

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
            is_continuation = self._is_continuation_signal(
                prev, tbl,
            )

            # Merge if: consecutive pages AND (similar headers OR
            # continuation signal detected)
            if (
                tbl["page"] <= prev["page"] + 2
                and (
                    similarity >= self._min_header_similarity
                    or is_continuation
                )
            ):
                # Skip repeated header row on continuation pages
                rows_to_add = tbl["rows"]
                if (
                    is_continuation
                    and rows_to_add
                    and self._header_similarity(
                        prev["header"],
                        rows_to_add[0],
                    ) >= 0.90
                ):
                    rows_to_add = rows_to_add[1:]

                prev["rows"].extend(rows_to_add)
                prev["cell_count"] = (
                    len(prev["header"])
                    * (len(prev["rows"]) + 1)
                )
                merge_reason = (
                    f"header_sim={similarity:.2f}"
                    if similarity >= self._min_header_similarity
                    else f"continuation_signal"
                )
                logger.info(
                    "Merged page %d into table starting at page %d "
                    "(%s)",
                    tbl["page"], prev["page"], merge_reason,
                )
            else:
                merged.append(tbl)

        return merged

    @staticmethod
    def _is_continuation_signal(
        prev_table: dict,
        next_table: dict,
    ) -> bool:
        """Detect table continuation signals (PTCV-218).

        Checks for:
        - "continued", "(cont'd)", "(cont.)" in first row text
        - Same column count as previous table
        - Repeated header row (first data row matches prev header)

        Args:
            prev_table: Previous table dict.
            next_table: Next table dict to check.

        Returns:
            True if continuation signals are detected.
        """
        # Check first row for continuation text
        continuation_re = re.compile(
            r"(?:continued|cont[''.]d|cont\.)"
            r"|table\s+\d+\s*\(",
            re.IGNORECASE,
        )

        # Check header for continuation text
        header_text = " ".join(str(c) for c in next_table["header"])
        if continuation_re.search(header_text):
            return True

        # Check first data row for continuation text
        if next_table["rows"]:
            first_row_text = " ".join(
                str(c) for c in next_table["rows"][0]
            )
            if continuation_re.search(first_row_text):
                return True

        # Same column count is a soft signal (combined with consecutive page)
        if (
            len(prev_table["header"]) == len(next_table["header"])
            and len(prev_table["header"]) >= 5
        ):
            # High column count match on consecutive page is a strong signal
            return True

        return False

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
                # PTCV-266: Separate footnote markers from names
                from .footnote_parser import parse_assessment_name
                _parsed = parse_assessment_name(str(row[0]).strip())
                activity_name = _parsed.clean_name
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
