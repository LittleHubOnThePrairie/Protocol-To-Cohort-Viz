"""Universal table extraction with cell structure (PTCV-223).

Runs Camelot (Stream+Lattice) and pdfplumber ``extract_tables()``
on ALL pages of a protocol PDF, producing structured
``ExtractedTable`` objects with cell-level data. Merges results
from multiple methods, keeping the highest-coverage result per
page per table region.

This replaces the SoA-specific table discovery in Stage 2 with
a universal Stage 1 extraction that benefits ALL downstream stages.

Risk tier: MEDIUM — data pipeline component (no patient data).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
from typing import Optional

from .models import ExtractedTable

logger = logging.getLogger(__name__)

# Minimum cells for a credible table (avoids noise)
_MIN_CELLS = 6

# Continuation signal patterns (from PTCV-218)
_CONTINUATION_RE = re.compile(
    r"(?:continued|cont[''.]d|cont\.)"
    r"|table\s+\d+\s*\(",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Camelot extraction
# ---------------------------------------------------------------------------


def _camelot_extract(
    pdf_path: str,
    page_count: int,
    flavor: str,
) -> list[dict]:
    """Run Camelot on all pages in a given flavor.

    Args:
        pdf_path: Path to temporary PDF file on disk.
        page_count: Total number of pages.
        flavor: ``"stream"`` or ``"lattice"``.

    Returns:
        List of dicts with page, header, rows, extractor, cell_count.
    """
    try:
        import camelot
    except ImportError:
        logger.info("camelot-py not installed; skipping %s extraction", flavor)
        return []

    results: list[dict] = []

    for page_num in range(1, page_count + 1):
        try:
            tables = camelot.read_pdf(
                pdf_path, pages=str(page_num), flavor=flavor,
            )
        except Exception as exc:
            logger.debug(
                "Camelot %s page %d: %s", flavor, page_num, exc,
            )
            continue

        for tbl in tables:
            df = tbl.df
            if df.empty or len(df) < 2:
                continue
            cell_count = df.size
            if cell_count < _MIN_CELLS:
                continue
            header = [str(c) for c in df.iloc[0].tolist()]
            rows = [
                [str(c) for c in row] for row in df.iloc[1:].values.tolist()
            ]
            results.append({
                "page": page_num,
                "header": header,
                "rows": rows,
                "extractor": f"camelot_{flavor}",
                "cell_count": cell_count,
            })

    return results


# ---------------------------------------------------------------------------
# pdfplumber extraction
# ---------------------------------------------------------------------------


def _pdfplumber_extract(pdf_bytes: bytes, page_count: int) -> list[dict]:
    """Run pdfplumber ``extract_tables()`` on all pages.

    Uses text-based strategies to preserve cell boundaries even
    for tables without full gridlines.

    Args:
        pdf_bytes: Raw PDF bytes.
        page_count: Total number of pages.

    Returns:
        List of dicts with page, header, rows, extractor, cell_count.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.info("pdfplumber not installed; skipping extract_tables")
        return []

    results: list[dict] = []
    table_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 5,
        "join_tolerance": 5,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
    }

    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as exc:
        logger.debug("pdfplumber failed to open PDF: %s", exc)
        return results

    try:
        for page_num in range(1, page_count + 1):
            idx = page_num - 1
            if idx >= len(pdf.pages):
                continue

            page = pdf.pages[idx]
            try:
                extracted = page.extract_tables(table_settings)
            except Exception as exc:
                logger.debug(
                    "pdfplumber extract_tables page %d: %s",
                    page_num, exc,
                )
                continue

            for tbl in extracted:
                if not tbl or len(tbl) < 2:
                    continue

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

                if cell_count < _MIN_CELLS:
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


# ---------------------------------------------------------------------------
# Merge + dedup
# ---------------------------------------------------------------------------


def _merge_methods(
    *method_results: list[dict],
) -> list[dict]:
    """Merge results from multiple methods, keeping best per page.

    When multiple methods find tables on the same page, the result
    with the highest cell count wins. Tables from pages where only
    one method found results are included unconditionally.

    Args:
        method_results: Variable number of result lists.

    Returns:
        Merged list sorted by page number.
    """
    all_tables: list[dict] = []
    for results in method_results:
        all_tables.extend(results)

    if not all_tables:
        return []

    # Group by page
    by_page: dict[int, list[dict]] = {}
    for t in all_tables:
        by_page.setdefault(t["page"], []).append(t)

    merged: list[dict] = []
    for page in sorted(by_page):
        candidates = by_page[page]
        # Keep the best table per page (highest cell count)
        best = max(candidates, key=lambda t: t["cell_count"])
        merged.append(best)

    return merged


# ---------------------------------------------------------------------------
# Multi-page merge
# ---------------------------------------------------------------------------


def _header_similarity(
    header_a: list[str], header_b: list[str],
) -> float:
    """Jaccard similarity between two header rows."""
    set_a = {c.strip().lower() for c in header_a if c.strip()}
    set_b = {c.strip().lower() for c in header_b if c.strip()}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _is_continuation(prev: dict, next_tbl: dict) -> bool:
    """Detect continuation signals between tables on consecutive pages."""
    header_text = " ".join(str(c) for c in next_tbl["header"])
    if _CONTINUATION_RE.search(header_text):
        return True

    if next_tbl["rows"]:
        first_row = " ".join(str(c) for c in next_tbl["rows"][0])
        if _CONTINUATION_RE.search(first_row):
            return True

    # Same high column count on consecutive page
    if (
        len(prev["header"]) == len(next_tbl["header"])
        and len(prev["header"]) >= 5
    ):
        return True

    return False


def _merge_multi_page(
    tables: list[dict],
    min_similarity: float = 0.80,
) -> list[dict]:
    """Merge tables spanning consecutive pages.

    Detects continuation by header similarity or text signals.

    Args:
        tables: Sorted list of table dicts.
        min_similarity: Jaccard threshold for header matching.

    Returns:
        Merged list of table dicts.
    """
    if not tables:
        return []

    merged: list[dict] = [tables[0]]

    for tbl in tables[1:]:
        prev = merged[-1]
        similarity = _header_similarity(prev["header"], tbl["header"])
        continuation = _is_continuation(prev, tbl)

        if (
            tbl["page"] <= prev["page"] + 2
            and (similarity >= min_similarity or continuation)
        ):
            # Skip repeated header row on continuation pages
            rows_to_add = tbl["rows"]
            if (
                continuation
                and rows_to_add
                and _header_similarity(prev["header"], rows_to_add[0]) >= 0.90
            ):
                rows_to_add = rows_to_add[1:]

            prev["rows"].extend(rows_to_add)
            prev["cell_count"] = (
                len(prev["header"]) * (len(prev["rows"]) + 1)
            )
        else:
            merged.append(tbl)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_all_tables(
    pdf_bytes: bytes,
    page_count: int,
    run_id: str,
    registry_id: str,
    source_sha256: str,
    existing_tables: Optional[list[ExtractedTable]] = None,
    enable_camelot: bool = False,
) -> list[ExtractedTable]:
    """Extract ALL tables from a PDF.

    Uses pdfplumber ``extract_tables()`` as the primary method
    (fast, in-memory). Optionally runs Camelot Stream+Lattice
    when ``enable_camelot=True`` (slower but handles bordered
    tables better).

    PTCV-244: pdfplumber is primary, Camelot is opt-in.

    Tables already found by pymupdf4llm (passed as
    ``existing_tables``) are included; pages where they have higher
    cell counts than pdfplumber results are kept.

    Args:
        pdf_bytes: Raw PDF file bytes.
        page_count: Total page count.
        run_id: Extraction run UUID4.
        registry_id: Trial identifier.
        source_sha256: Source file SHA-256.
        existing_tables: Tables already extracted by pymupdf4llm
            (from ``_parse_markdown_tables``). Merged with new
            discoveries.
        enable_camelot: If True, also run Camelot Stream+Lattice
            alongside pdfplumber (slower, better for bordered tables).
            Default False (PTCV-244).

    Returns:
        List of ExtractedTable with cell structure, sorted by
        page_number then table_index.
    """
    # Primary: pdfplumber extract_tables (fast, in-memory)
    plumber = _pdfplumber_extract(pdf_bytes, page_count)

    # Optional: Camelot Stream+Lattice (slow, requires temp file)
    stream: list[dict] = []
    lattice: list[dict] = []
    if enable_camelot:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False,
        ) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            stream = _camelot_extract(tmp_path, page_count, "stream")
            lattice = _camelot_extract(tmp_path, page_count, "lattice")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Convert existing pymupdf4llm tables to dict format for merge
    existing_dicts: list[dict] = []
    if existing_tables:
        for et in existing_tables:
            try:
                header = json.loads(et.header_row)
                rows = json.loads(et.data_rows)
            except (json.JSONDecodeError, TypeError):
                continue
            cell_count = len(header) * (len(rows) + 1) if header else 0
            if cell_count < _MIN_CELLS:
                continue
            existing_dicts.append({
                "page": et.page_number,
                "header": header,
                "rows": rows,
                "extractor": et.extractor_used,
                "cell_count": cell_count,
            })

    # Merge all methods: best per page
    merged = _merge_methods(
        stream, lattice, plumber, existing_dicts,
    )

    # Sort by page, then merge multi-page tables
    merged.sort(key=lambda t: t["page"])
    merged = _merge_multi_page(merged)

    # Convert to ExtractedTable objects
    result: list[ExtractedTable] = []
    for idx, tbl in enumerate(merged):
        result.append(ExtractedTable(
            run_id=run_id,
            source_registry_id=registry_id,
            source_sha256=source_sha256,
            page_number=tbl["page"],
            extractor_used=tbl["extractor"],
            table_index=idx,
            header_row=json.dumps(tbl["header"]),
            data_rows=json.dumps(tbl["rows"]),
        ))

    logger.info(
        "Universal table extraction: %d tables from %d pages "
        "(stream=%d, lattice=%d, plumber=%d, existing=%d) for %s",
        len(result), page_count,
        len(stream), len(lattice), len(plumber), len(existing_dicts),
        registry_id,
    )

    return result


__all__ = [
    "extract_all_tables",
]
