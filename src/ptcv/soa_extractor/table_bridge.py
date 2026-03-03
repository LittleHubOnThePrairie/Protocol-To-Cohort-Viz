"""Bridge pre-extracted tables to SoA pipeline (PTCV-51).

Converts ``ExtractedTable`` objects (from ``tables.parquet``) into
``RawSoaTable`` instances consumable by the USDM mapper, bypassing
the text-based ``SoaTableParser`` which cannot find tables in
flattened ICH section text.

Risk tier: LOW — data transformation only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from .models import RawSoaTable

if TYPE_CHECKING:
    from ..extraction.models import ExtractedTable

logger = logging.getLogger(__name__)

# Visit-indicative column header patterns
_VISIT_RE = re.compile(
    r"\bvisit\b"
    r"|\bday\s*-?\d+"
    r"|\bweek\s*-?\d+"
    r"|\bmonth\s*\d+"
    r"|\bscreening\b"
    r"|\bbaseline\b"
    r"|\bfollow[\s-]*up\b"
    r"|\bdischarge\b"
    r"|\badmission\b"
    r"|\brandomiz"
    r"|\bend\s+of\s+(?:study|treatment|trial)\b"
    r"|\beot\b"
    r"|\beos\b",
    re.IGNORECASE,
)

# Activity cell markers (same as SoaTableParser)
_MARKED_RE = re.compile(r"^[Xx✓•Y1]$|^[Yy]es$", re.IGNORECASE)

# Day/window row heuristic (same as SoaTableParser)
_DAY_ROW_RE = re.compile(
    r"[Dd]ay\s*-?\d+|[Ww]eek\s*\d+|[-]\d|\d+\s*[±]",
)

# Minimum thresholds for SoA identification
_MIN_COLUMNS = 5
_MIN_VISIT_MATCHES = 3
_MIN_DATA_ROWS = 3


def is_soa_table(header: list[str]) -> bool:
    """Check if a table header row indicates a Schedule of Activities.

    Args:
        header: List of column header strings.

    Returns:
        True if the table has enough columns and visit-pattern matches.
    """
    if len(header) < _MIN_COLUMNS:
        return False
    visit_matches = sum(
        1 for h in header if _VISIT_RE.search(_clean_cell(h))
    )
    return visit_matches >= _MIN_VISIT_MATCHES


def extracted_table_to_raw_soa(
    table: "ExtractedTable",
) -> RawSoaTable | None:
    """Convert an ExtractedTable to a RawSoaTable.

    Args:
        table: An ExtractedTable with JSON-serialised header_row and
            data_rows.

    Returns:
        RawSoaTable if the table looks like a Schedule of Activities,
        None otherwise.
    """
    try:
        header = json.loads(table.header_row)
        data = json.loads(table.data_rows)
    except (json.JSONDecodeError, TypeError):
        logger.debug(
            "Skipping table %d: invalid JSON in header/data",
            table.table_index,
        )
        return None

    header = [_clean_cell(h) for h in header]

    if not is_soa_table(header):
        return None

    if len(data) < _MIN_DATA_ROWS:
        return None

    # Check if first data row is a day/window sub-header
    day_headers: list[str] = []
    activity_start = 0
    if data:
        first_data = [_clean_cell(c) for c in data[0]]
        is_day_row = any(
            _DAY_ROW_RE.search(cell) for cell in first_data[1:]
        )
        if is_day_row:
            day_headers = first_data[1:]
            activity_start = 1

    # Build activities: first column is activity name, rest are flags
    activities: list[tuple[str, list[bool]]] = []
    n_visits = len(header) - 1  # exclude first column (activity label)

    for row in data[activity_start:]:
        if not row or not _clean_cell(row[0]):
            continue
        activity_name = _clean_cell(row[0])
        scheduled = [
            _MARKED_RE.match(_clean_cell(cell)) is not None
            for cell in row[1:]
        ]
        # Pad or truncate to match visit count
        while len(scheduled) < n_visits:
            scheduled.append(False)
        activities.append((activity_name, scheduled[:n_visits]))

    if not activities:
        return None

    visit_headers = header[1:]

    logger.info(
        "Converted extracted table (page %d, index %d) to SoA: "
        "%d visits, %d activities",
        table.page_number,
        table.table_index,
        len(visit_headers),
        len(activities),
    )

    return RawSoaTable(
        visit_headers=visit_headers,
        day_headers=day_headers,
        activities=activities,
        section_code="tables.parquet",
    )


def filter_soa_tables(
    tables: list["ExtractedTable"],
) -> list[RawSoaTable]:
    """Filter and convert a list of ExtractedTables to RawSoaTables.

    Args:
        tables: List of ExtractedTable objects (e.g. from tables.parquet).

    Returns:
        List of RawSoaTable for tables identified as SoA candidates.
    """
    results: list[RawSoaTable] = []
    for table in tables:
        raw = extracted_table_to_raw_soa(table)
        if raw is not None:
            results.append(raw)

    if results:
        logger.info(
            "Found %d SoA candidate(s) from %d pre-extracted tables",
            len(results),
            len(tables),
        )
    return results


def _clean_cell(value: object) -> str:
    """Normalise a cell value to a string, collapsing whitespace."""
    if value is None:
        return ""
    text = str(value).strip()
    # Collapse internal newlines and excessive whitespace
    text = re.sub(r"\s+", " ", text)
    return text
