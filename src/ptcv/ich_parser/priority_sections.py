"""Targeted extraction for ICH E6(R3) priority sections (PTCV-52).

Enhances extraction for B.4, B.5, B.10, and B.14 sections by:
  1. Replacing truncated text_excerpt with full section text
  2. Mapping tables from tables.parquet to sections by page range
  3. Flagging SoA-candidate tables for downstream pipeline

Risk tier: LOW — data enrichment only.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction.models import ExtractedTable, TextBlock

from .models import IchSection
from .schema_loader import (
    get_priority_sections,
    get_soa_min_columns,
    get_soa_min_visit_matches,
    get_soa_pattern,
)

logger = logging.getLogger(__name__)

# Loaded from YAML configuration (PTCV-69).
PRIORITY_SECTIONS: frozenset[str] = get_priority_sections()
_VISIT_RE: re.Pattern[str] = get_soa_pattern()
_MIN_SOA_COLUMNS: int = get_soa_min_columns()
_MIN_SOA_VISIT_MATCHES: int = get_soa_min_visit_matches()


@dataclasses.dataclass
class PrioritySectionResult:
    """Enhanced extraction result for a priority ICH section.

    Attributes:
        section_code: ICH section code (e.g. "B.4").
        section_name: Human-readable name.
        full_text: Complete section text without truncation.
        tables: All ExtractedTable objects whose pages fall within
            this section's page range.
        soa_candidate_tables: Subset of tables with visit-indicative
            column headers.
        cross_references: List of cross-reference strings found in text
            (e.g. "see Section B.4").
        page_range: Tuple of (start_page, end_page) inclusive.
    """

    section_code: str
    section_name: str
    full_text: str
    tables: list
    soa_candidate_tables: list
    cross_references: list[str]
    page_range: tuple[int, int]


def _map_sections_to_pages(
    sections: list[IchSection],
    text_blocks: list["TextBlock"],
) -> dict[str, tuple[int, int]]:
    """Map each classified section to a (start_page, end_page) range.

    Uses heading text matching: finds the text_excerpt prefix in the
    original text blocks to locate the starting page, then uses the
    next section's start as the boundary.

    Args:
        sections: Classified IchSection list.
        text_blocks: Original TextBlock list with page_number.

    Returns:
        Dict mapping section_code to (start_page, end_page) inclusive.
    """
    if not text_blocks:
        return {}

    sorted_blocks = sorted(
        text_blocks, key=lambda b: (b.page_number, b.block_index)
    )
    max_page = max(b.page_number for b in sorted_blocks)

    # Build a lookup: first 80 chars of each block → page_number
    block_page_index: list[tuple[str, int]] = []
    for b in sorted_blocks:
        prefix = b.text.strip()[:80].lower()
        if prefix:
            block_page_index.append((prefix, b.page_number))

    # For each section, find its start page by matching content_json text
    section_starts: list[tuple[str, int]] = []
    for sec in sections:
        try:
            content = json.loads(sec.content_json)
        except (json.JSONDecodeError, TypeError):
            continue

        excerpt = content.get("text_excerpt", "")
        if not excerpt:
            continue

        # Match the first 80 chars of the excerpt against text blocks
        search_prefix = excerpt.strip()[:80].lower()
        start_page = _find_page_for_text(search_prefix, block_page_index)
        if start_page > 0:
            section_starts.append((sec.section_code, start_page))

    if not section_starts:
        return {}

    # Sort by start page to determine boundaries
    section_starts.sort(key=lambda x: x[1])

    page_ranges: dict[str, tuple[int, int]] = {}
    for i, (code, start) in enumerate(section_starts):
        if i + 1 < len(section_starts):
            end = section_starts[i + 1][1] - 1
            end = max(end, start)
        else:
            end = max_page
        page_ranges[code] = (start, end)

    return page_ranges


def _find_page_for_text(
    search_prefix: str,
    block_index: list[tuple[str, int]],
) -> int:
    """Find the page number for a text prefix using substring matching.

    Returns 0 if no match found.
    """
    for block_prefix, page in block_index:
        if search_prefix[:40] in block_prefix or block_prefix in search_prefix[:80]:
            return page
    return 0


def _full_text_for_pages(
    text_blocks: list["TextBlock"],
    start_page: int,
    end_page: int,
) -> str:
    """Join all text blocks within a page range into full text.

    Args:
        text_blocks: Original TextBlock list.
        start_page: Start page (inclusive).
        end_page: End page (inclusive).

    Returns:
        Joined text from all blocks in the page range.
    """
    filtered = [
        b for b in text_blocks
        if start_page <= b.page_number <= end_page
    ]
    filtered.sort(key=lambda b: (b.page_number, b.block_index))
    return "\n".join(b.text for b in filtered if b.text.strip())


def _tables_for_pages(
    tables: list["ExtractedTable"],
    start_page: int,
    end_page: int,
) -> list["ExtractedTable"]:
    """Filter tables to those whose page falls within the range."""
    return [
        t for t in tables
        if start_page <= t.page_number <= end_page
    ]


def _is_soa_candidate(table: "ExtractedTable") -> bool:
    """Check if a table has SoA-indicative column headers."""
    try:
        header = json.loads(table.header_row)
    except (json.JSONDecodeError, TypeError):
        return False

    if len(header) < _MIN_SOA_COLUMNS:
        return False

    visit_matches = sum(
        1 for h in header
        if _VISIT_RE.search(re.sub(r"\s+", " ", str(h).strip()))
    )
    return visit_matches >= _MIN_SOA_VISIT_MATCHES


def _find_cross_references(text: str) -> list[str]:
    """Extract cross-reference mentions from section text."""
    pattern = re.compile(
        r"(?:see|refer\s+to|as\s+(?:described|defined)\s+in)"
        r"\s+(?:Section\s+)?B\.\d+",
        re.IGNORECASE,
    )
    return [m.group(0) for m in pattern.finditer(text)]


def extract_priority_sections(
    sections: list[IchSection],
    text_blocks: list["TextBlock"],
    extracted_tables: list["ExtractedTable"] | None = None,
) -> list[PrioritySectionResult]:
    """Extract enhanced data for priority ICH sections.

    For B.4, B.5, B.10, and B.14 sections:
    - Replaces truncated text_excerpt with full un-truncated text
    - Associates tables from tables.parquet by page range
    - Flags SoA candidate tables

    Non-priority sections are not included in the output.

    Args:
        sections: Classified IchSection list.
        text_blocks: Original TextBlock list from extraction.
        extracted_tables: Optional ExtractedTable list from tables.parquet.

    Returns:
        List of PrioritySectionResult for found priority sections.
    """
    if extracted_tables is None:
        extracted_tables = []

    # Map sections to page ranges
    page_ranges = _map_sections_to_pages(sections, text_blocks)

    results: list[PrioritySectionResult] = []
    for sec in sections:
        if sec.section_code not in PRIORITY_SECTIONS:
            continue

        page_range = page_ranges.get(sec.section_code)
        if page_range is None:
            # No page mapping found — use content_json text as-is
            try:
                content = json.loads(sec.content_json)
                full_text = content.get("text_excerpt", "")
            except (json.JSONDecodeError, TypeError):
                full_text = ""

            results.append(PrioritySectionResult(
                section_code=sec.section_code,
                section_name=sec.section_name,
                full_text=full_text,
                tables=[],
                soa_candidate_tables=[],
                cross_references=_find_cross_references(full_text),
                page_range=(0, 0),
            ))
            continue

        start, end = page_range
        full_text = _full_text_for_pages(text_blocks, start, end)
        section_tables = _tables_for_pages(extracted_tables, start, end)
        soa_candidates = [t for t in section_tables if _is_soa_candidate(t)]

        cross_refs = _find_cross_references(full_text)

        results.append(PrioritySectionResult(
            section_code=sec.section_code,
            section_name=sec.section_name,
            full_text=full_text,
            tables=section_tables,
            soa_candidate_tables=soa_candidates,
            cross_references=cross_refs,
            page_range=page_range,
        ))

        logger.info(
            "Priority section %s: pages %d-%d, %d chars, "
            "%d tables (%d SoA candidates)",
            sec.section_code, start, end, len(full_text),
            len(section_tables), len(soa_candidates),
        )

    return results
