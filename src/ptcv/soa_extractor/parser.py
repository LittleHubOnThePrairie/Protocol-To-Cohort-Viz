"""SoA table parser — extracts raw schedule tables from ICH section text.

Parses markdown-style pipe-delimited tables and aligned-column tables from
the content_json of IchSection objects produced by PTCV-20. Output is a
RawSoaTable suitable for UsdmMapper processing.

Recognised table formats:
  1. Markdown pipe tables (| Visit | Day 1 | Week 2 | ...)
  2. Aligned whitespace tables (two or more columns, space-separated)

The parser searches all provided IchSection objects and returns the first
table found. If no table is detected the result is None.

Risk tier: MEDIUM — data pipeline parsing (no patient data).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..ich_parser.models import IchSection
from .models import RawSoaTable


# ---------------------------------------------------------------------------
# Schedule-of-Activities indicator keywords
# ---------------------------------------------------------------------------

_SOA_KEYWORDS = re.compile(
    r"schedule\s+of\s+(?:activities|assessments|visits)"
    r"|visit\s+schedule"
    r"|assessment\s+schedule"
    r"|(?:study|trial)\s+procedures",
    re.IGNORECASE,
)

# A pipe-delimited table line has at least two | characters
_PIPE_LINE_RE = re.compile(r"\|[^|]+\|")

# A separator line in a markdown table: |---|---|...
_SEPARATOR_LINE_RE = re.compile(r"^\|[-|:\s]+\|$")

# Activity cell markers: "X", "x", "•", "✓", "Y", "yes", "1"
_MARKED_RE = re.compile(r"^[Xx✓•Y1]$|^[Yy]es$", re.IGNORECASE)


class SoaTableParser:
    """Extract the first SoA table from a list of IchSection objects.

    [PTCV-21 Scenario: Extract SoA to USDM Parquet with lineage]
    """

    def parse(self, sections: list[IchSection]) -> Optional[RawSoaTable]:
        """Find and parse the first SoA table in the given sections.

        Sections are searched in order. B.4 (Design), B.7 (Treatment),
        and B.11+ sections are tried first; remaining sections follow.

        Args:
            sections: IchSection list from PTCV-20.

        Returns:
            RawSoaTable if a table was found, None otherwise.
        """
        # Priority: sections most likely to contain SoA
        priority_codes = {"b.4", "b.7", "b.11", "b.10", "b.8"}
        ordered = sorted(
            sections,
            key=lambda s: (0 if s.section_code.lower() in priority_codes else 1),
        )

        for section in ordered:
            text = self._extract_text(section)
            if not text:
                continue
            table = self._parse_markdown_table(text, section.section_code)
            if table is not None:
                return table
            table = self._parse_aligned_table(text, section.section_code)
            if table is not None:
                return table

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(section: IchSection) -> str:
        """Pull the text field from content_json."""
        try:
            data = json.loads(section.content_json)
            return data.get("text", "") or data.get("text_excerpt", "")
        except (json.JSONDecodeError, AttributeError):
            return ""

    def _parse_markdown_table(
        self, text: str, section_code: str
    ) -> Optional[RawSoaTable]:
        """Parse a markdown pipe-delimited SoA table.

        Expects at least two non-separator rows with three or more columns.
        """
        lines = text.splitlines()

        # Find contiguous blocks of pipe-table lines
        table_blocks: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            stripped = line.strip()
            if _PIPE_LINE_RE.search(stripped):
                current.append(stripped)
            else:
                if len(current) >= 2:
                    table_blocks.append(current)
                current = []
        if len(current) >= 2:
            table_blocks.append(current)

        for block in table_blocks:
            result = self._process_pipe_block(block, section_code)
            if result is not None:
                return result

        return None

    def _process_pipe_block(
        self, block: list[str], section_code: str
    ) -> Optional[RawSoaTable]:
        """Convert a list of pipe-table lines into a RawSoaTable."""
        rows: list[list[str]] = []
        for line in block:
            if _SEPARATOR_LINE_RE.match(line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 2:
                rows.append(cells)

        if len(rows) < 2:
            return None

        # First row is the header row (visit names)
        header_row = rows[0]
        # Check if second row looks like day/window information
        second_row = rows[1]
        activity_start = 1

        day_row: list[str] = []
        # Heuristic: second row has day numbers or window info
        is_day_row = any(
            bool(re.search(r"[Dd]ay\s*-?\d+|[Ww]eek\s*\d+|[-]\d|\d+\s*[±]", cell))
            for cell in second_row[1:]
        )
        if is_day_row:
            day_row = second_row
            activity_start = 2

        activities: list[tuple[str, list[bool]]] = []
        for row in rows[activity_start:]:
            if not row or not row[0]:
                continue
            activity_name = row[0]
            scheduled = [
                _MARKED_RE.match(cell.strip()) is not None
                for cell in row[1:]
            ]
            # Pad to match header length
            n = len(header_row) - 1
            while len(scheduled) < n:
                scheduled.append(False)
            activities.append((activity_name, scheduled[:n]))

        if not activities:
            return None

        # Strip the first column header (typically "Activity" or "Assessment")
        visit_headers = header_row[1:]
        day_headers = day_row[1:] if day_row else []

        return RawSoaTable(
            visit_headers=visit_headers,
            day_headers=day_headers,
            activities=activities,
            section_code=section_code,
        )

    def _parse_aligned_table(
        self, text: str, section_code: str
    ) -> Optional[RawSoaTable]:
        """Parse a whitespace-aligned SoA table (two-or-more column columns).

        Only attempted if the text contains SoA indicator keywords.
        """
        if not _SOA_KEYWORDS.search(text):
            return None

        lines = text.splitlines()
        # Find lines with 3+ whitespace-separated tokens that look like
        # a table (first column non-numeric, subsequent columns look like
        # markers or visit labels).
        data_lines: list[list[str]] = []
        for line in lines:
            tokens = line.split()
            if len(tokens) >= 3:
                # First token is activity name (possibly multi-word but
                # in aligned tables row headers are often single words)
                data_lines.append(tokens)

        if len(data_lines) < 2:
            return None

        # The first data line is treated as column headers
        header_row = data_lines[0]
        visit_headers = header_row[1:]  # everything after row-header column
        activities: list[tuple[str, list[bool]]] = []

        for row in data_lines[1:]:
            name = row[0]
            scheduled = [
                _MARKED_RE.match(cell.strip()) is not None
                for cell in row[1:]
            ]
            n = len(visit_headers)
            while len(scheduled) < n:
                scheduled.append(False)
            activities.append((name, scheduled[:n]))

        if not activities:
            return None

        return RawSoaTable(
            visit_headers=visit_headers,
            day_headers=[],
            activities=activities,
            section_code=section_code,
        )
