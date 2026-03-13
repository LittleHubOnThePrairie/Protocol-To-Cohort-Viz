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
    r"|(?:study|trial)\s+procedures"
    r"|study\s+calendar"
    r"|study\s+timeline"
    r"|treatment\s+calendar"
    r"|assessment\s+calendar",
    re.IGNORECASE,
)

# A pipe-delimited table line has at least two | characters
_PIPE_LINE_RE = re.compile(r"\|[^|]+\|")

# A separator line in a markdown table: |---|---|...
_SEPARATOR_LINE_RE = re.compile(r"^\|[-|:\s]+\|$")

# Activity cell markers: "X", "x", "•", "✓", "Y", "yes", "1"
_MARKED_RE = re.compile(r"^[Xx✓•Y1]$|^[Yy]es$", re.IGNORECASE)

# Visit-header patterns — tokens that look like clinical visit/timepoint names.
# Used to validate aligned-table headers are not just random English words.
_VISIT_HEADER_RE = re.compile(
    r"(?:V(?:isit)?\s*\d|D(?:ay)?\s*-?\d|W(?:eek|k)?\s*\d"
    r"|Screen|Baseline|BL\b|EOT\b|EOS\b|Follow|FU\d"
    r"|Rand|Month\s*\d|Cycle\s*\d|C\d+D\d+"
    r"|Pre.?dose|Post.?dose|Early.?Term|ET\b"
    r"|Dose\s*\d|Period\s*\d|Enrol|End.of)",
    re.IGNORECASE,
)


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

        # Reject tables with zero scheduled markers — prose text parsed
        # as a table will have no X/✓ markers (PTCV-128).
        if not any(any(sched) for _, sched in activities):
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
        """Parse a whitespace-aligned SoA table (two-or-more columns).

        Only attempted if the text contains SoA indicator keywords *near*
        candidate table lines. This prevents a keyword buried in a 30K
        prose block from triggering table detection on unrelated text
        (PTCV-128).
        """
        kw_match = _SOA_KEYWORDS.search(text)
        if not kw_match:
            return None

        lines = text.splitlines()

        # Restrict search to a window around the keyword match to avoid
        # treating distant prose as table rows (PTCV-128).
        kw_line_idx = text[:kw_match.start()].count("\n")
        window = 80  # lines above/below the keyword
        start = max(0, kw_line_idx - window)
        end = min(len(lines), kw_line_idx + window)
        candidate_lines = lines[start:end]

        # Find lines with 3+ whitespace-separated tokens that look like
        # a table (first column non-numeric, subsequent columns look like
        # markers or visit labels).
        data_lines: list[list[str]] = []
        for line in candidate_lines:
            tokens = line.split()
            if len(tokens) >= 3:
                data_lines.append(tokens)

        if len(data_lines) < 2:
            return None

        # The first data line is treated as column headers
        header_row = data_lines[0]
        visit_headers = header_row[1:]  # everything after row-header column

        # Validate that enough headers look like visit/timepoint names
        # rather than random English words (PTCV-128, PTCV-151).
        # Require at least 3 visit columns and at least 2 (or 50% of
        # headers) matching visit patterns to prevent prose tokenisation.
        if len(visit_headers) < 3:
            return None
        visit_like = sum(
            1 for h in visit_headers if _VISIT_HEADER_RE.search(h)
        )
        min_required = max(2, len(visit_headers) // 2)
        if visit_like < min_required:
            return None

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

        # Reject tables with zero scheduled markers (PTCV-128).
        if not any(any(sched) for _, sched in activities):
            return None

        return RawSoaTable(
            visit_headers=visit_headers,
            day_headers=[],
            activities=activities,
            section_code=section_code,
        )
