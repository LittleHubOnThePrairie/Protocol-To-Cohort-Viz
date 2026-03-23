"""Column header validation for SoA tables (PTCV-215).

Validates extracted column headers against expected SoA visit patterns
and detects multi-column headers that need splitting. Flags tables
with column-count mismatches for Level 2 escalation.

Risk tier: LOW — validation only, no data mutation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Visit-header patterns: tokens that look like clinical visit/timepoint names
_VISIT_HEADER_RE = re.compile(
    r"(?:V(?:isit)?\s*\d|D(?:ay)?\s*-?\d|W(?:eek|k)?\s*\d"
    r"|Screen|Baseline|BL\b|EOT\b|EOS\b|Follow|FU\b"
    r"|Rand|Month\s*\d|Cycle\s*\d|C\d+D\d+"
    r"|Pre.?dose|Post.?dose|Early.?Term|ET\b"
    r"|Dose\s*\d|Period\s*\d|Enrol|End.of"
    r"|Unscheduled|Discontinuation|Completion)",
    re.IGNORECASE,
)

# Multi-column header patterns: "Future Cycles (even) (odd)",
# "Cycle 1 / Cycle 2", "Treatment A | Treatment B"
_MULTI_COL_RE = re.compile(
    r"\((?:even|odd)\)"
    r"|\beven\b.*\bodd\b"
    r"|\b(?:arm|group)\s+[AB]\b",
    re.IGNORECASE,
)


@dataclass
class ColumnValidationResult:
    """Result of column header validation.

    Attributes:
        header_count: Number of extracted column headers.
        visit_headers_matched: Number of headers matching visit patterns.
        match_ratio: Fraction of headers matching visit patterns.
        misaligned_rows: Row indices where cell count != header count.
        multi_column_headers: Headers that may need splitting.
        needs_escalation: Whether to escalate to Level 2 (Vision).
        escalation_reason: Why escalation was triggered (if any).
    """

    header_count: int = 0
    visit_headers_matched: int = 0
    match_ratio: float = 0.0
    misaligned_rows: list[int] = field(default_factory=list)
    multi_column_headers: list[str] = field(default_factory=list)
    needs_escalation: bool = False
    escalation_reason: str = ""


def validate_columns(
    header: list[str],
    rows: list[list[str]],
    min_match_ratio: float = 0.3,
    max_misaligned_ratio: float = 0.2,
) -> ColumnValidationResult:
    """Validate SoA table column headers and row alignment.

    Checks that:
    1. Headers match expected visit/timepoint patterns
    2. Each row has the same number of cells as the header
    3. Multi-column headers are detected for potential splitting

    Args:
        header: List of column header strings.
        rows: List of data rows (each a list of cell strings).
        min_match_ratio: Minimum fraction of headers that must match
            visit patterns. Below this, escalation is triggered.
        max_misaligned_ratio: Maximum fraction of rows that can have
            cell count != header count before escalation.

    Returns:
        ColumnValidationResult with validation details.
    """
    result = ColumnValidationResult(header_count=len(header))

    if not header:
        result.needs_escalation = True
        result.escalation_reason = "No column headers found"
        return result

    # Skip the first column (Assessment/Activity label)
    visit_headers = header[1:] if len(header) > 1 else header

    # Count headers matching visit patterns
    matched = 0
    for h in visit_headers:
        if _VISIT_HEADER_RE.search(h):
            matched += 1

    result.visit_headers_matched = matched
    result.match_ratio = (
        matched / len(visit_headers) if visit_headers else 0.0
    )

    # Detect multi-column headers
    for h in header:
        if _MULTI_COL_RE.search(h):
            result.multi_column_headers.append(h)

    # Check row alignment
    for i, row in enumerate(rows):
        if len(row) != len(header):
            result.misaligned_rows.append(i)

    misaligned_ratio = (
        len(result.misaligned_rows) / len(rows) if rows else 0.0
    )

    # Determine escalation
    if result.match_ratio < min_match_ratio and len(visit_headers) >= 3:
        result.needs_escalation = True
        result.escalation_reason = (
            f"Low visit header match ratio: {result.match_ratio:.0%} "
            f"({matched}/{len(visit_headers)} matched)"
        )
    elif misaligned_ratio > max_misaligned_ratio:
        result.needs_escalation = True
        result.escalation_reason = (
            f"High row misalignment: {misaligned_ratio:.0%} "
            f"({len(result.misaligned_rows)}/{len(rows)} rows)"
        )

    return result


def split_multi_column_header(header_text: str) -> list[str]:
    """Split a multi-column header into sub-columns.

    Handles patterns like:
    - "Future Cycles (even) (odd)" → ["Future Cycles (even)", "Future Cycles (odd)"]
    - "Cycle 1 / Cycle 2" → ["Cycle 1", "Cycle 2"]

    Args:
        header_text: Header text that may represent multiple columns.

    Returns:
        List of sub-column names. Returns [header_text] if no split
        is detected.
    """
    # Pattern: "X (even) (odd)"
    even_odd = re.match(
        r"(.+?)\s*\(even\)\s*\(odd\)",
        header_text,
        re.IGNORECASE,
    )
    if even_odd:
        base = even_odd.group(1).strip()
        return [f"{base} (even)", f"{base} (odd)"]

    # Pattern: "X / Y" or "X | Y"
    split = re.split(r"\s*[/|]\s*", header_text)
    if len(split) >= 2 and all(len(s.strip()) > 1 for s in split):
        return [s.strip() for s in split]

    return [header_text]


__all__ = [
    "ColumnValidationResult",
    "validate_columns",
    "split_multi_column_header",
]
