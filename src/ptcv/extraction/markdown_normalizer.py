"""pymupdf4llm post-processing normalizer — PTCV-176.

Normalizes raw Markdown output from pymupdf4llm for downstream
consumption (route building, Sonnet classification, SoA assembly).

Transforms applied in order:
    1. TOC dot-leader removal (and optional extraction)
    2. Boilerplate removal (page headers/footers)
    3. Heading normalization (strip bold wrappers)
    4. Bold-to-heading promotion (numbered bold lines → ATX headings)
    5. Table cleanup (replace <br> in cells)
    6. Whitespace normalization (collapse blank lines)

Risk tier: MEDIUM — data pipeline post-processing (no patient data).
"""

from __future__ import annotations

import dataclasses
import re

from ptcv.ich_parser.toc_extractor import _PAGE_BOILERPLATE_RES


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TOCItem:
    """One entry from the protocol table of contents."""

    number: str  # e.g. "1.1"
    title: str  # e.g. "Synopsis"
    page: int  # e.g. 11


@dataclasses.dataclass
class NormalizationStats:
    """Counts of normalization actions performed."""

    boilerplate_lines_removed: int = 0
    headings_cleaned: int = 0
    headings_promoted: int = 0
    toc_lines_removed: int = 0
    br_tags_removed: int = 0
    blank_lines_collapsed: int = 0


@dataclasses.dataclass
class NormalizedResult:
    """Return value from ``normalize_markdown``."""

    text: str
    stats: NormalizationStats
    toc: list[TOCItem] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Compiled regex constants
# ---------------------------------------------------------------------------

# TOC dot-leader line:
#   **1.** **Protocol Summary ........11**
#   1.1. Synopsis ..........11
_TOC_DOT_LEADER_RE = re.compile(
    r"^\*?\*?(\d+(?:\.\d+)*)\.?\*?\*?"  # number (with optional bold)
    r"\s+\*?\*?"  # separator
    r"(.+?)"  # title
    r"\.{2,}\s*"  # dot leader
    r"(\d+)\*?\*?\s*$",  # page number (with optional bold)
    re.MULTILINE,
)

# ATX heading with bold wrapper: ##### **Title**
_ATX_BOLD_RE = re.compile(
    r"^(#{1,6})\s+\*\*(.+?)\*\*\s*$",
    re.MULTILINE,
)

# Compound bold heading: ##### **1. Summary** **1.1. Synopsis**
_ATX_MULTI_BOLD_RE = re.compile(
    r"^(#{1,6})\s+(\*\*.+?\*\*(?:\s+\*\*.+?\*\*)+)\s*$",
    re.MULTILINE,
)

# Standalone bold-numbered line (section heading rendered as bold):
#   **1. Protocol Summary**
#   **1.2. Background**
#   **1.** **Title Text**  (pymupdf4llm split-bold)
_BOLD_SECTION_RE = re.compile(
    r"^\*\*(\d+(?:\.\d+)*\.?)\s*\*?\*?\s*\*?\*?"  # number part
    r"(.+?)\*\*\s*$",  # title part
    re.MULTILINE,
)

# <br> or <br/> inside text
_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)

# Three or more consecutive newlines (blank line runs)
_EXCESS_NEWLINES_RE = re.compile(r"\n{4,}")

# pymupdf4llm-specific boilerplate beyond toc_extractor patterns
_EXTRA_BOILERPLATE_RES: list[re.Pattern[str]] = [
    re.compile(r"^\s*Page\s+\d+\s*$", re.IGNORECASE),
    re.compile(
        r"Approved\s+on\s+\d{1,2}\s+\w+\s+\d{4}\s+GMT", re.IGNORECASE
    ),
    re.compile(r"^\s*\d{4,5}\s*$"),  # orphan numeric IDs
]

# All boilerplate patterns (reused from toc_extractor + extras)
_ALL_BOILERPLATE_RES = _PAGE_BOILERPLATE_RES + _EXTRA_BOILERPLATE_RES


# ---------------------------------------------------------------------------
# Private transform functions
# ---------------------------------------------------------------------------


def _extract_and_remove_toc(
    text: str, stats: NormalizationStats, extract: bool
) -> tuple[str, list[TOCItem]]:
    """Remove TOC dot-leader lines; optionally extract TOCItems."""
    toc_items: list[TOCItem] = []
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        m = _TOC_DOT_LEADER_RE.match(line)
        if m:
            stats.toc_lines_removed += 1
            if extract:
                toc_items.append(
                    TOCItem(
                        number=m.group(1),
                        title=m.group(2).strip(),
                        page=int(m.group(3)),
                    )
                )
        else:
            kept.append(line)
    return "\n".join(kept), toc_items


def _remove_boilerplate(text: str, stats: NormalizationStats) -> str:
    """Remove page headers, footers, and boilerplate lines."""
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if any(pat.search(stripped) for pat in _ALL_BOILERPLATE_RES):
            stats.boilerplate_lines_removed += 1
        else:
            kept.append(line)
    return "\n".join(kept)


def _normalize_headings(text: str, stats: NormalizationStats) -> str:
    """Strip bold wrappers from ATX headings."""

    def _merge_compound(m: re.Match[str]) -> str:
        hashes = m.group(1)
        bold_parts = re.findall(r"\*\*(.+?)\*\*", m.group(2))
        stats.headings_cleaned += 1
        return f"{hashes} {' — '.join(bold_parts)}"

    # First handle compound bold headings (must come before single bold)
    text = _ATX_MULTI_BOLD_RE.sub(_merge_compound, text)

    def _strip_bold(m: re.Match[str]) -> str:
        stats.headings_cleaned += 1
        return f"{m.group(1)} {m.group(2)}"

    text = _ATX_BOLD_RE.sub(_strip_bold, text)
    return text


def _section_depth(number: str) -> int:
    """Return ATX heading level from a section number string.

    ``"1"`` → 2 (``##``), ``"1.2"`` → 3 (``###``), ``"1.2.3"`` → 4,
    four-or-more parts → 5.  Level 1 (``#``) is reserved for the
    document title, so top-level sections start at ``##``.
    """
    parts = [p for p in number.rstrip(".").split(".") if p]
    return min(len(parts) + 1, 5)


def _promote_bold_headings(text: str, stats: NormalizationStats) -> str:
    """Promote standalone bold-numbered lines to ATX headings.

    pymupdf4llm often renders section headings as bold text without
    ``#`` prefixes.  This converts e.g. ``**1. Protocol Summary**`` to
    ``## 1. Protocol Summary`` with the heading level inferred from
    section depth.
    """
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip table rows and lines that already have ATX prefix
        if stripped.startswith("|") or stripped.startswith("#"):
            result.append(line)
            continue
        m = _BOLD_SECTION_RE.match(stripped)
        if m:
            number = m.group(1).rstrip(".")
            title = m.group(2).strip()
            # Skip if title is empty (e.g. just a bold number)
            if not title:
                result.append(line)
                continue
            level = _section_depth(number)
            hashes = "#" * level
            result.append(f"{hashes} {number}. {title}")
            stats.headings_promoted += 1
        else:
            result.append(line)
    return "\n".join(result)


def _clean_tables(text: str, stats: NormalizationStats) -> str:
    """Replace <br> tags with spaces inside markdown table rows."""
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        if line.lstrip().startswith("|"):
            cleaned, n = _BR_TAG_RE.subn(" ", line)
            stats.br_tags_removed += n
            result.append(cleaned)
        else:
            result.append(line)
    return "\n".join(result)


def _normalize_whitespace(text: str, stats: NormalizationStats) -> str:
    """Collapse excessive blank lines and strip trailing whitespace."""
    # Strip trailing whitespace per line
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)
    # Collapse 3+ consecutive blank lines to 2
    text, n = _EXCESS_NEWLINES_RE.subn("\n\n\n", text)
    stats.blank_lines_collapsed += n
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_markdown(
    text: str,
    *,
    extract_toc: bool = False,
) -> NormalizedResult:
    """Normalize pymupdf4llm Markdown output for downstream consumption.

    Args:
        text: Raw Markdown from pymupdf4llm.
        extract_toc: If True, parse TOC dot-leader lines into
            ``NormalizedResult.toc``.

    Returns:
        NormalizedResult with cleaned text, stats, and optional TOC.
    """
    stats = NormalizationStats()
    text, toc = _extract_and_remove_toc(text, stats, extract=extract_toc)
    text = _remove_boilerplate(text, stats)
    text = _normalize_headings(text, stats)
    text = _promote_bold_headings(text, stats)
    text = _clean_tables(text, stats)
    text = _normalize_whitespace(text, stats)
    return NormalizedResult(text=text, stats=stats, toc=toc)
