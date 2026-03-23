"""Protocol Table-of-Contents and section header extractor (PTCV-89).

Extracts a structured, navigable index from clinical protocol PDFs:

  1. Detect TOC pages (look for "Table of Contents" heading)
  2. Parse TOC entries — section numbers, titles, page references
  3. Scan document body for matching section headers
  4. Build page/character-offset ranges for each section
  5. Map each section to its text content

The resulting ``ProtocolIndex`` is the primary navigation layer for
the query-driven extraction pipeline (PTCV-87).

Risk tier: LOW — read-only document analysis.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a numbered section like "1", "1.0", "1.1", "1.1.1", "10.2.3"
_SECTION_NUM_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\.?\s+"
)

# TOC line: section number + title + optional dot-leaders + page number
# Examples:
#   "1.1 Background .............. 24"
#   "3  Study Design  29"
#   "2.0 BACKGROUND AND RATIONALE"
_TOC_LINE_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\.?\s+"    # section number
    r"(.+?)"                      # title (non-greedy)
    r"(?:\s*\.{2,}\s*|\s{3,})"   # dot-leaders or wide whitespace
    r"(\d+)\s*$"                  # page number
)

# TOC line without page number (simple TOC)
_TOC_LINE_SIMPLE_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\.?\s+"    # section number
    r"([A-Z].{2,})"              # title (starts with uppercase, 3+ chars)
    r"\s*$"
)

# Lettered appendix entry: "A  APPENDIX TITLE ... 45"
_APPENDIX_RE = re.compile(
    r"^([A-Z])\s{2,}"            # single letter + whitespace
    r"(.+?)"                      # title
    r"(?:\s*\.{2,}\s*|\s{3,})"   # dot-leaders or whitespace
    r"(\d+)\s*$"                  # page number
)

# "List of Tables/Figures" entries: Table/Figure/Listing + ref + title + page
# Examples:
#   "Table 10-1 Schedule of Assessments ............ 30"
#   "Figure 6-1 Trial Design ....................... 19"
#   "Listing 14.1.1 Demographics ................... 42"
_LIST_ENTRY_RE = re.compile(
    r"^(?:Table|Figure|Listing|Exhibit)\s+"  # prefix
    r"[\w.-]+"                               # reference (10-1, 6.1, etc.)
    r"\s+.+?"                                # title (non-greedy)
    r"(?:\s*\.{2,}\s*|\s{3,})"              # dot-leaders or wide whitespace
    r"\d+\s*$",                              # page number
    re.IGNORECASE,
)

# "List of ..." header lines (e.g. "LIST OF IN-TEXT TABLES")
_LIST_OF_HEADER_RE = re.compile(
    r"^LIST\s+OF\s+", re.IGNORECASE,
)

# Page header/footer boilerplate — patterns that reliably identify
# running headers/footers injected by PDF extraction.
_PAGE_BOILERPLATE_RES: list[re.Pattern[str]] = [
    re.compile(r"Proprietary\s+and\s+Confidential", re.IGNORECASE),
    re.compile(r"Proprietary\s+confidential", re.IGNORECASE),
    re.compile(r"^CONFIDENTIAL\s*$", re.IGNORECASE),
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    # Copyright/legal notices (PTCV-127)
    re.compile(r"\u00a9\s*\d{4}", re.IGNORECASE),  # © 2023
    re.compile(r"\(c\)\s*\d{4}", re.IGNORECASE),  # (c) 2023
    re.compile(r"Clinical\s+Trial\s+Protocol\b", re.IGNORECASE),
]

# Regex to normalise digits in a line for header/footer frequency analysis.
_DIGIT_NORM_RE = re.compile(r"\d+")

# Body heading: numbered section at start of a line
_BODY_HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\.?\s+([A-Z].*)",
    re.MULTILINE,
)

# Reject date-like lines: "24 January 2023", "15 March 2021"
_DATE_LINE_RE = re.compile(
    r"^\d+\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}",
    re.IGNORECASE,
)

# Maximum plausible leading section number component (e.g. "1"-"50")
_MAX_SECTION_NUM = 50

# TOC page indicators
_TOC_HEADER_RE = re.compile(
    r"(?:TABLE\s+OF\s+CONTENTS|CONTENTS)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TOCEntry:
    """A single entry from the protocol's Table of Contents.

    Attributes:
        level: Hierarchy depth (1 = top-level, 2 = subsection, etc.).
        number: Section number string (e.g. ``"1.1.2"``).
        title: Section title text.
        page_ref: Page number from the TOC (protocol-internal numbering).
            ``0`` if the TOC has no page references.
        page_start: Actual 1-based PDF page where this section starts
            (resolved from body scan). ``0`` if unresolved.
        page_end: Actual 1-based PDF page where this section ends
            (exclusive — i.e. the start of the next section).
            ``0`` if unresolved.
        char_offset_start: Character offset in the full-document text
            where this section begins. ``-1`` if unresolved.
        char_offset_end: Character offset where this section ends.
            ``-1`` if unresolved.
    """

    level: int
    number: str
    title: str
    page_ref: int = 0
    page_start: int = 0
    page_end: int = 0
    char_offset_start: int = -1
    char_offset_end: int = -1


@dataclasses.dataclass
class SectionHeader:
    """A section header detected in the document body.

    Attributes:
        number: Section number string (e.g. ``"3.2"``).
        title: Header text (without the number prefix).
        level: Hierarchy depth.
        page: 1-based PDF page number.
        char_offset: Character offset in the full-document text.
    """

    number: str
    title: str
    level: int
    page: int
    char_offset: int


@dataclasses.dataclass
class ProtocolIndex:
    """Complete navigable index of a protocol document.

    Attributes:
        source_path: Path to the source PDF.
        page_count: Total pages in the PDF.
        toc_entries: Entries parsed from the TOC page(s).
        section_headers: Headers detected in the document body.
        content_spans: Mapping from section number to the text content
            it covers.  Keys are section numbers (e.g. ``"1.1"``).
        full_text: Complete document text (page-concatenated).
        toc_found: Whether a TOC page was detected.
        toc_pages: 1-based page numbers of detected TOC pages.
        tables: Structured tables extracted from all pages
            (PTCV-237). Each dict has ``page_number``,
            ``header_row`` (JSON string), ``data_rows`` (JSON string),
            ``extractor_used``, and ``table_index``. Empty list when
            table extraction is disabled or fails.
        diagrams: Diagrams detected from PDF pages (PTCV-261).
            Each dict is the serialized form of a
            ``ptcv.extraction.diagram_finder.Diagram`` via
            ``Diagram.to_dict()``, containing ``diagram_type``,
            ``page_number``, ``nodes``, and ``edges``. Empty list when
            no diagrams are found.
        figures: Detected figures with Vision-generated captions
            (PTCV-263). Each dict has ``page_number``, ``bbox``,
            ``figure_type_hint``, ``caption``, and
            ``detection_method``. Vision captioning is gated behind
            ``PTCV_ENABLE_FIGURE_VISION=1``. Empty list when figure
            detection is disabled or no figures found.
        layout_graph: Spatial layout relationships between page
            elements (PTCV-263). Dict with ``page_count``,
            ``total_nodes``, ``total_edges``, and ``pages`` (list
            of per-page graphs with nodes and edges). None when
            layout graph construction fails or is skipped.
    """

    source_path: str
    page_count: int
    toc_entries: list[TOCEntry]
    section_headers: list[SectionHeader]
    content_spans: dict[str, str]
    full_text: str
    toc_found: bool
    toc_pages: list[int]
    tables: list[dict] = dataclasses.field(default_factory=list)
    diagrams: list[dict] = dataclasses.field(default_factory=list)
    figures: list[dict] = dataclasses.field(default_factory=list)
    layout_graph: Optional[dict] = None

    def get_section_text(self, section_number: str) -> Optional[str]:
        """Return the text content for a section, or *None*."""
        return self.content_spans.get(section_number)

    def get_children(self, parent_number: str) -> list[TOCEntry]:
        """Return direct child TOC entries of *parent_number*."""
        prefix = parent_number + "."
        parent_level = _section_level(parent_number)
        return [
            e for e in self.toc_entries
            if e.number.startswith(prefix)
            and e.level == parent_level + 1
        ]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _section_level(number: str) -> int:
    """Determine hierarchy level from a section number.

    ``"1"`` → 1, ``"1.1"`` → 2, ``"1.1.1"`` → 3.
    """
    return number.count(".") + 1


def _normalise_number(raw: str) -> str:
    """Normalise a section number — strip trailing dots/zeros.

    ``"1.0"`` → ``"1"``, ``"3.2."`` → ``"3.2"``.
    """
    s = raw.strip().rstrip(".")
    # Remove trailing ".0" for "1.0" → "1", but keep "10"
    if s.endswith(".0") and len(s) > 2:
        s = s[:-2]
    return s


def _clean_title(raw: str) -> str:
    """Strip dot-leaders, extra whitespace, and page numbers from title."""
    # Remove trailing dots and page numbers
    cleaned = re.sub(r"\.{2,}\s*\d*\s*$", "", raw)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# TOC page detection and parsing
# ---------------------------------------------------------------------------


def _is_toc_page(text: str) -> bool:
    """Return True if *text* appears to be a TOC page.

    Detects pages with an explicit "TABLE OF CONTENTS" header,
    AND pages where a high proportion of non-empty lines match
    the TOC-entry format (dotted leaders + page numbers), even
    without a header (PTCV-107: sub-TOC / continuation pages).
    """
    if _TOC_HEADER_RE.search(text):
        return True
    # PTCV-107: detect pages that are predominantly TOC lines
    non_empty = [
        ln for ln in text.split("\n") if ln.strip()
    ]
    if len(non_empty) < 3:
        return False
    toc_count = sum(
        1 for ln in non_empty
        if _TOC_LINE_RE.match(ln.strip())
    )
    # If >50% of non-empty lines are TOC-format, it's a TOC page
    return toc_count / len(non_empty) > 0.5


def _count_toc_lines(text: str) -> int:
    """Count how many lines look like TOC entries."""
    count = 0
    for line in text.split("\n"):
        line = line.strip()
        if _TOC_LINE_RE.match(line) or _TOC_LINE_SIMPLE_RE.match(line):
            count += 1
    return count


def _is_page_boilerplate(line: str) -> bool:
    """Return True if *line* looks like a page header/footer (PTCV-108).

    Matches common running-header patterns injected by PDF extraction,
    e.g. ``"Proprietary and Confidential Page 11 of 59 ..."``
    """
    return any(pat.search(line) for pat in _PAGE_BOILERPLATE_RES)


def strip_toc_lines(text: str) -> str:
    """Remove TOC-entry, list-of, and page-boilerplate lines (PTCV-107/108).

    Strips:
      - Section-number TOC lines  (``"7.3 Exclusion ... 20"``)
      - Appendix TOC lines        (``"A  APPENDIX ... 45"``)
      - Table/Figure list entries  (``"Table 10-1 Schedule ... 30"``)
      - "List of ..." headers      (``"LIST OF IN-TEXT TABLES"``)
      - Page header/footer lines   (``"Proprietary and Confidential ..."``)

    Returns:
        Text with junk lines removed.
    """
    cleaned: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if _TOC_LINE_RE.match(stripped):
            continue  # Skip TOC line
        if _APPENDIX_RE.match(stripped):
            continue  # Skip appendix TOC line
        # PTCV-108: skip Table/Figure/Listing list entries
        if _LIST_ENTRY_RE.match(stripped):
            continue
        # PTCV-108: skip "List of ..." headers
        if _LIST_OF_HEADER_RE.match(stripped):
            continue
        # PTCV-108: skip page header/footer boilerplate
        if _is_page_boilerplate(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Per-page repeating header/footer detection (PTCV-127)
# ---------------------------------------------------------------------------


def _detect_repeating_lines(
    page_texts: list[tuple[int, str]],
    *,
    max_header_lines: int = 5,
    max_footer_lines: int = 3,
    min_frequency: float = 0.3,
    min_pages: int = 5,
) -> tuple[set[str], set[str]]:
    """Detect repeating page headers and footers by frequency analysis.

    Examines the first/last *N* non-empty lines on each page.  Lines
    whose normalised form (digits replaced with ``#``) appears on
    >=*min_frequency* of all pages are classified as running headers
    or footers.

    Pharmaceutical protocol headers commonly span 4-5 lines (sponsor
    name, trial number, protocol title, confidentiality, copyright)
    so the default ``max_header_lines=5`` captures these.

    Args:
        page_texts: ``(page_number, text)`` pairs.
        max_header_lines: Number of top lines to examine per page.
        max_footer_lines: Number of bottom lines to examine per page.
        min_frequency: Fraction of pages a line must appear on.
        min_pages: Minimum page count to enable detection.

    Returns:
        ``(header_patterns, footer_patterns)`` — sets of normalised
        line strings identified as repeating.
    """
    if len(page_texts) < min_pages:
        return set(), set()

    threshold = len(page_texts) * min_frequency
    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    for _, text in page_texts:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        # Deduplicate per-page to avoid inflating counts
        top_seen: set[str] = set()
        for ln in lines[:max_header_lines]:
            norm = _DIGIT_NORM_RE.sub("#", ln)
            if norm not in top_seen:
                top_seen.add(norm)
                top_counter[norm] += 1
        bot_seen: set[str] = set()
        for ln in lines[-max_footer_lines:]:
            norm = _DIGIT_NORM_RE.sub("#", ln)
            if norm not in bot_seen:
                bot_seen.add(norm)
                bottom_counter[norm] += 1

    header_pats = {
        pat for pat, count in top_counter.items() if count >= threshold
    }
    footer_pats = {
        pat for pat, count in bottom_counter.items() if count >= threshold
    }
    return header_pats, footer_pats


def _strip_page_headers_footers(
    page_texts: list[tuple[int, str]],
    *,
    max_header_lines: int = 5,
    max_footer_lines: int = 3,
    min_frequency: float = 0.3,
    min_pages: int = 5,
) -> list[tuple[int, str]]:
    """Remove repeating header/footer lines from each page (PTCV-127).

    Performs frequency analysis across all pages to detect running
    headers and footers, then strips matching lines from the top
    and bottom of each page.

    Returns:
        Cleaned ``(page_number, text)`` list.
    """
    header_pats, footer_pats = _detect_repeating_lines(
        page_texts,
        max_header_lines=max_header_lines,
        max_footer_lines=max_footer_lines,
        min_frequency=min_frequency,
        min_pages=min_pages,
    )

    if not header_pats and not footer_pats:
        return page_texts

    total_stripped = 0
    cleaned: list[tuple[int, str]] = []
    for page_num, text in page_texts:
        lines = text.split("\n")
        non_empty_indices = [
            i for i, ln in enumerate(lines) if ln.strip()
        ]
        if not non_empty_indices:
            cleaned.append((page_num, text))
            continue

        strip_indices: set[int] = set()

        # Check top lines
        for idx in non_empty_indices[:max_header_lines]:
            norm = _DIGIT_NORM_RE.sub("#", lines[idx].strip())
            if norm in header_pats:
                strip_indices.add(idx)

        # Check bottom lines
        for idx in non_empty_indices[-max_footer_lines:]:
            norm = _DIGIT_NORM_RE.sub("#", lines[idx].strip())
            if norm in footer_pats:
                strip_indices.add(idx)

        if strip_indices:
            total_stripped += len(strip_indices)
            out = [
                ln for i, ln in enumerate(lines)
                if i not in strip_indices
            ]
            cleaned.append((page_num, "\n".join(out)))
        else:
            cleaned.append((page_num, text))

    if total_stripped > 0:
        logger.info(
            "Stripped %d repeating header/footer lines across %d pages",
            total_stripped,
            len(page_texts),
        )

    return cleaned


def _strip_patterns_from_text(
    text: str,
    patterns: set[str],
) -> str:
    """Remove lines matching header/footer patterns (PTCV-189).

    Unlike :func:`_strip_page_headers_footers` which operates on
    per-page text with positional constraints (top-N / bottom-N),
    this function strips matching lines *anywhere* in a continuous
    text.  Used to clean pymupdf4llm markdown which has already
    been flattened into a single string.

    Matching uses substring containment rather than exact equality
    because pymupdf4llm may wrap lines differently from the plain
    text extractor (e.g. concatenating header fragments that were
    separate lines in ``page.get_text()``).

    Only patterns longer than 10 characters are used for substring
    matching to avoid false positives on short normalised forms.

    Args:
        text: Full document text (markdown or plain).
        patterns: Normalised line patterns from
            :func:`_detect_repeating_lines`.

    Returns:
        Cleaned text with matching lines removed.
    """
    if not patterns:
        return text

    # Use longer patterns for substring matching to avoid
    # false positives (e.g. "C CI" is too short).
    long_pats = {p for p in patterns if len(p) > 10}
    short_pats = {p for p in patterns if len(p) <= 10}

    lines = text.split("\n")
    kept: list[str] = []
    stripped = 0
    for line in lines:
        norm = _DIGIT_NORM_RE.sub("#", line.strip())
        # Exact match (handles short patterns safely)
        if norm in patterns:
            stripped += 1
            continue
        # Substring containment for long patterns
        if long_pats and any(p in norm for p in long_pats):
            stripped += 1
            continue
        kept.append(line)

    if stripped > 0:
        logger.info(
            "PTCV-189: Stripped %d header/footer lines from "
            "pymupdf4llm markdown",
            stripped,
        )

    return "\n".join(kept)


def _parse_toc_page(text: str) -> list[TOCEntry]:
    """Parse TOC entries from a single page of text."""
    entries: list[TOCEntry] = []
    seen_numbers: set[str] = set()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Skip the "TABLE OF CONTENTS" header itself
        if _TOC_HEADER_RE.match(line):
            continue
        # Skip "List of Tables/Figures" lines
        if re.match(r"(?i)^list\s+of\s+", line):
            continue

        entry: TOCEntry | None = None

        # Try numbered entry with page reference
        m = _TOC_LINE_RE.match(line)
        if m:
            num = _normalise_number(m.group(1))
            title = _clean_title(m.group(2))
            page = int(m.group(3))
            entry = TOCEntry(
                level=_section_level(num),
                number=num,
                title=title,
                page_ref=page,
            )

        # Try numbered entry without page reference
        if entry is None:
            m = _TOC_LINE_SIMPLE_RE.match(line)
            if m:
                num = _normalise_number(m.group(1))
                title = _clean_title(m.group(2))
                entry = TOCEntry(
                    level=_section_level(num),
                    number=num,
                    title=title,
                )

        # Try lettered appendix
        if entry is None:
            m = _APPENDIX_RE.match(line)
            if m:
                letter = m.group(1)
                title = _clean_title(m.group(2))
                page = int(m.group(3))
                entry = TOCEntry(
                    level=1,
                    number=letter,
                    title=title,
                    page_ref=page,
                )

        if entry is not None and entry.number not in seen_numbers:
            seen_numbers.add(entry.number)
            entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Body header detection
# ---------------------------------------------------------------------------


def _detect_body_headers(
    page_texts: list[tuple[int, str]],
    full_text: str,
    toc_pages: list[int] | None = None,
    toc_numbers: set[str] | None = None,
    body_start_page: int = 1,
    page_offsets: dict[int, int] | None = None,
) -> list[SectionHeader]:
    """Detect section headers in the document body.

    Args:
        page_texts: List of ``(1-based page number, page text)`` tuples.
        full_text: Concatenated full document text.
        toc_pages: 1-based page numbers to skip (TOC pages).
        toc_numbers: Known section numbers from the TOC.  When provided,
            only section numbers whose leading component appears in the
            TOC are accepted (reduces false positives).
        body_start_page: First page to scan for body headers.
            Pages before this are skipped (front matter).
        page_offsets: Mapping from 1-based page number to the
            character offset where that page starts in *full_text*.
            When provided, char_offset searches start from the
            page's offset to avoid matching TOC occurrences
            (PTCV-109).

    Returns:
        Sorted list of :class:`SectionHeader` by character offset.
    """
    skip_pages = set(toc_pages or [])
    # Build set of leading components from TOC for validation
    toc_leading: set[str] | None = None
    if toc_numbers:
        toc_leading = {n.split(".")[0] for n in toc_numbers}

    headers: list[SectionHeader] = []
    seen: set[tuple[str, int]] = set()  # (number, page) dedup

    for page_num, page_text in page_texts:
        # Skip front matter and TOC pages
        if page_num < body_start_page:
            continue
        if page_num in skip_pages:
            continue

        for m in _BODY_HEADING_RE.finditer(page_text):
            raw_num = m.group(1)
            title = m.group(2).strip()
            num = _normalise_number(raw_num)

            # --- Noise filters ---

            # Reject implausibly large leading section numbers
            # (catches addresses like "1800 Century Park East")
            leading = int(num.split(".")[0])
            if leading > _MAX_SECTION_NUM:
                continue

            # Reject date-like lines ("24 January 2023")
            full_line = m.group(0)
            if _DATE_LINE_RE.match(full_line):
                continue

            # If TOC is available, only accept numbers whose leading
            # component is present in the TOC
            if toc_leading and num.split(".")[0] not in toc_leading:
                continue

            # Skip very long "headings" (likely paragraph starts)
            if len(title) > 120:
                continue
            # Skip lines that are clearly not headings
            if title.endswith(",") or title.endswith(";"):
                continue

            key = (num, page_num)
            if key in seen:
                continue
            seen.add(key)

            # Find character offset in full_text.
            # Search from this page's offset to avoid matching TOC
            # entries that appear earlier in the document (PTCV-109).
            search_start = 0
            if page_offsets is not None:
                search_start = page_offsets.get(page_num, 0)
            search_pat = (
                re.escape(raw_num) + r"\.?\s+"
                + re.escape(title[:40])
            )
            offset_match = re.search(
                search_pat, full_text[search_start:]
            )
            char_offset = (
                (search_start + offset_match.start())
                if offset_match
                else -1
            )

            headers.append(SectionHeader(
                number=num,
                title=title,
                level=_section_level(num),
                page=page_num,
                char_offset=char_offset,
            ))

    # Sort by character offset (or page if offset unknown)
    headers.sort(key=lambda h: (h.char_offset if h.char_offset >= 0 else 999999, h.page))

    # Deduplicate: keep only the first occurrence of each section number
    # (later occurrences are typically amendment tables or appendix recaps)
    seen_nums: set[str] = set()
    deduped: list[SectionHeader] = []
    for h in headers:
        if h.number not in seen_nums:
            seen_nums.add(h.number)
            deduped.append(h)
    return deduped


# ---------------------------------------------------------------------------
# Content span resolution
# ---------------------------------------------------------------------------


def _resolve_content_spans(
    headers: list[SectionHeader],
    full_text: str,
) -> dict[str, str]:
    """Map each section header to its text content.

    Each section's content runs from its header position to the start
    of the next section header at the same or higher level.
    """
    spans: dict[str, str] = {}
    if not headers:
        return spans

    # Only use headers with resolved offsets
    resolved = [h for h in headers if h.char_offset >= 0]
    if not resolved:
        return spans

    resolved.sort(key=lambda h: h.char_offset)

    for i, header in enumerate(resolved):
        start = header.char_offset
        # End is the start of the next header
        if i + 1 < len(resolved):
            end = resolved[i + 1].char_offset
        else:
            end = len(full_text)

        content = full_text[start:end].strip()
        # PTCV-107: strip residual TOC-format lines from span
        content = strip_toc_lines(content)
        spans[header.number] = content

    return spans


def _resolve_toc_pages(
    toc_entries: list[TOCEntry],
    headers: list[SectionHeader],
) -> list[TOCEntry]:
    """Enrich TOC entries with actual PDF page numbers from body headers.

    Updates ``page_start`` and ``page_end`` on each TOC entry by
    matching against detected body headers.
    """
    # Build lookup: section number → body header
    header_map: dict[str, SectionHeader] = {}
    for h in headers:
        if h.number not in header_map:
            header_map[h.number] = h

    resolved: list[TOCEntry] = []
    for entry in toc_entries:
        matched = header_map.get(entry.number)
        if matched:
            entry = dataclasses.replace(
                entry,
                page_start=matched.page,
                char_offset_start=matched.char_offset,
            )
        resolved.append(entry)

    # Compute page_end and char_offset_end from next entry
    for i, entry in enumerate(resolved):
        if i + 1 < len(resolved):
            nxt = resolved[i + 1]
            if nxt.page_start > 0:
                resolved[i] = dataclasses.replace(
                    entry,
                    page_end=nxt.page_start,
                    char_offset_end=nxt.char_offset_start,
                )
        else:
            # Last entry — leave page_end=0 (unknown)
            pass

    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_protocol_index(
    pdf_path: str | Path,
) -> ProtocolIndex:
    """Extract a navigable index from a protocol PDF.

    Args:
        pdf_path: Path to the protocol PDF file.

    Returns:
        :class:`ProtocolIndex` with TOC entries, section headers, and
        content spans.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_texts: list[tuple[int, str]] = []
    toc_page_nums: list[int] = []
    toc_entries: list[TOCEntry] = []
    _pymupdf4llm_text: Optional[str] = None

    import fitz  # PTCV-208: lazy import to avoid requiring PyMuPDF at package level

    doc = fitz.open(str(pdf_path))
    try:
        page_count = len(doc)

        for i in range(page_count):
            page_num = i + 1  # 1-based
            text = doc[i].get_text() or ""
            page_texts.append((page_num, text))

            # Detect TOC pages (only in first ~15 pages)
            if page_num <= 15 and _is_toc_page(text):
                toc_count = _count_toc_lines(text)
                if toc_count >= 3:
                    toc_page_nums.append(page_num)
                    toc_entries.extend(_parse_toc_page(text))

        # Also check pages adjacent to TOC pages (multi-page TOC)
        if toc_page_nums:
            last_toc = max(toc_page_nums)
            for check_page in range(
                last_toc + 1, min(last_toc + 4, page_count + 1)
            ):
                idx = check_page - 1
                text = page_texts[idx][1]
                toc_count = _count_toc_lines(text)
                if toc_count >= 3:
                    toc_page_nums.append(check_page)
                    toc_entries.extend(_parse_toc_page(text))
                else:
                    break  # Stop if no more TOC lines

        # PTCV-170: Get pymupdf4llm structured markdown while doc is open
        try:
            import pymupdf4llm

            from ptcv.extraction.markdown_normalizer import (
                normalize_markdown,
            )

            md_raw = pymupdf4llm.to_markdown(doc)
            normalized = normalize_markdown(md_raw, extract_toc=False)
            _pymupdf4llm_text = normalized.text
        except Exception as exc:
            logger.warning(
                "pymupdf4llm markdown extraction failed: %s "
                "— using plain text for full_text",
                exc,
            )
    finally:
        doc.close()

    toc_found = len(toc_page_nums) > 0

    if not toc_found:
        logger.warning(
            "No Table of Contents found in %s — using body header detection only",
            pdf_path.name,
        )

    # PTCV-127: Strip repeating page headers/footers before assembly.
    # This removes sponsor headers, confidentiality notices, copyright
    # lines, and document reference footers that repeat across pages.
    # PTCV-189: Detect patterns first so we can reuse them for
    # pymupdf4llm text (which bypasses per-page stripping).
    _hdr_pats, _ftr_pats = _detect_repeating_lines(page_texts)
    page_texts = _strip_page_headers_footers(page_texts)

    # Build full text and page offset map (PTCV-109)
    page_separator = "\n"
    page_offset_map: dict[int, int] = {}
    offset = 0
    for page_num, text in page_texts:
        page_offset_map[page_num] = offset
        offset += len(text) + len(page_separator)
    full_text = page_separator.join(text for _, text in page_texts)

    # Detect body headers — skip front matter, TOC pages, and filter noise
    toc_numbers = {e.number for e in toc_entries} if toc_entries else None
    # Body content starts after the last TOC page (skip front matter)
    body_start = max(toc_page_nums) + 1 if toc_page_nums else 1
    headers = _detect_body_headers(
        page_texts, full_text,
        toc_pages=toc_page_nums if toc_found else None,
        toc_numbers=toc_numbers,
        body_start_page=body_start,
        page_offsets=page_offset_map,
    )

    # Resolve TOC entry pages from body headers
    if toc_entries:
        toc_entries = _resolve_toc_pages(toc_entries, headers)

    # Build content spans from body headers
    content_spans = _resolve_content_spans(headers, full_text)

    # If no TOC was found, synthesise TOC entries from body headers
    if not toc_found and headers:
        toc_entries = [
            TOCEntry(
                level=h.level,
                number=h.number,
                title=h.title,
                page_start=h.page,
                char_offset_start=h.char_offset,
            )
            for h in headers
        ]
        # Compute page_end from next entry
        for i in range(len(toc_entries) - 1):
            toc_entries[i] = dataclasses.replace(
                toc_entries[i],
                page_end=toc_entries[i + 1].page_start,
                char_offset_end=toc_entries[i + 1].char_offset_start,
            )

    logger.info(
        "ProtocolIndex for %s: %d pages, %d TOC entries, "
        "%d body headers, TOC found=%s",
        pdf_path.name,
        page_count,
        len(toc_entries),
        len(headers),
        toc_found,
    )

    # PTCV-170: If pymupdf4llm markdown is available, use it as full_text
    # for richer downstream consumption (tables, headings, structure).
    # Re-resolve char_offsets and content_spans against the markdown.
    if _pymupdf4llm_text is not None:
        # PTCV-189: Strip detected header/footer patterns from the
        # pymupdf4llm text.  The per-page stripping at line 845
        # only cleaned page_texts; the pymupdf4llm markdown is a
        # separate extraction that still contains them.
        full_text = _strip_patterns_from_text(
            _pymupdf4llm_text, _hdr_pats | _ftr_pats,
        )
        # Re-resolve header char_offsets against the markdown text
        for header in headers:
            search_pat = (
                re.escape(header.number) + r"\.?\s+"
                + re.escape(header.title[:40])
            )
            offset_match = re.search(search_pat, full_text)
            header.char_offset = (
                offset_match.start() if offset_match else -1
            )
        # Re-build content_spans from the markdown full_text
        content_spans = _resolve_content_spans(headers, full_text)
    else:
        # PTCV-129: Clean plain-text full_text of TOC residue so the
        # query extractor's full-text fallback doesn't pick up TOC
        # entries.  Only needed when using plain text (pymupdf4llm
        # normalizer already strips TOC dot-leaders).
        full_text = strip_toc_lines(full_text)

    # PTCV-237: Universal table extraction for Query Pipeline.
    # Run Camelot + pdfplumber on all pages to produce structured
    # tables accessible to downstream QueryExtractor and SoA stages.
    structured_tables: list[dict] = []
    try:
        from ptcv.extraction.table_extractor import extract_all_tables
        import json as _json
        import uuid as _uuid

        pdf_bytes = pdf_path.read_bytes()
        run_id = str(_uuid.uuid4())
        extracted = extract_all_tables(
            pdf_bytes=pdf_bytes,
            page_count=page_count,
            run_id=run_id,
            registry_id=pdf_path.stem.split("_")[0],
            source_sha256="",
        )
        for et in extracted:
            structured_tables.append({
                "page_number": et.page_number,
                "header_row": et.header_row,
                "data_rows": et.data_rows,
                "extractor_used": et.extractor_used,
                "table_index": et.table_index,
            })
        if structured_tables:
            logger.info(
                "Universal table extraction: %d tables for %s",
                len(structured_tables), pdf_path.name,
            )
    except Exception as exc:
        logger.debug(
            "Universal table extraction failed: %s — "
            "continuing without structured tables",
            exc,
        )

    # PTCV-261: Diagram detection via DiagramFinder.
    # Vector-based detection (rects/lines/curves) runs unconditionally
    # (zero cost). Vision API fallback for rasterized diagrams is gated
    # behind PTCV_ENABLE_DIAGRAM_VISION=1 to control API costs.
    detected_diagrams: list[dict] = []
    try:
        import os as _os
        import pdfplumber

        from ptcv.extraction.diagram_finder import DiagramFinder

        enable_vision = _os.environ.get(
            "PTCV_ENABLE_DIAGRAM_VISION", ""
        ) == "1"
        finder = DiagramFinder(enable_vision_fallback=enable_vision)

        with pdfplumber.open(str(pdf_path)) as plumber_pdf:
            for plumber_page in plumber_pdf.pages:
                page_diagrams = finder.find_diagrams(plumber_page)
                for diag in page_diagrams:
                    detected_diagrams.append(diag.to_dict())
        if detected_diagrams:
            logger.info(
                "Diagram detection: %d diagram(s) for %s",
                len(detected_diagrams), pdf_path.name,
            )
    except Exception as exc:
        logger.debug(
            "Diagram detection failed: %s — "
            "continuing without diagrams",
            exc,
        )

    # PTCV-263: Figure detection + Vision captioning.
    # FigureDetector scans each page for embedded images and large
    # drawing regions (zero cost). Vision captioning via FigureCaptioner
    # is gated behind PTCV_ENABLE_FIGURE_VISION=1 to control API costs.
    detected_figures: list[dict] = []
    try:
        import os as _os2

        from ptcv.extraction.figure_detector import detect_figures

        pdf_bytes_fig = pdf_path.read_bytes()
        detection_result = detect_figures(pdf_bytes_fig)

        if detection_result.figures:
            enable_figure_vision = _os2.environ.get(
                "PTCV_ENABLE_FIGURE_VISION", ""
            ) == "1"

            if enable_figure_vision:
                from ptcv.extraction.figure_captioner import FigureCaptioner

                captioner = FigureCaptioner()
                caption_result = captioner.caption_figures(
                    detection_result.figures, pdf_bytes_fig,
                )
                for fc in caption_result.captions:
                    fig = fc.figure
                    detected_figures.append({
                        "page_number": fig.page_number,
                        "bbox": {
                            "x0": fig.bbox.x0,
                            "y0": fig.bbox.y0,
                            "x1": fig.bbox.x1,
                            "y1": fig.bbox.y1,
                        },
                        "figure_type_hint": fig.figure_type_hint,
                        "detection_method": fig.detection_method,
                        "caption": fc.caption,
                    })
                if caption_result.api_calls_made:
                    logger.info(
                        "Figure captioning: %d figures, %d API calls, "
                        "%d tokens for %s",
                        caption_result.figures_captioned,
                        caption_result.api_calls_made,
                        caption_result.total_token_cost,
                        pdf_path.name,
                    )
            else:
                for fig in detection_result.figures:
                    detected_figures.append({
                        "page_number": fig.page_number,
                        "bbox": {
                            "x0": fig.bbox.x0,
                            "y0": fig.bbox.y0,
                            "x1": fig.bbox.x1,
                            "y1": fig.bbox.y1,
                        },
                        "figure_type_hint": fig.figure_type_hint,
                        "detection_method": fig.detection_method,
                        "caption": fig.caption,
                    })

            logger.info(
                "Figure detection: %d figure(s) for %s",
                len(detected_figures), pdf_path.name,
            )
    except Exception as exc:
        logger.debug(
            "Figure detection failed: %s — "
            "continuing without figures",
            exc,
        )

    # PTCV-263: Layout graph — spatial relationships between text
    # blocks, tables, figures, and footnotes on each page.
    layout_graph_dict: Optional[dict] = None
    try:
        from ptcv.extraction.layout_graph import build_document_layout

        pages_data: list[dict] = []
        for page_num, text in page_texts:
            # Build text blocks from page text (split by double newline)
            blocks = [
                {"text": block.strip(), "block_index": i}
                for i, block in enumerate(text.split("\n\n"))
                if block.strip()
            ]
            # Filter tables and figures for this page
            page_tables = [
                t for t in structured_tables
                if t.get("page_number") == page_num
            ]
            page_figures = [
                f for f in detected_figures
                if f.get("page_number") == page_num
            ]
            pages_data.append({
                "page_number": page_num,
                "text_blocks": blocks,
                "tables": page_tables,
                "figures": page_figures,
            })

        doc_layout = build_document_layout(pages_data)

        layout_graph_dict = {
            "page_count": doc_layout.page_count,
            "total_nodes": doc_layout.total_nodes,
            "total_edges": doc_layout.total_edges,
            "pages": [
                {
                    "page_number": pl.page_number,
                    "nodes": [
                        {
                            "node_id": n.node_id,
                            "element_type": n.element_type.value,
                            "text_preview": n.text_preview,
                            "metadata": n.metadata,
                        }
                        for n in pl.nodes
                    ],
                    "edges": [
                        {
                            "source_id": e.source_id,
                            "target_id": e.target_id,
                            "relationship": e.relationship.value,
                            "confidence": e.confidence,
                        }
                        for e in pl.edges
                    ],
                }
                for pl in doc_layout.pages
            ],
        }

        if doc_layout.total_edges > 0:
            logger.info(
                "Layout graph: %d nodes, %d edges across %d pages for %s",
                doc_layout.total_nodes,
                doc_layout.total_edges,
                doc_layout.page_count,
                pdf_path.name,
            )
    except Exception as exc:
        logger.debug(
            "Layout graph construction failed: %s — "
            "continuing without layout graph",
            exc,
        )

    return ProtocolIndex(
        source_path=str(pdf_path),
        page_count=page_count,
        toc_entries=toc_entries,
        section_headers=headers,
        content_spans=content_spans,
        full_text=full_text,
        toc_found=toc_found,
        toc_pages=sorted(set(toc_page_nums)),
        tables=structured_tables,
        diagrams=detected_diagrams,
        figures=detected_figures,
        layout_graph=layout_graph_dict,
    )
