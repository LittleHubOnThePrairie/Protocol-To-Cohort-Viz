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
from pathlib import Path
from typing import Optional

import pdfplumber

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
    """

    source_path: str
    page_count: int
    toc_entries: list[TOCEntry]
    section_headers: list[SectionHeader]
    content_spans: dict[str, str]
    full_text: str
    toc_found: bool
    toc_pages: list[int]

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
    """Return True if *text* appears to be a TOC page."""
    if _TOC_HEADER_RE.search(text):
        return True
    return False


def _count_toc_lines(text: str) -> int:
    """Count how many lines look like TOC entries."""
    count = 0
    for line in text.split("\n"):
        line = line.strip()
        if _TOC_LINE_RE.match(line) or _TOC_LINE_SIMPLE_RE.match(line):
            count += 1
    return count


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

            # Find character offset in full_text
            # Build a search pattern for this specific heading
            search_pat = re.escape(raw_num) + r"\.?\s+" + re.escape(title[:40])
            offset_match = re.search(search_pat, full_text)
            char_offset = offset_match.start() if offset_match else -1

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

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            page_num = i + 1  # 1-based
            text = page.extract_text() or ""
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
            for check_page in range(last_toc + 1, min(last_toc + 4, page_count + 1)):
                idx = check_page - 1
                text = page_texts[idx][1]
                toc_count = _count_toc_lines(text)
                if toc_count >= 3:
                    toc_page_nums.append(check_page)
                    toc_entries.extend(_parse_toc_page(text))
                else:
                    break  # Stop if no more TOC lines

    toc_found = len(toc_page_nums) > 0

    if not toc_found:
        logger.warning(
            "No Table of Contents found in %s — using body header detection only",
            pdf_path.name,
        )

    # Build full text (page separator for offset tracking)
    page_separator = "\n"
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

    return ProtocolIndex(
        source_path=str(pdf_path),
        page_count=page_count,
        toc_entries=toc_entries,
        section_headers=headers,
        content_spans=content_spans,
        full_text=full_text,
        toc_found=toc_found,
        toc_pages=sorted(set(toc_page_nums)),
    )
