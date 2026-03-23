"""Query-driven extraction engine (PTCV-91, PTCV-98).

Runs Appendix B queries against protocol sections to extract
structured answers.  Given ``ProtocolIndex`` (content),
``MatchResult`` (routing), and ``AppendixBQuery`` (questions),
dispatches each query to a type-specific extractor.

Deterministic type-specific strategies:

    text_long   → paragraph relevance scoring + excerpt
    text_short  → label-value regex / first relevant sentence
    identifier  → NCT / EudraCT / protocol number regexes
    date        → date format regexes (4 patterns)
    list        → bullet / numbered item regex + criteria boundary
    table       → pipe-delimited / aligned column detection
    numeric     → sample size / power / significance regexes
    enum        → phase / design keyword matching
    statement   → regulatory keyword density scoring

Routing uses ``get_queries_by_schema_section()`` to map classifier
section codes (B.1-B.16) to their corresponding queries.

Optional LLM transformation (PTCV-98, PTCV-111): when
``enable_transformation=True`` and an Anthropic API key is
available, all scoped matches are reframed by Claude Sonnet to
directly address the Appendix B query intent while preserving
factual fidelity.  Queries are processed concurrently (up to 8
threads) to minimize wall-clock time from API calls.

Risk tier: LOW — read-only extraction, optional LLM calls.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from ptcv.ich_parser.query_schema import (
    AppendixBQuery,
    get_queries_by_schema_section,
    load_query_schema,
)
from ptcv.ich_parser.section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMapping,
)
from ptcv.ich_parser.summarization_matcher import (
    SubSectionDef,
    build_subsection_registry,
    get_subsections_for_parent,
)
from ptcv.ich_parser.toc_extractor import ProtocolIndex

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QueryExtraction:
    """A single extracted answer for an Appendix B query.

    Attributes:
        query_id: Unique query identifier (e.g. ``"B.1.1.q1"``).
        section_id: Appendix B sub-section (e.g. ``"B.1.1"``).
        content: Extracted answer text (may be LLM-transformed).
        confidence: Confidence score (0.0–1.0).
        extraction_method: Strategy used (``"regex"``,
            ``"heuristic"``, ``"passthrough"``,
            ``"relevance_excerpt"``, ``"table_detection"``,
            ``"unscoped_search"``, ``"llm_transform"``).
        source_section: Protocol section number(s) searched.
        verbatim_content: Pre-transformation extraction text.
            Populated only when ``extraction_method`` is
            ``"llm_transform"``; empty string otherwise.
    """

    query_id: str
    section_id: str
    content: str
    confidence: float
    extraction_method: str
    source_section: str
    verbatim_content: str = ""


@dataclasses.dataclass(frozen=True)
class ExtractionGap:
    """An unanswered query.

    Attributes:
        query_id: Unique query identifier.
        section_id: Appendix B sub-section.
        reason: Why extraction failed (``"no_match"``,
            ``"low_confidence"``, ``"unmapped_section"``,
            ``"no_content"``).
    """

    query_id: str
    section_id: str
    reason: str


@dataclasses.dataclass
class ExtractionResult:
    """Aggregate result from :meth:`QueryExtractor.extract`.

    Attributes:
        extractions: Successful extractions.
        gaps: Unanswered queries.
        coverage: Fraction answered with confidence >= threshold.
        total_queries: Total queries processed.
        answered_queries: Queries with confidence >= threshold.
    """

    extractions: list[QueryExtraction]
    gaps: list[ExtractionGap]
    coverage: float
    total_queries: int
    answered_queries: int


# -----------------------------------------------------------------------
# Regex constants
# -----------------------------------------------------------------------

_NCT_RE = re.compile(r"NCT\d{8}", re.IGNORECASE)
_EUDRACT_RE = re.compile(
    r"\d{4}-\d{6}-\d{2}"
)
_PROTOCOL_NUM_RE = re.compile(
    r"(?:Protocol\s*(?:No\.?|Number|#|ID)\s*[:.]?\s*)"
    r"([A-Z0-9][\w\-/]{3,})",
    re.IGNORECASE,
)

_DATE_PATTERNS: list[re.Pattern[str]] = [
    # "24 January 2023", "3 Mar 2021"
    re.compile(
        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{4}\b",
        re.IGNORECASE,
    ),
    # "January 24, 2023"
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    # "2023-01-24"
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # "01/24/2023" or "24/01/2023"
    re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
]

_SAMPLE_SIZE_RE = re.compile(
    r"(?:sample\s+size|(?:total|planned)\s+(?:enrollment|enrolment)"
    r"|(?:number|n)\s+(?:of\s+)?(?:participants?|subjects?|patients?))"
    r"[\s:=]*(?:(?:of|is|will\s+be|approximately|~|≈)\s*)?"
    r"(\d[\d,]*)",
    re.IGNORECASE,
)
_POWER_RE = re.compile(
    r"(?:power|statistical\s+power)"
    r"[\s:=]*(?:of\s+)?(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)
_SIGNIFICANCE_RE = re.compile(
    r"(?:significance\s+level|alpha|type\s+I\s+error|"
    r"one-sided|two-sided|p[\s<]*)"
    r"[\s:=]*(\d+\.\d+)",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(
    r"(\d[\d,]*\.?\d*)\s*"
    r"(%|mg|kg|mL|L|g|μg|mcg|IU|mmol|patients?|subjects?"
    r"|participants?|years?|months?|weeks?|days?|hours?)?",
    re.IGNORECASE,
)

_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-•●○▪◦]\s+|\d+[\.\)]\s+|[a-z][\.\)]\s+)",
    re.MULTILINE,
)

_TABLE_PIPE_RE = re.compile(
    r"^.*\|.*\|.*$", re.MULTILINE
)
_TABLE_ALIGNED_RE = re.compile(
    r"^.{10,}\s{3,}.{10,}\s{3,}.{10,}$",
    re.MULTILINE,
)

_STATEMENT_KEYWORDS: list[str] = [
    "in compliance with",
    "in accordance with",
    "good clinical practice",
    "gcp",
    "regulatory requirement",
    "applicable law",
    "declaration of helsinki",
    "informed consent",
    "irb",
    "iec",
    "ethics committee",
    "institutional review board",
    "confidentiality",
    "data protection",
    "retention",
    "archiving",
    "source data verification",
    "monitoring",
    "audit",
    "inspection",
    "direct access",
    "noncompliance",
    "deviation",
]

# Confidence multipliers
_REVIEW_PENALTY = 0.85
_UNSCOPED_PENALTY = 0.70

# -----------------------------------------------------------------------
# Criteria boundary detection
# -----------------------------------------------------------------------

_INCLUSION_HEADINGS = re.compile(
    r"(?:^|\n)\s*(?:\d+[\.\d]*\.?\s+)?"
    r"(?:inclusion\s+criteria|eligibility\s+criteria)",
    re.IGNORECASE,
)
_EXCLUSION_HEADINGS = re.compile(
    r"(?:^|\n)\s*(?:\d+[\.\d]*\.?\s+)?"
    r"(?:exclusion\s+criteria)",
    re.IGNORECASE,
)

# Numbered section heading pattern for heading-based subsection
# splitting (PTCV-144). Matches "11.1 Inclusion Criteria",
# "5.2.1 Exclusion Criteria", "3. Study Objectives", etc.
_NUMBERED_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(\d+(?:\.\d+)*\.?)\s+([A-Z][^\n]{3,})",
    re.MULTILINE,
)

# PTCV-171: Markdown heading pattern for subsection splitting.
# Matches "## 5.1 Inclusion Criteria", "### 5.2.1 Exclusion", etc.
# Captures heading level (hash count) and section number + title.
_MD_SECTION_HEADING_RE = re.compile(
    r"^#{1,6}\s+(\d+(?:\.\d+)*\.?)\s+([^\n]{3,})$",
    re.MULTILINE,
)


def _match_heading_to_subsection(
    heading_title: str,
    sub_defs: list[SubSectionDef],
) -> str | None:
    """Match a heading title to a sub-section by keyword overlap.

    Returns the sub-section code if the best overlap score is ≥ 0.30,
    otherwise *None*.
    """
    title_lower = heading_title.lower()
    best_code: str | None = None
    best_score = 0.0
    for sd in sub_defs:
        words = set(re.findall(r"[a-z]{3,}", sd.name.lower()))
        if not words:
            continue
        hits = sum(1 for w in words if w in title_lower)
        score = hits / len(words)
        if score > best_score:
            best_score = score
            best_code = sd.code
    if best_score >= 0.30:
        return best_code
    return None


def _extract_criteria_section(
    text: str, criteria_type: str
) -> str:
    """Extract inclusion or exclusion criteria sub-section.

    Splits on sub-headings to isolate inclusion from exclusion.
    Returns the full text if no boundary detected.
    """
    if criteria_type == "inclusion":
        start_pat = _INCLUSION_HEADINGS
        end_pat = _EXCLUSION_HEADINGS
    else:
        start_pat = _EXCLUSION_HEADINGS
        end_pat = _INCLUSION_HEADINGS

    start_match = start_pat.search(text)
    if not start_match:
        return text

    start_pos = start_match.start()
    remaining = text[start_pos:]

    end_match = end_pat.search(remaining)
    if end_match:
        return remaining[: end_match.start()].strip()
    return remaining.strip()


# -----------------------------------------------------------------------
# Type-specific extractors
# -----------------------------------------------------------------------


# Maximum characters for text_long excerpt output (PTCV-109, PTCV-144).
_TEXT_LONG_MAX_CHARS = 4000

# Maximum characters sent to LLM for transformation (PTCV-144).
_LLM_TRANSFORM_MAX_CHARS = 10000

# Minimum paragraph length to consider for relevance scoring.
_MIN_PARAGRAPH_LEN = 40

# Maximum characters for unscoped (full-doc) fallback (PTCV-157).
# Prevents full-document dumps (up to 500K chars) when section matching
# fails.  Content is pre-filtered via _select_relevant_paragraphs to
# keep only the most query-relevant chunks.
_UNSCOPED_MAX_CHARS = 10000

# PTCV-182: Route concatenation caps to prevent pathological merges.
_MAX_SECTIONS_PER_ROUTE = 5
_MAX_ROUTE_CHARS = 20000

# PTCV-182: Ordinal ranking for MatchConfidence (higher = better).
_CONFIDENCE_RANK: dict[MatchConfidence, int] = {
    MatchConfidence.HIGH: 3,
    MatchConfidence.REVIEW: 2,
    MatchConfidence.LOW: 1,
}

# PTCV-183: Refusal detection patterns for LLM transform.
# Lowercased prefixes/phrases that indicate a model refusal
# rather than genuine transformed content.
_REFUSAL_PATTERNS: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "i apologize",
    "i'm sorry",
    "i am sorry",
    "i'm not able",
    "i am not able",
    "as an ai",
    "as a language model",
    "i don't have access",
    "i do not have access",
    "unfortunately, i cannot",
    "unfortunately, i can't",
    "i cannot provide",
    "i can't provide",
    "i cannot process",
    "i can't process",
    "content not available",
)


def _is_refusal(text: str) -> bool:
    """Return *True* if *text* looks like an LLM refusal.

    Checks whether the text begins with or contains common
    refusal phrases.  Only the first 200 characters are
    inspected to avoid false positives on long legitimate
    content that coincidentally mentions these phrases.
    """
    prefix = text[:200].lower().strip()
    return any(prefix.startswith(p) for p in _REFUSAL_PATTERNS)


_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "what", "that", "this",
    "with", "from", "have", "has", "will", "been", "should",
    "there", "their", "which", "each",
})


def _query_keywords(query: AppendixBQuery) -> set[str]:
    """Extract meaningful keywords from a query's text."""
    words = set(
        w.lower()
        for w in re.findall(r"[a-z]{3,}", query.query_text.lower())
    )
    return words - _STOP_WORDS


def _score_paragraph(
    paragraph: str, keywords: set[str]
) -> float:
    """Score a paragraph by keyword overlap with query.

    Returns a value in ``[0.0, 1.0]``.
    """
    if not keywords:
        return 0.5
    para_lower = paragraph.lower()
    hits = sum(1 for w in keywords if w in para_lower)
    return hits / len(keywords)


def _select_relevant_paragraphs(
    text: str,
    query: AppendixBQuery,
    max_chars: int = _TEXT_LONG_MAX_CHARS,
) -> tuple[str, float]:
    """Select the most query-relevant paragraphs from *text*.

    Splits *text* on blank lines, scores each paragraph by
    keyword overlap with the query, and returns the top-scoring
    paragraphs up to *max_chars*.

    Returns:
        ``(excerpt, overlap_score)`` where *overlap_score* is the
        keyword overlap of the top-scoring paragraph.
    """
    keywords = _query_keywords(query)

    # Split on blank lines (two or more newlines)
    paragraphs = re.split(r"\n\s*\n", text.strip())
    # Filter out tiny fragments (headers, page numbers, etc.)
    paragraphs = [
        p.strip() for p in paragraphs
        if len(p.strip()) >= _MIN_PARAGRAPH_LEN
    ]

    if not paragraphs:
        # Fallback: return truncated text
        return text[:max_chars].strip(), 0.0

    # Score and rank
    scored = [
        (_score_paragraph(p, keywords), i, p)
        for i, p in enumerate(paragraphs)
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))

    best_overlap = scored[0][0] if scored else 0.0

    # Collect top paragraphs in original document order
    selected_indices: list[int] = []
    char_count = 0
    for _score, idx, para in scored:
        if char_count + len(para) > max_chars and selected_indices:
            break
        selected_indices.append(idx)
        char_count += len(para)

    # Restore original order for readability
    selected_indices.sort()
    excerpt = "\n\n".join(paragraphs[i] for i in selected_indices)

    return excerpt, best_overlap


def _extract_text_long(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Extract query-relevant paragraphs with confidence scoring.

    Instead of returning the full scoped text (passthrough), selects
    the most relevant paragraphs based on keyword overlap with the
    query (PTCV-109).
    """
    if not text.strip():
        return "", 0.0, "passthrough"

    # Short content: return as-is (no reduction needed)
    if len(text) <= _TEXT_LONG_MAX_CHARS:
        keywords = _query_keywords(query)
        if not keywords:
            return text.strip(), 0.60, "passthrough"
        text_lower = text.lower()
        hits = sum(1 for w in keywords if w in text_lower)
        overlap = hits / len(keywords)
        confidence = 0.45 + (overlap * 0.40)
        return text.strip(), round(confidence, 4), "passthrough"

    # Long content: select relevant paragraphs (PTCV-109)
    excerpt, overlap = _select_relevant_paragraphs(text, query)
    if not excerpt:
        return "", 0.0, "passthrough"

    confidence = 0.45 + (overlap * 0.40)
    return excerpt, round(confidence, 4), "relevance_excerpt"


def _extract_text_short(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Label-value regex or first relevant sentence."""
    if not text.strip():
        return "", 0.0, "heuristic"

    # Try label-value patterns: "Title: ...", "Sponsor: ..."
    label_patterns = [
        r"(?:protocol\s+)?title\s*[:]\s*(.+?)(?:\n|$)",
        r"(?:sponsor|company)\s*[:]\s*(.+?)(?:\n|$)",
        r"(?:name|address)\s*[:]\s*(.+?)(?:\n|$)",
        r"(?:investigator|person)\s*[:]\s*(.+?)(?:\n|$)",
    ]
    for pat in label_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            if len(value) > 5:
                return value, 0.92, "regex"

    # Fallback: first non-empty sentence (up to 300 chars)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 10:
            return sent[:300], 0.65, "heuristic"

    return text[:300].strip(), 0.50, "heuristic"


def _extract_identifier(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """NCT, EudraCT, or protocol number regexes."""
    if not text.strip():
        return "", 0.0, "regex"

    # NCT number
    m = _NCT_RE.search(text)
    if m:
        return m.group(0), 0.95, "regex"

    # EudraCT number
    m = _EUDRACT_RE.search(text)
    if m:
        return m.group(0), 0.92, "regex"

    # Protocol number
    m = _PROTOCOL_NUM_RE.search(text)
    if m:
        return m.group(1), 0.85, "regex"

    # Fallback: look for any alphanumeric identifier pattern
    m = re.search(
        r"\b([A-Z]{2,}[\-/]?\d{3,}[\w\-/]*)\b", text
    )
    if m:
        return m.group(1), 0.70, "regex"

    return "", 0.0, "regex"


def _extract_date(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Date format regexes."""
    if not text.strip():
        return "", 0.0, "regex"

    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0), 0.92, "regex"

    return "", 0.0, "regex"


def _extract_list(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Bullet/numbered item extraction with criteria boundary."""
    if not text.strip():
        return "", 0.0, "heuristic"

    # Detect inclusion vs exclusion criteria queries
    q_lower = query.query_text.lower()
    if "inclusion" in q_lower:
        text = _extract_criteria_section(text, "inclusion")
    elif "exclusion" in q_lower:
        text = _extract_criteria_section(text, "exclusion")

    # Find list items
    items = _LIST_ITEM_RE.findall(text)
    if len(items) >= 2:
        # Extract lines that start with list markers
        lines: list[str] = []
        for line in text.split("\n"):
            if _LIST_ITEM_RE.match(line):
                lines.append(line.strip())
        if lines:
            result = "\n".join(lines)
            confidence = min(0.92, 0.65 + len(lines) * 0.03)
            return result, round(confidence, 4), "heuristic"

    # Fallback: return text if it has content
    if len(text.strip()) > 20:
        return text.strip(), 0.50, "heuristic"

    return "", 0.0, "heuristic"


def _extract_table(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Pipe-delimited or aligned column detection."""
    if not text.strip():
        return "", 0.0, "table_detection"

    # Pipe-delimited tables
    pipe_lines = _TABLE_PIPE_RE.findall(text)
    if len(pipe_lines) >= 2:
        return "\n".join(pipe_lines), 0.85, "table_detection"

    # Aligned columns (3+ columns with wide whitespace)
    aligned = _TABLE_ALIGNED_RE.findall(text)
    if len(aligned) >= 3:
        return "\n".join(aligned), 0.65, "table_detection"

    # No table found — return text as fallback
    if len(text.strip()) > 20:
        return text.strip(), 0.50, "table_detection"

    return "", 0.0, "table_detection"


def _extract_numeric(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Sample size, power, significance, or generic numeric."""
    if not text.strip():
        return "", 0.0, "regex"

    q_lower = query.query_text.lower()

    # Sample size
    if "sample size" in q_lower or "number" in q_lower:
        m = _SAMPLE_SIZE_RE.search(text)
        if m:
            return m.group(0).strip(), 0.92, "regex"

    # Power
    if "power" in q_lower:
        m = _POWER_RE.search(text)
        if m:
            return m.group(0).strip(), 0.90, "regex"

    # Significance level
    if "significance" in q_lower or "alpha" in q_lower:
        m = _SIGNIFICANCE_RE.search(text)
        if m:
            return m.group(0).strip(), 0.90, "regex"

    # Generic numeric with unit
    m = _NUMERIC_RE.search(text)
    if m:
        return m.group(0).strip(), 0.75, "regex"

    return "", 0.0, "regex"


def _extract_enum(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Phase or design keyword matching."""
    if not text.strip():
        return "", 0.0, "heuristic"

    # Phase detection
    m = re.search(
        r"\bPhase\s*([1-4I]+[abAB]?(?:/[1-4I]+[abAB]?)?)\b",
        text,
    )
    if m:
        return f"Phase {m.group(1)}", 0.90, "heuristic"

    # Design type keywords
    design_keywords = [
        "double-blind",
        "single-blind",
        "open-label",
        "placebo-controlled",
        "randomised",
        "randomized",
        "parallel",
        "crossover",
        "adaptive",
        "non-inferiority",
        "superiority",
        "equivalence",
    ]
    found = [
        kw for kw in design_keywords if kw in text.lower()
    ]
    if found:
        return ", ".join(found), 0.80, "heuristic"

    return "", 0.0, "heuristic"


def _extract_statement(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Regulatory keyword density scoring per sentence."""
    if not text.strip():
        return "", 0.0, "heuristic"

    sentences = re.split(r"(?<=[.!?])\s+", text)
    best_sent = ""
    best_score = 0.0

    for sent in sentences:
        sent_lower = sent.lower()
        hits = sum(
            1 for kw in _STATEMENT_KEYWORDS if kw in sent_lower
        )
        if hits > best_score:
            best_score = hits
            best_sent = sent.strip()

    if best_score >= 3:
        confidence = min(0.92, 0.60 + best_score * 0.08)
        return best_sent, round(confidence, 4), "heuristic"
    elif best_score >= 1:
        confidence = 0.60 + best_score * 0.08
        return best_sent, round(confidence, 4), "heuristic"

    return "", 0.0, "heuristic"


# -----------------------------------------------------------------------
# Citation detection and filtering (PTCV-99)
# -----------------------------------------------------------------------
# Detects full bibliographic references, collects them for B.2.7,
# and strips them from content destined for other sections.
# Short parenthetical references like "(Smith et al., 2023)" or
# "[1]" are preserved inline.

# Full citation patterns:
# - APA:      "Author, A. B. (Year). Title. Journal, Vol(Issue), Pages."
# - Vancouver: "1. Author AB. Title. Journal. Year;Vol(Issue):Pages."
# - Numbered:  "[1] Author. Title. Journal 2023; ..."
# - DOI refs:  "... doi:10.xxxx/yyyy" or "https://doi.org/..."

# A full citation is >=60 chars and matches at least one pattern.
_CITATION_MIN_LEN = 60

# APA-style: Author, A. B. (Year).
_APA_RE = re.compile(
    r"[A-Z][a-z]+,?\s+[A-Z]\..*?\(\d{4}\)\.\s+.+?\.",
)

# Vancouver/numbered: "1." or "[1]" at start with author initials
_VANCOUVER_RE = re.compile(
    r"(?:^\d{1,3}[.)]\s+|^\[\d{1,3}\]\s+)"
    r"[A-Z][a-z]+\s+[A-Z]{1,3}[,.]",
    re.MULTILINE,
)

# DOI anywhere in a line
_DOI_RE = re.compile(
    r"(?:doi:\s*10\.\d{4,}/\S+|"
    r"https?://doi\.org/10\.\d{4,}/\S+)",
    re.IGNORECASE,
)

# PMID reference
_PMID_RE = re.compile(r"PMID:\s*\d+", re.IGNORECASE)

# Parenthetical references to PRESERVE (not strip).
# e.g. "(Smith et al., 2023)", "(Author, Year)", "[1-3]", "[12]"
_PARENTHETICAL_RE = re.compile(
    r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}\)"
    r"|\[\d{1,3}(?:[,\-–]\s*\d{1,3})*\]"
)


def _is_full_citation(line: str) -> bool:
    """Return ``True`` if *line* is a full bibliographic reference.

    Criteria: length >= ``_CITATION_MIN_LEN`` AND matches at least
    one citation pattern (APA, Vancouver, DOI, PMID).  Short
    parenthetical references are excluded.
    """
    if len(line.strip()) < _CITATION_MIN_LEN:
        return False
    if _APA_RE.search(line):
        return True
    if _VANCOUVER_RE.search(line):
        return True
    if _DOI_RE.search(line):
        return True
    if _PMID_RE.search(line):
        return True
    return False


def extract_citations(text: str) -> list[str]:
    """Extract full bibliographic references from *text*.

    Splits text into lines and returns lines that match citation
    patterns.  Does NOT return short parenthetical refs.

    Args:
        text: Content to scan for citations.

    Returns:
        List of full citation strings, preserving original order.
    """
    citations: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and _is_full_citation(stripped):
            citations.append(stripped)
    return citations


def filter_citations(text: str) -> str:
    """Remove full bibliographic references from *text*.

    Preserves short parenthetical references like
    ``"(Smith et al., 2023)"`` and ``"[1]"``.

    Args:
        text: Content to filter.

    Returns:
        Text with full citations removed.
    """
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        if line.strip() and _is_full_citation(line.strip()):
            continue
        kept.append(line)
    # Clean up excessive blank lines from removal
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    return result.strip()


# -----------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------

_EXTRACTORS: dict[
    str,
    type[
        object
    ],
] = {}  # unused placeholder for future registry pattern

_TYPE_DISPATCH = {
    "text_long": _extract_text_long,
    "text_short": _extract_text_short,
    "identifier": _extract_identifier,
    "date": _extract_date,
    "list": _extract_list,
    "table": _extract_table,
    "numeric": _extract_numeric,
    "enum": _extract_enum,
    "statement": _extract_statement,
}


# -----------------------------------------------------------------------
# QueryExtractor
# -----------------------------------------------------------------------


class QueryExtractor:
    """Run Appendix B queries against protocol sections.

    Args:
        confidence_threshold: Minimum confidence for a query to
            count as answered (default ``0.70``).
        enable_transformation: If *True*, all scoped extractions
            are reframed by Claude Sonnet to directly address the
            query intent (PTCV-98, PTCV-111).
        anthropic_api_key: Anthropic API key.  If *None*, reads
            ``ANTHROPIC_API_KEY`` from the environment.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        enable_transformation: bool = False,
        anthropic_api_key: str | None = None,
    ) -> None:
        self._threshold = confidence_threshold
        # PTCV-188: Primary model with graceful degradation.
        self._llm_model = "claude-sonnet-4-6"
        self._fallback_model = "claude-sonnet-4-20250514"
        self._client: Any = None
        self._use_llm = False
        self._transform_calls = 0
        self._refusal_count = 0
        self._fallback_count = 0
        self._transform_lock = threading.Lock()

        if enable_transformation:
            api_key = anthropic_api_key or os.environ.get(
                "ANTHROPIC_API_KEY", ""
            )
            if api_key:
                self._init_llm(api_key)
            else:
                logger.warning(
                    "enable_transformation=True but no "
                    "ANTHROPIC_API_KEY — running verbatim only"
                )

    def _init_llm(self, api_key: str) -> None:
        """Lazy-import Anthropic SDK and create client."""
        try:
            import anthropic
        except ImportError:
            logger.warning(
                "QueryExtractor: 'anthropic' package not installed "
                "— falling back to verbatim-only extraction. "
                "Install with: pip install anthropic"
            )
            return

        self._client = anthropic.Anthropic(api_key=api_key)
        self._use_llm = True
        logger.info(
            "QueryExtractor: LLM transformation enabled "
            "(model=%s)",
            self._llm_model,
        )

    # ------------------------------------------------------------------
    # LLM transformation (PTCV-98)
    # ------------------------------------------------------------------

    # PTCV-186: System prompt for transform calls — establishes
    # document-analysis context to reduce refusal risk on clinical
    # content (defense-in-depth alongside the model pin).
    _TRANSFORM_SYSTEM = (
        "You are a regulatory document reformatting tool. You "
        "reorganize text from publicly available clinical trial "
        "protocol documents (sourced from ClinicalTrials.gov) into "
        "ICH E6(R3) Appendix B structure. This is a read-only "
        "document restructuring task — no medical advice is "
        "generated."
    )

    _TRANSFORM_PROMPT = (
        "You are an ICH E6(R3) regulatory writing specialist.\n\n"
        "TASK: Transform the source content below to directly "
        "address this Appendix B query. Produce text suitable "
        "for an ICH E6(R3) Appendix B protocol section.\n\n"
        "QUERY: {query_text}\n\n"
        "SOURCE CONTENT:\n<source>\n{source}\n</source>\n\n"
        "RULES:\n"
        "1. Reframe content to directly answer the query\n"
        "2. Remove tangential information not relevant to "
        "the query\n"
        "3. Preserve ALL factual claims — numbers, dates, "
        "drug names, doses\n"
        "4. Do NOT introduce any new information not in the "
        "source\n"
        "5. Use formal regulatory tone appropriate for ICH "
        "GCP documentation\n"
        "6. Keep the response concise but complete\n\n"
        'Respond with a single JSON object (no markdown fences):\n'
        '{{"transformed": "your transformed text", '
        '"confidence": 0.85}}\n\n'
        "The confidence should reflect how well the source "
        "material answers the query (0.0 = not at all, "
        "1.0 = perfectly)."
    )

    def _retry_with_fallback(
        self,
        prompt: str,
        query_id: str,
        refusal_reason: str,
    ) -> tuple[str, float] | None:
        """Retry a refused LLM call with the fallback model.

        Returns:
            ``(transformed_text, confidence)`` on success, *None* if
            the fallback also refuses or fails.
        """
        logger.info(
            "PTCV-188: Retrying %s with fallback model %s "
            "(reason=%s)",
            query_id,
            self._fallback_model,
            refusal_reason,
        )
        try:
            resp = self._client.messages.create(
                model=self._fallback_model,
                max_tokens=2048,
                system=self._TRANSFORM_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            stop = getattr(resp, "stop_reason", None)
            if isinstance(stop, str) and stop in (
                "content_filtered",
                "refusal",
            ):
                logger.warning(
                    "PTCV-188: Fallback model also refused "
                    "for %s (stop_reason=%s)",
                    query_id,
                    stop,
                )
                return None

            first_block = resp.content[0]
            if not hasattr(first_block, "text"):
                return None
            raw = first_block.text.strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                cleaned = re.sub(
                    r"```(?:json)?", "", raw
                ).strip()
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    return None

            transformed = parsed.get("transformed", "")
            llm_conf = float(parsed.get("confidence", 0.0))
            if not transformed or _is_refusal(transformed):
                return None

            with self._transform_lock:
                self._fallback_count += 1
                self._transform_calls += 1
            logger.info(
                "PTCV-188: Fallback succeeded for %s",
                query_id,
            )
            return transformed, min(1.0, max(0.0, llm_conf))

        except Exception:
            logger.warning(
                "PTCV-188: Fallback model failed for %s",
                query_id,
                exc_info=True,
            )
            return None

    def _transform_content(
        self,
        source_content: str,
        query: AppendixBQuery,
    ) -> tuple[str, float] | None:
        """Call Claude Sonnet to reframe extracted content.

        Returns:
            ``(transformed_text, llm_confidence)`` on success,
            *None* on any error (caller falls back to verbatim).
        """
        # Select query-relevant portion if source is large
        # (PTCV-109: avoid blind prefix truncation,
        #  PTCV-144: raised from 6000 to _LLM_TRANSFORM_MAX_CHARS).
        if len(source_content) > _LLM_TRANSFORM_MAX_CHARS:
            source_for_llm, _ = _select_relevant_paragraphs(
                source_content, query,
                max_chars=_LLM_TRANSFORM_MAX_CHARS,
            )
        else:
            source_for_llm = source_content

        prompt = self._TRANSFORM_PROMPT.format(
            query_text=query.query_text,
            source=source_for_llm,
        )
        # PTCV-183: Retry with exponential backoff for
        # transient errors (rate limits, timeouts).
        max_retries = 3
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                resp = self._client.messages.create(
                    model=self._llm_model,
                    max_tokens=2048,
                    system=self._TRANSFORM_SYSTEM,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                with self._transform_lock:
                    self._transform_calls += 1

                # PTCV-183: Check stop_reason for content
                # filtering.
                stop = getattr(resp, "stop_reason", None)
                if isinstance(stop, str) and stop in (
                    "content_filtered",
                    "refusal",
                ):
                    # PTCV-188: Retry with fallback model
                    # before giving up.
                    fallback = self._retry_with_fallback(
                        prompt, query.query_id, stop,
                    )
                    if fallback is not None:
                        return fallback
                    with self._transform_lock:
                        self._refusal_count += 1
                    logger.warning(
                        "PTCV-183: LLM stop_reason=%s "
                        "for %s — fallback also failed",
                        stop,
                        query.query_id,
                    )
                    return None

                first_block = resp.content[0]
                if not hasattr(first_block, "text"):
                    return None
                raw = first_block.text.strip()

                # Parse JSON — strip markdown fences if
                # present.
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    cleaned = re.sub(
                        r"```(?:json)?", "", raw
                    ).strip()
                    try:
                        parsed = json.loads(cleaned)
                    except json.JSONDecodeError:
                        logger.warning(
                            "LLM transform: failed to parse "
                            "JSON for %s (attempt %d/%d)",
                            query.query_id,
                            attempt + 1,
                            max_retries,
                        )
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        return None

                transformed = parsed.get("transformed", "")
                llm_conf = float(
                    parsed.get("confidence", 0.0)
                )
                if not transformed:
                    return None

                # PTCV-183: Detect refusal language in the
                # transformed text before accepting it.
                if _is_refusal(transformed):
                    # PTCV-188: Retry with fallback model.
                    fallback = self._retry_with_fallback(
                        prompt, query.query_id, "text_refusal",
                    )
                    if fallback is not None:
                        return fallback
                    with self._transform_lock:
                        self._refusal_count += 1
                    logger.warning(
                        "PTCV-183: LLM refusal in text "
                        "for %s — fallback also failed: "
                        "%.100s",
                        query.query_id,
                        transformed,
                    )
                    return None

                return transformed, min(
                    1.0, max(0.0, llm_conf)
                )

            except Exception as exc:
                last_exc = exc
                # Retry on rate-limit (429) or server errors
                # (5xx); bail on other errors.
                retryable = False
                status = getattr(exc, "status_code", None)
                if status is not None:
                    retryable = status in (429, 500, 502, 529)
                elif "rate" in str(exc).lower():
                    retryable = True
                elif "timeout" in str(exc).lower():
                    retryable = True
                elif "overloaded" in str(exc).lower():
                    retryable = True

                if retryable and attempt < max_retries - 1:
                    delay = (2 ** attempt) + 0.5
                    logger.info(
                        "LLM transform: retrying %s "
                        "(attempt %d/%d, delay=%.1fs): %s",
                        query.query_id,
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue

                logger.warning(
                    "LLM transform failed for %s after "
                    "%d attempt(s) — falling back to "
                    "verbatim: %s",
                    query.query_id,
                    attempt + 1,
                    exc,
                )
                return None

        # Should not reach here, but safety net.
        logger.warning(
            "LLM transform exhausted retries for %s: %s",
            query.query_id,
            last_exc,
        )
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        protocol_index: ProtocolIndex,
        match_result: MatchResult,
        queries: Sequence[AppendixBQuery] | None = None,
        progress_callback: Any = None,
    ) -> ExtractionResult:
        """Run all queries against the protocol.

        Args:
            protocol_index: Navigable document index.
            match_result: Section mapping from
                :class:`SectionMatcher`.
            queries: Pre-loaded queries; loads from YAML if
                *None*.
            progress_callback: Optional ``(done, total)`` callback
                invoked after each query completes (PTCV-124).

        Returns:
            :class:`ExtractionResult` with extractions, gaps,
            and coverage statistics.
        """
        if queries is None:
            queries = load_query_schema()

        # Build sub-section registry for content splitting
        # (PTCV-110).
        registry = build_subsection_registry(queries)

        # Build route map: ich_section_code → concatenated content
        routes = self._build_routes(
            protocol_index,
            match_result,
            subsection_registry=registry,
        )

        # PTCV-155 Tier 3: Inject front-matter as B.1 fallback
        # when no HIGH/REVIEW content was matched for B.1.
        self._inject_front_matter_route(routes, protocol_index)

        # PTCV-230: Inject registry metadata content as
        # supplementary route content for sections where PDF
        # routes are absent or low-confidence.
        self._inject_registry_routes(routes, protocol_index)

        # PTCV-99: Citation extraction and consolidation.
        # Collect full citations from all routes, inject into B.2.7,
        # and strip from non-B.2 routes.
        self._consolidate_citations(routes)

        # PTCV-256: Build embedding-based Layer 2 routes.
        # Scores all content_spans against section centroids to
        # recover from Stage 2 misclassification.
        embedding_routes = self._build_embedding_routes(
            protocol_index,
        )

        extractions: list[QueryExtraction] = []
        gaps: list[ExtractionGap] = []

        # PTCV-258: Pre-load registry metadata for gap recovery.
        _registry_meta: dict | None = None
        try:
            nct_id = self._resolve_nct_id(protocol_index)
            if nct_id:
                from ptcv.registry.metadata_fetcher import (
                    RegistryMetadataFetcher,
                )
                _fetcher = RegistryMetadataFetcher()
                _registry_meta = _fetcher.fetch(nct_id)
        except Exception:
            pass  # Registry unavailable — recovery L3 skipped

        if self._use_llm:
            # Concurrent extraction — LLM calls are I/O-bound
            # so threading gives near-linear speedup.
            max_workers = min(8, len(queries))
            with ThreadPoolExecutor(
                max_workers=max_workers
            ) as pool:
                future_to_query = {
                    pool.submit(
                        self._process_query,
                        q,
                        routes,
                        protocol_index,
                        match_result,
                        embedding_routes,
                    ): q
                    for q in queries
                }
                done_count = 0
                total_queries = len(queries)
                for future in as_completed(future_to_query):
                    q = future_to_query[future]
                    extraction = future.result()
                    if extraction is not None:
                        extractions.append(extraction)
                    else:
                        # PTCV-258: Gap recovery cascade
                        from .gap_recovery import recover_gap
                        recovery = recover_gap(
                            q, routes, _TYPE_DISPATCH,
                            registry_metadata=_registry_meta,
                        )
                        if recovery.recovered:
                            extractions.append(QueryExtraction(
                                query_id=q.query_id,
                                section_id=q.section_id,
                                content=recovery.content,
                                confidence=recovery.confidence,
                                extraction_method=recovery.method,
                                source_section=recovery.source_section,
                                verbatim_content="",
                            ))
                        else:
                            reason = "no_match"
                            if recovery.exhaustive_search:
                                reason = (
                                    "exhaustive_no_match:"
                                    + ",".join(
                                        recovery.strategies_attempted
                                    )
                                )
                            gaps.append(ExtractionGap(
                                query_id=q.query_id,
                                section_id=q.section_id,
                                reason=reason,
                            ))
                    done_count += 1
                    if progress_callback is not None:
                        progress_callback(done_count, total_queries)
        else:
            total_queries = len(queries)
            for i, query in enumerate(queries, 1):
                extraction = self._process_query(
                    query, routes, protocol_index, match_result,
                    embedding_routes,
                )
                if extraction is not None:
                    extractions.append(extraction)
                else:
                    # PTCV-258: Gap recovery cascade
                    from .gap_recovery import recover_gap
                    recovery = recover_gap(
                        query, routes, _TYPE_DISPATCH,
                        registry_metadata=_registry_meta,
                    )
                    if recovery.recovered:
                        extractions.append(QueryExtraction(
                            query_id=query.query_id,
                            section_id=query.section_id,
                            content=recovery.content,
                            confidence=recovery.confidence,
                            extraction_method=recovery.method,
                            source_section=recovery.source_section,
                            verbatim_content="",
                        ))
                    else:
                        reason = "no_match"
                        if recovery.exhaustive_search:
                            reason = (
                                "exhaustive_no_match:"
                                + ",".join(
                                    recovery.strategies_attempted
                                )
                            )
                        gaps.append(ExtractionGap(
                            query_id=query.query_id,
                            section_id=query.section_id,
                            reason=reason,
                        ))
                if progress_callback is not None:
                    progress_callback(i, total_queries)

        # PTCV-259: Query-level registry injection — replace
        # low-confidence extractions with direct CT.gov answers.
        extractions = self._inject_registry_query_answers(
            extractions, protocol_index,
        )

        answered = sum(
            1
            for e in extractions
            if e.confidence >= self._threshold
        )
        total = len(queries)
        coverage = answered / total if total else 0.0

        return ExtractionResult(
            extractions=extractions,
            gaps=gaps,
            coverage=round(coverage, 4),
            total_queries=total,
            answered_queries=answered,
        )

    # ------------------------------------------------------------------
    # Route building
    # ------------------------------------------------------------------

    @staticmethod
    def _split_content_by_subsection(
        parent_content: str,
        parent_code: str,
        registry: dict[str, SubSectionDef],
    ) -> dict[str, str]:
        """Split parent section content into sub-section buckets.

        First attempts heading-based boundary detection using
        numbered section headings (PTCV-144).  Falls back to
        keyword-overlap scoring when fewer than 2 headings match.

        Args:
            parent_content: Concatenated text for a parent section.
            parent_code: Parent section code (e.g. ``"B.2"``).
            registry: Sub-section registry from
                :func:`build_subsection_registry`.

        Returns:
            Dict mapping sub-section code to paragraph content.
            Includes *parent_code* key for unassigned paragraphs.
        """
        sub_defs = get_subsections_for_parent(
            parent_code, registry
        )
        if not sub_defs:
            return {parent_code: parent_content}

        paragraphs = re.split(r"\n\s*\n", parent_content.strip())
        if not paragraphs:
            return {parent_code: parent_content}

        # Initialize buckets for all sub-sections + parent.
        buckets: dict[str, list[str]] = {
            sd.code: [] for sd in sub_defs
        }
        buckets[parent_code] = []

        # PTCV-144: Try heading-based splitting first.
        # Numbered headings like "11.1 Inclusion Criteria" are
        # far more reliable than keyword scoring.
        # PTCV-171: Also detect markdown headings (## 5.1 Title)
        # which are even more reliable structural signals.
        headings = list(
            _NUMBERED_HEADING_RE.finditer(parent_content)
        )
        # PTCV-171: Markdown headings from pymupdf4llm output
        md_headings = list(
            _MD_SECTION_HEADING_RE.finditer(parent_content)
        )
        # Merge both heading sources, deduplicating by position
        seen_positions: set[int] = set()
        heading_assignments: list[tuple[int, str]] = []
        for m in md_headings:
            title = m.group(2).strip()
            matched_code = _match_heading_to_subsection(
                title, sub_defs
            )
            if matched_code:
                heading_assignments.append(
                    (m.start(), matched_code)
                )
                seen_positions.add(m.start())
        for m in headings:
            if m.start() in seen_positions:
                continue
            title = m.group(2).strip()
            matched_code = _match_heading_to_subsection(
                title, sub_defs
            )
            if matched_code:
                heading_assignments.append(
                    (m.start(), matched_code)
                )

        if len(heading_assignments) >= 2:
            # Heading-based splitting is authoritative.
            heading_assignments.sort(key=lambda t: t[0])
            for i, (offset, code) in enumerate(
                heading_assignments
            ):
                end = (
                    heading_assignments[i + 1][0]
                    if i + 1 < len(heading_assignments)
                    else len(parent_content)
                )
                region = parent_content[offset:end].strip()
                if region:
                    buckets[code].append(region)
            # Preamble before first heading → parent bucket.
            first_offset = heading_assignments[0][0]
            if first_offset > 0:
                preamble = parent_content[
                    :first_offset
                ].strip()
                if preamble:
                    buckets[parent_code].append(preamble)
            return {
                code: "\n\n".join(paras)
                for code, paras in buckets.items()
                if paras
            }

        # Fallback: keyword-overlap scoring (original logic).
        _stop = frozenset({
            "the", "and", "for", "are", "what", "that",
            "this", "with", "from", "have", "has", "will",
            "been", "should", "there", "their", "which",
            "each", "any", "all", "how", "does", "used",
            "into", "may", "can", "also", "name",
        })
        sub_kws: dict[str, set[str]] = {}
        for sd in sub_defs:
            words = set(
                w
                for w in re.findall(
                    r"[a-z]{3,}", sd.description.lower()
                )
                if w not in _stop
            )
            sub_kws[sd.code] = words

        for para in paragraphs:
            if len(para.strip()) < 30:
                buckets[parent_code].append(para)
                continue

            para_lower = para.lower()
            best_code = parent_code
            best_score = 0.0

            for sd in sub_defs:
                kws = sub_kws[sd.code]
                if not kws:
                    continue
                hits = sum(1 for w in kws if w in para_lower)
                score = hits / len(kws)
                if score > best_score:
                    best_score = score
                    best_code = sd.code

            if best_score < 0.08:
                best_code = parent_code

            buckets[best_code].append(para)

        return {
            code: "\n\n".join(paras)
            for code, paras in buckets.items()
            if paras
        }

    @staticmethod
    def _build_routes(
        protocol_index: ProtocolIndex,
        match_result: MatchResult,
        subsection_registry: (
            dict[str, SubSectionDef] | None
        ) = None,
    ) -> dict[str, tuple[str, str, MatchConfidence]]:
        """Map ICH section codes to concatenated content.

        When *subsection_registry* is provided, also creates
        sub-section-level routes (B.x.y) by splitting parent
        content using keyword scoring (PTCV-110).

        Returns:
            Dict mapping ICH code to
            ``(content, source_sections, confidence)``.
            Multiple protocol sections mapping to the same ICH
            code are concatenated with section markers.
        """
        routes: dict[
            str, tuple[str, str, MatchConfidence]
        ] = {}

        for mapping in match_result.mappings:
            if not mapping.matches:
                continue

            top_match = mapping.matches[0]
            ich_code = top_match.ich_section_code
            confidence = top_match.confidence

            # Get content from protocol index
            content = protocol_index.content_spans.get(
                mapping.protocol_section_number, ""
            )
            if not content:
                continue

            if ich_code in routes:
                existing_content, existing_src, existing_conf = (
                    routes[ich_code]
                )

                # PTCV-182: Cap route concatenation to prevent
                # pathological merges (defense in depth).
                section_count = existing_src.count(",") + 1
                if section_count >= _MAX_SECTIONS_PER_ROUTE:
                    continue
                if (
                    len(existing_content) + len(content)
                    > _MAX_ROUTE_CHARS
                ):
                    continue

                # Concatenate with section marker
                combined_content = (
                    f"{existing_content}\n\n"
                    f"--- [{mapping.protocol_section_number}] "
                    f"{mapping.protocol_section_title} ---\n"
                    f"{content}"
                )
                combined_src = (
                    f"{existing_src}, "
                    f"{mapping.protocol_section_number}"
                )
                # PTCV-182: Use ordinal ranking for worst
                # confidence (fixes alphabetical comparison bug).
                worst_conf = (
                    existing_conf
                    if _CONFIDENCE_RANK.get(
                        existing_conf, 0
                    )
                    < _CONFIDENCE_RANK.get(confidence, 0)
                    else confidence
                )
                routes[ich_code] = (
                    combined_content,
                    combined_src,
                    worst_conf,
                )
            else:
                routes[ich_code] = (
                    content,
                    mapping.protocol_section_number,
                    confidence,
                )

        # PTCV-110: Split parent content into sub-section routes.
        if subsection_registry:
            sub_routes: dict[
                str, tuple[str, str, MatchConfidence]
            ] = {}
            for ich_code, (
                content,
                source,
                confidence,
            ) in routes.items():
                splits = (
                    QueryExtractor._split_content_by_subsection(
                        content, ich_code, subsection_registry
                    )
                )
                for sub_code, sub_content in splits.items():
                    if (
                        sub_code != ich_code
                        and sub_content.strip()
                    ):
                        sub_routes[sub_code] = (
                            sub_content,
                            source,
                            confidence,
                        )
            routes.update(sub_routes)

        return routes

    # ------------------------------------------------------------------
    # Embedding-based semantic routing — Layer 2 (PTCV-256)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_embedding_routes(
        protocol_index: ProtocolIndex,
    ) -> dict[str, list[tuple[str, float, str]]]:
        """Build Layer 2 routes via centroid similarity scoring.

        Uses the CentroidClassifier to score ALL content_spans
        against ALL section centroids. Returns the top-k most
        similar spans per ICH code, regardless of how Stage 2
        classified them. This recovers from Stage 2 misclassification.

        Returns:
            Dict mapping ICH code to list of
            ``(content_text, similarity_score, section_number)``
            tuples sorted by similarity descending.
            Empty dict if centroid classifier is unavailable.

        PTCV-256: Embedding-based semantic routing.
        """
        if not protocol_index.content_spans:
            return {}

        try:
            from .centroid_classifier import load_centroid_classifier
        except ImportError:
            logger.debug(
                "PTCV-256: centroid_classifier not available; "
                "skipping embedding routes"
            )
            return {}

        classifier = load_centroid_classifier()
        if classifier is None:
            return {}

        scored = classifier.score_spans(
            protocol_index.content_spans, top_k=3,
        )

        embedding_routes: dict[
            str, list[tuple[str, float, str]]
        ] = {}
        for ich_code, span_scores in scored.items():
            entries: list[tuple[str, float, str]] = []
            for section_num, similarity in span_scores:
                content = protocol_index.content_spans.get(
                    section_num, ""
                )
                if content.strip():
                    entries.append(
                        (content, similarity, section_num)
                    )
            if entries:
                embedding_routes[ich_code] = entries

        if embedding_routes:
            logger.info(
                "PTCV-256: Built embedding routes for %d "
                "section codes",
                len(embedding_routes),
            )

        return embedding_routes

    # ------------------------------------------------------------------
    # Front-matter extraction (PTCV-155, Tier 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_front_matter_route(
        routes: dict[str, tuple[str, str, MatchConfidence]],
        protocol_index: ProtocolIndex,
    ) -> None:
        """Add front-matter as B.1 route when no quality match exists.

        In 35/40 protocols, front-matter (title, sponsor, PI) appears
        on unnumbered cover pages before section 1 and is absent from
        the ToC.  This captures that pre-section content as a B.1
        route fallback so B.1 queries can find protocol titles and
        sponsor information.

        Skips injection if B.1 already has HIGH or REVIEW content.
        """
        # Skip if B.1 already has genuine good content.
        # PTCV-182: Detect contaminated B.1 routes — if the
        # source string lists ≥3 sections it was built from
        # concatenated unrelated content, so proceed with
        # front-matter injection anyway.
        if "B.1" in routes:
            _, src, conf = routes["B.1"]
            section_count = src.count(",") + 1
            if conf in (
                MatchConfidence.HIGH,
                MatchConfidence.REVIEW,
            ) and section_count < 3:
                return

        # Find the earliest section header's char offset
        resolved = [
            h
            for h in protocol_index.section_headers
            if h.char_offset >= 0
        ]
        if not resolved:
            return

        first_offset = min(h.char_offset for h in resolved)
        if first_offset <= 100:
            # Too little front matter to be useful
            return

        # Extract text before the first numbered section
        front_matter = protocol_index.full_text[
            :first_offset
        ].strip()
        if len(front_matter) < 50:
            return

        # Cap at 5000 chars to avoid dumping entire ToC remnants
        front_matter = front_matter[:5000]

        logger.info(
            "PTCV-155: Injecting %d chars of front-matter "
            "as B.1 route",
            len(front_matter),
        )

        if "B.1" in routes:
            # Prepend front matter to existing LOW-confidence B.1
            existing_content, existing_src, _ = routes["B.1"]
            routes["B.1"] = (
                f"{front_matter}\n\n"
                f"--- [{existing_src}] Matched Sections ---\n"
                f"{existing_content}",
                f"front_matter, {existing_src}",
                MatchConfidence.REVIEW,
            )
        else:
            routes["B.1"] = (
                front_matter,
                "front_matter",
                MatchConfidence.REVIEW,
            )

    # ------------------------------------------------------------------
    # Registry metadata injection (PTCV-230)
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_registry_routes(
        routes: dict[str, tuple[str, str, MatchConfidence]],
        protocol_index: ProtocolIndex,
    ) -> None:
        """Inject registry metadata as route content for missing/low sections.

        Fetches ClinicalTrials.gov registry metadata for the protocol's
        NCT ID, maps it to ICH E6(R3) sections via MetadataToIchMapper,
        and injects the content into routes where:
        - No PDF route exists (creates new route from registry)
        - PDF route has LOW confidence (appends registry as supplement)

        Skips injection for routes with HIGH or REVIEW confidence to
        avoid polluting high-quality PDF-extracted content.

        Follows the ``_inject_front_matter_route()`` pattern.
        """
        # Resolve NCT ID from filename or front-matter text
        nct_id = QueryExtractor._resolve_nct_id(protocol_index)
        if not nct_id:
            return

        # Fetch and map registry metadata
        try:
            from ptcv.registry.metadata_fetcher import (
                RegistryMetadataFetcher,
            )
            from ptcv.registry.ich_mapper import MetadataToIchMapper
        except ImportError:
            logger.debug(
                "PTCV-230: Registry modules not available; "
                "skipping registry injection"
            )
            return

        fetcher = RegistryMetadataFetcher()
        metadata = fetcher.fetch(nct_id)
        if not metadata:
            logger.debug(
                "PTCV-230: No registry metadata for %s", nct_id
            )
            return

        mapper = MetadataToIchMapper()
        mapped_sections = mapper.map(metadata)
        if not mapped_sections:
            return

        injected = 0
        for section in mapped_sections:
            code = section.section_code
            content = (
                f"[REGISTRY — ClinicalTrials.gov {nct_id}]\n"
                f"{section.content_text}"
            )

            if code not in routes:
                # No PDF route — create from registry
                routes[code] = (
                    content,
                    f"registry:{nct_id}",
                    MatchConfidence.REVIEW,
                )
                injected += 1

            else:
                _, src, conf = routes[code]
                if conf == MatchConfidence.LOW:
                    # Low-confidence PDF route — append registry
                    existing_content = routes[code][0]
                    routes[code] = (
                        f"{existing_content}\n\n"
                        f"--- [Registry Supplement] ---\n"
                        f"{content}",
                        f"{src}, registry:{nct_id}",
                        MatchConfidence.REVIEW,
                    )
                    injected += 1
                # HIGH/REVIEW — skip, PDF content is sufficient

        if injected:
            logger.info(
                "PTCV-230: Injected %d registry routes for %s",
                injected,
                nct_id,
            )

    @staticmethod
    def _resolve_nct_id(
        protocol_index: ProtocolIndex,
    ) -> str | None:
        """Extract NCT ID from protocol source path or front-matter.

        Checks filename first (most reliable), then scans the first
        2000 chars of full_text for an NCT pattern.

        Returns:
            NCT ID string or None if not found.
        """
        # Check filename
        if protocol_index.source_path:
            m = _NCT_RE.search(protocol_index.source_path)
            if m:
                return m.group(0)

        # Check front-matter (first 2000 chars)
        if protocol_index.full_text:
            m = _NCT_RE.search(protocol_index.full_text[:2000])
            if m:
                return m.group(0)

        return None

    # ------------------------------------------------------------------
    # Query-level registry injection (PTCV-259)
    # ------------------------------------------------------------------

    def _inject_registry_query_answers(
        self,
        extractions: list[QueryExtraction],
        protocol_index: ProtocolIndex,
    ) -> list[QueryExtraction]:
        """Replace low-confidence extractions with direct registry answers.

        Fetches registry metadata for the protocol's NCT ID, then uses
        ``inject_registry_answers()`` to replace low-confidence results
        with exact CT.gov field values. Unlike route-level injection
        (PTCV-230), this operates at query granularity — each query
        gets exactly the registry field that answers it.

        High-confidence PDF extractions are never overridden.
        """
        nct_id = self._resolve_nct_id(protocol_index)
        if not nct_id:
            return extractions

        try:
            from ptcv.registry.metadata_fetcher import (
                RegistryMetadataFetcher,
            )
            from ptcv.registry.query_injector import (
                inject_registry_answers,
            )
        except ImportError:
            logger.debug(
                "PTCV-259: Registry modules not available; "
                "skipping query-level injection"
            )
            return extractions

        fetcher = RegistryMetadataFetcher()
        metadata = fetcher.fetch(nct_id)
        if not metadata:
            return extractions

        return inject_registry_answers(
            extractions,
            metadata,
            confidence_threshold=self._threshold,
        )

    # ------------------------------------------------------------------
    # Citation consolidation (PTCV-99)
    # ------------------------------------------------------------------

    @staticmethod
    def _consolidate_citations(
        routes: dict[str, tuple[str, str, MatchConfidence]],
    ) -> None:
        """Collect citations from routes, inject into B.2.7, filter others.

        1. Scans all routes for full bibliographic references.
        2. Consolidates unique citations into B.2.7 (deduped, ordered).
        3. Strips full citations from non-B.2 routes.
        4. Preserves parenthetical references like "(Author, Year)".

        Modifies *routes* in place.
        """
        all_citations: list[str] = []
        seen: set[str] = set()

        # Collect citations from all routes
        for code, (content, _src, _conf) in list(routes.items()):
            found = extract_citations(content)
            for c in found:
                normalised = c.strip().lower()
                if normalised not in seen:
                    seen.add(normalised)
                    all_citations.append(c)

        if not all_citations:
            return

        logger.info(
            "PTCV-99: Collected %d unique citations for B.2.7",
            len(all_citations),
        )

        # Build consolidated bibliography for B.2.7
        bibliography = "\n".join(all_citations)

        if "B.2.7" in routes:
            existing, src, conf = routes["B.2.7"]
            routes["B.2.7"] = (
                f"{existing}\n\n"
                f"--- Consolidated Bibliography ---\n"
                f"{bibliography}",
                f"{src}, citation_extraction",
                conf,
            )
        elif "B.2" in routes:
            _, src, conf = routes["B.2"]
            routes["B.2.7"] = (
                bibliography,
                "citation_extraction",
                conf,
            )
        else:
            routes["B.2.7"] = (
                bibliography,
                "citation_extraction",
                MatchConfidence.REVIEW,
            )

        # Strip citations from non-B.2 routes
        for code in list(routes.keys()):
            if code.startswith("B.2"):
                continue
            content, src, conf = routes[code]
            filtered = filter_citations(content)
            if filtered != content:
                routes[code] = (filtered, src, conf)

    # ------------------------------------------------------------------
    # Per-query processing
    # ------------------------------------------------------------------

    def _process_query(
        self,
        query: AppendixBQuery,
        routes: dict[str, tuple[str, str, MatchConfidence]],
        protocol_index: ProtocolIndex,
        match_result: MatchResult,
        embedding_routes: (
            dict[str, list[tuple[str, float, str]]] | None
        ) = None,
    ) -> QueryExtraction | None:
        """Process a single query against routed content.

        Resolution order:
        1. Layer 1 sub-section route (B.x.y)
        2. Layer 1 parent route (B.x)
        3. Layer 2 embedding route (PTCV-256) — semantic similarity
        4. Unscoped full-text search (fallback)
        """
        # PTCV-110: Try sub-section route first (B.x.y).
        section_id = query.section_id
        if section_id in routes:
            content, source, route_confidence = routes[
                section_id
            ]
            return self._extract_from_content(
                query,
                content,
                source,
                route_confidence,
                scoped=True,
            )

        # Fall back to parent route (B.x).
        schema_code = query.schema_section
        if schema_code in routes:
            content, source, route_confidence = routes[
                schema_code
            ]
            return self._extract_from_content(
                query,
                content,
                source,
                route_confidence,
                scoped=True,
            )

        # PTCV-256: Layer 2 — embedding-based semantic routing.
        # Try centroid similarity to recover from Stage 2
        # misclassification before falling back to unscoped search.
        if embedding_routes:
            l2_result = self._try_embedding_route(
                query, embedding_routes,
            )
            if l2_result is not None:
                return l2_result

        # Fallback: unscoped full-text search (PTCV-157: capped)
        if protocol_index.full_text:
            # Pre-filter to the most query-relevant paragraphs
            # instead of dumping the entire document (up to 500K chars).
            capped_text, _ = _select_relevant_paragraphs(
                protocol_index.full_text,
                query,
                max_chars=_UNSCOPED_MAX_CHARS,
            )
            if not capped_text:
                return None
            return self._extract_from_content(
                query,
                capped_text,
                "full_text",
                MatchConfidence.LOW,
                scoped=False,
            )

        return None

    # ------------------------------------------------------------------
    # Embedding route resolution (PTCV-256)
    # ------------------------------------------------------------------

    _EMBEDDING_MIN_SIMILARITY = 0.30

    def _try_embedding_route(
        self,
        query: AppendixBQuery,
        embedding_routes: dict[
            str, list[tuple[str, float, str]]
        ],
    ) -> QueryExtraction | None:
        """Try to resolve a query via Layer 2 embedding routes.

        Checks both subsection (B.x.y) and parent (B.x) codes.
        Uses the highest-similarity span above the minimum threshold.
        Route confidence is REVIEW — better than unscoped, but not
        as reliable as Layer 1 keyword routing.

        Returns:
            QueryExtraction or None if no suitable embedding match.
        """
        for code in (query.section_id, query.schema_section):
            spans = embedding_routes.get(code)
            if not spans:
                continue
            # spans is sorted by similarity descending
            best_content, best_sim, best_section = spans[0]
            if best_sim < self._EMBEDDING_MIN_SIMILARITY:
                continue

            logger.debug(
                "PTCV-256: Layer 2 route for %s via %s "
                "(sim=%.3f)",
                query.query_id, best_section, best_sim,
            )

            return self._extract_from_content(
                query,
                best_content,
                f"embedding:{best_section}",
                MatchConfidence.REVIEW,
                scoped=True,
            )

        return None

    def _extract_from_content(
        self,
        query: AppendixBQuery,
        text: str,
        source_section: str,
        route_confidence: MatchConfidence,
        scoped: bool,
    ) -> QueryExtraction | None:
        """Dispatch to type-specific extractor and apply penalties."""
        # PTCV-257: Auto-detect content format and select best strategy.
        # The detected format overrides the YAML hint when confident.
        from .content_format_detector import (
            detect_content_format,
            select_strategy,
        )

        detected = detect_content_format(text)
        strategy = select_strategy(query.expected_type, detected)
        extractor_fn = _TYPE_DISPATCH.get(
            strategy, _extract_text_long
        )
        extracted, confidence, method = extractor_fn(
            text, query
        )

        # If detected strategy produced empty result and differs from
        # YAML hint, retry with the original hint as fallback.
        if (
            not extracted
            and strategy != query.expected_type
        ):
            fallback_fn = _TYPE_DISPATCH.get(
                query.expected_type, _extract_text_long
            )
            extracted, confidence, method = fallback_fn(
                text, query
            )
            if extracted:
                method = f"{method}(fallback)"

        if not extracted:
            return None

        # LLM transformation (PTCV-98, PTCV-111): reframe scoped
        # matches to directly address the query intent.
        # Runs BEFORE confidence penalties so that the LLM's
        # self-reported confidence participates in the final score.
        verbatim = ""  # pre-transformation text; set only on success
        if self._use_llm and scoped:
            logger.debug(
                "LLM transform: attempting query=%s "
                "source_len=%d",
                query.query_id,
                len(extracted),
            )
            transformed = self._transform_content(
                extracted, query
            )
            if transformed is not None:
                verbatim = extracted  # preserve pre-transformation
                extracted, llm_conf = transformed
                # Blend: 60% deterministic + 40% LLM self-report
                confidence = round(
                    0.6 * confidence + 0.4 * llm_conf, 4
                )
                method = "llm_transform"
                logger.debug(
                    "LLM transform: success query=%s "
                    "llm_conf=%.4f blended=%.4f",
                    query.query_id,
                    llm_conf,
                    confidence,
                )
            else:
                logger.debug(
                    "LLM transform: failed query=%s "
                    "— falling back to verbatim",
                    query.query_id,
                )
        elif self._use_llm and not scoped:
            logger.debug(
                "LLM transform: skipped query=%s "
                "reason=unscoped",
                query.query_id,
            )
        elif not self._use_llm:
            logger.debug(
                "LLM transform: skipped query=%s "
                "reason=llm_disabled",
                query.query_id,
            )

        # Apply confidence penalties AFTER transformation
        # (PTCV-111: penalties are post-transformation).
        if not scoped:
            confidence *= _UNSCOPED_PENALTY
            method = "unscoped_search"
        elif route_confidence == MatchConfidence.REVIEW:
            confidence *= _REVIEW_PENALTY

        confidence = round(min(1.0, confidence), 4)

        return QueryExtraction(
            query_id=query.query_id,
            section_id=query.section_id,
            content=extracted,
            confidence=confidence,
            extraction_method=method,
            source_section=source_section,
            verbatim_content=verbatim,
        )
