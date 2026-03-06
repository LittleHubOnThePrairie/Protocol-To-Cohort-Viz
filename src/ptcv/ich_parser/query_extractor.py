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

Routing uses ``get_queries_by_schema_section()`` because B.15/B.16
queries map to ``schema_section: "B.14"``.

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
        content: Extracted answer text.
        confidence: Confidence score (0.0–1.0).
        extraction_method: Strategy used (``"regex"``,
            ``"heuristic"``, ``"passthrough"``,
            ``"relevance_excerpt"``, ``"table_detection"``,
            ``"unscoped_search"``, ``"llm_transform"``).
        source_section: Protocol section number(s) searched.
    """

    query_id: str
    section_id: str
    content: str
    confidence: float
    extraction_method: str
    source_section: str


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


# Maximum characters for text_long excerpt output (PTCV-109).
_TEXT_LONG_MAX_CHARS = 2000

# Minimum paragraph length to consider for relevance scoring.
_MIN_PARAGRAPH_LEN = 40

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
        self._llm_model = "claude-sonnet-4-6"
        self._client: Any = None
        self._use_llm = False
        self._transform_calls = 0
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
        # (PTCV-109: avoid blind prefix truncation).
        if len(source_content) > 6000:
            source_for_llm, _ = _select_relevant_paragraphs(
                source_content, query, max_chars=6000
            )
        else:
            source_for_llm = source_content

        prompt = self._TRANSFORM_PROMPT.format(
            query_text=query.query_text,
            source=source_for_llm,
        )
        try:
            resp = self._client.messages.create(
                model=self._llm_model,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            with self._transform_lock:
                self._transform_calls += 1

            first_block = resp.content[0]
            if not hasattr(first_block, "text"):
                return None
            raw = first_block.text.strip()

            # Parse JSON — strip markdown fences if present
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
                        "JSON for %s",
                        query.query_id,
                    )
                    return None

            transformed = parsed.get("transformed", "")
            llm_conf = float(
                parsed.get("confidence", 0.0)
            )
            if not transformed:
                return None
            return transformed, min(1.0, max(0.0, llm_conf))

        except Exception:
            logger.warning(
                "LLM transform failed for %s — "
                "falling back to verbatim",
                query.query_id,
                exc_info=True,
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
    ) -> ExtractionResult:
        """Run all queries against the protocol.

        Args:
            protocol_index: Navigable document index.
            match_result: Section mapping from
                :class:`SectionMatcher`.
            queries: Pre-loaded queries; loads from YAML if
                *None*.

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

        extractions: list[QueryExtraction] = []
        gaps: list[ExtractionGap] = []

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
                    ): q
                    for q in queries
                }
                for future in as_completed(future_to_query):
                    q = future_to_query[future]
                    extraction = future.result()
                    if extraction is not None:
                        extractions.append(extraction)
                    else:
                        gaps.append(
                            ExtractionGap(
                                query_id=q.query_id,
                                section_id=q.section_id,
                                reason="no_match",
                            )
                        )
        else:
            for query in queries:
                extraction = self._process_query(
                    query, routes, protocol_index, match_result
                )
                if extraction is not None:
                    extractions.append(extraction)
                else:
                    gaps.append(
                        ExtractionGap(
                            query_id=query.query_id,
                            section_id=query.section_id,
                            reason="no_match",
                        )
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

        Scores each paragraph against sub-section descriptions
        using keyword overlap, assigning each to its best match.

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

        # Build keyword sets from sub-section descriptions.
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

        # Score and assign each paragraph.
        buckets: dict[str, list[str]] = {
            sd.code: [] for sd in sub_defs
        }
        buckets[parent_code] = []

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
                # Concatenate with section marker
                existing_content, existing_src, existing_conf = (
                    routes[ich_code]
                )
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
                # Use worst confidence of merged sections
                worst_conf = (
                    existing_conf
                    if existing_conf.value < confidence.value
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
    # Per-query processing
    # ------------------------------------------------------------------

    def _process_query(
        self,
        query: AppendixBQuery,
        routes: dict[str, tuple[str, str, MatchConfidence]],
        protocol_index: ProtocolIndex,
        match_result: MatchResult,
    ) -> QueryExtraction | None:
        """Process a single query against routed content."""
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

        # Fallback: unscoped full-text search
        if protocol_index.full_text:
            return self._extract_from_content(
                query,
                protocol_index.full_text,
                "full_text",
                MatchConfidence.LOW,
                scoped=False,
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
        extractor_fn = _TYPE_DISPATCH.get(
            query.expected_type, _extract_text_long
        )
        extracted, confidence, method = extractor_fn(
            text, query
        )

        if not extracted:
            return None

        # LLM transformation (PTCV-98, PTCV-111): reframe scoped
        # matches to directly address the query intent.
        # Runs BEFORE confidence penalties so that the LLM's
        # self-reported confidence participates in the final score.
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
        )
