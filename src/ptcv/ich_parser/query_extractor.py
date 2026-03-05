"""Query-driven extraction engine (PTCV-91).

Runs Appendix B queries against protocol sections to extract
structured answers.  Given ``ProtocolIndex`` (content),
``MatchResult`` (routing), and ``AppendixBQuery`` (questions),
dispatches each query to a type-specific extractor.

V1 is deterministic — no LLM calls.  Type-specific strategies:

    text_long   → passthrough + keyword-overlap confidence
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

Risk tier: LOW — read-only extraction, no external I/O.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Sequence

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
            ``"table_detection"``, ``"unscoped_search"``).
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


def _extract_text_long(
    text: str, query: AppendixBQuery
) -> tuple[str, float, str]:
    """Passthrough with keyword-overlap confidence scoring."""
    if not text.strip():
        return "", 0.0, "passthrough"

    # Keyword overlap: words from query_text found in content
    query_words = set(
        w.lower()
        for w in re.findall(r"[a-z]{3,}", query.query_text.lower())
    )
    # Remove common stop words
    stop = {
        "the", "and", "for", "are", "what", "that", "this",
        "with", "from", "have", "has", "will", "been", "should",
        "there", "their", "which", "each",
    }
    query_words -= stop

    if not query_words:
        return text.strip(), 0.60, "passthrough"

    text_lower = text.lower()
    hits = sum(1 for w in query_words if w in text_lower)
    overlap = hits / len(query_words)

    # Map overlap to confidence: [0.0, 1.0] → [0.45, 0.85]
    confidence = 0.45 + (overlap * 0.40)
    return text.strip(), round(confidence, 4), "passthrough"


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
    """

    def __init__(
        self, confidence_threshold: float = 0.70
    ) -> None:
        self._threshold = confidence_threshold

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

        # Build route map: ich_section_code → concatenated content
        routes = self._build_routes(
            protocol_index, match_result
        )

        extractions: list[QueryExtraction] = []
        gaps: list[ExtractionGap] = []

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
    def _build_routes(
        protocol_index: ProtocolIndex,
        match_result: MatchResult,
    ) -> dict[str, tuple[str, str, MatchConfidence]]:
        """Map ICH section codes to concatenated content.

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
        schema_code = query.schema_section

        # Try scoped route first
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

        # Apply confidence penalties
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
