"""Semantic section matcher — protocol headers to Appendix B (PTCV-90).

Maps protocol section headers (from ``ProtocolIndex``) to ICH E6(R3)
Appendix B section codes using deterministic keyword/pattern scoring
from ``IchSectionDef``.  No external API keys required, CI-safe.

A synonym boost table provides additive score increases for
well-known protocol header synonyms (e.g. "Inclusion Criteria"
→ B.5), improving precision without brittle hard-coded overrides.

PTCV-171: Markdown heading hierarchy from pymupdf4llm output is used
as a strong classification signal.  When content spans contain
``## Title`` headings, these are weighted higher than body text for
keyword scoring, improving confidence and reducing B.1 noise.

PTCV-160: When PTCV-171's heading matching produces ambiguous REVIEW-tier
results (top-2 candidates within a small score delta), TOC entry ordering
is used as a tiebreaker.  The candidate whose ICH section code fits the
sequential order of neighboring headers is boosted.  Protocols without
a TOC are handled gracefully (no-op).

.. note::

   Cohere embedding support was removed in PTCV-102.  The keyword
   matcher achieves equivalent results for the structured ICH E6(R3)
   domain and runs deterministically without API rate limits.

Risk tier: LOW — read-only semantic analysis.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import re
from typing import Any, Optional

from ptcv.ich_parser.schema_loader import (
    IchSectionDef,
    load_ich_schema,
)
from ptcv.ich_parser.toc_extractor import (
    ProtocolIndex,
    SectionHeader,
    TOCEntry,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Synonym boost table
# -----------------------------------------------------------------------
# Maps normalised header substrings to ICH section codes.  When a
# protocol header title (lowercased) contains one of these substrings,
# the corresponding ICH section gets +synonym_boost to its score.
#
# Section names below match ICH E6(R3) Appendix B (PTCV-113).

_SYNONYM_BOOSTS: dict[str, str] = {
    # B.1 General Information
    "general information": "B.1",
    "protocol summary": "B.1",
    "administrative information": "B.1",
    "protocol title": "B.1",
    "protocol number": "B.1",
    "synopsis": "B.1",
    "protocol synopsis": "B.1",
    "study information": "B.1",
    "sponsor information": "B.1",
    "sponsor responsibilities": "B.1",
    "sponsor obligations": "B.1",
    "study sponsor": "B.1",
    "protocol identification": "B.1",
    "study identification": "B.1",
    "cover page": "B.1",
    "title page": "B.1",
    "administrative requirements": "B.1",
    # B.2 Background Information
    "background": "B.2",
    "rationale": "B.2",
    "introduction": "B.2",
    "literature review": "B.2",
    "preclinical": "B.2",
    "nonclinical": "B.2",
    "references": "B.2",
    "bibliography": "B.2",
    "citations": "B.2",
    "literature": "B.2",
    # B.3 Objectives
    "objective": "B.3",
    "objectives": "B.3",
    "purpose": "B.3",
    "endpoints": "B.3",
    "hypothesis": "B.3",
    "primary endpoint": "B.3",
    "primary endpoints": "B.3",
    "secondary endpoint": "B.3",
    "secondary endpoints": "B.3",
    "primary objective": "B.3",
    "primary objectives": "B.3",
    "secondary objective": "B.3",
    "secondary objectives": "B.3",
    "objectives and endpoints": "B.3",
    "aims": "B.3",
    "study objectives": "B.3",
    "trial objectives": "B.3",
    "study purpose": "B.3",
    "trial purpose": "B.3",
    "estimand": "B.3",
    "estimands": "B.3",
    # B.4 Trial Design
    "study design": "B.4",
    "trial design": "B.4",
    "schedule of activities": "B.4",
    "schedule of assessments": "B.4",
    "study calendar": "B.4",
    "randomisation": "B.4",
    "randomization": "B.4",
    "blinding": "B.4",
    # B.5 Selection of Subjects
    "inclusion criteria": "B.5",
    "exclusion criteria": "B.5",
    "eligibility": "B.5",
    "selection of subjects": "B.5",
    "selection of patients": "B.5",
    "selection of participants": "B.5",
    "study population": "B.5",
    "entry criteria": "B.5",
    # B.6 Discontinuation and Participant Withdrawal
    "discontinuation": "B.6",
    "withdrawal": "B.6",
    "stopping rules": "B.6",
    "treatment termination": "B.6",
    "early termination": "B.6",
    "participant discontinuation": "B.6",
    "drop out": "B.6",
    # B.7 Treatment of Participants
    "investigational product": "B.7",
    "study treatment": "B.7",
    "study drug": "B.7",
    "dosage": "B.7",
    "dose modification": "B.7",
    "concomitant medication": "B.7",
    "treatment schedule": "B.7",
    # B.8 Assessment of Efficacy
    "efficacy assessment": "B.8",
    "clinical outcome": "B.8",
    "response criteria": "B.8",
    "tumour response": "B.8",
    "tumor response": "B.8",
    "assessment of response": "B.8",
    "clinical outcome assessment": "B.8",
    "outcome measures": "B.8",
    "study endpoints": "B.8",
    "efficacy analyses": "B.8",
    "efficacy endpoints": "B.8",
    "treatment response": "B.8",
    "endpoints and assessment": "B.8",
    "efficacy evaluation": "B.8",
    "efficacy measures": "B.8",
    # B.9 Assessment of Safety
    "adverse event": "B.9",
    "safety assessment": "B.9",
    "safety monitoring": "B.9",
    "vital signs": "B.9",
    "laboratory test": "B.9",
    "safety reporting": "B.9",
    # B.10 Statistics
    "statistical": "B.10",
    "sample size": "B.10",
    "analysis population": "B.10",
    "statistical analysis plan": "B.10",
    "power calculation": "B.10",
    # B.11 Direct Access to Source Data/Documents
    "source data verification": "B.11",
    "direct access": "B.11",
    "source document": "B.11",
    # B.12 Quality Control and Quality Assurance
    "quality control": "B.12",
    "quality assurance": "B.12",
    "monitoring plan": "B.12",
    "data management": "B.12",
    "risk-based monitoring": "B.12",
    "site monitoring": "B.12",
    # B.13 Ethics
    "ethics": "B.13",
    "informed consent": "B.13",
    "irb": "B.13",
    "ethics committee": "B.13",
    "ethical approval": "B.13",
    # B.14 Data Handling and Record Keeping
    "data handling": "B.14",
    "record keeping": "B.14",
    "case report form": "B.14",
    "crf": "B.14",
    "archiving": "B.14",
    "data retention": "B.14",
    "essential documents": "B.14",
    "data storage": "B.14",
    "data collection": "B.14",
    # B.15 Financing and Insurance
    "financing": "B.15",
    "insurance": "B.15",
    "compensation": "B.15",
    "indemnity": "B.15",
    # B.16 Publication Policy
    "publication": "B.16",
    "publication policy": "B.16",
}

# -----------------------------------------------------------------------
# Markdown heading detection (PTCV-171)
# -----------------------------------------------------------------------
# Detects ATX headings produced by pymupdf4llm + PTCV-176 normalizer.
# Heading text is a stronger classification signal than body text because
# headings are unambiguous structural markers in the document.

_MD_HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.+)$", re.MULTILINE,
)

# Additive score boost per heading level when heading text matches
# ICH section synonyms.  Level 1-2 headings (## / ###) are major
# section titles; deeper levels are sub-sections with weaker signal.
_HEADING_LEVEL_BOOST: dict[int, float] = {
    1: 0.12,
    2: 0.10,
    3: 0.07,
    4: 0.05,
    5: 0.03,
    6: 0.02,
}

# -----------------------------------------------------------------------
# B.1 exclusion terms (PTCV-155, Tier 2)
# -----------------------------------------------------------------------
# Headers containing these terms should never fall through to B.1.
# They either belong to other ICH sections (with dedicated synonym
# boosts) or are meta-content that doesn't map to any ICH section.
# Without this, the matcher assigns 30-89 unclassifiable headers to
# B.1 at LOW confidence, burying the actual front-matter signal.

_B1_EXCLUSION_TERMS: frozenset[str] = frozenset({
    # Meta-content (no ICH section)
    "definitions",
    "abbreviations",
    "glossary",
    "list of abbreviations",
    "list of tables",
    "list of figures",
    "references",
    "appendix",
    "appendices",
    # B.9 Safety
    "pregnancy",
    # B.10 Statistics
    "interim analysis",
    "statistical analyses",
    # B.12 Quality Control / GCP
    "good clinical practice",
})

# -----------------------------------------------------------------------
# Garbage header detection (PTCV-157, Tier 2)
# -----------------------------------------------------------------------
# Headers that are clearly not protocol section titles should be
# excluded from matching entirely.  Indicators:
# - Literature citations (author lists with "et al")
# - OCR-damaged text (broken word spacing)
# - Overly long "headers" (>100 chars — likely body text)

# Maximum header length before it's treated as body text.
_MAX_HEADER_LEN = 100

# Regex: author-list citation pattern (e.g. "Smith JA, Jones B, ...")
_CITATION_RE = re.compile(
    r"(?:[A-Z][a-z]+\s+[A-Z]{1,2},?\s*){2,}.*\bet\s*al\b",
    re.IGNORECASE,
)

# Common short English words that are NOT OCR artefacts.
_COMMON_SHORT_WORDS = frozenset({
    "a", "i", "an", "as", "at", "be", "by", "do", "go",
    "he", "if", "in", "is", "it", "me", "my", "no", "of",
    "on", "or", "so", "to", "up", "us", "we",
})


def _is_garbage_header(title: str) -> bool:
    """Return ``True`` if *title* looks like a garbage header.

    Garbage indicators (PTCV-157):
    - Length > ``_MAX_HEADER_LEN`` chars (likely body text).
    - Contains citation-style author list with "et al".
    - Contains OCR word-splitting damage: >= 2 artefact words
      (single-char orphans not in common words, or hyphen-prefixed
      fragments like "-resistance").
    """
    if len(title) > _MAX_HEADER_LEN:
        return True
    if _CITATION_RE.search(title):
        return True
    # OCR damage: count artefact words.
    words = title.split()
    artefact_count = 0
    for w in words:
        w_lower = w.lower()
        if len(w) <= 2 and w_lower not in _COMMON_SHORT_WORDS:
            artefact_count += 1
        elif w.startswith("-") and len(w) > 1:
            artefact_count += 1
    if artefact_count >= 2:
        return True
    return False


# -----------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------


class MatchConfidence(enum.Enum):
    """Confidence tier for a section match."""

    HIGH = "high"
    REVIEW = "review"
    LOW = "low"


@dataclasses.dataclass(frozen=True)
class SectionMatch:
    """A single candidate mapping from protocol header to ICH section.

    Attributes:
        ich_section_code: Appendix B section code (e.g. ``"B.5"``).
        ich_section_name: Human-readable ICH section name.
        similarity_score: Raw similarity score (0.0–1.0).
        boosted_score: Score after synonym boost applied.
        confidence: Confidence tier derived from *boosted_score*.
        match_method: ``"embedding"`` or ``"keyword_fallback"``.
    """

    ich_section_code: str
    ich_section_name: str
    similarity_score: float
    boosted_score: float
    confidence: MatchConfidence
    match_method: str
    # Optional enrichment fields (PTCV-96: sub-section matching)
    sub_section_code: str = ""
    summarization_score: float = -1.0
    composite_score: float = -1.0


@dataclasses.dataclass(frozen=True)
class SectionMapping:
    """Complete mapping result for one protocol header.

    Attributes:
        protocol_section_number: Section number from the protocol
            (e.g. ``"5.1"``).
        protocol_section_title: Section title (e.g.
            ``"Inclusion Criteria"``).
        matches: Ordered list of candidate matches (best first).
        auto_mapped: ``True`` if the top match is HIGH confidence.
    """

    protocol_section_number: str
    protocol_section_title: str
    matches: list[SectionMatch]
    auto_mapped: bool


@dataclasses.dataclass
class MatchResult:
    """Aggregate result from :meth:`SectionMatcher.match`.

    Attributes:
        mappings: Per-header mapping results.
        auto_mapped_count: Headers auto-mapped (HIGH).
        review_count: Headers flagged for review (REVIEW).
        unmapped_count: Headers with no strong match (LOW).
        auto_map_rate: Fraction auto-mapped (0.0–1.0).
    """

    mappings: list[SectionMapping]
    auto_mapped_count: int
    review_count: int
    unmapped_count: int
    auto_map_rate: float


# -----------------------------------------------------------------------
# Scoring constants (PTCV-135)
# -----------------------------------------------------------------------
# Cap the denominator in keyword scoring to prevent dense sections
# (e.g. B.4 with 35 max_possible) from diluting single-keyword
# matches relative to sparse sections (e.g. B.6 with 14).
_DENOM_CAP = 20

# Keywords with this many characters or fewer use word-boundary
# matching to prevent false positives (e.g. "dose" in "overdose").
_SHORT_KW_THRESHOLD = 5

# -----------------------------------------------------------------------
# Content-length hint for enriching query embeddings
# -----------------------------------------------------------------------
_CONTENT_HINT_LEN = 200  # chars from content_spans to append

# -----------------------------------------------------------------------
# TOC order disambiguation (PTCV-160)
# -----------------------------------------------------------------------
# When the top-2 boosted scores for a header are within this delta,
# the header is considered "ambiguous" and TOC ordering can act as a
# tiebreaker.
_TOC_AMBIGUITY_DELTA = 0.10

# Additive boost applied to the candidate whose ICH section fits the
# sequential TOC ordering relative to its neighbors.
_TOC_ORDER_BOOST = 0.06


# -----------------------------------------------------------------------
# SectionMatcher
# -----------------------------------------------------------------------


class SectionMatcher:
    """Map protocol section headers to ICH E6(R3) Appendix B sections.

    Uses deterministic keyword/pattern scoring from ``IchSectionDef``
    plus a synonym boost table for well-known protocol header synonyms.

    .. note::

        Cohere embedding support was removed in PTCV-102.  The
        ``cohere_api_key`` parameter is accepted but ignored for
        backward compatibility.

    Args:
        cohere_api_key: **Ignored** (kept for backward compatibility).
        auto_threshold: Minimum boosted score for auto-mapping
            (default ``0.75``).
        review_threshold: Minimum boosted score for review tier
            (default ``0.50``).
        synonym_boost: Additive boost for known synonyms
            (default ``0.15``).
    """

    def __init__(
        self,
        cohere_api_key: Optional[str] = None,
        auto_threshold: float = 0.75,
        review_threshold: float = 0.50,
        synonym_boost: float = 0.15,
    ) -> None:
        self._auto_threshold = auto_threshold
        self._review_threshold = review_threshold
        self._synonym_boost = synonym_boost

        # Load ICH schema — 16 sections (B.1 through B.16)
        schema = load_ich_schema()
        self._section_defs: dict[str, IchSectionDef] = (
            schema.sections
        )
        self._ref_codes: list[str] = sorted(
            self._section_defs.keys(),
            key=lambda c: self._section_defs[c].render_order,
        )

        # Keyword-only mode (PTCV-102: Cohere removed)
        self._use_embeddings = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self, protocol_index: ProtocolIndex
    ) -> MatchResult:
        """Map all protocol headers to ICH Appendix B sections.

        Args:
            protocol_index: Navigable index from the TOC extractor.

        Returns:
            :class:`MatchResult` with per-header mappings and
            aggregate statistics.
        """
        headers = self._collect_headers(protocol_index)
        if not headers:
            return MatchResult(
                mappings=[],
                auto_mapped_count=0,
                review_count=0,
                unmapped_count=0,
                auto_map_rate=0.0,
            )

        # Build query texts enriched with content hints
        query_texts: list[str] = []
        for number, title in headers:
            qt = f"{number} {title}"
            snippet = protocol_index.content_spans.get(
                number, ""
            )[:_CONTENT_HINT_LEN]
            if snippet:
                qt = f"{qt} {snippet}"
            query_texts.append(qt)

        # Score matrix: [n_headers x n_ref_sections]
        scores = self._keyword_scores(
            headers, protocol_index
        )
        method = "keyword_fallback"

        # --- Pass 1: compute boosted scores for all headers ----------
        all_raw: list[list[float]] = []
        all_boosted: list[list[float]] = []

        for idx, (number, title) in enumerate(headers):
            raw_scores = scores[idx]

            # PTCV-157 Tier 2: Zero all scores for garbage headers
            # (citations, OCR damage, body text posing as headers).
            if _is_garbage_header(title):
                raw_scores = [0.0] * len(raw_scores)

            # PTCV-171: Boost scores from markdown headings in
            # content_spans before applying synonym boost.
            content = protocol_index.content_spans.get(
                number, ""
            )
            heading_scores = self._apply_heading_boost(
                content, raw_scores
            )
            boosted = self._apply_synonym_boost(
                title, heading_scores
            )
            all_raw.append(raw_scores)
            all_boosted.append(boosted)

        # PTCV-160: Apply TOC order hints for ambiguous REVIEW cases
        all_boosted = self._apply_toc_order_hints(
            headers, all_boosted, protocol_index
        )

        # --- Pass 2: classify with potentially adjusted scores --------
        mappings: list[SectionMapping] = []
        auto_count = 0
        review_count = 0
        unmapped_count = 0

        for idx, (number, title) in enumerate(headers):
            mapping = self._classify_header(
                number, title, all_raw[idx],
                all_boosted[idx], method,
            )
            mappings.append(mapping)

            if mapping.auto_mapped:
                auto_count += 1
            elif (
                mapping.matches
                and mapping.matches[0].confidence
                == MatchConfidence.REVIEW
            ):
                review_count += 1
            else:
                unmapped_count += 1

        total = len(headers)
        return MatchResult(
            mappings=mappings,
            auto_mapped_count=auto_count,
            review_count=review_count,
            unmapped_count=unmapped_count,
            auto_map_rate=auto_count / total if total else 0.0,
        )

    # ------------------------------------------------------------------
    # Header collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_headers(
        protocol_index: ProtocolIndex,
    ) -> list[tuple[str, str]]:
        """Extract (number, title) pairs from the protocol index.

        Prefers ``toc_entries``; falls back to ``section_headers``.
        """
        if protocol_index.toc_entries:
            return [
                (e.number, e.title)
                for e in protocol_index.toc_entries
            ]
        return [
            (h.number, h.title)
            for h in protocol_index.section_headers
        ]

    # ------------------------------------------------------------------
    # Scoring — keyword/pattern matching
    # ------------------------------------------------------------------

    def _keyword_scores(
        self,
        headers: list[tuple[str, str]],
        protocol_index: ProtocolIndex,
    ) -> list[list[float]]:
        """Score each header against all ICH sections via patterns."""
        scores: list[list[float]] = []
        for number, title in headers:
            content = protocol_index.content_spans.get(
                number, ""
            )
            row = [
                self._keyword_score(
                    title, content, self._section_defs[code]
                )
                for code in self._ref_codes
            ]
            scores.append(row)
        return scores

    @staticmethod
    def _keyword_score(
        header_title: str,
        section_text: str,
        ich_def: IchSectionDef,
    ) -> float:
        """Score a header against one ICH section using patterns.

        Reuses the weighted formula from
        ``RuleBasedClassifier._score_block``:
        ``(pattern_hits * 2 + keyword_hits) / max_possible``.

        PTCV-135 improvements:
        - Denominator capped at ``_DENOM_CAP`` to normalise scores
          across sparse and dense sections.
        - Short keywords (<=5 chars) use word-boundary matching to
          prevent false positives (e.g. "dose" in "overdose").
        """
        combined = (
            header_title + " " + section_text[:500]
        ).lower()

        pattern_hits = sum(
            1
            for p in ich_def.patterns
            if re.search(p, combined, re.IGNORECASE)
        )
        keyword_hits = 0
        for kw in ich_def.keywords:
            kw_lower = kw.lower()
            if len(kw_lower) <= _SHORT_KW_THRESHOLD:
                # Word-boundary match for short keywords
                if re.search(
                    r"\b" + re.escape(kw_lower) + r"\b",
                    combined,
                ):
                    keyword_hits += 1
            else:
                # Substring match for longer keywords
                if kw_lower in combined:
                    keyword_hits += 1

        raw_max = (
            len(ich_def.patterns) * 2 + len(ich_def.keywords)
        )
        if raw_max == 0:
            return 0.0
        max_possible = min(raw_max, _DENOM_CAP)
        return min(
            1.0,
            (pattern_hits * 2 + keyword_hits) / max_possible,
        )

    # ------------------------------------------------------------------
    # Markdown heading boost (PTCV-171)
    # ------------------------------------------------------------------

    def _apply_heading_boost(
        self,
        content: str,
        raw_scores: list[float],
    ) -> list[float]:
        """Boost scores from markdown headings in content (PTCV-171).

        Markdown headings (``## Title``) are unambiguous structural
        signals from pymupdf4llm output.  When heading text matches
        ICH section synonyms, the corresponding section gets a
        confidence boost proportional to heading level.

        Args:
            content: Section content text (may contain Markdown).
            raw_scores: Per-ICH-section keyword scores.

        Returns:
            Boosted score list (new list; input not mutated).
        """
        headings = _MD_HEADING_RE.findall(content[:3000])
        if not headings:
            return raw_scores

        boosted = list(raw_scores)

        for hashes, heading_text in headings:
            level = len(hashes)
            boost_value = _HEADING_LEVEL_BOOST.get(level, 0.02)
            heading_lower = heading_text.strip().lower()
            # Strip section numbers from heading text so
            # "## 5.1 Inclusion Criteria" becomes "inclusion criteria"
            heading_lower = re.sub(
                r"^\d+(?:\.\d+)*\.?\s+", "", heading_lower
            )
            # Also strip bold markers from un-normalized headings
            heading_lower = heading_lower.replace("**", "")

            for synonym, ich_code in _SYNONYM_BOOSTS.items():
                if (
                    synonym in heading_lower
                    and ich_code in self._ref_codes
                ):
                    idx = self._ref_codes.index(ich_code)
                    boosted[idx] = min(
                        1.0, boosted[idx] + boost_value
                    )

        return boosted

    # ------------------------------------------------------------------
    # Synonym boost
    # ------------------------------------------------------------------

    def _apply_synonym_boost(
        self, title: str, raw_scores: list[float]
    ) -> list[float]:
        """Apply synonym boost and B.1 exclusions for one header.

        Positive boosts from ``_SYNONYM_BOOSTS`` are applied first,
        then B.1 exclusion terms (PTCV-155) zero out the B.1 score
        for headers that should never map to General Information.
        """
        boosted = list(raw_scores)
        title_lower = title.lower()
        for synonym, ich_code in _SYNONYM_BOOSTS.items():
            if synonym in title_lower and ich_code in self._ref_codes:
                idx = self._ref_codes.index(ich_code)
                boosted[idx] = min(
                    1.0, boosted[idx] + self._synonym_boost
                )

        # PTCV-155 Tier 2: Prevent noise headers from mapping to B.1.
        if "B.1" in self._ref_codes:
            for excl_term in _B1_EXCLUSION_TERMS:
                if excl_term in title_lower:
                    b1_idx = self._ref_codes.index("B.1")
                    boosted[b1_idx] = 0.0
                    break

        return boosted

    # ------------------------------------------------------------------
    # TOC order disambiguation (PTCV-160)
    # ------------------------------------------------------------------

    def _apply_toc_order_hints(
        self,
        headers: list[tuple[str, str]],
        all_boosted: list[list[float]],
        protocol_index: ProtocolIndex,
    ) -> list[list[float]]:
        """Boost correctly-ordered sections for ambiguous headers.

        When PTCV-171's heading classification produces close REVIEW-tier
        scores (top-2 within ``_TOC_AMBIGUITY_DELTA``), the TOC entry
        ordering disambiguates by checking which candidate fits the
        sequential ICH section order relative to neighboring headers.

        Args:
            headers: ``(number, title)`` pairs for all protocol headers.
            all_boosted: Per-header boosted score lists from pass 1.
            protocol_index: Source protocol index (checked for TOC).

        Returns:
            Possibly-adjusted boosted score lists (new outer list;
            inner lists copied only when modified).
        """
        if (
            not protocol_index.toc_found
            or not protocol_index.toc_entries
        ):
            return all_boosted

        code_order: dict[str, int] = {
            code: i for i, code in enumerate(self._ref_codes)
        }
        n_codes = len(self._ref_codes)
        result = list(all_boosted)  # shallow copy; inner lists shared

        def _top_order(boosted: list[float]) -> int:
            """Return ICH ordinal of the top-scoring section."""
            best_idx = max(range(n_codes), key=lambda j: boosted[j])
            return code_order.get(
                self._ref_codes[best_idx], -1
            )

        hints_applied = 0
        total_ambiguous = 0

        for i, boosted in enumerate(result):
            ranked = sorted(
                range(n_codes),
                key=lambda j: boosted[j],
                reverse=True,
            )
            top_score = boosted[ranked[0]]

            # Only REVIEW tier is eligible for disambiguation
            if (
                top_score >= self._auto_threshold
                or top_score < self._review_threshold
            ):
                continue
            if len(ranked) < 2:
                continue

            second_score = boosted[ranked[1]]
            if top_score - second_score > _TOC_AMBIGUITY_DELTA:
                continue  # Clear winner — no hint needed

            total_ambiguous += 1

            # Resolve neighbor ICH ordinals
            prev_ord = _top_order(result[i - 1]) if i > 0 else -1
            next_ord = (
                _top_order(result[i + 1])
                if i < len(result) - 1
                else -1
            )
            if prev_ord < 0 and next_ord < 0:
                continue

            cand1_ord = code_order.get(
                self._ref_codes[ranked[0]], -1
            )
            cand2_ord = code_order.get(
                self._ref_codes[ranked[1]], -1
            )

            def _fits(section_ord: int) -> bool:
                if prev_ord >= 0 and next_ord >= 0:
                    return prev_ord < section_ord < next_ord
                if prev_ord >= 0:
                    return section_ord > prev_ord
                return section_ord < next_ord

            c1_fits = _fits(cand1_ord)
            c2_fits = _fits(cand2_ord)

            if c2_fits and not c1_fits:
                # TOC ordering favours the second candidate — boost it
                adjusted = list(boosted)
                adjusted[ranked[1]] = min(
                    1.0, adjusted[ranked[1]] + _TOC_ORDER_BOOST
                )
                result[i] = adjusted
                hints_applied += 1
                logger.debug(
                    "TOC order hint: '%s' boosted %s over %s",
                    headers[i][1],
                    self._ref_codes[ranked[1]],
                    self._ref_codes[ranked[0]],
                )

        if total_ambiguous > 0:
            logger.info(
                "TOC order disambiguation: %d/%d ambiguous "
                "headers resolved",
                hints_applied,
                total_ambiguous,
            )

        return result

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_header(
        self,
        number: str,
        title: str,
        raw_scores: list[float],
        boosted_scores: list[float],
        method: str,
    ) -> SectionMapping:
        """Classify one header into HIGH/REVIEW/LOW matches."""
        # Build (index, boosted, raw) sorted by boosted desc
        ranked = sorted(
            range(len(self._ref_codes)),
            key=lambda i: boosted_scores[i],
            reverse=True,
        )

        top_boosted = boosted_scores[ranked[0]]

        if top_boosted >= self._auto_threshold:
            # HIGH — include all matches above auto_threshold
            matches = [
                self._make_match(
                    ranked[i],
                    raw_scores[ranked[i]],
                    boosted_scores[ranked[i]],
                    MatchConfidence.HIGH,
                    method,
                )
                for i in range(len(ranked))
                if boosted_scores[ranked[i]]
                >= self._auto_threshold
            ]
            return SectionMapping(
                protocol_section_number=number,
                protocol_section_title=title,
                matches=matches,
                auto_mapped=True,
            )

        if top_boosted >= self._review_threshold:
            # REVIEW — top-3 candidates
            matches = [
                self._make_match(
                    ranked[i],
                    raw_scores[ranked[i]],
                    boosted_scores[ranked[i]],
                    MatchConfidence.REVIEW,
                    method,
                )
                for i in range(min(3, len(ranked)))
            ]
            return SectionMapping(
                protocol_section_number=number,
                protocol_section_title=title,
                matches=matches,
                auto_mapped=False,
            )

        # LOW — best candidate only
        matches = [
            self._make_match(
                ranked[0],
                raw_scores[ranked[0]],
                boosted_scores[ranked[0]],
                MatchConfidence.LOW,
                method,
            )
        ]
        return SectionMapping(
            protocol_section_number=number,
            protocol_section_title=title,
            matches=matches,
            auto_mapped=False,
        )

    def _make_match(
        self,
        ref_idx: int,
        raw_score: float,
        boosted_score: float,
        confidence: MatchConfidence,
        method: str,
    ) -> SectionMatch:
        """Build a SectionMatch from a reference index."""
        code = self._ref_codes[ref_idx]
        return SectionMatch(
            ich_section_code=code,
            ich_section_name=self._section_defs[code].name,
            similarity_score=round(raw_score, 4),
            boosted_score=round(boosted_score, 4),
            confidence=confidence,
            match_method=method,
        )
