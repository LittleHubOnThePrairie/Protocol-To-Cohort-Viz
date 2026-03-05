"""Semantic section matcher — protocol headers to Appendix B (PTCV-90).

Maps protocol section headers (from ``ProtocolIndex``) to ICH E6(R3)
Appendix B section codes using semantic similarity.  Two strategies:

1. **Embedding-based** — Cohere ``embed-english-v3.0`` cosine
   similarity (same SDK pattern as ``RAGClassifier``).
2. **Deterministic fallback** — keyword/pattern scoring from
   ``IchSectionDef`` (no API key required, CI-safe).

A synonym boost table provides additive score increases for
well-known protocol header synonyms (e.g. "Inclusion Criteria"
→ B.5), improving precision without brittle hard-coded overrides.

.. note::

   The ``ich_e6r3_schema.yaml`` has a name/pattern mismatch for
   B.7–B.10: section *names* are offset from the *patterns* they
   contain.  The synonym table follows the **patterns** (which the
   existing classifier pipeline uses), not the names.  For example,
   "adverse event" maps to B.9 because B.9's patterns match safety
   content, even though B.9 is named "Statistics".

Risk tier: LOW — read-only semantic analysis.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import os
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
# Follows the *pattern* semantics in ich_e6r3_schema.yaml, NOT the
# section names (see module docstring for the B.7-B.10 caveat).

_SYNONYM_BOOSTS: dict[str, str] = {
    # B.1 General Information
    "general information": "B.1",
    "protocol summary": "B.1",
    "administrative information": "B.1",
    "protocol title": "B.1",
    "protocol number": "B.1",
    # B.2 Background Information
    "background": "B.2",
    "rationale": "B.2",
    "introduction": "B.2",
    "literature review": "B.2",
    "preclinical": "B.2",
    "nonclinical": "B.2",
    # B.3 Objectives
    "objective": "B.3",
    "purpose": "B.3",
    "endpoints": "B.3",
    "hypothesis": "B.3",
    "primary endpoint": "B.3",
    "secondary endpoint": "B.3",
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
    # B.6 Treatment of Subjects
    "discontinuation": "B.6",
    "withdrawal": "B.6",
    "stopping rules": "B.6",
    "treatment termination": "B.6",
    # B.7 Assessment of Efficacy (patterns = treatment/dosing/IMP)
    "investigational product": "B.7",
    "study treatment": "B.7",
    "study drug": "B.7",
    "dosage": "B.7",
    "dose modification": "B.7",
    "concomitant medication": "B.7",
    "treatment schedule": "B.7",
    # B.8 Assessment of Safety (patterns = efficacy assessment)
    "efficacy assessment": "B.8",
    "clinical outcome": "B.8",
    "response criteria": "B.8",
    "tumour response": "B.8",
    "tumor response": "B.8",
    # B.9 Statistics (patterns = safety/AEs)
    "adverse event": "B.9",
    "safety assessment": "B.9",
    "safety monitoring": "B.9",
    "vital signs": "B.9",
    "laboratory test": "B.9",
    "safety reporting": "B.9",
    # B.10 Direct Access (patterns = statistical analysis)
    "statistical": "B.10",
    "sample size": "B.10",
    "analysis population": "B.10",
    "statistical analysis plan": "B.10",
    "power calculation": "B.10",
    # B.11 Quality Control and Quality Assurance
    "quality control": "B.11",
    "quality assurance": "B.11",
    "monitoring plan": "B.11",
    "source data verification": "B.11",
    "data management": "B.11",
    # B.12 Ethics
    "ethics": "B.12",
    "informed consent": "B.12",
    "irb": "B.12",
    "ethics committee": "B.12",
    "ethical approval": "B.12",
    # B.13 Data Handling and Record Keeping
    "data handling": "B.13",
    "record keeping": "B.13",
    "case report form": "B.13",
    "crf": "B.13",
    "archiving": "B.13",
    "data retention": "B.13",
    # B.14 Financing, Insurance, and Publication Policy
    "financing": "B.14",
    "insurance": "B.14",
    "publication": "B.14",
    "compensation": "B.14",
    "indemnity": "B.14",
}


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
# Content-length hint for enriching query embeddings
# -----------------------------------------------------------------------
_CONTENT_HINT_LEN = 200  # chars from content_spans to append


# -----------------------------------------------------------------------
# SectionMatcher
# -----------------------------------------------------------------------


class SectionMatcher:
    """Map protocol section headers to ICH E6(R3) Appendix B sections.

    Supports embedding-based matching (Cohere embed-english-v3.0) and
    a deterministic keyword/pattern fallback for CI environments.

    Args:
        cohere_api_key: Cohere API key.  If *None* or empty, falls
            back to deterministic keyword matching.
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

        # Load ICH schema — 14 sections (B.1 through B.14)
        schema = load_ich_schema()
        self._section_defs: dict[str, IchSectionDef] = (
            schema.sections
        )
        self._ref_codes: list[str] = sorted(
            self._section_defs.keys(),
            key=lambda c: self._section_defs[c].render_order,
        )
        self._ref_texts: list[str] = [
            (
                f"{code} {self._section_defs[code].name}: "
                f"{self._section_defs[code].description} "
                f"{' '.join(self._section_defs[code].keywords)}"
            )
            for code in self._ref_codes
        ]

        # Embedding mode
        api_key = cohere_api_key or os.environ.get(
            "COHERE_API_KEY", ""
        )
        self._use_embeddings = bool(api_key)
        self._np: Any = None
        self._co: Any = None
        self._ref_embeddings: Any = None

        if self._use_embeddings:
            self._init_embeddings(api_key)

    def _init_embeddings(self, api_key: str) -> None:
        """Lazy-import Cohere/numpy and pre-compute ref embeddings."""
        import cohere as cohere_sdk
        import numpy as np

        self._np = np
        self._co = cohere_sdk.Client(api_key=api_key)

        resp = self._co.embed(
            texts=self._ref_texts,
            model="embed-english-v3.0",
            input_type="search_document",
        )
        self._ref_embeddings = np.array(
            resp.embeddings, dtype=np.float32
        )
        logger.info(
            "Section matcher: embedded %d reference sections",
            len(self._ref_codes),
        )

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
        if self._use_embeddings:
            scores = self._embedding_scores(query_texts)
            method = "embedding"
        else:
            scores = self._keyword_scores(
                headers, protocol_index
            )
            method = "keyword_fallback"

        # Apply synonym boosts and classify
        mappings: list[SectionMapping] = []
        auto_count = 0
        review_count = 0
        unmapped_count = 0

        for idx, (number, title) in enumerate(headers):
            raw_scores = scores[idx]
            boosted = self._apply_synonym_boost(
                title, raw_scores
            )
            mapping = self._classify_header(
                number, title, raw_scores, boosted, method
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
    # Scoring — embedding path
    # ------------------------------------------------------------------

    def _embedding_scores(
        self, query_texts: list[str]
    ) -> list[list[float]]:
        """Batch-embed query texts and compute cosine similarity."""
        np = self._np
        resp = self._co.embed(
            texts=[t[:8192] for t in query_texts],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        query_vecs = np.array(
            resp.embeddings, dtype=np.float32
        )

        # Cosine similarity: [n_queries x n_refs]
        q_norms = np.linalg.norm(
            query_vecs, axis=1, keepdims=True
        )
        r_norms = np.linalg.norm(
            self._ref_embeddings, axis=1, keepdims=True
        )
        q_norms = np.where(q_norms == 0, 1e-9, q_norms)
        r_norms = np.where(r_norms == 0, 1e-9, r_norms)
        sim_matrix = (query_vecs / q_norms) @ (
            self._ref_embeddings / r_norms
        ).T

        return sim_matrix.tolist()

    # ------------------------------------------------------------------
    # Scoring — deterministic fallback
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
        """
        combined = (
            header_title + " " + section_text[:500]
        ).lower()

        pattern_hits = sum(
            1
            for p in ich_def.patterns
            if re.search(p, combined, re.IGNORECASE)
        )
        keyword_hits = sum(
            1
            for kw in ich_def.keywords
            if kw.lower() in combined
        )

        max_possible = (
            len(ich_def.patterns) * 2 + len(ich_def.keywords)
        )
        if max_possible == 0:
            return 0.0
        return (pattern_hits * 2 + keyword_hits) / max_possible

    # ------------------------------------------------------------------
    # Synonym boost
    # ------------------------------------------------------------------

    def _apply_synonym_boost(
        self, title: str, raw_scores: list[float]
    ) -> list[float]:
        """Apply synonym boost to raw scores for one header."""
        boosted = list(raw_scores)
        title_lower = title.lower()
        for synonym, ich_code in _SYNONYM_BOOSTS.items():
            if synonym in title_lower and ich_code in self._ref_codes:
                idx = self._ref_codes.index(ich_code)
                boosted[idx] = min(
                    1.0, boosted[idx] + self._synonym_boost
                )
        return boosted

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
