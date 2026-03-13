"""Sub-section matching with LLM summarization (PTCV-96).

Enrichment layer that post-processes :class:`MatchResult` from
:class:`SectionMatcher` to produce sub-section-level matches
(B.5.1, B.5.2, etc.) with optional LLM-driven semantic fit scoring.

Three scoring signals are combined into a composite score:

1. **Embedding/keyword score** — inherited from the parent-level match.
2. **Keyword overlap** — sub-section description keywords vs content.
3. **LLM summarization** — Claude Sonnet assesses semantic fit
   (optional; graceful fallback when no API key).

Risk tier: LOW — read-only enrichment, no mutations to source data.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Sequence

from ptcv.ich_parser.query_schema import (
    AppendixBQuery,
    load_query_schema,
)
from ptcv.ich_parser.section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMatch,
    SectionMapping,
)
from ptcv.ich_parser.toc_extractor import ProtocolIndex

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SubSectionDef:
    """Definition of one ICH E6(R3) Appendix B sub-section.

    Attributes:
        code: Sub-section code (e.g. ``"B.5.1"``).
        parent_code: Top-level parent section (e.g. ``"B.5"``).
        name: Human-readable title derived from query text.
        description: Concatenated query texts for semantic matching.
        query_ids: Query IDs belonging to this sub-section.
    """

    code: str
    parent_code: str
    name: str
    description: str
    query_ids: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SummarizationResult:
    """Result of a single LLM semantic-fit assessment.

    Attributes:
        sub_section_code: The sub-section evaluated.
        score: Semantic fit score (0.0–1.0).
        rationale: Brief LLM-generated explanation.
    """

    sub_section_code: str
    score: float
    rationale: str


@dataclasses.dataclass(frozen=True)
class SubSectionMatch:
    """A candidate match at sub-section granularity.

    Attributes:
        sub_section_code: Sub-section code (e.g. ``"B.5.1"``).
        parent_section_code: Parent section (e.g. ``"B.5"``).
        sub_section_name: Human-readable name.
        embedding_score: From parent match similarity.
        keyword_score: Keyword overlap with sub-section description.
        summarization_score: LLM semantic fit (``-1.0`` if unavailable).
        composite_score: Weighted combination of all signals.
        confidence: Tier derived from composite_score.
        match_method: ``"embedding+summarization"`` or
            ``"keyword_fallback"``.
    """

    sub_section_code: str
    parent_section_code: str
    sub_section_name: str
    embedding_score: float
    keyword_score: float
    summarization_score: float
    composite_score: float
    confidence: MatchConfidence
    match_method: str


@dataclasses.dataclass(frozen=True)
class EnrichedSectionMapping:
    """A :class:`SectionMapping` enriched with sub-section detail.

    Preserves parent-level ``matches`` for backward compatibility
    and adds ``sub_section_matches`` with finer-grained scoring.

    Attributes:
        protocol_section_number: Protocol section number.
        protocol_section_title: Protocol section title.
        matches: Parent-level :class:`SectionMatch` list (preserved).
        auto_mapped: Whether top match was HIGH confidence.
        sub_section_matches: Ordered sub-section matches (best first).
        parent_coverage: Set of matched sub-section codes.
    """

    protocol_section_number: str
    protocol_section_title: str
    matches: list[SectionMatch]
    auto_mapped: bool
    sub_section_matches: list[SubSectionMatch]
    parent_coverage: frozenset[str]


@dataclasses.dataclass
class EnrichedMatchResult:
    """Match result enriched with sub-section granularity.

    First 5 fields mirror :class:`MatchResult` for backward
    compatibility.  Additional fields carry the enrichment.

    Attributes:
        mappings: Original parent-level mappings (unchanged).
        auto_mapped_count: Parent auto-mapped count.
        review_count: Parent review count.
        unmapped_count: Parent unmapped count.
        auto_map_rate: Parent auto-map fraction.
        enriched_mappings: Per-header enriched mappings.
        sub_section_auto_map_rate: Fraction of sub-section matches
            with HIGH confidence.
        llm_calls_made: Number of LLM API calls performed.
        llm_fallback: ``True`` if LLM was unavailable.
        elapsed_seconds: Wall-clock time for the enrichment stage.
    """

    mappings: list[SectionMapping]
    auto_mapped_count: int
    review_count: int
    unmapped_count: int
    auto_map_rate: float
    enriched_mappings: list[EnrichedSectionMapping]
    sub_section_auto_map_rate: float
    llm_calls_made: int
    llm_fallback: bool
    elapsed_seconds: float = 0.0


# -----------------------------------------------------------------------
# Sub-section registry
# -----------------------------------------------------------------------

_REGISTRY_CACHE: dict[str, SubSectionDef] | None = None


def _derive_name(query_text: str) -> str:
    """Derive a short sub-section name from a query_text string.

    Strips "What is/are the" prefixes and trailing punctuation.
    """
    text = query_text.strip()
    # Remove common question prefixes
    for prefix in (
        "What is the ",
        "What are the ",
        "What is ",
        "What are ",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    # Remove trailing question mark
    text = text.rstrip("?").strip()
    # Capitalise first letter
    if text:
        text = text[0].upper() + text[1:]
    return text


def build_subsection_registry(
    queries: Sequence[AppendixBQuery] | None = None,
) -> dict[str, SubSectionDef]:
    """Build sub-section definitions from the Appendix B query schema.

    Groups queries by ``section_id``, derives a human-readable name
    from the first query's text, and concatenates all query texts as
    the semantic description.

    Args:
        queries: Pre-loaded query list; loads from YAML if *None*.

    Returns:
        Dict mapping sub-section code (e.g. ``"B.5.1"``) to
        :class:`SubSectionDef`.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None and queries is None:
        return _REGISTRY_CACHE

    if queries is None:
        queries = load_query_schema()

    groups: dict[str, list[AppendixBQuery]] = defaultdict(list)
    for q in queries:
        groups[q.section_id].append(q)

    registry: dict[str, SubSectionDef] = {}
    for section_id, section_queries in sorted(groups.items()):
        first_q = section_queries[0]
        name = _derive_name(first_q.query_text)
        description = " ".join(q.query_text for q in section_queries)
        query_ids = tuple(q.query_id for q in section_queries)

        registry[section_id] = SubSectionDef(
            code=section_id,
            parent_code=first_q.parent_section,
            name=name,
            description=description,
            query_ids=query_ids,
        )

    if queries is None or _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = registry

    logger.info(
        "Built sub-section registry: %d sub-sections across %d parents",
        len(registry),
        len({d.parent_code for d in registry.values()}),
    )
    return registry


def _reset_registry_cache() -> None:
    """Clear the registry cache (testing only)."""
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = None


def get_subsections_for_parent(
    parent_code: str,
    registry: dict[str, SubSectionDef] | None = None,
) -> list[SubSectionDef]:
    """Return sub-section definitions for a given parent code.

    Args:
        parent_code: Top-level section code (e.g. ``"B.5"``).
        registry: Pre-built registry; builds from YAML if *None*.

    Returns:
        List of :class:`SubSectionDef` for the parent, sorted by code.
    """
    if registry is None:
        registry = build_subsection_registry()
    return sorted(
        (d for d in registry.values() if d.parent_code == parent_code),
        key=lambda d: d.code,
    )


# -----------------------------------------------------------------------
# Composite scoring
# -----------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "embedding": 0.45,
    "keyword": 0.15,
    "summarization": 0.40,
}

_FALLBACK_WEIGHTS: dict[str, float] = {
    "embedding": 0.65,
    "keyword": 0.35,
    "summarization": 0.0,
}


def compute_composite_score(
    embedding_score: float,
    keyword_score: float,
    summarization_score: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted composite of the three scoring signals.

    If *summarization_score* is ``-1.0`` (unavailable), automatically
    switches to fallback weights that redistribute the summarization
    share to embedding and keyword.

    Args:
        embedding_score: Embedding similarity (0.0–1.0).
        keyword_score: Keyword overlap (0.0–1.0).
        summarization_score: LLM semantic fit (0.0–1.0 or ``-1.0``).
        weights: Custom weight dict; uses defaults if *None*.

    Returns:
        Composite score (0.0–1.0), rounded to 4 decimals.
    """
    if summarization_score < 0:
        w = _FALLBACK_WEIGHTS
        summ = 0.0
    else:
        w = weights or _DEFAULT_WEIGHTS
        summ = summarization_score

    score = (
        w["embedding"] * embedding_score
        + w["keyword"] * keyword_score
        + w["summarization"] * summ
    )
    return round(min(1.0, max(0.0, score)), 4)


# -----------------------------------------------------------------------
# LLM prompt
# -----------------------------------------------------------------------

_SUMMARIZATION_PROMPT = """\
You are an ICH E6(R3) GCP expert assessing semantic fit between \
a clinical protocol section and an ICH Appendix B sub-section.

Protocol section: "{protocol_title}"
Content excerpt (first {max_chars} chars):
<content>
{content_excerpt}
</content>

Candidate ICH sub-section: {sub_code} — {sub_name}
Sub-section queries (expected content):
{sub_description}

Rate the semantic fit on a scale of 0.0 to 1.0:
- 1.0: Content directly and comprehensively addresses this sub-section
- 0.7-0.9: Strong match with most key concepts present
- 0.4-0.6: Partial match, some relevant content
- 0.1-0.3: Weak match, tangentially related
- 0.0: No relevant content

Respond with ONLY a JSON object (no markdown fences):
{{"score": <float>, "rationale": "<one sentence>"}}"""

# Cache key prefix length for content hashing
_MAX_CONTENT_CHARS = 2000


# -----------------------------------------------------------------------
# SummarizationMatcher
# -----------------------------------------------------------------------


class SummarizationMatcher:
    """Enrich parent-level section matches with sub-section detail.

    Post-processes a :class:`MatchResult` to classify content at
    sub-section granularity, optionally using LLM summarization
    to assess semantic fit.

    Args:
        anthropic_api_key: Anthropic API key.  If *None* or empty,
            LLM scoring is skipped (graceful fallback).
        claude_model: Model ID for summarization calls.
        auto_threshold: Minimum composite score for HIGH confidence.
        review_threshold: Minimum composite score for REVIEW tier.
        weights: Custom composite scoring weights.
        max_content_chars: Maximum content chars sent to LLM.
        max_concurrent: Maximum concurrent LLM calls.  Defaults to
            ``10``.  Set to ``1`` for sequential execution.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        # PTCV-188: Primary model with graceful degradation.
        claude_model: str = "claude-sonnet-4-6",
        auto_threshold: float = 0.75,
        review_threshold: float = 0.50,
        weights: dict[str, float] | None = None,
        max_content_chars: int = _MAX_CONTENT_CHARS,
        max_concurrent: int = 10,
    ) -> None:
        self._auto_threshold = auto_threshold
        self._review_threshold = review_threshold
        self._weights = weights
        self._max_chars = max_content_chars
        self._model = claude_model
        self._fallback_model = "claude-sonnet-4-20250514"
        self._max_concurrent = max(1, max_concurrent)

        api_key = anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
        self._use_llm = bool(api_key)
        self._client: Any = None

        if self._use_llm:
            self._init_client(api_key)

        # In-memory summarization cache
        self._cache: dict[str, SummarizationResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._llm_calls = 0
        self._refusal_count = 0
        self._fallback_count = 0

    def _init_client(self, api_key: str) -> None:
        """Lazy-import Anthropic SDK and create client."""
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(
                "SummarizationMatcher: Anthropic client initialised "
                "(model=%s)",
                self._model,
            )
        except Exception:
            logger.warning(
                "SummarizationMatcher: Failed to init Anthropic client, "
                "falling back to keyword-only scoring",
                exc_info=True,
            )
            self._use_llm = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(
        self,
        match_result: MatchResult,
        protocol_index: ProtocolIndex,
        progress_callback: Any = None,
    ) -> EnrichedMatchResult:
        """Enrich a :class:`MatchResult` with sub-section matches.

        For each parent-level mapping:

        1. Look up sub-sections of the matched parent code.
        2. Score content against each sub-section using keyword
           overlap and (optionally) LLM summarization.
        3. Compute composite score and classify confidence.
        4. Build :class:`EnrichedSectionMapping`.

        LLM calls are parallelized using a thread pool (PTCV-125).

        Args:
            match_result: Output from :meth:`SectionMatcher.match`.
            protocol_index: Document index for content access.
            progress_callback: Optional ``(done, total)`` callback
                invoked after each mapping is enriched (PTCV-126).

        Returns:
            :class:`EnrichedMatchResult` preserving all original
            fields plus sub-section enrichment.
        """
        t0 = time.monotonic()
        registry = build_subsection_registry()
        enriched_mappings: list[EnrichedSectionMapping] = []
        total_sub_high = 0
        total_sub = 0
        total_mappings = len(match_result.mappings)

        for i, mapping in enumerate(match_result.mappings):
            enriched = self._enrich_mapping(
                mapping, protocol_index, registry
            )
            enriched_mappings.append(enriched)
            if progress_callback is not None:
                progress_callback(i + 1, total_mappings)

            for sub in enriched.sub_section_matches:
                total_sub += 1
                if sub.confidence == MatchConfidence.HIGH:
                    total_sub_high += 1

        elapsed = time.monotonic() - t0
        sub_auto_rate = (
            total_sub_high / total_sub if total_sub else 0.0
        )

        logger.info(
            "SummarizationMatcher.refine completed in %.2fs "
            "(%d LLM calls, %d cache hits, %d cache misses)",
            elapsed,
            self._llm_calls,
            self._cache_hits,
            self._cache_misses,
        )

        return EnrichedMatchResult(
            mappings=match_result.mappings,
            auto_mapped_count=match_result.auto_mapped_count,
            review_count=match_result.review_count,
            unmapped_count=match_result.unmapped_count,
            auto_map_rate=match_result.auto_map_rate,
            enriched_mappings=enriched_mappings,
            sub_section_auto_map_rate=round(sub_auto_rate, 4),
            llm_calls_made=self._llm_calls,
            llm_fallback=not self._use_llm,
            elapsed_seconds=round(elapsed, 3),
        )

    # ------------------------------------------------------------------
    # Per-mapping enrichment
    # ------------------------------------------------------------------

    def _enrich_mapping(
        self,
        mapping: SectionMapping,
        protocol_index: ProtocolIndex,
        registry: dict[str, SubSectionDef],
    ) -> EnrichedSectionMapping:
        """Enrich a single :class:`SectionMapping`."""
        if not mapping.matches:
            return EnrichedSectionMapping(
                protocol_section_number=mapping.protocol_section_number,
                protocol_section_title=mapping.protocol_section_title,
                matches=mapping.matches,
                auto_mapped=mapping.auto_mapped,
                sub_section_matches=[],
                parent_coverage=frozenset(),
            )

        parent_code = mapping.matches[0].ich_section_code
        parent_score = mapping.matches[0].boosted_score
        sub_defs = get_subsections_for_parent(parent_code, registry)

        content = protocol_index.content_spans.get(
            mapping.protocol_section_number, ""
        )

        # Parallelize LLM calls across sub-sections (PTCV-125).
        # LLM calls are I/O-bound so threading provides near-linear
        # speedup.  When LLM is disabled, the overhead is negligible.
        if self._use_llm and len(sub_defs) > 1:
            sub_matches = self._score_subsections_parallel(
                sub_defs=sub_defs,
                content=content,
                protocol_title=mapping.protocol_section_title,
                parent_score=parent_score,
            )
        else:
            sub_matches = [
                self._score_subsection(
                    content=content,
                    protocol_title=mapping.protocol_section_title,
                    parent_score=parent_score,
                    sub_def=sub_def,
                )
                for sub_def in sub_defs
            ]

        # Sort by composite score descending
        sub_matches.sort(
            key=lambda m: m.composite_score, reverse=True
        )

        coverage = frozenset(
            m.sub_section_code
            for m in sub_matches
            if m.confidence in (MatchConfidence.HIGH, MatchConfidence.REVIEW)
        )

        return EnrichedSectionMapping(
            protocol_section_number=mapping.protocol_section_number,
            protocol_section_title=mapping.protocol_section_title,
            matches=mapping.matches,
            auto_mapped=mapping.auto_mapped,
            sub_section_matches=sub_matches,
            parent_coverage=coverage,
        )

    # ------------------------------------------------------------------
    # Parallel sub-section scoring (PTCV-125)
    # ------------------------------------------------------------------

    def _score_subsections_parallel(
        self,
        sub_defs: list[SubSectionDef],
        content: str,
        protocol_title: str,
        parent_score: float,
    ) -> list[SubSectionMatch]:
        """Score multiple sub-sections concurrently via thread pool.

        Each ``_score_subsection`` call is I/O-bound (LLM API), so
        threading provides near-linear speedup.  The thread count is
        capped by ``self._max_concurrent``.
        """
        workers = min(self._max_concurrent, len(sub_defs))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as pool:
            futures = {
                pool.submit(
                    self._score_subsection,
                    content=content,
                    protocol_title=protocol_title,
                    parent_score=parent_score,
                    sub_def=sub_def,
                ): sub_def
                for sub_def in sub_defs
            }
            results: list[SubSectionMatch] = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    sub_def = futures[future]
                    logger.warning(
                        "Parallel scoring failed for %s",
                        sub_def.code,
                        exc_info=True,
                    )
                    # Return a zero-score fallback match.
                    results.append(SubSectionMatch(
                        sub_section_code=sub_def.code,
                        parent_section_code=sub_def.parent_code,
                        sub_section_name=sub_def.name,
                        embedding_score=round(parent_score, 4),
                        keyword_score=0.0,
                        summarization_score=0.0,
                        composite_score=0.0,
                        confidence=MatchConfidence.LOW,
                        match_method="parallel_error",
                    ))
        return results

    # ------------------------------------------------------------------
    # Sub-section scoring
    # ------------------------------------------------------------------

    def _score_subsection(
        self,
        content: str,
        protocol_title: str,
        parent_score: float,
        sub_def: SubSectionDef,
    ) -> SubSectionMatch:
        """Score content against one sub-section definition."""
        kw_score = self._keyword_score(content, sub_def)

        if self._use_llm and content.strip():
            summ_result = self._llm_score(
                content, protocol_title, sub_def
            )
            summ_score = summ_result.score
            method = "embedding+summarization"
        else:
            summ_score = -1.0
            method = "keyword_fallback"

        composite = compute_composite_score(
            embedding_score=parent_score,
            keyword_score=kw_score,
            summarization_score=summ_score,
            weights=self._weights,
        )
        confidence = self._classify(composite)

        return SubSectionMatch(
            sub_section_code=sub_def.code,
            parent_section_code=sub_def.parent_code,
            sub_section_name=sub_def.name,
            embedding_score=round(parent_score, 4),
            keyword_score=round(kw_score, 4),
            summarization_score=round(summ_score, 4)
            if summ_score >= 0
            else -1.0,
            composite_score=composite,
            confidence=confidence,
            match_method=method,
        )

    @staticmethod
    def _keyword_score(
        content: str, sub_def: SubSectionDef
    ) -> float:
        """Keyword overlap between content and sub-section description.

        Counts the fraction of significant words from the sub-section
        description that appear in the content.
        """
        if not content.strip():
            return 0.0

        stop = {
            "the", "and", "for", "are", "what", "that", "this",
            "with", "from", "have", "has", "will", "been", "should",
            "there", "their", "which", "each", "any", "all", "how",
            "does", "being", "used", "who", "whom",
        }
        desc_words = set(
            w.lower()
            for w in re.findall(r"[a-z]{3,}", sub_def.description.lower())
        ) - stop

        if not desc_words:
            return 0.0

        content_lower = content.lower()
        hits = sum(1 for w in desc_words if w in content_lower)
        return round(hits / len(desc_words), 4)

    def _llm_score(
        self,
        content: str,
        protocol_title: str,
        sub_def: SubSectionDef,
    ) -> SummarizationResult:
        """Call LLM to assess semantic fit, with caching."""
        excerpt = content[: self._max_chars]
        cache_key = hashlib.sha256(
            f"{excerpt}|{sub_def.code}".encode()
        ).hexdigest()

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1
        result = self._call_llm(excerpt, protocol_title, sub_def)
        self._cache[cache_key] = result
        return result

    def _call_llm(
        self,
        excerpt: str,
        protocol_title: str,
        sub_def: SubSectionDef,
    ) -> SummarizationResult:
        """Make the actual Anthropic API call."""
        self._llm_calls += 1

        prompt = _SUMMARIZATION_PROMPT.format(
            protocol_title=protocol_title,
            max_chars=self._max_chars,
            content_excerpt=excerpt,
            sub_code=sub_def.code,
            sub_name=sub_def.name,
            sub_description=sub_def.description,
        )

        _SYSTEM = (
            "You are a regulatory document analysis tool for "
            "ICH E6(R3) GCP compliance. You classify sections "
            "of clinical trial protocol documents against the "
            "ICH Appendix B taxonomy. This is a read-only "
            "document classification task on publicly available "
            "clinical trial protocols from ClinicalTrials.gov. "
            "No real patients are involved."
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=200,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            # PTCV-188: Detect refusal and retry with fallback.
            stop = getattr(response, "stop_reason", None)
            is_refused = (
                isinstance(stop, str)
                and stop in ("content_filtered", "refusal")
            ) or not response.content

            if is_refused:
                logger.info(
                    "PTCV-188: Refusal for %s "
                    "(stop_reason=%s), retrying with %s",
                    sub_def.code,
                    stop,
                    self._fallback_model,
                )
                response = self._client.messages.create(
                    model=self._fallback_model,
                    max_tokens=200,
                    system=_SYSTEM,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                if not response.content:
                    self._refusal_count += 1
                    raise ValueError(
                        f"Fallback also empty for {sub_def.code}"
                    )
                self._fallback_count += 1

            text = response.content[0].text.strip()
            parsed = json.loads(text)
            score = float(parsed.get("score", 0.0))
            score = min(1.0, max(0.0, score))
            rationale = str(parsed.get("rationale", ""))

            return SummarizationResult(
                sub_section_code=sub_def.code,
                score=round(score, 4),
                rationale=rationale,
            )
        except Exception:
            logger.warning(
                "LLM call failed for sub-section %s, returning 0.0",
                sub_def.code,
                exc_info=True,
            )
            return SummarizationResult(
                sub_section_code=sub_def.code,
                score=0.0,
                rationale="LLM call failed",
            )

    def _classify(self, composite_score: float) -> MatchConfidence:
        """Map composite score to confidence tier."""
        if composite_score >= self._auto_threshold:
            return MatchConfidence.HIGH
        if composite_score >= self._review_threshold:
            return MatchConfidence.REVIEW
        return MatchConfidence.LOW
