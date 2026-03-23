"""Hybrid embedding-first section classifier (PTCV-279).

Stage 2 section classification with embeddings as the primary signal
and keywords demoted to search-space bounding:

Phase 1 — **Search space bounding**: ``SectionMatcher.match()`` provides
    a cheap keyword baseline for all headers (TOC + synonym + heading
    boost). This bounds candidate spans but does NOT make final
    classification decisions for sections with embedding coverage.

Phase 2 — **Embedding classification** (primary where available):
    FAISS ``CentroidClassifier`` scores content against section
    centroids. For sections with centroids, this OVERRIDES the
    keyword score. Confidence cascade: HIGH → accept, REVIEW → RAG
    boost → Sonnet fallback.

Phase 3 — **Sub-section enrichment**: ``SummarizationMatcher.refine()``
    scores sub-sections with LLM semantic fit. Sub-section scores
    propagate upward via bottom-up confidence boosting.

Sections without centroids fall back to keyword + Sonnet (Phase 2
fallback path), which is gradually replaced as the alignment corpus
expands (PTCV-281).

Risk tier: LOW — classification logic (no patient data).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from .section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMapping,
    SectionMatch,
    SectionMatcher,
)
from .summarization_matcher import (
    EnrichedMatchResult,
    SummarizationMatcher,
)
from .toc_extractor import ProtocolIndex

logger = logging.getLogger(__name__)


class SectionClassifier:
    """Hybrid embedding-first section classifier.

    Uses FAISS centroid embeddings as the primary classification signal
    for sections with centroids, keyword scoring as fallback for
    sections without, and LLM sub-section scoring via
    ``SummarizationMatcher``.

    Args:
        auto_threshold: Minimum score for HIGH confidence (default 0.75).
        review_threshold: Minimum score for REVIEW tier (default 0.50).
        synonym_boost: Additive boost for known synonyms (default 0.15).
        rag_index: Optional RAG exemplar index for disambiguation.
        rag_boost: Additive boost from RAG tiebreaker (default 0.06).
        anthropic_api_key: API key for Sonnet LLM sub-section scoring.
            If *None*, uses ``ANTHROPIC_API_KEY`` env var. If unavailable,
            LLM scoring is skipped (graceful fallback).
    """

    def __init__(
        self,
        auto_threshold: float = 0.75,
        review_threshold: float = 0.50,
        synonym_boost: float = 0.15,
        rag_index: Optional[Any] = None,
        rag_boost: float = 0.06,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        self._auto_threshold = auto_threshold
        self._review_threshold = review_threshold

        # Phase 1: Keyword baseline (search space bounding + fallback)
        self._matcher = SectionMatcher(
            auto_threshold=auto_threshold,
            review_threshold=review_threshold,
            synonym_boost=synonym_boost,
            rag_index=rag_index,
            rag_boost=rag_boost,
        )

        # Phase 2: Embedding classification (primary where available)
        self._centroid_classifier: Any = None
        self._centroid_codes: set[str] = set()
        self._rag_index = rag_index
        self._rag_boost = rag_boost
        self._load_centroid_classifier()

        # Phase 3: LLM sub-section enrichment (restored from
        # SummarizationMatcher — PTCV-279 fix for PTCV-255 regression)
        api_key = anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY", "",
        )
        self._summarizer = SummarizationMatcher(
            anthropic_api_key=api_key or None,
        )

    def _load_centroid_classifier(self) -> None:
        """Try to load FAISS centroid classifier."""
        try:
            from .centroid_classifier import load_centroid_classifier

            clf = load_centroid_classifier()
            if clf is not None and clf.section_count > 0:
                self._centroid_classifier = clf
                self._centroid_codes = set(clf.section_codes)
                logger.info(
                    "SectionClassifier: FAISS centroids loaded for "
                    "%d sections: %s",
                    clf.section_count,
                    ", ".join(sorted(clf.section_codes)),
                )
            else:
                logger.info(
                    "SectionClassifier: no FAISS centroids available "
                    "— using keyword fallback for all sections",
                )
        except Exception as exc:
            logger.debug(
                "SectionClassifier: centroid classifier unavailable "
                "(%s) — using keyword fallback",
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        protocol_index: ProtocolIndex,
        progress_callback: Any = None,
    ) -> tuple[MatchResult, EnrichedMatchResult]:
        """Classify all protocol headers using hybrid approach.

        1. Keyword baseline for all headers (bounding + fallback)
        2. FAISS embedding override for sections with centroids
        3. LLM sub-section enrichment via SummarizationMatcher
        4. Bottom-up confidence propagation

        Args:
            protocol_index: Navigable document index from Stage 1.
            progress_callback: Optional ``(done, total)`` callback.

        Returns:
            Tuple of (MatchResult, EnrichedMatchResult).
        """
        t0 = time.monotonic()

        # Phase 1: Keyword baseline (all headers)
        base_result = self._matcher.match(protocol_index)

        # Phase 2: Embedding override (sections with centroids)
        upgraded_result = self._apply_embedding_overrides(
            base_result, protocol_index,
        )

        # Phase 3: LLM sub-section enrichment
        # Restores SummarizationMatcher.refine() that was incorrectly
        # removed in PTCV-255.
        def _on_progress(done: int, total: int) -> None:
            if progress_callback is not None and total > 0:
                progress_callback(done, total)

        enriched_result = self._summarizer.refine(
            upgraded_result,
            protocol_index,
            progress_callback=_on_progress,
        )

        # Phase 4: Bottom-up propagation — upgrade parents from
        # strong sub-section matches in the enriched result.
        match_result, enriched_result = self._propagate_from_enriched(
            upgraded_result, enriched_result,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "SectionClassifier.classify completed in %.2fs: "
            "%d headers, %d HIGH, %d REVIEW, %d LOW, "
            "embedding overrides=%d, llm_calls=%d",
            elapsed,
            len(match_result.mappings),
            match_result.auto_mapped_count,
            match_result.review_count,
            match_result.unmapped_count,
            self._embedding_override_count,
            enriched_result.llm_calls_made,
        )

        return match_result, enriched_result

    # ------------------------------------------------------------------
    # Phase 2: Embedding overrides
    # ------------------------------------------------------------------

    _embedding_override_count: int = 0

    def _apply_embedding_overrides(
        self,
        base_result: MatchResult,
        protocol_index: ProtocolIndex,
    ) -> MatchResult:
        """Override keyword scores with FAISS embedding scores.

        For each mapping whose top ICH code has a centroid, replace
        the keyword-based confidence with the centroid cosine similarity.
        Sections without centroids keep their keyword scores.

        Also applies RAG exemplar boost for REVIEW-tier embeddings.
        """
        if self._centroid_classifier is None:
            return base_result

        self._embedding_override_count = 0
        new_mappings: list[SectionMapping] = []

        for mapping in base_result.mappings:
            upgraded = self._try_embedding_override(
                mapping, protocol_index,
            )
            new_mappings.append(upgraded)

        # Recount tiers
        auto = sum(1 for m in new_mappings if m.auto_mapped)
        review = sum(
            1 for m in new_mappings
            if not m.auto_mapped
            and m.matches
            and m.matches[0].confidence == MatchConfidence.REVIEW
        )
        low = len(new_mappings) - auto - review

        return MatchResult(
            mappings=new_mappings,
            auto_mapped_count=auto,
            review_count=review,
            unmapped_count=low,
            auto_map_rate=round(
                auto / len(new_mappings) if new_mappings else 0.0,
                4,
            ),
        )

    def _try_embedding_override(
        self,
        mapping: SectionMapping,
        protocol_index: ProtocolIndex,
    ) -> SectionMapping:
        """Try to override a single mapping with embedding confidence."""
        if not mapping.matches:
            return mapping

        top_match = mapping.matches[0]
        top_code = top_match.ich_section_code

        # Only override if we have a centroid for the top candidate
        if top_code not in self._centroid_codes:
            return mapping

        # Get content for this header
        content = protocol_index.content_spans.get(
            mapping.protocol_section_number, "",
        )
        if not content or len(content.strip()) < 20:
            return mapping  # Too little content for embedding

        # Classify via FAISS centroid
        matches = self._centroid_classifier.classify(content, top_k=3)
        if not matches:
            return mapping

        best = matches[0]
        embedding_confidence = best.confidence

        # RAG boost for REVIEW-tier embedding results
        if (
            self._review_threshold
            <= embedding_confidence
            < self._auto_threshold
            and self._rag_index is not None
        ):
            embedding_confidence = self._apply_rag_boost(
                content, best.section_code, embedding_confidence,
            )

        # Determine confidence tier
        if embedding_confidence >= self._auto_threshold:
            confidence = MatchConfidence.HIGH
            auto_mapped = True
        elif embedding_confidence >= self._review_threshold:
            confidence = MatchConfidence.REVIEW
            auto_mapped = False
        else:
            confidence = MatchConfidence.LOW
            auto_mapped = False

        self._embedding_override_count += 1

        new_match = SectionMatch(
            ich_section_code=best.section_code,
            ich_section_name=best.section_name,
            similarity_score=round(embedding_confidence, 4),
            boosted_score=round(embedding_confidence, 4),
            confidence=confidence,
            match_method="centroid_embedding",
        )

        logger.debug(
            "Embedding override: %s '%s' — %s (kw=%.3f → emb=%.3f %s)",
            mapping.protocol_section_number,
            mapping.protocol_section_title[:40],
            best.section_code,
            top_match.boosted_score,
            embedding_confidence,
            confidence.value,
        )

        # Keep remaining keyword matches as fallback candidates
        remaining = [
            m for m in mapping.matches
            if m.ich_section_code != best.section_code
        ]

        return SectionMapping(
            protocol_section_number=mapping.protocol_section_number,
            protocol_section_title=mapping.protocol_section_title,
            matches=[new_match] + remaining,
            auto_mapped=auto_mapped,
        )

    def _apply_rag_boost(
        self,
        content: str,
        section_code: str,
        current_confidence: float,
    ) -> float:
        """Boost REVIEW-tier embedding confidence via RAG exemplars."""
        try:
            exemplars = self._rag_index.query(content[:500], top_k=3)
        except Exception:
            return current_confidence

        if not exemplars:
            return current_confidence

        # Count votes for the section code
        votes_for = sum(
            1 for ex in exemplars
            if ex.section_code == section_code
        )
        if votes_for >= 2:
            boosted = min(1.0, current_confidence + self._rag_boost)
            logger.debug(
                "RAG boost: %s — %d/%d exemplars agree "
                "(%.3f → %.3f)",
                section_code, votes_for, len(exemplars),
                current_confidence, boosted,
            )
            return boosted

        return current_confidence

    # ------------------------------------------------------------------
    # Phase 4: Bottom-up propagation from enriched sub-sections
    # ------------------------------------------------------------------

    def _propagate_from_enriched(
        self,
        match_result: MatchResult,
        enriched_result: EnrichedMatchResult,
    ) -> tuple[MatchResult, EnrichedMatchResult]:
        """Upgrade parent confidence from strong LLM sub-section scores.

        If ``SummarizationMatcher`` found HIGH-confidence sub-sections,
        propagate those scores upward to upgrade REVIEW parents to HIGH.
        """
        new_mappings: list[SectionMapping] = []
        changed = False

        for mapping, enriched in zip(
            match_result.mappings,
            enriched_result.enriched_mappings,
        ):
            if not mapping.matches:
                new_mappings.append(mapping)
                continue

            top = mapping.matches[0]
            if top.confidence == MatchConfidence.HIGH:
                new_mappings.append(mapping)
                continue

            # Check if any sub-section scored HIGH
            best_sub = None
            for sm in enriched.sub_section_matches:
                if sm.confidence == MatchConfidence.HIGH:
                    if (
                        best_sub is None
                        or sm.composite_score > best_sub.composite_score
                    ):
                        best_sub = sm

            if best_sub is None:
                new_mappings.append(mapping)
                continue

            # Bottom-up upgrade
            propagated_score = max(
                top.boosted_score, best_sub.composite_score,
            )
            upgraded_match = SectionMatch(
                ich_section_code=top.ich_section_code,
                ich_section_name=top.ich_section_name,
                similarity_score=top.similarity_score,
                boosted_score=round(propagated_score, 4),
                confidence=MatchConfidence.HIGH,
                match_method=top.match_method,
                sub_section_code=best_sub.sub_section_code,
            )
            new_matches = [upgraded_match] + list(mapping.matches[1:])
            new_mappings.append(SectionMapping(
                protocol_section_number=mapping.protocol_section_number,
                protocol_section_title=mapping.protocol_section_title,
                matches=new_matches,
                auto_mapped=True,
            ))
            changed = True

            logger.debug(
                "Bottom-up upgrade: %s — sub-section %s (%.3f) "
                "lifts parent from %s to HIGH",
                mapping.protocol_section_number,
                best_sub.sub_section_code,
                best_sub.composite_score,
                top.confidence.value,
            )

        if not changed:
            return match_result, enriched_result

        # Recount tiers
        auto = sum(1 for m in new_mappings if m.auto_mapped)
        review = sum(
            1 for m in new_mappings
            if not m.auto_mapped
            and m.matches
            and m.matches[0].confidence == MatchConfidence.REVIEW
        )
        low = len(new_mappings) - auto - review

        new_match = MatchResult(
            mappings=new_mappings,
            auto_mapped_count=auto,
            review_count=review,
            unmapped_count=low,
            auto_map_rate=round(
                auto / len(new_mappings) if new_mappings else 0.0,
                4,
            ),
        )

        # Update enriched result with new mappings
        from dataclasses import replace
        new_enriched = replace(
            enriched_result,
            mappings=new_mappings,
            auto_mapped_count=auto,
            review_count=review,
            unmapped_count=low,
            auto_map_rate=new_match.auto_map_rate,
        )

        return new_match, new_enriched
