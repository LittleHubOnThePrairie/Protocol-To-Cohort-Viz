"""Tests for hybrid embedding-first SectionClassifier (PTCV-279).

Feature: Hybrid embedding-first section classification

  Scenario: FAISS centroids classify where available
  Scenario: Keyword fallback for sections without centroids
  Scenario: HIGH embedding confidence skips Sonnet
  Scenario: REVIEW confidence triggers RAG then Sonnet cascade
  Scenario: Sonnet handles sub-section classification
  Scenario: No regression on protocols without FAISS index
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMapping,
    SectionMatch,
)
from ptcv.ich_parser.summarization_matcher import (
    EnrichedMatchResult,
    EnrichedSectionMapping,
    SubSectionMatch,
)
from ptcv.ich_parser.section_classifier import SectionClassifier


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_protocol_index(
    headers: list[tuple[str, str]],
    content_spans: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock ProtocolIndex."""
    toc_entries = []
    for number, title in headers:
        entry = MagicMock()
        entry.number = number
        entry.title = title
        entry.level = number.count(".") + 1
        toc_entries.append(entry)

    idx = MagicMock()
    idx.toc_entries = toc_entries
    idx.section_headers = []
    idx.content_spans = content_spans or {}
    idx.toc_found = True
    return idx


def _make_mapping(
    number: str,
    title: str,
    code: str,
    score: float,
    confidence: MatchConfidence = MatchConfidence.HIGH,
) -> SectionMapping:
    return SectionMapping(
        protocol_section_number=number,
        protocol_section_title=title,
        matches=[SectionMatch(
            ich_section_code=code,
            ich_section_name=title,
            similarity_score=score,
            boosted_score=score,
            confidence=confidence,
            match_method="keyword_fallback",
        )],
        auto_mapped=confidence == MatchConfidence.HIGH,
    )


def _make_match_result(
    mappings: list[SectionMapping],
) -> MatchResult:
    auto = sum(1 for m in mappings if m.auto_mapped)
    review = sum(
        1 for m in mappings
        if not m.auto_mapped
        and m.matches
        and m.matches[0].confidence == MatchConfidence.REVIEW
    )
    return MatchResult(
        mappings=mappings,
        auto_mapped_count=auto,
        review_count=review,
        unmapped_count=len(mappings) - auto - review,
        auto_map_rate=auto / len(mappings) if mappings else 0.0,
    )


def _make_enriched_result(
    match_result: MatchResult,
    sub_matches_per_mapping: list[list[SubSectionMatch]] | None = None,
) -> EnrichedMatchResult:
    enriched_mappings = []
    for i, m in enumerate(match_result.mappings):
        subs = (
            sub_matches_per_mapping[i]
            if sub_matches_per_mapping and i < len(sub_matches_per_mapping)
            else []
        )
        enriched_mappings.append(EnrichedSectionMapping(
            protocol_section_number=m.protocol_section_number,
            protocol_section_title=m.protocol_section_title,
            matches=m.matches,
            auto_mapped=m.auto_mapped,
            sub_section_matches=subs,
            parent_coverage=frozenset(),
        ))
    return EnrichedMatchResult(
        mappings=match_result.mappings,
        auto_mapped_count=match_result.auto_mapped_count,
        review_count=match_result.review_count,
        unmapped_count=match_result.unmapped_count,
        auto_map_rate=match_result.auto_map_rate,
        enriched_mappings=enriched_mappings,
        sub_section_auto_map_rate=0.0,
        llm_calls_made=3,
        llm_fallback=False,
        elapsed_seconds=0.5,
    )


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestEmbeddingOverride:
    """Scenario: FAISS centroids classify where available."""

    def test_centroid_overrides_keyword_score(self) -> None:
        """Given a content span matching B.5 (has centroid)
        When the embedding classifier runs
        Then FAISS cosine similarity is the primary signal."""
        # Keyword gives B.5 a REVIEW score of 0.60
        mapping = _make_mapping(
            "5", "Study Population", "B.5", 0.60,
            MatchConfidence.REVIEW,
        )
        base_result = _make_match_result([mapping])
        enriched = _make_enriched_result(base_result)

        protocol_index = _make_protocol_index(
            [("5", "Study Population")],
            content_spans={"5": "Inclusion criteria for eligibility."},
        )

        # Mock centroid classifier returning HIGH confidence
        mock_centroid = MagicMock()
        mock_centroid.section_count = 6
        mock_centroid.section_codes = ["B.5"]
        mock_centroid.classify.return_value = [
            MagicMock(
                section_code="B.5",
                section_name="Selection of Subjects",
                confidence=0.88,
            ),
        ]

        with patch.object(
            SectionClassifier, "__init__", lambda self, **kw: None,
        ):
            clf = SectionClassifier.__new__(SectionClassifier)
            clf._matcher = MagicMock()
            clf._matcher.match.return_value = base_result
            clf._centroid_classifier = mock_centroid
            clf._centroid_codes = {"B.5"}
            clf._rag_index = None
            clf._rag_boost = 0.06
            clf._auto_threshold = 0.75
            clf._review_threshold = 0.50
            clf._embedding_override_count = 0
            clf._summarizer = MagicMock()
            clf._summarizer.refine.return_value = enriched

        match_result, _ = clf.classify(protocol_index)

        top = match_result.mappings[0].matches[0]
        assert top.match_method == "centroid_embedding"
        assert top.boosted_score >= 0.85
        assert top.confidence == MatchConfidence.HIGH


class TestKeywordFallback:
    """Scenario: Keyword fallback for sections without centroids."""

    def test_sections_without_centroids_use_keywords(self) -> None:
        """Given a content span matching B.7 (no centroid)
        When the classifier runs
        Then SectionMatcher keyword scoring is used."""
        mapping = _make_mapping(
            "7", "Treatment", "B.7", 0.70,
            MatchConfidence.REVIEW,
        )
        base_result = _make_match_result([mapping])
        enriched = _make_enriched_result(base_result)

        protocol_index = _make_protocol_index(
            [("7", "Treatment")],
            content_spans={"7": "Study drug dosing."},
        )

        # Centroid exists but NOT for B.7
        mock_centroid = MagicMock()
        mock_centroid.section_count = 6
        mock_centroid.section_codes = ["B.5"]

        with patch.object(
            SectionClassifier, "__init__", lambda self, **kw: None,
        ):
            clf = SectionClassifier.__new__(SectionClassifier)
            clf._matcher = MagicMock()
            clf._matcher.match.return_value = base_result
            clf._centroid_classifier = mock_centroid
            clf._centroid_codes = {"B.5"}
            clf._rag_index = None
            clf._rag_boost = 0.06
            clf._auto_threshold = 0.75
            clf._review_threshold = 0.50
            clf._embedding_override_count = 0
            clf._summarizer = MagicMock()
            clf._summarizer.refine.return_value = enriched

        match_result, _ = clf.classify(protocol_index)

        top = match_result.mappings[0].matches[0]
        # Should keep original keyword method, NOT centroid_embedding
        assert top.match_method == "keyword_fallback"
        assert top.boosted_score == 0.70


class TestHighEmbeddingSkipsSonnet:
    """Scenario: HIGH embedding confidence skips Sonnet."""

    def test_high_centroid_accepted_directly(self) -> None:
        mapping = _make_mapping(
            "5", "Eligibility", "B.5", 0.55,
            MatchConfidence.REVIEW,
        )
        base_result = _make_match_result([mapping])
        enriched = _make_enriched_result(base_result)
        protocol_index = _make_protocol_index(
            [("5", "Eligibility")],
            content_spans={"5": "Inclusion criteria text."},
        )

        mock_centroid = MagicMock()
        mock_centroid.section_count = 1
        mock_centroid.section_codes = ["B.5"]
        mock_centroid.classify.return_value = [
            MagicMock(
                section_code="B.5",
                section_name="Selection of Subjects",
                confidence=0.92,
            ),
        ]

        with patch.object(
            SectionClassifier, "__init__", lambda self, **kw: None,
        ):
            clf = SectionClassifier.__new__(SectionClassifier)
            clf._matcher = MagicMock()
            clf._matcher.match.return_value = base_result
            clf._centroid_classifier = mock_centroid
            clf._centroid_codes = {"B.5"}
            clf._rag_index = None
            clf._rag_boost = 0.06
            clf._auto_threshold = 0.75
            clf._review_threshold = 0.50
            clf._embedding_override_count = 0
            clf._summarizer = MagicMock()
            clf._summarizer.refine.return_value = enriched

        match_result, _ = clf.classify(protocol_index)

        # HIGH — accepted directly, no Sonnet needed
        assert match_result.mappings[0].auto_mapped is True
        assert match_result.mappings[0].matches[0].confidence == (
            MatchConfidence.HIGH
        )


class TestReviewTriggersCascade:
    """Scenario: REVIEW confidence triggers RAG then Sonnet cascade."""

    def test_rag_boost_applied_to_review(self) -> None:
        mapping = _make_mapping(
            "3", "Objectives", "B.3", 0.55,
            MatchConfidence.REVIEW,
        )
        base_result = _make_match_result([mapping])
        enriched = _make_enriched_result(base_result)
        protocol_index = _make_protocol_index(
            [("3", "Objectives")],
            content_spans={"3": "Primary endpoint is overall survival."},
        )

        # Centroid returns REVIEW confidence
        mock_centroid = MagicMock()
        mock_centroid.section_count = 1
        mock_centroid.section_codes = ["B.3"]
        mock_centroid.classify.return_value = [
            MagicMock(
                section_code="B.3",
                section_name="Objectives",
                confidence=0.68,
            ),
        ]

        # RAG returns 2 votes for B.3 → should boost
        mock_rag = MagicMock()
        mock_rag.query.return_value = [
            MagicMock(section_code="B.3"),
            MagicMock(section_code="B.3"),
            MagicMock(section_code="B.4"),
        ]

        with patch.object(
            SectionClassifier, "__init__", lambda self, **kw: None,
        ):
            clf = SectionClassifier.__new__(SectionClassifier)
            clf._matcher = MagicMock()
            clf._matcher.match.return_value = base_result
            clf._centroid_classifier = mock_centroid
            clf._centroid_codes = {"B.3"}
            clf._rag_index = mock_rag
            clf._rag_boost = 0.08
            clf._auto_threshold = 0.75
            clf._review_threshold = 0.50
            clf._embedding_override_count = 0
            clf._summarizer = MagicMock()
            clf._summarizer.refine.return_value = enriched

        match_result, _ = clf.classify(protocol_index)

        top = match_result.mappings[0].matches[0]
        # 0.68 + 0.08 RAG boost = 0.76 → HIGH
        assert top.boosted_score >= 0.75
        assert top.confidence == MatchConfidence.HIGH


class TestLlmSubSectionScoring:
    """Scenario: Sonnet handles sub-section classification."""

    def test_summarizer_refine_called(self) -> None:
        """SummarizationMatcher.refine() is called for LLM sub-section
        scoring (restored from PTCV-255 regression)."""
        mapping = _make_mapping(
            "5", "Population", "B.5", 0.85,
            MatchConfidence.HIGH,
        )
        base_result = _make_match_result([mapping])
        enriched = _make_enriched_result(base_result)
        enriched_result_with_llm = dataclasses.replace(
            enriched, llm_calls_made=5, llm_fallback=False,
        )
        protocol_index = _make_protocol_index(
            [("5", "Population")],
        )

        with patch.object(
            SectionClassifier, "__init__", lambda self, **kw: None,
        ):
            clf = SectionClassifier.__new__(SectionClassifier)
            clf._matcher = MagicMock()
            clf._matcher.match.return_value = base_result
            clf._centroid_classifier = None
            clf._centroid_codes = set()
            clf._rag_index = None
            clf._rag_boost = 0.06
            clf._auto_threshold = 0.75
            clf._review_threshold = 0.50
            clf._embedding_override_count = 0
            clf._summarizer = MagicMock()
            clf._summarizer.refine.return_value = enriched_result_with_llm

        _, enriched_out = clf.classify(protocol_index)

        # SummarizationMatcher.refine() was called
        clf._summarizer.refine.assert_called_once()
        # LLM calls reported (not hardcoded to 0)
        assert enriched_out.llm_calls_made == 5
        assert enriched_out.llm_fallback is False


class TestNoFaissRegression:
    """Scenario: No regression on protocols without FAISS index."""

    def test_no_centroids_uses_keyword_only(self) -> None:
        mapping = _make_mapping(
            "1", "Summary", "B.1", 0.80,
            MatchConfidence.HIGH,
        )
        base_result = _make_match_result([mapping])
        enriched = _make_enriched_result(base_result)
        protocol_index = _make_protocol_index(
            [("1", "Summary")],
        )

        with patch.object(
            SectionClassifier, "__init__", lambda self, **kw: None,
        ):
            clf = SectionClassifier.__new__(SectionClassifier)
            clf._matcher = MagicMock()
            clf._matcher.match.return_value = base_result
            clf._centroid_classifier = None  # No FAISS
            clf._centroid_codes = set()
            clf._rag_index = None
            clf._rag_boost = 0.06
            clf._auto_threshold = 0.75
            clf._review_threshold = 0.50
            clf._embedding_override_count = 0
            clf._summarizer = MagicMock()
            clf._summarizer.refine.return_value = enriched

        match_result, _ = clf.classify(protocol_index)

        # Should work fine with keyword scores preserved
        assert match_result.auto_mapped_count == 1
        top = match_result.mappings[0].matches[0]
        assert top.match_method == "keyword_fallback"


class TestBottomUpFromEnriched:
    """Bottom-up propagation from LLM sub-section scores."""

    def test_high_subsection_upgrades_review_parent(self) -> None:
        mapping = _make_mapping(
            "5", "Population", "B.5", 0.65,
            MatchConfidence.REVIEW,
        )
        base_result = _make_match_result([mapping])

        # SummarizationMatcher found a HIGH sub-section
        high_sub = SubSectionMatch(
            sub_section_code="B.5.1",
            parent_section_code="B.5",
            sub_section_name="Inclusion Criteria",
            embedding_score=0.65,
            keyword_score=0.50,
            summarization_score=0.85,
            composite_score=0.80,
            confidence=MatchConfidence.HIGH,
            match_method="embedding+summarization",
        )
        enriched = _make_enriched_result(
            base_result,
            sub_matches_per_mapping=[[high_sub]],
        )

        protocol_index = _make_protocol_index(
            [("5", "Population")],
        )

        with patch.object(
            SectionClassifier, "__init__", lambda self, **kw: None,
        ):
            clf = SectionClassifier.__new__(SectionClassifier)
            clf._matcher = MagicMock()
            clf._matcher.match.return_value = base_result
            clf._centroid_classifier = None
            clf._centroid_codes = set()
            clf._rag_index = None
            clf._rag_boost = 0.06
            clf._auto_threshold = 0.75
            clf._review_threshold = 0.50
            clf._embedding_override_count = 0
            clf._summarizer = MagicMock()
            clf._summarizer.refine.return_value = enriched

        match_result, _ = clf.classify(protocol_index)

        # Parent should be upgraded from REVIEW to HIGH
        assert match_result.mappings[0].auto_mapped is True
        top = match_result.mappings[0].matches[0]
        assert top.confidence == MatchConfidence.HIGH
        assert top.sub_section_code == "B.5.1"
        assert top.boosted_score >= 0.80
