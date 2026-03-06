"""Tests for sub-section matching with LLM summarization (PTCV-96).

Covers: sub-section registry, data models, composite scoring,
SectionMatch new fields, EnrichedMatchResult, fallback mode,
mocked LLM scoring, and caching.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.query_schema import AppendixBQuery
from ptcv.ich_parser.section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMatch,
    SectionMapping,
)
from ptcv.ich_parser.summarization_matcher import (
    EnrichedMatchResult,
    EnrichedSectionMapping,
    SubSectionDef,
    SubSectionMatch,
    SummarizationMatcher,
    SummarizationResult,
    _derive_name,
    _reset_registry_cache,
    build_subsection_registry,
    compute_composite_score,
    get_subsections_for_parent,
)
from ptcv.ich_parser.toc_extractor import ProtocolIndex, TOCEntry


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_registry_cache() -> None:
    """Reset the module-level registry cache before each test."""
    _reset_registry_cache()


@pytest.fixture()
def sample_queries() -> list[AppendixBQuery]:
    """Minimal set of queries spanning 2 parents with sub-sections."""
    return [
        AppendixBQuery(
            query_id="B.5.1_Q1",
            section_id="B.5.1",
            parent_section="B.5",
            schema_section="B.5",
            query_text="What are the inclusion criteria?",
            expected_type="list",
            required=True,
        ),
        AppendixBQuery(
            query_id="B.5.1_Q2",
            section_id="B.5.1",
            parent_section="B.5",
            schema_section="B.5",
            query_text="What are the age requirements?",
            expected_type="text",
            required=False,
        ),
        AppendixBQuery(
            query_id="B.5.2_Q1",
            section_id="B.5.2",
            parent_section="B.5",
            schema_section="B.5",
            query_text="What are the exclusion criteria?",
            expected_type="list",
            required=True,
        ),
        AppendixBQuery(
            query_id="B.1.1_Q1",
            section_id="B.1.1",
            parent_section="B.1",
            schema_section="B.1",
            query_text="What is the protocol title?",
            expected_type="text",
            required=True,
        ),
    ]


@pytest.fixture()
def sample_protocol_index() -> ProtocolIndex:
    """Synthetic ProtocolIndex with content spans for matching."""
    return ProtocolIndex(
        source_path="test.pdf",
        page_count=20,
        toc_entries=[
            TOCEntry(level=1, number="5", title="Subject Selection"),
            TOCEntry(level=2, number="5.1", title="Inclusion Criteria"),
        ],
        section_headers=[],
        content_spans={
            "5": (
                "Subjects must meet all inclusion criteria and none "
                "of the exclusion criteria to be eligible. Age "
                "requirements specify adults aged 18 or older."
            ),
        },
        full_text="",
        toc_found=True,
        toc_pages=[1],
    )


def _make_match_result(
    parent_code: str = "B.5",
    parent_name: str = "Study Population",
    score: float = 0.85,
    confidence: MatchConfidence = MatchConfidence.HIGH,
) -> MatchResult:
    """Build a minimal MatchResult with one mapping."""
    match = SectionMatch(
        ich_section_code=parent_code,
        ich_section_name=parent_name,
        similarity_score=score,
        boosted_score=score,
        confidence=confidence,
        match_method="keyword_fallback",
    )
    mapping = SectionMapping(
        protocol_section_number="5",
        protocol_section_title="Subject Selection",
        matches=[match],
        auto_mapped=confidence == MatchConfidence.HIGH,
    )
    return MatchResult(
        mappings=[mapping],
        auto_mapped_count=1 if confidence == MatchConfidence.HIGH else 0,
        review_count=0 if confidence == MatchConfidence.HIGH else 1,
        unmapped_count=0,
        auto_map_rate=1.0 if confidence == MatchConfidence.HIGH else 0.0,
    )


# -------------------------------------------------------------------
# TestSubSectionRegistry
# -------------------------------------------------------------------


class TestSubSectionRegistry:
    """Sub-section registry from query schema."""

    def test_builds_from_queries(
        self, sample_queries: list[AppendixBQuery]
    ) -> None:
        """Registry contains all unique section_ids."""
        reg = build_subsection_registry(sample_queries)
        assert set(reg.keys()) == {"B.5.1", "B.5.2", "B.1.1"}

    def test_parent_code_correct(
        self, sample_queries: list[AppendixBQuery]
    ) -> None:
        """Each sub-section points to correct parent."""
        reg = build_subsection_registry(sample_queries)
        assert reg["B.5.1"].parent_code == "B.5"
        assert reg["B.5.2"].parent_code == "B.5"
        assert reg["B.1.1"].parent_code == "B.1"

    def test_descriptions_non_empty(
        self, sample_queries: list[AppendixBQuery]
    ) -> None:
        """Each sub-section has a non-empty description."""
        reg = build_subsection_registry(sample_queries)
        for sub in reg.values():
            assert sub.description.strip()

    def test_query_ids_tracked(
        self, sample_queries: list[AppendixBQuery]
    ) -> None:
        """Query IDs are collected per sub-section."""
        reg = build_subsection_registry(sample_queries)
        assert reg["B.5.1"].query_ids == ("B.5.1_Q1", "B.5.1_Q2")
        assert reg["B.5.2"].query_ids == ("B.5.2_Q1",)

    def test_get_subsections_for_parent(
        self, sample_queries: list[AppendixBQuery]
    ) -> None:
        """Retrieving sub-sections for a parent returns sorted list."""
        reg = build_subsection_registry(sample_queries)
        subs = get_subsections_for_parent("B.5", registry=reg)
        assert len(subs) == 2
        assert subs[0].code == "B.5.1"
        assert subs[1].code == "B.5.2"

    def test_get_subsections_for_parent_no_match(
        self, sample_queries: list[AppendixBQuery]
    ) -> None:
        """Parent with no sub-sections returns empty list."""
        reg = build_subsection_registry(sample_queries)
        assert get_subsections_for_parent("B.99", registry=reg) == []


# -------------------------------------------------------------------
# TestDeriveNameHelper
# -------------------------------------------------------------------


class TestDeriveNameHelper:
    """Name derivation from query text."""

    def test_strips_what_are_the(self) -> None:
        assert _derive_name("What are the inclusion criteria?") == (
            "Inclusion criteria"
        )

    def test_strips_what_is_the(self) -> None:
        assert _derive_name("What is the protocol title?") == (
            "Protocol title"
        )

    def test_no_prefix(self) -> None:
        assert _derive_name("Inclusion criteria for subjects?") == (
            "Inclusion criteria for subjects"
        )

    def test_empty_string(self) -> None:
        assert _derive_name("") == ""


# -------------------------------------------------------------------
# TestSubSectionMatchDataModel
# -------------------------------------------------------------------


class TestSubSectionMatchDataModel:
    """SubSectionMatch data model integrity."""

    def test_frozen(self) -> None:
        m = SubSectionMatch(
            sub_section_code="B.5.1",
            parent_section_code="B.5",
            sub_section_name="Inclusion criteria",
            embedding_score=0.85,
            keyword_score=0.60,
            summarization_score=0.90,
            composite_score=0.82,
            confidence=MatchConfidence.HIGH,
            match_method="embedding+summarization",
        )
        with pytest.raises(AttributeError):
            m.composite_score = 0.5  # type: ignore[misc]

    def test_field_access(self) -> None:
        m = SubSectionMatch(
            sub_section_code="B.5.2",
            parent_section_code="B.5",
            sub_section_name="Exclusion criteria",
            embedding_score=0.70,
            keyword_score=0.40,
            summarization_score=-1.0,
            composite_score=0.59,
            confidence=MatchConfidence.REVIEW,
            match_method="keyword_fallback",
        )
        assert m.sub_section_code == "B.5.2"
        assert m.summarization_score == -1.0
        assert m.match_method == "keyword_fallback"


# -------------------------------------------------------------------
# TestCompositeScoring
# -------------------------------------------------------------------


class TestCompositeScoring:
    """Composite score weighting logic."""

    def test_default_weights_with_llm(self) -> None:
        """Full scoring with all three signals."""
        score = compute_composite_score(0.8, 0.6, 0.9)
        # 0.45*0.8 + 0.15*0.6 + 0.40*0.9 = 0.36 + 0.09 + 0.36 = 0.81
        assert score == pytest.approx(0.81, abs=0.01)

    def test_fallback_weights_when_llm_unavailable(self) -> None:
        """Summarization = -1.0 triggers fallback weights."""
        score = compute_composite_score(0.8, 0.6, -1.0)
        # 0.65*0.8 + 0.35*0.6 + 0.0*0 = 0.52 + 0.21 = 0.73
        assert score == pytest.approx(0.73, abs=0.01)

    def test_all_max_equals_one(self) -> None:
        """Perfect scores yield 1.0."""
        assert compute_composite_score(1.0, 1.0, 1.0) == 1.0

    def test_all_zero(self) -> None:
        """Zero scores yield 0.0."""
        assert compute_composite_score(0.0, 0.0, 0.0) == 0.0

    def test_clamped_to_range(self) -> None:
        """Scores are clamped to [0.0, 1.0]."""
        # Even with values > 1.0 the result is clamped
        assert compute_composite_score(2.0, 2.0, 2.0) == 1.0

    def test_custom_weights(self) -> None:
        """Custom weights are applied when LLM available."""
        w = {"embedding": 0.5, "keyword": 0.5, "summarization": 0.0}
        score = compute_composite_score(0.8, 0.4, 0.9, weights=w)
        # 0.5*0.8 + 0.5*0.4 + 0.0*0.9 = 0.60
        assert score == pytest.approx(0.60, abs=0.01)


# -------------------------------------------------------------------
# TestSectionMatchNewFields
# -------------------------------------------------------------------


class TestSectionMatchNewFields:
    """SectionMatch backward-compatible optional fields (PTCV-96)."""

    def test_old_construction_works(self) -> None:
        """Existing 6-field construction still works with defaults."""
        m = SectionMatch(
            ich_section_code="B.1",
            ich_section_name="General Information",
            similarity_score=0.9,
            boosted_score=0.9,
            confidence=MatchConfidence.HIGH,
            match_method="keyword_fallback",
        )
        assert m.sub_section_code == ""
        assert m.summarization_score == -1.0
        assert m.composite_score == -1.0

    def test_new_fields_settable(self) -> None:
        """New optional fields can be set at construction time."""
        m = SectionMatch(
            ich_section_code="B.5",
            ich_section_name="Study Population",
            similarity_score=0.8,
            boosted_score=0.85,
            confidence=MatchConfidence.HIGH,
            match_method="embedding",
            sub_section_code="B.5.1",
            summarization_score=0.92,
            composite_score=0.88,
        )
        assert m.sub_section_code == "B.5.1"
        assert m.summarization_score == 0.92
        assert m.composite_score == 0.88


# -------------------------------------------------------------------
# TestEnrichedMatchResult
# -------------------------------------------------------------------


class TestEnrichedMatchResult:
    """EnrichedMatchResult preserves MatchResult fields."""

    def test_has_all_match_result_fields(self) -> None:
        """First 5 fields mirror MatchResult."""
        er = EnrichedMatchResult(
            mappings=[],
            auto_mapped_count=3,
            review_count=1,
            unmapped_count=0,
            auto_map_rate=0.75,
            enriched_mappings=[],
            sub_section_auto_map_rate=0.50,
            llm_calls_made=10,
            llm_fallback=False,
        )
        assert er.auto_mapped_count == 3
        assert er.auto_map_rate == 0.75
        assert er.llm_calls_made == 10
        assert er.llm_fallback is False

    def test_enriched_mappings_accessible(self) -> None:
        """EnrichedSectionMapping instances are in enriched_mappings."""
        em = EnrichedSectionMapping(
            protocol_section_number="5",
            protocol_section_title="Subject Selection",
            matches=[],
            auto_mapped=True,
            sub_section_matches=[],
            parent_coverage=frozenset(),
        )
        er = EnrichedMatchResult(
            mappings=[],
            auto_mapped_count=1,
            review_count=0,
            unmapped_count=0,
            auto_map_rate=1.0,
            enriched_mappings=[em],
            sub_section_auto_map_rate=0.0,
            llm_calls_made=0,
            llm_fallback=True,
        )
        assert len(er.enriched_mappings) == 1
        assert er.enriched_mappings[0].protocol_section_number == "5"


# -------------------------------------------------------------------
# TestSummarizationMatcherFallback
# -------------------------------------------------------------------


class TestSummarizationMatcherFallback:
    """SummarizationMatcher without API key (keyword-only fallback)."""

    def test_no_api_key_sets_fallback(self) -> None:
        """No API key means LLM is disabled."""
        matcher = SummarizationMatcher(anthropic_api_key=None)
        assert matcher._use_llm is False

    def test_fallback_llm_flag_true(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Enriched result reports llm_fallback=True."""
        matcher = SummarizationMatcher(anthropic_api_key=None)
        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)
        assert result.llm_fallback is True

    def test_fallback_scores_are_negative(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Summarization scores are -1.0 in fallback mode."""
        matcher = SummarizationMatcher(anthropic_api_key=None)
        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        for em in result.enriched_mappings:
            for sub in em.sub_section_matches:
                assert sub.summarization_score == -1.0
                assert sub.match_method == "keyword_fallback"

    def test_fallback_method_excludes_summarization(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """All sub-matches use keyword_fallback method."""
        matcher = SummarizationMatcher(anthropic_api_key=None)
        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        for em in result.enriched_mappings:
            for sub in em.sub_section_matches:
                assert "summarization" not in sub.match_method


# -------------------------------------------------------------------
# TestSummarizationMatcherWithLLM
# -------------------------------------------------------------------


class TestSummarizationMatcherWithLLM:
    """SummarizationMatcher with mocked Anthropic client."""

    def _make_mock_response(self, score: float, rationale: str) -> MagicMock:
        """Build a mock Anthropic response."""
        resp = MagicMock()
        content_block = MagicMock()
        content_block.text = json.dumps(
            {"score": score, "rationale": rationale}
        )
        resp.content = [content_block]
        return resp

    def test_llm_called_per_subsection(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """LLM is called for each sub-section under the parent."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            self._make_mock_response(0.85, "Good match")
        )
        matcher._client = mock_client
        matcher._use_llm = True

        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        # B.5 has 2 sub-sections (B.5.1, B.5.2), so 2 LLM calls
        assert mock_client.messages.create.call_count == 2
        assert result.llm_calls_made == 2

    def test_method_includes_summarization(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Match method is 'embedding+summarization' when LLM active."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            self._make_mock_response(0.80, "Strong match")
        )
        matcher._client = mock_client
        matcher._use_llm = True

        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        for em in result.enriched_mappings:
            for sub in em.sub_section_matches:
                assert sub.match_method == "embedding+summarization"

    def test_llm_score_propagated(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """LLM score appears in sub-section match."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            self._make_mock_response(0.92, "Excellent fit")
        )
        matcher._client = mock_client
        matcher._use_llm = True

        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        subs = result.enriched_mappings[0].sub_section_matches
        assert all(s.summarization_score == 0.92 for s in subs)

    def test_composite_computed_with_llm(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Composite score uses default weights when LLM active."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            self._make_mock_response(0.90, "Match")
        )
        matcher._client = mock_client
        matcher._use_llm = True

        mr = _make_match_result(score=0.85)
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        for em in result.enriched_mappings:
            for sub in em.sub_section_matches:
                # composite uses embedding=0.85, keyword=some, summ=0.90
                assert sub.composite_score > 0
                assert sub.composite_score <= 1.0

    def test_parent_coverage_rollup(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Parent coverage includes HIGH and REVIEW sub-sections."""
        matcher = SummarizationMatcher(
            anthropic_api_key="test-key",
            auto_threshold=0.75,
            review_threshold=0.50,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            self._make_mock_response(0.90, "Strong match")
        )
        matcher._client = mock_client
        matcher._use_llm = True

        mr = _make_match_result(score=0.85)
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        coverage = result.enriched_mappings[0].parent_coverage
        # With high parent score + high LLM score, both should be covered
        assert len(coverage) > 0

    def test_llm_fallback_false(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """llm_fallback is False when LLM is active."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            self._make_mock_response(0.80, "Match")
        )
        matcher._client = mock_client
        matcher._use_llm = True

        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        assert result.llm_fallback is False


# -------------------------------------------------------------------
# TestSummarizationCache
# -------------------------------------------------------------------


class TestSummarizationCache:
    """In-memory LLM result caching."""

    def test_cache_hit(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Second call with same content+subsection uses cache."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            MagicMock(
                content=[
                    MagicMock(
                        text=json.dumps(
                            {"score": 0.8, "rationale": "Match"}
                        )
                    )
                ]
            )
        )
        matcher._client = mock_client
        matcher._use_llm = True

        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            # First call
            matcher.refine(mr, sample_protocol_index)
            first_calls = mock_client.messages.create.call_count

            # Second call — same content, should hit cache
            matcher.refine(mr, sample_protocol_index)

        # No additional LLM calls on second run
        assert mock_client.messages.create.call_count == first_calls

    def test_cache_miss_on_different_content(
        self,
        sample_queries: list[AppendixBQuery],
    ) -> None:
        """Different content produces a cache miss."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    text=json.dumps(
                        {"score": 0.7, "rationale": "Partial"}
                    )
                )
            ]
        )
        matcher._client = mock_client
        matcher._use_llm = True

        idx1 = ProtocolIndex(
            source_path="a.pdf",
            page_count=10,
            toc_entries=[
                TOCEntry(level=1, number="5", title="Selection"),
            ],
            section_headers=[],
            content_spans={"5": "Content about inclusion criteria"},
            full_text="",
            toc_found=True,
            toc_pages=[1],
        )
        idx2 = ProtocolIndex(
            source_path="b.pdf",
            page_count=10,
            toc_entries=[
                TOCEntry(level=1, number="5", title="Selection"),
            ],
            section_headers=[],
            content_spans={"5": "Completely different protocol content"},
            full_text="",
            toc_found=True,
            toc_pages=[1],
        )

        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            matcher.refine(mr, idx1)
            calls_after_first = mock_client.messages.create.call_count
            matcher.refine(mr, idx2)

        # Different content → additional LLM calls
        assert mock_client.messages.create.call_count > calls_after_first

    def test_cache_stats_tracked(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Cache hit/miss counters are incremented."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    text=json.dumps(
                        {"score": 0.8, "rationale": "Match"}
                    )
                )
            ]
        )
        matcher._client = mock_client
        matcher._use_llm = True

        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            matcher.refine(mr, sample_protocol_index)
            assert matcher._cache_misses > 0
            first_misses = matcher._cache_misses

            matcher.refine(mr, sample_protocol_index)
            assert matcher._cache_hits > 0
            assert matcher._cache_misses == first_misses


# -------------------------------------------------------------------
# TestEmptyAndEdgeCases
# -------------------------------------------------------------------


class TestEmptyAndEdgeCases:
    """Edge cases: empty content, no matches, unmapped sections."""

    def test_empty_matches_produce_empty_subsections(
        self,
        sample_queries: list[AppendixBQuery],
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """A mapping with no parent matches produces no sub-section matches."""
        mapping = SectionMapping(
            protocol_section_number="99",
            protocol_section_title="Unknown Section",
            matches=[],
            auto_mapped=False,
        )
        mr = MatchResult(
            mappings=[mapping],
            auto_mapped_count=0,
            review_count=0,
            unmapped_count=1,
            auto_map_rate=0.0,
        )
        matcher = SummarizationMatcher(anthropic_api_key=None)
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, sample_protocol_index)

        assert result.enriched_mappings[0].sub_section_matches == []
        assert result.enriched_mappings[0].parent_coverage == frozenset()

    def test_empty_content_span_keyword_zero(
        self,
        sample_queries: list[AppendixBQuery],
    ) -> None:
        """Missing content span yields keyword_score of 0.0."""
        idx = ProtocolIndex(
            source_path="test.pdf",
            page_count=5,
            toc_entries=[
                TOCEntry(level=1, number="5", title="Selection"),
            ],
            section_headers=[],
            content_spans={},  # No content for section "5"
            full_text="",
            toc_found=True,
            toc_pages=[1],
        )
        matcher = SummarizationMatcher(anthropic_api_key=None)
        mr = _make_match_result()
        with patch(
            "ptcv.ich_parser.summarization_matcher.build_subsection_registry"
        ) as mock_reg:
            mock_reg.return_value = build_subsection_registry(
                sample_queries
            )
            result = matcher.refine(mr, idx)

        for em in result.enriched_mappings:
            for sub in em.sub_section_matches:
                assert sub.keyword_score == 0.0

    def test_llm_call_failure_returns_zero(self) -> None:
        """LLM call exception is caught and returns score 0.0."""
        matcher = SummarizationMatcher(anthropic_api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")
        matcher._client = mock_client
        matcher._use_llm = True

        sub_def = SubSectionDef(
            code="B.5.1",
            parent_code="B.5",
            name="Inclusion criteria",
            description="inclusion criteria for subjects",
            query_ids=("B.5.1_Q1",),
        )
        result = matcher._call_llm("Some content", "Selection", sub_def)
        assert result.score == 0.0
        assert "failed" in result.rationale.lower()

    def test_builds_from_yaml_when_no_queries(self) -> None:
        """build_subsection_registry() loads from YAML if no args given."""
        reg = build_subsection_registry()
        # Should load the real schema and have multiple sub-sections
        assert len(reg) > 0
        # All entries have non-empty parent codes
        for sub in reg.values():
            assert sub.parent_code.startswith("B.")
