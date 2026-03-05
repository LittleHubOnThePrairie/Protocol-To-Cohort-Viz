"""Tests for the semantic section matcher (PTCV-90).

Covers data models, deterministic keyword fallback, synonym boost,
the ``match()`` entry point, one-to-many mapping, edge cases, and
embedding mode (mocked Cohere).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMatch,
    SectionMatcher,
    SectionMapping,
)
from ptcv.ich_parser.toc_extractor import (
    ProtocolIndex,
    SectionHeader,
    TOCEntry,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture()
def sample_protocol_index() -> ProtocolIndex:
    """Synthetic ProtocolIndex with common protocol headers."""
    return ProtocolIndex(
        source_path="test.pdf",
        page_count=50,
        toc_entries=[
            TOCEntry(level=1, number="1", title="General Information"),
            TOCEntry(level=1, number="2", title="Background and Rationale"),
            TOCEntry(level=1, number="3", title="Objectives and Purpose"),
            TOCEntry(level=1, number="4", title="Study Design"),
            TOCEntry(level=1, number="5", title="Selection of Subjects"),
            TOCEntry(level=2, number="5.1", title="Inclusion Criteria"),
            TOCEntry(level=2, number="5.2", title="Exclusion Criteria"),
            TOCEntry(level=1, number="6", title="Discontinuation"),
            TOCEntry(level=1, number="7", title="Study Treatment"),
            TOCEntry(level=1, number="8", title="Efficacy Assessment"),
            TOCEntry(level=1, number="9", title="Safety Assessment"),
            TOCEntry(level=1, number="10", title="Statistical Methods"),
            TOCEntry(level=1, number="11", title="Quality Assurance"),
            TOCEntry(level=1, number="12", title="Ethics"),
            TOCEntry(level=1, number="13", title="Data Handling"),
            TOCEntry(level=1, number="14", title="Publication Policy"),
        ],
        section_headers=[],
        content_spans={
            "1": "Protocol Title: A Phase 2 Study of Drug X",
            "5.1": "Adults aged 18 or older with confirmed diagnosis",
        },
        full_text="",
        toc_found=True,
        toc_pages=[2, 3],
    )


@pytest.fixture()
def matcher() -> SectionMatcher:
    """SectionMatcher in deterministic (no-API) mode."""
    return SectionMatcher(cohere_api_key=None)


# -------------------------------------------------------------------
# TestSectionMatchDataModels
# -------------------------------------------------------------------


class TestSectionMatchDataModels:
    """Data model integrity tests."""

    def test_match_confidence_enum_values(self) -> None:
        """Given the MatchConfidence enum, values are correct."""
        assert MatchConfidence.HIGH.value == "high"
        assert MatchConfidence.REVIEW.value == "review"
        assert MatchConfidence.LOW.value == "low"

    def test_section_match_frozen(self) -> None:
        """Given a SectionMatch, it is immutable."""
        m = SectionMatch(
            ich_section_code="B.1",
            ich_section_name="General Information",
            similarity_score=0.9,
            boosted_score=0.9,
            confidence=MatchConfidence.HIGH,
            match_method="keyword_fallback",
        )
        with pytest.raises(AttributeError):
            m.similarity_score = 0.5  # type: ignore[misc]

    def test_section_mapping_frozen(self) -> None:
        """Given a SectionMapping, it is immutable."""
        mapping = SectionMapping(
            protocol_section_number="1",
            protocol_section_title="Test",
            matches=[],
            auto_mapped=False,
        )
        with pytest.raises(AttributeError):
            mapping.auto_mapped = True  # type: ignore[misc]


# -------------------------------------------------------------------
# TestSectionMatcherInit
# -------------------------------------------------------------------


class TestSectionMatcherInit:
    """Construction and initialisation tests."""

    def test_init_without_api_key(self, matcher: SectionMatcher) -> None:
        """Given no API key, embeddings are disabled."""
        assert matcher._use_embeddings is False

    def test_init_loads_14_reference_sections(
        self, matcher: SectionMatcher
    ) -> None:
        """Given default schema, 14 reference sections loaded."""
        assert len(matcher._ref_codes) == 14
        assert matcher._ref_codes[0] == "B.1"
        assert matcher._ref_codes[-1] == "B.14"

    def test_init_custom_thresholds(self) -> None:
        """Given custom thresholds, they are stored."""
        m = SectionMatcher(
            cohere_api_key=None,
            auto_threshold=0.80,
            review_threshold=0.60,
            synonym_boost=0.20,
        )
        assert m._auto_threshold == 0.80
        assert m._review_threshold == 0.60
        assert m._synonym_boost == 0.20


# -------------------------------------------------------------------
# TestDeterministicFallback
# -------------------------------------------------------------------


class TestDeterministicFallback:
    """Keyword/pattern scoring produces correct mappings."""

    def test_inclusion_criteria_maps_to_b5(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Inclusion Criteria', best match is B.5."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="5",
                    title="Inclusion Criteria",
                ),
            ],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        top = result.mappings[0].matches[0]
        assert top.ich_section_code == "B.5"

    def test_background_maps_to_b2(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Background and Rationale', best match is B.2."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="2",
                    title="Background and Rationale",
                ),
            ],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        top = result.mappings[0].matches[0]
        assert top.ich_section_code == "B.2"

    def test_study_design_maps_to_b4(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Study Design', best match is B.4."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1, number="4", title="Study Design"
                ),
            ],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        top = result.mappings[0].matches[0]
        assert top.ich_section_code == "B.4"

    def test_statistical_methods_maps_to_b10(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Statistical Methods', best match is B.10."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="10",
                    title="Statistical Methods",
                ),
            ],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        top = result.mappings[0].matches[0]
        assert top.ich_section_code == "B.10"

    def test_safety_assessment_maps_to_b9(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Safety Assessment', best match is B.9."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="9",
                    title="Safety Assessment",
                ),
            ],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        top = result.mappings[0].matches[0]
        assert top.ich_section_code == "B.9"

    def test_unknown_header_gets_low_confidence(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'List of Abbreviations', confidence is LOW."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="99",
                    title="List of Abbreviations",
                ),
            ],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        top = result.mappings[0].matches[0]
        assert top.confidence == MatchConfidence.LOW

    def test_method_is_keyword_fallback(
        self, matcher: SectionMatcher
    ) -> None:
        """Given no API key, match_method is 'keyword_fallback'."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="5",
                    title="Inclusion Criteria",
                ),
            ],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        assert (
            result.mappings[0].matches[0].match_method
            == "keyword_fallback"
        )


# -------------------------------------------------------------------
# TestSynonymBoost
# -------------------------------------------------------------------


class TestSynonymBoost:
    """Synonym boost table tests."""

    def test_inclusion_criteria_boosted(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Inclusion Criteria', B.5 score is boosted."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Inclusion Criteria", raw
        )
        b5_idx = matcher._ref_codes.index("B.5")
        assert boosted[b5_idx] == pytest.approx(0.15, abs=0.01)

    def test_boost_capped_at_1(
        self, matcher: SectionMatcher
    ) -> None:
        """Given raw score 0.95 + boost 0.15, capped at 1.0."""
        raw = [0.0] * len(matcher._ref_codes)
        b5_idx = matcher._ref_codes.index("B.5")
        raw[b5_idx] = 0.95
        boosted = matcher._apply_synonym_boost(
            "Inclusion Criteria", raw
        )
        assert boosted[b5_idx] == 1.0

    def test_no_boost_for_unrecognised_header(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Appendix A', no synonym boost applied."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost("Appendix A", raw)
        assert boosted == raw

    def test_boost_is_additive(
        self, matcher: SectionMatcher
    ) -> None:
        """Given raw score 0.60, boosted is 0.60 + 0.15 = 0.75."""
        raw = [0.0] * len(matcher._ref_codes)
        b5_idx = matcher._ref_codes.index("B.5")
        raw[b5_idx] = 0.60
        boosted = matcher._apply_synonym_boost(
            "Inclusion Criteria", raw
        )
        assert boosted[b5_idx] == pytest.approx(0.75, abs=0.01)


# -------------------------------------------------------------------
# TestMatchMethod
# -------------------------------------------------------------------


class TestMatchMethod:
    """Tests for the match() entry point."""

    def test_returns_match_result(
        self,
        matcher: SectionMatcher,
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Given a protocol index, returns a MatchResult."""
        result = matcher.match(sample_protocol_index)
        assert isinstance(result, MatchResult)

    def test_all_headers_have_mappings(
        self,
        matcher: SectionMatcher,
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Given 16 TOC entries, 16 mappings produced."""
        result = matcher.match(sample_protocol_index)
        assert len(result.mappings) == 16

    def test_counts_sum_to_total(
        self,
        matcher: SectionMatcher,
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Given any result, counts sum to total headers."""
        result = matcher.match(sample_protocol_index)
        total = (
            result.auto_mapped_count
            + result.review_count
            + result.unmapped_count
        )
        assert total == len(result.mappings)

    def test_auto_map_rate_is_fraction(
        self,
        matcher: SectionMatcher,
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Given a result, auto_map_rate is between 0 and 1."""
        result = matcher.match(sample_protocol_index)
        assert 0.0 <= result.auto_map_rate <= 1.0

    def test_auto_mapped_flag_matches_confidence(
        self,
        matcher: SectionMatcher,
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Given mappings, auto_mapped iff top match is HIGH."""
        result = matcher.match(sample_protocol_index)
        for mapping in result.mappings:
            if mapping.auto_mapped:
                assert (
                    mapping.matches[0].confidence
                    == MatchConfidence.HIGH
                )

    def test_review_tier_has_up_to_3_candidates(
        self,
        matcher: SectionMatcher,
        sample_protocol_index: ProtocolIndex,
    ) -> None:
        """Given REVIEW-tier mappings, at most 3 candidates."""
        result = matcher.match(sample_protocol_index)
        for mapping in result.mappings:
            if (
                mapping.matches
                and mapping.matches[0].confidence
                == MatchConfidence.REVIEW
            ):
                assert len(mapping.matches) <= 3


# -------------------------------------------------------------------
# TestOneToMany
# -------------------------------------------------------------------


class TestOneToMany:
    """One-to-many mapping tests."""

    def test_eligibility_criteria_maps_to_b5(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Eligibility Criteria' covering inclusion+exclusion,
        maps to B.5 (which handles both via patterns)."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="5",
                    title="Eligibility Criteria",
                ),
            ],
            section_headers=[],
            content_spans={
                "5": (
                    "Inclusion criteria: age >= 18 "
                    "Exclusion criteria: prior therapy"
                ),
            },
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        codes = [m.ich_section_code for m in result.mappings[0].matches]
        assert "B.5" in codes


# -------------------------------------------------------------------
# TestEdgeCases
# -------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_toc_entries_returns_empty(
        self, matcher: SectionMatcher
    ) -> None:
        """Given empty toc_entries and section_headers, empty result."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
        )
        result = matcher.match(idx)
        assert result.mappings == []
        assert result.auto_map_rate == 0.0

    def test_no_content_spans_still_works(
        self, matcher: SectionMatcher
    ) -> None:
        """Given toc_entries but no content_spans, matching works."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="5",
                    title="Inclusion Criteria",
                ),
            ],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        assert len(result.mappings) == 1

    def test_section_headers_fallback(
        self, matcher: SectionMatcher
    ) -> None:
        """Given empty toc_entries, falls back to section_headers."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[],
            section_headers=[
                SectionHeader(
                    number="5",
                    title="Inclusion Criteria",
                    level=1,
                    page=10,
                    char_offset=500,
                ),
            ],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
        )
        result = matcher.match(idx)
        assert len(result.mappings) == 1
        assert result.mappings[0].matches[0].ich_section_code == "B.5"


# -------------------------------------------------------------------
# TestEmbeddingMode (mocked Cohere)
# -------------------------------------------------------------------


_HAS_COHERE = True
try:
    import cohere  # noqa: F401
except ImportError:
    _HAS_COHERE = False

_skip_no_cohere = pytest.mark.skipif(
    not _HAS_COHERE, reason="cohere SDK not installed"
)


@_skip_no_cohere
class TestEmbeddingMode:
    """Embedding path tests with mocked Cohere client."""

    def test_embedding_mode_calls_cohere(self) -> None:
        """Given an API key, Cohere client is initialised."""
        import numpy as np

        fake_embeddings = np.random.default_rng(42).standard_normal(
            (14, 64)
        )

        mock_client = MagicMock()
        embed_resp = MagicMock()
        embed_resp.embeddings = fake_embeddings.tolist()
        mock_client.embed.return_value = embed_resp

        with patch(
            "cohere.Client", return_value=mock_client
        ):
            m = SectionMatcher(cohere_api_key="fake-key")

        assert m._use_embeddings is True
        mock_client.embed.assert_called_once()

    def test_embedding_scores_shape(self) -> None:
        """Given mocked embeddings, score matrix has correct shape."""
        import numpy as np

        rng = np.random.default_rng(42)
        mock_client = MagicMock()

        # Reference embeddings (14 sections x 64 dims)
        ref_emb = rng.standard_normal((14, 64)).astype(
            np.float32
        )
        ref_resp = MagicMock()
        ref_resp.embeddings = ref_emb.tolist()

        # Query embeddings (3 headers x 64 dims)
        query_emb = rng.standard_normal((3, 64)).astype(
            np.float32
        )
        query_resp = MagicMock()
        query_resp.embeddings = query_emb.tolist()

        mock_client.embed.side_effect = [ref_resp, query_resp]

        with patch(
            "cohere.Client", return_value=mock_client
        ):
            m = SectionMatcher(cohere_api_key="fake-key")

        # Reset side_effect for query call
        mock_client.embed.side_effect = None
        mock_client.embed.return_value = query_resp

        scores = m._embedding_scores(
            ["Header 1", "Header 2", "Header 3"]
        )
        assert len(scores) == 3
        assert len(scores[0]) == 14

    def test_embedding_scores_in_valid_range(self) -> None:
        """Given mocked embeddings, all scores in [-1, 1]."""
        import numpy as np

        rng = np.random.default_rng(42)
        mock_client = MagicMock()

        ref_emb = rng.standard_normal((14, 64)).astype(
            np.float32
        )
        ref_resp = MagicMock()
        ref_resp.embeddings = ref_emb.tolist()

        query_emb = rng.standard_normal((2, 64)).astype(
            np.float32
        )
        query_resp = MagicMock()
        query_resp.embeddings = query_emb.tolist()

        mock_client.embed.side_effect = [ref_resp, query_resp]

        with patch(
            "cohere.Client", return_value=mock_client
        ):
            m = SectionMatcher(cohere_api_key="fake-key")

        mock_client.embed.side_effect = None
        mock_client.embed.return_value = query_resp

        scores = m._embedding_scores(["Test 1", "Test 2"])
        for row in scores:
            for s in row:
                assert -1.0 <= s <= 1.0 + 1e-6
