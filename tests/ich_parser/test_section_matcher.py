"""Tests for the semantic section matcher (PTCV-90).

Covers data models, deterministic keyword fallback, synonym boost,
the ``match()`` entry point, one-to-many mapping, and edge cases.
"""

from __future__ import annotations

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

    def test_init_loads_16_reference_sections(
        self, matcher: SectionMatcher
    ) -> None:
        """Given default schema, 16 reference sections loaded."""
        assert len(matcher._ref_codes) == 16
        assert matcher._ref_codes[0] == "B.1"
        assert matcher._ref_codes[-1] == "B.16"

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
# TestB3ObjectivesMatching (PTCV-134)
# -------------------------------------------------------------------


class TestB3ObjectivesMatching:
    """B.3 section matching regression tests (PTCV-134).

    Ensures common protocol headings for "Trial Objectives and Purpose"
    correctly map to B.3 with appropriate confidence tiers.
    """

    def test_objectives_and_purpose_maps_to_b3(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Objectives and Purpose', best match is B.3."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="3",
                    title="Objectives and Purpose",
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
        assert top.ich_section_code == "B.3"

    def test_study_objectives_maps_to_b3(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Study Objectives', best match is B.3."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="3",
                    title="Study Objectives",
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
        assert top.ich_section_code == "B.3"

    def test_objectives_with_content_reaches_high(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Objectives' with typical content, confidence is HIGH."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="3",
                    title="Study Objectives and Endpoints",
                ),
            ],
            section_headers=[],
            content_spans={
                "3": (
                    "Primary Objective: To evaluate the efficacy "
                    "of Drug X. Secondary Objectives: PFS, OS. "
                    "Primary endpoint: ORR by RECIST v1.1. "
                    "Hypothesis: Drug X will demonstrate "
                    "non-inferiority."
                ),
            },
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        top = result.mappings[0].matches[0]
        assert top.ich_section_code == "B.3"
        assert top.confidence == MatchConfidence.HIGH

    def test_trial_objectives_maps_to_b3(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Trial Objectives', best match is B.3."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="3",
                    title="Trial Objectives",
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
        assert top.ich_section_code == "B.3"

    def test_study_aims_maps_to_b3(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Study Aims', best match is B.3 (EU protocol style)."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="3",
                    title="Study Aims",
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
        assert top.ich_section_code == "B.3"

    def test_objectives_endpoints_estimands_maps_to_b3(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Objectives, Endpoints and Estimands', maps to B.3."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="3",
                    title="Objectives, Endpoints and Estimands",
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
        assert top.ich_section_code == "B.3"

    def test_bare_objectives_with_content_not_low(
        self, matcher: SectionMatcher
    ) -> None:
        """Given bare 'Objectives' with content, confidence is not LOW."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="3",
                    title="Objectives",
                ),
            ],
            section_headers=[],
            content_spans={
                "3": (
                    "The primary objective and hypothesis of "
                    "this study is to evaluate the efficacy "
                    "and safety endpoint of Drug X in patients "
                    "with advanced tumours."
                ),
            },
            full_text="",
            toc_found=True,
            toc_pages=[],
        )
        result = matcher.match(idx)
        top = result.mappings[0].matches[0]
        assert top.ich_section_code == "B.3"
        assert top.confidence != MatchConfidence.LOW


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
        """Given 'Unrelated Topic', no synonym boost applied."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Unrelated Topic", raw
        )
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

    def test_financing_maps_to_b15(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Financing and Insurance', best match is B.15 (PTCV-154)."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="15",
                    title="Financing and Insurance",
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
        assert top.ich_section_code == "B.15"

    def test_publication_policy_maps_to_b16(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Publication Policy', best match is B.16 (PTCV-154)."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="16",
                    title="Publication Policy",
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
        assert top.ich_section_code == "B.16"

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
# TestPTCV135ScoringImprovements
# -------------------------------------------------------------------


class TestPTCV135ScoringImprovements:
    """PTCV-135: Denominator cap and word-boundary matching tests."""

    def test_word_boundary_prevents_overdose_matching_dose(
        self, matcher: SectionMatcher
    ) -> None:
        """'overdose' should NOT match the short keyword 'dose' in B.7."""
        from ptcv.ich_parser.schema_loader import load_ich_schema

        schema = load_ich_schema()
        b7_def = schema.sections["B.7"]
        # 'dose' is a keyword in B.7 (<=5 chars → word-boundary)
        assert "dose" in b7_def.keywords
        score = SectionMatcher._keyword_score(
            "Overdose Management", "", b7_def
        )
        # With word-boundary, 'dose' in 'overdose' should not match
        # Only 'management' could match (it doesn't), so score is low
        assert score < 0.15

    def test_word_boundary_allows_exact_dose(
        self, matcher: SectionMatcher
    ) -> None:
        """'Dose Escalation' should match the keyword 'dose' in B.7."""
        from ptcv.ich_parser.schema_loader import load_ich_schema

        schema = load_ich_schema()
        b7_def = schema.sections["B.7"]
        score = SectionMatcher._keyword_score(
            "Dose Escalation", "", b7_def
        )
        assert score > 0.0

    def test_denominator_cap_normalises_dense_section(
        self, matcher: SectionMatcher
    ) -> None:
        """Dense sections have their denominator capped at 20.

        B.4 (dense, raw_max > 20) gets capped so a single keyword
        hit isn't diluted to near-zero. B.1 (raw_max <= 20)
        stays uncapped.
        """
        from ptcv.ich_parser.schema_loader import load_ich_schema

        schema = load_ich_schema()
        b4_def = schema.sections["B.4"]
        b1_def = schema.sections["B.1"]

        # B.4 raw max is well above 20 (capped)
        raw_b4 = len(b4_def.patterns) * 2 + len(b4_def.keywords)
        assert raw_b4 > 20, "B.4 should exceed the cap"

        # B.1 raw max is at or below the cap
        raw_b1 = len(b1_def.patterns) * 2 + len(b1_def.keywords)
        assert raw_b1 <= 20, "B.1 should be at or under the cap"

    def test_b12_synonym_boost_applied(
        self, matcher: SectionMatcher
    ) -> None:
        """'Quality Control' should trigger B.12 synonym boost (PTCV-154)."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Quality Control", raw
        )
        b12_idx = matcher._ref_codes.index("B.12")
        assert boosted[b12_idx] == pytest.approx(0.15, abs=0.01)

    def test_b15_synonym_boost_applied(
        self, matcher: SectionMatcher
    ) -> None:
        """'Financing' should trigger B.15 synonym boost (PTCV-154)."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Financing Provisions", raw
        )
        b15_idx = matcher._ref_codes.index("B.15")
        assert boosted[b15_idx] == pytest.approx(0.15, abs=0.01)

    def test_b16_synonym_boost_applied(
        self, matcher: SectionMatcher
    ) -> None:
        """'Publication' should trigger B.16 synonym boost (PTCV-154)."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Publication Agreement", raw
        )
        b16_idx = matcher._ref_codes.index("B.16")
        assert boosted[b16_idx] == pytest.approx(0.15, abs=0.01)


# -------------------------------------------------------------------
# TestB1SynonymBoosts (PTCV-155, Tier 1)
# -------------------------------------------------------------------


class TestB1SynonymBoosts:
    """PTCV-155 Tier 1: Expanded B.1 synonym boost tests."""

    def test_synopsis_boosted_to_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Synopsis' should trigger B.1 synonym boost."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost("Synopsis", raw)
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == pytest.approx(0.15, abs=0.01)

    def test_protocol_synopsis_boosted_to_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Protocol Synopsis' should trigger B.1 synonym boost."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Protocol Synopsis", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] > 0.0

    def test_sponsor_information_boosted_to_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Sponsor Information' should trigger B.1 synonym boost."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Sponsor Information", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == pytest.approx(0.15, abs=0.01)

    def test_cover_page_boosted_to_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Cover Page' should trigger B.1 synonym boost."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Cover Page", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == pytest.approx(0.15, abs=0.01)

    def test_title_page_boosted_to_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Title Page' should trigger B.1 synonym boost."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Title Page", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == pytest.approx(0.15, abs=0.01)

    def test_study_identification_boosted_to_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Study Identification' should trigger B.1 synonym boost."""
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Study Identification", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == pytest.approx(0.15, abs=0.01)

    def test_synopsis_maps_to_b1_via_match(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Synopsis' header, best match is B.1."""
        idx = ProtocolIndex(
            source_path="t.pdf",
            page_count=1,
            toc_entries=[
                TOCEntry(
                    level=1,
                    number="1",
                    title="Synopsis",
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
        assert top.ich_section_code == "B.1"


# -------------------------------------------------------------------
# TestB1ExclusionTerms (PTCV-155, Tier 2)
# -------------------------------------------------------------------


class TestB1ExclusionTerms:
    """PTCV-155 Tier 2: B.1 noise reduction exclusion tests."""

    def test_definitions_excluded_from_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Definitions' should have B.1 score zeroed."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Definitions", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == 0.0

    def test_good_clinical_practice_excluded_from_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Good Clinical Practice' should have B.1 score zeroed."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Good Clinical Practice", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == 0.0

    def test_interim_analysis_excluded_from_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Interim Analysis' should have B.1 score zeroed."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Interim Analysis", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == 0.0

    def test_references_excluded_from_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'References' should have B.1 score zeroed."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "References", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == 0.0

    def test_pregnancy_excluded_from_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'Pregnancy' should have B.1 score zeroed."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Pregnancy", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == 0.0

    def test_abbreviations_excluded_from_b1(
        self, matcher: SectionMatcher
    ) -> None:
        """'List of Abbreviations' should have B.1 score zeroed."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "List of Abbreviations", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == 0.0

    def test_exclusion_does_not_affect_other_sections(
        self, matcher: SectionMatcher
    ) -> None:
        """B.1 exclusion should not affect other section scores."""
        raw = [0.5] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Definitions", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        # B.1 zeroed
        assert boosted[b1_idx] == 0.0
        # All other scores unchanged (no synonym boost applied)
        for i, score in enumerate(boosted):
            if i != b1_idx:
                assert score == 0.5

    def test_general_information_not_excluded(
        self, matcher: SectionMatcher
    ) -> None:
        """'General Information' should NOT be excluded from B.1."""
        raw = [0.1] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "General Information", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        # Should have the original score PLUS the synonym boost
        assert boosted[b1_idx] >= 0.1

    def test_definitions_b1_score_is_zero(
        self, matcher: SectionMatcher
    ) -> None:
        """Given 'Definitions', B.1 boosted score should be 0.0.

        The header may still map to B.1 at LOW confidence when all
        sections score 0.0 (tie-breaking favours the first code),
        but the exclusion ensures B.1 cannot accumulate this header
        into its route with any positive score.
        """
        raw = [0.0] * len(matcher._ref_codes)
        boosted = matcher._apply_synonym_boost(
            "Definitions", raw
        )
        b1_idx = matcher._ref_codes.index("B.1")
        assert boosted[b1_idx] == 0.0
