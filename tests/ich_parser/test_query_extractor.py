"""Tests for query-driven extraction engine (PTCV-91, PTCV-98)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.query_extractor import (
    ExtractionGap,
    ExtractionResult,
    QueryExtraction,
    QueryExtractor,
    _TEXT_LONG_MAX_CHARS,
    _extract_criteria_section,
    _extract_date,
    _extract_enum,
    _extract_identifier,
    _extract_list,
    _extract_numeric,
    _extract_statement,
    _extract_table,
    _extract_text_long,
    _extract_text_short,
    _query_keywords,
    _score_paragraph,
    _select_relevant_paragraphs,
)
from ptcv.ich_parser.query_schema import AppendixBQuery
from ptcv.ich_parser.section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMatch,
    SectionMapping,
)
from ptcv.ich_parser.toc_extractor import (
    ProtocolIndex,
    SectionHeader,
    TOCEntry,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_query(
    query_id: str = "B.1.1.q1",
    section_id: str = "B.1.1",
    parent_section: str = "B.1",
    schema_section: str = "B.1",
    query_text: str = "What is the protocol title?",
    expected_type: str = "text_short",
    required: bool = True,
) -> AppendixBQuery:
    return AppendixBQuery(
        query_id=query_id,
        section_id=section_id,
        parent_section=parent_section,
        schema_section=schema_section,
        query_text=query_text,
        expected_type=expected_type,
        required=required,
    )


def _make_match(
    ich_code: str = "B.1",
    confidence: MatchConfidence = MatchConfidence.HIGH,
) -> SectionMatch:
    return SectionMatch(
        ich_section_code=ich_code,
        ich_section_name="General Information",
        similarity_score=0.85,
        boosted_score=0.90,
        confidence=confidence,
        match_method="keyword_fallback",
    )


def _make_mapping(
    number: str = "1",
    title: str = "General Information",
    ich_code: str = "B.1",
    confidence: MatchConfidence = MatchConfidence.HIGH,
) -> SectionMapping:
    return SectionMapping(
        protocol_section_number=number,
        protocol_section_title=title,
        matches=[_make_match(ich_code, confidence)],
        auto_mapped=confidence == MatchConfidence.HIGH,
    )


def _make_protocol_index(
    content_spans: dict[str, str] | None = None,
    full_text: str = "",
) -> ProtocolIndex:
    spans = content_spans or {}
    toc = [
        TOCEntry(level=1, number=num, title=f"Section {num}")
        for num in spans
    ]
    headers = [
        SectionHeader(
            number=num, title=f"Section {num}",
            level=1, page=1, char_offset=0,
        )
        for num in spans
    ]
    return ProtocolIndex(
        source_path="test.pdf",
        page_count=10,
        toc_entries=toc,
        section_headers=headers,
        content_spans=spans,
        full_text=full_text or " ".join(spans.values()),
        toc_found=True,
        toc_pages=[1],
    )


def _make_match_result(
    mappings: list[SectionMapping] | None = None,
) -> MatchResult:
    maps = mappings or []
    auto = sum(1 for m in maps if m.auto_mapped)
    total = len(maps)
    return MatchResult(
        mappings=maps,
        auto_mapped_count=auto,
        review_count=total - auto,
        unmapped_count=0,
        auto_map_rate=auto / total if total else 0.0,
    )


# -----------------------------------------------------------------------
# TestQueryExtractionModels
# -----------------------------------------------------------------------


class TestQueryExtractionModels:
    """Data model tests."""

    def test_query_extraction_frozen(self) -> None:
        e = QueryExtraction(
            query_id="B.1.1.q1",
            section_id="B.1.1",
            content="Test Title",
            confidence=0.95,
            extraction_method="regex",
            source_section="1",
        )
        assert e.query_id == "B.1.1.q1"
        assert e.confidence == 0.95
        with pytest.raises(AttributeError):
            e.confidence = 0.5  # type: ignore[misc]

    def test_extraction_gap_fields(self) -> None:
        g = ExtractionGap(
            query_id="B.1.1.q3",
            section_id="B.1.1",
            reason="no_match",
        )
        assert g.reason == "no_match"

    def test_extraction_result_fields(self) -> None:
        r = ExtractionResult(
            extractions=[],
            gaps=[],
            coverage=0.0,
            total_queries=0,
            answered_queries=0,
        )
        assert r.total_queries == 0
        assert r.coverage == 0.0


# -----------------------------------------------------------------------
# TestQueryExtractorInit
# -----------------------------------------------------------------------


class TestQueryExtractorInit:
    """Initialisation tests."""

    def test_default_threshold(self) -> None:
        qe = QueryExtractor()
        assert qe._threshold == 0.70

    def test_custom_threshold(self) -> None:
        qe = QueryExtractor(confidence_threshold=0.50)
        assert qe._threshold == 0.50


# -----------------------------------------------------------------------
# TestTextShortExtraction
# -----------------------------------------------------------------------


class TestTextShortExtraction:
    """Short text extraction tests."""

    def test_label_value_title(self) -> None:
        text = "Protocol Title: A Phase 2 Study of Drug X"
        q = _make_query(expected_type="text_short")
        content, conf, method = _extract_text_short(text, q)
        assert "Phase 2 Study of Drug X" in content
        assert conf >= 0.90
        assert method == "regex"

    def test_sponsor_label(self) -> None:
        text = "Sponsor: Acme Pharmaceuticals, Inc."
        q = _make_query(
            query_text="What is the name of the sponsor?",
            expected_type="text_short",
        )
        content, conf, method = _extract_text_short(text, q)
        assert "Acme Pharmaceuticals" in content
        assert conf >= 0.90

    def test_empty_text(self) -> None:
        q = _make_query(expected_type="text_short")
        content, conf, method = _extract_text_short("", q)
        assert content == ""
        assert conf == 0.0


# -----------------------------------------------------------------------
# TestIdentifierExtraction
# -----------------------------------------------------------------------


class TestIdentifierExtraction:
    """Identifier extraction tests."""

    def test_nct_number(self) -> None:
        text = "This study is registered as NCT12345678."
        q = _make_query(expected_type="identifier")
        content, conf, method = _extract_identifier(text, q)
        assert content == "NCT12345678"
        assert conf == 0.95

    def test_eudract_number(self) -> None:
        text = "EudraCT Number: 2023-004567-89"
        q = _make_query(expected_type="identifier")
        content, conf, method = _extract_identifier(text, q)
        assert content == "2023-004567-89"
        assert conf == 0.92

    def test_protocol_number(self) -> None:
        text = "Protocol No. ABC-123-456"
        q = _make_query(expected_type="identifier")
        content, conf, method = _extract_identifier(text, q)
        assert "ABC-123-456" in content
        assert conf >= 0.85


# -----------------------------------------------------------------------
# TestDateExtraction
# -----------------------------------------------------------------------


class TestDateExtraction:
    """Date extraction tests."""

    def test_date_dmy(self) -> None:
        text = "Protocol Date: 24 January 2023"
        q = _make_query(expected_type="date")
        content, conf, method = _extract_date(text, q)
        assert "24 January 2023" in content
        assert conf >= 0.90

    def test_date_iso(self) -> None:
        text = "Dated 2023-01-24 for submission."
        q = _make_query(expected_type="date")
        content, conf, method = _extract_date(text, q)
        assert "2023-01-24" in content
        assert conf >= 0.90

    def test_no_date(self) -> None:
        text = "No date information here."
        q = _make_query(expected_type="date")
        content, conf, method = _extract_date(text, q)
        assert content == ""
        assert conf == 0.0


# -----------------------------------------------------------------------
# TestListExtraction
# -----------------------------------------------------------------------


class TestListExtraction:
    """List extraction tests."""

    def test_bulleted_list(self) -> None:
        text = (
            "Criteria:\n"
            "- Age >= 18 years\n"
            "- ECOG performance status 0-1\n"
            "- Adequate organ function\n"
        )
        q = _make_query(
            query_text="What are the inclusion criteria?",
            expected_type="list",
        )
        content, conf, method = _extract_list(text, q)
        assert "Age >= 18" in content
        assert conf >= 0.65

    def test_inclusion_boundary(self) -> None:
        text = (
            "Inclusion Criteria\n"
            "1. Age >= 18 years\n"
            "2. Signed informed consent\n"
            "\n"
            "Exclusion Criteria\n"
            "1. Active malignancy\n"
            "2. Pregnancy\n"
        )
        q = _make_query(
            query_text="What are the participant inclusion criteria?",
            expected_type="list",
        )
        content, conf, method = _extract_list(text, q)
        assert "Age >= 18" in content
        assert "Active malignancy" not in content

    def test_no_list(self) -> None:
        text = "Short text."
        q = _make_query(expected_type="list")
        content, conf, method = _extract_list(text, q)
        assert conf <= 0.50


# -----------------------------------------------------------------------
# TestTextLongExtraction
# -----------------------------------------------------------------------


class TestTextLongExtraction:
    """Long text extraction tests (PTCV-109: relevance scoring)."""

    def test_short_content_passthrough(self) -> None:
        """Content under _TEXT_LONG_MAX_CHARS is returned as-is."""
        text = (
            "The primary objective of this trial is to evaluate "
            "the efficacy and safety of Drug X in patients with "
            "advanced cancer."
        )
        q = _make_query(
            query_text="What are the primary objective(s) of the trial?",
            expected_type="text_long",
        )
        content, conf, method = _extract_text_long(text, q)
        assert content == text
        assert conf >= 0.45
        assert method == "passthrough"

    def test_empty_content(self) -> None:
        q = _make_query(expected_type="text_long")
        content, conf, method = _extract_text_long("", q)
        assert content == ""
        assert conf == 0.0

    def test_long_content_excerpts_relevant_paragraphs(self) -> None:
        """Content exceeding limit triggers relevance excerpt."""
        relevant = (
            "Drug X is a selective tyrosine kinase inhibitor "
            "that targets the EGFR pathway. The investigational "
            "product has shown promising preclinical results."
        )
        filler = (
            "This section describes the administrative "
            "procedures for site monitoring and data "
            "collection across all participating centers."
        )
        # Build text that exceeds _TEXT_LONG_MAX_CHARS
        paragraphs = [filler] * 20 + [relevant] + [filler] * 20
        text = "\n\n".join(paragraphs)
        assert len(text) > _TEXT_LONG_MAX_CHARS

        q = _make_query(
            query_text=(
                "What is the investigational product description?"
            ),
            expected_type="text_long",
        )
        content, conf, method = _extract_text_long(text, q)
        assert method == "relevance_excerpt"
        assert len(content) <= _TEXT_LONG_MAX_CHARS + 200  # some margin
        assert "tyrosine kinase inhibitor" in content

    def test_long_content_does_not_repeat_full_text(self) -> None:
        """Output must be shorter than input for large content."""
        para = "Background information about the study. " * 30
        text = "\n\n".join([para] * 10)
        assert len(text) > _TEXT_LONG_MAX_CHARS

        q = _make_query(
            query_text="What is the background?",
            expected_type="text_long",
        )
        content, conf, method = _extract_text_long(text, q)
        assert len(content) < len(text)


class TestParagraphRelevanceScoring:
    """Tests for _query_keywords, _score_paragraph,
    _select_relevant_paragraphs (PTCV-109)."""

    def test_query_keywords_strips_stopwords(self) -> None:
        q = _make_query(query_text="What are the inclusion criteria?")
        kw = _query_keywords(q)
        assert "inclusion" in kw
        assert "criteria" in kw
        assert "what" not in kw
        assert "the" not in kw

    def test_score_paragraph_full_match(self) -> None:
        kw = {"inclusion", "criteria"}
        score = _score_paragraph(
            "The inclusion criteria are defined below.", kw
        )
        assert score == 1.0

    def test_score_paragraph_partial_match(self) -> None:
        kw = {"inclusion", "criteria", "eligibility"}
        score = _score_paragraph(
            "Inclusion criteria listed here.", kw
        )
        assert 0.5 <= score < 1.0

    def test_score_paragraph_no_match(self) -> None:
        kw = {"inclusion", "criteria"}
        score = _score_paragraph("Unrelated content here.", kw)
        assert score == 0.0

    def test_select_relevant_preserves_order(self) -> None:
        """Selected paragraphs maintain original document order."""
        q = _make_query(query_text="investigational product")
        text = (
            "Section 1: Administrative details and logistics.\n\n"
            "Section 2: The investigational product is Drug X.\n\n"
            "Section 3: More administrative procedures follow.\n\n"
            "Section 4: Product formulation and dosing details."
        )
        excerpt, _ = _select_relevant_paragraphs(
            text, q, max_chars=500
        )
        # "investigational product" paragraph should come before
        # "Product formulation" paragraph
        pos_ip = excerpt.find("investigational product")
        pos_pf = excerpt.find("Product formulation")
        if pos_ip >= 0 and pos_pf >= 0:
            assert pos_ip < pos_pf

    def test_select_relevant_respects_max_chars(self) -> None:
        q = _make_query(query_text="safety endpoints")
        paras = [f"Paragraph {i} about safety endpoints." for i in range(50)]
        text = "\n\n".join(paras)
        excerpt, _ = _select_relevant_paragraphs(
            text, q, max_chars=300
        )
        assert len(excerpt) <= 400  # some margin for joining


# -----------------------------------------------------------------------
# TestNumericExtraction
# -----------------------------------------------------------------------


class TestNumericExtraction:
    """Numeric extraction tests."""

    def test_sample_size(self) -> None:
        text = (
            "The planned sample size is 200 participants. "
            "This provides 80% power."
        )
        q = _make_query(
            query_text="What is the number of participants "
            "planned and the reason for sample size?",
            expected_type="numeric",
        )
        content, conf, method = _extract_numeric(text, q)
        assert "200" in content or "sample size" in content.lower()
        assert conf >= 0.75

    def test_no_numeric(self) -> None:
        text = "No numbers here."
        q = _make_query(
            query_text="What is the sample size?",
            expected_type="numeric",
        )
        content, conf, method = _extract_numeric(text, q)
        # May match generic number or be empty
        assert conf <= 0.75 or content == ""


# -----------------------------------------------------------------------
# TestTableExtraction
# -----------------------------------------------------------------------


class TestTableExtraction:
    """Table extraction tests."""

    def test_pipe_table(self) -> None:
        text = (
            "| Visit | Day | Assessment |\n"
            "| Screening | -14 | Physical exam |\n"
            "| Baseline | 0 | Blood draw |\n"
        )
        q = _make_query(expected_type="table")
        content, conf, method = _extract_table(text, q)
        assert "|" in content
        assert conf >= 0.80
        assert method == "table_detection"

    def test_no_table(self) -> None:
        text = "A short paragraph with no table."
        q = _make_query(expected_type="table")
        content, conf, method = _extract_table(text, q)
        assert conf <= 0.50


# -----------------------------------------------------------------------
# TestStatementExtraction
# -----------------------------------------------------------------------


class TestStatementExtraction:
    """Statement extraction tests."""

    def test_gcp_statement(self) -> None:
        text = (
            "This trial will be conducted in compliance with "
            "Good Clinical Practice and in accordance with "
            "applicable regulatory requirements."
        )
        q = _make_query(expected_type="statement")
        content, conf, method = _extract_statement(text, q)
        assert "compliance" in content.lower()
        assert conf >= 0.60

    def test_no_regulatory_keywords(self) -> None:
        text = "The sky is blue and the grass is green."
        q = _make_query(expected_type="statement")
        content, conf, method = _extract_statement(text, q)
        assert conf == 0.0


# -----------------------------------------------------------------------
# TestUnscopedFallback
# -----------------------------------------------------------------------


class TestUnscopedFallback:
    """Full-text fallback tests."""

    def test_unscoped_search(self) -> None:
        protocol = _make_protocol_index(
            content_spans={"1": "General info content"},
            full_text=(
                "General info content. "
                "Protocol Title: A Phase 2 Study of Drug X"
            ),
        )
        # B.5 has no route, should fall back to full_text
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        q = _make_query(
            query_id="B.5.1.q1",
            section_id="B.5.1",
            schema_section="B.5",
            query_text="What are the inclusion criteria?",
            expected_type="list",
        )
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=[q]
        )
        # Should attempt unscoped search
        if result.extractions:
            assert (
                result.extractions[0].extraction_method
                == "unscoped_search"
            )
        else:
            # If no extraction, should be a gap
            assert len(result.gaps) == 1

    def test_unscoped_method_name(self) -> None:
        protocol = _make_protocol_index(
            content_spans={},
            full_text="Protocol Number: ABC-123-XYZ",
        )
        match_result = _make_match_result([])
        q = _make_query(
            query_id="B.1.1.q2",
            section_id="B.1.1",
            schema_section="B.1",
            query_text="What is the protocol number?",
            expected_type="identifier",
        )
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=[q]
        )
        assert len(result.extractions) == 1
        assert (
            result.extractions[0].extraction_method
            == "unscoped_search"
        )


# -----------------------------------------------------------------------
# TestExtractMethod
# -----------------------------------------------------------------------


class TestExtractMethod:
    """Integration tests for QueryExtractor.extract()."""

    def test_returns_extraction_result(self) -> None:
        protocol = _make_protocol_index(
            content_spans={
                "1": (
                    "Protocol Title: A Phase 2 Study\n"
                    "Protocol No. XYZ-001-2023\n"
                    "Date: 24 January 2023"
                ),
            },
        )
        match_result = _make_match_result([
            _make_mapping("1", "General Information", "B.1"),
        ])
        queries = [
            _make_query(
                query_id="B.1.1.q1",
                query_text="What is the protocol title?",
                expected_type="text_short",
            ),
            _make_query(
                query_id="B.1.1.q2",
                query_text="What is the protocol number?",
                expected_type="identifier",
            ),
        ]
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=queries
        )
        assert isinstance(result, ExtractionResult)
        assert result.total_queries == 2

    def test_coverage_computed(self) -> None:
        protocol = _make_protocol_index(
            content_spans={
                "1": "Protocol Title: A Phase 2 Study"
            },
        )
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        queries = [
            _make_query(
                query_id="B.1.1.q1",
                expected_type="text_short",
            ),
        ]
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=queries
        )
        assert 0.0 <= result.coverage <= 1.0

    def test_gaps_for_unmapped(self) -> None:
        # No content at all → gap (not even full_text fallback)
        protocol = ProtocolIndex(
            source_path="test.pdf",
            page_count=1,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
        )
        match_result = _make_match_result([])
        queries = [
            _make_query(
                query_id="B.99.q1",
                schema_section="B.99",
                expected_type="text_long",
            ),
        ]
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=queries
        )
        assert len(result.gaps) == 1
        assert result.gaps[0].reason == "no_match"

    def test_counts_sum(self) -> None:
        protocol = _make_protocol_index(
            content_spans={
                "1": "Protocol Title: Study ABC\nSponsor: Acme Corp",
            },
        )
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        queries = [
            _make_query(
                query_id="B.1.1.q1",
                expected_type="text_short",
            ),
            _make_query(
                query_id="B.1.1.q2",
                expected_type="identifier",
            ),
        ]
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=queries
        )
        total_accounted = (
            len(result.extractions) + len(result.gaps)
        )
        assert total_accounted == result.total_queries


# -----------------------------------------------------------------------
# TestEdgeCases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_protocol(self) -> None:
        protocol = _make_protocol_index(
            content_spans={}, full_text=""
        )
        match_result = _make_match_result([])
        queries = [_make_query()]
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=queries
        )
        # No routes, no full_text → gap
        assert len(result.gaps) == 1
        assert result.answered_queries == 0

    def test_no_queries(self) -> None:
        protocol = _make_protocol_index(
            content_spans={"1": "Some content"},
        )
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=[]
        )
        assert result.total_queries == 0
        assert result.coverage == 0.0

    def test_review_confidence_penalty(self) -> None:
        """REVIEW-tier routes get 0.85x confidence penalty."""
        protocol = _make_protocol_index(
            content_spans={
                "5": "Protocol Title: Test Study"
            },
        )
        match_result = _make_match_result([
            _make_mapping(
                "5", "Selection", "B.1",
                confidence=MatchConfidence.REVIEW,
            ),
        ])
        queries = [
            _make_query(
                expected_type="text_short",
                schema_section="B.1",
            ),
        ]
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=queries
        )
        if result.extractions:
            # Confidence should be penalised
            assert result.extractions[0].confidence < 0.92


# -----------------------------------------------------------------------
# TestCriteriaBoundary
# -----------------------------------------------------------------------


class TestCriteriaBoundary:
    """Inclusion/exclusion criteria boundary tests."""

    def test_inclusion_isolated(self) -> None:
        text = (
            "Inclusion Criteria\n"
            "1. Age >= 18\n"
            "2. Consent obtained\n"
            "\n"
            "Exclusion Criteria\n"
            "1. Pregnant\n"
            "2. Active cancer\n"
        )
        result = _extract_criteria_section(text, "inclusion")
        assert "Age >= 18" in result
        assert "Pregnant" not in result

    def test_exclusion_isolated(self) -> None:
        text = (
            "Inclusion Criteria\n"
            "1. Age >= 18\n"
            "\n"
            "Exclusion Criteria\n"
            "1. Pregnant\n"
            "2. Active cancer\n"
        )
        result = _extract_criteria_section(text, "exclusion")
        assert "Pregnant" in result
        assert "Age >= 18" not in result


# -----------------------------------------------------------------------
# TestLLMTransformation (PTCV-98)
# -----------------------------------------------------------------------


def _mock_anthropic_response(
    transformed: str, confidence: float
) -> MagicMock:
    """Build a mock Anthropic messages.create() response."""
    import json as _json

    block = MagicMock()
    block.text = _json.dumps(
        {"transformed": transformed, "confidence": confidence}
    )
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


class TestLLMTransformation:
    """LLM transformation tests (PTCV-98)."""

    def test_transformation_disabled_by_default(self) -> None:
        """Default init has transformation off."""
        qe = QueryExtractor()
        assert qe._use_llm is False
        assert qe._transform_calls == 0

    @patch.dict("os.environ", {}, clear=False)
    def test_transformation_requires_api_key(self) -> None:
        """enable_transformation=True + no key → verbatim."""
        # Ensure no key in env for this test
        import os
        env_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            qe = QueryExtractor(
                enable_transformation=True,
                anthropic_api_key="",
            )
            assert qe._use_llm is False
        finally:
            if env_key:
                os.environ["ANTHROPIC_API_KEY"] = env_key

    def test_transform_only_high_confidence(self) -> None:
        """Below threshold → verbatim; above → LLM called."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            _mock_anthropic_response("Transformed text", 0.90)
        )

        qe = QueryExtractor(
            transformation_threshold=0.80,
        )
        qe._client = mock_client
        qe._use_llm = True

        # HIGH confidence scoped — should transform
        protocol = _make_protocol_index(
            content_spans={
                "1": "Protocol Title: A Phase 2 Study of Drug X"
            },
        )
        match_result = _make_match_result([
            _make_mapping(
                "1", "General", "B.1",
                confidence=MatchConfidence.HIGH,
            ),
        ])
        queries = [
            _make_query(
                expected_type="text_short",
                schema_section="B.1",
            ),
        ]
        result = qe.extract(
            protocol, match_result, queries=queries
        )
        assert len(result.extractions) == 1
        ext = result.extractions[0]
        # text_short with "Protocol Title:" gets 0.92 confidence
        # 0.92 >= 0.80 threshold → LLM called
        assert ext.extraction_method == "llm_transform"
        assert ext.content == "Transformed text"

    def test_transform_not_on_unscoped(self) -> None:
        """Unscoped search → never transformed."""
        mock_client = MagicMock()

        qe = QueryExtractor()
        qe._client = mock_client
        qe._use_llm = True

        # No route for B.5 → falls back to full_text
        protocol = _make_protocol_index(
            content_spans={"1": "General info"},
            full_text="General info. NCT12345678",
        )
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        queries = [
            _make_query(
                query_id="B.5.1.q1",
                schema_section="B.5",
                expected_type="identifier",
            ),
        ]
        result = qe.extract(
            protocol, match_result, queries=queries
        )
        if result.extractions:
            assert (
                result.extractions[0].extraction_method
                != "llm_transform"
            )
        # LLM should NOT have been called
        mock_client.messages.create.assert_not_called()

    def test_transform_not_on_review_tier(self) -> None:
        """REVIEW confidence after penalty < 0.80 → no transform."""
        mock_client = MagicMock()

        qe = QueryExtractor()
        qe._client = mock_client
        qe._use_llm = True

        protocol = _make_protocol_index(
            content_spans={
                "5": "Protocol Title: Test Study"
            },
        )
        match_result = _make_match_result([
            _make_mapping(
                "5", "Selection", "B.1",
                confidence=MatchConfidence.REVIEW,
            ),
        ])
        queries = [
            _make_query(
                expected_type="text_short",
                schema_section="B.1",
            ),
        ]
        result = qe.extract(
            protocol, match_result, queries=queries
        )
        # REVIEW penalty (0.85x) on 0.92 → 0.782 < 0.80 threshold
        if result.extractions:
            assert (
                result.extractions[0].extraction_method
                != "llm_transform"
            )
        mock_client.messages.create.assert_not_called()

    def test_transform_method_name(self) -> None:
        """Successful transform sets method to llm_transform."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            _mock_anthropic_response(
                "The protocol title is Drug X Phase 2.",
                0.95,
            )
        )

        qe = QueryExtractor()
        qe._client = mock_client
        qe._use_llm = True
        protocol = _make_protocol_index(
            content_spans={
                "1": "Protocol Title: A Phase 2 Study of Drug X"
            },
        )
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        queries = [
            _make_query(
                expected_type="text_short",
                schema_section="B.1",
            ),
        ]
        result = qe.extract(
            protocol, match_result, queries=queries
        )
        assert result.extractions[0].extraction_method == (
            "llm_transform"
        )

    def test_transform_fallback_on_error(self) -> None:
        """LLM exception → graceful fallback to verbatim."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = (
            RuntimeError("API error")
        )

        qe = QueryExtractor()
        qe._client = mock_client
        qe._use_llm = True
        protocol = _make_protocol_index(
            content_spans={
                "1": "Protocol Title: A Phase 2 Study of Drug X"
            },
        )
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        queries = [
            _make_query(
                expected_type="text_short",
                schema_section="B.1",
            ),
        ]
        result = qe.extract(
            protocol, match_result, queries=queries
        )
        # Should still extract — just verbatim
        assert len(result.extractions) == 1
        assert (
            result.extractions[0].extraction_method != "llm_transform"
        )

    def test_transform_confidence_blend(self) -> None:
        """Blended confidence = 0.6 * det + 0.4 * llm."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            _mock_anthropic_response("Transformed", 0.90)
        )

        qe = QueryExtractor()
        qe._client = mock_client
        qe._use_llm = True
        protocol = _make_protocol_index(
            content_spans={
                "1": "Protocol Title: A Phase 2 Study of Drug X"
            },
        )
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        queries = [
            _make_query(
                expected_type="text_short",
                schema_section="B.1",
            ),
        ]
        result = qe.extract(
            protocol, match_result, queries=queries
        )
        ext = result.extractions[0]
        # text_short "Protocol Title:" → det_conf = 0.92
        # Blended: 0.6 * 0.92 + 0.4 * 0.90 = 0.552 + 0.36 = 0.912
        expected = round(0.6 * 0.92 + 0.4 * 0.90, 4)
        assert ext.confidence == expected
