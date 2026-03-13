"""Tests for query-driven extraction engine (PTCV-91, PTCV-98)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.query_extractor import (
    ExtractionGap,
    ExtractionResult,
    QueryExtraction,
    QueryExtractor,
    _CITATION_MIN_LEN,
    _LLM_TRANSFORM_MAX_CHARS,
    _NUMBERED_HEADING_RE,
    _TEXT_LONG_MAX_CHARS,
    _CONFIDENCE_RANK,
    _MAX_ROUTE_CHARS,
    _MAX_SECTIONS_PER_ROUTE,
    _REFUSAL_PATTERNS,
    _UNSCOPED_MAX_CHARS,
    _is_refusal,
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
    _is_full_citation,
    _match_heading_to_subsection,
    _query_keywords,
    _score_paragraph,
    _select_relevant_paragraphs,
    extract_citations,
    filter_citations,
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

    def test_transform_all_scoped(self) -> None:
        """All scoped extractions are transformed (PTCV-111)."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            _mock_anthropic_response("Transformed text", 0.90)
        )

        qe = QueryExtractor()
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

    def test_transform_fires_on_review_tier(self) -> None:
        """REVIEW-tier scoped matches are transformed (PTCV-111).

        Confidence penalty is applied AFTER transformation, so
        REVIEW matches still get sent to the LLM.
        """
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            _mock_anthropic_response("Reviewed text", 0.85)
        )

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
        assert len(result.extractions) == 1
        ext = result.extractions[0]
        # PTCV-111: transformation fires first, then REVIEW penalty
        assert ext.extraction_method == "llm_transform"
        assert ext.content == "Reviewed text"
        # Confidence: 0.6*(0.92) + 0.4*(0.85) = 0.892 → *0.85 = 0.7582
        assert ext.confidence < 0.80  # penalty pushes it down
        mock_client.messages.create.assert_called_once()

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

    def test_review_penalty_applied_post_blend(self) -> None:
        """REVIEW penalty applied AFTER LLM blend (PTCV-111).

        Flow: det_conf=0.92 → blend(0.6*0.92 + 0.4*0.85)=0.892
              → REVIEW penalty 0.85 → 0.892*0.85 = 0.7582
        """
        mock_client = MagicMock()
        mock_client.messages.create.return_value = (
            _mock_anthropic_response("Reviewed text", 0.85)
        )

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
        ext = result.extractions[0]
        # Blend first: 0.6*0.92 + 0.4*0.85 = 0.892
        # Then REVIEW penalty: 0.892 * 0.85 = 0.7582
        blended = round(0.6 * 0.92 + 0.4 * 0.85, 4)
        expected = round(blended * 0.85, 4)
        assert ext.confidence == expected
        assert ext.extraction_method == "llm_transform"

    def test_unscoped_penalty_no_transform(self) -> None:
        """Unscoped extractions skip transform and get penalty
        (PTCV-111)."""
        mock_client = MagicMock()

        qe = QueryExtractor()
        qe._client = mock_client
        qe._use_llm = True

        # B.5 query has no route → full_text fallback (unscoped)
        protocol = _make_protocol_index(
            content_spans={"1": "General info"},
            full_text=(
                "Protocol Title: NCT12345678 Drug Study.\n"
                "General info. NCT12345678"
            ),
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
            ext = result.extractions[0]
            assert ext.extraction_method != "llm_transform"
        mock_client.messages.create.assert_not_called()


# -----------------------------------------------------------------------
# TestSubSectionRouting (PTCV-110)
# -----------------------------------------------------------------------


class TestSubSectionRouting:
    """Sub-section level routing tests (PTCV-110)."""

    def test_b2x_queries_get_different_content(self) -> None:
        """B.2.1 and B.2.2 queries receive differentiated content."""
        protocol = _make_protocol_index(
            content_spans={
                "2": (
                    "The investigational product is Drug X, a "
                    "selective inhibitor targeting the EGFR "
                    "pathway. The formulation is a 100mg oral "
                    "capsule administered once daily.\n\n"
                    "Summary of nonclinical studies showed no "
                    "significant toxicity in animal models at "
                    "therapeutic doses. Preclinical findings "
                    "indicate favorable safety margins.\n\n"
                    "Known risks and benefits: nausea, fatigue, "
                    "and rash are the primary adverse events. "
                    "The benefit-risk balance is favorable."
                ),
            },
        )
        match_result = _make_match_result([
            _make_mapping("2", "Background", "B.2"),
        ])
        queries = [
            _make_query(
                query_id="B.2.1.q1",
                section_id="B.2.1",
                parent_section="B.2",
                schema_section="B.2",
                query_text=(
                    "What is the name and description of the "
                    "investigational product(s)?"
                ),
                expected_type="text_long",
            ),
            _make_query(
                query_id="B.2.2.q1",
                section_id="B.2.2",
                parent_section="B.2",
                schema_section="B.2",
                query_text=(
                    "What is the summary of findings from "
                    "nonclinical studies that potentially have "
                    "clinical significance?"
                ),
                expected_type="text_long",
            ),
        ]
        qe = QueryExtractor()
        result = qe.extract(
            protocol, match_result, queries=queries
        )
        assert len(result.extractions) == 2
        # The two extractions should have DIFFERENT content
        # because they route to different sub-section buckets.
        c0 = result.extractions[0].content
        c1 = result.extractions[1].content
        assert c0 != c1, (
            "B.2.1 and B.2.2 should get different content"
        )

    def test_subsection_route_preferred_over_parent(
        self,
    ) -> None:
        """section_id lookup happens before schema_section."""
        protocol = _make_protocol_index(
            content_spans={
                "5": (
                    "Inclusion criteria: adult patients aged "
                    "18 or older with confirmed diagnosis.\n\n"
                    "Exclusion criteria: pregnant women, "
                    "patients with severe hepatic impairment."
                ),
            },
        )
        match_result = _make_match_result([
            _make_mapping("5", "Eligibility", "B.5"),
        ])
        q_inclusion = _make_query(
            query_id="B.5.1.q1",
            section_id="B.5.1",
            parent_section="B.5",
            schema_section="B.5",
            query_text=(
                "What are the inclusion criteria for trial "
                "participants?"
            ),
            expected_type="list",
        )
        q_exclusion = _make_query(
            query_id="B.5.2.q1",
            section_id="B.5.2",
            parent_section="B.5",
            schema_section="B.5",
            query_text=(
                "What are the exclusion criteria for trial "
                "participants?"
            ),
            expected_type="list",
        )
        qe = QueryExtractor()
        result = qe.extract(
            protocol, match_result,
            queries=[q_inclusion, q_exclusion],
        )
        assert len(result.extractions) == 2
        inc = result.extractions[0].content
        exc = result.extractions[1].content
        assert "inclusion" in inc.lower()
        assert "exclusion" in exc.lower()

    def test_fallback_to_parent_when_no_subsections(
        self,
    ) -> None:
        """Sections without sub-sections still route correctly."""
        protocol = _make_protocol_index(
            content_spans={
                "1": "Protocol Title: Study of Drug X",
            },
        )
        match_result = _make_match_result([
            _make_mapping("1", "General", "B.1"),
        ])
        # B.1 has only one sub-section query, should still work
        q = _make_query(
            query_id="B.1.1.q1",
            section_id="B.1.1",
            parent_section="B.1",
            schema_section="B.1",
            query_text="What is the protocol title?",
            expected_type="text_short",
        )
        qe = QueryExtractor()
        result = qe.extract(
            protocol, match_result, queries=[q]
        )
        assert len(result.extractions) == 1
        assert "Drug X" in result.extractions[0].content

    def test_split_content_by_subsection(self) -> None:
        """Unit test for _split_content_by_subsection."""
        from ptcv.ich_parser.summarization_matcher import (
            SubSectionDef,
        )

        registry = {
            "B.2.1": SubSectionDef(
                code="B.2.1",
                parent_code="B.2",
                name="Investigational Product",
                description=(
                    "What is the name and description of the "
                    "investigational product formulation dose?"
                ),
                query_ids=("B.2.1.q1",),
            ),
            "B.2.2": SubSectionDef(
                code="B.2.2",
                parent_code="B.2",
                name="Nonclinical Findings",
                description=(
                    "What is the summary of findings from "
                    "nonclinical preclinical studies toxicity "
                    "clinical significance?"
                ),
                query_ids=("B.2.2.q1",),
            ),
        }
        content = (
            "Drug X is a selective EGFR inhibitor. The "
            "formulation is a 100mg oral capsule for daily "
            "investigational product administration.\n\n"
            "Nonclinical studies in rodent models showed no "
            "significant toxicity. Preclinical findings "
            "demonstrate favorable safety margins."
        )
        result = QueryExtractor._split_content_by_subsection(
            content, "B.2", registry
        )
        assert "B.2.1" in result
        assert "B.2.2" in result
        assert "investigational" in result["B.2.1"].lower()
        assert "nonclinical" in result["B.2.2"].lower()


# -----------------------------------------------------------------------
# TestExtractProgressCallback (PTCV-124)
# -----------------------------------------------------------------------


class TestExtractProgressCallback:
    """Tests for progress_callback in QueryExtractor.extract()."""

    def test_callback_called_for_each_query(self) -> None:
        """Callback fires once per query with (done, total)."""
        protocol = _make_protocol_index(
            content_spans={
                "1": "Protocol Title: A Phase 2 Study",
            },
        )
        match_result = _make_match_result([
            _make_mapping("1", "General Information", "B.1"),
        ])
        queries = [
            _make_query(query_id=f"B.1.1.q{i}")
            for i in range(5)
        ]
        calls: list[tuple[int, int]] = []
        extractor = QueryExtractor()
        extractor.extract(
            protocol, match_result, queries=queries,
            progress_callback=lambda d, t: calls.append((d, t)),
        )
        assert len(calls) == 5
        assert calls[-1] == (5, 5)

    def test_callback_done_monotonically_increases(self) -> None:
        """The done count increases with each callback."""
        protocol = _make_protocol_index(
            content_spans={"1": "Some content"},
        )
        match_result = _make_match_result([
            _make_mapping("1", "Info", "B.1"),
        ])
        queries = [
            _make_query(query_id=f"B.1.1.q{i}")
            for i in range(3)
        ]
        calls: list[tuple[int, int]] = []
        extractor = QueryExtractor()
        extractor.extract(
            protocol, match_result, queries=queries,
            progress_callback=lambda d, t: calls.append((d, t)),
        )
        done_values = [c[0] for c in calls]
        assert done_values == sorted(done_values)
        assert all(c[1] == 3 for c in calls)

    def test_no_callback_no_error(self) -> None:
        """None callback does not raise."""
        protocol = _make_protocol_index(
            content_spans={"1": "Content"},
        )
        match_result = _make_match_result([
            _make_mapping("1", "Info", "B.1"),
        ])
        queries = [_make_query()]
        extractor = QueryExtractor()
        result = extractor.extract(
            protocol, match_result, queries=queries,
            progress_callback=None,
        )
        assert isinstance(result, ExtractionResult)


# -----------------------------------------------------------------------
# PTCV-144: Content extraction bug fixes
# -----------------------------------------------------------------------


class TestHeadingBasedSplitting:
    """Heading-based subsection splitting tests (PTCV-144)."""

    def test_heading_based_split_inclusion_exclusion(
        self,
    ) -> None:
        """Explicit headings correctly split B.5.1/B.5.2."""
        from ptcv.ich_parser.summarization_matcher import (
            SubSectionDef,
        )

        registry = {
            "B.5.1": SubSectionDef(
                code="B.5.1",
                parent_code="B.5",
                name="Inclusion Criteria",
                description=(
                    "What are the participant inclusion "
                    "criteria for trial eligibility?"
                ),
                query_ids=("B.5.1.q1",),
            ),
            "B.5.2": SubSectionDef(
                code="B.5.2",
                parent_code="B.5",
                name="Exclusion Criteria",
                description=(
                    "What are the participant exclusion "
                    "criteria?"
                ),
                query_ids=("B.5.2.q1",),
            ),
        }
        content = (
            "Preamble text about eligibility.\n\n"
            "11.1 Inclusion Criteria\n"
            "1. Age >= 18 years\n"
            "2. Written informed consent\n"
            "3. Confirmed diagnosis of Type 2 diabetes\n\n"
            "11.2 Exclusion Criteria\n"
            "1. Pregnant or breastfeeding\n"
            "2. Active malignancy\n"
            "3. Severe hepatic impairment\n"
        )
        result = QueryExtractor._split_content_by_subsection(
            content, "B.5", registry
        )
        assert "B.5.1" in result
        assert "B.5.2" in result
        assert "Age >= 18" in result["B.5.1"]
        assert "Pregnant" not in result["B.5.1"]
        assert "Pregnant" in result["B.5.2"]
        assert "Age >= 18" not in result["B.5.2"]

    def test_heading_split_fallback_to_keyword_scoring(
        self,
    ) -> None:
        """No headings → keyword scoring fallback (PTCV-144)."""
        from ptcv.ich_parser.summarization_matcher import (
            SubSectionDef,
        )

        registry = {
            "B.2.1": SubSectionDef(
                code="B.2.1",
                parent_code="B.2",
                name="Investigational Product",
                description=(
                    "What is the name and description of "
                    "the investigational product formulation "
                    "dose?"
                ),
                query_ids=("B.2.1.q1",),
            ),
            "B.2.2": SubSectionDef(
                code="B.2.2",
                parent_code="B.2",
                name="Nonclinical Findings",
                description=(
                    "What is the summary of findings from "
                    "nonclinical preclinical studies "
                    "toxicity?"
                ),
                query_ids=("B.2.2.q1",),
            ),
        }
        # No numbered headings — just paragraphs.
        content = (
            "Drug X is a selective EGFR inhibitor. The "
            "formulation is a 100mg oral capsule for daily "
            "investigational product administration.\n\n"
            "Nonclinical studies in rodent models showed no "
            "significant toxicity. Preclinical findings "
            "demonstrate favorable safety margins."
        )
        result = QueryExtractor._split_content_by_subsection(
            content, "B.2", registry
        )
        assert "B.2.1" in result
        assert "B.2.2" in result

    def test_heading_split_single_heading_falls_back(
        self,
    ) -> None:
        """Single heading match falls back to keywords."""
        from ptcv.ich_parser.summarization_matcher import (
            SubSectionDef,
        )

        registry = {
            "B.5.1": SubSectionDef(
                code="B.5.1",
                parent_code="B.5",
                name="Inclusion Criteria",
                description="inclusion criteria eligibility",
                query_ids=("B.5.1.q1",),
            ),
            "B.5.2": SubSectionDef(
                code="B.5.2",
                parent_code="B.5",
                name="Exclusion Criteria",
                description="exclusion criteria",
                query_ids=("B.5.2.q1",),
            ),
        }
        # Only one heading matches — should fall back.
        content = (
            "11.1 Inclusion Criteria\n"
            "1. Age >= 18\n\n"
            "Patients who are pregnant are excluded.\n"
        )
        result = QueryExtractor._split_content_by_subsection(
            content, "B.5", registry
        )
        assert isinstance(result, dict)

    def test_heading_regex_matches_various_formats(
        self,
    ) -> None:
        """Heading regex handles diverse numbering formats."""
        cases = [
            ("11.1 Inclusion Criteria", True),
            ("5.2.1 Exclusion Criteria", True),
            ("3. Study Objectives", True),
            ("1. age >= 18", False),  # lowercase = not a heading
            ("A short line", False),  # no number prefix
        ]
        for text, should_match in cases:
            matches = list(_NUMBERED_HEADING_RE.finditer(text))
            if should_match:
                assert len(matches) >= 1, (
                    f"Expected match for: {text}"
                )
            else:
                assert len(matches) == 0, (
                    f"Unexpected match for: {text}"
                )

    def test_match_heading_to_subsection(self) -> None:
        """Helper matches headings to subsections by name."""
        from ptcv.ich_parser.summarization_matcher import (
            SubSectionDef,
        )

        sub_defs = [
            SubSectionDef(
                code="B.5.1",
                parent_code="B.5",
                name="Inclusion Criteria",
                description="inclusion criteria",
                query_ids=("B.5.1.q1",),
            ),
            SubSectionDef(
                code="B.5.2",
                parent_code="B.5",
                name="Exclusion Criteria",
                description="exclusion criteria",
                query_ids=("B.5.2.q1",),
            ),
        ]
        assert _match_heading_to_subsection(
            "Inclusion Criteria", sub_defs
        ) == "B.5.1"
        assert _match_heading_to_subsection(
            "Exclusion Criteria", sub_defs
        ) == "B.5.2"
        assert _match_heading_to_subsection(
            "Unrelated Topic", sub_defs
        ) is None

    def test_preamble_assigned_to_parent(self) -> None:
        """Text before first heading goes to parent bucket."""
        from ptcv.ich_parser.summarization_matcher import (
            SubSectionDef,
        )

        registry = {
            "B.5.1": SubSectionDef(
                code="B.5.1",
                parent_code="B.5",
                name="Inclusion Criteria",
                description="inclusion criteria",
                query_ids=("B.5.1.q1",),
            ),
            "B.5.2": SubSectionDef(
                code="B.5.2",
                parent_code="B.5",
                name="Exclusion Criteria",
                description="exclusion criteria",
                query_ids=("B.5.2.q1",),
            ),
        }
        content = (
            "This section describes eligibility.\n\n"
            "11.1 Inclusion Criteria\n"
            "1. Age >= 18\n\n"
            "11.2 Exclusion Criteria\n"
            "1. Pregnant\n"
        )
        result = QueryExtractor._split_content_by_subsection(
            content, "B.5", registry
        )
        assert "B.5" in result
        assert "eligibility" in result["B.5"].lower()


class TestTextLongLimitIncrease:
    """Tests for increased _TEXT_LONG_MAX_CHARS (PTCV-144)."""

    def test_limit_value_is_4000(self) -> None:
        """_TEXT_LONG_MAX_CHARS raised to 4000."""
        assert _TEXT_LONG_MAX_CHARS == 4000

    def test_medium_content_passthrough(self) -> None:
        """Content 2000-4000 chars passes through intact."""
        para = (
            "The adverse event follow-up procedure requires "
            "investigators to document all events within "
            "24 hours of occurrence. Visit schedules must "
            "be maintained on a weekly basis for the first "
            "month following the event. Outcome "
            "classifications include resolved, ongoing, "
            "and fatal. All serious adverse events require "
            "expedited reporting to the sponsor."
        )
        text = "\n\n".join([para] * 8)
        assert 2000 < len(text) < 4000

        q = _make_query(
            query_text=(
                "What are the adverse event follow-up "
                "procedures?"
            ),
            expected_type="text_long",
        )
        content, conf, method = _extract_text_long(text, q)
        assert method == "passthrough"
        assert content == text.strip()


class TestLLMTransformLimitIncrease:
    """Tests for increased _LLM_TRANSFORM_MAX_CHARS (PTCV-144)."""

    def test_limit_value_is_10000(self) -> None:
        """_LLM_TRANSFORM_MAX_CHARS raised to 10000."""
        assert _LLM_TRANSFORM_MAX_CHARS == 10000


class TestEndToEndPTCV144:
    """Integration tests for PTCV-144 fixes."""

    def test_criteria_with_explicit_headings(self) -> None:
        """Full pipeline: explicit headings → correct criteria."""
        content = (
            "This section describes patient eligibility.\n\n"
            "11.1 Inclusion Criteria\n"
            "1. Age 18 or older\n"
            "2. Confirmed diagnosis\n"
            "3. Written informed consent obtained\n\n"
            "11.2 Exclusion Criteria\n"
            "1. Pregnant or nursing\n"
            "2. Known hypersensitivity to study drug\n"
            "3. Active malignancy within 5 years\n"
        )
        protocol = _make_protocol_index(
            content_spans={"5": content},
        )
        match_result = _make_match_result([
            _make_mapping("5", "Eligibility", "B.5"),
        ])
        q_inc = _make_query(
            query_id="B.5.1.q1",
            section_id="B.5.1",
            parent_section="B.5",
            schema_section="B.5",
            query_text="What are the inclusion criteria?",
            expected_type="list",
        )
        q_exc = _make_query(
            query_id="B.5.2.q1",
            section_id="B.5.2",
            parent_section="B.5",
            schema_section="B.5",
            query_text="What are the exclusion criteria?",
            expected_type="list",
        )
        qe = QueryExtractor()
        result = qe.extract(
            protocol, match_result,
            queries=[q_inc, q_exc],
        )
        assert len(result.extractions) == 2
        inc = result.extractions[0].content
        exc = result.extractions[1].content
        # Inclusion must contain inclusion items.
        assert (
            "Age 18" in inc or "informed consent" in inc
        )
        # Exclusion must contain exclusion items.
        assert "Pregnant" in exc or "malignancy" in exc
        # Critically: inclusion must NOT contain exclusion.
        assert "Pregnant" not in inc
        assert "hypersensitivity" not in inc

    def test_long_ae_content_not_truncated(self) -> None:
        """AE follow-up procedures fully preserved."""
        # Build AE content ~3000 chars (above old 2000 limit,
        # below new 4000 limit).
        ae_text = (
            "13.2.8 Adverse Event Follow-up Procedures\n"
            "AEs (whether serious or non-serious) and "
            "clinically significant abnormal laboratory "
            "test values will be evaluated by the PI or "
            "designee and treated and/or followed up until "
            "the symptoms or values return to normal or "
            "acceptable levels.\n\n"
            "Treatment of SAEs will be performed by a "
            "licensed medical provider, either at the CRU "
            "or at a nearby hospital emergency room. Where "
            "appropriate, medical tests and examinations "
            "will be performed to document resolution of "
            "events. The outcome may be classified as "
            "resolved, improved, unchanged, worse, fatal, "
            "or unknown (lost to follow up).\n\n"
            "Follow-up visit schedule:\n"
            "- Day 7: Clinical assessment and vital signs\n"
            "- Day 14: Laboratory tests and ECG\n"
            "- Day 30: Final follow-up phone call\n"
            "- Day 90: Long-term safety assessment\n\n"
            "All SAEs must be reported to the sponsor "
            "within 24 hours of the investigator becoming "
            "aware of the event. The initial report should "
            "include all available information about the "
            "event, including the date of onset, severity, "
            "and any actions taken."
        )
        protocol = _make_protocol_index(
            content_spans={"13": ae_text},
        )
        match_result = _make_match_result([
            _make_mapping(
                "13", "Safety Assessment", "B.9"
            ),
        ])
        q = _make_query(
            query_id="B.9.4.q1",
            section_id="B.9.4",
            parent_section="B.9",
            schema_section="B.9",
            query_text=(
                "What is the type and duration of "
                "follow-up of participants after adverse "
                "events?"
            ),
            expected_type="text_long",
        )
        qe = QueryExtractor()
        result = qe.extract(
            protocol, match_result, queries=[q]
        )
        assert len(result.extractions) >= 1
        extracted = result.extractions[0].content
        # Day 30 follow-up must be preserved (was truncated
        # with the old 2000-char limit).
        assert "Day 30" in extracted or "Day 7" in extracted


# -------------------------------------------------------------------
# TestFrontMatterExtraction (PTCV-155, Tier 3)
# -------------------------------------------------------------------


class TestFrontMatterExtraction:
    """PTCV-155 Tier 3: Front-matter injection for B.1 routes."""

    def _make_protocol_with_front_matter(
        self,
        front_matter: str = "",
        body_content: str = "",
        b1_route_confidence: MatchConfidence | None = None,
    ) -> tuple[ProtocolIndex, MatchResult]:
        """Build a ProtocolIndex with front matter before section 1."""
        # Front matter + section boundary
        full_text = front_matter + "\n" + body_content
        first_header_offset = len(front_matter) + 1

        headers = [
            SectionHeader(
                number="1",
                title="Background",
                level=1,
                page=4,
                char_offset=first_header_offset,
            ),
        ]
        idx = ProtocolIndex(
            source_path="test.pdf",
            page_count=50,
            toc_entries=[
                TOCEntry(level=1, number="1", title="Background"),
            ],
            section_headers=headers,
            content_spans={"1": body_content},
            full_text=full_text,
            toc_found=True,
            toc_pages=[2, 3],
        )

        mappings = [
            _make_mapping(
                "1", "Background", "B.2", MatchConfidence.HIGH
            ),
        ]
        if b1_route_confidence is not None:
            mappings.append(
                _make_mapping(
                    "0",
                    "General Information",
                    "B.1",
                    b1_route_confidence,
                ),
            )
        match_result = _make_match_result(mappings)
        return idx, match_result

    def test_front_matter_injected_when_no_b1_route(
        self,
    ) -> None:
        """Given no B.1 route, front matter creates one."""
        cover = (
            "Protocol Title: A Phase 3 Study of Drug X\n"
            "Sponsor: Acme Pharma Inc.\n"
            "Protocol Number: ACME-001-2024\n"
            "Date: 15 March 2025\n"
        ) * 3  # Repeat to exceed 100 chars

        idx, match_result = (
            self._make_protocol_with_front_matter(
                front_matter=cover,
                body_content="1 Background\nDisease background...",
            )
        )

        routes: dict[
            str, tuple[str, str, MatchConfidence]
        ] = {}
        # Simulate route building with no B.1 match
        QueryExtractor._inject_front_matter_route(
            routes, idx
        )

        assert "B.1" in routes
        content, source, conf = routes["B.1"]
        assert "Acme Pharma" in content
        assert "ACME-001-2024" in content
        assert source == "front_matter"
        assert conf == MatchConfidence.REVIEW

    def test_front_matter_skipped_when_b1_has_high_route(
        self,
    ) -> None:
        """Given B.1 already has HIGH route, front matter not injected."""
        idx, _ = self._make_protocol_with_front_matter(
            front_matter="Cover page content " * 20,
            body_content="1 Background\nContent...",
        )

        routes: dict[
            str, tuple[str, str, MatchConfidence]
        ] = {
            "B.1": (
                "General Information content",
                "1",
                MatchConfidence.HIGH,
            ),
        }
        QueryExtractor._inject_front_matter_route(
            routes, idx
        )

        # Route unchanged
        content, source, conf = routes["B.1"]
        assert content == "General Information content"
        assert conf == MatchConfidence.HIGH

    def test_front_matter_prepended_to_low_b1_route(
        self,
    ) -> None:
        """Given B.1 has LOW route, front matter prepended."""
        cover = "Protocol: Drug X Study\nSponsor: BigPharma\n" * 5
        idx, _ = self._make_protocol_with_front_matter(
            front_matter=cover,
            body_content="1 Background\nContent...",
        )

        routes: dict[
            str, tuple[str, str, MatchConfidence]
        ] = {
            "B.1": (
                "Some noisy content about definitions",
                "99",
                MatchConfidence.LOW,
            ),
        }
        QueryExtractor._inject_front_matter_route(
            routes, idx
        )

        content, source, conf = routes["B.1"]
        assert "BigPharma" in content
        assert "front_matter" in source
        assert conf == MatchConfidence.REVIEW

    def test_front_matter_skipped_when_too_short(
        self,
    ) -> None:
        """Given <50 chars of front matter, no injection."""
        idx, _ = self._make_protocol_with_front_matter(
            front_matter="Short",
            body_content="1 Background\nContent...",
        )

        routes: dict[
            str, tuple[str, str, MatchConfidence]
        ] = {}
        QueryExtractor._inject_front_matter_route(
            routes, idx
        )

        assert "B.1" not in routes

    def test_front_matter_capped_at_5000_chars(
        self,
    ) -> None:
        """Front matter content capped at 5000 characters."""
        # 10000 chars of cover content
        cover = "A" * 10000
        idx, _ = self._make_protocol_with_front_matter(
            front_matter=cover,
            body_content="1 Background\nContent...",
        )

        routes: dict[
            str, tuple[str, str, MatchConfidence]
        ] = {}
        QueryExtractor._inject_front_matter_route(
            routes, idx
        )

        assert "B.1" in routes
        content, _, _ = routes["B.1"]
        assert len(content) <= 5000


# ===================================================================
# TestUnscopedLengthCap (PTCV-157, Tier 3)
# ===================================================================


class TestUnscopedLengthCap:
    """Verify unscoped_search fallback is capped at _UNSCOPED_MAX_CHARS."""

    def test_constant_value(self) -> None:
        assert _UNSCOPED_MAX_CHARS == 10000

    def test_unscoped_paragraphs_capped(self) -> None:
        """_select_relevant_paragraphs with UNSCOPED cap limits output."""
        large_text = (
            "Unrelated content paragraph here.\n\n" * 1000
        )  # ~35K chars
        query = AppendixBQuery(
            query_id="B.8.1.q1",
            section_id="B.8.1",
            parent_section="B.8",
            schema_section="B.8",
            query_text="What are the efficacy parameters?",
            expected_type="text_long",
            required=True,
        )
        excerpt, _ = _select_relevant_paragraphs(
            large_text, query, max_chars=_UNSCOPED_MAX_CHARS,
        )
        assert len(excerpt) <= _UNSCOPED_MAX_CHARS + 500

    def test_unscoped_selects_relevant_content(self) -> None:
        """Unscoped fallback should prefer query-relevant paragraphs."""
        irrelevant = "Lorem ipsum dolor sit amet. " * 200
        relevant = (
            "The primary efficacy endpoint is PASI 75 at Week 12. "
            "Secondary efficacy endpoints include sPGA scores and "
            "PsA joint counts."
        )
        full_text = irrelevant + "\n\n" + relevant + "\n\n" + irrelevant

        query = AppendixBQuery(
            query_id="B.8.1.q1",
            section_id="B.8.1",
            parent_section="B.8",
            schema_section="B.8",
            query_text="What are the efficacy parameters?",
            expected_type="text_long",
            required=True,
        )

        excerpt, _ = _select_relevant_paragraphs(
            full_text, query, max_chars=_UNSCOPED_MAX_CHARS,
        )
        # The relevant paragraph should appear in the output
        assert "PASI 75" in excerpt


# ===================================================================
# TestCitationDetection (PTCV-99)
# ===================================================================


class TestIsFullCitation:
    """Unit tests for _is_full_citation() pattern matching."""

    def test_apa_style(self) -> None:
        line = (
            "Smith, J. A., & Jones, B. C. (2021). Efficacy of "
            "pembrolizumab in advanced NSCLC: a meta-analysis. "
            "J Clin Oncol, 39(15), 1234-1245."
        )
        assert _is_full_citation(line) is True

    def test_vancouver_numbered(self) -> None:
        line = (
            "1. Anderson BC, Chen L, Williams DP. Phase III trial "
            "of nivolumab in hepatocellular carcinoma. Lancet Oncol. "
            "2022;23(4):512-521."
        )
        assert _is_full_citation(line) is True

    def test_vancouver_bracketed(self) -> None:
        line = (
            "[12] Garcia MR, Patel S. Biomarker-driven patient "
            "stratification in oncology trials. Nat Rev Clin Oncol. "
            "2020;17(2):89-104."
        )
        assert _is_full_citation(line) is True

    def test_doi_reference(self) -> None:
        line = (
            "Thompson K, et al. Novel endpoints in dermatology "
            "trials: a systematic review. doi: 10.1016/j.jaad.2023.01.045"
        )
        assert _is_full_citation(line) is True

    def test_pmid_reference(self) -> None:
        line = (
            "Wilson D, Brown E. Adaptive designs in oncology: "
            "current landscape and future directions. PMID: 34567890"
        )
        assert _is_full_citation(line) is True

    def test_short_parenthetical_rejected(self) -> None:
        assert _is_full_citation("(Smith et al., 2023)") is False

    def test_short_text_rejected(self) -> None:
        assert _is_full_citation("See reference 12.") is False

    def test_empty_rejected(self) -> None:
        assert _is_full_citation("") is False

    def test_long_but_no_pattern(self) -> None:
        line = "A" * (_CITATION_MIN_LEN + 10)
        assert _is_full_citation(line) is False

    def test_whitespace_stripped(self) -> None:
        line = (
            "   Smith, J. A. (2020). Long title that exceeds the "
            "minimum length threshold for detection purposes.   "
        )
        assert _is_full_citation(line) is True


class TestExtractCitations:
    """Unit tests for extract_citations()."""

    def test_extracts_from_mixed_text(self) -> None:
        text = (
            "The study showed efficacy (Smith et al., 2020).\n"
            "\n"
            "Smith, J. A., & Jones, B. C. (2021). Efficacy of "
            "pembrolizumab in advanced NSCLC: a meta-analysis. "
            "J Clin Oncol, 39(15), 1234-1245.\n"
            "\n"
            "Results were consistent with prior work."
        )
        result = extract_citations(text)
        assert len(result) == 1
        assert "pembrolizumab" in result[0]

    def test_preserves_order(self) -> None:
        text = (
            "1. Anderson BC, Chen L. Phase III trial of nivolumab "
            "in hepatocellular carcinoma. Lancet Oncol. 2022.\n"
            "2. Garcia MR, Patel S. Biomarker-driven patient "
            "stratification in oncology. Nat Rev Clin Oncol. 2020."
        )
        result = extract_citations(text)
        assert len(result) == 2
        assert "Anderson" in result[0]
        assert "Garcia" in result[1]

    def test_no_citations_returns_empty(self) -> None:
        text = "This is a normal paragraph with no references."
        assert extract_citations(text) == []

    def test_empty_input(self) -> None:
        assert extract_citations("") == []

    def test_skips_parenthetical(self) -> None:
        text = (
            "As noted by (Author, 2021) the results were "
            "significant [1-3]."
        )
        assert extract_citations(text) == []


class TestFilterCitations:
    """Unit tests for filter_citations()."""

    def test_removes_full_citations(self) -> None:
        text = (
            "Results showed improvement.\n"
            "\n"
            "Smith, J. A. (2021). A comprehensive review of "
            "clinical trial design methods in modern oncology. "
            "J Clin Oncol, 39(15), 1234.\n"
            "\n"
            "Conclusions were drawn."
        )
        result = filter_citations(text)
        assert "Results showed improvement" in result
        assert "Conclusions were drawn" in result
        assert "Smith, J. A." not in result

    def test_preserves_parenthetical_refs(self) -> None:
        text = "The effect was significant (Smith et al., 2023)."
        result = filter_citations(text)
        assert "(Smith et al., 2023)" in result

    def test_cleans_excessive_blank_lines(self) -> None:
        text = (
            "Line 1.\n\n"
            "Smith, J. A. (2021). Very long citation that is at "
            "least sixty characters so it meets the minimum. "
            "J Clin Oncol, 39(15).\n\n\n"
            "Line 2."
        )
        result = filter_citations(text)
        assert "\n\n\n" not in result

    def test_no_citations_unchanged(self) -> None:
        text = "Normal text with no citations at all."
        assert filter_citations(text) == text

    def test_empty_input(self) -> None:
        assert filter_citations("") == ""


class TestConsolidateCitations:
    """Unit tests for QueryExtractor._consolidate_citations()."""

    def test_collects_into_b27(self) -> None:
        citation = (
            "Smith, J. A. (2021). Efficacy of treatment X in "
            "advanced cancer: a randomized controlled trial. "
            "J Clin Oncol, 39(15), 1234-1245."
        )
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.5": (
                f"Inclusion criteria.\n{citation}",
                "section_5",
                MatchConfidence.HIGH,
            ),
        }
        QueryExtractor._consolidate_citations(routes)
        assert "B.2.7" in routes
        assert "Smith" in routes["B.2.7"][0]
        # Citation stripped from B.5
        assert "Smith" not in routes["B.5"][0]

    def test_appends_to_existing_b27(self) -> None:
        citation = (
            "Jones, B. C. (2022). Novel biomarkers for patient "
            "stratification in clinical oncology trials. "
            "Nat Rev Clin Oncol, 19(8), 501-515."
        )
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.2.7": (
                "Existing references content.",
                "existing_src",
                MatchConfidence.HIGH,
            ),
            "B.8": (
                f"Efficacy endpoints.\n{citation}",
                "section_8",
                MatchConfidence.HIGH,
            ),
        }
        QueryExtractor._consolidate_citations(routes)
        content_b27 = routes["B.2.7"][0]
        assert "Existing references content" in content_b27
        assert "Consolidated Bibliography" in content_b27
        assert "Jones" in content_b27

    def test_uses_b2_confidence_when_no_b27(self) -> None:
        citation = (
            "Wilson, D. (2020). Adaptive designs in oncology: "
            "current landscape and future directions for practice. "
            "doi: 10.1016/j.example.2020.01.001"
        )
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.2": (
                "Background text.",
                "section_2",
                MatchConfidence.REVIEW,
            ),
            "B.5": (
                f"Criteria.\n{citation}",
                "section_5",
                MatchConfidence.HIGH,
            ),
        }
        QueryExtractor._consolidate_citations(routes)
        assert "B.2.7" in routes
        assert routes["B.2.7"][2] == MatchConfidence.REVIEW

    def test_no_citations_noop(self) -> None:
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.5": (
                "Normal text with no citations.",
                "section_5",
                MatchConfidence.HIGH,
            ),
        }
        QueryExtractor._consolidate_citations(routes)
        assert "B.2.7" not in routes

    def test_deduplicates_citations(self) -> None:
        citation = (
            "Smith, J. A. (2021). Efficacy of treatment X in "
            "advanced cancer: a randomized controlled trial. "
            "J Clin Oncol, 39(15), 1234-1245."
        )
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.5": (
                f"Criteria text.\n{citation}",
                "section_5",
                MatchConfidence.HIGH,
            ),
            "B.8": (
                f"Efficacy text.\n{citation}",
                "section_8",
                MatchConfidence.HIGH,
            ),
        }
        QueryExtractor._consolidate_citations(routes)
        b27_content = routes["B.2.7"][0]
        # Should appear only once despite being in two routes
        assert b27_content.count("J Clin Oncol") == 1

    def test_preserves_b2_route_citations(self) -> None:
        citation = (
            "Smith, J. A. (2021). Efficacy of treatment X in "
            "advanced cancer: a randomized controlled trial. "
            "J Clin Oncol, 39(15), 1234-1245."
        )
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.2": (
                f"Background info.\n{citation}",
                "section_2",
                MatchConfidence.HIGH,
            ),
            "B.5": (
                f"Criteria.\n{citation}",
                "section_5",
                MatchConfidence.HIGH,
            ),
        }
        QueryExtractor._consolidate_citations(routes)
        # B.2 content should NOT be filtered
        assert "Smith" in routes["B.2"][0]
        # B.5 content SHOULD be filtered
        assert "Smith" not in routes["B.5"][0]

    def test_review_confidence_when_no_b2(self) -> None:
        citation = (
            "Anderson, K. (2023). Long citation text that exceeds "
            "the sixty character minimum for detection purposes. "
            "PMID: 12345678"
        )
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.8": (
                f"Efficacy.\n{citation}",
                "section_8",
                MatchConfidence.HIGH,
            ),
        }
        QueryExtractor._consolidate_citations(routes)
        assert routes["B.2.7"][2] == MatchConfidence.REVIEW


# -------------------------------------------------------------------
# TestMarkdownHeadingSplitting (PTCV-171)
# -------------------------------------------------------------------


class TestMarkdownHeadingSplitting:
    """Markdown heading detection for subsection splitting."""

    def test_md_headings_split_inclusion_exclusion(self) -> None:
        """Markdown headings (## / ###) correctly split sub-sections.

        [PTCV-171 Scenario: Markdown headings used for section
        classification]
        """
        from ptcv.ich_parser.summarization_matcher import (
            SubSectionDef,
        )

        registry = {
            "B.5.1": SubSectionDef(
                code="B.5.1",
                parent_code="B.5",
                name="Inclusion Criteria",
                description=(
                    "What are the participant inclusion "
                    "criteria for trial eligibility?"
                ),
                query_ids=("B.5.1.q1",),
            ),
            "B.5.2": SubSectionDef(
                code="B.5.2",
                parent_code="B.5",
                name="Exclusion Criteria",
                description=(
                    "What are the participant exclusion "
                    "criteria?"
                ),
                query_ids=("B.5.2.q1",),
            ),
        }
        content = (
            "Preamble text about eligibility.\n\n"
            "### 11.1 Inclusion Criteria\n\n"
            "1. Age >= 18 years\n"
            "2. Written informed consent\n"
            "3. Confirmed diagnosis of Type 2 diabetes\n\n"
            "### 11.2 Exclusion Criteria\n\n"
            "1. Pregnant or breastfeeding\n"
            "2. Active malignancy\n"
            "3. Severe hepatic impairment\n"
        )
        result = QueryExtractor._split_content_by_subsection(
            content, "B.5", registry
        )
        assert "B.5.1" in result
        assert "B.5.2" in result
        assert "Age >= 18" in result["B.5.1"]
        assert "Pregnant" not in result["B.5.1"]
        assert "Pregnant" in result["B.5.2"]

    def test_md_tables_preserved_through_routes(self) -> None:
        """Markdown tables survive route building concatenation.

        [PTCV-171 Scenario: Tables preserved through route
        concatenation]
        """
        protocol = ProtocolIndex(
            source_path="test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=[],
            content_spans={
                "8": (
                    "## 8 Efficacy Assessment\n\n"
                    "| Endpoint | Measure | Timepoint |\n"
                    "|----------|---------|----------|\n"
                    "| Primary | OS | 12 months |\n"
                    "| Secondary | PFS | 6 months |\n"
                ),
                "8.1": (
                    "### 8.1 Primary Endpoint\n\n"
                    "| Assessment | Schedule |\n"
                    "|------------|----------|\n"
                    "| CT scan | Every 8 weeks |\n"
                ),
            },
            full_text="",
            toc_found=True,
            toc_pages=[1],
        )

        match_result = MatchResult(
            mappings=[
                _make_mapping("8", "Efficacy Assessment", "B.8"),
                _make_mapping("8.1", "Primary Endpoint", "B.8"),
            ],
            auto_mapped_count=2,
            review_count=0,
            unmapped_count=0,
            auto_map_rate=1.0,
        )

        routes = QueryExtractor._build_routes(
            protocol, match_result
        )

        assert "B.8" in routes
        content = routes["B.8"][0]
        # Table pipe syntax preserved through concatenation
        assert "| Endpoint |" in content
        assert "| CT scan |" in content

    def test_md_heading_preferred_over_plain_text(self) -> None:
        """Markdown headings are preferred over plain-text headings
        when both are present at the same position.
        """
        from ptcv.ich_parser.summarization_matcher import (
            SubSectionDef,
        )

        registry = {
            "B.5.1": SubSectionDef(
                code="B.5.1",
                parent_code="B.5",
                name="Inclusion Criteria",
                description="Inclusion criteria for eligibility",
                query_ids=("B.5.1.q1",),
            ),
            "B.5.2": SubSectionDef(
                code="B.5.2",
                parent_code="B.5",
                name="Exclusion Criteria",
                description="Exclusion criteria",
                query_ids=("B.5.2.q1",),
            ),
        }
        # Content has BOTH ## headings AND plain-text numbered
        # headings (the ## lines also contain the number).
        content = (
            "## 5.1 Inclusion Criteria\n\n"
            "- Age >= 18 years\n\n"
            "## 5.2 Exclusion Criteria\n\n"
            "- Pregnant women\n"
        )
        result = QueryExtractor._split_content_by_subsection(
            content, "B.5", registry
        )
        assert "B.5.1" in result
        assert "B.5.2" in result
        assert "Age >= 18" in result["B.5.1"]
        assert "Pregnant" in result["B.5.2"]


# -----------------------------------------------------------------------
# PTCV-182: Route building quality tests
# -----------------------------------------------------------------------


class TestPTCV182RouteBuildingQuality:
    """Validate LOW-confidence gating, route caps, and
    confidence merge fixes (PTCV-182)."""

    def test_low_confidence_mappings_included_but_capped(
        self,
    ) -> None:
        """LOW-confidence mappings produce routes (best-effort)
        but are subject to section count caps."""
        low_mappings = [
            _make_mapping(
                number=str(i),
                title=f"Section {i}",
                ich_code="B.1",
                confidence=MatchConfidence.LOW,
            )
            for i in range(1, 11)
        ]
        high_mapping = _make_mapping(
            number="20",
            title="Study Objectives",
            ich_code="B.5",
            confidence=MatchConfidence.HIGH,
        )
        spans = {str(i): f"Low content {i}" for i in range(1, 11)}
        spans["20"] = "High confidence objectives"

        idx = _make_protocol_index(content_spans=spans)
        mr = _make_match_result(low_mappings + [high_mapping])

        routes = QueryExtractor._build_routes(idx, mr)

        assert "B.1" in routes
        # Capped at _MAX_SECTIONS_PER_ROUTE
        section_count = routes["B.1"][1].count(",") + 1
        assert section_count <= _MAX_SECTIONS_PER_ROUTE
        assert routes["B.1"][2] == MatchConfidence.LOW
        assert "B.5" in routes
        assert routes["B.5"][2] == MatchConfidence.HIGH

    def test_review_mappings_included(self) -> None:
        """REVIEW-confidence mappings must produce routes."""
        mapping = _make_mapping(
            number="3",
            title="Study Design",
            ich_code="B.4",
            confidence=MatchConfidence.REVIEW,
        )
        idx = _make_protocol_index(
            content_spans={"3": "Study design content"}
        )
        mr = _make_match_result([mapping])

        routes = QueryExtractor._build_routes(idx, mr)

        assert "B.4" in routes
        assert routes["B.4"][2] == MatchConfidence.REVIEW

    def test_route_section_count_cap(self) -> None:
        """Routes must cap at _MAX_SECTIONS_PER_ROUTE sections."""
        mappings = [
            _make_mapping(
                number=str(i),
                title=f"Section {i}",
                ich_code="B.7",
                confidence=MatchConfidence.REVIEW,
            )
            for i in range(1, 9)  # 8 mappings
        ]
        spans = {
            str(i): f"Content block {i}" for i in range(1, 9)
        }
        idx = _make_protocol_index(content_spans=spans)
        mr = _make_match_result(mappings)

        routes = QueryExtractor._build_routes(idx, mr)

        assert "B.7" in routes
        source_sections = routes["B.7"][1]
        count = source_sections.count(",") + 1
        assert count <= _MAX_SECTIONS_PER_ROUTE

    def test_route_char_limit_cap(self) -> None:
        """Routes must cap at _MAX_ROUTE_CHARS characters."""
        big_chunk = "x" * 8000
        mappings = [
            _make_mapping(
                number=str(i),
                title=f"Section {i}",
                ich_code="B.9",
                confidence=MatchConfidence.REVIEW,
            )
            for i in range(1, 5)
        ]
        spans = {str(i): big_chunk for i in range(1, 5)}
        idx = _make_protocol_index(content_spans=spans)
        mr = _make_match_result(mappings)

        routes = QueryExtractor._build_routes(idx, mr)

        assert "B.9" in routes
        assert len(routes["B.9"][0]) <= _MAX_ROUTE_CHARS + 500

    def test_worst_confidence_high_plus_low(self) -> None:
        """Ordinal merge: HIGH + LOW = LOW (was HIGH due to bug)."""
        assert _CONFIDENCE_RANK[MatchConfidence.HIGH] == 3
        assert _CONFIDENCE_RANK[MatchConfidence.LOW] == 1

        # LOW mappings are now excluded, so test the rank dict
        # directly to confirm the fix.
        high_rank = _CONFIDENCE_RANK[MatchConfidence.HIGH]
        low_rank = _CONFIDENCE_RANK[MatchConfidence.LOW]
        # Worst = min rank
        worst = (
            MatchConfidence.HIGH
            if high_rank < low_rank
            else MatchConfidence.LOW
        )
        assert worst == MatchConfidence.LOW

    def test_worst_confidence_review_plus_low(self) -> None:
        """Ordinal merge: REVIEW + LOW = LOW."""
        review_rank = _CONFIDENCE_RANK[MatchConfidence.REVIEW]
        low_rank = _CONFIDENCE_RANK[MatchConfidence.LOW]
        worst = (
            MatchConfidence.REVIEW
            if review_rank < low_rank
            else MatchConfidence.LOW
        )
        assert worst == MatchConfidence.LOW

    def test_worst_confidence_high_plus_review(self) -> None:
        """Ordinal merge: HIGH + REVIEW = REVIEW."""
        # Two REVIEW mappings to same ICH code — verify merge
        m1 = _make_mapping(
            number="1", title="A", ich_code="B.3",
            confidence=MatchConfidence.HIGH,
        )
        m2 = _make_mapping(
            number="2", title="B", ich_code="B.3",
            confidence=MatchConfidence.REVIEW,
        )
        idx = _make_protocol_index(
            content_spans={"1": "aaa", "2": "bbb"}
        )
        mr = _make_match_result([m1, m2])

        routes = QueryExtractor._build_routes(idx, mr)

        assert routes["B.3"][2] == MatchConfidence.REVIEW


# -----------------------------------------------------------------------
# PTCV-182: Front-matter injection resilience tests
# -----------------------------------------------------------------------


class TestPTCV182FrontMatterResilience:
    """Validate front-matter injection when B.1 is contaminated."""

    def test_front_matter_injection_when_b1_contaminated(
        self,
    ) -> None:
        """B.1 REVIEW with >=3 source sections → front-matter
        injected."""
        # Simulate contaminated B.1 route (4 source sections)
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.1": (
                "garbage concatenated content",
                "5.2.1, 6.3.3, 7.1.1, 8.1.2",
                MatchConfidence.REVIEW,
            ),
        }
        # Build protocol index with front-matter
        front_text = "A" * 200  # 200 chars of front matter
        full_text = front_text + "\n## 1. Introduction\nBody text."
        headers = [
            SectionHeader(
                number="1",
                title="Introduction",
                level=1,
                page=1,
                char_offset=201,
            ),
        ]
        idx = ProtocolIndex(
            source_path="test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=headers,
            content_spans={"1": "Body text."},
            full_text=full_text,
            toc_found=True,
            toc_pages=[1],
        )

        QueryExtractor._inject_front_matter_route(routes, idx)

        # Front-matter should have been injected (prepended)
        assert routes["B.1"][1].startswith("front_matter")
        assert routes["B.1"][2] == MatchConfidence.REVIEW

    def test_front_matter_skipped_when_b1_genuine(self) -> None:
        """B.1 HIGH with 1 source section → skip injection."""
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.1": (
                "Real B.1 content about protocol title",
                "1",
                MatchConfidence.HIGH,
            ),
        }
        headers = [
            SectionHeader(
                number="1",
                title="Introduction",
                level=1,
                page=1,
                char_offset=500,
            ),
        ]
        idx = ProtocolIndex(
            source_path="test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=headers,
            content_spans={"1": "Introduction content."},
            full_text="A" * 500 + "\n## 1. Introduction\nContent.",
            toc_found=True,
            toc_pages=[1],
        )

        QueryExtractor._inject_front_matter_route(routes, idx)

        # Should NOT have been modified
        assert routes["B.1"][1] == "1"
        assert routes["B.1"][2] == MatchConfidence.HIGH


# -----------------------------------------------------------------------
# PTCV-183: LLM refusal detection tests
# -----------------------------------------------------------------------


class TestPTCV183RefusalDetection:
    """Validate refusal detection in _transform_content (PTCV-183)."""

    def test_is_refusal_detects_common_patterns(self) -> None:
        """Known refusal prefixes must be detected."""
        assert _is_refusal("I cannot process this content")
        assert _is_refusal("I can't provide medical advice")
        assert _is_refusal("I'm sorry, but I cannot")
        assert _is_refusal("I apologize, I'm unable to")
        assert _is_refusal("As an AI, I cannot")
        assert _is_refusal("I'm unable to assist with")
        assert _is_refusal("Unfortunately, I cannot process")
        assert _is_refusal("I do not have access to")
        assert _is_refusal("Content not available for this")

    def test_is_refusal_case_insensitive(self) -> None:
        """Refusal detection must be case-insensitive."""
        assert _is_refusal("I CANNOT process this content")
        assert _is_refusal("I Can't provide this")
        assert _is_refusal("I'M SORRY, but I cannot")

    def test_is_refusal_rejects_legitimate_content(self) -> None:
        """Legitimate clinical content must not trigger refusal."""
        assert not _is_refusal(
            "The study design is a randomized controlled trial"
        )
        assert not _is_refusal(
            "Patients will be enrolled in two arms"
        )
        assert not _is_refusal(
            "This protocol cannot be modified without IRB"
        )
        assert not _is_refusal("")

    def test_is_refusal_inspects_only_prefix(self) -> None:
        """Refusal phrases deep in text must not trigger."""
        long_text = (
            "The study follows ICH E6 guidelines. " * 10
            + "I cannot process this content."
        )
        assert not _is_refusal(long_text)

    def test_refusal_patterns_all_lowercase(self) -> None:
        """All patterns in _REFUSAL_PATTERNS must be lowercase."""
        for pattern in _REFUSAL_PATTERNS:
            assert pattern == pattern.lower(), (
                f"Pattern not lowercase: {pattern!r}"
            )

    def test_transform_content_rejects_refusal_json(
        self,
    ) -> None:
        """_transform_content must return None when transformed
        text contains refusal language."""
        extractor = QueryExtractor(
            enable_transformation=False,
        )
        # Force LLM enabled for this test
        extractor._use_llm = True

        query = _make_query(
            query_id="B.4.1.q1",
            section_id="B.4.1",
            parent_section="B.4",
            query_text="What is the study design?",
        )

        # Mock the Anthropic client response with refusal
        mock_resp = MagicMock()
        mock_resp.stop_reason = "end_turn"
        mock_block = MagicMock()
        mock_block.text = (
            '{"transformed": "I cannot process this '
            'clinical content", "confidence": 0.5}'
        )
        mock_resp.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        extractor._client = mock_client

        result = extractor._transform_content(
            "Study is a Phase 3 RCT", query
        )
        assert result is None
        assert extractor._refusal_count == 1

    def test_transform_content_rejects_bad_stop_reason(
        self,
    ) -> None:
        """_transform_content must return None for unexpected
        stop_reason values."""
        extractor = QueryExtractor(
            enable_transformation=False,
        )
        extractor._use_llm = True

        query = _make_query(
            query_id="B.4.2.q1",
            section_id="B.4.2",
            parent_section="B.4",
            query_text="What is the primary endpoint?",
        )

        mock_resp = MagicMock()
        mock_resp.stop_reason = "content_filtered"
        mock_block = MagicMock()
        mock_block.text = '{"transformed": "", "confidence": 0}'
        mock_resp.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        extractor._client = mock_client

        result = extractor._transform_content(
            "Primary endpoint is OS", query
        )
        assert result is None
        assert extractor._refusal_count == 1

    def test_transform_content_accepts_legitimate(
        self,
    ) -> None:
        """_transform_content must accept valid transformed text."""
        extractor = QueryExtractor(
            enable_transformation=False,
        )
        extractor._use_llm = True

        query = _make_query(
            query_id="B.4.1.q1",
            section_id="B.4.1",
            parent_section="B.4",
            query_text="What is the study design?",
        )

        mock_resp = MagicMock()
        mock_resp.stop_reason = "end_turn"
        mock_block = MagicMock()
        mock_block.text = (
            '{"transformed": "This is a Phase 3 '
            'randomized controlled trial.", '
            '"confidence": 0.92}'
        )
        mock_resp.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        extractor._client = mock_client

        result = extractor._transform_content(
            "Study is a Phase 3 RCT", query
        )
        assert result is not None
        text, conf = result
        assert "Phase 3" in text
        assert conf == pytest.approx(0.92)
        assert extractor._refusal_count == 0

    @patch("ptcv.ich_parser.query_extractor.time.sleep")
    def test_transform_retries_on_rate_limit(
        self, mock_sleep: MagicMock,
    ) -> None:
        """_transform_content retries on rate-limit errors."""
        extractor = QueryExtractor(
            enable_transformation=False,
        )
        extractor._use_llm = True

        query = _make_query(
            query_id="B.4.1.q1",
            section_id="B.4.1",
            parent_section="B.4",
            query_text="What is the study design?",
        )

        # First call raises rate limit, second succeeds
        rate_err = Exception("rate limit exceeded")
        rate_err.status_code = 429  # type: ignore[attr-defined]

        mock_resp = MagicMock()
        mock_resp.stop_reason = "end_turn"
        mock_block = MagicMock()
        mock_block.text = (
            '{"transformed": "Phase 3 RCT.", '
            '"confidence": 0.9}'
        )
        mock_resp.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            rate_err, mock_resp,
        ]
        extractor._client = mock_client

        result = extractor._transform_content(
            "Study is a Phase 3 RCT", query
        )
        assert result is not None
        assert "Phase 3" in result[0]
        assert mock_client.messages.create.call_count == 2
        mock_sleep.assert_called_once()

    @patch("ptcv.ich_parser.query_extractor.time.sleep")
    def test_transform_no_retry_on_non_retryable(
        self, mock_sleep: MagicMock,
    ) -> None:
        """_transform_content does not retry non-retryable errors."""
        extractor = QueryExtractor(
            enable_transformation=False,
        )
        extractor._use_llm = True

        query = _make_query(
            query_id="B.4.1.q1",
            section_id="B.4.1",
            parent_section="B.4",
            query_text="What is the study design?",
        )

        auth_err = Exception("authentication failed")
        auth_err.status_code = 401  # type: ignore[attr-defined]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = auth_err
        extractor._client = mock_client

        result = extractor._transform_content(
            "Study is a Phase 3 RCT", query
        )
        assert result is None
        assert mock_client.messages.create.call_count == 1
        mock_sleep.assert_not_called()
