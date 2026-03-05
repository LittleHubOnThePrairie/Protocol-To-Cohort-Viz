"""Tests for query-driven extraction engine (PTCV-91)."""

from __future__ import annotations

import pytest

from ptcv.ich_parser.query_extractor import (
    ExtractionGap,
    ExtractionResult,
    QueryExtraction,
    QueryExtractor,
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
    """Long text passthrough tests."""

    def test_passthrough_with_keywords(self) -> None:
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
