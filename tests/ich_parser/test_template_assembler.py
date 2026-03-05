"""Tests for ICH E6(R3) Appendix B template assembler (PTCV-92).

Feature: Template assembly into ICH E6(R3) Appendix B format

  Scenario: Extracted content populates correct template sections
  Scenario: Gap sections are flagged clearly
  Scenario: Source traceability preserved
  Scenario: Coverage report summarizes completeness
  Scenario: Output in multiple formats
"""

from __future__ import annotations

import json

import pytest

from ptcv.ich_parser.query_schema import load_query_schema, _reset_cache
from ptcv.ich_parser.template_assembler import (
    APPENDIX_B_SECTION_NAMES,
    GAP_PLACEHOLDER,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    AssembledProtocol,
    AssembledSection,
    CoverageReport,
    QueryExtractionHit,
    SourceReference,
    assemble_template,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset query cache before each test."""
    _reset_cache()
    yield
    _reset_cache()


# -----------------------------------------------------------------------
# Helpers — build hits for a subset of sections
# -----------------------------------------------------------------------


def _make_hit(
    query_id: str,
    section_id: str,
    parent: str,
    content: str = "Extracted text.",
    confidence: float = 0.90,
    page: int = 0,
    header: str = "",
) -> QueryExtractionHit:
    return QueryExtractionHit(
        query_id=query_id,
        section_id=section_id,
        parent_section=parent,
        query_text=f"Query for {query_id}",
        extracted_content=content,
        confidence=confidence,
        source=SourceReference(
            pdf_page=page,
            section_header=header,
            char_offset_start=100,
            char_offset_end=500,
        ),
    )


def _make_section_hits(parent: str, **kwargs) -> list[QueryExtractionHit]:
    """Create hits for all queries in a parent section."""
    queries = load_query_schema()
    section_queries = [q for q in queries if q.parent_section == parent]
    return [
        _make_hit(
            q.query_id, q.section_id, parent, **kwargs,
        )
        for q in section_queries
    ]


# -----------------------------------------------------------------------
# Scenario: Extracted content populates correct template sections
# -----------------------------------------------------------------------

class TestPopulatesSections:
    """Given an ExtractionResult with content for B.1, B.2, B.6, B.7,
    B.8, B.10,
    When the template assembler runs,
    Then the output contains all populated sections in Appendix B order
    (B.1 before B.2 before B.6...)
    And each section content matches the extraction.
    """

    def test_sections_in_appendix_b_order(self):
        hits = (
            _make_section_hits("B.1")
            + _make_section_hits("B.2")
            + _make_section_hits("B.6")
            + _make_section_hits("B.7")
            + _make_section_hits("B.8")
            + _make_section_hits("B.10")
        )
        result = assemble_template(hits)

        codes = [s.section_code for s in result.sections]
        # All 16 sections present, in B.1-B.16 order
        assert codes[0] == "B.1"
        assert codes[1] == "B.10"  # sorted alphanumerically
        # Verify populated ones appear and are marked populated
        populated_codes = {s.section_code for s in result.sections if s.populated}
        assert {"B.1", "B.2", "B.6", "B.7", "B.8", "B.10"} == populated_codes

    def test_content_matches_extraction(self):
        hits = [_make_hit("B.1.1.q1", "B.1.1", "B.1", content="Protocol Title")]
        result = assemble_template(hits)
        b1 = result.get_section("B.1")
        assert b1 is not None
        assert b1.populated is True
        assert b1.hits[0].extracted_content == "Protocol Title"

    def test_all_16_sections_present(self):
        result = assemble_template([])
        assert len(result.sections) == 16

    def test_section_names_populated(self):
        result = assemble_template([])
        for s in result.sections:
            assert s.section_name != ""
            assert s.section_code in APPENDIX_B_SECTION_NAMES


# -----------------------------------------------------------------------
# Scenario: Gap sections are flagged clearly
# -----------------------------------------------------------------------

class TestGapSections:
    """Given an ExtractionResult missing content for B.9,
    When the template assembler runs,
    Then section B.9 appears in the output with a gap placeholder,
    And the coverage report lists B.9 as a gap.
    """

    def test_missing_section_is_gap(self):
        # Populate B.1 only, leaving B.9 as a gap
        hits = _make_section_hits("B.1")
        result = assemble_template(hits)
        b9 = result.get_section("B.9")
        assert b9 is not None
        assert b9.is_gap is True
        assert b9.populated is False
        assert len(b9.hits) == 0

    def test_gap_in_coverage_report(self):
        hits = _make_section_hits("B.1")
        result = assemble_template(hits)
        assert "B.9" in result.coverage.gap_sections

    def test_gap_placeholder_in_markdown(self):
        hits = _make_section_hits("B.1")
        result = assemble_template(hits)
        md = result.to_markdown()
        assert GAP_PLACEHOLDER in md

    def test_no_gaps_when_fully_populated(self):
        queries = load_query_schema()
        hits = [
            _make_hit(q.query_id, q.section_id, q.parent_section)
            for q in queries
        ]
        result = assemble_template(hits)
        assert result.coverage.gap_count == 0
        assert result.coverage.gap_sections == []


# -----------------------------------------------------------------------
# Scenario: Source traceability preserved
# -----------------------------------------------------------------------

class TestSourceTraceability:
    """Given an assembled protocol section B.6,
    When the source_traceability is queried,
    Then it returns the original PDF page numbers and section header,
    And the character offset range in the source document.
    """

    def test_traceability_has_page_and_header(self):
        hits = [
            _make_hit(
                "B.6.1.q1", "B.6.1", "B.6",
                page=42, header="6.1 Description of Treatment",
            ),
        ]
        result = assemble_template(hits)
        assert "B.6" in result.source_traceability
        refs = result.source_traceability["B.6"]
        assert len(refs) == 1
        assert refs[0].pdf_page == 42
        assert refs[0].section_header == "6.1 Description of Treatment"

    def test_traceability_has_char_offsets(self):
        hits = [
            _make_hit(
                "B.6.1.q1", "B.6.1", "B.6",
                page=42, header="6.1 Description",
            ),
        ]
        result = assemble_template(hits)
        refs = result.source_traceability["B.6"]
        assert refs[0].char_offset_start == 100
        assert refs[0].char_offset_end == 500

    def test_no_traceability_when_no_page(self):
        hits = [_make_hit("B.1.1.q1", "B.1.1", "B.1", page=0)]
        result = assemble_template(hits)
        assert "B.1" not in result.source_traceability

    def test_traceability_in_json_output(self):
        hits = [
            _make_hit(
                "B.6.1.q1", "B.6.1", "B.6",
                page=42, header="6.1 Treatment",
            ),
        ]
        result = assemble_template(hits)
        d = result.to_dict()
        assert "source_traceability" in d
        assert "B.6" in d["source_traceability"]
        ref = d["source_traceability"]["B.6"][0]
        assert ref["pdf_page"] == 42

    def test_traceability_in_markdown(self):
        hits = [
            _make_hit(
                "B.6.1.q1", "B.6.1", "B.6",
                page=42, header="6.1 Treatment",
            ),
        ]
        result = assemble_template(hits)
        md = result.to_markdown()
        assert "Page 42" in md
        assert "6.1 Treatment" in md


# -----------------------------------------------------------------------
# Scenario: Coverage report summarizes completeness
# -----------------------------------------------------------------------

class TestCoverageReport:
    """Given an assembled protocol,
    When the coverage_report is generated,
    Then it shows total sections, populated count, gap count, avg confidence,
    And sections are categorized as high/medium/low confidence.
    """

    def test_basic_counts(self):
        hits = _make_section_hits("B.1") + _make_section_hits("B.2")
        result = assemble_template(hits)
        c = result.coverage
        assert c.total_sections == 16
        assert c.populated_count == 2
        assert c.gap_count == 14

    def test_average_confidence(self):
        hits = [
            _make_hit("B.1.1.q1", "B.1.1", "B.1", confidence=0.90),
            _make_hit("B.2.1.q1", "B.2.1", "B.2", confidence=0.80),
        ]
        result = assemble_template(hits)
        assert result.coverage.average_confidence == 0.85

    def test_confidence_categories(self):
        hits = [
            _make_hit("B.1.1.q1", "B.1.1", "B.1", confidence=0.95),
            _make_hit("B.2.1.q1", "B.2.1", "B.2", confidence=0.75),
            _make_hit("B.3.q1", "B.3", "B.3", confidence=0.50),
        ]
        result = assemble_template(hits)
        c = result.coverage
        assert c.high_confidence_count == 1   # B.1 at 0.95
        assert c.medium_confidence_count == 1  # B.2 at 0.75
        assert c.low_confidence_count == 1     # B.3 at 0.50

    def test_query_counts(self):
        queries = load_query_schema()
        all_hits = [
            _make_hit(q.query_id, q.section_id, q.parent_section)
            for q in queries
        ]
        result = assemble_template(all_hits)
        c = result.coverage
        assert c.total_queries == len(queries)
        assert c.answered_queries == len(queries)
        assert c.required_queries > 0
        assert c.answered_required == c.required_queries

    def test_low_confidence_sections_listed(self):
        hits = [
            _make_hit("B.3.q1", "B.3", "B.3", confidence=0.40),
        ]
        result = assemble_template(hits)
        assert "B.3" in result.coverage.low_confidence_sections

    def test_empty_protocol_all_gaps(self):
        result = assemble_template([])
        c = result.coverage
        assert c.populated_count == 0
        assert c.gap_count == 16
        assert c.average_confidence == 0.0
        assert len(c.gap_sections) == 16


# -----------------------------------------------------------------------
# Scenario: Output in multiple formats
# -----------------------------------------------------------------------

class TestOutputFormats:
    """Given an assembled protocol,
    When JSON and Markdown outputs are generated,
    Then both contain the same section content,
    And Markdown includes human-readable headers and gap flags.
    """

    def test_json_sections_match_markdown_sections(self):
        hits = _make_section_hits("B.1") + _make_section_hits("B.4")
        result = assemble_template(hits)

        d = result.to_dict()
        md = result.to_markdown()

        # JSON has all 16 sections
        assert len(d["sections"]) == 16

        # Markdown has all section headers
        for s in result.sections:
            assert f"## {s.section_code}" in md

    def test_json_is_serializable(self):
        hits = _make_section_hits("B.1")
        result = assemble_template(hits)
        d = result.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_json_has_format_field(self):
        result = assemble_template([])
        d = result.to_dict()
        assert d["format"] == "ICH E6(R3) Appendix B"

    def test_json_coverage_matches(self):
        hits = _make_section_hits("B.1")
        result = assemble_template(hits)
        d = result.to_dict()
        assert d["coverage"]["total_sections"] == 16
        assert d["coverage"]["populated_count"] == 1

    def test_markdown_has_gap_flags(self):
        result = assemble_template([])
        md = result.to_markdown()
        # All sections are gaps, so placeholder should appear
        assert md.count(GAP_PLACEHOLDER) == 16

    def test_markdown_has_confidence_flags(self):
        hits = [
            _make_hit("B.1.1.q1", "B.1.1", "B.1", confidence=0.50),
        ]
        result = assemble_template(hits)
        md = result.to_markdown()
        assert "LOW CONFIDENCE" in md

    def test_markdown_moderate_confidence_flag(self):
        hits = [
            _make_hit("B.1.1.q1", "B.1.1", "B.1", confidence=0.75),
        ]
        result = assemble_template(hits)
        md = result.to_markdown()
        assert "moderate confidence" in md

    def test_markdown_coverage_summary(self):
        hits = _make_section_hits("B.1")
        result = assemble_template(hits)
        md = result.to_markdown()
        assert "Coverage Summary" in md
        assert "Sections populated:" in md
        assert "Gap sections:" in md


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Additional edge case tests."""

    def test_multiple_hits_per_section(self):
        hits = [
            _make_hit("B.1.1.q1", "B.1.1", "B.1", content="Title"),
            _make_hit("B.1.1.q2", "B.1.1", "B.1", content="Number"),
        ]
        result = assemble_template(hits)
        b1 = result.get_section("B.1")
        assert b1 is not None
        assert len(b1.hits) == 2

    def test_get_section_nonexistent(self):
        result = assemble_template([])
        assert result.get_section("B.99") is None

    def test_low_confidence_flag_on_section(self):
        hits = [
            _make_hit("B.1.1.q1", "B.1.1", "B.1", confidence=0.95),
            _make_hit("B.1.1.q2", "B.1.1", "B.1", confidence=0.50),
        ]
        result = assemble_template(hits)
        b1 = result.get_section("B.1")
        assert b1 is not None
        assert b1.has_low_confidence is True

    def test_required_query_tracking(self):
        queries = load_query_schema()
        b1_queries = [q for q in queries if q.parent_section == "B.1"]
        required_count = sum(1 for q in b1_queries if q.required)

        # Answer only one required query
        required_q = next(q for q in b1_queries if q.required)
        hits = [_make_hit(required_q.query_id, required_q.section_id, "B.1")]
        result = assemble_template(hits)
        b1 = result.get_section("B.1")
        assert b1 is not None
        assert b1.required_query_count == required_count
        assert b1.answered_required_count == 1
