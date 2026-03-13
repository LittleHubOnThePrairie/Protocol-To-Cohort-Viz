"""Tests for classified section assembly (PTCV-165).

Feature: ICH E6(R3) document assembly from classified pipeline output

  Scenario: Classified sections are assembled in canonical order
  Scenario: Missing required sections are detected
  Scenario: Markdown tables preserved in assembled output
  Scenario: Assembly output includes full provenance
  Scenario: Markdown output matches legacy format structure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.template_assembler import (
    APPENDIX_B_SECTION_NAMES,
    GAP_PLACEHOLDER,
    AssembledProtocol,
    ClassifiedSection,
    assemble_from_classified,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _cs(
    code: str,
    content: str = "",
    confidence: float = 0.90,
    extraction_method: str = "",
    classification_method: str = "",
    source_page: int = 0,
    source_header: str = "",
) -> ClassifiedSection:
    """Build a ClassifiedSection with defaults."""
    if not content:
        content = f"Sample content for section {code}."
    return ClassifiedSection(
        section_code=code,
        section_name=APPENDIX_B_SECTION_NAMES.get(code, code),
        content_text=content,
        confidence=confidence,
        extraction_method=extraction_method,
        classification_method=classification_method,
        source_page=source_page,
        source_header=source_header,
    )


# -------------------------------------------------------------------
# Scenario 1: Classified sections assembled in canonical order
# -------------------------------------------------------------------


class TestCanonicalOrder:
    """Given sections classified as B.1, B.3, B.5, B.4,
    When assembly runs,
    Then sections are ordered B.1, B.3, B.4, B.5."""

    def test_out_of_order_input_sorted_canonically(self):
        sections = [_cs("B.1"), _cs("B.3"), _cs("B.5"), _cs("B.4")]
        result = assemble_from_classified(sections)
        populated = [
            s.section_code for s in result.sections if s.populated
        ]
        assert populated == ["B.1", "B.3", "B.4", "B.5"]

    def test_all_16_sections_present_in_output(self):
        sections = [_cs("B.1")]
        result = assemble_from_classified(sections)
        assert len(result.sections) == 16

    def test_gap_sections_fill_missing_codes(self):
        sections = [_cs("B.1"), _cs("B.10")]
        result = assemble_from_classified(sections)
        gaps = {s.section_code for s in result.sections if s.is_gap}
        assert "B.2" in gaps
        assert "B.9" in gaps
        assert "B.1" not in gaps
        assert "B.10" not in gaps

    def test_b9_before_b10_natural_order(self):
        result = assemble_from_classified([])
        codes = [s.section_code for s in result.sections]
        assert codes.index("B.9") < codes.index("B.10")


# -------------------------------------------------------------------
# Scenario 2: Missing required sections are detected
# -------------------------------------------------------------------


class TestMissingRequiredSections:
    """Given classified sections that do not include B.5,
    When assembly runs,
    Then a gap warning reports 'B.5 (Study Population) is missing'."""

    def test_b5_missing_detected_as_gap(self):
        sections = [_cs("B.1"), _cs("B.3"), _cs("B.4")]
        result = assemble_from_classified(sections)
        b5 = result.get_section("B.5")
        assert b5 is not None
        assert b5.is_gap is True
        assert "B.5" in result.coverage.gap_sections

    def test_b3_b4_b5_all_missing_all_flagged(self):
        sections = [_cs("B.1"), _cs("B.6")]
        result = assemble_from_classified(sections)
        gaps = set(result.coverage.gap_sections)
        assert {"B.3", "B.4", "B.5"}.issubset(gaps)

    def test_all_required_present_no_gaps_in_required(self):
        sections = [_cs("B.3"), _cs("B.4"), _cs("B.5")]
        result = assemble_from_classified(sections)
        required_gaps = {"B.3", "B.4", "B.5"} & set(
            result.coverage.gap_sections,
        )
        assert required_gaps == set()

    def test_coverage_required_counts(self):
        sections = [_cs("B.3"), _cs("B.5")]  # B.4 missing
        result = assemble_from_classified(sections)
        assert result.coverage.required_queries == 3
        assert result.coverage.answered_required == 2

    def test_missing_required_warning_logged(self, caplog):
        sections = [_cs("B.1")]
        with caplog.at_level("WARNING"):
            assemble_from_classified(sections)
        assert "B.5" in caplog.text
        assert "missing" in caplog.text.lower()


# -------------------------------------------------------------------
# Scenario 3: Markdown tables preserved in assembled output
# -------------------------------------------------------------------


TABLE_MD = """\
| Visit | Week | Assessment |
|-------|------|------------|
| V1    | 0    | Screening  |
| V2    | 4    | Treatment  |
| V3    | 12   | Follow-up  |"""


class TestMarkdownTablePreservation:
    """Given a B.8 section containing Markdown pipe tables,
    When assembly produces retemplated_protocol.md,
    Then the Markdown table syntax is preserved."""

    def test_table_pipes_preserved(self):
        sections = [_cs("B.8", content=TABLE_MD)]
        result = assemble_from_classified(sections)
        md = result.to_markdown()
        assert "| Visit | Week | Assessment |" in md
        assert "|-------|------|------------|" in md
        assert "| V1    | 0    | Screening  |" in md

    def test_heading_formatting_preserved(self):
        content = "## Efficacy Endpoints\n\n### Primary\n\nORR by RECIST"
        sections = [_cs("B.8", content=content)]
        result = assemble_from_classified(sections)
        md = result.to_markdown()
        assert "## Efficacy Endpoints" in md
        assert "### Primary" in md

    def test_list_formatting_preserved(self):
        content = "- Inclusion criterion 1\n- Inclusion criterion 2\n  - Sub-item"
        sections = [_cs("B.5", content=content)]
        result = assemble_from_classified(sections)
        md = result.to_markdown()
        assert "- Inclusion criterion 1" in md
        assert "  - Sub-item" in md

    def test_table_in_json_output(self):
        sections = [_cs("B.8", content=TABLE_MD)]
        result = assemble_from_classified(sections)
        d = result.to_dict()
        b8 = next(
            s for s in d["sections"] if s["section_code"] == "B.8"
        )
        assert "| Visit |" in b8["hits"][0]["extracted_content"]


# -------------------------------------------------------------------
# Scenario 4: Assembly output includes full provenance
# -------------------------------------------------------------------


class TestProvenance:
    """Given a section extracted by Docling, classified by NeoBERT
    at confidence 0.82,
    When assembly produces output,
    Then metadata includes extraction_method, classification_method,
    confidence."""

    def test_provenance_on_assembled_section(self):
        sections = [
            _cs(
                "B.4",
                confidence=0.82,
                extraction_method="E2:docling",
                classification_method="C2:neobert_sonnet",
            ),
        ]
        result = assemble_from_classified(sections)
        b4 = result.get_section("B.4")
        assert b4 is not None
        assert b4.extraction_method == "E2:docling"
        assert b4.classification_method == "C2:neobert_sonnet"
        assert b4.average_confidence == 0.82

    def test_provenance_in_json_dict(self):
        sections = [
            _cs(
                "B.4",
                extraction_method="E2:docling",
                classification_method="cascade:local",
            ),
        ]
        result = assemble_from_classified(sections)
        d = result.to_dict()
        b4 = next(
            s for s in d["sections"] if s["section_code"] == "B.4"
        )
        assert b4["extraction_method"] == "E2:docling"
        assert b4["classification_method"] == "cascade:local"

    def test_provenance_in_markdown(self):
        sections = [
            _cs(
                "B.4",
                confidence=0.82,
                extraction_method="E3:pdfplumber",
                classification_method="C2:neobert_sonnet",
            ),
        ]
        result = assemble_from_classified(sections)
        md = result.to_markdown()
        assert "Extraction: E3:pdfplumber" in md
        assert "Classification: C2:neobert_sonnet" in md
        assert "Confidence: 0.82" in md

    def test_no_provenance_for_query_path(self):
        """Legacy query-pipeline output has no provenance metadata."""
        sections = [_cs("B.1")]
        result = assemble_from_classified(sections)
        b1 = result.get_section("B.1")
        assert b1 is not None
        assert b1.extraction_method == ""
        assert b1.classification_method == ""

    def test_provenance_absent_from_json_when_empty(self):
        sections = [_cs("B.1")]
        result = assemble_from_classified(sections)
        d = result.to_dict()
        b1 = next(
            s for s in d["sections"] if s["section_code"] == "B.1"
        )
        assert "extraction_method" not in b1
        assert "classification_method" not in b1

    def test_provenance_absent_from_markdown_when_empty(self):
        sections = [_cs("B.1")]
        result = assemble_from_classified(sections)
        md = result.to_markdown()
        assert "Extraction:" not in md


# -------------------------------------------------------------------
# Scenario 5: Markdown output matches legacy format structure
# -------------------------------------------------------------------


class TestMarkdownLegacyFormat:
    """Given assembled classified sections,
    When rendered to Markdown,
    Then output has same structure as legacy assembler."""

    def test_title_present(self):
        result = assemble_from_classified([])
        md = result.to_markdown()
        assert "# ICH E6(R3) Appendix B" in md

    def test_coverage_summary_block(self):
        sections = [_cs("B.1")]
        result = assemble_from_classified(sections)
        md = result.to_markdown()
        assert "## Coverage Summary" in md
        assert "Sections populated:" in md
        assert "Gap sections:" in md

    def test_gap_placeholder_for_missing(self):
        result = assemble_from_classified([])
        md = result.to_markdown()
        assert md.count(GAP_PLACEHOLDER) == 16

    def test_section_headers_present(self):
        sections = [_cs("B.4")]
        result = assemble_from_classified(sections)
        md = result.to_markdown()
        assert "## B.4 Trial Design" in md
        assert "## B.1 General Information" in md


# -------------------------------------------------------------------
# Deduplication
# -------------------------------------------------------------------


class TestDeduplication:
    """DO NOT duplicate content when classified by both
    local and Sonnet."""

    def test_same_content_hash_deduped(self):
        content = "Identical content from both classifiers."
        s1 = _cs("B.4", content=content,
                  classification_method="cascade:local")
        s2 = _cs("B.4", content=content,
                  classification_method="cascade:sonnet")
        assert s1.content_hash == s2.content_hash

        result = assemble_from_classified([s1, s2])
        b4 = result.get_section("B.4")
        assert b4 is not None
        assert len(b4.hits) == 1

    def test_different_content_both_kept(self):
        s1 = _cs("B.4", content="Content version A.")
        s2 = _cs("B.4", content="Content version B.")
        assert s1.content_hash != s2.content_hash

        result = assemble_from_classified([s1, s2])
        b4 = result.get_section("B.4")
        assert b4 is not None
        assert len(b4.hits) == 2


# -------------------------------------------------------------------
# Factory methods
# -------------------------------------------------------------------


class TestFactoryMethods:
    """Test ClassifiedSection factory methods."""

    def test_from_ich_section(self):
        from ptcv.ich_parser.models import IchSection

        section = IchSection(
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc123",
            registry_id="NCT00000001",
            section_code="B.4",
            section_name="Trial Design",
            content_json='{"text": "Design details"}',
            confidence_score=0.85,
            review_required=False,
            legacy_format=False,
            content_text="Design details about the trial.",
        )
        cs = ClassifiedSection.from_ich_section(
            section,
            extraction_method="E3:pdfplumber",
            classification_method="C3:rulebased_sonnet",
        )
        assert cs.section_code == "B.4"
        assert cs.section_name == "Trial Design"
        assert cs.content_text == "Design details about the trial."
        assert cs.confidence == 0.85
        assert cs.extraction_method == "E3:pdfplumber"
        assert cs.classification_method == "C3:rulebased_sonnet"

    def test_from_ich_section_extracts_page_range(self):
        from ptcv.ich_parser.models import IchSection

        section = IchSection(
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
            registry_id="NCT1",
            section_code="B.1",
            section_name="General",
            content_json=json.dumps({"page_range": [5, 12]}),
            confidence_score=0.90,
            review_required=False,
            legacy_format=False,
            content_text="General info.",
        )
        cs = ClassifiedSection.from_ich_section(section)
        assert cs.source_page == 5

    def test_from_cascade_result(self):
        from ptcv.ich_parser.models import IchSection
        from ptcv.ich_parser.classification_router import (
            CascadeResult,
            RoutingDecision,
            RoutingStats,
            LocalCandidate,
        )

        section = IchSection(
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
            registry_id="NCT1",
            section_code="B.4",
            section_name="Trial Design",
            content_json="{}",
            confidence_score=0.60,
            review_required=True,
            legacy_format=False,
            content_text="Low confidence design text.",
        )
        decision = RoutingDecision(
            block_index=0,
            content_hash="deadbeef",
            local_candidates=[
                LocalCandidate("B.4", "Trial Design", 0.60),
            ],
            route="sonnet",
            threshold_used=0.70,
            final_section_code="B.4",
            final_confidence=0.60,
        )
        cascade = CascadeResult(
            run_id="run-1",
            registry_id="NCT1",
            decisions=[decision],
            stats=RoutingStats(),
            sections=[section],
        )
        result = ClassifiedSection.from_cascade_result(
            cascade, extraction_method="E3:pdfplumber",
        )
        assert len(result) == 1
        assert result[0].section_code == "B.4"
        assert result[0].classification_method == "cascade:sonnet"
        assert result[0].extraction_method == "E3:pdfplumber"

    def test_content_hash_auto_computed(self):
        cs = _cs("B.1", content="Test content")
        assert cs.content_hash != ""
        assert len(cs.content_hash) == 64  # SHA-256 hex

    def test_content_hash_stable(self):
        cs1 = _cs("B.1", content="Stable content")
        cs2 = _cs("B.1", content="Stable content")
        assert cs1.content_hash == cs2.content_hash

    def test_content_hash_empty_for_empty_content(self):
        cs = ClassifiedSection(
            section_code="B.1",
            section_name="General",
            content_text="",
            confidence=0.0,
        )
        assert cs.content_hash == ""


# -------------------------------------------------------------------
# Query bridge compatibility
# -------------------------------------------------------------------


class TestQueryBridgeCompat:
    """Verify query_bridge.assembled_to_sections works with
    classified assembly output."""

    def test_bridge_produces_ich_sections(self):
        from ptcv.soa_extractor.query_bridge import (
            assembled_to_sections,
        )

        sections = [
            _cs("B.1", content="General information content."),
            _cs("B.4", content="Trial design content."),
        ]
        assembled = assemble_from_classified(sections)
        ich_sections = assembled_to_sections(
            assembled,
            registry_id="NCT00000001",
            source_sha256="abc123",
        )
        codes = {s.section_code for s in ich_sections}
        assert "B.1" in codes
        assert "B.4" in codes
        for s in ich_sections:
            assert s.content_text != ""
            assert s.registry_id == "NCT00000001"


# -------------------------------------------------------------------
# Backward compatibility: existing assembler unaffected
# -------------------------------------------------------------------


class TestBackwardCompat:
    """Existing AssembledSection defaults are preserved."""

    def test_assembled_section_default_provenance_empty(self):
        from ptcv.ich_parser.template_assembler import (
            AssembledSection,
            QueryExtractionHit,
        )

        section = AssembledSection(
            section_code="B.1",
            section_name="General Information",
            populated=True,
            hits=[],
            average_confidence=0.90,
            is_gap=False,
            has_low_confidence=False,
            required_query_count=0,
            answered_required_count=0,
        )
        assert section.extraction_method == ""
        assert section.classification_method == ""
