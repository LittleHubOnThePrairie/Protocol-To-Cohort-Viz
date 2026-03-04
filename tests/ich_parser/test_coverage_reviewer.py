"""Tests for CoverageReviewer — deterministic text overlap (PTCV-60)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.coverage_reviewer import (
    CoverageResult,
    CoverageReviewer,
    UncoveredBlock,
)
from ptcv.ich_parser.models import IchSection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(code: str, text: str) -> IchSection:
    """Build a minimal IchSection with the given text."""
    return IchSection(
        run_id="r1",
        source_run_id="",
        source_sha256="a" * 64,
        registry_id="NCT0001",
        section_code=code,
        section_name=f"Section {code}",
        content_json=json.dumps({"text_excerpt": text}),
        confidence_score=0.90,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
    )


def _make_blocks(texts: list[tuple[int, str]]) -> list[dict]:
    """Build text blocks from (page, text) pairs."""
    return [{"page_number": p, "text": t} for p, t in texts]


# ---------------------------------------------------------------------------
# Perfect coverage
# ---------------------------------------------------------------------------


class TestPerfectCoverage:
    def test_identical_text_scores_one(self):
        text = "This is a sufficiently long sentence for coverage testing purposes."
        blocks = _make_blocks([(1, text)])
        sections = [_make_section("B.1", text)]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert result.coverage_score >= 0.99
        assert result.passed is True
        assert result.uncovered_blocks == []

    def test_all_sentences_covered(self):
        text = (
            "First sentence with enough characters to pass minimum length. "
            "Second sentence also long enough for the coverage check."
        )
        blocks = _make_blocks([(1, text)])
        sections = [_make_section("B.1", text)]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert result.coverage_score >= 0.99
        assert result.passed is True


# ---------------------------------------------------------------------------
# Zero coverage
# ---------------------------------------------------------------------------


class TestZeroCoverage:
    def test_no_overlap_scores_zero(self):
        blocks = _make_blocks([
            (1, "This original sentence is not found in any section at all."),
        ])
        sections = [_make_section("B.1", "Completely different content here.")]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert result.coverage_score < 0.50
        assert result.passed is False
        assert len(result.uncovered_blocks) >= 1


# ---------------------------------------------------------------------------
# Partial coverage
# ---------------------------------------------------------------------------


class TestPartialCoverage:
    def test_half_covered_scores_around_half(self):
        covered = "This sentence appears in the retemplated section output."
        uncovered = "This other sentence does not appear anywhere at all really."
        blocks = _make_blocks([
            (1, covered),
            (2, uncovered),
        ])
        sections = [_make_section("B.1", covered)]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert 0.30 < result.coverage_score < 0.70
        assert len(result.uncovered_blocks) >= 1


# ---------------------------------------------------------------------------
# Boilerplate exclusion
# ---------------------------------------------------------------------------


class TestBoilerplateExclusion:
    def test_page_numbers_excluded(self):
        blocks = _make_blocks([
            (1, "Page 1"),
            (1, "Page 2"),
            (1, "This is a sufficiently long sentence for coverage testing purposes."),
        ])
        sections = [_make_section("B.1",
            "This is a sufficiently long sentence for coverage testing purposes.")]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert result.coverage_score >= 0.99

    def test_confidential_excluded(self):
        blocks = _make_blocks([
            (1, "Confidential"),
            (1, "This is real protocol content that matters for coverage analysis."),
        ])
        sections = [_make_section("B.1",
            "This is real protocol content that matters for coverage analysis.")]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert result.coverage_score >= 0.99


# ---------------------------------------------------------------------------
# Short fragments
# ---------------------------------------------------------------------------


class TestShortFragments:
    def test_fragments_under_min_len_auto_covered(self):
        blocks = _make_blocks([
            (1, "Short."),
            (1, "Also tiny."),
            (1, "This is a sufficiently long sentence for coverage testing purposes."),
        ])
        sections = [_make_section("B.1",
            "This is a sufficiently long sentence for coverage testing purposes.")]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert result.coverage_score >= 0.99


# ---------------------------------------------------------------------------
# Uncovered blocks
# ---------------------------------------------------------------------------


class TestUncoveredBlocks:
    def test_uncovered_block_has_page_and_text(self):
        blocks = _make_blocks([
            (5, "This sentence only appears in the original text and not in sections."),
        ])
        sections = [_make_section("B.1", "Completely unrelated section content here.")]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert len(result.uncovered_blocks) >= 1
        block = result.uncovered_blocks[0]
        assert isinstance(block, UncoveredBlock)
        assert block.page_number == 5
        assert block.char_count > 0


# ---------------------------------------------------------------------------
# Section coverage breakdown
# ---------------------------------------------------------------------------


class TestSectionCoverage:
    def test_section_coverage_dict_populated(self):
        text = "This sentence is classified into section B.3 with enough text."
        blocks = _make_blocks([(1, text)])
        sections = [_make_section("B.3", text)]

        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)

        assert "B.3" in result.section_coverage
        assert result.section_coverage["B.3"] > 0


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_text_blocks_scores_one(self):
        reviewer = CoverageReviewer()
        result = reviewer.review([], [_make_section("B.1", "content")])

        assert result.coverage_score == 1.0
        assert result.passed is True

    def test_empty_sections_scores_based_on_overlap(self):
        blocks = _make_blocks([
            (1, "Some original text that cannot be found anywhere in zero sections."),
        ])
        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, [])

        assert result.coverage_score < 1.0


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_custom_threshold(self):
        text = "This sentence is in both the original and the retemplated output."
        blocks = _make_blocks([(1, text)])
        sections = [_make_section("B.1", text)]

        reviewer = CoverageReviewer(pass_threshold=0.95)
        result = reviewer.review(blocks, sections)

        assert result.pass_threshold == 0.95
        assert result.passed is True

    def test_result_type(self):
        reviewer = CoverageReviewer()
        result = reviewer.review([], [])
        assert isinstance(result, CoverageResult)
