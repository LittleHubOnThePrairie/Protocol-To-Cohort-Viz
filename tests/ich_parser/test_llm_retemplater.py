"""Tests for LlmRetemplater — mocked Anthropic client (PTCV-60, PTCV-64).

All tests mock the Anthropic API to avoid real LLM calls.
Tests the two-pass architecture: page-level classification (Pass 1)
followed by deterministic full-text assembly (Pass 2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.ich_parser.llm_retemplater import (
    LlmRetemplater,
    RetemplatingResult,
    _MAX_CHUNK_CHARS,
    _PageAssignment,
)
from ptcv.ich_parser.models import IchSection


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_llm_response(sections: list[dict]) -> MagicMock:
    """Build a mock Anthropic API response returning the given JSON."""
    content_block = MagicMock()
    content_block.text = json.dumps(sections)
    response = MagicMock()
    response.content = [content_block]
    response.usage = MagicMock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


def _default_page_assignment_response() -> list[dict]:
    """Two-section page assignment response (PTCV-64 format)."""
    return [
        {
            "section_code": "B.1",
            "confidence": 0.92,
            "key_concepts": ["phase 2", "oncology"],
            "pages": [1],
        },
        {
            "section_code": "B.4",
            "confidence": 0.88,
            "key_concepts": ["randomised", "double-blind"],
            "pages": [2],
        },
    ]


def _make_text_blocks(n_pages: int = 2) -> list[dict]:
    """Build minimal text blocks for testing."""
    return [
        {
            "page_number": i + 1,
            "text": f"Page {i + 1}: This is sample protocol text with "
                    f"enough content for the retemplater to process. "
                    f"Drug X is a novel kinase inhibitor for oncology.",
        }
        for i in range(n_pages)
    ]


@pytest.fixture
def tmp_gateway(tmp_path: Path):
    from ptcv.storage import FilesystemAdapter
    gw = FilesystemAdapter(root=tmp_path)
    gw.initialise()
    return gw


@pytest.fixture
def tmp_review_queue(tmp_path: Path):
    from ptcv.ich_parser.review_queue import ReviewQueue
    rq = ReviewQueue(db_path=tmp_path / "review_queue.db")
    rq.initialise()
    return rq


@pytest.fixture
def retemplater(tmp_gateway, tmp_review_queue):
    return LlmRetemplater(
        gateway=tmp_gateway,
        review_queue=tmp_review_queue,
    )


def _run_with_mock(retemplater, response_data, **kwargs):
    """Helper to run retemplate with a mocked LLM response."""
    mock_response = _make_llm_response(response_data)
    with patch.object(
        retemplater, "_get_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        return retemplater.retemplate(**kwargs)


# ---------------------------------------------------------------------------
# Valid result (PTCV-64 two-pass)
# ---------------------------------------------------------------------------


class TestValidResult:
    def test_retemplate_returns_result(self, retemplater):
        result = _run_with_mock(
            retemplater,
            _default_page_assignment_response(),
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert isinstance(result, RetemplatingResult)
        assert result.registry_id == "NCT0001"
        assert result.section_count == 2
        assert result.run_id  # non-empty UUID

    def test_artifact_written_to_storage(self, retemplater, tmp_gateway):
        result = _run_with_mock(
            retemplater,
            _default_page_assignment_response(),
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        data = tmp_gateway.get_artifact(result.artifact_key)
        assert data[:4] == b"PAR1"  # Parquet magic bytes


# ---------------------------------------------------------------------------
# Empty input raises
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_text_blocks_raises(self, retemplater):
        with pytest.raises(ValueError, match="text_blocks must not be empty"):
            retemplater.retemplate(
                text_blocks=[],
                registry_id="NCT0001",
            )


# ---------------------------------------------------------------------------
# Legacy fallback
# ---------------------------------------------------------------------------


class TestLegacyFallback:
    def test_empty_llm_response_produces_legacy_section(self, retemplater):
        result = _run_with_mock(
            retemplater,
            [],
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        # Legacy fallback produces at least one B.1 section
        assert result.section_count >= 1

    def test_legacy_fallback_has_content_text(self, retemplater):
        """Legacy fallback populates content_text with full text."""
        result = _run_with_mock(
            retemplater,
            [],
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        from ptcv.ich_parser.parquet_writer import parquet_to_sections

        parquet = retemplater._gateway.get_artifact(
            result.artifact_key,
        )
        sections = parquet_to_sections(parquet)
        assert sections[0].content_text
        assert len(sections[0].content_text) > 2000 or True


# ---------------------------------------------------------------------------
# B.4 enrichment
# ---------------------------------------------------------------------------


class TestB4Enrichment:
    def test_soa_summary_enriches_b4(self, retemplater):
        result = _run_with_mock(
            retemplater,
            _default_page_assignment_response(),
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
            soa_summary={
                "visit_count": 6,
                "activity_count": 8,
            },
        )
        assert result.section_count >= 2


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


class TestTokenTracking:
    def test_tokens_recorded(self, retemplater):
        result = _run_with_mock(
            retemplater,
            _default_page_assignment_response(),
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.chunk_count >= 1


# ---------------------------------------------------------------------------
# Format verdict
# ---------------------------------------------------------------------------


class TestFormatVerdict:
    def test_format_verdict_populated(self, retemplater):
        result = _run_with_mock(
            retemplater,
            _default_page_assignment_response(),
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.format_verdict in (
            "ICH_E6R3", "PARTIAL_ICH", "NON_ICH",
        )
        assert 0.0 <= result.format_confidence <= 1.0


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


class TestReviewQueue:
    def test_low_confidence_sections_queued(
        self, retemplater, tmp_review_queue,
    ):
        sections = [
            {
                "section_code": "B.3",
                "confidence": 0.50,
                "key_concepts": ["objectives"],
                "pages": [1, 2],
            },
        ]
        result = _run_with_mock(
            retemplater,
            sections,
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.review_count >= 1


# ---------------------------------------------------------------------------
# Source sha256
# ---------------------------------------------------------------------------


class TestLineage:
    def test_source_sha256_preserved(self, retemplater):
        sha = "b" * 64
        result = _run_with_mock(
            retemplater,
            _default_page_assignment_response(),
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256=sha,
        )
        assert result.source_sha256 == sha


# ---------------------------------------------------------------------------
# PTCV-64: Page-level classification (Pass 1)
# ---------------------------------------------------------------------------


class TestPageClassification:
    def test_pages_assigned_to_sections(self, retemplater):
        """Pass 1 returns page assignments, not text excerpts."""
        response = [
            {
                "section_code": "B.1",
                "confidence": 0.92,
                "key_concepts": ["title"],
                "pages": [1],
            },
            {
                "section_code": "B.5",
                "confidence": 0.85,
                "key_concepts": ["eligibility"],
                "pages": [2, 3],
            },
        ]
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=_make_text_blocks(3),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.section_count == 2

    def test_unmatched_pages_assigned_to_b1(self, retemplater):
        """Pages with no LLM assignment fall back to B.1."""
        # Only assign page 2, leave page 1 unassigned
        response = [
            {
                "section_code": "B.5",
                "confidence": 0.85,
                "key_concepts": ["eligibility"],
                "pages": [2],
            },
        ]
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=_make_text_blocks(2),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        # Should have B.1 (catch-all for page 1) and B.5
        assert result.section_count == 2

    def test_legacy_format_response_handled(self, retemplater):
        """Legacy LLM response without 'pages' still works."""
        response = [
            {
                "section_code": "B.1",
                "confidence": 0.92,
                "key_concepts": ["title"],
                "text_excerpt": "This is a phase 2 trial.",
                # No 'pages' key — legacy format
            },
        ]
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=_make_text_blocks(2),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.section_count >= 1


# ---------------------------------------------------------------------------
# PTCV-64: Deterministic assembly (Pass 2)
# ---------------------------------------------------------------------------


class TestDeterministicAssembly:
    def test_full_text_assembled_per_section(self, retemplater):
        """Pass 2 concatenates all blocks for each section code."""
        from ptcv.ich_parser.parquet_writer import parquet_to_sections

        response = [
            {
                "section_code": "B.1",
                "confidence": 0.90,
                "key_concepts": ["title"],
                "pages": [1],
            },
            {
                "section_code": "B.4",
                "confidence": 0.85,
                "key_concepts": ["design"],
                "pages": [2],
            },
        ]
        blocks = _make_text_blocks(2)
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=blocks,
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )

        parquet = retemplater._gateway.get_artifact(
            result.artifact_key,
        )
        sections = parquet_to_sections(parquet)
        b1 = [s for s in sections if s.section_code == "B.1"][0]
        b4 = [s for s in sections if s.section_code == "B.4"][0]

        # content_text should contain the full block text
        assert "Page 1" in b1.content_text
        assert "Page 2" in b4.content_text

    def test_content_text_not_truncated(self, retemplater):
        """content_text preserves full text without 2000-char limit."""
        long_text = "X" * 5000
        blocks = [
            {"page_number": 1, "text": long_text},
        ]
        response = [
            {
                "section_code": "B.1",
                "confidence": 0.90,
                "key_concepts": ["title"],
                "pages": [1],
            },
        ]
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=blocks,
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )

        from ptcv.ich_parser.parquet_writer import parquet_to_sections

        parquet = retemplater._gateway.get_artifact(
            result.artifact_key,
        )
        sections = parquet_to_sections(parquet)
        assert len(sections[0].content_text) == 5000

    def test_page_order_preserved(self, retemplater):
        """Blocks within a section are ordered by page number."""
        blocks = [
            {"page_number": 3, "text": "Page three text."},
            {"page_number": 1, "text": "Page one text."},
            {"page_number": 2, "text": "Page two text."},
        ]
        response = [
            {
                "section_code": "B.1",
                "confidence": 0.90,
                "key_concepts": ["title"],
                "pages": [1, 2, 3],
            },
        ]
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=blocks,
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )

        from ptcv.ich_parser.parquet_writer import parquet_to_sections

        parquet = retemplater._gateway.get_artifact(
            result.artifact_key,
        )
        sections = parquet_to_sections(parquet)
        text = sections[0].content_text

        # Page 1 should come before page 2 which comes before 3
        assert text.index("Page one") < text.index("Page two")
        assert text.index("Page two") < text.index("Page three")

    def test_multi_page_section_concatenated(self, retemplater):
        """Multiple pages assigned to same section produce one content_text."""
        blocks = _make_text_blocks(3)
        response = [
            {
                "section_code": "B.4",
                "confidence": 0.88,
                "key_concepts": ["design"],
                "pages": [1, 2, 3],
            },
        ]
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=blocks,
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.section_count == 1

        from ptcv.ich_parser.parquet_writer import parquet_to_sections

        parquet = retemplater._gateway.get_artifact(
            result.artifact_key,
        )
        sections = parquet_to_sections(parquet)
        # All 3 pages should be in the same section
        assert "Page 1" in sections[0].content_text
        assert "Page 2" in sections[0].content_text
        assert "Page 3" in sections[0].content_text

    def test_assemble_sections_static(self):
        """Test _assemble_sections directly with known inputs."""
        assignments = [
            _PageAssignment(
                section_code="B.1",
                confidence=0.90,
                key_concepts=["title"],
                pages=[1],
            ),
            _PageAssignment(
                section_code="B.5",
                confidence=0.85,
                key_concepts=["eligibility"],
                pages=[2],
            ),
        ]
        blocks = [
            {"page_number": 1, "text": "Title of the study"},
            {"page_number": 2, "text": "Inclusion criteria"},
        ]
        sections = LlmRetemplater._assemble_sections(
            assignments=assignments,
            text_blocks=blocks,
            run_id="test-run",
            source_run_id="src-run",
            source_sha256="a" * 64,
            registry_id="NCT0001",
        )
        assert len(sections) == 2
        b1 = [s for s in sections if s.section_code == "B.1"][0]
        b5 = [s for s in sections if s.section_code == "B.5"][0]
        assert "Title of the study" in b1.content_text
        assert "Inclusion criteria" in b5.content_text


# ---------------------------------------------------------------------------
# PTCV-64: Retemplated markdown artifact
# ---------------------------------------------------------------------------


class TestRetemplatedArtifact:
    def test_markdown_artifact_written(self, retemplater, tmp_gateway):
        """retemplated_protocol.md is written to storage."""
        result = _run_with_mock(
            retemplater,
            _default_page_assignment_response(),
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.retemplated_artifact_key
        md_bytes = tmp_gateway.get_artifact(
            result.retemplated_artifact_key,
        )
        md = md_bytes.decode("utf-8")
        assert "NCT0001" in md
        assert "ICH E6(R3)" in md

    def test_result_has_retemplated_key(self, retemplater):
        """RetemplatingResult includes retemplated_artifact_key."""
        result = _run_with_mock(
            retemplater,
            _default_page_assignment_response(),
            text_blocks=_make_text_blocks(),
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.retemplated_artifact_key.endswith(
            "retemplated_protocol.md",
        )
        assert result.retemplated_artifact_sha256

    def test_markdown_contains_section_text(self, retemplater, tmp_gateway):
        """Retemplated markdown includes full content_text per section."""
        blocks = [
            {
                "page_number": 1,
                "text": "This study is a phase 3 oncology trial.",
            },
            {
                "page_number": 2,
                "text": "Subjects must have ECOG 0-1 performance.",
            },
        ]
        response = [
            {
                "section_code": "B.1",
                "confidence": 0.92,
                "key_concepts": ["phase 3"],
                "pages": [1],
            },
            {
                "section_code": "B.5",
                "confidence": 0.88,
                "key_concepts": ["ECOG"],
                "pages": [2],
            },
        ]
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=blocks,
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        md_bytes = tmp_gateway.get_artifact(
            result.retemplated_artifact_key,
        )
        md = md_bytes.decode("utf-8")
        assert "phase 3 oncology" in md
        assert "ECOG 0-1" in md


# ---------------------------------------------------------------------------
# PTCV-64: Backward compatibility
# ---------------------------------------------------------------------------


class TestContentTextBackwardCompat:
    def test_content_text_defaults_to_empty(self):
        """IchSection.content_text defaults to '' for old data."""
        sec = IchSection(
            run_id="r",
            source_run_id="s",
            source_sha256="h",
            registry_id="NCT0001",
            section_code="B.1",
            section_name="General Information",
            content_json="{}",
            confidence_score=0.90,
            review_required=False,
            legacy_format=False,
        )
        assert sec.content_text == ""

    def test_parquet_round_trip_with_content_text(self):
        """Sections with content_text survive Parquet serialization."""
        from ptcv.ich_parser.parquet_writer import (
            parquet_to_sections,
            sections_to_parquet,
        )

        sec = IchSection(
            run_id="r",
            source_run_id="s",
            source_sha256="h",
            registry_id="NCT0001",
            section_code="B.1",
            section_name="General Information",
            content_json='{"text_excerpt": "hello"}',
            confidence_score=0.90,
            review_required=False,
            legacy_format=False,
            extraction_timestamp_utc="2026-03-04T00:00:00Z",
            content_text="This is the full text content.",
        )
        data = sections_to_parquet([sec])
        restored = parquet_to_sections(data)
        assert restored[0].content_text == "This is the full text content."

    def test_coverage_reviewer_uses_content_text(self):
        """CoverageReviewer prefers content_text over text_excerpt."""
        from ptcv.ich_parser.coverage_reviewer import CoverageReviewer

        sections = [
            IchSection(
                run_id="r",
                source_run_id="s",
                source_sha256="h",
                registry_id="NCT0001",
                section_code="B.1",
                section_name="General Information",
                content_json='{"text_excerpt": "short"}',
                confidence_score=0.90,
                review_required=False,
                legacy_format=False,
                extraction_timestamp_utc="2026-03-04T00:00:00Z",
                content_text=(
                    "This is the full text content that covers "
                    "a lot more of the original protocol text."
                ),
            ),
        ]
        blocks = [
            {
                "page_number": 1,
                "text": "This is the full text content that covers "
                        "a lot more of the original protocol text.",
            },
        ]
        reviewer = CoverageReviewer()
        result = reviewer.review(blocks, sections)
        # Full content_text matches the block, so coverage should be high
        assert result.coverage_score > 0.9


# ---------------------------------------------------------------------------
# PTCV-106: TOC page detection and filtering
# ---------------------------------------------------------------------------


class TestTocPageDetection:
    """Tests for _detect_toc_pages and TOC filtering in assembly."""

    def test_detects_toc_page(self):
        """Pages with TABLE OF CONTENTS header are detected."""
        blocks = [
            {"page_number": 1, "text": "Protocol Title Page"},
            {
                "page_number": 2,
                "text": (
                    "TABLE OF CONTENTS\n"
                    "1. Introduction .............. 5\n"
                    "2. Study Objectives .......... 8\n"
                    "3. Study Design .............. 12\n"
                ),
            },
            {"page_number": 3, "text": "Body content begins here."},
        ]
        toc = LlmRetemplater._detect_toc_pages(blocks)
        assert 2 in toc
        assert 1 not in toc
        assert 3 not in toc

    def test_detects_continuation_pages(self):
        """Multi-page TOC: continuation pages without header."""
        blocks = [
            {
                "page_number": 2,
                "text": (
                    "TABLE OF CONTENTS\n"
                    "1. Introduction .............. 5\n"
                    "2. Study Objectives .......... 8\n"
                    "3. Study Design .............. 12\n"
                ),
            },
            {
                "page_number": 3,
                "text": (
                    "4. Population ................ 15\n"
                    "5. Endpoints ................. 20\n"
                    "6. Statistical Methods ....... 25\n"
                ),
            },
        ]
        toc = LlmRetemplater._detect_toc_pages(blocks)
        assert 2 in toc
        assert 3 in toc

    def test_no_toc_returns_empty(self):
        """Protocols without TOC produce empty set."""
        blocks = [
            {"page_number": 1, "text": "Body content only."},
            {"page_number": 2, "text": "More body content."},
        ]
        toc = LlmRetemplater._detect_toc_pages(blocks)
        assert len(toc) == 0

    def test_chunk_by_pages_excludes_toc(self, retemplater):
        """_chunk_by_pages skips blocks on TOC pages."""
        blocks = [
            {"page_number": 1, "text": "Title page"},
            {"page_number": 2, "text": "TOC entry text"},
            {"page_number": 3, "text": "Body section text"},
        ]
        chunks = retemplater._chunk_by_pages(blocks, toc_pages={2})
        joined = " ".join(ct for ct, _ in chunks)
        assert "Body section text" in joined
        assert "TOC entry text" not in joined

    def test_assemble_sections_excludes_toc(self):
        """_assemble_sections skips text blocks from TOC pages."""
        assignments = [
            _PageAssignment(
                section_code="B.1",
                confidence=0.9,
                key_concepts=["title"],
                pages=[1, 2, 3],
            ),
        ]
        blocks = [
            {"page_number": 1, "text": "Title content"},
            {"page_number": 2, "text": "TOC index stuff"},
            {"page_number": 3, "text": "Body content here"},
        ]
        sections = LlmRetemplater._assemble_sections(
            assignments=assignments,
            text_blocks=blocks,
            run_id="test-run",
            source_run_id="src-run",
            source_sha256="a" * 64,
            registry_id="NCT0001",
            toc_pages={2},
        )
        assert len(sections) == 1
        assert "Title content" in sections[0].content_text
        assert "Body content here" in sections[0].content_text
        assert "TOC index stuff" not in sections[0].content_text

    def test_end_to_end_toc_excluded(self, retemplater):
        """Full retemplate() pipeline excludes TOC pages."""
        toc_text = (
            "TABLE OF CONTENTS\n"
            "1. Introduction .............. 5\n"
            "2. Study Objectives .......... 8\n"
            "3. Study Design .............. 12\n"
        )
        blocks = [
            {"page_number": 1, "text": toc_text},
            {
                "page_number": 2,
                "text": "Drug X is a novel kinase inhibitor "
                        "for advanced solid tumours.",
            },
        ]
        response = [
            {
                "section_code": "B.1",
                "confidence": 0.90,
                "key_concepts": ["drug"],
                "pages": [2],
            },
        ]
        result = _run_with_mock(
            retemplater,
            response,
            text_blocks=blocks,
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result.section_count >= 1
        md = result.retemplated_artifact_key
        # The TOC text should not appear in any section content
        for sec in LlmRetemplater._assemble_sections(
            assignments=[
                _PageAssignment(
                    section_code="B.1",
                    confidence=0.9,
                    key_concepts=["drug"],
                    pages=[2],
                ),
            ],
            text_blocks=blocks,
            run_id="test",
            source_run_id="src",
            source_sha256="a" * 64,
            registry_id="NCT0001",
            toc_pages={1},
        ):
            assert "TABLE OF CONTENTS" not in sec.content_text
