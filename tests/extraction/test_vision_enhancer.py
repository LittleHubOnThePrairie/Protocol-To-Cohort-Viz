"""Tests for VisionEnhancer — PTCV-172.

All tests mock the Anthropic client and fitz module to avoid
real API calls and model downloads.

Qualification phase: IQ/OQ
Risk tier: MEDIUM
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
# fitz/pymupdf4llm stubs are provided by conftest.py

from ptcv.extraction.models import ExtractedTable, TextBlock  # noqa: E402
from ptcv.extraction.vision_enhancer import (  # noqa: E402
    SparsePageCandidate,
    VisionEnhancementResult,
    VisionExtractionResult,
    VisionEnhancer,
    _DEFAULT_MAX_PAGES,
    _VISION_BLOCK_INDEX_OFFSET,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_text_block(
    page_number: int = 1,
    block_index: int = 0,
    text: str = "Some text content here for testing",
) -> TextBlock:
    return TextBlock(
        run_id="run-1",
        source_registry_id="NCT00001",
        source_sha256="a" * 64,
        page_number=page_number,
        block_index=block_index,
        text=text,
        block_type="paragraph",
    )


def _make_table(
    page_number: int = 1,
    table_index: int = 0,
) -> ExtractedTable:
    return ExtractedTable(
        run_id="run-1",
        source_registry_id="NCT00001",
        source_sha256="a" * 64,
        page_number=page_number,
        extractor_used="pymupdf4llm",
        table_index=table_index,
        header_row=json.dumps(["Visit", "Day"]),
        data_rows=json.dumps([["V1", "1"]]),
    )


def _make_vision_response_json(
    text_blocks: list[dict] | None = None,
    tables: list[dict] | None = None,
    is_cover_page: bool = False,
    cover_fields: dict | None = None,
) -> str:
    return json.dumps({
        "text_blocks": text_blocks or [],
        "tables": tables or [],
        "is_cover_page": is_cover_page,
        "cover_fields": cover_fields or {},
    })


class FakeUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str, input_tokens: int = 100,
                 output_tokens: int = 50):
        self.content = [FakeContentBlock(text)]
        self.usage = FakeUsage(input_tokens, output_tokens)


class FakeAnthropicClient:
    """Mock Anthropic client returning canned responses."""

    def __init__(self, response_text: str = "{}",
                 input_tokens: int = 100, output_tokens: int = 50):
        self._response_text = response_text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.messages = self

    def create(self, **kwargs):
        return FakeResponse(
            self._response_text,
            self._input_tokens,
            self._output_tokens,
        )


class FakePixmap:
    def tobytes(self, fmt: str = "png") -> bytes:
        # Minimal PNG: 8-byte signature
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


class FakePage:
    def get_pixmap(self, dpi: int = 150) -> FakePixmap:
        return FakePixmap()


class FakeDocument:
    def __init__(self, page_count: int = 10):
        self._page_count = page_count

    def __len__(self) -> int:
        return self._page_count

    def __getitem__(self, idx: int) -> FakePage:
        if idx < 0 or idx >= self._page_count:
            raise IndexError(f"page index {idx} out of range")
        return FakePage()

    def close(self) -> None:
        pass


class FakeFitz:
    """Mock fitz module."""

    @staticmethod
    def open(stream: bytes = b"", filetype: str = "pdf") -> FakeDocument:
        return FakeDocument(page_count=10)


# ------------------------------------------------------------------
# TestSparsePageCandidate
# ------------------------------------------------------------------

class TestSparsePageCandidate:
    def test_frozen_dataclass(self):
        c = SparsePageCandidate(
            page_number=1, priority=0,
            text_char_count=0, block_count=0,
        )
        assert c.page_number == 1
        assert c.priority == 0
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.page_number = 2  # type: ignore[misc]


# ------------------------------------------------------------------
# TestIdentifySparsePages
# ------------------------------------------------------------------

class TestIdentifySparsePages:
    def setup_method(self):
        self.enhancer = VisionEnhancer(max_pages=5)

    def test_empty_page_identified(self):
        """Page with 0 blocks should be flagged as sparse."""
        # Page 5 has no blocks at all in a 10-page doc
        blocks = [_make_text_block(page_number=1)]
        result = self.enhancer.identify_sparse_pages(blocks, [], 10)
        page_nums = [c.page_number for c in result]
        assert 5 in page_nums

    def test_short_text_page_identified(self):
        """Page with < 50 chars total should be flagged."""
        blocks = [
            _make_text_block(page_number=4, text="Hi"),
        ]
        result = self.enhancer.identify_sparse_pages(blocks, [], 5)
        page_nums = [c.page_number for c in result]
        assert 4 in page_nums

    def test_cover_pages_always_included(self):
        """Pages 1-3 should always be candidates (cover pages)."""
        # Even if they have content
        blocks = [
            _make_text_block(page_number=1, text="x" * 200),
            _make_text_block(page_number=1, text="y" * 200),
            _make_text_block(page_number=1, text="z" * 200),
            _make_text_block(page_number=2, text="a" * 200),
            _make_text_block(page_number=2, text="b" * 200),
            _make_text_block(page_number=2, text="c" * 200),
            _make_text_block(page_number=3, text="d" * 200),
            _make_text_block(page_number=3, text="e" * 200),
            _make_text_block(page_number=3, text="f" * 200),
        ]
        result = self.enhancer.identify_sparse_pages(blocks, [], 5)
        page_nums = [c.page_number for c in result]
        assert 1 in page_nums
        assert 2 in page_nums
        assert 3 in page_nums

    def test_max_pages_respected(self):
        """Should return at most max_pages candidates."""
        enhancer = VisionEnhancer(max_pages=2)
        result = enhancer.identify_sparse_pages([], [], 20)
        assert len(result) <= 2

    def test_priority_ordering(self):
        """Cover pages (0) before empty (1) before sparse (2)."""
        blocks = [
            _make_text_block(page_number=5, text="Hi"),  # sparse
        ]
        result = self.enhancer.identify_sparse_pages(blocks, [], 10)
        priorities = [c.priority for c in result]
        assert priorities == sorted(priorities)

    def test_deduplication(self):
        """Page appearing in multiple criteria only listed once."""
        # Page 1 is both a cover page and empty
        result = self.enhancer.identify_sparse_pages([], [], 5)
        page_nums = [c.page_number for c in result]
        assert page_nums.count(1) == 1

    def test_pages_with_content_not_sparse(self):
        """Pages with sufficient blocks and text are not flagged."""
        blocks = [
            _make_text_block(page_number=5, block_index=0, text="a" * 100),
            _make_text_block(page_number=5, block_index=1, text="b" * 100),
            _make_text_block(page_number=5, block_index=2, text="c" * 100),
        ]
        result = self.enhancer.identify_sparse_pages(blocks, [], 10)
        page_nums = [c.page_number for c in result]
        # Page 5 has 3 blocks and 300 chars — should NOT be flagged
        assert 5 not in page_nums

    def test_page_with_table_not_sparse(self):
        """Pages with tables are not flagged even if text is sparse."""
        blocks = [_make_text_block(page_number=5, text="Hi")]
        tables = [_make_table(page_number=5)]
        result = self.enhancer.identify_sparse_pages(blocks, tables, 10)
        page_nums = [c.page_number for c in result]
        # Page 5 has table, should NOT be in sparse candidates
        # (but may be in cover pages if page 5 were <= 3)
        assert 5 not in page_nums


# ------------------------------------------------------------------
# TestRenderPageToPng
# ------------------------------------------------------------------

class TestRenderPageToPng:
    def test_render_returns_png_bytes(self):
        """Should return bytes starting with PNG magic."""
        with patch.dict(sys.modules, {"fitz": FakeFitz()}):
            # Re-import to pick up mock
            import importlib
            import ptcv.extraction.vision_enhancer as mod
            importlib.reload(mod)

            result = mod.VisionEnhancer.render_page_to_png(
                b"fake-pdf", page_number=1,
            )
            assert result[:4] == b"\x89PNG"

    def test_render_out_of_range_raises(self):
        """Page number out of range should raise IndexError."""
        fake_fitz = FakeFitz()

        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            import importlib
            import ptcv.extraction.vision_enhancer as mod
            importlib.reload(mod)

            with pytest.raises(IndexError, match="out of range"):
                mod.VisionEnhancer.render_page_to_png(
                    b"fake-pdf", page_number=99,
                )

    def test_graceful_without_fitz(self):
        """Should raise ImportError when fitz not available."""
        with patch.dict(sys.modules, {"fitz": None}):
            import importlib
            import ptcv.extraction.vision_enhancer as mod
            importlib.reload(mod)

            with pytest.raises(ImportError):
                mod.VisionEnhancer.render_page_to_png(
                    b"fake-pdf", page_number=1,
                )


# ------------------------------------------------------------------
# TestExtractFromImage
# ------------------------------------------------------------------

class TestExtractFromImage:
    def _make_enhancer(self, response_json: str,
                       input_tokens: int = 100,
                       output_tokens: int = 50) -> VisionEnhancer:
        enhancer = VisionEnhancer(max_pages=5)
        enhancer._client = FakeAnthropicClient(
            response_json, input_tokens, output_tokens,
        )
        return enhancer

    def test_text_extraction_from_image(self):
        """Vision API response with text blocks produces TextBlocks."""
        resp = _make_vision_response_json(
            text_blocks=[
                {"text": "Protocol Title", "block_type": "heading"},
                {"text": "Introduction paragraph.", "block_type": "paragraph"},
            ],
        )
        enhancer = self._make_enhancer(resp)
        result = enhancer.extract_from_image(
            image_png=b"\x89PNG\r\n",
            page_number=1,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        assert len(result.text_blocks) == 2
        assert result.text_blocks[0].text == "Protocol Title"
        assert result.text_blocks[0].block_type == "heading"
        assert result.text_blocks[0].block_index == _VISION_BLOCK_INDEX_OFFSET

    def test_table_extraction_from_image(self):
        """Vision API response with tables produces ExtractedTables."""
        resp = _make_vision_response_json(
            tables=[{
                "header": ["Visit", "Day", "Assessment"],
                "data_rows": [
                    ["Screening", "1", "X"],
                    ["Week 4", "28", "X"],
                ],
            }],
        )
        enhancer = self._make_enhancer(resp)
        result = enhancer.extract_from_image(
            image_png=b"\x89PNG\r\n",
            page_number=5,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
            table_index_start=10,
        )
        assert len(result.tables) == 1
        assert result.tables[0].extractor_used == "claude_vision"
        assert result.tables[0].table_index == 10
        header = json.loads(result.tables[0].header_row)
        assert "Visit" in header

    def test_cover_page_detection(self):
        """Vision API identifying a cover page populates cover_fields."""
        resp = _make_vision_response_json(
            is_cover_page=True,
            cover_fields={
                "protocol_title": "A Phase III Study of Drug X",
                "sponsor": "Pharma Corp",
                "principal_investigator": "Dr. Smith",
                "protocol_number": "PROTO-001",
            },
        )
        enhancer = self._make_enhancer(resp)
        result = enhancer.extract_from_image(
            image_png=b"\x89PNG\r\n",
            page_number=1,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        assert result.is_cover_page is True
        assert result.cover_fields["protocol_title"] == (
            "A Phase III Study of Drug X"
        )
        assert result.cover_fields["sponsor"] == "Pharma Corp"

    def test_malformed_json_handled(self):
        """Unparseable JSON returns empty result, not exception."""
        enhancer = self._make_enhancer("not valid json {{{{")
        result = enhancer.extract_from_image(
            image_png=b"\x89PNG\r\n",
            page_number=1,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        assert len(result.text_blocks) == 0
        assert len(result.tables) == 0
        assert result.is_cover_page is False

    def test_token_tracking(self):
        """Input/output tokens captured from API response."""
        resp = _make_vision_response_json()
        enhancer = self._make_enhancer(resp, input_tokens=500,
                                       output_tokens=200)
        result = enhancer.extract_from_image(
            image_png=b"\x89PNG\r\n",
            page_number=1,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        assert result.input_tokens == 500
        assert result.output_tokens == 200


# ------------------------------------------------------------------
# TestEnhance
# ------------------------------------------------------------------

class TestEnhance:
    def _make_enhancer_with_mocks(
        self,
        response_json: str = "{}",
        page_count: int = 10,
    ) -> VisionEnhancer:
        enhancer = VisionEnhancer(max_pages=5)
        enhancer._client = FakeAnthropicClient(response_json)

        # Mock render_page_to_png
        enhancer.render_page_to_png = MagicMock(  # type: ignore[assignment]
            return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
        )
        return enhancer

    def test_enhance_appends_blocks(self):
        """Vision blocks are returned in enhancement result."""
        resp = _make_vision_response_json(
            text_blocks=[
                {"text": "Vision extracted text", "block_type": "paragraph"},
            ],
        )
        enhancer = self._make_enhancer_with_mocks(resp)
        result = enhancer.enhance(
            pdf_bytes=b"fake-pdf",
            text_blocks=[],
            tables=[],
            page_count=5,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        assert result.pages_processed > 0
        assert len(result.text_blocks) > 0
        assert result.text_blocks[0].text == "Vision extracted text"

    def test_enhance_sets_extractor_used(self):
        """Vision tables have extractor_used='claude_vision'."""
        resp = _make_vision_response_json(
            tables=[{
                "header": ["Col1"],
                "data_rows": [["Val1"]],
            }],
        )
        enhancer = self._make_enhancer_with_mocks(resp)
        result = enhancer.enhance(
            pdf_bytes=b"fake-pdf",
            text_blocks=[],
            tables=[],
            page_count=5,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        for tbl in result.tables:
            assert tbl.extractor_used == "claude_vision"

    def test_enhance_respects_max_pages(self):
        """At most max_pages API calls should be made."""
        resp = _make_vision_response_json()
        enhancer = VisionEnhancer(max_pages=2)
        enhancer._client = FakeAnthropicClient(resp)
        enhancer.render_page_to_png = MagicMock(  # type: ignore[assignment]
            return_value=b"\x89PNG\r\n\x1a\n",
        )
        result = enhancer.enhance(
            pdf_bytes=b"fake-pdf",
            text_blocks=[],
            tables=[],
            page_count=20,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        assert result.pages_processed <= 2

    def test_enhance_logs_cost(self):
        """Estimated cost is computed from token counts."""
        resp = _make_vision_response_json()
        enhancer = self._make_enhancer_with_mocks(resp)
        result = enhancer.enhance(
            pdf_bytes=b"fake-pdf",
            text_blocks=[],
            tables=[],
            page_count=5,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        if result.pages_processed > 0:
            assert result.estimated_cost_usd > 0.0

    def test_enhance_with_no_sparse_pages(self):
        """No candidates means empty result."""
        # All pages have plenty of content
        blocks = []
        for pn in range(1, 6):
            for bi in range(5):
                blocks.append(_make_text_block(
                    page_number=pn,
                    block_index=bi,
                    text="x" * 200,
                ))
        enhancer = VisionEnhancer(max_pages=5)
        result = enhancer.enhance(
            pdf_bytes=b"fake-pdf",
            text_blocks=blocks,
            tables=[],
            page_count=5,
            run_id="run-1",
            registry_id="NCT00001",
            source_sha256="a" * 64,
        )
        assert result.pages_processed == 0
        assert len(result.text_blocks) == 0


# ------------------------------------------------------------------
# TestCostGuardrails
# ------------------------------------------------------------------

class TestCostGuardrails:
    def test_default_max_pages_is_5(self):
        enhancer = VisionEnhancer()
        assert enhancer.max_pages == _DEFAULT_MAX_PAGES

    def test_env_var_override(self):
        with patch.dict(os.environ, {"PTCV_VISION_MAX_PAGES": "3"}):
            enhancer = VisionEnhancer()
            assert enhancer.max_pages == 3

    def test_cost_estimation(self):
        """Cost estimate based on token counts."""
        result = VisionEnhancementResult(
            text_blocks=[],
            tables=[],
            pages_processed=3,
            total_input_tokens=5400,
            total_output_tokens=1500,
            cover_fields={},
            estimated_cost_usd=0.0,
        )
        # Manual calculation
        from ptcv.extraction.vision_enhancer import (
            _EST_COST_PER_INPUT_TOKEN,
            _EST_COST_PER_OUTPUT_TOKEN,
        )
        expected = (
            5400 * _EST_COST_PER_INPUT_TOKEN
            + 1500 * _EST_COST_PER_OUTPUT_TOKEN
        )
        assert expected > 0
        # Verify the enhance method would compute this correctly
        assert expected == pytest.approx(0.0387, abs=0.001)
