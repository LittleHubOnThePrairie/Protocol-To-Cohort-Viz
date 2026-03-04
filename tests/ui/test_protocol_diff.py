"""Tests for side-by-side protocol diff view helpers (PTCV-79).

Pure-Python tests — no Streamlit dependency.

Covers GHERKIN scenarios:
  - Diff view shows both documents (build_original_text produces text)
  - Word-level differences highlighted (inputs differ, enabling diff)
  - Toggle between display modes (build_diff_label returns labels)
  - Comparison unavailable before retemplating (empty input handling)
"""

from __future__ import annotations

from ptcv.ui.components.protocol_diff import (
    build_diff_label,
    build_original_text,
)


# ---------------------------------------------------------------------------
# Scenario: build_original_text concatenation
# ---------------------------------------------------------------------------

class TestBuildOriginalText:
    """build_original_text produces correct concatenated text."""

    def test_single_block(self) -> None:
        blocks = [{"page_number": 1, "text": "Hello world"}]
        result = build_original_text(blocks)
        assert result == "Hello world"

    def test_multiple_blocks_same_page(self) -> None:
        blocks = [
            {"page_number": 1, "text": "First paragraph"},
            {"page_number": 1, "text": "Second paragraph"},
        ]
        result = build_original_text(blocks)
        assert "First paragraph" in result
        assert "Second paragraph" in result
        assert "--- Page" not in result

    def test_page_break_marker(self) -> None:
        blocks = [
            {"page_number": 1, "text": "Page 1 content"},
            {"page_number": 2, "text": "Page 2 content"},
        ]
        result = build_original_text(blocks)
        assert "--- Page 2 ---" in result
        assert "Page 1 content" in result
        assert "Page 2 content" in result

    def test_empty_input(self) -> None:
        assert build_original_text([]) == ""

    def test_whitespace_only_blocks_skipped(self) -> None:
        blocks = [
            {"page_number": 1, "text": "   "},
            {"page_number": 1, "text": "Real content"},
        ]
        result = build_original_text(blocks)
        assert result == "Real content"

    def test_text_stripped(self) -> None:
        blocks = [{"page_number": 1, "text": "  padded  "}]
        result = build_original_text(blocks)
        assert result == "padded"

    def test_multiple_page_breaks(self) -> None:
        blocks = [
            {"page_number": 1, "text": "P1"},
            {"page_number": 3, "text": "P3"},
            {"page_number": 5, "text": "P5"},
        ]
        result = build_original_text(blocks)
        assert "--- Page 3 ---" in result
        assert "--- Page 5 ---" in result

    def test_missing_text_key_treated_as_empty(self) -> None:
        blocks = [{"page_number": 1}]
        result = build_original_text(blocks)
        assert result == ""

    def test_preserves_block_order(self) -> None:
        blocks = [
            {"page_number": 1, "text": "Alpha"},
            {"page_number": 1, "text": "Beta"},
            {"page_number": 1, "text": "Gamma"},
        ]
        result = build_original_text(blocks)
        alpha_idx = result.index("Alpha")
        beta_idx = result.index("Beta")
        gamma_idx = result.index("Gamma")
        assert alpha_idx < beta_idx < gamma_idx


# ---------------------------------------------------------------------------
# Scenario: build_diff_label
# ---------------------------------------------------------------------------

class TestBuildDiffLabel:
    """build_diff_label returns appropriate panel labels."""

    def test_labels_contain_registry_id(self) -> None:
        left, right = build_diff_label("NCT00112827")
        assert "NCT00112827" in left

    def test_right_label_contains_ich(self) -> None:
        _, right = build_diff_label("NCT00112827")
        assert "ICH E6(R3)" in right

    def test_llm_method_tag(self) -> None:
        _, right = build_diff_label("NCT001", "llm")
        assert "(LLM)" in right

    def test_rag_method_tag(self) -> None:
        _, right = build_diff_label("NCT001", "rag")
        assert "(RAG)" in right

    def test_no_method_tag_for_unknown(self) -> None:
        _, right = build_diff_label("NCT001", "ich_parser")
        assert "(LLM)" not in right
        assert "(RAG)" not in right

    def test_empty_method(self) -> None:
        _, right = build_diff_label("NCT001", "")
        assert "(LLM)" not in right


# ---------------------------------------------------------------------------
# Scenario: Comparison unavailable before retemplating
# ---------------------------------------------------------------------------

class TestEmptyInputHandling:
    """Edge cases produce safe outputs for the diff component."""

    def test_empty_blocks_returns_empty_string(self) -> None:
        assert build_original_text([]) == ""

    def test_all_whitespace_blocks_returns_empty(self) -> None:
        blocks = [
            {"page_number": 1, "text": "   "},
            {"page_number": 2, "text": "\n\t"},
        ]
        assert build_original_text(blocks) == ""
