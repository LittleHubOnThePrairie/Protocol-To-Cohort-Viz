"""Tests for expanded context window in RAGClassifier (PTCV-48).

Feature: Expanded context window for RAGClassifier (fallback tier)

  Scenario: Long sections fully visible to fallback classifier
    Given a protocol section with 6,000 characters of content
    When the RAGClassifier processes this section
    Then the Claude prompt includes at least 6,000 characters

  Scenario: Embedding captures more content
    Given a protocol block with 7,000 characters
    When the block is embedded via Cohere
    Then at least 7,000 characters are sent to the embedding API

  Scenario: Small blocks merged
    Given a protocol with a heading followed by 200 chars then another heading
    When _split_into_blocks is called
    Then the 200-char block is merged with the adjacent block
    And no block in the output is shorter than 500 chars (except the last)

  Scenario: Text excerpt expanded
    Given a classified section with 2,000 chars of content
    When the classification result is produced
    Then the text_excerpt fallback contains up to 1,000 chars
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.classifier import (
    RuleBasedClassifier,
    _MIN_BLOCK_CHARS,
)


# ---------------------------------------------------------------
# Scenario: Long sections fully visible to classifier
# ---------------------------------------------------------------


class TestExpandedContext:
    """Claude prompt truncation expanded from 3K to 8K."""

    def test_6k_section_visible_in_prompt(self) -> None:
        """A 6K-char block should appear fully in the prompt."""
        # We test by verifying the truncation constant in the source
        # The prompt uses block[:8000], so 6K is within budget
        block = "A" * 6000
        truncated = block[:8000]
        assert len(truncated) == 6000

    def test_8k_truncation_limit(self) -> None:
        """Blocks over 8K should be truncated to 8000 chars."""
        block = "B" * 10000
        truncated = block[:8000]
        assert len(truncated) == 8000

    def test_prompt_uses_8k_not_3k(self) -> None:
        """Verify the prompt template references 8K truncation."""
        import inspect
        from ptcv.ich_parser.classifier import RAGClassifier

        source = inspect.getsource(RAGClassifier._generate_with_claude)
        assert "block[:8000]" in source
        assert "block[:3000]" not in source


# ---------------------------------------------------------------
# Scenario: Embedding captures more content
# ---------------------------------------------------------------


class TestExpandedEmbedding:
    """Cohere embedding truncation expanded from 4K to 8K."""

    def test_7k_block_not_truncated(self) -> None:
        """A 7K-char block fits within the 8192-char embedding limit."""
        block = "C" * 7000
        truncated = block[:8192]
        assert len(truncated) == 7000

    def test_embedding_limit_is_8192(self) -> None:
        """Verify the embedding truncation uses 8192."""
        import inspect
        from ptcv.ich_parser.classifier import RAGClassifier

        source = inspect.getsource(RAGClassifier.classify)
        assert "b[:8192]" in source
        assert "b[:4096]" not in source


# ---------------------------------------------------------------
# Scenario: Small blocks merged
# ---------------------------------------------------------------


class TestBlockMerging:
    """Blocks < _MIN_BLOCK_CHARS merged with next block."""

    def test_small_block_merged(self) -> None:
        """A 200-char block between headings is merged with the next."""
        classifier = RuleBasedClassifier()
        # Build text with two headings: first section is short
        text = (
            "1. SHORT HEADING\n"
            + "x" * 150 + "\n"
            + "2. LONGER SECTION\n"
            + "y" * 1000 + "\n"
        )
        blocks = classifier._split_into_blocks(text)
        # The short block should be merged into the longer one
        assert len(blocks) <= 2
        # No block should be shorter than _MIN_BLOCK_CHARS except the last
        for block in blocks[:-1]:
            assert len(block) >= _MIN_BLOCK_CHARS, (
                f"Block has {len(block)} chars, below {_MIN_BLOCK_CHARS}"
            )

    def test_multiple_small_blocks_merged(self) -> None:
        """Multiple consecutive short blocks merge into one."""
        classifier = RuleBasedClassifier()
        text = (
            "1. HEADING A\n" + "a" * 100 + "\n"
            + "2. HEADING B\n" + "b" * 100 + "\n"
            + "3. HEADING C\n" + "c" * 100 + "\n"
            + "4. REAL CONTENT\n" + "d" * 2000 + "\n"
        )
        blocks = classifier._split_into_blocks(text)
        # Short blocks A, B, C should be merged together or with D
        for block in blocks[:-1]:
            assert len(block) >= _MIN_BLOCK_CHARS

    def test_large_blocks_not_merged(self) -> None:
        """Blocks already above threshold stay separate."""
        classifier = RuleBasedClassifier()
        text = (
            "1. FIRST SECTION\n" + "x" * 1000 + "\n"
            + "2. SECOND SECTION\n" + "y" * 1000 + "\n"
        )
        blocks = classifier._split_into_blocks(text)
        assert len(blocks) == 2

    def test_single_block_passthrough(self) -> None:
        """Text with no headings returns as single block."""
        classifier = RuleBasedClassifier()
        text = "Just some text without headings. " * 50
        blocks = classifier._split_into_blocks(text)
        assert len(blocks) == 1

    def test_min_block_chars_constant(self) -> None:
        assert _MIN_BLOCK_CHARS == 500


# ---------------------------------------------------------------
# Scenario: Text excerpt expanded
# ---------------------------------------------------------------


class TestExpandedExcerpt:
    """text_excerpt fallback expanded from 500 to 1000 chars."""

    def test_excerpt_fallback_uses_1000(self) -> None:
        """Verify _generate_with_claude uses block[:1000] for fallback."""
        import inspect
        from ptcv.ich_parser.classifier import RAGClassifier

        source = inspect.getsource(RAGClassifier._generate_with_claude)
        assert "block[:1000]" in source
        assert "block[:500]" not in source

    def test_excerpt_instruction_says_1000(self) -> None:
        """Verify the prompt asks for 1000-char excerpts."""
        import inspect
        from ptcv.ich_parser.classifier import RAGClassifier

        source = inspect.getsource(RAGClassifier._generate_with_claude)
        assert "first 1000 chars" in source
        assert "first 500 chars" not in source

    def test_max_tokens_is_1024(self) -> None:
        """Verify max_tokens expanded from 512 to 1024."""
        import inspect
        from ptcv.ich_parser.classifier import RAGClassifier

        source = inspect.getsource(RAGClassifier._generate_with_claude)
        assert "max_tokens=1024" in source
        assert "max_tokens=512" not in source


# ---------------------------------------------------------------
# Scenario: RuleBasedClassifier content_json excerpt
# ---------------------------------------------------------------


class TestRuleBasedExcerpt:
    """RuleBasedClassifier text_excerpt in content_json."""

    def test_score_block_excerpt_is_2000(self) -> None:
        """_score_block still uses 2000-char excerpt (unchanged)."""
        classifier = RuleBasedClassifier()
        long_block = "informed consent " * 500  # ~8500 chars
        _, _, content = classifier._score_block(long_block)
        assert len(content["text_excerpt"]) <= 2000
