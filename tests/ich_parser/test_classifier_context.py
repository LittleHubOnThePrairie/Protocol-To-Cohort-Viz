"""Tests for classifier context handling (PTCV-48, PTCV-102).

Feature: Block merging and excerpt handling in RuleBasedClassifier

  Scenario: Small blocks merged
    Given a protocol with a heading followed by 200 chars then another heading
    When _split_into_blocks is called
    Then the 200-char block is merged with the adjacent block
    And no block in the output is shorter than 500 chars (except the last)

  Scenario: Text excerpt expanded
    Given a classified section with 2,000 chars of content
    When the classification result is produced
    Then the text_excerpt fallback contains up to 2,000 chars

.. note::

    Tests for RAGClassifier context/embedding windows were removed in
    PTCV-102 (Cohere dependency removed).  RAGClassifier is now a
    deprecated stub delegating to RuleBasedClassifier.
"""

from __future__ import annotations

import json

import pytest

from ptcv.ich_parser.classifier import (
    RuleBasedClassifier,
    _MIN_BLOCK_CHARS,
)


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
