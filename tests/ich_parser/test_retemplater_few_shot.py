"""Tests for dynamic few-shot prompting in LlmRetemplater (PTCV-47).

Feature: Dynamic few-shot prompting in LlmRetemplater

  Scenario: Few-shot examples included in classification prompt
    Given the gold-standard corpus is loaded
    And a protocol text chunk is being classified
    When the LlmRetemplater generates a classification prompt
    Then the prompt includes at least 1 few-shot example from the corpus

  Scenario: Examples are dynamically selected by relevance
    Given two protocol chunks with different chunk indices
    When each generates a classification prompt
    Then the few-shot examples selected for each chunk are different

  Scenario: Graceful fallback without corpus
    Given the corpus directory is empty or missing
    When a protocol chunk is classified
    Then classification proceeds without few-shot examples
    And a warning is logged

  Scenario: Prompt stays within budget
    Given a protocol chunk at the 80K char truncation limit
    When few-shot examples are injected
    Then total prompt size does not exceed 90K characters
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.corpus_loader import CorpusExemplar, load_corpus
from ptcv.ich_parser.llm_retemplater import (
    LlmRetemplater,
    _MAX_EXEMPLAR_CHARS,
    _MAX_EXEMPLARS,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture()
def retemplater() -> LlmRetemplater:
    """LlmRetemplater with mocked storage and review queue."""
    gateway = MagicMock()
    gateway.initialise = MagicMock()
    review_queue = MagicMock()
    review_queue.initialise = MagicMock()
    return LlmRetemplater(
        gateway=gateway,
        review_queue=review_queue,
    )


@pytest.fixture()
def retemplater_no_corpus() -> LlmRetemplater:
    """LlmRetemplater with empty corpus."""
    gateway = MagicMock()
    gateway.initialise = MagicMock()
    review_queue = MagicMock()
    review_queue.initialise = MagicMock()
    with patch(
        "ptcv.ich_parser.llm_retemplater.load_corpus",
        return_value={},
    ):
        rt = LlmRetemplater(
            gateway=gateway,
            review_queue=review_queue,
        )
    return rt


# ---------------------------------------------------------------
# Scenario: Few-shot examples included in classification prompt
# ---------------------------------------------------------------


class TestFewShotPromptInjection:
    """Scenario: Few-shot examples included in prompt."""

    def test_prompt_contains_example_tags(
        self, retemplater: LlmRetemplater,
    ) -> None:
        prompt = retemplater._build_classification_prompt(
            chunk_text="Test protocol text about objectives.",
            page_range=(1, 5),
            registry_id="NCT_TEST",
            soa_summary=None,
        )
        assert "<example>" in prompt
        assert "</example>" in prompt

    def test_prompt_contains_section_code_labels(
        self, retemplater: LlmRetemplater,
    ) -> None:
        prompt = retemplater._build_classification_prompt(
            chunk_text="Test protocol text.",
            page_range=(1, 5),
            registry_id="NCT_TEST",
            soa_summary=None,
        )
        # Exemplar block should contain section code labels
        assert "Example 1 (B." in prompt

    def test_at_least_one_exemplar(
        self, retemplater: LlmRetemplater,
    ) -> None:
        exemplars = retemplater._select_exemplars(chunk_index=0)
        assert len(exemplars) >= 1
        assert all(
            isinstance(e, CorpusExemplar) for e in exemplars
        )

    def test_max_exemplars_respected(
        self, retemplater: LlmRetemplater,
    ) -> None:
        exemplars = retemplater._select_exemplars(chunk_index=0)
        assert len(exemplars) <= _MAX_EXEMPLARS


# ---------------------------------------------------------------
# Scenario: Examples are dynamically selected by relevance
# ---------------------------------------------------------------


class TestExemplarRotation:
    """Scenario: Examples are dynamically selected per chunk."""

    def test_different_chunks_get_different_exemplars(
        self, retemplater: LlmRetemplater,
    ) -> None:
        ex0 = retemplater._select_exemplars(chunk_index=0)
        ex1 = retemplater._select_exemplars(chunk_index=1)

        codes0 = {e.section_code for e in ex0}
        codes1 = {e.section_code for e in ex1}
        # Different chunk indices should yield different sections
        assert codes0 != codes1

    def test_rotation_wraps_around(
        self, retemplater: LlmRetemplater,
    ) -> None:
        """Large chunk indices wrap via modulo."""
        ex = retemplater._select_exemplars(chunk_index=100)
        assert len(ex) >= 1

    def test_exemplars_from_different_sections(
        self, retemplater: LlmRetemplater,
    ) -> None:
        """Each prompt's exemplars come from different codes."""
        exemplars = retemplater._select_exemplars(chunk_index=0)
        if len(exemplars) >= 2:
            codes = [e.section_code for e in exemplars]
            assert codes[0] != codes[1]


# ---------------------------------------------------------------
# Scenario: Graceful fallback without corpus
# ---------------------------------------------------------------


class TestGracefulFallback:
    """Scenario: Graceful fallback without corpus."""

    def test_empty_corpus_no_exemplars(
        self, retemplater_no_corpus: LlmRetemplater,
    ) -> None:
        exemplars = retemplater_no_corpus._select_exemplars(
            chunk_index=0,
        )
        assert exemplars == []

    def test_empty_corpus_prompt_unchanged(
        self, retemplater_no_corpus: LlmRetemplater,
    ) -> None:
        prompt = retemplater_no_corpus._build_classification_prompt(
            chunk_text="Test text.",
            page_range=(1, 3),
            registry_id="NCT_TEST",
            soa_summary=None,
        )
        assert "<example>" not in prompt
        # Core prompt structure still present
        assert "ICH E6(R3) Appendix B sections:" in prompt
        assert "Protocol text:" in prompt

    def test_format_exemplar_block_empty_list(self) -> None:
        block = LlmRetemplater._format_exemplar_block([])
        assert block == ""


# ---------------------------------------------------------------
# Scenario: Prompt stays within budget
# ---------------------------------------------------------------


class TestPromptBudget:
    """Scenario: Prompt stays within budget."""

    def test_prompt_under_90k_with_large_chunk(
        self, retemplater: LlmRetemplater,
    ) -> None:
        """80K chunk + exemplars should stay under 90K total."""
        large_chunk = "A" * 80_000
        prompt = retemplater._build_classification_prompt(
            chunk_text=large_chunk,
            page_range=(1, 100),
            registry_id="NCT_LARGE",
            soa_summary=None,
        )
        assert len(prompt) < 90_000, (
            f"Prompt is {len(prompt):,} chars, exceeds 90K budget"
        )

    def test_exemplar_text_truncated(
        self, retemplater: LlmRetemplater,
    ) -> None:
        """Exemplar text in prompt is capped at _MAX_EXEMPLAR_CHARS."""
        prompt = retemplater._build_classification_prompt(
            chunk_text="Short text.",
            page_range=(1, 5),
            registry_id="NCT_TEST",
            soa_summary=None,
        )
        # Each <example>...</example> block should be under limit
        import re

        examples = re.findall(
            r"<example>\n(.*?)\n</example>",
            prompt,
            re.DOTALL,
        )
        for ex_text in examples:
            assert len(ex_text) <= _MAX_EXEMPLAR_CHARS, (
                f"Exemplar text is {len(ex_text)} chars, "
                f"exceeds {_MAX_EXEMPLAR_CHARS} limit"
            )

    def test_total_exemplar_budget(
        self, retemplater: LlmRetemplater,
    ) -> None:
        """Total exemplar text < _MAX_EXEMPLARS * _MAX_EXEMPLAR_CHARS."""
        exemplars = retemplater._select_exemplars(chunk_index=0)
        total = sum(len(e.text[:_MAX_EXEMPLAR_CHARS]) for e in exemplars)
        assert total <= _MAX_EXEMPLARS * _MAX_EXEMPLAR_CHARS
