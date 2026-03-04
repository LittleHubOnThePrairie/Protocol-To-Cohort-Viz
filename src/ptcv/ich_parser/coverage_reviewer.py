"""Deterministic text coverage reviewer for retemplating (PTCV-60).

Checks how much of the original extracted text is represented in the
retemplated ICH sections. Uses sentence-level overlap rather than
LLM judgment to avoid compounding LLM errors.

Risk tier: LOW -- deterministic text analysis only.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import IchSection

from .schema_loader import get_boilerplate_pattern, get_min_sentence_length


@dataclasses.dataclass
class UncoveredBlock:
    """A block of original text not found in any retemplated section.

    Attributes:
        page_number: Page where the uncovered text appeared.
        text: The uncovered text (truncated to 500 chars).
        char_count: Full character count of the uncovered block.
    """

    page_number: int
    text: str
    char_count: int


@dataclasses.dataclass
class CoverageResult:
    """Result of the text coverage review.

    Attributes:
        coverage_score: Float 0.0-1.0, ratio of original characters
            covered by retemplated sections.
        total_original_chars: Total characters in original text
            blocks (after whitespace normalization, excluding
            boilerplate).
        covered_chars: Characters from original found in sections.
        uncovered_blocks: Significant text blocks with no coverage.
        section_coverage: Dict mapping section_code to the number
            of original characters that mapped to it.
        pass_threshold: The threshold used (default 0.70).
        passed: True if coverage_score >= pass_threshold.
    """

    coverage_score: float
    total_original_chars: int
    covered_chars: int
    uncovered_blocks: list[UncoveredBlock]
    section_coverage: dict[str, int]
    pass_threshold: float
    passed: bool


# Loaded from YAML configuration (PTCV-69).
_BOILERPLATE_RE: re.Pattern[str] = get_boilerplate_pattern()
_MIN_SENTENCE_LEN: int = get_min_sentence_length()


class CoverageReviewer:
    """Deterministic text-overlap coverage reviewer.

    Algorithm:
    1. Split original text into sentences (period/newline
       delimited)
    2. Normalize whitespace in both original and retemplated text
    3. For each original sentence, check if a significant substring
       appears in any retemplated section's content_json
    4. Compute coverage as covered_chars / total_chars
    5. Collect uncovered blocks for human review

    Args:
        pass_threshold: Minimum coverage score to pass
            (default 0.70).
        min_sentence_len: Minimum chars for a sentence to count
            (default 20). Shorter fragments are excluded.
    """

    def __init__(
        self,
        pass_threshold: float = 0.70,
        min_sentence_len: int = _MIN_SENTENCE_LEN,
    ) -> None:
        self._threshold = pass_threshold
        self._min_len = min_sentence_len

    def review(
        self,
        original_text_blocks: list[dict],
        retemplated_sections: list["IchSection"],
    ) -> CoverageResult:
        """Check coverage of original text by retemplated sections.

        Args:
            original_text_blocks: List of dicts with 'page_number'
                and 'text' keys from the extraction stage.
            retemplated_sections: IchSection list from retemplating.

        Returns:
            CoverageResult with score, uncovered blocks, etc.
        """
        # 1. Build normalized section text corpus
        section_texts: dict[str, str] = {}
        for sec in retemplated_sections:
            # Prefer full content_text (PTCV-64) over excerpt
            if sec.content_text:
                text = sec.content_text
            else:
                try:
                    content = json.loads(sec.content_json)
                    text = content.get("text_excerpt", "")
                except (
                    json.JSONDecodeError,
                    TypeError,
                    AttributeError,
                ):
                    text = ""
            section_texts[sec.section_code] = self._normalize(text)

        all_section_text = " ".join(section_texts.values())

        # 2. Split original text into sentences
        sentences = self._extract_sentences(original_text_blocks)

        # 3. Check each sentence against section corpus
        total_chars = 0
        covered_chars = 0
        section_coverage: dict[str, int] = {
            code: 0 for code in section_texts
        }
        uncovered: list[UncoveredBlock] = []

        for page, sentence, char_count in sentences:
            total_chars += char_count
            normalized = self._normalize(sentence)

            # Use first 60 chars as a check substring
            check_str = normalized[:60]
            if len(check_str) < self._min_len:
                covered_chars += char_count  # Too short to matter
                continue

            found_in = self._find_in_sections(
                check_str, section_texts,
            )
            if found_in:
                covered_chars += char_count
                section_coverage[found_in] = (
                    section_coverage.get(found_in, 0) + char_count
                )
            elif check_str in all_section_text:
                covered_chars += char_count
            else:
                uncovered.append(
                    UncoveredBlock(
                        page_number=page,
                        text=sentence[:500],
                        char_count=char_count,
                    )
                )

        if total_chars == 0:
            score = 1.0
        else:
            score = covered_chars / total_chars

        return CoverageResult(
            coverage_score=round(score, 4),
            total_original_chars=total_chars,
            covered_chars=covered_chars,
            uncovered_blocks=uncovered,
            section_coverage=section_coverage,
            pass_threshold=self._threshold,
            passed=score >= self._threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse whitespace and lowercase for comparison."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _extract_sentences(
        self, text_blocks: list[dict],
    ) -> list[tuple[int, str, int]]:
        """Split text blocks into (page, sentence, char_count).

        Excludes boilerplate lines and fragments shorter than
        _min_sentence_len.
        """
        results: list[tuple[int, str, int]] = []
        for block in text_blocks:
            page = block.get("page_number", 0)
            text = block.get("text", "")

            # Remove boilerplate
            text = _BOILERPLATE_RE.sub("", text)

            # Split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) >= self._min_len:
                    results.append((page, sent, len(sent)))

        return results

    @staticmethod
    def _find_in_sections(
        needle: str,
        section_texts: dict[str, str],
    ) -> Optional[str]:
        """Find which section contains the needle substring.

        Returns section_code or None.
        """
        for code, text in section_texts.items():
            if needle in text:
                return code
        return None
