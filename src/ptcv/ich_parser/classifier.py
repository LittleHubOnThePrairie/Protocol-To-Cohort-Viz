"""ICH E6(R3) section classifiers.

Two implementations are provided:

RuleBasedClassifier — pure regex/keyword matching, no external
    dependencies. Suitable for dev/test and for protocols with standard
    ICH E6(R3) headings. Confidence is scored by keyword density.

RAGClassifier — **DEPRECATED** (PTCV-102). Cohere dependency removed.
    Superseded by the query-driven pipeline (PTCV-87/88/89/91/96).
    Kept for backward compatibility; raises ``DeprecationWarning``
    on instantiation.

Risk tier: MEDIUM — data pipeline ML component (no patient data).

Regulatory references:
- ALCOA+ Accurate: confidence_score required; low-confidence sections
  flagged for human review (review_required=True)
"""

from __future__ import annotations

import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any

from .models import IchSection
from .schema_loader import get_classifier_sections, get_review_threshold


# ---------------------------------------------------------------------------
# ICH E6(R3) Appendix B section definitions — loaded from YAML (PTCV-67)
# ---------------------------------------------------------------------------

_ICH_SECTIONS: dict[str, dict[str, Any]] = get_classifier_sections()

# Minimum block size for splitting — blocks below this are merged (PTCV-48)
_MIN_BLOCK_CHARS = 500


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SectionClassifier(ABC):
    """Abstract base for ICH E6(R3) section classifiers."""

    @abstractmethod
    def classify(
        self,
        text: str,
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        """Classify ``text`` into ICH E6(R3) Appendix B sections.

        Args:
            text: Full protocol text to classify.
            registry_id: Trial identifier (EUCT-Code or NCT-ID).
            run_id: UUID4 for the current ICH-parse run.
            source_run_id: run_id from the upstream PTCV-19 extraction.
            source_sha256: SHA-256 of the upstream extraction artifact.

        Returns:
            List of IchSection instances, one per detected section.
            extraction_timestamp_utc is left empty; IchParser sets it
            immediately before writing to storage.
        """


# ---------------------------------------------------------------------------
# Rule-based classifier (no external deps)
# ---------------------------------------------------------------------------

class RuleBasedClassifier(SectionClassifier):
    """Regex + keyword density classifier for ICH E6(R3) sections.

    Splits the protocol text into candidate blocks by heading detection,
    then scores each block against the ICH section keyword sets.
    Confidence is the normalised keyword hit ratio (0.0–1.0).

    Sections with no keyword hits are assigned confidence=0.0 and
    legacy_format=True (non-standard headings detected).
    """

    # Heading detection: numbered or lettered headings or ALL CAPS lines
    _HEADING_RE = re.compile(
        r"(?m)^(?:"
        r"\d+(?:\.\d+)*[\.\)]\s+[A-Z]"  # 1. Title or 1.2. Title
        r"|[A-Z][A-Z\s]{3,}$"            # ALL CAPS HEADING
        r"|(?:Section|SECTION|Part|PART)\s+[A-Z\d]"
        r")"
    )

    def classify(
        self,
        text: str,
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        blocks = self._split_into_blocks(text)
        sections: list[IchSection] = []

        for block_text in blocks:
            best_code, best_score, best_content = self._score_block(block_text)
            if best_score == 0.0:
                continue  # Skip blocks with no keyword hits

            review_required = best_score < get_review_threshold(best_code)
            legacy = best_score < 0.40  # Very low score → likely legacy format

            sections.append(
                IchSection(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    section_code=best_code,
                    section_name=_ICH_SECTIONS[best_code]["name"],
                    content_json=json.dumps(best_content, ensure_ascii=False),
                    confidence_score=round(best_score, 4),
                    review_required=review_required,
                    legacy_format=legacy,
                    content_text=block_text,
                )
            )

        # Deduplicate: keep highest-confidence assignment per section_code
        return self._deduplicate(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split text into candidate section blocks by heading detection.

        Blocks shorter than ``_MIN_BLOCK_CHARS`` are merged with the
        following block to avoid over-splitting (PTCV-48).
        """
        positions = [m.start() for m in self._HEADING_RE.finditer(text)]
        if not positions:
            return [text]
        positions.append(len(text))
        raw = [text[positions[i]: positions[i + 1]] for i in range(len(positions) - 1)]

        # Merge small blocks with the next block
        merged: list[str] = []
        carry = ""
        for block in raw:
            combined = carry + block
            if len(combined) < _MIN_BLOCK_CHARS:
                carry = combined
            else:
                merged.append(combined)
                carry = ""
        if carry:
            if merged:
                merged[-1] += carry
            else:
                merged.append(carry)
        return merged

    def _score_block(
        self, block: str
    ) -> tuple[str, float, dict[str, Any]]:
        """Score ``block`` against all ICH sections and return best match.

        Returns:
            Tuple of (section_code, confidence_score, content_dict).
        """
        lower = block.lower()
        best_code = "B.1"
        best_score = 0.0

        for code, defn in _ICH_SECTIONS.items():
            # Pattern hits (weighted higher)
            pattern_hits = sum(
                1 for p in defn["patterns"] if re.search(p, lower)
            )
            # Keyword hits (weighted lower)
            keyword_hits = sum(
                1 for kw in defn["keywords"] if kw.lower() in lower
            )

            max_possible = len(defn["patterns"]) * 2 + len(defn["keywords"])
            raw = pattern_hits * 2 + keyword_hits
            score = raw / max_possible if max_possible > 0 else 0.0

            if score > best_score:
                best_score = score
                best_code = code

        content: dict[str, Any] = {
            "text_excerpt": block[:2000].strip(),
            "word_count": len(block.split()),
        }
        return best_code, best_score, content

    def _deduplicate(self, sections: list[IchSection]) -> list[IchSection]:
        """Keep one IchSection per section_code — highest confidence wins.

        When multiple blocks classify to the same section, the
        highest-confidence one is kept and content_text from all
        blocks is merged (PTCV-64).
        """
        import dataclasses as _dc

        best: dict[str, IchSection] = {}
        extra_texts: dict[str, list[str]] = {}
        for sec in sections:
            existing = best.get(sec.section_code)
            if existing is None:
                best[sec.section_code] = sec
                extra_texts[sec.section_code] = []
            elif sec.confidence_score > existing.confidence_score:
                extra_texts[sec.section_code].append(
                    existing.content_text,
                )
                best[sec.section_code] = sec
            else:
                extra_texts[sec.section_code].append(
                    sec.content_text,
                )

        # Merge content_text from duplicate blocks
        result: list[IchSection] = []
        for code in sorted(best.keys()):
            sec = best[code]
            extras = [
                t for t in extra_texts.get(code, []) if t
            ]
            if extras and sec.content_text:
                merged = sec.content_text + "\n" + "\n".join(
                    extras,
                )
                sec = _dc.replace(sec, content_text=merged)
            result.append(sec)

        return result


# ---------------------------------------------------------------------------
# RAG classifier — DEPRECATED (PTCV-102)
# ---------------------------------------------------------------------------


class RAGClassifier(SectionClassifier):
    """**DEPRECATED** — Cohere dependency removed (PTCV-102).

    This classifier was superseded by the query-driven pipeline
    (``SectionMatcher`` + ``QueryExtractor`` + ``SummarizationMatcher``).

    Instantiation raises ``DeprecationWarning`` and delegates to
    ``RuleBasedClassifier`` as a transparent fallback.

    .. deprecated:: PTCV-102
        Use the query-driven pipeline instead.
    """

    def __init__(
        self,
        cohere_model: str = "embed-english-v3.0",
        claude_model: str = "claude-sonnet-4-6",
        top_k: int = 3,
    ) -> None:
        warnings.warn(
            "RAGClassifier is deprecated (PTCV-102). Cohere dependency "
            "removed. Use the query-driven pipeline (SectionMatcher + "
            "QueryExtractor) instead. Falling back to RuleBasedClassifier.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._fallback = RuleBasedClassifier()

    def classify(
        self,
        text: str,
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        """Delegate to ``RuleBasedClassifier``."""
        return self._fallback.classify(
            text, registry_id, run_id, source_run_id, source_sha256,
        )
