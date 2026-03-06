"""Map classified ICH sections onto full protocol text (PTCV-43).

Finds the character offsets of each classified section's text excerpt
within the full raw protocol text, producing a list of spans that
can be used to render highlighted regions in the full-protocol view.

Risk tier: LOW — text processing only (no patient data).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptcv.ich_parser.models import IchSection


@dataclasses.dataclass
class TextSpan:
    """A character range in the full protocol text.

    Attributes:
        start: Start character offset (inclusive).
        end: End character offset (exclusive).
        section_code: ICH section code (e.g. "B.3"), or empty
            for unclassified gaps.
        section_name: Human-readable section name, or empty.
        confidence: Classifier confidence 0.0-1.0, or 0.0 for gaps.
        classified: True if this span was classified by the parser.
    """

    start: int
    end: int
    section_code: str = ""
    section_name: str = ""
    confidence: float = 0.0
    classified: bool = False

    @property
    def length(self) -> int:
        """Character length of this span."""
        return self.end - self.start


def _extract_search_text(section: "IchSection") -> str:
    """Extract the best search string from a section's content_json.

    Args:
        section: Classified ICH section.

    Returns:
        Text string to search for in the full protocol.
    """
    import json

    try:
        data = json.loads(section.content_json)
    except (json.JSONDecodeError, TypeError):
        return section.content_json

    if isinstance(data, dict):
        for key in ("text_excerpt", "text", "content", "summary"):
            if key in data and data[key]:
                return str(data[key]).strip()
        # Fallback: first string value
        for v in data.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    elif isinstance(data, str):
        return data

    return str(data)


def map_sections_to_spans(
    protocol_text: str,
    sections: list["IchSection"],
) -> list[TextSpan]:
    """Map classified sections to character spans in the protocol text.

    Searches for each section's text excerpt in the protocol text.
    Returns a sorted list of spans covering the entire document:
    classified spans interleaved with unclassified gap spans.

    Args:
        protocol_text: Full raw protocol text.
        sections: Classified ICH sections from the parser.

    Returns:
        List of TextSpan covering [0, len(protocol_text)), sorted
        by start offset. Classified spans have classified=True.
    """
    classified: list[TextSpan] = []
    text_lower = protocol_text.lower()

    for sec in sections:
        search = _extract_search_text(sec)
        if not search:
            continue

        # Try exact match first, then case-insensitive
        idx = protocol_text.find(search)
        if idx == -1:
            idx = text_lower.find(search.lower())
        if idx == -1:
            # Try matching first 200 chars as fallback
            short = search[:200]
            idx = text_lower.find(short.lower())
            if idx != -1:
                # Extend to full length if possible
                end = min(idx + len(search), len(protocol_text))
                classified.append(TextSpan(
                    start=idx,
                    end=end,
                    section_code=sec.section_code,
                    section_name=sec.section_name,
                    confidence=sec.confidence_score,
                    classified=True,
                ))
                continue

        if idx != -1:
            classified.append(TextSpan(
                start=idx,
                end=idx + len(search),
                section_code=sec.section_code,
                section_name=sec.section_name,
                confidence=sec.confidence_score,
                classified=True,
            ))

    # Sort by start offset; resolve overlaps by keeping first match
    classified.sort(key=lambda s: s.start)
    deduped: list[TextSpan] = []
    last_end = 0
    for span in classified:
        if span.start < last_end:
            continue  # Skip overlapping span
        deduped.append(span)
        last_end = span.end

    # Fill gaps with unclassified spans
    result: list[TextSpan] = []
    pos = 0
    for span in deduped:
        if span.start > pos:
            result.append(TextSpan(
                start=pos,
                end=span.start,
                classified=False,
            ))
        result.append(span)
        pos = span.end

    if pos < len(protocol_text):
        result.append(TextSpan(
            start=pos,
            end=len(protocol_text),
            classified=False,
        ))

    return result


def compute_coverage(spans: list[TextSpan], total_length: int) -> dict:
    """Compute classification coverage statistics.

    Args:
        spans: Spans from map_sections_to_spans().
        total_length: Length of the full protocol text.

    Returns:
        Dict with classified_chars, unclassified_chars,
        coverage_pct, gap_count, classified_count.
    """
    classified_chars = sum(
        s.length for s in spans if s.classified
    )
    gap_count = sum(1 for s in spans if not s.classified)
    classified_count = sum(1 for s in spans if s.classified)

    return {
        "classified_chars": classified_chars,
        "unclassified_chars": total_length - classified_chars,
        "coverage_pct": (
            classified_chars / total_length * 100
            if total_length > 0
            else 0.0
        ),
        "gap_count": gap_count,
        "classified_count": classified_count,
        "total_length": total_length,
    }
