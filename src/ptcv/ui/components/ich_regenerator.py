"""ICH E6(R3) protocol regeneration component (PTCV-35).

Reconstructs a protocol document in ICH E6(R3) Appendix B structure
from classified IchSection objects and renders it as markdown.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptcv.ich_parser.models import IchSection

# Canonical ICH E6(R3) Appendix B section ordering and names
ICH_SECTIONS: list[tuple[str, str]] = [
    ("B.1", "General Information"),
    ("B.2", "Background Information"),
    ("B.3", "Trial Objectives and Purpose"),
    ("B.4", "Trial Design"),
    ("B.5", "Selection of Subjects"),
    ("B.6", "Treatment of Subjects"),
    ("B.7", "Assessment of Efficacy"),
    ("B.8", "Assessment of Safety"),
    ("B.9", "Statistics"),
    ("B.10", "Direct Access to Source Data and Documents"),
    ("B.11", "Quality Control and Quality Assurance"),
]

_REVIEW_THRESHOLD = 0.70


def _extract_text(section: "IchSection") -> str:
    """Extract readable text from an IchSection.

    Prefers full content_text (PTCV-64) over content_json excerpt.

    Args:
        section: Classified ICH section.

    Returns:
        Best-effort text content.
    """
    # PTCV-64: Use full content_text when available
    if section.content_text:
        return section.content_text

    # Fallback to content_json extraction (pre-PTCV-64 data)
    try:
        data = json.loads(section.content_json)
    except (json.JSONDecodeError, TypeError):
        return section.content_json

    # content_json may have various shapes; extract useful text
    parts: list[str] = []
    if isinstance(data, dict):
        for key in ("text_excerpt", "text", "content", "summary"):
            if key in data and data[key]:
                parts.append(str(data[key]).strip())
        if "key_concepts" in data and isinstance(data["key_concepts"], list):
            concepts = ", ".join(str(c) for c in data["key_concepts"])
            if concepts:
                parts.append(f"**Key concepts:** {concepts}")
        # If nothing matched, dump all string values
        if not parts:
            for v in data.values():
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
    elif isinstance(data, str):
        parts.append(data)

    return "\n\n".join(parts) if parts else str(data)


def regenerate_ich_markdown(
    sections: list["IchSection"],
    registry_id: str,
    format_verdict: str,
    format_confidence: float,
) -> str:
    """Build a full ICH E6(R3) formatted markdown document.

    Iterates B.1 through B.11 in order. For each section code:
    - If a matching IchSection exists, its extracted content is rendered.
    - If no match, a placeholder is inserted.
    - Low-confidence sections get a warning indicator.

    Args:
        sections: List of IchSection objects from parquet_to_sections().
        registry_id: Protocol registry ID for the title.
        format_verdict: ICH_E6R3, PARTIAL_ICH, or NON_ICH.
        format_confidence: Overall format confidence score.

    Returns:
        Complete markdown string.
    """
    # Index sections by code (take highest-confidence if duplicates)
    by_code: dict[str, "IchSection"] = {}
    for sec in sections:
        code = sec.section_code
        if code not in by_code or sec.confidence_score > by_code[code].confidence_score:
            by_code[code] = sec

    lines: list[str] = []

    # Title
    lines.append(f"# {registry_id}: ICH E6(R3) Reformatted Protocol")
    lines.append("")
    lines.append(
        f"**Format verdict:** {format_verdict} | "
        f"**Confidence:** {format_confidence:.2f} | "
        f"**Sections detected:** {len(by_code)} / {len(ICH_SECTIONS)}"
    )
    lines.append("")

    # NON_ICH banner
    if format_verdict == "NON_ICH":
        lines.append(
            "> **Warning:** Most sections could not be mapped from "
            "this non-ICH protocol. Only sections that were extracted "
            "(even if low-confidence) show content below."
        )
        lines.append("")

    # Iterate all 11 sections in order
    for code, name in ICH_SECTIONS:
        sec = by_code.get(code)
        if sec is None:
            # Missing section placeholder
            lines.append(f"## {code} {name}")
            lines.append("")
            lines.append("> *Section not detected in source protocol*")
            lines.append("")
        else:
            # Header with optional low-confidence warning
            if sec.confidence_score < _REVIEW_THRESHOLD:
                lines.append(
                    f"## {code} {name} "
                    f"(confidence: {sec.confidence_score:.2f})"
                )
                lines.append("")
                lines.append(
                    "> *Low confidence — content may be misclassified*"
                )
            else:
                lines.append(f"## {code} {name}")
            lines.append("")
            lines.append(_extract_text(sec))
            lines.append("")

    return "\n".join(lines)


def make_download_filename(
    registry_id: str,
    amendment: str,
) -> str:
    """Build the download filename for the regenerated markdown.

    Args:
        registry_id: Protocol registry ID.
        amendment: Amendment/version string.

    Returns:
        Filename like ``NCT00112827_1.0_ich_reformatted.md``.
    """
    return f"{registry_id}_{amendment}_ich_reformatted.md"
