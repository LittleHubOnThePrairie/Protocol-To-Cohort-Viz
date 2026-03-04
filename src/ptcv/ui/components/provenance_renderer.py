"""Annotation hover provenance renderer (PTCV-81).

Renders retemplated ICH sections as HTML with hover tooltips
showing source page provenance and confidence scores. Uses
``st.components.v1.html()`` for rich tooltip rendering within
Streamlit.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptcv.ich_parser.models import IchSection

# Confidence threshold for low-confidence visual distinction
_LOW_CONFIDENCE_THRESHOLD = 0.70

# CSS for the provenance viewer
_PROVENANCE_CSS = """\
<style>
.provenance-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
    padding: 1rem;
}
.provenance-section {
    margin-bottom: 1.5rem;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    position: relative;
}
.provenance-section.high-confidence {
    border-left: 4px solid #3b82f6;
    background: #f0f7ff;
}
.provenance-section.low-confidence {
    border-left: 4px solid #f59e0b;
    background: #fffbeb;
}
.provenance-section.missing {
    border-left: 4px solid #9ca3af;
    background: #f9fafb;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.section-body {
    border-bottom: 1px dotted #999;
    cursor: help;
    position: relative;
    display: inline;
}
.section-body:hover {
    background-color: rgba(59, 130, 246, 0.1);
}
.low-confidence .section-body:hover {
    background-color: rgba(245, 158, 11, 0.1);
}
.tooltip-badge {
    display: inline-block;
    font-size: 0.75rem;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.badge-high {
    background: #dbeafe;
    color: #1e40af;
}
.badge-low {
    background: #fef3c7;
    color: #92400e;
}
.section-content {
    white-space: pre-wrap;
    font-size: 0.9rem;
    margin-top: 0.25rem;
}
.provenance-meta {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.25rem;
}
</style>
"""


def _parse_page_range(section: "IchSection") -> str:
    """Extract page range string from section content_json.

    Args:
        section: IchSection with content_json containing page_range.

    Returns:
        Human-readable page range like "Pages 12-15" or "Page 3".
    """
    try:
        data = json.loads(section.content_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    page_range = data.get("page_range", [])
    if not page_range:
        return ""
    if isinstance(page_range, list) and len(page_range) == 2:
        lo, hi = int(page_range[0]), int(page_range[1])
        if lo == hi:
            return f"Page {lo}"
        return f"Pages {lo}\u2013{hi}"
    if isinstance(page_range, list) and len(page_range) == 1:
        return f"Page {int(page_range[0])}"
    return ""


def _build_tooltip(section: "IchSection") -> str:
    """Build tooltip text for a section.

    Args:
        section: IchSection with provenance data.

    Returns:
        Tooltip string for the title attribute.
    """
    parts: list[str] = []
    page_str = _parse_page_range(section)
    if page_str:
        parts.append(f"Source: {page_str}")
    parts.append(
        f"Confidence: {section.confidence_score:.0%}",
    )
    if section.confidence_score < _LOW_CONFIDENCE_THRESHOLD:
        parts.append("Low confidence \u2014 review recommended")
    return " | ".join(parts)


def build_provenance_html(
    sections: list["IchSection"],
    registry_id: str,
) -> str:
    """Build HTML with hover-tooltip provenance annotations.

    Each section is rendered as a block with:
    - Color-coded left border (blue=high, amber=low confidence)
    - Hover tooltip showing source pages and confidence
    - Dotted underline on content for hover affordance

    Args:
        sections: List of IchSection objects from parquet_to_sections().
        registry_id: Protocol registry ID for the title.

    Returns:
        Complete HTML string for rendering via st.components.v1.html().
    """
    from .ich_regenerator import ICH_SECTIONS

    by_code: dict[str, "IchSection"] = {}
    for s in sections:
        code = s.section_code
        if (
            code not in by_code
            or s.confidence_score > by_code[code].confidence_score
        ):
            by_code[code] = s

    parts: list[str] = [_PROVENANCE_CSS]
    parts.append('<div class="provenance-container">')
    parts.append(
        f"<h2>{html.escape(registry_id)}: ICH E6(R3) "
        f"Reformatted Protocol</h2>"
    )

    for code, name in ICH_SECTIONS:
        sec = by_code.get(code)
        if sec is None:
            parts.append(
                '<div class="provenance-section missing">'
                f'<div class="section-header">'
                f"{html.escape(code)} {html.escape(name)}</div>"
                '<div class="provenance-meta">'
                "<em>Section not detected in source protocol</em>"
                "</div></div>"
            )
            continue

        is_low = sec.confidence_score < _LOW_CONFIDENCE_THRESHOLD
        conf_class = "low-confidence" if is_low else "high-confidence"
        badge_class = "badge-low" if is_low else "badge-high"
        tooltip = _build_tooltip(sec)
        page_str = _parse_page_range(sec)
        conf_pct = f"{sec.confidence_score:.0%}"

        # Section header with badge
        badge_label = f"{conf_pct}"
        if page_str:
            badge_label = f"{page_str} \u2022 {conf_pct}"

        parts.append(
            f'<div class="provenance-section {conf_class}">'
        )
        parts.append(
            f'<div class="section-header">'
            f"{html.escape(code)} {html.escape(name)}"
            f'<span class="tooltip-badge {badge_class}">'
            f"{html.escape(badge_label)}</span></div>"
        )

        # Content with hover tooltip
        content = sec.content_text or ""
        if not content:
            try:
                data = json.loads(sec.content_json)
                content = data.get("text_excerpt", "")
            except (json.JSONDecodeError, TypeError):
                content = ""

        # Truncate for display (full text can be very long)
        display_text = content[:3000]
        if len(content) > 3000:
            display_text += "\n\n[...truncated...]"

        escaped_content = html.escape(display_text)
        escaped_tooltip = html.escape(tooltip)

        parts.append(
            f'<div class="section-content">'
            f'<span class="section-body" title="{escaped_tooltip}">'
            f"{escaped_content}</span></div>"
        )

        # Provenance metadata line
        meta_parts = []
        if page_str:
            meta_parts.append(page_str)
        meta_parts.append(f"Confidence: {conf_pct}")
        if is_low:
            meta_parts.append(
                "\u26a0\ufe0f Low confidence \u2014 review recommended",
            )
        parts.append(
            f'<div class="provenance-meta">'
            f"{' &middot; '.join(meta_parts)}</div>"
        )
        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


def estimate_html_height(sections: list["IchSection"]) -> int:
    """Estimate the rendered HTML height for st.components.v1.html().

    Args:
        sections: List of IchSection objects.

    Returns:
        Estimated pixel height (minimum 400, max 3000).
    """
    # Rough: 150px per section header + ~1px per 3 chars of content
    total = 100  # title
    for sec in sections:
        content_len = len(sec.content_text) if sec.content_text else 200
        total += 80 + min(content_len // 3, 500)
    return max(400, min(total, 3000))
