"""Enhanced side-by-side comparison with section alignment (PTCV-80).

Builds aligned section pairs from IchSection data and generates
an HTML component with JavaScript-driven synchronized scrolling.

Pure-Python module — no Streamlit dependency for the data layer.
"""

from __future__ import annotations

import html
from typing import NamedTuple, Sequence


class SectionPair(NamedTuple):
    """Aligned original + retemplated content for one ICH section."""

    section_code: str
    section_name: str
    original_text: str
    retemplated_text: str


def align_sections(
    sections: Sequence[object],
    retemplated_md: str,
) -> list[SectionPair]:
    """Build aligned section pairs from IchSection objects and markdown.

    Args:
        sections: List of IchSection-like objects with section_code,
            section_name, and content_text attributes.
        retemplated_md: Full retemplated markdown string.

    Returns:
        List of SectionPair in canonical ICH order.
    """
    # Parse retemplated markdown into section blocks by heading
    md_sections = _parse_markdown_sections(retemplated_md)

    pairs: list[SectionPair] = []
    for sec in sections:
        code = getattr(sec, "section_code", "")
        name = getattr(sec, "section_name", "")
        original = getattr(sec, "content_text", "") or ""
        retemplated = md_sections.get(code, "")
        pairs.append(SectionPair(code, name, original.strip(), retemplated.strip()))

    return pairs


def _parse_markdown_sections(md: str) -> dict[str, str]:
    """Split retemplated markdown into sections keyed by code.

    Looks for headings like ``## B.1 Title Page`` or ``# B.7``.

    Returns:
        Dict mapping section_code (e.g. "B.1") to the text under
        that heading (up to the next heading).
    """
    import re

    result: dict[str, str] = {}
    # Match markdown headings that start with B.<digit>
    pattern = re.compile(r"^(#{1,3})\s+(B\.\d+)\b", re.MULTILINE)

    matches = list(pattern.finditer(md))
    for i, m in enumerate(matches):
        code = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        result[code] = md[start:end].strip()

    return result


def build_comparison_html(
    pairs: list[SectionPair],
    height: int = 600,
) -> str:
    """Build an HTML component with synchronized scrolling panels.

    Args:
        pairs: Aligned section pairs from align_sections().
        height: Pixel height for scrollable panels.

    Returns:
        Complete HTML string with embedded CSS and JavaScript.
    """
    if not pairs:
        return "<p>No sections available for comparison.</p>"

    nav_items = []
    left_items = []
    right_items = []

    for p in pairs:
        esc_code = html.escape(p.section_code)
        esc_name = html.escape(p.section_name)
        esc_orig = html.escape(p.original_text or "(no original content)")
        esc_retempl = html.escape(p.retemplated_text or "(no retemplated content)")

        nav_items.append(
            f'<div class="nav-item" onclick="jumpTo(\'{esc_code}\')">'
            f"<b>{esc_code}</b> {esc_name}</div>"
        )
        left_items.append(
            f'<div id="left-{esc_code}" class="section">'
            f"<h4>{esc_code} {esc_name}</h4>"
            f"<pre>{esc_orig}</pre></div>"
        )
        right_items.append(
            f'<div id="right-{esc_code}" class="section">'
            f"<h4>{esc_code} {esc_name}</h4>"
            f"<pre>{esc_retempl}</pre></div>"
        )

    nav_html = "\n".join(nav_items)
    left_html = "\n".join(left_items)
    right_html = "\n".join(right_items)

    return _HTML_TEMPLATE.format(
        height=height,
        nav_html=nav_html,
        left_html=left_html,
        right_html=right_html,
    )


_HTML_TEMPLATE = """\
<style>
  .compare-container {{
    display: flex;
    gap: 6px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
  }}
  .nav-panel {{
    width: 180px;
    min-width: 180px;
    overflow-y: auto;
    max-height: {height}px;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 4px;
    background: #fafafa;
  }}
  .nav-item {{
    padding: 4px 6px;
    cursor: pointer;
    border-radius: 3px;
    margin-bottom: 2px;
  }}
  .nav-item:hover, .nav-item.active {{
    background: #e3f2fd;
  }}
  .scroll-panel {{
    flex: 1;
    overflow-y: auto;
    max-height: {height}px;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px;
  }}
  .scroll-panel h4 {{
    position: sticky;
    top: 0;
    background: #fff;
    margin: 0 0 4px 0;
    padding: 4px 0;
    border-bottom: 1px solid #eee;
    color: #1565c0;
  }}
  .section {{
    margin-bottom: 16px;
  }}
  .section pre {{
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: inherit;
    margin: 4px 0;
    line-height: 1.5;
  }}
  .panel-label {{
    font-weight: 600;
    text-align: center;
    padding: 4px;
    background: #f5f5f5;
    border-radius: 4px 4px 0 0;
    border: 1px solid #ddd;
    border-bottom: none;
    font-size: 12px;
  }}
</style>

<div class="compare-container">
  <div class="nav-panel" id="nav">
    <div style="font-weight:600; padding:4px 6px; margin-bottom:4px;">Sections</div>
    {nav_html}
  </div>
  <div style="flex:1; display:flex; flex-direction:column;">
    <div class="panel-label">Original</div>
    <div class="scroll-panel" id="left" onscroll="syncScroll('left')">
      {left_html}
    </div>
  </div>
  <div style="flex:1; display:flex; flex-direction:column;">
    <div class="panel-label">Retemplated ICH E6(R3)</div>
    <div class="scroll-panel" id="right" onscroll="syncScroll('right')">
      {right_html}
    </div>
  </div>
</div>

<script>
  let syncing = false;

  function syncScroll(source) {{
    if (syncing) return;
    syncing = true;

    const src = document.getElementById(source);
    const tgt = document.getElementById(source === 'left' ? 'right' : 'left');
    const ratio = src.scrollTop / (src.scrollHeight - src.clientHeight || 1);
    tgt.scrollTop = ratio * (tgt.scrollHeight - tgt.clientHeight);

    highlightVisibleSection(source);
    setTimeout(() => {{ syncing = false; }}, 50);
  }}

  function jumpTo(code) {{
    const left = document.getElementById('left-' + code);
    const right = document.getElementById('right-' + code);
    if (left) left.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    if (right) right.scrollIntoView({{ behavior: 'smooth', block: 'start' }});

    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => {{
      if (el.textContent.startsWith(code)) el.classList.add('active');
    }});
  }}

  function highlightVisibleSection(panelId) {{
    const panel = document.getElementById(panelId);
    const sections = panel.querySelectorAll('.section');
    let closest = null;
    let minDist = Infinity;

    sections.forEach(sec => {{
      const rect = sec.getBoundingClientRect();
      const panelRect = panel.getBoundingClientRect();
      const dist = Math.abs(rect.top - panelRect.top);
      if (dist < minDist) {{
        minDist = dist;
        closest = sec;
      }}
    }});

    if (closest) {{
      const code = closest.id.replace(panelId + '-', '');
      document.querySelectorAll('.nav-item').forEach(el => {{
        el.classList.remove('active');
        if (el.textContent.startsWith(code)) el.classList.add('active');
      }});
    }}
  }}
</script>
"""
