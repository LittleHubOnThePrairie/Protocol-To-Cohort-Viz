"""Color-coded ICH section coverage diagram (PTCV-270).

Renders a visual grid of ICH E6(R3) Appendix B sections (B.1-B.16)
color-coded by extraction confidence:
  - Green: high confidence (≥0.85)
  - Amber: moderate confidence (0.70-0.85)
  - Red: low confidence (<0.70) or gap (no content)

Data source: AssembledProtocol.sections from the template assembler.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptcv.ich_parser.template_assembler import AssembledProtocol


# Thresholds aligned with template_assembler.py
_HIGH = 0.85
_LOW = 0.70


@dataclasses.dataclass
class SectionBlock:
    """One block in the coverage diagram.

    Attributes:
        code: ICH section code (e.g. "B.4").
        name: Section short name.
        confidence: Average confidence (0.0 for gaps).
        status: "high", "moderate", "low", or "gap".
        color: CSS hex color for the block.
        hit_count: Number of extraction hits.
        is_gap: Whether the section has no content.
    """

    code: str
    name: str
    confidence: float
    status: str
    color: str
    hit_count: int = 0
    is_gap: bool = False


def build_section_blocks(
    assembled: "AssembledProtocol",
) -> list[SectionBlock]:
    """Build coverage diagram blocks from an AssembledProtocol.

    Args:
        assembled: AssembledProtocol from template assembler.

    Returns:
        Ordered list of SectionBlock, one per parent section.
    """
    blocks: list[SectionBlock] = []

    for section in assembled.sections:
        conf = section.average_confidence

        if section.is_gap or not section.populated:
            status = "gap"
            color = "#dc3545"  # Bootstrap danger red
        elif conf >= _HIGH:
            status = "high"
            color = "#28a745"  # Bootstrap success green
        elif conf >= _LOW:
            status = "moderate"
            color = "#ffc107"  # Bootstrap warning amber
        else:
            status = "low"
            color = "#dc3545"  # Red for low confidence

        blocks.append(SectionBlock(
            code=section.section_code,
            name=section.section_name,
            confidence=conf,
            status=status,
            color=color,
            hit_count=len(section.hits),
            is_gap=section.is_gap,
        ))

    return blocks


def render_coverage_diagram(
    assembled: "AssembledProtocol",
) -> None:
    """Render the color-coded coverage diagram in Streamlit.

    Displays B.1-B.16 as a 4x4 grid of colored blocks with
    expandable detail per section.

    Args:
        assembled: AssembledProtocol from template assembler.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    blocks = build_section_blocks(assembled)

    st.subheader("Section Coverage")

    # Summary legend
    high_n = sum(1 for b in blocks if b.status == "high")
    mod_n = sum(1 for b in blocks if b.status == "moderate")
    low_n = sum(1 for b in blocks if b.status == "low")
    gap_n = sum(1 for b in blocks if b.status == "gap")

    legend_cols = st.columns(4)
    legend_cols[0].markdown(
        f"🟢 **High** ({high_n})"
    )
    legend_cols[1].markdown(
        f"🟡 **Moderate** ({mod_n})"
    )
    legend_cols[2].markdown(
        f"🔴 **Low** ({low_n})"
    )
    legend_cols[3].markdown(
        f"⬜ **Gap** ({gap_n})"
    )

    # Render 4 columns x 4 rows grid
    rows_per_grid = 4
    cols_per_grid = 4

    for row_idx in range(rows_per_grid):
        cols = st.columns(cols_per_grid)
        for col_idx in range(cols_per_grid):
            block_idx = row_idx * cols_per_grid + col_idx
            if block_idx >= len(blocks):
                break

            block = blocks[block_idx]

            with cols[col_idx]:
                # Colored container using markdown
                if block.is_gap:
                    label = "Gap"
                    conf_str = ""
                else:
                    label = f"{block.confidence:.0%}"
                    conf_str = f" ({block.hit_count} hits)"

                st.markdown(
                    f'<div style="'
                    f"background-color: {block.color};"
                    f"color: white;"
                    f"padding: 8px;"
                    f"border-radius: 6px;"
                    f"text-align: center;"
                    f"margin-bottom: 4px;"
                    f"font-size: 0.85em;"
                    f'">'
                    f"<strong>{block.code}</strong><br>"
                    f"{label}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                with st.expander(
                    block.name, expanded=False,
                ):
                    if block.is_gap:
                        st.caption(
                            "No content extracted for this section."
                        )
                    else:
                        st.caption(
                            f"**Confidence:** {block.confidence:.2f}\n\n"
                            f"**Hits:** {block.hit_count}\n\n"
                            f"**Status:** {block.status.title()}"
                        )


__all__ = [
    "SectionBlock",
    "build_section_blocks",
    "render_coverage_diagram",
]
