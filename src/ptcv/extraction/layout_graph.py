"""Spatial layout graph for page element relationships (PTCV-227).

Builds a per-page graph of spatial relationships between text blocks,
tables, figures, footnotes, and headers/footers. Enables downstream
stages to resolve context that depends on spatial proximity — e.g.,
linking a footnote marker in a table cell to the footnote text at
the page bottom, or associating a caption with its figure.

Risk tier: LOW — read-only analysis, no external calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .figure_detector import BoundingBox


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ElementType(str, Enum):
    """Type of page element."""

    TEXT_BLOCK = "text_block"
    TABLE = "table"
    FIGURE = "figure"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"


class RelationshipType(str, Enum):
    """Spatial relationship between two elements."""

    ABOVE = "above"
    BELOW = "below"
    ADJACENT = "adjacent"
    CONTAINS = "contains"
    REFERENCES = "references"
    CAPTION_OF = "caption_of"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class LayoutNode:
    """A single element in the layout graph.

    Attributes:
        node_id: Unique identifier within the page (e.g., "text_0").
        element_type: Type of the element.
        page_number: 1-based page number.
        bbox: Bounding box (None if unknown).
        text_preview: First 200 chars of text content.
        metadata: Extra attributes (block_index, table_index, etc.).
    """

    node_id: str
    element_type: ElementType
    page_number: int
    bbox: Optional[BoundingBox] = None
    text_preview: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class LayoutEdge:
    """A relationship between two layout nodes.

    Attributes:
        source_id: Node ID of the source element.
        target_id: Node ID of the target element.
        relationship: Type of spatial relationship.
        confidence: Confidence in the relationship (0.0–1.0).
    """

    source_id: str
    target_id: str
    relationship: RelationshipType
    confidence: float = 1.0


@dataclass
class PageLayout:
    """Layout graph for a single page.

    Attributes:
        page_number: 1-based page number.
        nodes: All elements on the page.
        edges: Relationships between elements.
    """

    page_number: int
    nodes: list[LayoutNode] = field(default_factory=list)
    edges: list[LayoutEdge] = field(default_factory=list)

    def get_node(self, node_id: str) -> Optional[LayoutNode]:
        """Find a node by ID."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def get_edges_from(self, node_id: str) -> list[LayoutEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_edges_to(self, node_id: str) -> list[LayoutEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target_id == node_id]

    def get_related(
        self,
        node_id: str,
        relationship: RelationshipType,
    ) -> list[LayoutNode]:
        """Get nodes related to a given node by relationship type."""
        target_ids = [
            e.target_id for e in self.edges
            if e.source_id == node_id
            and e.relationship == relationship
        ]
        return [n for n in self.nodes if n.node_id in target_ids]


@dataclass
class DocumentLayout:
    """Layout graph for an entire document.

    Attributes:
        pages: Per-page layout graphs.
        page_count: Total number of pages.
    """

    pages: list[PageLayout] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_nodes(self) -> int:
        return sum(len(p.nodes) for p in self.pages)

    @property
    def total_edges(self) -> int:
        return sum(len(p.edges) for p in self.pages)

    def get_page(self, page_number: int) -> Optional[PageLayout]:
        """Get layout for a specific page."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None


# ---------------------------------------------------------------------------
# Caption detection
# ---------------------------------------------------------------------------

_CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table|Exhibit|Diagram|Scheme)"
    r"\s*\d*[.:]?\s",
    re.IGNORECASE | re.MULTILINE,
)

_FOOTNOTE_RE = re.compile(
    r"^\s*(?:\d+[.)]\s|[*†‡§]\s|[a-z][.)]\s)",
    re.MULTILINE,
)

_HEADER_FOOTER_RE = re.compile(
    r"(?:page\s+\d+|protocol\s+(?:no|number|version))"
    r"|(?:confidential|draft|final)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_page_layout(
    page_number: int,
    text_blocks: list[dict],
    tables: list[dict],
    figures: list[dict],
) -> PageLayout:
    """Build a layout graph for one page.

    Creates nodes for each element and infers spatial relationships:
    - Caption text → nearest figure/table (``caption_of``)
    - Footnote markers → footnote text (``references``)
    - Headers/footers identified by position and patterns

    Args:
        page_number: 1-based page number.
        text_blocks: Dicts with ``text``, ``block_index``, and
            optionally ``in_table``.
        tables: Dicts with ``table_index``, ``header_row``,
            ``page_number``.
        figures: Dicts with ``page_number``, ``bbox`` (dict with
            x0/y0/x1/y1), ``figure_type_hint``, ``caption``.

    Returns:
        PageLayout with nodes and inferred edges.
    """
    layout = PageLayout(page_number=page_number)

    # Add text block nodes
    for block in text_blocks:
        text = block.get("text", "")
        block_idx = block.get("block_index", 0)
        node_id = f"text_{block_idx}"
        in_table = block.get("in_table", False)

        element_type = _classify_text_element(text, block_idx, len(text_blocks))

        layout.nodes.append(LayoutNode(
            node_id=node_id,
            element_type=element_type,
            page_number=page_number,
            text_preview=text[:200],
            metadata={"block_index": block_idx, "in_table": in_table},
        ))

    # Add table nodes
    for tbl in tables:
        tbl_idx = tbl.get("table_index", 0)
        node_id = f"table_{tbl_idx}"
        header = tbl.get("header_row", "")
        preview = header[:200] if isinstance(header, str) else str(header)[:200]

        layout.nodes.append(LayoutNode(
            node_id=node_id,
            element_type=ElementType.TABLE,
            page_number=page_number,
            text_preview=preview,
            metadata={"table_index": tbl_idx},
        ))

    # Add figure nodes
    for i, fig in enumerate(figures):
        node_id = f"figure_{i}"
        bbox_dict = fig.get("bbox")
        bbox = None
        if bbox_dict and isinstance(bbox_dict, dict):
            bbox = BoundingBox(
                x0=bbox_dict.get("x0", 0),
                y0=bbox_dict.get("y0", 0),
                x1=bbox_dict.get("x1", 0),
                y1=bbox_dict.get("y1", 0),
            )

        layout.nodes.append(LayoutNode(
            node_id=node_id,
            element_type=ElementType.FIGURE,
            page_number=page_number,
            bbox=bbox,
            text_preview=fig.get("caption", "")[:200],
            metadata={
                "figure_type_hint": fig.get("figure_type_hint", "generic"),
            },
        ))

    # Infer edges
    _infer_caption_edges(layout)
    _infer_footnote_edges(layout)

    return layout


def _classify_text_element(
    text: str,
    block_index: int,
    total_blocks: int,
) -> ElementType:
    """Classify a text block's element type by content and position.

    Args:
        text: Block text.
        block_index: 0-based position on page.
        total_blocks: Total blocks on page.

    Returns:
        ElementType classification.
    """
    stripped = text.strip()

    # Caption detection
    if _CAPTION_RE.match(stripped):
        return ElementType.CAPTION

    # Footnote detection
    if _FOOTNOTE_RE.match(stripped) and block_index > total_blocks * 0.6:
        return ElementType.FOOTNOTE

    # Header/footer by position
    if block_index == 0 and _HEADER_FOOTER_RE.search(stripped):
        return ElementType.HEADER

    if block_index >= total_blocks - 1 and _HEADER_FOOTER_RE.search(stripped):
        return ElementType.FOOTER

    return ElementType.TEXT_BLOCK


def _infer_caption_edges(layout: PageLayout) -> None:
    """Link caption nodes to their nearest figure or table."""
    caption_nodes = [
        n for n in layout.nodes
        if n.element_type == ElementType.CAPTION
    ]
    target_nodes = [
        n for n in layout.nodes
        if n.element_type in (ElementType.FIGURE, ElementType.TABLE)
    ]

    for cap in caption_nodes:
        text_lower = cap.text_preview.lower()

        # Match caption to figure or table by type keyword
        best_target: Optional[LayoutNode] = None
        if any(kw in text_lower for kw in ("figure", "fig.", "diagram")):
            figures = [n for n in target_nodes if n.element_type == ElementType.FIGURE]
            if figures:
                best_target = figures[0]  # First figure on page
        elif "table" in text_lower:
            tables = [n for n in target_nodes if n.element_type == ElementType.TABLE]
            if tables:
                best_target = tables[0]  # First table on page

        if best_target is None and target_nodes:
            best_target = target_nodes[0]

        if best_target is not None:
            layout.edges.append(LayoutEdge(
                source_id=cap.node_id,
                target_id=best_target.node_id,
                relationship=RelationshipType.CAPTION_OF,
                confidence=0.8,
            ))


def _infer_footnote_edges(layout: PageLayout) -> None:
    """Link footnote nodes to table nodes on the same page."""
    footnote_nodes = [
        n for n in layout.nodes
        if n.element_type == ElementType.FOOTNOTE
    ]
    table_nodes = [
        n for n in layout.nodes
        if n.element_type == ElementType.TABLE
    ]

    if not footnote_nodes or not table_nodes:
        return

    # Link each footnote to the last table on the page
    # (footnotes typically appear below the table they reference)
    last_table = table_nodes[-1]
    for fn in footnote_nodes:
        layout.edges.append(LayoutEdge(
            source_id=fn.node_id,
            target_id=last_table.node_id,
            relationship=RelationshipType.REFERENCES,
            confidence=0.7,
        ))


def build_document_layout(
    pages_data: list[dict],
) -> DocumentLayout:
    """Build layout graphs for all pages in a document.

    Args:
        pages_data: List of dicts, one per page, with keys:
            ``page_number``, ``text_blocks``, ``tables``, ``figures``.

    Returns:
        DocumentLayout with per-page graphs.
    """
    doc = DocumentLayout()

    for page_data in pages_data:
        page_num = page_data.get("page_number", 0)
        if page_num <= 0:
            continue

        layout = build_page_layout(
            page_number=page_num,
            text_blocks=page_data.get("text_blocks", []),
            tables=page_data.get("tables", []),
            figures=page_data.get("figures", []),
        )
        doc.pages.append(layout)

    return doc


__all__ = [
    "BoundingBox",
    "DocumentLayout",
    "ElementType",
    "LayoutEdge",
    "LayoutNode",
    "PageLayout",
    "RelationshipType",
    "build_document_layout",
    "build_page_layout",
]
