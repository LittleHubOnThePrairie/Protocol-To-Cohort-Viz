"""Tests for spatial layout graph (PTCV-227).

Tests node/edge creation, element classification, caption-to-figure
linking, and footnote-to-table referencing.
"""

from __future__ import annotations

import pytest

from ptcv.extraction.layout_graph import (
    DocumentLayout,
    ElementType,
    LayoutEdge,
    LayoutNode,
    PageLayout,
    RelationshipType,
    build_document_layout,
    build_page_layout,
)
from ptcv.extraction.figure_detector import BoundingBox


class TestLayoutNode:
    def test_creation(self):
        node = LayoutNode(
            node_id="text_0",
            element_type=ElementType.TEXT_BLOCK,
            page_number=1,
            text_preview="Some text",
        )
        assert node.node_id == "text_0"
        assert node.element_type == ElementType.TEXT_BLOCK

    def test_with_bbox(self):
        node = LayoutNode(
            node_id="figure_0",
            element_type=ElementType.FIGURE,
            page_number=5,
            bbox=BoundingBox(10, 20, 300, 400),
        )
        assert node.bbox is not None
        assert node.bbox.width == 290


class TestPageLayout:
    def test_get_node(self):
        layout = PageLayout(page_number=1, nodes=[
            LayoutNode("a", ElementType.TEXT_BLOCK, 1),
            LayoutNode("b", ElementType.TABLE, 1),
        ])
        assert layout.get_node("a") is not None
        assert layout.get_node("z") is None

    def test_get_edges_from(self):
        layout = PageLayout(page_number=1, edges=[
            LayoutEdge("a", "b", RelationshipType.CAPTION_OF),
            LayoutEdge("a", "c", RelationshipType.REFERENCES),
            LayoutEdge("d", "e", RelationshipType.ABOVE),
        ])
        assert len(layout.get_edges_from("a")) == 2
        assert len(layout.get_edges_from("d")) == 1
        assert len(layout.get_edges_from("z")) == 0

    def test_get_related(self):
        layout = PageLayout(
            page_number=1,
            nodes=[
                LayoutNode("cap", ElementType.CAPTION, 1),
                LayoutNode("fig", ElementType.FIGURE, 1),
                LayoutNode("tbl", ElementType.TABLE, 1),
            ],
            edges=[
                LayoutEdge("cap", "fig", RelationshipType.CAPTION_OF),
            ],
        )
        related = layout.get_related("cap", RelationshipType.CAPTION_OF)
        assert len(related) == 1
        assert related[0].node_id == "fig"


class TestDocumentLayout:
    def test_page_count(self):
        doc = DocumentLayout(pages=[
            PageLayout(1),
            PageLayout(2),
        ])
        assert doc.page_count == 2

    def test_get_page(self):
        doc = DocumentLayout(pages=[
            PageLayout(1),
            PageLayout(5),
        ])
        assert doc.get_page(5) is not None
        assert doc.get_page(3) is None

    def test_total_counts(self):
        doc = DocumentLayout(pages=[
            PageLayout(1, nodes=[
                LayoutNode("a", ElementType.TEXT_BLOCK, 1),
            ], edges=[
                LayoutEdge("a", "b", RelationshipType.ABOVE),
            ]),
            PageLayout(2, nodes=[
                LayoutNode("c", ElementType.TABLE, 2),
                LayoutNode("d", ElementType.FIGURE, 2),
            ]),
        ])
        assert doc.total_nodes == 3
        assert doc.total_edges == 1


class TestBuildPageLayout:
    def test_text_blocks_become_nodes(self):
        layout = build_page_layout(
            page_number=1,
            text_blocks=[
                {"text": "Introduction to the study", "block_index": 0},
                {"text": "Methods and materials used", "block_index": 1},
            ],
            tables=[],
            figures=[],
        )
        assert len(layout.nodes) == 2
        assert layout.nodes[0].node_id == "text_0"
        assert layout.nodes[1].node_id == "text_1"

    def test_tables_become_nodes(self):
        layout = build_page_layout(
            page_number=3,
            text_blocks=[],
            tables=[{"table_index": 0, "header_row": '["A","B"]'}],
            figures=[],
        )
        assert len(layout.nodes) == 1
        assert layout.nodes[0].element_type == ElementType.TABLE

    def test_figures_become_nodes(self):
        layout = build_page_layout(
            page_number=12,
            text_blocks=[],
            tables=[],
            figures=[{
                "bbox": {"x0": 50, "y0": 100, "x1": 500, "y1": 600},
                "figure_type_hint": "study_design",
                "caption": "Figure 1: Study Design",
            }],
        )
        assert len(layout.nodes) == 1
        fig = layout.nodes[0]
        assert fig.element_type == ElementType.FIGURE
        assert fig.bbox is not None
        assert fig.metadata["figure_type_hint"] == "study_design"

    def test_caption_linked_to_figure(self):
        layout = build_page_layout(
            page_number=1,
            text_blocks=[
                {"text": "Figure 1: Study Design Schema", "block_index": 0},
                {"text": "More prose text here", "block_index": 1},
            ],
            tables=[],
            figures=[{
                "bbox": {"x0": 0, "y0": 0, "x1": 500, "y1": 400},
                "figure_type_hint": "study_design",
                "caption": "",
            }],
        )
        # Caption node should link to figure node
        caption_edges = [
            e for e in layout.edges
            if e.relationship == RelationshipType.CAPTION_OF
        ]
        assert len(caption_edges) == 1
        assert caption_edges[0].target_id == "figure_0"

    def test_caption_linked_to_table(self):
        layout = build_page_layout(
            page_number=1,
            text_blocks=[
                {"text": "Table 3: Dosing Schedule", "block_index": 0},
            ],
            tables=[{"table_index": 0, "header_row": "[]"}],
            figures=[],
        )
        caption_edges = [
            e for e in layout.edges
            if e.relationship == RelationshipType.CAPTION_OF
        ]
        assert len(caption_edges) == 1
        assert caption_edges[0].target_id == "table_0"

    def test_footnote_linked_to_table(self):
        # Footnotes at bottom of page (high block_index)
        blocks = [
            {"text": "Protocol Design Section", "block_index": 0},
            {"text": "Some assessment data", "block_index": 1},
            {"text": "Additional context here", "block_index": 2},
            {"text": "More details about methods", "block_index": 3},
            {"text": "1. CBC with differential required", "block_index": 4},
        ]
        layout = build_page_layout(
            page_number=1,
            text_blocks=blocks,
            tables=[{"table_index": 0, "header_row": "[]"}],
            figures=[],
        )
        ref_edges = [
            e for e in layout.edges
            if e.relationship == RelationshipType.REFERENCES
        ]
        assert len(ref_edges) >= 1
        assert ref_edges[0].target_id == "table_0"

    def test_no_edges_when_no_relationships(self):
        layout = build_page_layout(
            page_number=1,
            text_blocks=[
                {"text": "Just plain text here", "block_index": 0},
            ],
            tables=[],
            figures=[],
        )
        assert len(layout.edges) == 0

    def test_in_table_metadata_preserved(self):
        layout = build_page_layout(
            page_number=1,
            text_blocks=[
                {"text": "Table content text", "block_index": 0, "in_table": True},
            ],
            tables=[],
            figures=[],
        )
        assert layout.nodes[0].metadata["in_table"] is True


class TestBuildDocumentLayout:
    def test_multi_page(self):
        doc = build_document_layout([
            {
                "page_number": 1,
                "text_blocks": [{"text": "Page 1 text", "block_index": 0}],
                "tables": [],
                "figures": [],
            },
            {
                "page_number": 2,
                "text_blocks": [{"text": "Page 2 text", "block_index": 0}],
                "tables": [{"table_index": 0, "header_row": "[]"}],
                "figures": [],
            },
        ])
        assert doc.page_count == 2
        assert doc.total_nodes == 3

    def test_skips_invalid_pages(self):
        doc = build_document_layout([
            {"page_number": 0, "text_blocks": [], "tables": [], "figures": []},
            {"page_number": 1, "text_blocks": [], "tables": [], "figures": []},
        ])
        assert doc.page_count == 1

    def test_empty_input(self):
        doc = build_document_layout([])
        assert doc.page_count == 0
        assert doc.total_nodes == 0


class TestElementTypeClassification:
    def test_caption_detected(self):
        layout = build_page_layout(
            page_number=1,
            text_blocks=[
                {"text": "Figure 1: Overview of Treatment Arms", "block_index": 0},
            ],
            tables=[], figures=[],
        )
        assert layout.nodes[0].element_type == ElementType.CAPTION

    def test_table_caption_detected(self):
        layout = build_page_layout(
            page_number=1,
            text_blocks=[
                {"text": "Table 5. Dose Modification Guidelines", "block_index": 0},
            ],
            tables=[], figures=[],
        )
        assert layout.nodes[0].element_type == ElementType.CAPTION

    def test_regular_text_not_caption(self):
        layout = build_page_layout(
            page_number=1,
            text_blocks=[
                {"text": "The primary endpoint is overall survival", "block_index": 0},
            ],
            tables=[], figures=[],
        )
        assert layout.nodes[0].element_type == ElementType.TEXT_BLOCK
