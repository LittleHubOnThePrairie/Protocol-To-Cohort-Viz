"""Tests for figure captioning and layout graph in Stage 1 (PTCV-263).

GHERKIN Scenarios:
  Feature: Figure captioning and layout graph in Stage 1

    Scenario: Figures detected and captioned
      Given a protocol PDF with embedded figures
      When extract_protocol_index() runs with PTCV_ENABLE_FIGURE_VISION=1
      Then ProtocolIndex.figures contains detected figures with Vision-generated captions

    Scenario: Layout graph built
      Given a protocol page with text, a table, and a figure
      When extract_protocol_index() runs
      Then ProtocolIndex.layout_graph contains spatial relationships
      And footnotes are linked to their nearest table
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.toc_extractor import ProtocolIndex, extract_protocol_index


def _make_mock_fitz_doc(page_texts: list[str]) -> MagicMock:
    """Create a mock fitz.Document from a list of page text strings."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.get_text.return_value = text
        pages.append(page)

    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])
    doc.close = MagicMock()
    return doc


# ---------------------------------------------------------------------------
# ProtocolIndex field tests
# ---------------------------------------------------------------------------


class TestProtocolIndexFields:
    """ProtocolIndex should have figures and layout_graph fields."""

    def test_figures_field_default_empty(self):
        idx = ProtocolIndex(
            source_path="/tmp/test.pdf",
            page_count=1,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
        )
        assert idx.figures == []

    def test_layout_graph_field_default_none(self):
        idx = ProtocolIndex(
            source_path="/tmp/test.pdf",
            page_count=1,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
        )
        assert idx.layout_graph is None

    def test_figures_field_populated(self):
        figures = [
            {
                "page_number": 3,
                "bbox": {"x0": 10, "y0": 20, "x1": 500, "y1": 400},
                "figure_type_hint": "study_design",
                "detection_method": "image_object",
                "caption": "Figure 1: Study Design",
            }
        ]
        idx = ProtocolIndex(
            source_path="/tmp/test.pdf",
            page_count=10,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
            figures=figures,
        )
        assert len(idx.figures) == 1
        assert idx.figures[0]["caption"] == "Figure 1: Study Design"

    def test_layout_graph_populated(self):
        layout_graph = {
            "page_count": 2,
            "total_nodes": 5,
            "total_edges": 2,
            "pages": [],
        }
        idx = ProtocolIndex(
            source_path="/tmp/test.pdf",
            page_count=2,
            toc_entries=[],
            section_headers=[],
            content_spans={},
            full_text="",
            toc_found=False,
            toc_pages=[],
            layout_graph=layout_graph,
        )
        assert idx.layout_graph is not None
        assert idx.layout_graph["total_nodes"] == 5


# ---------------------------------------------------------------------------
# Layout graph relationship tests (unit-level)
# ---------------------------------------------------------------------------


class TestLayoutGraphFootnoteRelationships:
    """Footnotes should be linked to their nearest table."""

    def test_footnote_linked_to_table(self):
        from ptcv.extraction.layout_graph import (
            RelationshipType,
            build_page_layout,
        )

        text_blocks = [
            {"text": "Some header text", "block_index": 0},
            {"text": "Body paragraph content here", "block_index": 1},
            {"text": "More content", "block_index": 2},
            {"text": "Another block", "block_index": 3},
            {"text": "1. This is a footnote about dosing", "block_index": 4},
        ]
        tables = [
            {
                "table_index": 0,
                "header_row": '["Visit","Day 1","Week 2"]',
                "page_number": 1,
            }
        ]
        figures: list[dict] = []

        layout = build_page_layout(
            page_number=1,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
        )

        # Find REFERENCES edges (footnote → table)
        ref_edges = [
            e for e in layout.edges
            if e.relationship == RelationshipType.REFERENCES
        ]
        assert len(ref_edges) >= 1
        # Target should be the table node
        assert ref_edges[0].target_id == "table_0"

    def test_caption_linked_to_figure(self):
        from ptcv.extraction.layout_graph import (
            RelationshipType,
            build_page_layout,
        )

        text_blocks = [
            {"text": "Figure 1: Study Design Schema", "block_index": 0},
            {"text": "Body text", "block_index": 1},
        ]
        tables: list[dict] = []
        figures = [
            {
                "page_number": 1,
                "bbox": {"x0": 50, "y0": 100, "x1": 500, "y1": 400},
                "figure_type_hint": "study_design",
                "caption": "",
            }
        ]

        layout = build_page_layout(
            page_number=1,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
        )

        caption_edges = [
            e for e in layout.edges
            if e.relationship == RelationshipType.CAPTION_OF
        ]
        assert len(caption_edges) >= 1
        assert caption_edges[0].target_id == "figure_0"


# ---------------------------------------------------------------------------
# Figure detection integration (mocked)
# ---------------------------------------------------------------------------


class TestFigureDetectionIntegration:
    """Verify figure detection is wired into extract_protocol_index()."""

    def test_figures_populated_from_detect_figures(self, tmp_path: Path):
        """When detect_figures returns results, ProtocolIndex.figures is populated."""
        from ptcv.extraction.figure_detector import (
            BoundingBox,
            DetectedFigure,
            FigureDetectionResult,
        )

        mock_figures = FigureDetectionResult(figures=[
            DetectedFigure(
                page_number=1,
                bbox=BoundingBox(x0=50, y0=100, x1=500, y1=400),
                figure_type_hint="study_design",
                detection_method="image_object",
                caption="Figure 1: Study Design",
            ),
        ])

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 minimal")

        mock_doc = _make_mock_fitz_doc(
            ["1. Introduction\nSome text here."]
        )
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with (
            patch.dict(sys.modules, {"fitz": mock_fitz}),
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "ptcv.extraction.figure_detector.detect_figures",
                return_value=mock_figures,
            ),
        ):
            result = extract_protocol_index(pdf_file)

        assert len(result.figures) == 1
        assert result.figures[0]["page_number"] == 1
        assert result.figures[0]["figure_type_hint"] == "study_design"
        assert result.figures[0]["caption"] == "Figure 1: Study Design"

    def test_layout_graph_built(self, tmp_path: Path):
        """ProtocolIndex.layout_graph should be populated."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 minimal")

        mock_doc = _make_mock_fitz_doc(
            ["1. Introduction\n\nSome body text here."]
        )
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with (
            patch.dict(sys.modules, {"fitz": mock_fitz}),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = extract_protocol_index(pdf_file)

        assert result.layout_graph is not None
        assert "page_count" in result.layout_graph
        assert "total_nodes" in result.layout_graph
        assert "total_edges" in result.layout_graph
        assert "pages" in result.layout_graph
        assert result.layout_graph["page_count"] >= 1


# ---------------------------------------------------------------------------
# Document layout serialization tests
# ---------------------------------------------------------------------------


class TestDocumentLayoutSerialization:
    """Layout graph dict should be well-formed."""

    def test_serialization_structure(self):
        from ptcv.extraction.layout_graph import build_document_layout

        pages_data = [
            {
                "page_number": 1,
                "text_blocks": [
                    {"text": "Figure 1: Design", "block_index": 0},
                    {"text": "Body paragraph", "block_index": 1},
                ],
                "tables": [
                    {"table_index": 0, "header_row": "[]", "page_number": 1},
                ],
                "figures": [
                    {
                        "page_number": 1,
                        "bbox": {"x0": 50, "y0": 100, "x1": 500, "y1": 400},
                        "figure_type_hint": "study_design",
                        "caption": "",
                    },
                ],
            },
        ]

        doc_layout = build_document_layout(pages_data)

        # Serialize as we do in the pipeline
        result = {
            "page_count": doc_layout.page_count,
            "total_nodes": doc_layout.total_nodes,
            "total_edges": doc_layout.total_edges,
            "pages": [
                {
                    "page_number": pl.page_number,
                    "nodes": [
                        {
                            "node_id": n.node_id,
                            "element_type": n.element_type.value,
                            "text_preview": n.text_preview,
                        }
                        for n in pl.nodes
                    ],
                    "edges": [
                        {
                            "source_id": e.source_id,
                            "target_id": e.target_id,
                            "relationship": e.relationship.value,
                        }
                        for e in pl.edges
                    ],
                }
                for pl in doc_layout.pages
            ],
        }

        assert result["page_count"] == 1
        assert result["total_nodes"] >= 3  # 2 text + 1 table + 1 figure
        # Caption "Figure 1: Design" should create a caption_of edge
        caption_edges = [
            e for page in result["pages"] for e in page["edges"]
            if e["relationship"] == "caption_of"
        ]
        assert len(caption_edges) >= 1
