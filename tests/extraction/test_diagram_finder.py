"""Tests for PTCV-243: DiagramFinder prototype.

Tests verify shape detection, connector tracing, text association,
graph assembly, diagram classification, and edge cases.
"""

import json
import math
import pytest
from unittest.mock import MagicMock

from ptcv.extraction.diagram_finder import (
    DiagramFinder,
    DiagramNode,
    DiagramEdge,
    Diagram,
)


def _make_rect(
    x0: float, top: float, x1: float, bottom: float,
    fill: Any = None, stroke: Any = (0, 0, 0),
) -> dict:
    """Create a mock pdfplumber rect object."""
    return {
        "x0": x0, "top": top, "x1": x1, "bottom": bottom,
        "width": x1 - x0, "height": bottom - top,
        "non_stroking_color": fill,
        "stroking_color": stroke,
    }


def _make_line(
    x0: float, top: float, x1: float, bottom: float,
) -> dict:
    """Create a mock pdfplumber line object."""
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom}


def _make_char(x0: float, top: float, text: str) -> dict:
    """Create a mock pdfplumber char object."""
    return {"x0": x0, "top": top, "x1": x0 + 6, "bottom": top + 10, "text": text}


def _make_page(
    rects: list | None = None,
    lines: list | None = None,
    curves: list | None = None,
    chars: list | None = None,
    width: float = 612,
    height: float = 792,
    page_number: int = 1,
) -> MagicMock:
    """Create a mock pdfplumber Page."""
    page = MagicMock()
    page.rects = rects or []
    page.lines = lines or []
    page.curves = curves or []
    page.chars = chars or []
    page.width = width
    page.height = height
    page.page_number = page_number
    return page


# -- Flowchart fixture: 4 boxes connected vertically --
#
#   [Screening]
#       |
#   [Randomization]
#      / \
#  [Arm A] [Arm B]

def _make_flowchart_page() -> MagicMock:
    """Create a page with a simple 4-node flowchart."""
    rects = [
        _make_rect(200, 100, 400, 150),   # Box 0: Screening
        _make_rect(200, 200, 400, 250),   # Box 1: Randomization
        _make_rect(100, 350, 250, 400),   # Box 2: Arm A
        _make_rect(350, 350, 500, 400),   # Box 3: Arm B
    ]
    lines = [
        _make_line(300, 150, 300, 200),   # Box 0 -> Box 1
        _make_line(250, 250, 175, 350),   # Box 1 -> Box 2
        _make_line(350, 250, 425, 350),   # Box 1 -> Box 3
    ]
    chars = [
        # "Screening" inside Box 0
        _make_char(220, 115, "S"), _make_char(226, 115, "c"),
        _make_char(232, 115, "r"), _make_char(238, 115, "e"),
        _make_char(244, 115, "e"), _make_char(250, 115, "n"),
        # "Arm A" inside Box 2
        _make_char(120, 365, "A"), _make_char(126, 365, "r"),
        _make_char(132, 365, "m"), _make_char(140, 365, "A"),
        # "Arm B" inside Box 3
        _make_char(370, 365, "A"), _make_char(376, 365, "r"),
        _make_char(382, 365, "m"), _make_char(390, 365, "B"),
    ]
    return _make_page(rects=rects, lines=lines, chars=chars)


from typing import Any


class TestDiagramNode:
    """Tests for DiagramNode dataclass."""

    def test_creation(self):
        node = DiagramNode(0, "rect", "Test", 10, 20, 100, 60)
        assert node.node_id == 0
        assert node.shape_type == "rect"
        assert node.label == "Test"

    def test_center(self):
        node = DiagramNode(0, "rect", "", 0, 0, 100, 50)
        assert node.cx == 50.0
        assert node.cy == 25.0

    def test_bbox(self):
        node = DiagramNode(0, "rect", "", 10, 20, 110, 70)
        assert node.bbox == (10, 20, 110, 70)

    def test_dimensions(self):
        node = DiagramNode(0, "rect", "", 0, 0, 80, 40)
        assert node.width == 80
        assert node.height == 40


class TestDiagram:
    """Tests for Diagram dataclass."""

    def test_counts(self):
        nodes = [DiagramNode(i, "rect", f"N{i}", 0, 0, 10, 10) for i in range(3)]
        edges = [DiagramEdge(0, 1), DiagramEdge(1, 2)]
        d = Diagram(nodes=nodes, edges=edges)
        assert d.node_count == 3
        assert d.edge_count == 2

    def test_adjacency_list(self):
        nodes = [DiagramNode(i, "rect", "", 0, 0, 10, 10) for i in range(3)]
        edges = [DiagramEdge(0, 1), DiagramEdge(0, 2)]
        d = Diagram(nodes=nodes, edges=edges)
        adj = d.to_adjacency_list()
        assert adj[0] == [1, 2]
        assert adj[1] == []
        assert adj[2] == []

    def test_to_dict(self):
        nodes = [DiagramNode(0, "rect", "Start", 10, 20, 100, 60)]
        edges = [DiagramEdge(0, 1, "yes")]
        d = Diagram(nodes=nodes, edges=edges, diagram_type="flowchart", page_number=5)
        data = d.to_dict()
        assert data["diagram_type"] == "flowchart"
        assert data["page_number"] == 5
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["label"] == "Start"
        assert data["edges"][0]["label"] == "yes"


class TestShapeDetection:
    """Tests for DiagramFinder shape detection."""

    def test_detects_valid_rects(self):
        finder = DiagramFinder()
        rects = [_make_rect(100, 100, 250, 160)]  # 150x60
        shapes = finder._detect_shapes(rects, [], 612)
        assert len(shapes) == 1
        assert shapes[0].shape_type == "rect"

    def test_filters_tiny_rects(self):
        finder = DiagramFinder()
        rects = [_make_rect(0, 0, 5, 5)]  # Too small
        shapes = finder._detect_shapes(rects, [], 612)
        assert len(shapes) == 0

    def test_filters_page_borders(self):
        finder = DiagramFinder()
        rects = [_make_rect(0, 0, 600, 50)]  # 98% of page width
        shapes = finder._detect_shapes(rects, [], 612)
        assert len(shapes) == 0

    def test_filters_flat_table_rows(self):
        finder = DiagramFinder()
        # Width/height ratio = 400/10 = 40 (way over max_aspect_ratio=8)
        rects = [_make_rect(50, 100, 450, 110)]
        shapes = finder._detect_shapes(rects, [], 612)
        assert len(shapes) == 0

    def test_multiple_shapes(self):
        finder = DiagramFinder()
        rects = [
            _make_rect(50, 50, 200, 100),
            _make_rect(50, 150, 200, 200),
            _make_rect(50, 250, 200, 300),
        ]
        shapes = finder._detect_shapes(rects, [], 612)
        assert len(shapes) == 3
        assert [s.node_id for s in shapes] == [0, 1, 2]


class TestConnectorTracing:
    """Tests for DiagramFinder connector tracing."""

    def test_connects_adjacent_boxes(self):
        finder = DiagramFinder()
        shapes = [
            DiagramNode(0, "rect", "", 100, 100, 200, 150),
            DiagramNode(1, "rect", "", 100, 200, 200, 250),
        ]
        # Line from bottom of box 0 to top of box 1
        lines = [_make_line(150, 150, 150, 200)]

        edges = finder._trace_connectors(lines, [], shapes)
        assert len(edges) == 1
        assert edges[0].source_id == 0  # Higher box = source
        assert edges[0].target_id == 1

    def test_no_connection_far_apart(self):
        finder = DiagramFinder(connector_snap_distance=10)
        shapes = [
            DiagramNode(0, "rect", "", 0, 0, 50, 50),
            DiagramNode(1, "rect", "", 400, 400, 450, 450),
        ]
        # Line not near either shape
        lines = [_make_line(200, 200, 210, 210)]

        edges = finder._trace_connectors(lines, [], shapes)
        assert len(edges) == 0

    def test_no_self_loops(self):
        finder = DiagramFinder()
        shapes = [DiagramNode(0, "rect", "", 100, 100, 200, 200)]
        # Line starts and ends near the same shape
        lines = [_make_line(150, 100, 150, 200)]

        edges = finder._trace_connectors(lines, [], shapes)
        assert len(edges) == 0

    def test_direction_inferred_from_position(self):
        finder = DiagramFinder()
        shapes = [
            DiagramNode(0, "rect", "", 100, 300, 200, 350),  # Lower
            DiagramNode(1, "rect", "", 100, 100, 200, 150),  # Higher
        ]
        # Line from lower to higher
        lines = [_make_line(150, 300, 150, 150)]

        edges = finder._trace_connectors(lines, [], shapes)
        assert len(edges) == 1
        # Higher shape (node 1) should be source
        assert edges[0].source_id == 1
        assert edges[0].target_id == 0


class TestPointToRectDistance:
    """Tests for point-to-rectangle distance calculation."""

    def test_inside_rect(self):
        dist = DiagramFinder._point_to_rect_distance(50, 50, 0, 0, 100, 100)
        assert dist == 0.0

    def test_on_edge(self):
        dist = DiagramFinder._point_to_rect_distance(0, 50, 0, 0, 100, 100)
        assert dist == 0.0

    def test_outside_horizontally(self):
        dist = DiagramFinder._point_to_rect_distance(110, 50, 0, 0, 100, 100)
        assert dist == 10.0

    def test_outside_corner(self):
        dist = DiagramFinder._point_to_rect_distance(103, 104, 0, 0, 100, 100)
        assert abs(dist - 5.0) < 0.01  # sqrt(3^2 + 4^2) = 5


class TestTextAssociation:
    """Tests for text-to-shape association."""

    def test_chars_inside_shape(self):
        finder = DiagramFinder()
        shapes = [DiagramNode(0, "rect", "", 100, 100, 300, 160)]
        chars = [
            _make_char(120, 120, "H"),
            _make_char(126, 120, "i"),
        ]
        finder._associate_text(shapes, chars)
        assert shapes[0].label == "Hi"

    def test_chars_outside_shape_ignored(self):
        finder = DiagramFinder()
        shapes = [DiagramNode(0, "rect", "", 100, 100, 200, 150)]
        chars = [_make_char(300, 300, "X")]  # Outside
        finder._associate_text(shapes, chars)
        assert shapes[0].label == ""

    def test_multiline_text(self):
        finder = DiagramFinder()
        shapes = [DiagramNode(0, "rect", "", 100, 100, 300, 200)]
        chars = [
            _make_char(120, 120, "A"),
            _make_char(126, 120, "B"),
            _make_char(120, 140, "C"),  # Different Y = new line
            _make_char(126, 140, "D"),
        ]
        finder._associate_text(shapes, chars)
        assert shapes[0].label == "AB\nCD"


class TestGraphAssembly:
    """Tests for connected component grouping."""

    def test_single_component(self):
        finder = DiagramFinder(min_nodes=2)
        shapes = [
            DiagramNode(0, "rect", "A", 0, 0, 50, 50),
            DiagramNode(1, "rect", "B", 0, 100, 50, 150),
            DiagramNode(2, "rect", "C", 0, 200, 50, 250),
        ]
        edges = [DiagramEdge(0, 1), DiagramEdge(1, 2)]

        diagrams = finder._assemble_diagrams(shapes, edges, page_number=3)
        assert len(diagrams) == 1
        assert diagrams[0].node_count == 3
        assert diagrams[0].edge_count == 2
        assert diagrams[0].page_number == 3

    def test_two_separate_components(self):
        finder = DiagramFinder(min_nodes=2)
        shapes = [
            DiagramNode(0, "rect", "A", 0, 0, 50, 50),
            DiagramNode(1, "rect", "B", 0, 100, 50, 150),
            DiagramNode(2, "rect", "C", 300, 0, 350, 50),
            DiagramNode(3, "rect", "D", 300, 100, 350, 150),
        ]
        edges = [DiagramEdge(0, 1), DiagramEdge(2, 3)]

        diagrams = finder._assemble_diagrams(shapes, edges, page_number=1)
        assert len(diagrams) == 2

    def test_small_component_filtered(self):
        finder = DiagramFinder(min_nodes=3)
        shapes = [
            DiagramNode(0, "rect", "A", 0, 0, 50, 50),
            DiagramNode(1, "rect", "B", 0, 100, 50, 150),
        ]
        edges = [DiagramEdge(0, 1)]

        diagrams = finder._assemble_diagrams(shapes, edges, page_number=1)
        assert len(diagrams) == 0  # Only 2 nodes, min is 3

    def test_diagram_bbox(self):
        finder = DiagramFinder(min_nodes=2)
        shapes = [
            DiagramNode(0, "rect", "", 10, 20, 100, 60),
            DiagramNode(1, "rect", "", 50, 80, 200, 120),
        ]
        edges = [DiagramEdge(0, 1)]

        diagrams = finder._assemble_diagrams(shapes, edges, page_number=1)
        assert diagrams[0].bbox == (10, 20, 200, 120)


class TestDiagramClassification:
    """Tests for diagram type inference."""

    def test_consort_detection(self):
        nodes = [
            DiagramNode(i, "rect", label, 0, 0, 10, 10)
            for i, label in enumerate([
                "Enrolled (n=100)", "Allocated to treatment",
                "Follow-up assessment", "Analyzed (n=95)",
            ])
        ]
        d = Diagram(nodes=nodes, edges=[DiagramEdge(0, 1)])
        assert DiagramFinder._classify_diagram(d) == "consort"

    def test_study_design_detection(self):
        nodes = [
            DiagramNode(i, "rect", label, 0, 0, 10, 10)
            for i, label in enumerate([
                "Screening Period", "Treatment Period",
                "End of Study", "Arm A: Drug X",
            ])
        ]
        d = Diagram(nodes=nodes, edges=[DiagramEdge(0, 1)])
        assert DiagramFinder._classify_diagram(d) == "study_design"

    def test_generic_flowchart(self):
        nodes = [
            DiagramNode(i, "rect", f"Step {i}", 0, 0, 10, 10)
            for i in range(4)
        ]
        edges = [DiagramEdge(0, 1), DiagramEdge(1, 2)]
        d = Diagram(nodes=nodes, edges=edges)
        assert DiagramFinder._classify_diagram(d) == "flowchart"

    def test_unknown_no_edges(self):
        nodes = [
            DiagramNode(i, "rect", f"Box {i}", 0, 0, 10, 10)
            for i in range(4)
        ]
        d = Diagram(nodes=nodes, edges=[])
        assert DiagramFinder._classify_diagram(d) == "unknown"


class TestFindDiagramsIntegration:
    """Integration tests for find_diagrams on mock pages."""

    def test_flowchart_page(self):
        """Full pipeline on a 4-node flowchart page."""
        page = _make_flowchart_page()
        finder = DiagramFinder(min_nodes=3)

        diagrams = finder.find_diagrams(page)

        assert len(diagrams) >= 1
        diagram = diagrams[0]
        assert diagram.node_count == 4
        assert diagram.edge_count == 3
        assert diagram.page_number == 1
        # Check labels were extracted
        labels = {n.label for n in diagram.nodes if n.label}
        assert len(labels) >= 1  # At least some text found

    def test_empty_page(self):
        """Empty page returns no diagrams."""
        page = _make_page()
        finder = DiagramFinder()
        assert finder.find_diagrams(page) == []

    def test_page_with_only_tiny_rects(self):
        """Page with only decorative elements returns no diagrams."""
        page = _make_page(rects=[
            _make_rect(0, 0, 5, 5),
            _make_rect(10, 10, 15, 15),
        ])
        finder = DiagramFinder()
        assert finder.find_diagrams(page) == []

    def test_diagram_type_classified(self):
        """Diagram type is classified after assembly."""
        page = _make_flowchart_page()
        finder = DiagramFinder(min_nodes=3)
        diagrams = finder.find_diagrams(page)
        assert diagrams[0].diagram_type in (
            "flowchart", "study_design", "consort", "unknown"
        )

    def test_vision_fallback_disabled(self):
        """No raster fallback when enable_vision_fallback=False."""
        page = _make_page(
            rects=[],  # No vector shapes
            chars=[],
        )
        # Add a large image that would trigger raster detection
        page.images = [{
            "x0": 50, "top": 50, "x1": 500, "bottom": 400,
            "width": 450, "height": 350,
        }]
        finder = DiagramFinder(enable_vision_fallback=False)
        diagrams = finder.find_diagrams(page)
        assert diagrams == []

    def test_vision_fallback_skips_small_images(self):
        """Small images are not candidates for raster diagram detection."""
        page = _make_page(rects=[], chars=[])
        page.images = [{
            "x0": 10, "top": 10, "x1": 50, "bottom": 40,
            "width": 40, "height": 30,
        }]
        finder = DiagramFinder(enable_vision_fallback=True)
        # Should not attempt Vision (image too small) — no API key needed
        diagrams = finder._find_raster_diagrams(page)
        assert diagrams == []

    def test_vision_fallback_no_api_key(self):
        """Raster fallback gracefully skips when no API key."""
        import os
        page = _make_page(rects=[], chars=[])
        page.images = [{
            "x0": 50, "top": 50, "x1": 500, "bottom": 400,
            "width": 450, "height": 350,
        }]
        # Ensure no API key
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            finder = DiagramFinder(enable_vision_fallback=True)
            diagrams = finder._find_raster_diagrams(page)
            assert diagrams == []  # Graceful skip
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key


class TestVisionResponseParsing:
    """Tests for _parse_vision_response."""

    def test_valid_json(self):
        finder = DiagramFinder()
        img = {"x0": 50, "top": 50, "x1": 500, "bottom": 400}
        response = json.dumps({
            "diagram_type": "consort",
            "nodes": [
                {"node_id": 0, "shape_type": "rect", "label": "Enrolled (n=100)"},
                {"node_id": 1, "shape_type": "rect", "label": "Randomized (n=80)"},
                {"node_id": 2, "shape_type": "rect", "label": "Arm A (n=40)"},
                {"node_id": 3, "shape_type": "rect", "label": "Arm B (n=40)"},
            ],
            "edges": [
                {"source_id": 0, "target_id": 1},
                {"source_id": 1, "target_id": 2},
                {"source_id": 1, "target_id": 3},
            ],
        })

        diagram = finder._parse_vision_response(response, page_number=9, img=img)

        assert diagram is not None
        assert diagram.node_count == 4
        assert diagram.edge_count == 3
        assert diagram.diagram_type == "consort"
        assert diagram.page_number == 9
        assert diagram.nodes[0].label == "Enrolled (n=100)"

    def test_json_with_markdown_fencing(self):
        finder = DiagramFinder()
        img = {"x0": 0, "top": 0, "x1": 100, "bottom": 100}
        response = '```json\n{"diagram_type": "flowchart", "nodes": [{"node_id": 0, "label": "Start"}], "edges": []}\n```'

        diagram = finder._parse_vision_response(response, page_number=1, img=img)

        assert diagram is not None
        assert diagram.node_count == 1

    def test_invalid_json(self):
        finder = DiagramFinder()
        img = {"x0": 0, "top": 0, "x1": 100, "bottom": 100}

        diagram = finder._parse_vision_response("not json", page_number=1, img=img)

        assert diagram is None

    def test_empty_nodes(self):
        finder = DiagramFinder()
        img = {"x0": 0, "top": 0, "x1": 100, "bottom": 100}
        response = json.dumps({"diagram_type": "flowchart", "nodes": [], "edges": []})

        diagram = finder._parse_vision_response(response, page_number=1, img=img)

        assert diagram is None  # No nodes = no diagram

    def test_invalid_edge_references_filtered(self):
        finder = DiagramFinder()
        img = {"x0": 0, "top": 0, "x1": 100, "bottom": 100}
        response = json.dumps({
            "diagram_type": "flowchart",
            "nodes": [{"node_id": 0, "label": "A"}, {"node_id": 1, "label": "B"}],
            "edges": [
                {"source_id": 0, "target_id": 1},
                {"source_id": 0, "target_id": 99},  # Invalid target
            ],
        })

        diagram = finder._parse_vision_response(response, page_number=1, img=img)

        assert diagram is not None
        assert diagram.edge_count == 1  # Invalid edge filtered
