"""DiagramFinder: Extract flowcharts and study diagrams from PDF pages.

PTCV-243: Extends pdfplumber's object extraction to detect and interpret
flowcharts, CONSORT flow diagrams, screening cascades, and study design
diagrams found in clinical trial protocols.

Follows the same architectural pattern as pdfplumber's ``TableFinder``:
raw PDF objects (rects, lines, curves, chars) are classified into
semantic diagram components (shapes, connectors, labels) and assembled
into a directed graph.

Pipeline::

    page.rects  ─┐
    page.lines  ─┤    1. Shape         2. Connector      3. Text          4. Graph
    page.curves ─┼─→  Detection   ──→  Tracing      ──→  Association ──→  Assembly
    page.chars  ─┘    (nodes)          (edges)           (labels)         (Diagram)

Usage::

    from ptcv.extraction.diagram_finder import DiagramFinder

    import pdfplumber
    with pdfplumber.open("protocol.pdf") as pdf:
        finder = DiagramFinder()
        for page in pdf.pages:
            diagrams = finder.find_diagrams(page)
            for diagram in diagrams:
                print(f"Found {diagram.diagram_type} with "
                      f"{len(diagram.nodes)} nodes")
                for node in diagram.nodes:
                    print(f"  [{node.shape_type}] {node.label}")
"""

import dataclasses
import io
import json
import logging
import math
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DiagramNode:
    """A shape in a diagram (process box, decision diamond, terminal).

    Attributes:
        node_id: Unique identifier within the diagram.
        shape_type: Shape classification ("rect", "diamond", "rounded_rect",
            "oval", "circle").
        label: Text content extracted from inside the shape.
        x0: Left edge coordinate.
        top: Top edge coordinate.
        x1: Right edge coordinate.
        bottom: Bottom edge coordinate.
        fill_color: Fill color tuple or None.
        stroke_color: Stroke color tuple or None.
    """

    node_id: int
    shape_type: str
    label: str
    x0: float
    top: float
    x1: float
    bottom: float
    fill_color: Any = None
    stroke_color: Any = None

    @property
    def cx(self) -> float:
        """Horizontal center."""
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        """Vertical center."""
        return (self.top + self.bottom) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x0, self.top, self.x1, self.bottom)


@dataclasses.dataclass
class DiagramEdge:
    """A directed connection between two diagram nodes.

    Attributes:
        source_id: Node ID of the source (arrow tail).
        target_id: Node ID of the target (arrow head).
        label: Optional text label on the connector.
        edge_type: "line", "arrow", or "curve".
    """

    source_id: int
    target_id: int
    label: str = ""
    edge_type: str = "arrow"


@dataclasses.dataclass
class Diagram:
    """A complete extracted diagram (flowchart, study design, CONSORT).

    Attributes:
        nodes: All detected shapes.
        edges: All directed connections.
        diagram_type: Inferred type ("flowchart", "consort", "study_design",
            "unknown").
        page_number: 1-based page number.
        bbox: Bounding box of the entire diagram on the page.
    """

    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    diagram_type: str = "unknown"
    page_number: int = 0
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_adjacency_list(self) -> dict[int, list[int]]:
        """Convert edges to an adjacency list."""
        adj: dict[int, list[int]] = {n.node_id: [] for n in self.nodes}
        for edge in self.edges:
            adj.setdefault(edge.source_id, []).append(edge.target_id)
        return adj

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "diagram_type": self.diagram_type,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "shape_type": n.shape_type,
                    "label": n.label,
                    "bbox": n.bbox,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "label": e.label,
                    "edge_type": e.edge_type,
                }
                for e in self.edges
            ],
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum shape dimensions (filters out tiny decorative elements)
_MIN_SHAPE_WIDTH = 20.0
_MIN_SHAPE_HEIGHT = 12.0

# Maximum shape dimensions (filters out page-level borders)
_MAX_SHAPE_WIDTH_RATIO = 0.9  # Max 90% of page width

# Proximity threshold for connecting lines to shapes (points)
_CONNECTOR_SNAP_DISTANCE = 15.0

# Minimum nodes to consider a group a diagram (not just decoration)
_MIN_NODES_FOR_DIAGRAM = 3

# Aspect ratio threshold: width/height > this = likely a table row, not a box
_MAX_ASPECT_RATIO = 8.0

# Raster diagram detection: minimum image dimensions (points) to consider
# as a potential diagram (filters out icons, logos, signatures)
_MIN_IMAGE_WIDTH = 150.0
_MIN_IMAGE_HEIGHT = 100.0

# Minimum fraction of page area for an image to be a diagram candidate
_MIN_IMAGE_PAGE_FRACTION = 0.05


# ---------------------------------------------------------------------------
# DiagramFinder
# ---------------------------------------------------------------------------


class DiagramFinder:
    """Detect and extract diagrams from pdfplumber Page objects.

    Processes a page's raw PDF objects (rects, lines, curves, chars)
    to identify shapes, trace connectors, associate text labels, and
    assemble directed graphs.

    Args:
        min_shape_width: Minimum shape width in points.
        min_shape_height: Minimum shape height in points.
        connector_snap_distance: Max distance to snap a line endpoint
            to a shape edge.
        min_nodes: Minimum nodes to form a diagram.
    """

    def __init__(
        self,
        min_shape_width: float = _MIN_SHAPE_WIDTH,
        min_shape_height: float = _MIN_SHAPE_HEIGHT,
        connector_snap_distance: float = _CONNECTOR_SNAP_DISTANCE,
        min_nodes: int = _MIN_NODES_FOR_DIAGRAM,
        enable_vision_fallback: bool = True,
        vision_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._min_w = min_shape_width
        self._min_h = min_shape_height
        self._snap_dist = connector_snap_distance
        self._min_nodes = min_nodes
        self._vision_fallback = enable_vision_fallback
        self._vision_model = vision_model

    def find_diagrams(self, page: Any) -> list[Diagram]:
        """Extract all diagrams from a pdfplumber Page.

        Tries vector-based detection first (rects/lines/curves). If no
        vector diagram is found but the page contains large embedded
        images, falls back to Vision API interpretation of raster
        diagrams.

        Args:
            page: A ``pdfplumber.page.Page`` instance.

        Returns:
            List of Diagram objects found on the page.
        """
        # Try vector-based detection first
        diagrams = self._find_vector_diagrams(page)

        # Fallback: check for raster diagrams (embedded images)
        if not diagrams and self._vision_fallback:
            raster_diagrams = self._find_raster_diagrams(page)
            diagrams.extend(raster_diagrams)

        return diagrams

    def _find_vector_diagrams(self, page: Any) -> list[Diagram]:
        """Detect diagrams from vector PDF objects (rects/lines/curves).

        This is the primary detection path for PDFs with native
        drawing operators.
        """
        page_width = float(page.width)

        # Step 1: Detect shapes from rects and curves
        shapes = self._detect_shapes(
            page.rects, page.curves, page_width,
        )

        if len(shapes) < self._min_nodes:
            return []

        # Step 2: Trace connectors from lines and curves
        connectors = self._trace_connectors(
            page.lines, page.curves, shapes,
        )

        # Step 3: Associate text with shapes
        self._associate_text(shapes, page.chars)

        # Step 4: Group connected shapes into diagrams
        diagrams = self._assemble_diagrams(
            shapes, connectors, page.page_number,
        )

        # Step 5: Classify diagram types
        for diagram in diagrams:
            diagram.diagram_type = self._classify_diagram(diagram)

        return diagrams

    def _find_raster_diagrams(self, page: Any) -> list[Diagram]:
        """Detect diagrams from embedded raster images via Vision API.

        For pages where the diagram is an embedded image (common when
        protocols paste PowerPoint/Visio exports), extracts the image
        and sends it to Claude Vision for interpretation.
        """
        page_width = float(page.width)
        page_height = float(page.height)
        page_area = page_width * page_height

        # Find candidate diagram images
        candidates = []
        for img in getattr(page, "images", []):
            img_w = float(img.get("x1", 0) - img.get("x0", 0))
            img_h = float(img.get("bottom", 0) - img.get("top", 0))

            if img_w < _MIN_IMAGE_WIDTH or img_h < _MIN_IMAGE_HEIGHT:
                continue
            if (img_w * img_h) / page_area < _MIN_IMAGE_PAGE_FRACTION:
                continue

            candidates.append(img)

        if not candidates:
            return []

        logger.info(
            "DiagramFinder: %d raster diagram candidate(s) on page %d",
            len(candidates),
            page.page_number,
        )

        diagrams: list[Diagram] = []
        for img in candidates:
            diagram = self._interpret_raster_diagram(img, page)
            if diagram is not None:
                diagrams.append(diagram)

        return diagrams

    def _interpret_raster_diagram(
        self,
        img: dict[str, Any],
        page: Any,
    ) -> Optional[Diagram]:
        """Send a raster image to Claude Vision for diagram interpretation.

        Renders the page region containing the image to PNG, sends to
        Claude Vision with a structured extraction prompt, and parses
        the response into DiagramNode/DiagramEdge objects.

        Args:
            img: pdfplumber image dict with bbox and stream data.
            page: The pdfplumber Page object.

        Returns:
            Diagram if Vision successfully interprets the image, else None.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.debug(
                "No ANTHROPIC_API_KEY; skipping raster diagram interpretation"
            )
            return None

        # Render page to image and crop to diagram region
        image_bytes = self._render_page_region(page, img)
        if not image_bytes:
            return None

        try:
            import anthropic
        except ImportError:
            logger.debug("anthropic package not installed; skipping Vision")
            return None

        import base64

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "This image shows a diagram from a clinical trial protocol. "
            "Analyze the diagram and extract its structure as JSON with "
            "this exact format:\n"
            "{\n"
            '  "diagram_type": "flowchart" or "consort" or "study_design",\n'
            '  "nodes": [\n'
            '    {"node_id": 0, "shape_type": "rect", "label": "text inside box"},\n'
            "    ...\n"
            "  ],\n"
            '  "edges": [\n'
            '    {"source_id": 0, "target_id": 1, "label": "optional edge label"},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Extract ALL boxes/shapes as nodes with their text labels\n"
            "- Extract ALL arrows/connectors as edges\n"
            "- node_id should be sequential integers starting at 0\n"
            "- source_id/target_id reference node_ids\n"
            "- Preserve the flow direction (typically top-to-bottom)\n"
            "- Include any text on arrows as edge labels\n"
            "- Respond with ONLY the JSON, no markdown fencing"
        )

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=self._vision_model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }],
            )

            response_text = response.content[0].text.strip()
            return self._parse_vision_response(
                response_text, page.page_number, img,
            )

        except Exception as e:
            logger.warning(
                "Vision API diagram interpretation failed: %s", e
            )
            return None

    def _render_page_region(
        self,
        page: Any,
        img: dict[str, Any],
    ) -> Optional[bytes]:
        """Render the page to PNG and crop to the image region.

        Uses PyMuPDF (fitz) for rendering since pdfplumber doesn't
        have a built-in page renderer.

        Args:
            page: pdfplumber Page.
            img: Image dict with bbox coordinates.

        Returns:
            PNG bytes of the cropped region, or None on failure.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.debug("PyMuPDF not available for page rendering")
            return None

        # Resolve PDF path from pdfplumber page
        pdf_path: Optional[str] = None
        try:
            pdf_path = str(page.pdf.stream.name)
        except (AttributeError, TypeError):
            pass

        if not pdf_path:
            try:
                pdf_path = str(page.pdf.path)
            except (AttributeError, TypeError):
                pass

        if not pdf_path:
            logger.debug("Cannot determine PDF path from pdfplumber page")
            return None

        try:
            doc = fitz.open(str(pdf_path))
            fitz_page = doc[page.page_number - 1]  # 0-indexed

            # Clip to diagram region directly during rendering
            # (avoids Pixmap crop constructor issues on Python 3.14)
            clip_rect = fitz.Rect(
                float(img.get("x0", 0)),
                float(img.get("top", 0)),
                float(img.get("x1", 0)),
                float(img.get("bottom", 0)),
            )

            # Render at 2x resolution for better Vision quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = fitz_page.get_pixmap(matrix=mat, clip=clip_rect)

            png_bytes = pix.tobytes("png")
            doc.close()

            logger.info(
                "Rendered diagram region: %dx%d pixels",
                pix.width,
                pix.height,
            )
            return png_bytes

        except Exception as e:
            logger.warning("Page rendering failed: %s", e)
            return None

    def _parse_vision_response(
        self,
        response_text: str,
        page_number: int,
        img: dict[str, Any],
    ) -> Optional[Diagram]:
        """Parse Claude Vision's JSON response into a Diagram.

        Args:
            response_text: JSON string from Vision API.
            page_number: Page number for the diagram.
            img: Image dict with bbox for diagram positioning.

        Returns:
            Diagram or None if parsing fails.
        """
        try:
            # Strip markdown fencing if present
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()

            data = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            logger.warning(
                "Failed to parse Vision response as JSON: %.100s",
                response_text,
            )
            return None

        # Build nodes
        nodes: list[DiagramNode] = []
        raw_nodes = data.get("nodes", [])
        for raw in raw_nodes:
            nodes.append(DiagramNode(
                node_id=raw.get("node_id", len(nodes)),
                shape_type=raw.get("shape_type", "rect"),
                label=raw.get("label", ""),
                x0=float(img.get("x0", 0)),
                top=float(img.get("top", 0)),
                x1=float(img.get("x1", 0)),
                bottom=float(img.get("bottom", 0)),
            ))

        if not nodes:
            return None

        # Build edges
        edges: list[DiagramEdge] = []
        raw_edges = data.get("edges", [])
        node_ids = {n.node_id for n in nodes}
        for raw in raw_edges:
            src = raw.get("source_id")
            tgt = raw.get("target_id")
            if src in node_ids and tgt in node_ids:
                edges.append(DiagramEdge(
                    source_id=src,
                    target_id=tgt,
                    label=raw.get("label", ""),
                    edge_type="arrow",
                ))

        diagram_type = data.get("diagram_type", "unknown")
        bbox = (
            float(img.get("x0", 0)),
            float(img.get("top", 0)),
            float(img.get("x1", 0)),
            float(img.get("bottom", 0)),
        )

        logger.info(
            "Vision interpreted raster diagram: %d nodes, %d edges, type=%s",
            len(nodes),
            len(edges),
            diagram_type,
        )

        return Diagram(
            nodes=nodes,
            edges=edges,
            diagram_type=diagram_type,
            page_number=page_number,
            bbox=bbox,
        )

    # ------------------------------------------------------------------
    # Step 1: Shape Detection
    # ------------------------------------------------------------------

    def _detect_shapes(
        self,
        rects: list[dict[str, Any]],
        curves: list[dict[str, Any]],
        page_width: float,
    ) -> list[DiagramNode]:
        """Classify rectangles and curves into diagram shapes.

        Filters out decorative elements (too small), table rows
        (too wide/flat), and page borders (too large).
        """
        shapes: list[DiagramNode] = []
        node_id = 0

        max_width = page_width * _MAX_SHAPE_WIDTH_RATIO

        for rect in rects:
            w = float(rect.get("width", 0) or (
                rect.get("x1", 0) - rect.get("x0", 0)
            ))
            h = float(rect.get("height", 0) or (
                rect.get("bottom", 0) - rect.get("top", 0)
            ))

            if w < self._min_w or h < self._min_h:
                continue
            if w > max_width:
                continue
            if h > 0 and w / h > _MAX_ASPECT_RATIO:
                continue

            shape_type = "rect"
            # Detect if this is a filled box (common in flowcharts)
            has_fill = bool(rect.get("non_stroking_color"))
            has_stroke = bool(rect.get("stroking_color"))

            shapes.append(DiagramNode(
                node_id=node_id,
                shape_type=shape_type,
                label="",
                x0=float(rect.get("x0", 0)),
                top=float(rect.get("top", 0)),
                x1=float(rect.get("x1", 0)),
                bottom=float(rect.get("bottom", 0)),
                fill_color=rect.get("non_stroking_color"),
                stroke_color=rect.get("stroking_color"),
            ))
            node_id += 1

        # Detect diamond shapes from curves (4-point paths)
        diamonds = self._detect_diamonds(curves)
        for bbox in diamonds:
            shapes.append(DiagramNode(
                node_id=node_id,
                shape_type="diamond",
                label="",
                x0=bbox[0],
                top=bbox[1],
                x1=bbox[2],
                bottom=bbox[3],
            ))
            node_id += 1

        return shapes

    def _detect_diamonds(
        self,
        curves: list[dict[str, Any]],
    ) -> list[tuple[float, float, float, float]]:
        """Detect diamond shapes from curve objects.

        A diamond is typically drawn as a closed path with 4 points
        forming a rotated square.
        """
        diamonds: list[tuple[float, float, float, float]] = []

        for curve in curves:
            pts = curve.get("pts", [])
            if not pts or len(pts) < 4 or len(pts) > 6:
                continue

            # Check if points form a diamond (4 corners, roughly
            # symmetric around center)
            xs = [p[0] for p in pts[:4]]
            ys = [p[1] for p in pts[:4]]

            width = max(xs) - min(xs)
            height = max(ys) - min(ys)

            if width < self._min_w or height < self._min_h:
                continue

            # Diamond test: center-of-mass should be near geometric center
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            geo_cx = (min(xs) + max(xs)) / 2
            geo_cy = (min(ys) + max(ys)) / 2

            if (abs(cx - geo_cx) < width * 0.2
                    and abs(cy - geo_cy) < height * 0.2
                    and 0.3 < width / max(height, 0.1) < 3.0):
                diamonds.append((min(xs), min(ys), max(xs), max(ys)))

        return diamonds

    # ------------------------------------------------------------------
    # Step 2: Connector Tracing
    # ------------------------------------------------------------------

    def _trace_connectors(
        self,
        lines: list[dict[str, Any]],
        curves: list[dict[str, Any]],
        shapes: list[DiagramNode],
    ) -> list[DiagramEdge]:
        """Match line endpoints to shapes to build directed edges.

        For each line, checks if one endpoint is near a shape edge
        and the other endpoint is near a different shape edge.
        Direction is inferred from vertical position (top = source,
        bottom = target for top-to-bottom flow).
        """
        edges: list[DiagramEdge] = []
        used_pairs: set[tuple[int, int]] = set()

        for line in lines:
            x0 = float(line.get("x0", 0))
            top = float(line.get("top", 0))
            x1 = float(line.get("x1", 0))
            bottom = float(line.get("bottom", 0))

            # Find which shapes each endpoint is near
            start_node = self._find_nearest_shape(
                x0, top, shapes,
            )
            end_node = self._find_nearest_shape(
                x1, bottom, shapes,
            )

            if (start_node is not None
                    and end_node is not None
                    and start_node != end_node):
                pair = (start_node, end_node)
                if pair not in used_pairs:
                    used_pairs.add(pair)

                    # Infer direction: the shape higher on the page
                    # is the source (top-to-bottom flow convention)
                    src_node = shapes[start_node]
                    tgt_node = shapes[end_node]

                    if src_node.cy <= tgt_node.cy:
                        source, target = start_node, end_node
                    else:
                        source, target = end_node, start_node

                    edges.append(DiagramEdge(
                        source_id=source,
                        target_id=target,
                        edge_type="arrow",
                    ))

        return edges

    def _find_nearest_shape(
        self,
        x: float,
        y: float,
        shapes: list[DiagramNode],
    ) -> Optional[int]:
        """Find the shape whose edge is nearest to point (x, y).

        Returns node_id or None if no shape is within snap distance.
        """
        best_id: Optional[int] = None
        best_dist = self._snap_dist

        for shape in shapes:
            dist = self._point_to_rect_distance(
                x, y,
                shape.x0, shape.top, shape.x1, shape.bottom,
            )
            if dist < best_dist:
                best_dist = dist
                best_id = shape.node_id

        return best_id

    @staticmethod
    def _point_to_rect_distance(
        px: float, py: float,
        rx0: float, ry0: float, rx1: float, ry1: float,
    ) -> float:
        """Minimum distance from point (px, py) to rectangle edges."""
        # Clamp point to rect bounds
        cx = max(rx0, min(px, rx1))
        cy = max(ry0, min(py, ry1))
        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    # ------------------------------------------------------------------
    # Step 3: Text Association
    # ------------------------------------------------------------------

    def _associate_text(
        self,
        shapes: list[DiagramNode],
        chars: list[dict[str, Any]],
    ) -> None:
        """Associate character objects with shapes by bbox containment.

        Mutates shapes in-place, setting their ``label`` attribute.
        """
        for shape in shapes:
            label_chars: list[tuple[float, float, str]] = []

            for char in chars:
                cx = float(char.get("x0", 0))
                cy = float(char.get("top", 0))
                text = char.get("text", "")

                # Check if char center is inside shape bbox
                if (shape.x0 <= cx <= shape.x1
                        and shape.top <= cy <= shape.bottom
                        and text.strip()):
                    label_chars.append((cy, cx, text))

            if label_chars:
                # Sort by vertical then horizontal position
                label_chars.sort()
                # Group into lines by vertical proximity
                lines: list[list[str]] = [[]]
                prev_y = label_chars[0][0]

                for cy, cx, text in label_chars:
                    if abs(cy - prev_y) > 3.0:
                        lines.append([])
                    lines[-1].append(text)
                    prev_y = cy

                shape.label = "\n".join(
                    "".join(line) for line in lines
                ).strip()

    # ------------------------------------------------------------------
    # Step 4: Graph Assembly
    # ------------------------------------------------------------------

    def _assemble_diagrams(
        self,
        shapes: list[DiagramNode],
        edges: list[DiagramEdge],
        page_number: int,
    ) -> list[Diagram]:
        """Group connected shapes into separate diagrams.

        Uses union-find to identify connected components. Each
        component with >= min_nodes shapes becomes a Diagram.
        """
        n = len(shapes)
        if n == 0:
            return []

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Union connected shapes
        for edge in edges:
            if edge.source_id < n and edge.target_id < n:
                union(edge.source_id, edge.target_id)

        # Group by component
        components: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            components.setdefault(root, []).append(i)

        diagrams: list[Diagram] = []
        for component_ids in components.values():
            if len(component_ids) < self._min_nodes:
                continue

            component_set = set(component_ids)
            component_nodes = [shapes[i] for i in component_ids]
            component_edges = [
                e for e in edges
                if e.source_id in component_set
                and e.target_id in component_set
            ]

            # Compute bounding box
            x0 = min(n.x0 for n in component_nodes)
            top = min(n.top for n in component_nodes)
            x1 = max(n.x1 for n in component_nodes)
            bottom = max(n.bottom for n in component_nodes)

            diagrams.append(Diagram(
                nodes=component_nodes,
                edges=component_edges,
                page_number=page_number,
                bbox=(x0, top, x1, bottom),
            ))

        return diagrams

    # ------------------------------------------------------------------
    # Step 5: Diagram Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_diagram(diagram: Diagram) -> str:
        """Infer diagram type from structure and labels.

        Heuristics:
        - CONSORT: labels contain "enrolled", "allocated", "follow-up",
          "analyzed", or "randomized"
        - Study design: labels contain "screening", "treatment",
          "follow-up", or "arm"
        - Flowchart: default for connected directed graphs
        """
        all_text = " ".join(
            n.label.lower() for n in diagram.nodes if n.label
        )

        consort_keywords = {
            "enrolled", "allocated", "follow-up", "analyzed",
            "randomized", "excluded", "discontinued", "assessed",
            "eligibility",
        }
        study_keywords = {
            "screening", "treatment period", "end of study",
            "arm a", "arm b", "cohort", "dose level",
            "run-in", "washout",
        }

        consort_hits = sum(1 for kw in consort_keywords if kw in all_text)
        study_hits = sum(1 for kw in study_keywords if kw in all_text)

        if consort_hits >= 3:
            return "consort"
        if study_hits >= 2:
            return "study_design"
        if diagram.edge_count > 0:
            return "flowchart"
        return "unknown"


__all__ = [
    "DiagramFinder",
    "DiagramNode",
    "DiagramEdge",
    "Diagram",
]
