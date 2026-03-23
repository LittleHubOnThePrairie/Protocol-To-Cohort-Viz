"""Visual element detection for protocol PDFs (PTCV-225).

Detects non-text visual elements — study design diagrams, CONSORT
flow charts, PK sampling schemas, dose-response curves — by
identifying embedded images and large drawing regions on each page.

Uses PyMuPDF (fitz) for image/drawing detection. Falls back
gracefully when fitz is not installed.

Risk tier: LOW — read-only analysis, no data mutation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Minimum image dimensions (pixels) to consider as a figure
_MIN_IMAGE_WIDTH = 100
_MIN_IMAGE_HEIGHT = 80

# Minimum area ratio (image area / page area) to qualify as a figure
_MIN_AREA_RATIO = 0.03

# Caption patterns found above or below figures
_CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Exhibit|Diagram|Scheme|Chart|Graph)"
    r"\s*\d*[.:]?\s",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box on a page.

    Coordinates are in PDF points (1/72 inch), origin at top-left.

    Attributes:
        x0: Left edge.
        y0: Top edge.
        x1: Right edge.
        y1: Bottom edge.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class DetectedFigure:
    """A detected visual element on a page.

    Attributes:
        page_number: 1-based page number.
        bbox: Bounding box in PDF points.
        figure_type_hint: Heuristic guess at figure type based on
            context (e.g., "study_design", "consort", "generic").
        detection_method: How the figure was detected
            ("image_object", "drawing_region", "caption_heuristic").
        caption: Caption text found near the figure (if any).
        image_index: Index of the image on the page (for image objects).
    """

    page_number: int
    bbox: BoundingBox
    figure_type_hint: str = "generic"
    detection_method: str = "image_object"
    caption: str = ""
    image_index: int = 0


@dataclass
class FigureDetectionResult:
    """Result of figure detection across an entire PDF.

    Attributes:
        figures: All detected figures.
        pages_with_figures: Set of page numbers containing figures.
        total_figures: Total count.
    """

    figures: list[DetectedFigure] = field(default_factory=list)

    @property
    def pages_with_figures(self) -> set[int]:
        return {f.page_number for f in self.figures}

    @property
    def total_figures(self) -> int:
        return len(self.figures)


def _classify_figure_type(caption: str) -> str:
    """Heuristic figure type classification from caption text.

    Args:
        caption: Caption text near the figure.

    Returns:
        Figure type hint string.
    """
    lower = caption.lower()

    if any(kw in lower for kw in (
        "study design", "study schema", "trial design",
        "treatment schema", "study overview",
    )):
        return "study_design"

    if any(kw in lower for kw in (
        "consort", "patient flow", "disposition",
        "participant flow", "subject flow",
    )):
        return "consort"

    if any(kw in lower for kw in (
        "kaplan", "survival", "km curve",
    )):
        return "kaplan_meier"

    if any(kw in lower for kw in (
        "pharmacokinetic", "pk profile", "concentration",
        "pk sampling",
    )):
        return "pk_profile"

    if any(kw in lower for kw in (
        "forest plot", "subgroup",
    )):
        return "forest_plot"

    if any(kw in lower for kw in (
        "dose", "response", "dose-response",
    )):
        return "dose_response"

    return "generic"


def _find_nearby_caption(
    page_text: str,
    bbox: BoundingBox,
    page_height: float,
) -> str:
    """Search for figure caption text near a bounding box.

    Looks for "Figure N:" patterns in the page text. Since we
    don't have per-character positions from fitz in this path,
    we scan all caption-like lines on the page.

    Args:
        page_text: Full text of the page.
        bbox: Bounding box of the detected figure.
        page_height: Page height in PDF points.

    Returns:
        Caption text if found, empty string otherwise.
    """
    for line in page_text.splitlines():
        stripped = line.strip()
        if _CAPTION_RE.match(stripped):
            return stripped[:200]  # Truncate long captions
    return ""


def detect_figures(
    pdf_bytes: bytes,
    table_pages: Optional[set[int]] = None,
) -> FigureDetectionResult:
    """Detect visual elements in a PDF.

    Scans each page for:
    1. Embedded image objects (raster images, vector art rasterized)
    2. Large drawing regions (vector graphics clusters)

    Pages that are entirely tables (from PTCV-223) can be excluded
    via ``table_pages`` to avoid false positives.

    Args:
        pdf_bytes: Raw PDF file bytes.
        table_pages: Set of page numbers known to contain only
            tables (excluded from figure detection).

    Returns:
        FigureDetectionResult with all detected figures.
    """
    try:
        import fitz
    except ImportError:
        logger.info(
            "PyMuPDF (fitz) not installed; skipping figure detection"
        )
        return FigureDetectionResult()

    if table_pages is None:
        table_pages = set()

    result = FigureDetectionResult()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.debug("fitz failed to open PDF: %s", exc)
        return result

    try:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1

            if page_num in table_pages:
                continue

            page = doc[page_idx]
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height

            if page_area <= 0:
                continue

            page_text = page.get_text("text") or ""
            img_index = 0

            # Method 1: Detect embedded image objects
            try:
                image_list = page.get_images(full=True)
            except Exception:
                image_list = []

            for img_info in image_list:
                try:
                    xref = img_info[0]
                    img_rects = page.get_image_rects(xref)

                    for rect in img_rects:
                        w = rect.width
                        h = rect.height
                        area = w * h

                        if (
                            w < _MIN_IMAGE_WIDTH
                            or h < _MIN_IMAGE_HEIGHT
                        ):
                            continue

                        if area / page_area < _MIN_AREA_RATIO:
                            continue

                        bbox = BoundingBox(
                            x0=rect.x0, y0=rect.y0,
                            x1=rect.x1, y1=rect.y1,
                        )

                        caption = _find_nearby_caption(
                            page_text, bbox, page_rect.height,
                        )
                        fig_type = (
                            _classify_figure_type(caption)
                            if caption
                            else "generic"
                        )

                        result.figures.append(DetectedFigure(
                            page_number=page_num,
                            bbox=bbox,
                            figure_type_hint=fig_type,
                            detection_method="image_object",
                            caption=caption,
                            image_index=img_index,
                        ))
                        img_index += 1

                except Exception as exc:
                    logger.debug(
                        "Image rect extraction failed on page %d: %s",
                        page_num, exc,
                    )

            # Method 2: Detect pages with figure captions but no
            # image objects (vector-only diagrams)
            if img_index == 0 and _CAPTION_RE.search(page_text):
                caption = _find_nearby_caption(
                    page_text,
                    BoundingBox(0, 0, page_rect.width, page_rect.height),
                    page_rect.height,
                )
                if caption:
                    fig_type = _classify_figure_type(caption)
                    result.figures.append(DetectedFigure(
                        page_number=page_num,
                        bbox=BoundingBox(
                            x0=0, y0=0,
                            x1=page_rect.width,
                            y1=page_rect.height,
                        ),
                        figure_type_hint=fig_type,
                        detection_method="caption_heuristic",
                        caption=caption,
                    ))

    finally:
        doc.close()

    logger.info(
        "Figure detection: %d figures on %d pages",
        result.total_figures,
        len(result.pages_with_figures),
    )

    return result


def crop_figure_image(
    pdf_bytes: bytes,
    figure: DetectedFigure,
    scale: float = 2.0,
) -> Optional[bytes]:
    """Crop a detected figure region as a PNG image.

    Renders the page at the given scale and crops to the figure's
    bounding box. Used by Phase 4 (PTCV-226) for Vision API captioning.

    Args:
        pdf_bytes: Raw PDF bytes.
        figure: Detected figure with page number and bounding box.
        scale: Rendering scale factor (2.0 = 144 DPI).

    Returns:
        PNG bytes of the cropped figure, or None on failure.
    """
    try:
        import fitz
    except ImportError:
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[figure.page_number - 1]

        # Define clip rectangle from bounding box
        clip = fitz.Rect(
            figure.bbox.x0, figure.bbox.y0,
            figure.bbox.x1, figure.bbox.y1,
        )

        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes

    except Exception as exc:
        logger.debug("Figure crop failed: %s", exc)
        return None


__all__ = [
    "BoundingBox",
    "DetectedFigure",
    "FigureDetectionResult",
    "crop_figure_image",
    "detect_figures",
]
