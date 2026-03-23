"""Content format auto-detection for adaptive extraction (PTCV-257).

Inspects routed protocol text to determine its structural format
(table, list, prose, etc.) so the extraction engine can select the
optimal strategy instead of relying solely on the YAML-hardcoded type.

The detected format is a *suggestion*; the YAML ``expected_type``
remains the fallback when detection is ambiguous.

Risk tier: LOW — read-only text analysis.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Pipe-delimited table: at least two | characters on a line
_PIPE_LINE_RE = re.compile(r"^.*\|.*\|.*$", re.MULTILINE)

# Aligned columns: 3+ segments separated by 3+ spaces
_ALIGNED_RE = re.compile(
    r"^.{10,}\s{3,}.{10,}\s{3,}.{10,}$", re.MULTILINE,
)

# List markers: bullets, numbered items, lettered items
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:[-•●○▪◦]\s+|\d+[\.\)]\s+|[a-z][\.\)]\s+)",
    re.MULTILINE,
)

# Date patterns — broad detection
_DATE_RE = re.compile(
    r"\d{1,2}\s+(?:January|February|March|April|May|June"
    r"|July|August|September|October|November|December)\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}",
    re.IGNORECASE,
)

# Numeric value with optional units
_NUMERIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|mg|kg|mL|units?|participants?|subjects?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class FormatDetection(NamedTuple):
    """Result of content format detection.

    Attributes:
        detected_type: Best-guess format (e.g. ``"table"``, ``"list"``).
        confidence: Detection confidence (0.0–1.0).
        signals: Dict of signal names to their raw counts/scores.
    """

    detected_type: str
    confidence: float
    signals: dict[str, int | float]


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

# Types that should NOT be overridden by format detection because they
# use highly specific regex patterns that format detection can't improve.
_SKIP_DETECTION_TYPES = frozenset({
    "identifier",
    "date",
    "numeric",
    "enum",
})


def should_detect(yaml_type: str) -> bool:
    """Return True if auto-detection should run for this YAML type.

    Some types (identifier, date, numeric, enum) use highly specific
    regex patterns that format detection cannot improve upon.
    """
    return yaml_type not in _SKIP_DETECTION_TYPES


def detect_content_format(text: str) -> FormatDetection:
    """Inspect text content to determine its structural format.

    Detection priority (highest signal first):
    1. **table** — pipe-delimited or aligned columns
    2. **list** — bullet/numbered items
    3. **text_short** — very short content (< 200 chars)
    4. **text_long** — default for prose

    Args:
        text: The routed section content to analyze.

    Returns:
        FormatDetection with detected type, confidence, and signals.
    """
    if not text or not text.strip():
        return FormatDetection("text_long", 0.0, {})

    stripped = text.strip()
    lines = stripped.splitlines()
    n_lines = len(lines)

    # Count structural signals
    pipe_lines = len(_PIPE_LINE_RE.findall(stripped))
    aligned_lines = len(_ALIGNED_RE.findall(stripped))
    list_items = len(_LIST_MARKER_RE.findall(stripped))

    signals: dict[str, int | float] = {
        "pipe_lines": pipe_lines,
        "aligned_lines": aligned_lines,
        "list_items": list_items,
        "char_count": len(stripped),
        "line_count": n_lines,
    }

    # --- Table detection (strongest signal) ---
    # Pipe tables: need ≥ 2 pipe lines for a real table
    if pipe_lines >= 2:
        confidence = min(0.95, 0.70 + pipe_lines * 0.05)
        return FormatDetection("table", confidence, signals)

    # Aligned columns: need ≥ 3 aligned lines
    if aligned_lines >= 3:
        confidence = min(0.85, 0.55 + aligned_lines * 0.05)
        return FormatDetection("table", confidence, signals)

    # --- List detection ---
    # Need ≥ 3 list items AND they represent a significant fraction
    # of total lines to avoid false positives in prose with occasional
    # numbered references.
    if list_items >= 3 and n_lines > 0:
        list_fraction = list_items / n_lines
        if list_fraction >= 0.3:
            confidence = min(0.90, 0.60 + list_items * 0.03)
            return FormatDetection("list", confidence, signals)

    # --- Short text detection ---
    if len(stripped) < 200:
        return FormatDetection("text_short", 0.70, signals)

    # --- Default: prose ---
    return FormatDetection("text_long", 0.50, signals)


def select_strategy(
    yaml_type: str,
    detected: FormatDetection,
    confidence_threshold: float = 0.65,
) -> str:
    """Choose the best extraction strategy.

    Uses the detected format when confidence is above threshold,
    otherwise falls back to the YAML hint.

    Args:
        yaml_type: The ``expected_type`` from the YAML schema.
        detected: Result from ``detect_content_format()``.
        confidence_threshold: Minimum detection confidence to
            override the YAML hint (default 0.65).

    Returns:
        The extraction type string to use (e.g. ``"table"``).
    """
    # Don't override types that use specialized regex patterns
    if not should_detect(yaml_type):
        return yaml_type

    # If detection agrees with YAML, always use it (reinforced)
    if detected.detected_type == yaml_type:
        return yaml_type

    # Override only when detection is confident enough
    if detected.confidence >= confidence_threshold:
        return detected.detected_type

    # Low-confidence detection → fall back to YAML hint
    return yaml_type
