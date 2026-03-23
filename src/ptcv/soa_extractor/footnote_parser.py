"""Footnote parser for SoA assessment names (PTCV-218).

Separates superscript footnote markers from assessment names and
cross-references them against footnote text found below the table.
Prevents footnote markers from corrupting column alignment.

Risk tier: LOW — text parsing only, no external calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Superscript/caret footnote markers: "Hematology^2", "ECG¹²",
# "Labs (see note 3)", "Vitals*"
_FOOTNOTE_MARKER_RE = re.compile(
    r"[\^](\d+)"              # Caret style: ^2, ^13
    r"|([¹²³⁴⁵⁶⁷⁸⁹⁰]+)"    # Unicode superscript: ¹²
    r"|[*†‡§]"                # Symbol markers: *, †, ‡
    r"|\(see\s+note\s+(\d+)\)"  # Parenthetical: (see note 3)
)

# Footnote text line: starts with a number, letter, *, or symbol
_FOOTNOTE_TEXT_RE = re.compile(
    r"^\s*(?:(\d+)[.):]\s*"   # "1. ...", "2) ...", "3: ..."
    r"|([*†‡§])\s*"           # "* ..."
    r"|([a-z])[.)]\s*"        # "a. ...", "b) ..."
    r")(.+)",
    re.IGNORECASE,
)

# Unicode superscript digit mapping
_SUPERSCRIPT_MAP: dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}


@dataclass
class FootnoteMarker:
    """A footnote marker found in an assessment name.

    Attributes:
        marker_id: The marker identifier (e.g., "2", "a", "*").
        marker_text: The raw marker text as it appeared.
        position: Character position in the original name.
    """

    marker_id: str
    marker_text: str
    position: int = 0


@dataclass
class ParsedAssessment:
    """An assessment name with footnote markers separated.

    Attributes:
        clean_name: Assessment name with markers removed.
        original_name: Original name before parsing.
        markers: List of footnote markers found.
    """

    clean_name: str
    original_name: str
    markers: list[FootnoteMarker] = field(default_factory=list)

    @property
    def has_footnotes(self) -> bool:
        """Whether any footnote markers were found."""
        return bool(self.markers)


@dataclass
class Footnote:
    """A resolved footnote with its text.

    Attributes:
        footnote_id: The footnote identifier (e.g., "1", "a", "*").
        text: The footnote content text.
    """

    footnote_id: str
    text: str


@dataclass
class FootnoteReport:
    """Result of footnote parsing and cross-referencing.

    Attributes:
        parsed_assessments: Assessments with markers separated.
        footnotes: Resolved footnote texts.
        unresolved_markers: Marker IDs with no matching footnote text.
    """

    parsed_assessments: list[ParsedAssessment] = field(
        default_factory=list
    )
    footnotes: list[Footnote] = field(default_factory=list)
    unresolved_markers: list[str] = field(default_factory=list)


def _decode_superscript(text: str) -> str:
    """Convert unicode superscript digits to normal digits."""
    return "".join(_SUPERSCRIPT_MAP.get(c, c) for c in text)


def parse_assessment_name(name: str) -> ParsedAssessment:
    """Separate footnote markers from an assessment name.

    Args:
        name: Raw assessment name (e.g., "Hematology^2").

    Returns:
        ParsedAssessment with clean name and extracted markers.
    """
    markers: list[FootnoteMarker] = []
    clean = name

    for match in _FOOTNOTE_MARKER_RE.finditer(name):
        caret_id = match.group(1)       # ^2
        super_id = match.group(2)       # ¹²
        note_id = match.group(3)        # (see note 3)

        if caret_id:
            marker_id = caret_id
            marker_text = f"^{caret_id}"
        elif super_id:
            marker_id = _decode_superscript(super_id)
            marker_text = super_id
        elif note_id:
            marker_id = note_id
            marker_text = match.group(0)
        else:
            # Symbol marker (*, †, etc.)
            marker_id = match.group(0)
            marker_text = match.group(0)

        markers.append(FootnoteMarker(
            marker_id=marker_id,
            marker_text=marker_text,
            position=match.start(),
        ))

    # Remove all markers from the name
    if markers:
        clean = _FOOTNOTE_MARKER_RE.sub("", name).strip()
        # Clean up trailing/leading whitespace and punctuation artifacts
        clean = re.sub(r"\s+", " ", clean).strip()
        clean = clean.rstrip(",;:")

    return ParsedAssessment(
        clean_name=clean,
        original_name=name,
        markers=markers,
    )


def extract_footnotes(text: str) -> list[Footnote]:
    """Extract footnote definitions from text below a table.

    Parses lines that start with a number, symbol, or letter
    followed by punctuation and text content.

    Args:
        text: Text block below the SoA table (may contain
            multiple footnote lines).

    Returns:
        List of Footnote objects with IDs and text.
    """
    footnotes: list[Footnote] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = _FOOTNOTE_TEXT_RE.match(line)
        if not match:
            continue

        num_id = match.group(1)     # "1", "2", etc.
        sym_id = match.group(2)     # "*", "†", etc.
        letter_id = match.group(3)  # "a", "b", etc.
        content = match.group(4).strip()

        footnote_id = num_id or sym_id or letter_id or ""
        if footnote_id and content:
            footnotes.append(Footnote(
                footnote_id=footnote_id,
                text=content,
            ))

    return footnotes


def parse_and_crossref(
    activity_names: list[str],
    footnote_text: str = "",
) -> FootnoteReport:
    """Parse assessment names and cross-reference with footnote text.

    Args:
        activity_names: List of raw assessment names from the table.
        footnote_text: Text below the table containing footnote
            definitions.

    Returns:
        FootnoteReport with parsed assessments, resolved footnotes,
        and any unresolved marker IDs.
    """
    parsed = [parse_assessment_name(name) for name in activity_names]
    footnotes = extract_footnotes(footnote_text) if footnote_text else []

    # Collect all marker IDs
    all_marker_ids: set[str] = set()
    for p in parsed:
        for m in p.markers:
            all_marker_ids.add(m.marker_id)

    # Check which markers have matching footnote text
    footnote_ids = {f.footnote_id for f in footnotes}
    unresolved = sorted(all_marker_ids - footnote_ids)

    return FootnoteReport(
        parsed_assessments=parsed,
        footnotes=footnotes,
        unresolved_markers=unresolved,
    )


__all__ = [
    "FootnoteMarker",
    "ParsedAssessment",
    "Footnote",
    "FootnoteReport",
    "parse_assessment_name",
    "extract_footnotes",
    "parse_and_crossref",
]
