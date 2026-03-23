"""Diagram SoA extractor — Stream C (PTCV-260).

Extracts assessment-visit pairs from diagram node labels produced
by DiagramFinder (PTCV-243). Study design diagrams often show
which assessments happen at which visits via labeled boxes and
arrows.

Risk tier: LOW — pattern matching on diagram metadata, no API calls.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .narrative_extractor import AssessmentVisitPair

logger = logging.getLogger(__name__)

# Patterns for extracting visit references from diagram labels
_VISIT_IN_LABEL = re.compile(
    r"(?:Screening|Baseline|Day\s*\d+|Week\s*\d+|Month\s*\d+"
    r"|Cycle\s*\d+|Visit\s*\d+|End\s+of\s+(?:Treatment|Study)"
    r"|Early\s+Termination|Follow[\s-]*up)",
    re.IGNORECASE,
)

# Common assessment keywords
_ASSESSMENT_KEYWORDS = {
    "vital signs", "vitals", "ecg", "electrocardiogram",
    "physical exam", "physical examination", "pe",
    "laboratory", "labs", "lab tests", "hematology",
    "chemistry", "cbc", "urinalysis", "coagulation",
    "tumor assessment", "ct scan", "mri", "imaging",
    "pharmacokinetics", "pk", "biomarker",
    "adverse events", "ae", "concomitant medications",
    "informed consent", "pregnancy test",
    "ecog", "performance status",
}


def _is_assessment(label: str) -> bool:
    """Check if a diagram node label refers to an assessment."""
    lower = label.strip().lower()
    return any(kw in lower for kw in _ASSESSMENT_KEYWORDS)


def _extract_visits(label: str) -> list[str]:
    """Extract visit labels from a diagram node or edge label."""
    return [m.group(0) for m in _VISIT_IN_LABEL.finditer(label)]


def extract_from_diagrams(
    diagrams: list[dict[str, Any]],
    source_label: str = "diagram",
) -> list[AssessmentVisitPair]:
    """Extract assessment-visit pairs from diagram data.

    Examines diagram nodes and edges for assessment-visit
    relationships. A node containing an assessment keyword paired
    with a visit reference (in the same node or a connected edge)
    produces a pair.

    Args:
        diagrams: List of diagram dicts from DiagramFinder.to_dict().
            Each has ``nodes``, ``edges``, ``diagram_type``,
            ``page_number``.
        source_label: Source tag for provenance.

    Returns:
        List of AssessmentVisitPair objects.
    """
    pairs: list[AssessmentVisitPair] = []
    seen: set[tuple[str, str]] = set()

    for diagram in diagrams:
        nodes = diagram.get("nodes", [])
        edges = diagram.get("edges", [])

        # Build node label lookup
        node_labels: dict[int, str] = {}
        for node in nodes:
            node_id = node.get("node_id", 0)
            label = node.get("label", "").strip()
            if label:
                node_labels[node_id] = label

        # Build adjacency: node_id → connected labels (via edges)
        connected_labels: dict[int, list[str]] = {}
        for edge in edges:
            src = edge.get("source_id")
            tgt = edge.get("target_id")
            edge_label = edge.get("label", "").strip()

            for nid, other_nid in [(src, tgt), (tgt, src)]:
                if nid not in connected_labels:
                    connected_labels[nid] = []
                if edge_label:
                    connected_labels[nid].append(edge_label)
                other_label = node_labels.get(other_nid, "")
                if other_label:
                    connected_labels[nid].append(other_label)

        # Extract pairs from nodes
        for node in nodes:
            label = node.get("label", "").strip()
            if not label:
                continue
            node_id = node.get("node_id", 0)

            # Case 1: Node label contains both assessment and visit
            if _is_assessment(label):
                visits = _extract_visits(label)
                if visits:
                    for visit in visits:
                        _add_pair(
                            pairs, seen, label, visit,
                            source_label,
                        )
                    continue

                # Case 2: Assessment node connected to visit node
                for connected in connected_labels.get(node_id, []):
                    visits = _extract_visits(connected)
                    for visit in visits:
                        _add_pair(
                            pairs, seen, label, visit,
                            source_label,
                        )

            # Case 3: Visit node connected to assessment text
            elif _extract_visits(label):
                visits = _extract_visits(label)
                for connected in connected_labels.get(node_id, []):
                    if _is_assessment(connected):
                        for visit in visits:
                            _add_pair(
                                pairs, seen, connected, visit,
                                source_label,
                            )

    if pairs:
        logger.info(
            "PTCV-260: Diagram extraction: %d assessment-visit "
            "pairs from %d diagrams",
            len(pairs), len(diagrams),
        )

    return pairs


def _add_pair(
    pairs: list[AssessmentVisitPair],
    seen: set[tuple[str, str]],
    assessment_label: str,
    visit_label: str,
    source: str,
) -> None:
    """Add a pair if not already seen (case-insensitive dedup)."""
    # Clean assessment: extract just the keyword part
    name = assessment_label.strip()
    key = (name.lower(), visit_label.lower())
    if key in seen:
        return
    seen.add(key)
    pairs.append(AssessmentVisitPair(
        assessment_name=name,
        visit_label=visit_label,
        source=source,
        confidence=0.60,
    ))


__all__ = [
    "extract_from_diagrams",
]
