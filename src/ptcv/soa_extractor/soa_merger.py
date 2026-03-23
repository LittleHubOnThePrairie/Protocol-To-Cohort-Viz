"""SoA multi-stream merger — deduplication and union (PTCV-260).

Merges assessment-visit pairs from table (Stream A), narrative
(Stream B), and diagram (Stream C) into a unified set. Handles
fuzzy name matching, deduplication, and multi-stream confidence
boosting.

Risk tier: LOW — read-only merge logic, no API calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .narrative_extractor import AssessmentVisitPair

logger = logging.getLogger(__name__)

# Confidence boost when 2+ streams agree on an assessment-visit pair
_MULTI_STREAM_BOOST = 0.10

# Maximum confidence after boost
_MAX_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MergedAssessment:
    """A merged assessment-visit pair with provenance.

    Attributes:
        assessment_name: Canonical assessment name.
        visit_label: Visit name or timing.
        confidence: Merged confidence (boosted for multi-stream).
        sources: Set of source streams that found this pair.
        source_tag: Joined source string for display.
    """

    assessment_name: str
    visit_label: str
    confidence: float = 0.70
    sources: set[str] = field(default_factory=set)

    @property
    def source_tag(self) -> str:
        """Joined source string (e.g., 'table+narrative')."""
        return "+".join(sorted(self.sources))


@dataclass
class MergeResult:
    """Result of merging assessment-visit pairs from all streams.

    Attributes:
        assessments: All merged assessment-visit pairs.
        total_input: Total input pairs before merge.
        total_merged: Output count after deduplication.
        multi_stream_count: Pairs confirmed by 2+ streams.
    """

    assessments: list[MergedAssessment] = field(default_factory=list)
    total_input: int = 0
    total_merged: int = 0
    multi_stream_count: int = 0


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Normalise assessment name for fuzzy matching."""
    n = name.strip().lower()
    # Common abbreviation expansions
    n = re.sub(r"\bpe\b", "physical exam", n)
    n = re.sub(r"\becg\b", "electrocardiogram", n)
    n = re.sub(r"\bcbc\b", "complete blood count", n)
    n = re.sub(r"\bpk\b", "pharmacokinetics", n)
    n = re.sub(r"\bae\b", "adverse events", n)
    # Remove extra whitespace/punctuation
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _normalize_visit(label: str) -> str:
    """Normalise visit label for matching."""
    v = label.strip().lower()
    v = re.sub(r"\s+", " ", v)
    return v


def _names_match(a: str, b: str) -> bool:
    """Check if two assessment names match (fuzzy)."""
    na = _normalize_name(a)
    nb = _normalize_name(b)

    if na == nb:
        return True

    # Substring containment (handles "Physical Exam" vs "Physical Examination")
    if na in nb or nb in na:
        return True

    return False


def _visits_match(a: str, b: str) -> bool:
    """Check if two visit labels match."""
    va = _normalize_visit(a)
    vb = _normalize_visit(b)

    if va == vb:
        return True

    # "screening" vs "screening visit"
    if va in vb or vb in va:
        return True

    return False


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_streams(
    *streams: list[AssessmentVisitPair],
) -> MergeResult:
    """Merge assessment-visit pairs from multiple streams.

    Deduplicates by fuzzy matching on (assessment_name, visit_label).
    Pairs found in 2+ streams get a confidence boost. Union semantics:
    assessments from ANY stream are included.

    Args:
        *streams: Variable number of pair lists from different streams.

    Returns:
        MergeResult with deduplicated, confidence-boosted pairs.
    """
    all_pairs: list[AssessmentVisitPair] = []
    for stream in streams:
        all_pairs.extend(stream)

    total_input = len(all_pairs)

    # Build merged list with dedup
    merged: list[MergedAssessment] = []

    for pair in all_pairs:
        # Check if this pair matches any existing merged entry
        found = False
        for existing in merged:
            if (
                _names_match(pair.assessment_name, existing.assessment_name)
                and _visits_match(pair.visit_label, existing.visit_label)
            ):
                # Merge: add source, boost confidence
                existing.sources.add(pair.source)
                existing.confidence = min(
                    max(existing.confidence, pair.confidence)
                    + _MULTI_STREAM_BOOST,
                    _MAX_CONFIDENCE,
                )
                found = True
                break

        if not found:
            merged.append(MergedAssessment(
                assessment_name=pair.assessment_name,
                visit_label=pair.visit_label,
                confidence=pair.confidence,
                sources={pair.source},
            ))

    multi_stream = sum(
        1 for m in merged if len(m.sources) > 1
    )

    if merged:
        logger.info(
            "PTCV-260: SoA merge: %d input → %d merged "
            "(%d multi-stream confirmed)",
            total_input, len(merged), multi_stream,
        )

    return MergeResult(
        assessments=merged,
        total_input=total_input,
        total_merged=len(merged),
        multi_stream_count=multi_stream,
    )


def pairs_to_raw_table(
    pairs: list[MergedAssessment],
) -> Optional["RawSoaTable"]:
    """Convert merged pairs into a RawSoaTable for USDM mapping.

    Builds the visit_headers from all unique visits and the
    activities list from all unique assessments.

    Args:
        pairs: Merged assessment-visit pairs.

    Returns:
        RawSoaTable or None if no pairs.
    """
    from .models import RawSoaTable

    if not pairs:
        return None

    # Collect unique visits (preserving order of first appearance)
    visit_order: list[str] = []
    visit_set: set[str] = set()
    for p in pairs:
        vn = _normalize_visit(p.visit_label)
        if vn not in visit_set:
            visit_set.add(vn)
            visit_order.append(p.visit_label)

    # Collect unique assessments
    assessment_order: list[str] = []
    assessment_set: set[str] = set()
    for p in pairs:
        an = _normalize_name(p.assessment_name)
        if an not in assessment_set:
            assessment_set.add(an)
            assessment_order.append(p.assessment_name)

    # Build activity matrix
    activities: list[tuple[str, list[bool]]] = []
    for assessment in assessment_order:
        scheduled = [False] * len(visit_order)
        for p in pairs:
            if _names_match(p.assessment_name, assessment):
                for i, visit in enumerate(visit_order):
                    if _visits_match(p.visit_label, visit):
                        scheduled[i] = True
        activities.append((assessment, scheduled))

    return RawSoaTable(
        visit_headers=visit_order,
        day_headers=[],
        activities=activities,
        section_code="merged",
        construction_method="multi_source",
    )


__all__ = [
    "MergeResult",
    "MergedAssessment",
    "merge_streams",
    "pairs_to_raw_table",
]
