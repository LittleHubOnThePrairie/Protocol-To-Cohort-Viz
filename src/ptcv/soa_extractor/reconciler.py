"""Multi-method SoA table reconciler (PTCV-216).

When multiple extraction methods produce partial SoA tables,
the reconciler merges them by:
1. Taking the union of assessment rows
2. Using majority voting for X/blank values per cell on overlaps
3. Flagging cells with method disagreement

Risk tier: LOW — data merging logic, no external calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .models import RawSoaTable

logger = logging.getLogger(__name__)


@dataclass
class CellConfidence:
    """Confidence metadata for a single cell.

    Attributes:
        value: Resolved boolean (scheduled or not).
        methods_agree: Number of methods that produced this value.
        methods_disagree: Number that produced the opposite.
        confidence: Agreement ratio (0.0–1.0).
        needs_review: True if methods disagreed.
    """

    value: bool
    methods_agree: int = 1
    methods_disagree: int = 0
    confidence: float = 1.0
    needs_review: bool = False


@dataclass
class ReconciliationReport:
    """Result of reconciling multiple extraction methods.

    Attributes:
        merged_table: The reconciled RawSoaTable.
        source_count: Number of input tables merged.
        total_activities: Activity count in merged result.
        overlapping_activities: Activities present in 2+ sources.
        unique_to_one: Activities present in only one source.
        disagreement_count: Cells where methods disagreed.
        cell_confidences: Per-activity, per-visit confidence map.
    """

    merged_table: RawSoaTable = field(
        default_factory=lambda: RawSoaTable(
            visit_headers=[], day_headers=[], activities=[],
            section_code="B.4",
        )
    )
    source_count: int = 0
    total_activities: int = 0
    overlapping_activities: int = 0
    unique_to_one: int = 0
    disagreement_count: int = 0
    cell_confidences: dict[str, list[CellConfidence]] = field(
        default_factory=dict,
    )


def _normalize_activity(name: str) -> str:
    """Normalize activity name for matching across methods."""
    import re
    n = name.strip().lower()
    n = re.sub(r"[\^¹²³⁴⁵⁶⁷⁸⁹⁰]+\d*", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def reconcile(
    tables: list[RawSoaTable],
    min_confidence: float = 0.5,
) -> ReconciliationReport:
    """Reconcile multiple SoA tables from different extraction methods.

    Merges by:
    1. Selecting the visit headers from the table with the most columns
    2. Union of all activity rows (matched by normalized name)
    3. Majority voting for X/blank per cell on overlapping rows
    4. Flagging cells where methods disagreed

    Args:
        tables: List of RawSoaTable objects from different methods.
        min_confidence: Below this, cells are flagged for review.

    Returns:
        ReconciliationReport with merged table and confidence data.
    """
    if not tables:
        return ReconciliationReport()

    if len(tables) == 1:
        return ReconciliationReport(
            merged_table=tables[0],
            source_count=1,
            total_activities=len(tables[0].activities),
            unique_to_one=len(tables[0].activities),
        )

    # Pick headers from the table with most visit columns
    best_header_table = max(tables, key=lambda t: len(t.visit_headers))
    visit_headers = best_header_table.visit_headers
    day_headers = best_header_table.day_headers
    n_visits = len(visit_headers)

    # Collect all activities by normalized name
    # For each, store list of (scheduled_flags, source_index) tuples
    activity_votes: dict[str, list[tuple[list[bool], int]]] = {}
    # Keep the best display name per normalized key
    display_names: dict[str, str] = {}

    for src_idx, table in enumerate(tables):
        for name, flags in table.activities:
            norm = _normalize_activity(name)
            if not norm:
                continue

            # Pad or truncate flags to match target header count
            padded = list(flags[:n_visits])
            while len(padded) < n_visits:
                padded.append(False)

            activity_votes.setdefault(norm, []).append(
                (padded, src_idx)
            )

            # Prefer longer display name
            if (
                norm not in display_names
                or len(name) > len(display_names[norm])
            ):
                display_names[norm] = name

    # Resolve each activity by majority voting
    merged_activities: list[tuple[str, list[bool]]] = []
    cell_confidences: dict[str, list[CellConfidence]] = {}
    overlapping = 0
    unique = 0
    disagreements = 0

    for norm, votes in sorted(activity_votes.items()):
        display_name = display_names[norm]
        n_sources = len(votes)

        if n_sources > 1:
            overlapping += 1
        else:
            unique += 1

        resolved_flags: list[bool] = []
        cell_confs: list[CellConfidence] = []

        for col_idx in range(n_visits):
            true_count = sum(1 for flags, _ in votes if flags[col_idx])
            false_count = n_sources - true_count

            # Majority vote
            value = true_count > false_count
            agree = max(true_count, false_count)
            disagree = min(true_count, false_count)
            conf = agree / n_sources if n_sources > 0 else 1.0

            if disagree > 0:
                disagreements += 1

            resolved_flags.append(value)
            cell_confs.append(CellConfidence(
                value=value,
                methods_agree=agree,
                methods_disagree=disagree,
                confidence=conf,
                needs_review=conf < min_confidence,
            ))

        merged_activities.append((display_name, resolved_flags))
        cell_confidences[norm] = cell_confs

    merged_table = RawSoaTable(
        visit_headers=visit_headers,
        day_headers=day_headers,
        activities=merged_activities,
        section_code=best_header_table.section_code,
    )

    return ReconciliationReport(
        merged_table=merged_table,
        source_count=len(tables),
        total_activities=len(merged_activities),
        overlapping_activities=overlapping,
        unique_to_one=unique,
        disagreement_count=disagreements,
        cell_confidences=cell_confidences,
    )


__all__ = [
    "CellConfidence",
    "ReconciliationReport",
    "reconcile",
]
