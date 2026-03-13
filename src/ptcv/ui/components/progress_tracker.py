"""Unified 8-stage pipeline progress tracker (PTCV-179).

Pure-Python module — no Streamlit dependency, fully testable.
Maps both query pipeline and classified pipeline stages into a
unified 8-stage model for progress display.
"""

from __future__ import annotations

import dataclasses
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StageStatus:
    """Status of one pipeline stage in the unified tracker.

    Attributes:
        key: Machine-readable stage identifier.
        name: Human-readable display name.
        status: One of ``"pending"``, ``"complete"``, ``"error"``,
            ``"skipped"``.
        elapsed_seconds: Time spent in this stage (0.0 if not run).
        source_pipeline: Which pipeline ran this stage
            (``"query"``, ``"classified"``, or ``""``).
        metrics: Stage-specific metrics dict (e.g.
            ``{"local_count": 8, "sonnet_count": 2}``).
    """

    key: str
    name: str
    status: str
    elapsed_seconds: float = 0.0
    source_pipeline: str = ""
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# 8-stage constant
# ---------------------------------------------------------------------------

UNIFIED_PIPELINE_STAGES: tuple[tuple[str, str], ...] = (
    ("pdf_extraction", "PDF Content Extraction"),
    ("header_footer", "Header/Footer Removal"),
    ("ml_classification", "ML Classification (NeoBERT/RAG)"),
    ("sonnet_classification", "Sonnet-Enhanced Classification"),
    ("content_transforms", "Content Transformations"),
    ("query_route", "Query Route Building"),
    ("document_assembly", "ICH E6(R3) Document Assembly"),
    ("soa_construction", "SoA Table Construction"),
)
"""Ordered 8-stage unified pipeline model."""


# ---------------------------------------------------------------------------
# Query pipeline mapping
# ---------------------------------------------------------------------------

# Query pipeline stage name → unified stage key(s).
_QUERY_TO_UNIFIED: dict[str, list[str]] = {
    "Document Assembly": ["pdf_extraction", "query_route"],
    "Section Classification": ["content_transforms"],
    "Query Extraction": ["query_route"],
    "Result Aggregation": ["document_assembly"],
}

# Unified stages always skipped by the query pipeline.
_QUERY_SKIPPED = frozenset({
    "header_footer",
    "ml_classification",
    "sonnet_classification",
    "soa_construction",
})


def map_query_stages_to_unified(
    stage_timings: dict[str, float],
) -> list[StageStatus]:
    """Map query pipeline 4-stage timings to the 8-stage model.

    Stages not run by the query pipeline are marked ``"skipped"``.
    Stages with timing data are marked ``"complete"``.

    Args:
        stage_timings: Dict from ``run_query_pipeline()``
            result ``stage_timings``.

    Returns:
        List of 8 ``StageStatus`` objects in canonical order.
    """
    # Collect which unified keys have timing data.
    unified_timings: dict[str, float] = {}
    for query_stage, elapsed in stage_timings.items():
        for unified_key in _QUERY_TO_UNIFIED.get(query_stage, []):
            # Use max elapsed when multiple query stages map to same key.
            unified_timings[unified_key] = max(
                unified_timings.get(unified_key, 0.0), elapsed,
            )

    stages: list[StageStatus] = []
    for key, name in UNIFIED_PIPELINE_STAGES:
        if key in _QUERY_SKIPPED:
            stages.append(StageStatus(
                key=key, name=name, status="skipped",
            ))
        elif key in unified_timings:
            stages.append(StageStatus(
                key=key, name=name, status="complete",
                elapsed_seconds=unified_timings[key],
                source_pipeline="query",
            ))
        elif stage_timings:
            # Query pipeline ran but this stage had no timing.
            stages.append(StageStatus(
                key=key, name=name, status="complete",
                source_pipeline="query",
            ))
        else:
            stages.append(StageStatus(
                key=key, name=name, status="pending",
            ))

    return stages


# ---------------------------------------------------------------------------
# Classified pipeline mapping
# ---------------------------------------------------------------------------

def map_classified_stages_to_unified(
    cascade_stats: dict[str, Any] | None = None,
    stage_timings: dict[str, float] | None = None,
) -> list[StageStatus]:
    """Map classified pipeline metrics to the 8-stage model.

    Uses ``cascade_stats`` (from ``RoutingStats``) to populate
    ML Classification and Sonnet Classification metrics.

    Args:
        cascade_stats: Serialised ``RoutingStats`` dict from
            ``CascadeResult``.
        stage_timings: Optional timing dict for classified stages.

    Returns:
        List of 8 ``StageStatus`` objects in canonical order.
    """
    stats = cascade_stats or {}
    timings = stage_timings or {}

    has_data = bool(stats) or bool(timings)

    stages: list[StageStatus] = []
    for key, name in UNIFIED_PIPELINE_STAGES:
        if key == "pdf_extraction":
            stages.append(StageStatus(
                key=key, name=name,
                status="complete" if has_data else "pending",
                elapsed_seconds=timings.get("extraction", 0.0),
                source_pipeline="classified" if has_data else "",
            ))
        elif key == "header_footer":
            stages.append(StageStatus(
                key=key, name=name,
                status="complete" if has_data else "pending",
                elapsed_seconds=timings.get("header_footer", 0.0),
                source_pipeline="classified" if has_data else "",
            ))
        elif key == "ml_classification":
            local_count = stats.get("local_count", 0)
            total = stats.get("total_sections", 0)
            stages.append(StageStatus(
                key=key, name=name,
                status="complete" if stats else "pending",
                elapsed_seconds=timings.get(
                    "classification_cascade", 0.0,
                ),
                source_pipeline="classified" if stats else "",
                metrics={
                    k: v for k, v in {
                        "local_count": local_count,
                        "total_sections": total,
                        "local_pct": stats.get("local_pct", 0.0),
                    }.items() if v
                },
            ))
        elif key == "sonnet_classification":
            sonnet_count = stats.get("sonnet_count", 0)
            stages.append(StageStatus(
                key=key, name=name,
                status="complete" if sonnet_count else "skipped",
                source_pipeline="classified" if stats else "",
                metrics={
                    k: v for k, v in {
                        "sonnet_count": sonnet_count,
                        "agreements": stats.get("agreements", 0),
                        "disagreements": stats.get(
                            "disagreements", 0,
                        ),
                    }.items() if v
                },
            ))
        elif key == "document_assembly":
            stages.append(StageStatus(
                key=key, name=name,
                status="complete" if has_data else "pending",
                elapsed_seconds=timings.get("assembly", 0.0),
                source_pipeline="classified" if has_data else "",
            ))
        elif key in ("content_transforms", "query_route"):
            # These are query-pipeline-specific stages.
            stages.append(StageStatus(
                key=key, name=name, status="skipped",
            ))
        elif key == "soa_construction":
            stages.append(StageStatus(
                key=key, name=name,
                status="complete" if timings.get("soa") else "skipped",
                elapsed_seconds=timings.get("soa", 0.0),
                source_pipeline="classified" if timings.get("soa") else "",
            ))
        else:
            stages.append(StageStatus(
                key=key, name=name, status="pending",
            ))

    return stages


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

_STATUS_PRIORITY = {
    "complete": 0,
    "error": 1,
    "pending": 2,
    "skipped": 3,
}


def merge_pipeline_stages(
    query_stages: list[StageStatus] | None,
    classified_stages: list[StageStatus] | None,
) -> list[StageStatus]:
    """Merge stages from both pipelines, preferring complete over skipped.

    Args:
        query_stages: From ``map_query_stages_to_unified()``, or None.
        classified_stages: From ``map_classified_stages_to_unified()``,
            or None.

    Returns:
        Merged list of 8 ``StageStatus`` objects.
    """
    if query_stages is None and classified_stages is None:
        return [
            StageStatus(key=k, name=n, status="pending")
            for k, n in UNIFIED_PIPELINE_STAGES
        ]
    if query_stages is None:
        return classified_stages  # type: ignore[return-value]
    if classified_stages is None:
        return query_stages

    # Build lookup by key for classified stages.
    classified_by_key = {s.key: s for s in classified_stages}

    merged: list[StageStatus] = []
    for qs in query_stages:
        cs = classified_by_key.get(qs.key)
        if cs is None:
            merged.append(qs)
            continue

        q_pri = _STATUS_PRIORITY.get(qs.status, 9)
        c_pri = _STATUS_PRIORITY.get(cs.status, 9)

        if c_pri < q_pri:
            merged.append(cs)
        elif q_pri < c_pri:
            merged.append(qs)
        else:
            # Same priority — prefer the one with more info.
            if cs.metrics or cs.elapsed_seconds > qs.elapsed_seconds:
                merged.append(cs)
            else:
                merged.append(qs)

    return merged


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

_STATUS_ICONS: dict[str, str] = {
    "complete": "\u2705",   # white_check_mark
    "error": "\u274c",      # x
    "pending": "\u23f3",    # hourglass
    "skipped": "\u2796",    # heavy_minus_sign
}

_STATUS_COLORS: dict[str, str] = {
    "complete": "green",
    "error": "red",
    "pending": "orange",
    "skipped": "grey",
}


def format_stage_display(stage: StageStatus) -> dict[str, str]:
    """Format a StageStatus for UI display.

    Returns:
        Dict with keys: ``"label"``, ``"icon"``, ``"color"``,
        ``"detail_text"``.
    """
    icon = _STATUS_ICONS.get(stage.status, "\u2753")
    color = _STATUS_COLORS.get(stage.status, "grey")

    detail_parts: list[str] = []
    if stage.elapsed_seconds > 0:
        detail_parts.append(f"{stage.elapsed_seconds:.1f}s")
    if stage.source_pipeline:
        detail_parts.append(stage.source_pipeline)
    for mk, mv in stage.metrics.items():
        detail_parts.append(f"{mk}: {mv}")

    return {
        "label": stage.name,
        "icon": icon,
        "color": color,
        "detail_text": " | ".join(detail_parts) if detail_parts else "",
    }
