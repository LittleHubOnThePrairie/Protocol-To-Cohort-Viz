"""Pipeline stage DAG and dependency resolution (PTCV-77).

Defines the 7 UI-visible pipeline stages, their dependency graph,
and helpers for auto-enabling prerequisites, cascading disables,
and topological execution ordering.

Pure-Python module — no Streamlit dependency, fully testable.
"""

from __future__ import annotations

from typing import NamedTuple


class PipelineStage(NamedTuple):
    """A single pipeline stage definition."""

    key: str
    label: str
    prerequisites: tuple[str, ...]
    cache_key: str  # session state cache name ("" = display-only)


# The 7 pipeline stages exposed to the UI.
#
# "extraction" and "retemplating" both resolve via _run_parse() on
# the backend — they share the same cache_key "parse_cache".
# The UI splits them so users see the conceptual pipeline, while
# the backend deduplicates execution automatically.
PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage("extraction", "Extraction", (), "parse_cache"),
    PipelineStage(
        "retemplating", "ICH Retemplating", ("extraction",), "parse_cache",
    ),
    PipelineStage("soa", "SoA Extraction", ("extraction",), "soa_cache"),
    PipelineStage(
        "fidelity", "Fidelity Check", ("retemplating",), "fidelity_cache",
    ),
    PipelineStage(
        "sdtm", "SDTM Generation",
        ("extraction", "soa", "retemplating"), "sdtm_cache",
    ),
    PipelineStage("sov", "Schedule of Visits", ("soa",), ""),
    PipelineStage("annotations", "Annotation Review", ("extraction",), ""),
    # PTCV-95: Query-driven pipeline stages.
    PipelineStage(
        "query_index", "Protocol Index", ("extraction",), "query_cache",
    ),
    PipelineStage(
        "query_match", "Section Matching", ("query_index",), "query_cache",
    ),
    PipelineStage(
        "query_extract", "Query Extraction",
        ("query_match",), "query_cache",
    ),
    PipelineStage(
        "query_assemble", "Template Assembly",
        ("query_extract",), "query_cache",
    ),
    PipelineStage(
        "benchmark", "Benchmark",
        ("retemplating", "query_assemble"), "benchmark_cache",
    ),
)

STAGE_BY_KEY: dict[str, PipelineStage] = {s.key: s for s in PIPELINE_STAGES}


def get_all_prerequisites(stage_key: str) -> set[str]:
    """Return all transitive prerequisites (not including *stage_key*)."""
    result: set[str] = set()
    stack = list(STAGE_BY_KEY[stage_key].prerequisites)
    while stack:
        prereq = stack.pop()
        if prereq not in result:
            result.add(prereq)
            stack.extend(STAGE_BY_KEY[prereq].prerequisites)
    return result


def get_all_dependents(stage_key: str) -> set[str]:
    """Return all transitive dependents (not including *stage_key*)."""
    result: set[str] = set()
    stack = [
        s.key for s in PIPELINE_STAGES
        if stage_key in s.prerequisites
    ]
    while stack:
        dep = stack.pop()
        if dep not in result:
            result.add(dep)
            stack.extend(
                s.key for s in PIPELINE_STAGES
                if dep in s.prerequisites
            )
    return result


def compute_active_stages(user_stages: set[str]) -> set[str]:
    """Compute active stages = user-selected + all transitive prerequisites."""
    active = set(user_stages)
    for key in user_stages:
        active |= get_all_prerequisites(key)
    return active


def get_execution_order(active_stages: set[str]) -> list[str]:
    """Return active stages in topological (dependency) order."""
    return [s.key for s in PIPELINE_STAGES if s.key in active_stages]
