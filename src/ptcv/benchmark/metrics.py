"""Quality metrics computation from pipeline output (PTCV-185).

Computes per-protocol quality metrics including coverage, confidence,
extraction method breakdown, and content duplication detection.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any


@dataclasses.dataclass
class ProtocolMetrics:
    """Quality metrics for a single protocol run."""

    protocol_file: str
    total_sections: int
    populated_count: int
    gap_count: int
    average_confidence: float
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    total_queries: int
    answered_queries: int
    required_queries: int
    answered_required: int
    gap_sections: list[str]
    low_confidence_sections: list[str]
    method_breakdown: dict[str, int]
    duplicate_content_count: int
    duplicate_content_groups: int
    stage_timings: dict[str, float]


def compute_protocol_metrics(
    protocol_file: str,
    pipeline_result: dict[str, Any],
) -> ProtocolMetrics:
    """Compute quality metrics from a pipeline result dict.

    Args:
        protocol_file: Name of the protocol PDF.
        pipeline_result: Dict returned by ``run_query_pipeline()``.

    Returns:
        Computed metrics for the protocol.
    """
    coverage = pipeline_result["coverage"]
    extraction_result = pipeline_result["extraction_result"]
    stage_timings = pipeline_result.get("stage_timings", {})

    # Method breakdown
    method_counts: Counter[str] = Counter()
    for ext in extraction_result.extractions:
        method_counts[ext.extraction_method] += 1

    # Content duplication detection — find extractions with identical
    # content across different queries (regurgitation signal)
    content_hashes: dict[str, list[str]] = {}
    for ext in extraction_result.extractions:
        text = ext.content.strip()
        if len(text) > 50:  # ignore short common strings
            if text not in content_hashes:
                content_hashes[text] = []
            content_hashes[text].append(ext.query_id)

    dup_groups = {
        k: v for k, v in content_hashes.items() if len(v) > 1
    }
    dup_count = sum(len(v) for v in dup_groups.values())

    return ProtocolMetrics(
        protocol_file=protocol_file,
        total_sections=coverage.total_sections,
        populated_count=coverage.populated_count,
        gap_count=coverage.gap_count,
        average_confidence=coverage.average_confidence,
        high_confidence_count=coverage.high_confidence_count,
        medium_confidence_count=coverage.medium_confidence_count,
        low_confidence_count=coverage.low_confidence_count,
        total_queries=coverage.total_queries,
        answered_queries=coverage.answered_queries,
        required_queries=coverage.required_queries,
        answered_required=coverage.answered_required,
        gap_sections=list(coverage.gap_sections),
        low_confidence_sections=list(coverage.low_confidence_sections),
        method_breakdown=dict(method_counts),
        duplicate_content_count=dup_count,
        duplicate_content_groups=len(dup_groups),
        stage_timings=dict(stage_timings),
    )
