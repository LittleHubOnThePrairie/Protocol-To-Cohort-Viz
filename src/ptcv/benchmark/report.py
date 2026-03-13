"""Human-readable benchmark report generation (PTCV-185)."""

from __future__ import annotations

from typing import Any

from ptcv.benchmark.comparator import ComparisonResult
from ptcv.benchmark.runner import BenchmarkResult


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Generate a Markdown benchmark report.

    Args:
        result: Completed benchmark result.

    Returns:
        Formatted Markdown string.
    """
    lines: list[str] = []
    lines.append("# Pipeline Benchmark Report")
    lines.append("")
    lines.append(f"**Timestamp:** {result.timestamp}")
    lines.append(
        f"**Corpus:** {result.successful_runs}/{result.corpus_size} "
        f"protocols succeeded"
    )
    lines.append(
        f"**LLM Transform:** "
        f"{'enabled' if result.enable_transformation else 'disabled'}"
    )
    lines.append(
        f"**Summarization:** "
        f"{'enabled' if result.enable_summarization else 'disabled'}"
    )
    lines.append("")

    # Per-protocol results table
    lines.append("## Per-Protocol Results")
    lines.append("")
    lines.append(
        "| Protocol | Populated | Gaps | Avg Conf | "
        "Answered Req | Duplicates | Time (s) |"
    )
    lines.append(
        "|----------|-----------|------|----------|"
        "--------------|------------|----------|"
    )

    for r in result.results:
        if not r.success:
            lines.append(
                f"| {r.protocol_file} | FAILED | - | - | "
                f"- | - | {r.elapsed_seconds:.1f} |"
            )
            continue
        m = r.metrics
        assert m is not None
        req_pct = (
            f"{m.answered_required}/{m.required_queries}"
        )
        lines.append(
            f"| {r.protocol_file} | "
            f"{m.populated_count}/{m.total_sections} | "
            f"{m.gap_count} | "
            f"{m.average_confidence:.3f} | "
            f"{req_pct} | "
            f"{m.duplicate_content_count} | "
            f"{r.elapsed_seconds:.1f} |"
        )

    lines.append("")

    # Method breakdown
    lines.append("## Extraction Method Breakdown")
    lines.append("")
    lines.append("| Protocol | " + " | ".join([
        "llm_transform", "passthrough", "relevance_excerpt",
        "regex", "heuristic", "unscoped_search", "table_detection",
    ]) + " |")
    lines.append(
        "|----------|" + "|".join(["---"] * 7) + "|"
    )

    methods = [
        "llm_transform", "passthrough", "relevance_excerpt",
        "regex", "heuristic", "unscoped_search", "table_detection",
    ]
    for r in result.results:
        if not r.success or not r.metrics:
            continue
        m = r.metrics
        cols = [str(m.method_breakdown.get(method, 0)) for method in methods]
        lines.append(f"| {r.protocol_file} | " + " | ".join(cols) + " |")

    lines.append("")

    # Gap sections
    lines.append("## Gap Sections")
    lines.append("")
    for r in result.results:
        if not r.success or not r.metrics:
            continue
        m = r.metrics
        if m.gap_sections:
            lines.append(
                f"- **{r.protocol_file}:** {', '.join(m.gap_sections)}"
            )
    lines.append("")

    # Aggregate timings
    if result.aggregate_timings:
        lines.append("## Aggregate Timings (seconds)")
        lines.append("")
        lines.append(
            "| Stage | Mean | Median | P95 | Min | Max |"
        )
        lines.append(
            "|-------|------|--------|-----|-----|-----|"
        )
        for stage, stats in result.aggregate_timings.items():
            lines.append(
                f"| {stage} | "
                f"{stats['mean']:.1f} | "
                f"{stats['median']:.1f} | "
                f"{stats['p95']:.1f} | "
                f"{stats['min']:.1f} | "
                f"{stats['max']:.1f} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_comparison_report(comparison: ComparisonResult) -> str:
    """Generate a Markdown comparison report.

    Args:
        comparison: Result from ``compare_runs()``.

    Returns:
        Formatted Markdown string.
    """
    lines: list[str] = []
    lines.append("# Benchmark Comparison Report")
    lines.append("")
    lines.append(
        f"**Baseline:** {comparison.baseline_timestamp}"
    )
    lines.append(
        f"**Current:** {comparison.current_timestamp}"
    )
    lines.append(
        f"**Protocols compared:** {comparison.total_protocols}"
    )
    lines.append("")

    if comparison.has_regressions:
        lines.append(
            f"## REGRESSIONS DETECTED ({len(comparison.regressions)})"
        )
        lines.append("")
        lines.append(
            "| Protocol | Metric | Baseline | Current | Delta |"
        )
        lines.append(
            "|----------|--------|----------|---------|-------|"
        )
        for r in comparison.regressions:
            lines.append(
                f"| {r.protocol_file} | {r.metric} | "
                f"{r.baseline_value} | {r.current_value} | "
                f"{r.delta:+.3f} ({r.delta_pct:+.1f}%) |"
            )
        lines.append("")
    else:
        lines.append("## No Regressions Detected")
        lines.append("")

    if comparison.improvements:
        lines.append(
            f"## Improvements ({len(comparison.improvements)})"
        )
        lines.append("")
        lines.append(
            "| Protocol | Metric | Baseline | Current | Delta |"
        )
        lines.append(
            "|----------|--------|----------|---------|-------|"
        )
        for r in comparison.improvements:
            lines.append(
                f"| {r.protocol_file} | {r.metric} | "
                f"{r.baseline_value} | {r.current_value} | "
                f"{r.delta:+.3f} ({r.delta_pct:+.1f}%) |"
            )
        lines.append("")

    return "\n".join(lines)
