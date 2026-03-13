"""Pipeline integration harness (PTCV-150).

Connects batch runner output to the analysis and comparison tools.
Provides:
- Regression detection with configurable thresholds
- Summary report generation (markdown for Jira/Confluence)
- Before/after comparison reporting

Usage::

    harness = AnalysisHarness("data/analysis/results.db")
    report = harness.generate_summary_report("run_001")
    print(report)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ptcv.analysis.data_store import AnalysisStore


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def detect_regressions(
    comparison: dict[str, Any],
    threshold: float = 0.02,
) -> list[dict[str, Any]]:
    """Flag sections with confidence drops exceeding threshold.

    Args:
        comparison: Output from ``AnalysisStore.compare_runs()``.
        threshold: Minimum absolute delta to flag as regressed.

    Returns:
        List of regressed section dicts with delta and severity.
    """
    regressions: list[dict[str, Any]] = []
    for section in comparison.get("sections", []):
        delta = section.get("delta", 0.0)
        if delta < -abs(threshold):
            severity = "critical" if delta < -0.10 else "warning"
            regressions.append({
                "ich_section_code": section["ich_section_code"],
                "run_a_avg_score": section["run_a_avg_score"],
                "run_b_avg_score": section["run_b_avg_score"],
                "delta": delta,
                "severity": severity,
            })
    return sorted(regressions, key=lambda r: r["delta"])


# ---------------------------------------------------------------------------
# Summary report generator
# ---------------------------------------------------------------------------

def generate_summary_report(
    store: AnalysisStore,
    run_id: str,
    top_n: int = 5,
) -> str:
    """Generate a markdown summary report for a batch run.

    Suitable for pasting into a Jira comment or Confluence page.

    Args:
        store: Open AnalysisStore instance.
        run_id: Batch run ID to summarize.
        top_n: Number of top/bottom sections to include.

    Returns:
        Markdown string.
    """
    run = store.get_run(run_id)
    if run is None:
        return f"**Error:** Run `{run_id}` not found."

    stats = store.get_section_stats(run_id)
    cov = store.get_coverage_distribution(run_id)

    total = run.get("protocol_count", 0) or 1
    pass_count = run.get("pass_count", 0)
    error_count = run.get("error_count", 0)
    pass_rate = round(pass_count / total * 100, 1)
    error_rate = round(error_count / total * 100, 1)

    avg_coverage = cov.get("avg_coverage_pct", 0.0)
    avg_confidence = cov.get("avg_confidence", 0.0)

    # Sort sections by avg confidence ascending
    sorted_stats = sorted(stats, key=lambda s: s["avg_boosted"])

    lines: list[str] = []
    lines.append(f"## Batch Run Summary: `{run_id}`\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Protocols | {total} |")
    lines.append(f"| Pass rate | {pass_rate}% ({pass_count}/{total}) |")
    lines.append(f"| Error rate | {error_rate}% ({error_count}/{total}) |")
    lines.append(f"| Avg coverage | {avg_coverage:.1f}% |")
    lines.append(f"| Avg confidence | {avg_confidence:.4f} |")
    lines.append(
        f"| Elapsed | {run.get('elapsed_seconds', 0.0):.1f}s |",
    )
    lines.append("")

    # Top-N problem sections
    if sorted_stats:
        worst = sorted_stats[:top_n]
        lines.append(f"### Top-{top_n} Problem Sections\n")
        lines.append(
            "| ICH Section | Avg Confidence | Auto-Map Rate "
            "| Hit Rate | Protocols |",
        )
        lines.append("|-------------|---------------|"
                      "--------------|----------|-----------|")
        for s in worst:
            lines.append(
                f"| {s['ich_section_code']} "
                f"({s['ich_section_name']}) "
                f"| {s['avg_boosted']:.4f} "
                f"| {s['auto_map_rate']:.1%} "
                f"| {s['hit_rate']:.1%} "
                f"| {s['protocol_count']} |",
            )
        lines.append("")

    # Best sections
    if sorted_stats:
        best = list(reversed(sorted_stats[-top_n:]))
        lines.append(f"### Top-{top_n} Best Sections\n")
        lines.append(
            "| ICH Section | Avg Confidence | Auto-Map Rate "
            "| Hit Rate |",
        )
        lines.append("|-------------|---------------|"
                      "--------------|----------|")
        for s in best:
            lines.append(
                f"| {s['ich_section_code']} "
                f"({s['ich_section_name']}) "
                f"| {s['avg_boosted']:.4f} "
                f"| {s['auto_map_rate']:.1%} "
                f"| {s['hit_rate']:.1%} |",
            )
        lines.append("")

    # Coverage distribution
    buckets = cov.get("buckets", {})
    if buckets:
        lines.append("### Coverage Distribution\n")
        lines.append("| Bucket | Protocols |")
        lines.append("|--------|-----------|")
        for bucket, count in buckets.items():
            lines.append(f"| {bucket} | {count} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regression report generator
# ---------------------------------------------------------------------------

def generate_regression_report(
    store: AnalysisStore,
    run_id_a: str,
    run_id_b: str,
    threshold: float = 0.02,
) -> str:
    """Generate a markdown regression report comparing two runs.

    Args:
        store: Open AnalysisStore instance.
        run_id_a: Baseline run ID.
        run_id_b: New run ID.
        threshold: Delta threshold for flagging regressions.

    Returns:
        Markdown string.
    """
    comparison = store.compare_runs(run_id_a, run_id_b)
    regressions = detect_regressions(comparison, threshold)

    lines: list[str] = []
    lines.append(
        f"## Regression Report: `{run_id_a}` → `{run_id_b}`\n",
    )

    cov_delta = comparison.get("coverage_delta", 0.0)
    conf_delta = comparison.get("confidence_delta", 0.0)
    cov_direction = "+" if cov_delta >= 0 else ""
    conf_direction = "+" if conf_delta >= 0 else ""

    lines.append("| Metric | Delta |")
    lines.append("|--------|-------|")
    lines.append(
        f"| Coverage | {cov_direction}{cov_delta:.2f} pp |",
    )
    lines.append(
        f"| Confidence | {conf_direction}{conf_delta:.4f} |",
    )
    lines.append("")

    if not regressions:
        lines.append("**No regressions detected.**\n")
    else:
        lines.append(
            f"### Regressions ({len(regressions)} sections)\n",
        )
        lines.append(
            "| ICH Section | Baseline | New | Delta | Severity |",
        )
        lines.append(
            "|-------------|----------|-----|-------|----------|",
        )
        for r in regressions:
            lines.append(
                f"| {r['ich_section_code']} "
                f"| {r['run_a_avg_score']:.4f} "
                f"| {r['run_b_avg_score']:.4f} "
                f"| {r['delta']:.4f} "
                f"| {r['severity']} |",
            )
        lines.append("")

    # Improvements
    improvements = [
        s for s in comparison.get("sections", [])
        if s.get("status") == "improved"
    ]
    if improvements:
        lines.append(
            f"### Improvements ({len(improvements)} sections)\n",
        )
        lines.append(
            "| ICH Section | Baseline | New | Delta |",
        )
        lines.append(
            "|-------------|----------|-----|-------|",
        )
        for s in sorted(
            improvements, key=lambda x: x["delta"], reverse=True,
        ):
            lines.append(
                f"| {s['ich_section_code']} "
                f"| {s['run_a_avg_score']:.4f} "
                f"| {s['run_b_avg_score']:.4f} "
                f"| +{s['delta']:.4f} |",
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis harness
# ---------------------------------------------------------------------------

class AnalysisHarness:
    """High-level integration layer for the analysis pipeline.

    Wraps AnalysisStore with report generation and regression
    detection capabilities.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._store: AnalysisStore | None = None

    @property
    def store(self) -> AnalysisStore:
        """Lazy-open the store."""
        if self._store is None:
            self._store = AnalysisStore(self._db_path)
        return self._store

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def list_runs(self) -> list[dict[str, Any]]:
        """List all batch runs, most recent first."""
        return self.store.list_runs()

    def get_run_summary(self, run_id: str) -> str:
        """Generate markdown summary for a batch run."""
        return generate_summary_report(self.store, run_id)

    def get_regression_report(
        self,
        run_id_a: str,
        run_id_b: str,
        threshold: float = 0.02,
    ) -> str:
        """Generate markdown regression report."""
        return generate_regression_report(
            self.store, run_id_a, run_id_b, threshold,
        )

    def get_regressions(
        self,
        run_id_a: str,
        run_id_b: str,
        threshold: float = 0.02,
    ) -> list[dict[str, Any]]:
        """Detect regressions between two runs."""
        comparison = self.store.compare_runs(run_id_a, run_id_b)
        return detect_regressions(comparison, threshold)

    def get_section_stats(
        self, run_id: str,
    ) -> list[dict[str, Any]]:
        """Get per-section statistics for a run."""
        return self.store.get_section_stats(run_id)

    def get_comparison_pairs(
        self,
        run_id: str,
        nct_id: str,
        ich_section_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get original vs extracted text pairs."""
        return self.store.get_comparison_pairs(
            run_id, nct_id, ich_section_code,
        )

    def list_protocols(
        self, run_id: str, status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List protocols in a run."""
        return self.store.list_protocols(run_id, status)
