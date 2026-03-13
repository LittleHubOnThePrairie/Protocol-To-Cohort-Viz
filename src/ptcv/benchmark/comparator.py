"""Regression comparator — compares benchmark runs against baselines.

Flags metrics that regress beyond configurable thresholds.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default regression thresholds (fraction of decline before flagging)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "populated_count": 0.0,       # any drop flags
    "average_confidence": 0.05,   # 5% drop
    "answered_required": 0.0,     # any drop flags
    "duplicate_content_count": 0.0,  # any increase flags (inverted)
}


@dataclasses.dataclass
class RegressionFlag:
    """A single regression detected between baseline and current run."""

    protocol_file: str
    metric: str
    baseline_value: float
    current_value: float
    threshold: float
    severity: str  # "warning" or "regression"

    @property
    def delta(self) -> float:
        return self.current_value - self.baseline_value

    @property
    def delta_pct(self) -> float:
        if self.baseline_value == 0:
            return 0.0
        return self.delta / self.baseline_value * 100


@dataclasses.dataclass
class ComparisonResult:
    """Result of comparing a benchmark run against a baseline."""

    baseline_timestamp: str
    current_timestamp: str
    total_protocols: int
    regressions: list[RegressionFlag]
    improvements: list[RegressionFlag]

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_timestamp": self.baseline_timestamp,
            "current_timestamp": self.current_timestamp,
            "total_protocols": self.total_protocols,
            "regression_count": len(self.regressions),
            "improvement_count": len(self.improvements),
            "regressions": [
                dataclasses.asdict(r) for r in self.regressions
            ],
            "improvements": [
                dataclasses.asdict(r) for r in self.improvements
            ],
        }


def _compare_metric(
    protocol: str,
    metric: str,
    baseline_val: float,
    current_val: float,
    threshold: float,
    higher_is_better: bool = True,
) -> RegressionFlag | None:
    """Compare a single metric and return a flag if regressed."""
    if higher_is_better:
        delta = baseline_val - current_val
    else:
        # For metrics where lower is better (e.g. duplicate count)
        delta = current_val - baseline_val

    if delta <= 0:
        return None

    if baseline_val > 0 and (delta / baseline_val) > threshold:
        return RegressionFlag(
            protocol_file=protocol,
            metric=metric,
            baseline_value=baseline_val,
            current_value=current_val,
            threshold=threshold,
            severity="regression",
        )
    elif delta > 0:
        return RegressionFlag(
            protocol_file=protocol,
            metric=metric,
            baseline_value=baseline_val,
            current_value=current_val,
            threshold=threshold,
            severity="warning",
        )
    return None


def compare_runs(
    baseline: dict[str, Any],
    current: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> ComparisonResult:
    """Compare current benchmark results against a baseline.

    Args:
        baseline: Benchmark results dict (from benchmark_results.json).
        current: Current benchmark results dict.
        thresholds: Per-metric regression thresholds. Uses
            ``DEFAULT_THRESHOLDS`` if not provided.

    Returns:
        Comparison result with regressions and improvements.
    """
    thresh = thresholds or DEFAULT_THRESHOLDS

    # Index baseline results by protocol
    baseline_by_proto = {
        r["protocol_file"]: r["metrics"]
        for r in baseline.get("results", [])
        if r.get("success") and r.get("metrics")
    }
    current_by_proto = {
        r["protocol_file"]: r["metrics"]
        for r in current.get("results", [])
        if r.get("success") and r.get("metrics")
    }

    regressions: list[RegressionFlag] = []
    improvements: list[RegressionFlag] = []

    # Higher-is-better metrics
    hib_metrics = [
        "populated_count",
        "average_confidence",
        "answered_required",
    ]
    # Lower-is-better metrics
    lib_metrics = ["duplicate_content_count"]

    common = set(baseline_by_proto) & set(current_by_proto)
    for proto in sorted(common):
        bm = baseline_by_proto[proto]
        cm = current_by_proto[proto]

        for metric in hib_metrics:
            bv = bm.get(metric, 0)
            cv = cm.get(metric, 0)
            flag = _compare_metric(
                proto, metric, bv, cv, thresh.get(metric, 0.0),
                higher_is_better=True,
            )
            if flag:
                regressions.append(flag)
            elif cv > bv:
                improvements.append(RegressionFlag(
                    protocol_file=proto, metric=metric,
                    baseline_value=bv, current_value=cv,
                    threshold=thresh.get(metric, 0.0),
                    severity="improvement",
                ))

        for metric in lib_metrics:
            bv = bm.get(metric, 0)
            cv = cm.get(metric, 0)
            flag = _compare_metric(
                proto, metric, bv, cv, thresh.get(metric, 0.0),
                higher_is_better=False,
            )
            if flag:
                regressions.append(flag)
            elif cv < bv:
                improvements.append(RegressionFlag(
                    protocol_file=proto, metric=metric,
                    baseline_value=bv, current_value=cv,
                    threshold=thresh.get(metric, 0.0),
                    severity="improvement",
                ))

    return ComparisonResult(
        baseline_timestamp=baseline.get("timestamp", "unknown"),
        current_timestamp=current.get("timestamp", "unknown"),
        total_protocols=len(common),
        regressions=regressions,
        improvements=improvements,
    )


def load_baseline(baseline_path: Path) -> dict[str, Any]:
    """Load a baseline from disk.

    Args:
        baseline_path: Path to a ``benchmark_results.json`` file,
            or a directory containing one.

    Returns:
        Parsed baseline dict.
    """
    if baseline_path.is_dir():
        baseline_path = baseline_path / "benchmark_results.json"
    with open(baseline_path) as f:
        return json.load(f)


def find_latest_baseline(baselines_dir: Path) -> Path | None:
    """Find the most recent baseline in the baselines directory."""
    if not baselines_dir.exists():
        return None
    baselines = sorted(baselines_dir.glob("*/benchmark_results.json"))
    if not baselines:
        # Check for a single file
        single = baselines_dir / "benchmark_results.json"
        return single if single.exists() else None
    return baselines[-1]
