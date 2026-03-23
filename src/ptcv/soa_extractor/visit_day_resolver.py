"""Visit Day Resolver: Convert Visit Labels to Real Study Days.

PTCV-253: Resolves human-readable visit labels (e.g., "Cycle 3 Day 1",
"Screening", "Week 4") to absolute study day numbers for chronological
timeline rendering.

Day conventions:
    Day 1   = first treatment/baseline day
    Day -14 = typical screening (14 days before Day 1)
    Day 0   = baseline (when distinct from Day 1)

Strategies (applied in priority order):
    A. Parse explicit day/week from visit label (regex)
    B. Resolve cycle-based visits using detected cycle length
    C. Assign known milestone days (Screening, Baseline, EOT, Follow-up)

Risk tier: LOW — data transformation only (no API calls).

Usage::

    from ptcv.soa_extractor.visit_day_resolver import (
        VisitDayResolver,
        resolve_timepoint_days,
    )

    resolver = VisitDayResolver(cycle_length=21)
    real_day = resolver.resolve("Cycle 3 Day 1")  # → 43
    real_day = resolver.resolve("Screening")       # → -14
    real_day = resolver.resolve("Week 4")          # → 28
"""

import dataclasses
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CYCLE_LENGTH = 21  # Most common oncology cycle
_DEFAULT_SCREENING_DAY = -14
_DEFAULT_BASELINE_DAY = 1
_DEFAULT_EOT_OFFSET = 1  # EOT = last resolved day + 1
_DEFAULT_FOLLOW_UP_OFFSET = 30  # Follow-up = EOT + 30


# ---------------------------------------------------------------------------
# Regex patterns for visit label parsing
# ---------------------------------------------------------------------------

# "Day 15", "Day 1", "D15", "D1"
_DAY_RE = re.compile(
    r"\bDay\s*(\d+)\b|\bD(\d+)\b",
    re.IGNORECASE,
)

# "Week 4", "Wk 4", "W4"
_WEEK_RE = re.compile(
    r"\bWeek\s*(\d+)\b|\bWk\s*(\d+)\b|\bW(\d+)\b",
    re.IGNORECASE,
)

# "Month 6", "Mo 3"
_MONTH_RE = re.compile(
    r"\bMonth\s*(\d+)\b|\bMo\s*(\d+)\b",
    re.IGNORECASE,
)

# "Cycle 3 Day 1", "C3D1", "Cycle 3, Day 1", "C3 D1"
_CYCLE_DAY_RE = re.compile(
    r"(?:Cycle\s*|C)(\d+)\s*[,/]?\s*(?:Day\s*|D)(\d+)",
    re.IGNORECASE,
)

# "Cycle 3" without day (implies Day 1 of that cycle)
_CYCLE_ONLY_RE = re.compile(
    r"(?:Cycle\s*|C)(\d+)\b(?!\s*[,/]?\s*(?:Day|D)\d)",
    re.IGNORECASE,
)

# Cycle length from protocol text: "21-day cycle", "q3w", "every 28 days"
_CYCLE_LENGTH_RE = re.compile(
    r"(\d+)\s*-?\s*day\s+cycle|"
    r"q(\d+)w|"
    r"every\s+(\d+)\s+days|"
    r"(\d+)\s*-?\s*day\s+treatment\s+cycle",
    re.IGNORECASE,
)

# Milestones
_SCREENING_RE = re.compile(r"\bscreen(?:ing)?\b", re.IGNORECASE)
_BASELINE_RE = re.compile(r"\bbaseline\b", re.IGNORECASE)
_EOT_RE = re.compile(
    r"\bend\s+of\s+(?:treatment|study)\b|\bEOT\b|\bEOS\b",
    re.IGNORECASE,
)
_FOLLOW_UP_RE = re.compile(
    r"\bfollow[\s-]*up\b|\bFU\b|\bpost[\s-]*treatment\b",
    re.IGNORECASE,
)

# "Every N Cycles Thereafter", "q6w thereafter"
_REPEATING_RE = re.compile(
    r"\bevery\s+(\d+)\s+cycles?\s+thereafter\b|"
    r"\bq(\d+)(?:w|c)\s+thereafter\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ResolvedVisit:
    """A visit label resolved to a real study day.

    Attributes:
        visit_label: Original visit label string.
        real_day: Absolute study day (Day 1 = first treatment).
        resolution_method: How the day was determined.
        confidence: Confidence in the resolution (0.0-1.0).
    """

    visit_label: str
    real_day: int
    resolution_method: str = ""
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# VisitDayResolver
# ---------------------------------------------------------------------------


class VisitDayResolver:
    """Resolve visit labels to absolute study day numbers.

    Args:
        cycle_length: Treatment cycle length in days (default: 21).
        screening_day: Day number for screening visits (default: -14).
        baseline_day: Day number for baseline (default: 1).
    """

    def __init__(
        self,
        cycle_length: int = _DEFAULT_CYCLE_LENGTH,
        screening_day: int = _DEFAULT_SCREENING_DAY,
        baseline_day: int = _DEFAULT_BASELINE_DAY,
    ) -> None:
        self._cycle_length = cycle_length
        self._screening_day = screening_day
        self._baseline_day = baseline_day

    @property
    def cycle_length(self) -> int:
        return self._cycle_length

    def resolve(self, visit_label: str) -> ResolvedVisit:
        """Resolve a single visit label to a study day.

        Tries strategies in priority order:
        1. Explicit day number ("Day 15")
        2. Cycle + day ("Cycle 3 Day 1")
        3. Week/month conversion ("Week 4")
        4. Milestone keywords (Screening, Baseline, EOT)
        5. Cycle-only ("Cycle 3" → Day 1 of cycle 3)

        Args:
            visit_label: Human-readable visit label.

        Returns:
            ResolvedVisit with real_day and method.
        """
        label = visit_label.strip()

        # Strategy 1: Cycle + Day (must check before plain Day)
        m = _CYCLE_DAY_RE.search(label)
        if m:
            cycle = int(m.group(1))
            day = int(m.group(2))
            real_day = (cycle - 1) * self._cycle_length + day
            return ResolvedVisit(
                visit_label=label,
                real_day=real_day,
                resolution_method="cycle_day",
                confidence=0.95,
            )

        # Strategy 2: Plain day number
        m = _DAY_RE.search(label)
        if m:
            day = int(m.group(1) or m.group(2))
            # Check if baseline is mentioned alongside day
            if _BASELINE_RE.search(label) and day == 1:
                return ResolvedVisit(
                    visit_label=label,
                    real_day=self._baseline_day,
                    resolution_method="baseline_day1",
                    confidence=1.0,
                )
            return ResolvedVisit(
                visit_label=label,
                real_day=day,
                resolution_method="explicit_day",
                confidence=1.0,
            )

        # Strategy 3: Week
        m = _WEEK_RE.search(label)
        if m:
            week = int(m.group(1) or m.group(2) or m.group(3))
            return ResolvedVisit(
                visit_label=label,
                real_day=(week - 1) * 7 + 1,
                resolution_method="week",
                confidence=0.9,
            )

        # Strategy 4: Month
        m = _MONTH_RE.search(label)
        if m:
            month = int(m.group(1) or m.group(2))
            return ResolvedVisit(
                visit_label=label,
                real_day=(month - 1) * 30 + 1,
                resolution_method="month",
                confidence=0.8,
            )

        # Strategy 5: Milestones
        if _SCREENING_RE.search(label):
            return ResolvedVisit(
                visit_label=label,
                real_day=self._screening_day,
                resolution_method="screening",
                confidence=1.0,
            )

        if _BASELINE_RE.search(label):
            return ResolvedVisit(
                visit_label=label,
                real_day=self._baseline_day,
                resolution_method="baseline",
                confidence=1.0,
            )

        if _EOT_RE.search(label):
            return ResolvedVisit(
                visit_label=label,
                real_day=0,  # Placeholder — resolved in batch
                resolution_method="eot",
                confidence=0.7,
            )

        if _FOLLOW_UP_RE.search(label):
            return ResolvedVisit(
                visit_label=label,
                real_day=0,  # Placeholder — resolved in batch
                resolution_method="follow_up",
                confidence=0.7,
            )

        # Strategy 6: Cycle-only
        m = _CYCLE_ONLY_RE.search(label)
        if m:
            cycle = int(m.group(1))
            real_day = (cycle - 1) * self._cycle_length + 1
            return ResolvedVisit(
                visit_label=label,
                real_day=real_day,
                resolution_method="cycle_only",
                confidence=0.85,
            )

        # Strategy 7: Repeating pattern
        m = _REPEATING_RE.search(label)
        if m:
            interval = int(m.group(1) or m.group(2))
            return ResolvedVisit(
                visit_label=label,
                real_day=0,  # Placeholder
                resolution_method="repeating",
                confidence=0.5,
            )

        # Fallback: unresolved
        return ResolvedVisit(
            visit_label=label,
            real_day=0,
            resolution_method="unresolved",
            confidence=0.0,
        )

    def resolve_batch(
        self,
        visit_labels: list[str],
    ) -> list[ResolvedVisit]:
        """Resolve a list of visit labels and fix relative milestones.

        After individual resolution, EOT and Follow-up days are
        computed relative to the last resolved concrete day.

        Args:
            visit_labels: Ordered list of visit label strings.

        Returns:
            List of ResolvedVisit in chronological order (sorted by real_day).
        """
        resolved = [self.resolve(label) for label in visit_labels]

        # Find the maximum concrete day for EOT/follow-up
        max_day = max(
            (r.real_day for r in resolved if r.real_day > 0),
            default=1,
        )

        # Fix placeholder days
        for r in resolved:
            if r.resolution_method == "eot" and r.real_day == 0:
                r.real_day = max_day + _DEFAULT_EOT_OFFSET
            elif r.resolution_method == "follow_up" and r.real_day == 0:
                r.real_day = max_day + _DEFAULT_FOLLOW_UP_OFFSET
            elif r.resolution_method == "repeating" and r.real_day == 0:
                r.real_day = max_day + self._cycle_length

        # Sort by real_day for chronological order
        resolved.sort(key=lambda r: r.real_day)

        return resolved

    @staticmethod
    def detect_cycle_length(protocol_text: str) -> Optional[int]:
        """Extract cycle length from protocol design text.

        Searches for patterns like "21-day cycle", "q3w", "every 28 days".

        Args:
            protocol_text: Text from B.4 (Trial Design) or similar.

        Returns:
            Cycle length in days, or None if not detected.
        """
        if not protocol_text:
            return None

        m = _CYCLE_LENGTH_RE.search(protocol_text)
        if m:
            # Groups: (N-day cycle, qNw, every N days, N-day treatment cycle)
            for group in m.groups():
                if group is not None:
                    val = int(group)
                    # qNw means N weeks
                    if m.group(2) is not None:  # qNw pattern
                        return val * 7
                    return val

        return None


def resolve_timepoint_days(
    timepoints: list[Any],
    protocol_text: str = "",
    cycle_length: Optional[int] = None,
) -> list[ResolvedVisit]:
    """Convenience: resolve UsdmTimepoint visit names to real days.

    Args:
        timepoints: List of UsdmTimepoint objects (or dicts with visit_name).
        protocol_text: Optional B.4 text for cycle length detection.
        cycle_length: Override cycle length (auto-detected if None).

    Returns:
        List of ResolvedVisit sorted chronologically.
    """
    # Detect cycle length
    if cycle_length is None:
        cycle_length = VisitDayResolver.detect_cycle_length(protocol_text)
    if cycle_length is None:
        cycle_length = _DEFAULT_CYCLE_LENGTH

    resolver = VisitDayResolver(cycle_length=cycle_length)

    labels = []
    for tp in timepoints:
        if hasattr(tp, "visit_name"):
            labels.append(tp.visit_name)
        elif isinstance(tp, dict):
            labels.append(tp.get("visit_name", str(tp)))
        else:
            labels.append(str(tp))

    return resolver.resolve_batch(labels)


__all__ = [
    "VisitDayResolver",
    "ResolvedVisit",
    "resolve_timepoint_days",
]
