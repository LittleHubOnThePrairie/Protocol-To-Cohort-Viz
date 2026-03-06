"""Visit schedule feasibility validator (PTCV-57).

Validates cross-visit relationships in TV/TE domain DataFrames to catch
protocol design errors and SoA extraction artifacts. Implements 12 rules
(VS-001 through VS-012) modelled on Pinnacle 21 validation patterns.

Risk tier: MEDIUM — regulatory validation; no patient data.

Regulatory references:
- PTCV-1 Research: Pinnacle 21 validation patterns (Citation [7])
- PTCV-1 Research: Faro Smart Design visit matrix validation (Citation [1])
- SDTMIG v3.4 Section 7.4: Trial Design Domains
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from ..models import SoaCellMetadata


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScheduleIssue:
    """One visit-schedule validation finding.

    Attributes:
        rule_id: Rule identifier (VS-001 through VS-012).
        severity: "Error" or "Warning".
        domain: SDTM domain ("TV" or "TE").
        variable: Variable involved (e.g. "VISITDY", "VISITNUM").
        description: Human-readable issue description.
        remediation_guidance: Suggested fix.
        row_number: 1-based TV row number, or None for domain-level.
    """

    rule_id: str
    severity: str
    domain: str
    variable: str
    description: str
    remediation_guidance: str
    row_number: int | None = None


@dataclasses.dataclass
class ScheduleValidationReport:
    """Aggregated schedule validation result.

    Attributes:
        issues: All issues found.
        error_count: Number of Error-severity issues.
        warning_count: Number of Warning-severity issues.
        feasible: True when no errors are present.
    """

    issues: list[ScheduleIssue]
    error_count: int
    warning_count: int
    feasible: bool


# ---------------------------------------------------------------------------
# Visit type classification from VISIT name
# ---------------------------------------------------------------------------

_VISIT_KEYWORDS: list[tuple[str, str]] = [
    ("screen", "Screening"),
    ("baseline", "Baseline"),
    ("follow-up", "Follow-up"),
    ("follow up", "Follow-up"),
    ("followup", "Follow-up"),
    ("early term", "Early Termination"),
    ("early withdraw", "Early Termination"),
    ("end of study", "End of Study"),
    ("end-of-study", "End of Study"),
    ("eos", "End of Study"),
]


def _classify_visit(visit_name: str) -> str:
    """Classify a VISIT name to a visit type via keyword matching.

    Returns one of: Screening, Baseline, Treatment, Follow-up,
    Early Termination, End of Study. Defaults to Treatment.
    """
    lower = visit_name.lower()
    for keyword, visit_type in _VISIT_KEYWORDS:
        if keyword in lower:
            return visit_type
    return "Treatment"


# Epoch code → visit types that belong to it
_EPOCH_VISIT_TYPES: dict[str, set[str]] = {
    "SCRN": {"Screening"},
    "TRT": {"Baseline", "Treatment"},
    "FUP": {"Follow-up"},
    "EOS": {"End of Study", "Early Termination"},
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class VisitScheduleValidator:
    """Validates visit schedule feasibility from TV + TE DataFrames.

    Args:
        tv_df: Trial Visits domain DataFrame.
        te_df: Trial Elements domain DataFrame.
        soa_matrix: Optional cell-level metadata for VS-012 check.
    """

    def __init__(
        self,
        tv_df: pd.DataFrame,
        te_df: pd.DataFrame,
        soa_matrix: dict[tuple[int, str], SoaCellMetadata] | None = None,
    ) -> None:
        self._tv = tv_df.copy()
        self._te = te_df.copy()
        self._soa_matrix = soa_matrix or {}

    def validate(self) -> ScheduleValidationReport:
        """Run all 12 validation rules.

        Returns:
            ScheduleValidationReport with issues sorted by severity.
        """
        issues: list[ScheduleIssue] = []

        issues.extend(self._check_screening_before_treatment())
        issues.extend(self._check_followup_after_treatment())
        issues.extend(self._check_early_term_after_screening())
        issues.extend(self._check_window_overlap())
        issues.extend(self._check_window_proportion())
        issues.extend(self._check_duplicate_visits())
        issues.extend(self._check_visit_frequency())
        issues.extend(self._check_epoch_coverage())
        issues.extend(self._check_epoch_ordering())
        issues.extend(self._check_visitdy_completeness())
        issues.extend(self._check_study_duration())
        issues.extend(self._check_empty_visits())

        # Sort: Errors first, then Warnings
        severity_order = {"Error": 0, "Warning": 1}
        issues.sort(
            key=lambda i: (severity_order.get(i.severity, 9), i.rule_id)
        )

        error_count = sum(1 for i in issues if i.severity == "Error")
        warning_count = sum(1 for i in issues if i.severity == "Warning")

        return ScheduleValidationReport(
            issues=issues,
            error_count=error_count,
            warning_count=warning_count,
            feasible=error_count == 0,
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _visits_by_type(self) -> dict[str, list[dict[str, Any]]]:
        """Group TV rows by classified visit type."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for _, row in self._tv.iterrows():
            vtype = _classify_visit(str(row.get("VISIT", "")))
            groups.setdefault(vtype, []).append(dict(row))
        return groups

    # ------------------------------------------------------------------
    # VS-001: Screening must precede Treatment
    # ------------------------------------------------------------------

    def _check_screening_before_treatment(self) -> list[ScheduleIssue]:
        groups = self._visits_by_type()
        screening = groups.get("Screening", [])
        treatment = groups.get("Treatment", []) + groups.get("Baseline", [])

        if not screening or not treatment:
            return []

        max_screen_day = max(v["VISITDY"] for v in screening)
        min_treat_day = min(v["VISITDY"] for v in treatment)

        if max_screen_day >= min_treat_day:
            return [ScheduleIssue(
                rule_id="VS-001",
                severity="Error",
                domain="TV",
                variable="VISITDY",
                description=(
                    f"Screening visit (Day {max_screen_day}) does not "
                    f"precede first treatment visit (Day {min_treat_day})."
                ),
                remediation_guidance=(
                    "Verify visit ordering: Screening visits must have "
                    "VISITDY < first Treatment/Baseline VISITDY."
                ),
            )]
        return []

    # ------------------------------------------------------------------
    # VS-002: Follow-up must follow Treatment
    # ------------------------------------------------------------------

    def _check_followup_after_treatment(self) -> list[ScheduleIssue]:
        groups = self._visits_by_type()
        followup = groups.get("Follow-up", [])
        treatment = groups.get("Treatment", []) + groups.get("Baseline", [])

        if not followup or not treatment:
            return []

        min_fu_day = min(v["VISITDY"] for v in followup)
        max_treat_day = max(v["VISITDY"] for v in treatment)

        if min_fu_day <= max_treat_day:
            return [ScheduleIssue(
                rule_id="VS-002",
                severity="Error",
                domain="TV",
                variable="VISITDY",
                description=(
                    f"Follow-up visit (Day {min_fu_day}) does not follow "
                    f"last treatment visit (Day {max_treat_day})."
                ),
                remediation_guidance=(
                    "Verify visit ordering: Follow-up visits must have "
                    "VISITDY > last Treatment VISITDY."
                ),
            )]
        return []

    # ------------------------------------------------------------------
    # VS-003: Early Termination must not precede Screening
    # ------------------------------------------------------------------

    def _check_early_term_after_screening(self) -> list[ScheduleIssue]:
        groups = self._visits_by_type()
        et = groups.get("Early Termination", [])
        screening = groups.get("Screening", [])

        if not et or not screening:
            return []

        min_et_day = min(v["VISITDY"] for v in et)
        min_screen_day = min(v["VISITDY"] for v in screening)

        if min_et_day <= min_screen_day:
            return [ScheduleIssue(
                rule_id="VS-003",
                severity="Error",
                domain="TV",
                variable="VISITDY",
                description=(
                    f"Early Termination visit (Day {min_et_day}) precedes "
                    f"Screening visit (Day {min_screen_day})."
                ),
                remediation_guidance=(
                    "Verify visit ordering: Early Termination must have "
                    "VISITDY > Screening VISITDY."
                ),
            )]
        return []

    # ------------------------------------------------------------------
    # VS-004: Visit windows must not overlap
    # ------------------------------------------------------------------

    def _check_window_overlap(self) -> list[ScheduleIssue]:
        if self._tv.empty or len(self._tv) < 2:
            return []

        issues: list[ScheduleIssue] = []
        sorted_tv = self._tv.sort_values("VISITDY").reset_index(drop=True)

        for i in range(len(sorted_tv) - 1):
            curr = sorted_tv.iloc[i]
            nxt = sorted_tv.iloc[i + 1]

            curr_day = float(curr["VISITDY"])
            nxt_day = float(nxt["VISITDY"])

            # TVSTRL is stored as negative of early window
            curr_end = curr_day + float(curr.get("TVENRL", 0))
            nxt_start = nxt_day + float(nxt.get("TVSTRL", 0))

            if curr_end > nxt_start:
                overlap = curr_end - nxt_start
                issues.append(ScheduleIssue(
                    rule_id="VS-004",
                    severity="Warning",
                    domain="TV",
                    variable="TVSTRL",
                    description=(
                        f"Visit {curr['VISIT']} (Day {curr_day} +{curr.get('TVENRL', 0)}) "
                        f"overlaps with {nxt['VISIT']} (Day {nxt_day} "
                        f"{nxt.get('TVSTRL', 0)}) by {overlap:.0f} day(s)."
                    ),
                    remediation_guidance=(
                        "Reduce visit window width or increase inter-visit "
                        "interval to eliminate overlap."
                    ),
                    row_number=i + 1,
                ))
        return issues

    # ------------------------------------------------------------------
    # VS-005: Window duration <= 50% of inter-visit interval
    # ------------------------------------------------------------------

    def _check_window_proportion(self) -> list[ScheduleIssue]:
        if self._tv.empty or len(self._tv) < 2:
            return []

        issues: list[ScheduleIssue] = []
        sorted_tv = self._tv.sort_values("VISITDY").reset_index(drop=True)

        for i in range(len(sorted_tv) - 1):
            curr = sorted_tv.iloc[i]
            nxt = sorted_tv.iloc[i + 1]

            interval = float(nxt["VISITDY"]) - float(curr["VISITDY"])
            if interval <= 0:
                continue

            # Window duration = TVENRL - TVSTRL (TVSTRL is negative)
            window_dur = float(curr.get("TVENRL", 0)) - float(
                curr.get("TVSTRL", 0)
            )

            if window_dur > 0.5 * interval:
                issues.append(ScheduleIssue(
                    rule_id="VS-005",
                    severity="Warning",
                    domain="TV",
                    variable="TVENRL",
                    description=(
                        f"Visit {curr['VISIT']} window ({window_dur:.0f} days) "
                        f"exceeds 50% of interval to next visit "
                        f"({interval:.0f} days)."
                    ),
                    remediation_guidance=(
                        "Window duration should be <= 50% of the interval "
                        "to the next visit to avoid ambiguous scheduling."
                    ),
                    row_number=i + 1,
                ))
        return issues

    # ------------------------------------------------------------------
    # VS-006: No duplicate VISITNUM or VISIT name
    # ------------------------------------------------------------------

    def _check_duplicate_visits(self) -> list[ScheduleIssue]:
        if self._tv.empty:
            return []

        issues: list[ScheduleIssue] = []

        # Check duplicate VISITNUM
        visitnums = list(self._tv["VISITNUM"])
        seen_nums: set[float] = set()
        for i, vn in enumerate(visitnums):
            if vn in seen_nums:
                issues.append(ScheduleIssue(
                    rule_id="VS-006",
                    severity="Error",
                    domain="TV",
                    variable="VISITNUM",
                    description=f"Duplicate VISITNUM {vn}.",
                    remediation_guidance=(
                        "Each visit must have a unique VISITNUM."
                    ),
                    row_number=i + 1,
                ))
            seen_nums.add(vn)

        # Check duplicate VISIT names
        visit_names = list(self._tv["VISIT"])
        seen_names: set[str] = set()
        for i, name in enumerate(visit_names):
            normed = str(name).strip().lower()
            if normed in seen_names:
                issues.append(ScheduleIssue(
                    rule_id="VS-006",
                    severity="Error",
                    domain="TV",
                    variable="VISIT",
                    description=f"Duplicate visit name '{name}'.",
                    remediation_guidance=(
                        "Each visit must have a unique VISIT name."
                    ),
                    row_number=i + 1,
                ))
            seen_names.add(normed)

        return issues

    # ------------------------------------------------------------------
    # VS-007: Visit frequency <= 20 per 28-day window
    # ------------------------------------------------------------------

    def _check_visit_frequency(self) -> list[ScheduleIssue]:
        if self._tv.empty or len(self._tv) <= 20:
            return []

        sorted_days = sorted(float(d) for d in self._tv["VISITDY"])

        for i, day in enumerate(sorted_days):
            window_end = day + 28
            count = sum(
                1 for d in sorted_days if day <= d <= window_end
            )
            if count > 20:
                return [ScheduleIssue(
                    rule_id="VS-007",
                    severity="Warning",
                    domain="TV",
                    variable="VISITDY",
                    description=(
                        f"{count} visits in 28-day window starting "
                        f"Day {day:.0f} (anomalous density)."
                    ),
                    remediation_guidance=(
                        "More than 20 visits in a 4-week period suggests "
                        "an extraction error or unusual protocol design."
                    ),
                )]
        return []

    # ------------------------------------------------------------------
    # VS-008: Each epoch must contain at least one visit
    # ------------------------------------------------------------------

    def _check_epoch_coverage(self) -> list[ScheduleIssue]:
        if self._te.empty or self._tv.empty:
            return []

        issues: list[ScheduleIssue] = []

        # Classify all TV visits
        visit_types: set[str] = set()
        for _, row in self._tv.iterrows():
            visit_types.add(_classify_visit(str(row.get("VISIT", ""))))

        # Check each TE epoch has at least one visit
        for _, te_row in self._te.iterrows():
            etcd = str(te_row.get("ETCD", ""))
            expected_types = _EPOCH_VISIT_TYPES.get(etcd, set())
            if not expected_types:
                continue

            if not expected_types & visit_types:
                element = str(te_row.get("ELEMENT", etcd))
                issues.append(ScheduleIssue(
                    rule_id="VS-008",
                    severity="Warning",
                    domain="TE",
                    variable="ETCD",
                    description=(
                        f"Epoch '{element}' (ETCD={etcd}) has no "
                        f"corresponding visits in TV."
                    ),
                    remediation_guidance=(
                        "Each protocol epoch should have at least one "
                        "planned visit in the TV domain."
                    ),
                ))
        return issues

    # ------------------------------------------------------------------
    # VS-009: Epoch transitions chronologically ordered
    # ------------------------------------------------------------------

    def _check_epoch_ordering(self) -> list[ScheduleIssue]:
        if self._tv.empty or len(self._tv) < 2:
            return []

        # Map epoch order
        epoch_order = {"Screening": 0, "Baseline": 1, "Treatment": 2,
                       "Follow-up": 3, "End of Study": 4,
                       "Early Termination": 5}

        sorted_tv = self._tv.sort_values("VISITDY").reset_index(drop=True)

        prev_epoch_rank = -1
        prev_type = ""
        for _, row in sorted_tv.iterrows():
            vtype = _classify_visit(str(row.get("VISIT", "")))
            rank = epoch_order.get(vtype, 2)

            # Allow same-rank and allow Treatment after Baseline
            if rank < prev_epoch_rank and not (
                vtype == "Treatment" and prev_type == "Baseline"
            ):
                return [ScheduleIssue(
                    rule_id="VS-009",
                    severity="Error",
                    domain="TV",
                    variable="VISITDY",
                    description=(
                        f"Epoch transition out of order: {vtype} "
                        f"(Day {row['VISITDY']}) appears after {prev_type}."
                    ),
                    remediation_guidance=(
                        "Epoch transitions must follow chronological "
                        "order: Screening -> Treatment -> Follow-up."
                    ),
                )]

            if rank > prev_epoch_rank:
                prev_epoch_rank = rank
                prev_type = vtype

        return []

    # ------------------------------------------------------------------
    # VS-010: All visits must have VISITDY
    # ------------------------------------------------------------------

    def _check_visitdy_completeness(self) -> list[ScheduleIssue]:
        if self._tv.empty:
            return []

        issues: list[ScheduleIssue] = []
        for i, (_, row) in enumerate(self._tv.iterrows()):
            visitdy = row.get("VISITDY")
            if visitdy is None or (
                isinstance(visitdy, float) and pd.isna(visitdy)
            ):
                issues.append(ScheduleIssue(
                    rule_id="VS-010",
                    severity="Warning",
                    domain="TV",
                    variable="VISITDY",
                    description=(
                        f"Visit '{row.get('VISIT', '?')}' has no "
                        f"planned study day (VISITDY)."
                    ),
                    remediation_guidance=(
                        "All planned visits should have a VISITDY value "
                        "for schedule validation."
                    ),
                    row_number=i + 1,
                ))
        return issues

    # ------------------------------------------------------------------
    # VS-011: Total study duration <= 10 years
    # ------------------------------------------------------------------

    def _check_study_duration(self) -> list[ScheduleIssue]:
        if self._tv.empty:
            return []

        days = [float(d) for d in self._tv["VISITDY"] if pd.notna(d)]
        if not days:
            return []

        span = max(days) - min(days)
        if span > 3650:
            return [ScheduleIssue(
                rule_id="VS-011",
                severity="Warning",
                domain="TV",
                variable="VISITDY",
                description=(
                    f"Study duration spans {span:.0f} days "
                    f"({span / 365:.1f} years), exceeding 10-year "
                    f"threshold."
                ),
                remediation_guidance=(
                    "Study duration > 10 years is likely an extraction "
                    "error. Verify day offsets in source protocol."
                ),
            )]
        return []

    # ------------------------------------------------------------------
    # VS-012: Visits with zero scheduled activities
    # ------------------------------------------------------------------

    def _check_empty_visits(self) -> list[ScheduleIssue]:
        if not self._soa_matrix or self._tv.empty:
            return []

        issues: list[ScheduleIssue] = []
        # Count activities per visitnum
        activities_per_visit: dict[int, int] = {}
        for (visitnum, _assessment) in self._soa_matrix:
            activities_per_visit[visitnum] = (
                activities_per_visit.get(visitnum, 0) + 1
            )

        for _, row in self._tv.iterrows():
            visitnum = int(row["VISITNUM"])
            if activities_per_visit.get(visitnum, 0) == 0:
                issues.append(ScheduleIssue(
                    rule_id="VS-012",
                    severity="Warning",
                    domain="TV",
                    variable="VISITNUM",
                    description=(
                        f"Visit '{row.get('VISIT', '?')}' (VISITNUM "
                        f"{visitnum}) has zero scheduled activities."
                    ),
                    remediation_guidance=(
                        "A visit with no scheduled activities is likely "
                        "incomplete. Verify SoA table extraction."
                    ),
                    row_number=visitnum,
                ))
        return issues
