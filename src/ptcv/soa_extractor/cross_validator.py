"""SoA cross-validation framework (PTCV-216).

Checks completeness and column alignment of extracted SoA tables
against a reference assessment list. When coverage is insufficient,
recommends escalation to the next cascade level.

Until the FAISS knowledge base (PTCV-219) is built, uses static
reference lists for common protocol types.

Risk tier: LOW — read-only analysis, no data mutation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .models import RawSoaTable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static reference assessment lists (interim until FAISS, PTCV-219)
# ---------------------------------------------------------------------------

ONCOLOGY_PHASE12_ASSESSMENTS: list[str] = [
    "Physical Exam",
    "Vital Signs",
    "ECG",
    "Hematology",
    "Chemistry",
    "Urinalysis",
    "Coagulation",
    "Tumor Assessment",
    "Informed Consent",
    "Medical History",
    "Concomitant Medications",
    "Adverse Events",
    "ECOG Performance",
]

GENERAL_ASSESSMENTS: list[str] = [
    "Physical Exam",
    "Vital Signs",
    "ECG",
    "Hematology",
    "Chemistry",
    "Informed Consent",
    "Adverse Events",
    "Concomitant Medications",
]

_REFERENCE_LISTS: dict[str, list[str]] = {
    "oncology_phase12": ONCOLOGY_PHASE12_ASSESSMENTS,
    "general": GENERAL_ASSESSMENTS,
}

# Minimum coverage ratio before escalation is recommended
DEFAULT_MIN_COVERAGE = 0.50


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class AssessmentMatch:
    """Match result for a single expected assessment.

    Attributes:
        expected: Expected assessment name from reference list.
        found: Whether a matching row was found in extracted table.
        matched_name: Actual name in the extracted table (if found).
    """

    expected: str
    found: bool = False
    matched_name: str = ""


@dataclass
class CompletenessReport:
    """Result of assessment completeness check.

    Attributes:
        reference_type: Which reference list was used.
        expected_count: Number of assessments in the reference list.
        found_count: Number found in extracted table.
        coverage_ratio: found_count / expected_count.
        matches: Per-assessment match details.
        missing: Names of expected assessments not found.
        extra: Activity names in extracted table not in reference.
        needs_escalation: Whether coverage is below threshold.
    """

    reference_type: str = "general"
    expected_count: int = 0
    found_count: int = 0
    coverage_ratio: float = 0.0
    matches: list[AssessmentMatch] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    needs_escalation: bool = False


@dataclass
class AlignmentIssue:
    """A single column alignment issue in an extracted row.

    Attributes:
        row_index: 0-based index of the misaligned row.
        activity_name: Name of the assessment/activity.
        expected_cells: Number of cells expected (header count).
        actual_cells: Number of cells found.
    """

    row_index: int
    activity_name: str
    expected_cells: int
    actual_cells: int


@dataclass
class AlignmentReport:
    """Result of column alignment validation.

    Attributes:
        header_count: Number of visit columns.
        total_rows: Total activity rows checked.
        misaligned_count: Number of misaligned rows.
        issues: Per-row alignment issues.
        misalignment_ratio: Fraction of rows with wrong cell count.
    """

    header_count: int = 0
    total_rows: int = 0
    misaligned_count: int = 0
    issues: list[AlignmentIssue] = field(default_factory=list)
    misalignment_ratio: float = 0.0


@dataclass
class CrossValidationResult:
    """Combined cross-validation result.

    Attributes:
        completeness: Assessment completeness report.
        alignment: Column alignment report.
        needs_escalation: Whether any check recommends escalation.
        escalation_reasons: Why escalation was recommended.
    """

    completeness: Optional[CompletenessReport] = None
    alignment: Optional[AlignmentReport] = None
    needs_escalation: bool = False
    escalation_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Normalize an assessment name for fuzzy matching.

    Strips whitespace, lowercases, removes common suffixes/prefixes,
    and collapses whitespace.

    Args:
        name: Raw assessment name.

    Returns:
        Normalized string for comparison.
    """
    n = name.strip().lower()
    # Remove footnote markers (e.g., "^2", "¹")
    n = re.sub(r"[\^¹²³⁴⁵⁶⁷⁸⁹⁰]+\d*", "", n)
    # Collapse whitespace
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _fuzzy_match(expected: str, actual: str) -> bool:
    """Check if an expected assessment name matches an extracted one.

    Uses substring matching and common abbreviation handling.

    Args:
        expected: Expected name from reference list.
        actual: Extracted activity name from table.

    Returns:
        True if the names are considered a match.
    """
    e = _normalize_name(expected)
    a = _normalize_name(actual)

    if not e or not a:
        return False

    # Exact match
    if e == a:
        return True

    # Substring match (either direction)
    if e in a or a in e:
        return True

    # Common abbreviations
    abbrevs = {
        "ecg": "electrocardiogram",
        "ecog": "ecog performance",
        "vitals": "vital signs",
        "pe": "physical exam",
        "conmeds": "concomitant medications",
        "aes": "adverse events",
        "labs": "laboratory",
        "heme": "hematology",
        "chem": "chemistry",
        "coags": "coagulation",
    }
    for short, long in abbrevs.items():
        if (short in e and long in a) or (long in e and short in a):
            return True

    return False


def check_completeness(
    table: RawSoaTable,
    reference_type: str = "general",
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> CompletenessReport:
    """Check assessment row completeness against a reference list.

    Compares extracted activity names against a static reference
    list for the given protocol type. Reports coverage ratio and
    lists missing/extra assessments.

    Args:
        table: Extracted SoA table to validate.
        reference_type: Key into _REFERENCE_LISTS (e.g.,
            "oncology_phase12", "general").
        min_coverage: Minimum coverage ratio before escalation
            is recommended.

    Returns:
        CompletenessReport with match details.
    """
    reference = _REFERENCE_LISTS.get(reference_type, GENERAL_ASSESSMENTS)
    extracted_names = [name for name, _ in table.activities]

    matches: list[AssessmentMatch] = []
    found_count = 0
    matched_extracted: set[int] = set()

    for expected in reference:
        match = AssessmentMatch(expected=expected)
        for i, actual in enumerate(extracted_names):
            if i in matched_extracted:
                continue
            if _fuzzy_match(expected, actual):
                match.found = True
                match.matched_name = actual
                matched_extracted.add(i)
                found_count += 1
                break
        matches.append(match)

    missing = [m.expected for m in matches if not m.found]
    extra = [
        name for i, name in enumerate(extracted_names)
        if i not in matched_extracted
    ]

    coverage = found_count / len(reference) if reference else 0.0

    return CompletenessReport(
        reference_type=reference_type,
        expected_count=len(reference),
        found_count=found_count,
        coverage_ratio=coverage,
        matches=matches,
        missing=missing,
        extra=extra,
        needs_escalation=coverage < min_coverage,
    )


def check_alignment(table: RawSoaTable) -> AlignmentReport:
    """Check column alignment for all rows in a table.

    Verifies that each activity row has the same number of
    scheduled flags as there are visit columns.

    Args:
        table: Extracted SoA table to validate.

    Returns:
        AlignmentReport with per-row issues.
    """
    header_count = len(table.visit_headers)
    issues: list[AlignmentIssue] = []

    for i, (name, flags) in enumerate(table.activities):
        if len(flags) != header_count:
            issues.append(AlignmentIssue(
                row_index=i,
                activity_name=name,
                expected_cells=header_count,
                actual_cells=len(flags),
            ))

    total = len(table.activities)
    ratio = len(issues) / total if total > 0 else 0.0

    return AlignmentReport(
        header_count=header_count,
        total_rows=total,
        misaligned_count=len(issues),
        issues=issues,
        misalignment_ratio=ratio,
    )


def cross_validate(
    table: RawSoaTable,
    reference_type: str = "general",
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_misalignment: float = 0.2,
) -> CrossValidationResult:
    """Run all cross-validation checks on a table.

    Combines completeness and alignment checks into a single
    result with a unified escalation recommendation.

    Args:
        table: Extracted SoA table to validate.
        reference_type: Reference list key for completeness check.
        min_coverage: Minimum coverage ratio.
        max_misalignment: Maximum misalignment ratio.

    Returns:
        CrossValidationResult with combined analysis.
    """
    completeness = check_completeness(
        table, reference_type, min_coverage,
    )
    alignment = check_alignment(table)

    reasons: list[str] = []
    needs_escalation = False

    if completeness.needs_escalation:
        needs_escalation = True
        reasons.append(
            f"Low assessment coverage: {completeness.coverage_ratio:.0%} "
            f"({completeness.found_count}/{completeness.expected_count}). "
            f"Missing: {', '.join(completeness.missing[:5])}"
        )

    if alignment.misalignment_ratio > max_misalignment:
        needs_escalation = True
        reasons.append(
            f"Column misalignment: {alignment.misalignment_ratio:.0%} "
            f"({alignment.misaligned_count}/{alignment.total_rows} rows)"
        )

    return CrossValidationResult(
        completeness=completeness,
        alignment=alignment,
        needs_escalation=needs_escalation,
        escalation_reasons=reasons,
    )


__all__ = [
    "AssessmentMatch",
    "AlignmentIssue",
    "AlignmentReport",
    "CompletenessReport",
    "CrossValidationResult",
    "GENERAL_ASSESSMENTS",
    "ONCOLOGY_PHASE12_ASSESSMENTS",
    "check_alignment",
    "check_completeness",
    "cross_validate",
]
