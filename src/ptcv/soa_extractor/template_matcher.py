"""Template matching for SoA completeness checking (PTCV-219).

Queries the FAISS knowledge base for similar verified protocols
and compares extracted assessment lists to detect missing items.

Risk tier: LOW — read-only analysis, no data mutation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .knowledge_base import SoaKnowledgeBase, SimilarProtocol, VerifiedSoaEntry
from .models import RawSoaTable

logger = logging.getLogger(__name__)


@dataclass
class CompletenessMatch:
    """Completeness comparison against a single template.

    Attributes:
        template_id: Registry ID of the template protocol.
        template_similarity: Cosine similarity score.
        expected_assessments: Assessment names in the template.
        found_assessments: Template assessments found in extracted.
        missing_assessments: Template assessments NOT in extracted.
        coverage_ratio: found / expected.
    """

    template_id: str
    template_similarity: float
    expected_assessments: list[str] = field(default_factory=list)
    found_assessments: list[str] = field(default_factory=list)
    missing_assessments: list[str] = field(default_factory=list)
    coverage_ratio: float = 0.0


@dataclass
class TemplateMatchReport:
    """Result of template matching against the knowledge base.

    Attributes:
        templates_checked: Number of templates compared.
        best_match: The most similar template comparison.
        all_matches: All template comparisons.
        consensus_missing: Assessments missing across multiple
            templates (high confidence they're truly missing).
        consensus_threshold: Minimum fraction of templates an
            assessment must be missing from to appear in consensus.
        has_template: Whether any template was found.
    """

    templates_checked: int = 0
    best_match: Optional[CompletenessMatch] = None
    all_matches: list[CompletenessMatch] = field(default_factory=list)
    consensus_missing: list[str] = field(default_factory=list)
    consensus_threshold: float = 0.5
    has_template: bool = False

    @property
    def best_coverage(self) -> float:
        """Coverage ratio from the best match."""
        return self.best_match.coverage_ratio if self.best_match else 0.0


def _normalize_for_match(name: str) -> str:
    """Normalize assessment name for fuzzy matching."""
    import re
    n = name.strip().lower()
    n = re.sub(r"[\^¹²³⁴⁵⁶⁷⁸⁹⁰]+\d*", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _fuzzy_contains(
    extracted_names: list[str],
    template_name: str,
) -> bool:
    """Check if a template assessment name is in extracted list."""
    t = _normalize_for_match(template_name)
    for name in extracted_names:
        e = _normalize_for_match(name)
        if t == e or t in e or e in t:
            return True
    return False


def match_against_template(
    table: RawSoaTable,
    template: VerifiedSoaEntry,
) -> CompletenessMatch:
    """Compare an extracted table against a single verified template.

    Args:
        table: Extracted SoA table to check.
        template: Verified template to compare against.

    Returns:
        CompletenessMatch with coverage details.
    """
    extracted_names = [name for name, _ in table.activities]

    found: list[str] = []
    missing: list[str] = []

    for template_name in template.assessment_names:
        if _fuzzy_contains(extracted_names, template_name):
            found.append(template_name)
        else:
            missing.append(template_name)

    expected = len(template.assessment_names)
    ratio = len(found) / expected if expected > 0 else 0.0

    return CompletenessMatch(
        template_id=template.registry_id,
        template_similarity=0.0,  # Set by caller
        expected_assessments=template.assessment_names,
        found_assessments=found,
        missing_assessments=missing,
        coverage_ratio=ratio,
    )


def match_completeness(
    table: RawSoaTable,
    kb: SoaKnowledgeBase,
    top_k: int = 3,
    consensus_threshold: float = 0.5,
    phase: str = "",
    condition: str = "",
) -> TemplateMatchReport:
    """Check assessment completeness against knowledge base templates.

    Queries the FAISS index for the K most similar verified protocols
    and compares the extracted assessment list against each template.

    Args:
        table: Extracted SoA table to check.
        kb: SoA knowledge base to query.
        top_k: Number of templates to compare against.
        consensus_threshold: Minimum fraction of templates an
            assessment must be missing from to appear in
            consensus_missing list.
        phase: Trial phase for query context.
        condition: Disease condition for query context.

    Returns:
        TemplateMatchReport with completeness analysis.
    """
    if kb.size == 0:
        return TemplateMatchReport(has_template=False)

    # Build query text from the extracted table
    extracted_names = [name for name, _ in table.activities]
    query_parts = []
    if phase:
        query_parts.append(f"Phase: {phase}")
    if condition:
        query_parts.append(f"Condition: {condition}")
    query_parts.append(
        "Assessments: " + ", ".join(extracted_names[:30])
    )
    query_text = ". ".join(query_parts)

    similar = kb.query(query_text, top_k=top_k)

    if not similar:
        return TemplateMatchReport(has_template=False)

    matches: list[CompletenessMatch] = []
    missing_counts: dict[str, int] = {}

    for sp in similar:
        cm = match_against_template(table, sp.entry)
        cm.template_similarity = sp.similarity
        matches.append(cm)

        for name in cm.missing_assessments:
            norm = _normalize_for_match(name)
            missing_counts[norm] = missing_counts.get(norm, 0) + 1

    # Consensus missing: assessments missing from >= threshold of templates
    n_templates = len(matches)
    consensus: list[str] = []
    if n_templates > 0:
        for name, count in sorted(
            missing_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if count / n_templates >= consensus_threshold:
                consensus.append(name)

    best = min(matches, key=lambda m: -m.coverage_ratio)

    return TemplateMatchReport(
        templates_checked=len(matches),
        best_match=best,
        all_matches=matches,
        consensus_missing=consensus,
        consensus_threshold=consensus_threshold,
        has_template=True,
    )


__all__ = [
    "CompletenessMatch",
    "TemplateMatchReport",
    "match_against_template",
    "match_completeness",
]
