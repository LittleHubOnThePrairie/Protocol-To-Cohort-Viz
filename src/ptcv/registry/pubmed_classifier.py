"""PubMed Publication Relevance Classifier (PTCV-282).

Classifies PubMed articles as PRIMARY, SECONDARY, or EXCLUDED
based on publication type, accession number count, and title signals.

PRIMARY articles are *about* a specific trial (design papers, results).
SECONDARY articles *cite* the trial among many (reviews, meta-analyses).
EXCLUDED articles are tangentially related (editorials, errata).

Usage::

    from ptcv.registry.pubmed_classifier import (
        classify_publication,
        PublicationRelevance,
    )

    relevance = classify_publication(article)
    if relevance == PublicationRelevance.PRIMARY:
        # Embed full abstract
    elif relevance == PublicationRelevance.SECONDARY:
        # Embed title + MeSH only
"""

from __future__ import annotations

import enum
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pubmed_adapter import PubmedArticle


class PublicationRelevance(enum.Enum):
    """Classification of a PubMed article's relevance to a trial."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXCLUDED = "excluded"


# Publication types that indicate the article IS about a trial
_PRIMARY_PUB_TYPES = frozenset({
    "Clinical Trial",
    "Clinical Trial, Phase I",
    "Clinical Trial, Phase II",
    "Clinical Trial, Phase III",
    "Clinical Trial, Phase IV",
    "Clinical Trial Protocol",
    "Randomized Controlled Trial",
    "Controlled Clinical Trial",
    "Pragmatic Clinical Trial",
    "Adaptive Clinical Trial",
    "Equivalence Trial",
    "Multicenter Study",
    "Observational Study",
    "Comparative Study",
})

# Publication types that indicate a review/aggregation
_REVIEW_PUB_TYPES = frozenset({
    "Review",
    "Systematic Review",
    "Meta-Analysis",
    "Pooled Analysis",
    "Guideline",
    "Practice Guideline",
    "Consensus Development Conference",
})

# Publication types to exclude entirely
_EXCLUDED_PUB_TYPES = frozenset({
    "Editorial",
    "Comment",
    "Letter",
    "Published Erratum",
    "News",
    "Newspaper Article",
    "Retraction of Publication",
    "Retracted Publication",
    "Expression of Concern",
})

# Title patterns suggesting a review/meta-analysis
_REVIEW_TITLE_RE = re.compile(
    r"\b(?:systematic\s+review|meta-analysis|pooled\s+analysis"
    r"|network\s+meta|umbrella\s+review)\b",
    re.IGNORECASE,
)

# Max accession numbers for a "focused" review to be treated as PRIMARY
_FOCUSED_REVIEW_MAX_ACCESSIONS = 5


def classify_publication(
    article: "PubmedArticle",
) -> PublicationRelevance:
    """Classify a PubMed article's relevance to its linked trial.

    Classification logic:
    1. If any publication type is in EXCLUDED set → EXCLUDED
    2. If any publication type is in PRIMARY set → PRIMARY
    3. If any publication type is in REVIEW set:
       a. If accession_numbers ≤ 5 → PRIMARY (focused review)
       b. If accession_numbers > 5 → SECONDARY
    4. If title matches review patterns → SECONDARY
    5. Default → PRIMARY (assume article about the trial)

    Args:
        article: PubmedArticle with publication_types and
            accession_numbers populated.

    Returns:
        PublicationRelevance classification.
    """
    pub_types = set(article.publication_types)

    # Step 1: Check for excluded types
    if pub_types & _EXCLUDED_PUB_TYPES:
        return PublicationRelevance.EXCLUDED

    # Step 2: Check for primary types (strongest signal)
    if pub_types & _PRIMARY_PUB_TYPES:
        return PublicationRelevance.PRIMARY

    # Step 3: Check for review types
    if pub_types & _REVIEW_PUB_TYPES:
        accession_count = len(article.accession_numbers)
        if accession_count <= _FOCUSED_REVIEW_MAX_ACCESSIONS:
            return PublicationRelevance.PRIMARY
        return PublicationRelevance.SECONDARY

    # Step 4: Title-based review detection
    if _REVIEW_TITLE_RE.search(article.title):
        return PublicationRelevance.SECONDARY

    # Step 5: Default — assume primary
    return PublicationRelevance.PRIMARY


def get_embedding_text(
    article: "PubmedArticle",
    relevance: PublicationRelevance,
) -> str:
    """Extract text for embedding based on relevance classification.

    PRIMARY: full abstract + title + MeSH terms
    SECONDARY: title + MeSH terms only (no abstract)
    EXCLUDED: empty string (no embedding)

    Args:
        article: PubmedArticle to extract text from.
        relevance: Classification result.

    Returns:
        Text string for embedding generation. Empty if EXCLUDED.
    """
    if relevance == PublicationRelevance.EXCLUDED:
        return ""

    parts: list[str] = [article.title]

    if relevance == PublicationRelevance.PRIMARY and article.abstract:
        parts.append(article.abstract)

    if article.mesh_terms:
        parts.append("MeSH: " + ", ".join(article.mesh_terms))

    return "\n".join(parts)


__all__ = [
    "PublicationRelevance",
    "classify_publication",
    "get_embedding_text",
]
