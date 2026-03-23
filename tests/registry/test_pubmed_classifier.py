"""Tests for PubMed publication relevance classifier (PTCV-282).

Tests classification of PubMed articles as PRIMARY, SECONDARY,
or EXCLUDED based on publication type, accession count, and title.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.registry.pubmed_adapter import PubmedArticle
from ptcv.registry.pubmed_classifier import (
    PublicationRelevance,
    classify_publication,
    get_embedding_text,
)


def _make_article(
    publication_types: list[str] | None = None,
    accession_numbers: list[str] | None = None,
    title: str = "Test Article",
    abstract: str = "Test abstract text.",
    mesh_terms: list[str] | None = None,
) -> PubmedArticle:
    return PubmedArticle(
        pmid="12345",
        title=title,
        abstract=abstract,
        publication_types=publication_types or [],
        accession_numbers=accession_numbers or [],
        mesh_terms=mesh_terms or [],
    )


class TestClassifyPrimary:
    """GHERKIN: Primary publication identified."""

    def test_clinical_trial_pub_type(self):
        article = _make_article(
            publication_types=["Clinical Trial"],
        )
        assert classify_publication(article) == PublicationRelevance.PRIMARY

    def test_rct_pub_type(self):
        article = _make_article(
            publication_types=["Randomized Controlled Trial"],
        )
        assert classify_publication(article) == PublicationRelevance.PRIMARY

    def test_phase_iii_pub_type(self):
        article = _make_article(
            publication_types=["Clinical Trial, Phase III"],
        )
        assert classify_publication(article) == PublicationRelevance.PRIMARY

    def test_protocol_publication(self):
        article = _make_article(
            publication_types=["Clinical Trial Protocol"],
        )
        assert classify_publication(article) == PublicationRelevance.PRIMARY

    def test_default_is_primary(self):
        """No publication type → assume primary."""
        article = _make_article(publication_types=[])
        assert classify_publication(article) == PublicationRelevance.PRIMARY

    def test_journal_article_only_is_primary(self):
        article = _make_article(
            publication_types=["Journal Article"],
        )
        assert classify_publication(article) == PublicationRelevance.PRIMARY


class TestClassifySecondary:
    """GHERKIN: Systematic review scoped to metadata only."""

    def test_review_with_many_accessions(self):
        article = _make_article(
            publication_types=["Systematic Review"],
            accession_numbers=[f"NCT{i:08d}" for i in range(25)],
        )
        assert classify_publication(article) == PublicationRelevance.SECONDARY

    def test_meta_analysis_with_many_accessions(self):
        article = _make_article(
            publication_types=["Meta-Analysis"],
            accession_numbers=[f"NCT{i:08d}" for i in range(10)],
        )
        assert classify_publication(article) == PublicationRelevance.SECONDARY

    def test_title_systematic_review(self):
        article = _make_article(
            title="A systematic review of treatment efficacy",
        )
        assert classify_publication(article) == PublicationRelevance.SECONDARY

    def test_title_meta_analysis(self):
        article = _make_article(
            title="Meta-analysis of phase III trials in NSCLC",
        )
        assert classify_publication(article) == PublicationRelevance.SECONDARY


class TestClassifyFocusedReview:
    """GHERKIN: Focused review treated as primary."""

    def test_review_with_few_accessions_is_primary(self):
        article = _make_article(
            publication_types=["Review"],
            accession_numbers=["NCT00112827", "NCT00112828", "NCT00112829"],
        )
        assert classify_publication(article) == PublicationRelevance.PRIMARY

    def test_review_with_5_accessions_is_primary(self):
        article = _make_article(
            publication_types=["Review"],
            accession_numbers=[f"NCT{i:08d}" for i in range(5)],
        )
        assert classify_publication(article) == PublicationRelevance.PRIMARY

    def test_review_with_6_accessions_is_secondary(self):
        article = _make_article(
            publication_types=["Review"],
            accession_numbers=[f"NCT{i:08d}" for i in range(6)],
        )
        assert classify_publication(article) == PublicationRelevance.SECONDARY


class TestClassifyExcluded:
    """GHERKIN: Editorial excluded from corpus."""

    def test_editorial(self):
        article = _make_article(publication_types=["Editorial"])
        assert classify_publication(article) == PublicationRelevance.EXCLUDED

    def test_comment(self):
        article = _make_article(publication_types=["Comment"])
        assert classify_publication(article) == PublicationRelevance.EXCLUDED

    def test_letter(self):
        article = _make_article(publication_types=["Letter"])
        assert classify_publication(article) == PublicationRelevance.EXCLUDED

    def test_erratum(self):
        article = _make_article(
            publication_types=["Published Erratum"],
        )
        assert classify_publication(article) == PublicationRelevance.EXCLUDED

    def test_excluded_takes_priority(self):
        """Even if also tagged as Clinical Trial, editorial wins."""
        article = _make_article(
            publication_types=["Editorial", "Clinical Trial"],
        )
        assert classify_publication(article) == PublicationRelevance.EXCLUDED


class TestGetEmbeddingText:
    def test_primary_includes_abstract(self):
        article = _make_article(
            title="Trial Results",
            abstract="The trial showed efficacy.",
            mesh_terms=["Neoplasms", "Drug Therapy"],
        )
        text = get_embedding_text(article, PublicationRelevance.PRIMARY)
        assert "Trial Results" in text
        assert "The trial showed efficacy." in text
        assert "MeSH:" in text
        assert "Neoplasms" in text

    def test_secondary_excludes_abstract(self):
        article = _make_article(
            title="A Review of Trials",
            abstract="This review covers 50 trials...",
            mesh_terms=["Oncology"],
        )
        text = get_embedding_text(article, PublicationRelevance.SECONDARY)
        assert "A Review of Trials" in text
        assert "MeSH:" in text
        assert "This review covers 50 trials" not in text

    def test_excluded_returns_empty(self):
        article = _make_article(title="Editorial Comment")
        text = get_embedding_text(article, PublicationRelevance.EXCLUDED)
        assert text == ""

    def test_primary_no_abstract(self):
        article = _make_article(title="Protocol Paper", abstract="")
        text = get_embedding_text(article, PublicationRelevance.PRIMARY)
        assert "Protocol Paper" in text
        assert len(text) > 0
