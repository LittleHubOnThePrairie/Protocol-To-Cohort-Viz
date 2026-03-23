"""Tests for PubMed abstract vector builder (PTCV-286).

Feature: RAG index with anchored PubMed embeddings

  Scenario: PubMed chunks use section-anchored embedding
  Scenario: Mixed METHODS sections sub-classified
  Scenario: Structured abstracts chunked by section headers
  Scenario: Unstructured abstracts chunked by sentences
  Scenario: Deduplication prevents double-indexing
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ptcv.registry.pubmed_vector_builder import (
    AnchoredChunk,
    build_chunks_from_cache,
    seed_pubmed_vectors,
    _chunk_abstract,
    _classify_by_content,
    _map_header_to_ich,
    _split_if_needed,
    _subclassify_methods,
    _subclassify_results,
)


# -----------------------------------------------------------------------
# Test data
# -----------------------------------------------------------------------

STRUCTURED_ABSTRACT = (
    "BACKGROUND: Drug X inhibits kinase Y, which is overexpressed "
    "in solid tumors. Prior studies showed preclinical efficacy.\n"
    "OBJECTIVE: To evaluate the efficacy and safety of Drug X "
    "in patients with advanced solid tumors.\n"
    "METHODS: This was a randomized, double-blind, "
    "placebo-controlled phase 3 study. Patients received 200mg "
    "daily or placebo in 28-day cycles.\n"
    "RESULTS: Overall response rate was 42%. Grade 3 adverse "
    "events included neutropenia (23%) and fatigue (11%). "
    "Median PFS was 8.3 months (HR 0.72, 95% CI 0.58-0.89).\n"
    "CONCLUSIONS: Drug X demonstrated significant clinical "
    "benefit with a manageable safety profile."
)

UNSTRUCTURED_ABSTRACT = (
    "We conducted a randomized double-blind study of Drug X "
    "versus placebo in 500 patients with solid tumors. "
    "The primary endpoint was progression-free survival. "
    "Median PFS was 8.3 months vs 4.2 months (HR 0.72). "
    "Grade 3 adverse events included neutropenia in 23%. "
    "Drug X showed significant clinical benefit."
)

SAFETY_ONLY_RESULTS = (
    "Grade 3 or higher adverse events were reported in 45% of "
    "patients. The most common were neutropenia (23%), "
    "thrombocytopenia (15%), and hepatotoxicity (8%). "
    "Treatment discontinuation due to adverse events occurred "
    "in 12% of patients."
)


def _make_cache(tmp: str, articles: list[dict]) -> Path:
    """Write a PubMed cache file and return the directory."""
    cache_dir = Path(tmp) / "pubmed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "nct_id": "NCT12345678",
        "articles": articles,
        "pmid_count": len(articles),
        "fetched_utc": "2026-01-01T00:00:00Z",
    }
    (cache_dir / "NCT12345678_pubmed.json").write_text(
        json.dumps(data), encoding="utf-8",
    )
    return cache_dir


# -----------------------------------------------------------------------
# Chunking tests
# -----------------------------------------------------------------------


class TestChunkStructuredAbstract:
    """Structured abstracts split by section headers."""

    def test_produces_chunks_per_section(self) -> None:
        chunks = _chunk_abstract(
            STRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        assert len(chunks) >= 4  # BACKGROUND, OBJECTIVE, METHODS, RESULTS+

    def test_chunks_have_ich_anchors(self) -> None:
        chunks = _chunk_abstract(
            STRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        for chunk in chunks:
            assert chunk.anchored_text.startswith("B.")
            assert chunk.ich_code in chunk.anchored_text
            assert chunk.ich_name in chunk.anchored_text

    def test_background_maps_to_b2(self) -> None:
        chunks = _chunk_abstract(
            STRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        bg = [c for c in chunks if c.abstract_section == "BACKGROUND"]
        assert len(bg) == 1
        assert bg[0].ich_code == "B.2"

    def test_objective_maps_to_b3(self) -> None:
        chunks = _chunk_abstract(
            STRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        obj = [c for c in chunks if c.abstract_section == "OBJECTIVE"]
        assert len(obj) == 1
        assert obj[0].ich_code == "B.3"

    def test_conclusions_maps_to_b3(self) -> None:
        chunks = _chunk_abstract(
            STRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        conc = [c for c in chunks if c.abstract_section == "CONCLUSIONS"]
        assert len(conc) == 1
        assert conc[0].ich_code == "B.3"


class TestChunkUnstructured:
    """Unstructured abstracts grouped by sentences."""

    def test_produces_chunks(self) -> None:
        chunks = _chunk_abstract(
            UNSTRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        assert len(chunks) >= 1

    def test_chunks_classified_by_content(self) -> None:
        chunks = _chunk_abstract(
            UNSTRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        for chunk in chunks:
            assert chunk.ich_code.startswith("B.")
            assert chunk.abstract_section == "UNSTRUCTURED"


# -----------------------------------------------------------------------
# Sub-classification tests
# -----------------------------------------------------------------------


class TestSubclassifyMethods:
    """METHODS sections split into B.4 vs B.7."""

    def test_design_content_maps_to_b4(self) -> None:
        result = _subclassify_methods(
            "This was a randomized double-blind phase 3 study.",
        )
        assert result[0][0] == "B.4"

    def test_dosing_content_maps_to_b7(self) -> None:
        result = _subclassify_methods(
            "Patients received 200mg daily administered orally "
            "in 28-day treatment cycles.",
        )
        assert result[0][0] == "B.7"

    def test_mixed_defaults_to_b4(self) -> None:
        result = _subclassify_methods(
            "This randomized study used 200mg daily dosing.",
        )
        assert result[0][0] == "B.4"


class TestSubclassifyResults:
    """RESULTS sections split into B.8 vs B.9."""

    def test_efficacy_content_maps_to_b8(self) -> None:
        result = _subclassify_results(
            "Overall response rate was 42%. Median PFS 8.3 months.",
        )
        assert result[0][0] == "B.8"

    def test_safety_content_maps_to_b9(self) -> None:
        result = _subclassify_results(SAFETY_ONLY_RESULTS)
        assert result[0][0] == "B.9"

    def test_mixed_with_dominant_efficacy_maps_to_b8(self) -> None:
        result = _subclassify_results(
            "Overall response rate was 42%. Median PFS 8.3 months. "
            "Grade 3 adverse events in 23%.",
        )
        assert result[0][0] == "B.8"


# -----------------------------------------------------------------------
# Anchoring tests
# -----------------------------------------------------------------------


class TestAnchoring:
    """Section-anchored embedding text."""

    def test_anchor_prefix_format(self) -> None:
        chunks = _chunk_abstract(
            STRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        for chunk in chunks:
            # Format: "B.X Section Name: content..."
            assert chunk.anchored_text.startswith(chunk.ich_code)
            assert ": " in chunk.anchored_text

    def test_raw_text_does_not_have_prefix(self) -> None:
        chunks = _chunk_abstract(
            STRUCTURED_ABSTRACT, "NCT123", "99999",
        )
        for chunk in chunks:
            assert not chunk.raw_text.startswith("B.")


# -----------------------------------------------------------------------
# Cache reading tests
# -----------------------------------------------------------------------


class TestBuildFromCache:
    """Reading cached PubMed files."""

    def test_reads_cache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = _make_cache(tmp, [
                {"pmid": "99999", "abstract": STRUCTURED_ABSTRACT},
            ])
            chunks = build_chunks_from_cache(cache_dir)
            assert len(chunks) >= 4
            assert all(c.nct_id == "NCT12345678" for c in chunks)
            assert all(c.pmid == "99999" for c in chunks)

    def test_skips_empty_abstracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = _make_cache(tmp, [
                {"pmid": "11111", "abstract": ""},
                {"pmid": "22222", "abstract": STRUCTURED_ABSTRACT},
            ])
            chunks = build_chunks_from_cache(cache_dir)
            assert all(c.pmid == "22222" for c in chunks)

    def test_missing_directory_returns_empty(self) -> None:
        chunks = build_chunks_from_cache("/nonexistent/path")
        assert chunks == []


# -----------------------------------------------------------------------
# Seeding tests
# -----------------------------------------------------------------------


class TestSeedPubmedVectors:
    """Adding chunks to RAG index."""

    def test_adds_vectors_to_index(self) -> None:
        mock_index = MagicMock()
        mock_index._index = MagicMock()
        mock_index._metadata = []
        mock_index._encode.return_value = MagicMock()

        chunks = [
            AnchoredChunk(
                anchored_text="B.4 Trial Design: randomized study",
                raw_text="randomized study",
                ich_code="B.4",
                ich_name="Trial Design",
                abstract_section="DESIGN",
                nct_id="NCT123",
                pmid="99999",
            ),
        ]

        count = seed_pubmed_vectors(mock_index, chunks)

        assert count == 1
        assert len(mock_index._metadata) == 1
        meta = mock_index._metadata[0]
        assert meta["source"] == "pubmed"
        assert meta["section_code"] == "B.4"
        assert meta["pmid"] == "99999"

    def test_deduplicates_existing(self) -> None:
        mock_index = MagicMock()
        mock_index._index = MagicMock()
        mock_index._metadata = [
            {
                "source": "pubmed",
                "nct_id": "NCT123",
                "pmid": "99999",
                "section_code": "B.4",
            },
        ]
        mock_index._encode.return_value = MagicMock()

        chunks = [
            AnchoredChunk(
                anchored_text="B.4 Trial Design: randomized study",
                raw_text="randomized study",
                ich_code="B.4",
                ich_name="Trial Design",
                abstract_section="DESIGN",
                nct_id="NCT123",
                pmid="99999",
            ),
        ]

        count = seed_pubmed_vectors(mock_index, chunks)
        assert count == 0  # Already indexed

    def test_empty_chunks_returns_zero(self) -> None:
        mock_index = MagicMock()
        count = seed_pubmed_vectors(mock_index, [])
        assert count == 0


# -----------------------------------------------------------------------
# Utility tests
# -----------------------------------------------------------------------


class TestSplitIfNeeded:
    """Chunk splitting for oversized content."""

    def test_short_text_not_split(self) -> None:
        result = _split_if_needed("Short text here.")
        assert len(result) == 1

    def test_long_text_split_by_sentences(self) -> None:
        long_text = ". ".join(
            [f"Sentence {i} with enough content" for i in range(50)]
        ) + "."
        result = _split_if_needed(long_text)
        assert len(result) > 1
        assert all(len(chunk) <= 1300 for chunk in result)


class TestClassifyByContent:
    """Content-based classification for unstructured text."""

    def test_design_keywords(self) -> None:
        code = _classify_by_content(
            "randomized double-blind phase 3 study",
        )
        assert code == "B.4"

    def test_safety_keywords(self) -> None:
        code = _classify_by_content(
            "adverse events grade 3 neutropenia safety profile",
        )
        assert code == "B.9"

    def test_efficacy_keywords(self) -> None:
        code = _classify_by_content(
            "overall response rate progression-free survival",
        )
        assert code == "B.8"
