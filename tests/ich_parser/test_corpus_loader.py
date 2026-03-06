"""Unit tests for corpus_loader (PTCV-46).

Tests the gold-standard ICH section corpus loading, exemplar
retrieval, and coverage validation against GHERKIN acceptance
criteria.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ptcv.ich_parser.corpus_loader import (
    CorpusExemplar,
    corpus_stats,
    get_exemplars,
    load_corpus,
)

# All 11 ICH section codes
_ALL_CODES = [
    "B.1", "B.2", "B.3", "B.4", "B.5", "B.6",
    "B.7", "B.8", "B.9", "B.10", "B.11",
]


class TestLoadCorpus:
    """Tests for load_corpus function."""

    def test_loads_default_corpus(self) -> None:
        corpus = load_corpus()
        assert len(corpus) > 0

    def test_all_sections_covered(self) -> None:
        """Scenario: Corpus covers all ICH sections."""
        corpus = load_corpus()
        for code in _ALL_CODES:
            assert code in corpus, f"Missing section {code}"
            assert len(corpus[code]) >= 2, (
                f"{code} has {len(corpus[code])} exemplars, need >=2"
            )

    def test_exemplars_have_required_fields(self) -> None:
        """Scenario: Corpus loads correctly."""
        corpus = load_corpus()
        for code, exemplars in corpus.items():
            for ex in exemplars:
                assert ex.section_code == code
                assert ex.section_name
                assert ex.registry_id
                assert ex.text
                assert len(ex.text) > 0

    def test_returns_dict_of_lists(self) -> None:
        """Scenario: Corpus loads correctly — returns dict."""
        corpus = load_corpus()
        assert isinstance(corpus, dict)
        for code, exemplars in corpus.items():
            assert isinstance(exemplars, list)
            for ex in exemplars:
                assert isinstance(ex, CorpusExemplar)

    def test_exemplars_substantive_content(self) -> None:
        """Scenario: Exemplars contain substantive content."""
        corpus = load_corpus()
        for code, exemplars in corpus.items():
            for ex in exemplars:
                assert len(ex.text) >= 500, (
                    f"{code} {ex.registry_id}: "
                    f"{len(ex.text)} chars < 500 minimum"
                )

    def test_cross_protocol_diversity(self) -> None:
        """Scenario: Cross-protocol diversity."""
        corpus = load_corpus()
        diverse_count = 0
        for code in _ALL_CODES:
            exemplars = corpus.get(code, [])
            registry_ids = {e.registry_id for e in exemplars}
            if len(registry_ids) >= 2:
                diverse_count += 1
        assert diverse_count >= 8, (
            f"Only {diverse_count}/11 codes have 2+ protocols, need >=8"
        )

    def test_missing_file_returns_empty(self) -> None:
        corpus = load_corpus(path="/nonexistent/path/corpus.jsonl")
        assert corpus == {}

    def test_custom_path(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
        ) as f:
            entry = {
                "section_code": "B.1",
                "section_name": "General Information",
                "registry_id": "NCT_TEST",
                "text": "A" * 600,
                "confidence": 0.95,
                "verified_by": "test",
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            corpus = load_corpus(path=f.name)

        assert "B.1" in corpus
        assert len(corpus["B.1"]) == 1
        assert corpus["B.1"][0].registry_id == "NCT_TEST"

        Path(f.name).unlink()


class TestGetExemplars:
    """Tests for get_exemplars function."""

    def test_returns_exemplars_for_code(self) -> None:
        corpus = load_corpus()
        exemplars = get_exemplars(corpus, "B.3")
        assert len(exemplars) > 0
        for ex in exemplars:
            assert ex.section_code == "B.3"

    def test_respects_max_exemplars(self) -> None:
        corpus = load_corpus()
        exemplars = get_exemplars(corpus, "B.3", max_exemplars=1)
        assert len(exemplars) == 1

    def test_sorted_by_confidence(self) -> None:
        corpus = load_corpus()
        exemplars = get_exemplars(corpus, "B.3", max_exemplars=4)
        for i in range(1, len(exemplars)):
            assert exemplars[i - 1].confidence >= exemplars[i].confidence

    def test_missing_code_returns_empty(self) -> None:
        corpus = load_corpus()
        exemplars = get_exemplars(corpus, "B.99")
        assert exemplars == []


class TestCorpusStats:
    """Tests for corpus_stats function."""

    def test_total_exemplars(self) -> None:
        corpus = load_corpus()
        stats = corpus_stats(corpus)
        assert stats["total_exemplars"] >= 22  # 11 codes * 2 min

    def test_section_count(self) -> None:
        corpus = load_corpus()
        stats = corpus_stats(corpus)
        assert stats["section_count"] == 11

    def test_exemplars_per_section(self) -> None:
        corpus = load_corpus()
        stats = corpus_stats(corpus)
        for code in _ALL_CODES:
            assert stats["exemplars_per_section"][code] >= 2

    def test_registry_diversity(self) -> None:
        corpus = load_corpus()
        stats = corpus_stats(corpus)
        for code in _ALL_CODES:
            assert stats["registry_diversity"][code] >= 1
