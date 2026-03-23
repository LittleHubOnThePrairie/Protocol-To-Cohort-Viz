"""Tests for RAG exemplar tiebreaker in SectionMatcher (PTCV-238).

Tests that RAG exemplars disambiguate REVIEW-tier matches,
skip HIGH/LOW tiers, and degrade gracefully.
"""

from __future__ import annotations

import inspect
import pytest
from unittest.mock import MagicMock

from ptcv.ich_parser.section_matcher import SectionMatcher


def _make_rag_index(exemplar_codes: list[str]) -> MagicMock:
    """Build a mock RAG index returning exemplars with given codes."""
    mock = MagicMock()
    exemplars = []
    for code in exemplar_codes:
        ex = MagicMock()
        ex.section_code = code
        exemplars.append(ex)
    mock.query.return_value = exemplars
    return mock


class TestRagParameterAccepted:
    """Tests that SectionMatcher accepts rag_index parameter."""

    def test_default_none(self):
        matcher = SectionMatcher()
        assert matcher._rag_index is None

    def test_accepts_rag_index(self):
        mock_rag = MagicMock()
        matcher = SectionMatcher(rag_index=mock_rag)
        assert matcher._rag_index is mock_rag

    def test_rag_boost_default(self):
        matcher = SectionMatcher()
        assert matcher._rag_boost == 0.06

    def test_custom_rag_boost(self):
        matcher = SectionMatcher(rag_boost=0.10)
        assert matcher._rag_boost == 0.10


class TestRagTiebreakerMethod:
    """Tests for _apply_rag_tiebreaker method."""

    def test_method_exists(self):
        assert hasattr(SectionMatcher, '_apply_rag_tiebreaker')

    def test_no_rag_returns_unchanged(self):
        """Test scores unchanged when rag_index is None."""
        matcher = SectionMatcher()
        headers = [("1", "Title")]
        scores = [[0.6, 0.55, 0.0]]
        mock_index = MagicMock()
        mock_index.content_spans = {"1": "some text"}

        result = matcher._apply_rag_tiebreaker(
            headers, scores, mock_index,
        )
        assert result == scores

    def test_skips_high_confidence(self):
        """Test RAG not queried for HIGH-tier (>= 0.75)."""
        rag = _make_rag_index(["B.5", "B.5", "B.5"])
        matcher = SectionMatcher(rag_index=rag)

        headers = [("5", "Inclusion Criteria")]
        # Score 0.85 is HIGH — should skip
        scores = [[0.0] * len(matcher._ref_codes)]
        b5_idx = matcher._ref_codes.index("B.5")
        scores[0][b5_idx] = 0.85

        mock_pi = MagicMock()
        mock_pi.content_spans = {"5": "text"}

        matcher._apply_rag_tiebreaker(headers, scores, mock_pi)

        # RAG should NOT have been queried
        rag.query.assert_not_called()

    def test_skips_low_confidence(self):
        """Test RAG not queried for LOW-tier (< 0.50)."""
        rag = _make_rag_index(["B.1"])
        matcher = SectionMatcher(rag_index=rag)

        headers = [("1", "Unknown Section")]
        scores = [[0.3] * len(matcher._ref_codes)]

        mock_pi = MagicMock()
        mock_pi.content_spans = {"1": "text"}

        matcher._apply_rag_tiebreaker(headers, scores, mock_pi)
        rag.query.assert_not_called()

    def test_boosts_exemplar_consensus(self):
        """Test RAG boosts candidate with more exemplar votes."""
        rag = _make_rag_index(["B.8", "B.8", "B.4"])
        matcher = SectionMatcher(rag_index=rag, rag_boost=0.06)

        headers = [("7", "Study Procedures and Assessments")]
        n = len(matcher._ref_codes)
        scores = [[0.0] * n]

        b4_idx = matcher._ref_codes.index("B.4")
        b8_idx = matcher._ref_codes.index("B.8")

        # B.4 slightly ahead of B.8 — both REVIEW tier
        scores[0][b4_idx] = 0.62
        scores[0][b8_idx] = 0.58

        mock_pi = MagicMock()
        mock_pi.content_spans = {"7": "Schedule of assessments and procedures"}

        matcher._apply_rag_tiebreaker(headers, scores, mock_pi)

        # B.8 should have been boosted (2 votes vs 1)
        assert scores[0][b8_idx] == pytest.approx(0.58 + 0.06)
        # B.4 unchanged
        assert scores[0][b4_idx] == pytest.approx(0.62)

    def test_confirms_top_when_consensus_agrees(self):
        """Test RAG confirms top candidate when exemplars agree."""
        rag = _make_rag_index(["B.4", "B.4", "B.4"])
        matcher = SectionMatcher(rag_index=rag, rag_boost=0.06)

        headers = [("4", "Trial Design")]
        n = len(matcher._ref_codes)
        scores = [[0.0] * n]

        b4_idx = matcher._ref_codes.index("B.4")
        b8_idx = matcher._ref_codes.index("B.8")

        scores[0][b4_idx] = 0.62
        scores[0][b8_idx] = 0.58

        mock_pi = MagicMock()
        mock_pi.content_spans = {"4": "Randomized double-blind design"}

        matcher._apply_rag_tiebreaker(headers, scores, mock_pi)

        # B.4 boosted (confirmed)
        assert scores[0][b4_idx] == pytest.approx(0.62 + 0.06)

    def test_skips_when_gap_too_large(self):
        """Test no boost when top-2 gap > 0.15."""
        rag = _make_rag_index(["B.8", "B.8", "B.8"])
        matcher = SectionMatcher(rag_index=rag)

        headers = [("7", "Assessments")]
        n = len(matcher._ref_codes)
        scores = [[0.0] * n]

        b4_idx = matcher._ref_codes.index("B.4")
        b8_idx = matcher._ref_codes.index("B.8")

        # Gap of 0.20 — too large for tiebreaker
        scores[0][b4_idx] = 0.70
        scores[0][b8_idx] = 0.50

        mock_pi = MagicMock()
        mock_pi.content_spans = {"7": "text"}

        original_b8 = scores[0][b8_idx]
        matcher._apply_rag_tiebreaker(headers, scores, mock_pi)

        # No boost applied
        assert scores[0][b8_idx] == original_b8

    def test_rag_query_failure_graceful(self):
        """Test RAG query exception doesn't crash."""
        rag = MagicMock()
        rag.query.side_effect = RuntimeError("FAISS error")
        matcher = SectionMatcher(rag_index=rag)

        headers = [("1", "Section")]
        n = len(matcher._ref_codes)
        scores = [[0.6] + [0.55] + [0.0] * (n - 2)]

        mock_pi = MagicMock()
        mock_pi.content_spans = {"1": "text"}

        # Should not raise
        matcher._apply_rag_tiebreaker(headers, scores, mock_pi)


class TestMatchMethodIntegration:
    """Tests that match() calls RAG tiebreaker."""

    def test_match_source_references_rag(self):
        """Test match() method contains RAG tiebreaker call."""
        source = inspect.getsource(SectionMatcher.match)
        assert "_apply_rag_tiebreaker" in source
        assert "PTCV-238" in source
