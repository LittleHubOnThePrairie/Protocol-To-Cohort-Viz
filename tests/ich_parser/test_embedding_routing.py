"""Tests for embedding-based semantic routing Layer 2 (PTCV-256).

GHERKIN Scenarios:
  Feature: Embedding-based semantic routing in QueryExtractor

    Scenario: Layer 2 recovers from Stage 2 misclassification
    Scenario: Layer 2 supplements Layer 1 for better coverage
    Scenario: Centroid similarity scores discriminate sections
    Scenario: No performance regression for well-classified protocols
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.centroid_classifier import (
    CentroidClassifier,
    CentroidMatch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_centroids(
    codes: list[str],
    dim: int = 384,
) -> dict[str, tuple[np.ndarray, str]]:
    """Create mock centroids with well-separated directions."""
    rng = np.random.RandomState(42)
    centroids = {}
    n = len(codes)
    for i, code in enumerate(codes):
        vec = rng.randn(dim).astype(np.float32)
        # Make each centroid point strongly in a distinct subspace
        start = i * (dim // n)
        end = (i + 1) * (dim // n)
        vec[start:end] += 5.0
        vec = vec / np.linalg.norm(vec)
        centroids[code] = (vec, f"Section {code}")
    return centroids


def _make_mock_model(centroids: dict[str, tuple[np.ndarray, str]]):
    """Create a mock sentence-transformer that returns centroid vectors.

    When encoding a text like "B.4 content", returns the centroid
    vector for B.4 (with slight noise for realism).
    """
    model = MagicMock()
    code_map = {code: vec for code, (vec, _) in centroids.items()}

    def encode_fn(texts, **kwargs):
        results = []
        rng = np.random.RandomState(99)
        for text in (texts if isinstance(texts, list) else [texts]):
            matched = False
            for code, vec in code_map.items():
                if code.lower() in text.lower():
                    # Return centroid + small noise
                    noise = rng.randn(*vec.shape).astype(np.float32) * 0.05
                    result = vec + noise
                    result = result / np.linalg.norm(result)
                    results.append(result)
                    matched = True
                    break
            if not matched:
                # Random vector for unmatched text
                v = rng.randn(384).astype(np.float32)
                v = v / np.linalg.norm(v)
                results.append(v)
        return np.stack(results).astype(np.float32)

    model.encode = encode_fn
    return model


# ---------------------------------------------------------------------------
# Scenario: Centroid similarity scores discriminate sections
# ---------------------------------------------------------------------------


class TestCentroidSimilarityDiscriminates:
    """Content aligned to B.4 should score highest against B.4 centroid."""

    def test_b4_content_scores_highest_against_b4(self):
        codes = ["B.3", "B.4", "B.5", "B.8"]
        centroids = _make_centroids(codes)
        model = _make_mock_model(centroids)
        classifier = CentroidClassifier(centroids=centroids, model=model)

        matches = classifier.classify("B.4 primary endpoints content")
        assert matches[0].section_code == "B.4"
        assert matches[0].confidence > 0.5

    def test_b5_content_scores_highest_against_b5(self):
        codes = ["B.3", "B.4", "B.5", "B.8"]
        centroids = _make_centroids(codes)
        model = _make_mock_model(centroids)
        classifier = CentroidClassifier(centroids=centroids, model=model)

        matches = classifier.classify("B.5 eligibility criteria content")
        assert matches[0].section_code == "B.5"

    def test_cross_section_similarity_is_low(self):
        codes = ["B.3", "B.4", "B.5", "B.8"]
        centroids = _make_centroids(codes)
        model = _make_mock_model(centroids)
        classifier = CentroidClassifier(centroids=centroids, model=model)

        matches = classifier.classify("B.4 primary endpoints content")
        # Top match (B.4) should be much higher than others
        assert matches[0].confidence > matches[1].confidence + 0.1


# ---------------------------------------------------------------------------
# Scenario: score_spans batch scoring
# ---------------------------------------------------------------------------


class TestScoreSpans:
    """score_spans should rank content spans correctly per section."""

    def test_returns_highest_similarity_for_matching_section(self):
        codes = ["B.4", "B.5", "B.8"]
        centroids = _make_centroids(codes)
        model = _make_mock_model(centroids)
        classifier = CentroidClassifier(centroids=centroids, model=model)

        spans = {
            "5.1": "B.4 Trial Design with primary endpoints",
            "6.1": "B.5 Eligibility criteria and inclusion",
            "7.1": "B.8 Efficacy assessments outcomes module",
        }
        result = classifier.score_spans(spans, top_k=3)

        # B.4 centroid should rank span "5.1" (B.4 content) highest
        assert result["B.4"][0][0] == "5.1"
        # B.5 centroid should rank span "6.1" (B.5 content) highest
        assert result["B.5"][0][0] == "6.1"
        # B.8 centroid should rank span "7.1" (B.8 content) highest
        assert result["B.8"][0][0] == "7.1"

    def test_empty_spans_returns_empty(self):
        codes = ["B.4"]
        centroids = _make_centroids(codes)
        model = _make_mock_model(centroids)
        classifier = CentroidClassifier(centroids=centroids, model=model)

        assert classifier.score_spans({}) == {}

    def test_top_k_respected(self):
        codes = ["B.4"]
        centroids = _make_centroids(codes)
        model = _make_mock_model(centroids)
        classifier = CentroidClassifier(centroids=centroids, model=model)

        spans = {f"s{i}": f"B.4 content block {i}" for i in range(10)}
        result = classifier.score_spans(spans, top_k=2)
        assert len(result["B.4"]) <= 2


# ---------------------------------------------------------------------------
# Scenario: Layer 2 recovers from Stage 2 misclassification
# ---------------------------------------------------------------------------


class TestLayer2RecoveryFromMisclassification:
    """When Stage 2 misroutes content, Layer 2 finds it semantically."""

    def test_embedding_route_used_when_layer1_missing(self):
        """B.4.1 has no Layer 1 route but Layer 2 finds relevant content."""
        from ptcv.ich_parser.query_extractor import QueryExtractor

        # Build embedding routes with B.4.1 content
        embedding_routes = {
            "B.4.1": [
                ("Primary endpoint: Overall Response Rate", 0.88, "8.1"),
            ],
            "B.4": [
                ("Primary endpoint: Overall Response Rate", 0.88, "8.1"),
            ],
        }

        extractor = QueryExtractor(confidence_threshold=0.70)

        # Mock query
        query = MagicMock()
        query.query_id = "B.4.1.q1"
        query.section_id = "B.4.1"
        query.schema_section = "B.4"
        query.expected_type = "text_long"
        query.keywords = ["endpoint", "primary"]

        # Empty Layer 1 routes
        routes: dict = {}

        # Mock protocol_index with full_text
        protocol_index = MagicMock()
        protocol_index.full_text = ""

        match_result = MagicMock()

        result = extractor._process_query(
            query, routes, protocol_index, match_result,
            embedding_routes,
        )

        assert result is not None
        assert "embedding" in result.source_section

    def test_layer1_takes_precedence_over_layer2(self):
        """When Layer 1 has content, Layer 2 is not used."""
        from ptcv.ich_parser.query_extractor import QueryExtractor
        from ptcv.ich_parser.section_matcher import MatchConfidence

        embedding_routes = {
            "B.4": [
                ("Layer 2 content", 0.90, "8.1"),
            ],
        }

        extractor = QueryExtractor(confidence_threshold=0.70)

        query = MagicMock()
        query.query_id = "B.4.q1"
        query.section_id = "B.4"
        query.schema_section = "B.4"
        query.expected_type = "text_long"
        query.keywords = ["design"]

        routes = {
            "B.4": (
                "Layer 1: Trial uses randomized double-blind design",
                "4.1",
                MatchConfidence.HIGH,
            ),
        }

        protocol_index = MagicMock()
        protocol_index.full_text = ""
        match_result = MagicMock()

        result = extractor._process_query(
            query, routes, protocol_index, match_result,
            embedding_routes,
        )

        assert result is not None
        assert "embedding" not in result.source_section


# ---------------------------------------------------------------------------
# Scenario: No performance regression for well-classified protocols
# ---------------------------------------------------------------------------


class TestNoPerformanceRegression:
    """Embedding routes add no Sonnet calls; only local computation."""

    def test_no_api_calls_for_embedding_routes(self):
        """_build_embedding_routes makes zero external API calls."""
        from ptcv.ich_parser.query_extractor import QueryExtractor

        protocol_index = MagicMock()
        protocol_index.content_spans = {
            "1.1": "Introduction to the study",
            "5.1": "Eligibility criteria",
        }

        # Mock the centroid classifier to avoid real model loading
        mock_classifier = MagicMock()
        mock_classifier.score_spans.return_value = {
            "B.5": [("5.1", 0.85)],
        }

        with patch(
            "ptcv.ich_parser.centroid_classifier.load_centroid_classifier",
            return_value=mock_classifier,
        ):
            result = QueryExtractor._build_embedding_routes(
                protocol_index,
            )

        # Verify score_spans was called (local computation)
        mock_classifier.score_spans.assert_called_once()
        assert "B.5" in result

    def test_empty_content_spans_returns_empty(self):
        """No embedding routes when content_spans is empty."""
        from ptcv.ich_parser.query_extractor import QueryExtractor

        protocol_index = MagicMock()
        protocol_index.content_spans = {}

        result = QueryExtractor._build_embedding_routes(
            protocol_index,
        )
        assert result == {}

    def test_classifier_unavailable_returns_empty(self):
        """Graceful degradation when classifier can't load."""
        from ptcv.ich_parser.query_extractor import QueryExtractor

        protocol_index = MagicMock()
        protocol_index.content_spans = {"1.1": "Some text"}

        with patch(
            "ptcv.ich_parser.centroid_classifier.load_centroid_classifier",
            return_value=None,
        ):
            result = QueryExtractor._build_embedding_routes(
                protocol_index,
            )
        assert result == {}


# ---------------------------------------------------------------------------
# Scenario: Layer 2 supplements Layer 1 for better coverage
# ---------------------------------------------------------------------------


class TestLayer2SupplementsLayer1:
    """Layer 2 provides content when Layer 1 is empty for a section."""

    def test_embedding_route_fills_gap(self):
        """Query with no Layer 1 route gets answered via Layer 2."""
        from ptcv.ich_parser.query_extractor import QueryExtractor

        extractor = QueryExtractor(confidence_threshold=0.70)

        query = MagicMock()
        query.query_id = "B.8.q1"
        query.section_id = "B.8"
        query.schema_section = "B.8"
        query.expected_type = "text_long"
        query.keywords = ["efficacy", "outcome"]

        routes: dict = {}  # Empty Layer 1

        embedding_routes = {
            "B.8": [
                (
                    "Primary efficacy endpoint is overall survival "
                    "measured at 24 months. Secondary endpoints "
                    "include progression-free survival and response rate.",
                    0.75,
                    "7.2",
                ),
            ],
        }

        protocol_index = MagicMock()
        protocol_index.full_text = ""
        match_result = MagicMock()

        result = extractor._process_query(
            query, routes, protocol_index, match_result,
            embedding_routes,
        )

        assert result is not None
        assert "embedding:7.2" == result.source_section

    def test_low_similarity_not_used(self):
        """Embedding route below min similarity threshold is skipped."""
        from ptcv.ich_parser.query_extractor import QueryExtractor

        extractor = QueryExtractor(confidence_threshold=0.70)

        query = MagicMock()
        query.query_id = "B.8.q1"
        query.section_id = "B.8"
        query.schema_section = "B.8"
        query.expected_type = "text_long"
        query.keywords = ["efficacy"]

        routes: dict = {}

        embedding_routes = {
            "B.8": [
                ("Irrelevant text", 0.15, "1.1"),  # Below threshold
            ],
        }

        protocol_index = MagicMock()
        protocol_index.full_text = ""
        match_result = MagicMock()

        result = extractor._process_query(
            query, routes, protocol_index, match_result,
            embedding_routes,
        )

        # Should be None — embedding similarity too low, no full_text
        assert result is None
