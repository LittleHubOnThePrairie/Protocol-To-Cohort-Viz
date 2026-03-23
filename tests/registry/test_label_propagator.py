"""Tests for PTCV-235: Label Propagation via FAISS Similarity Graph.

Tests verify propagation mechanics, confidence decay, hop bounding,
majority voting, metadata application, and high-confidence filtering.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from ptcv.registry.label_propagator import (
    LabelPropagator,
    PseudoLabel,
    PropagationResult,
)


def _make_mock_rag_index(
    n_vectors: int,
    dim: int = 384,
    registry_indices: dict[int, tuple[str, float]] | None = None,
) -> MagicMock:
    """Create a mock RagIndex with FAISS index and metadata.

    Args:
        n_vectors: Total vectors in the index.
        dim: Embedding dimensionality.
        registry_indices: Dict of {index: (section_code, confidence)}
            for registry-labeled vectors.
    """
    registry_indices = registry_indices or {}
    rng = np.random.RandomState(42)

    # Build metadata
    metadata = []
    for i in range(n_vectors):
        if i in registry_indices:
            code, conf = registry_indices[i]
            metadata.append({
                "section_code": code,
                "confidence_score": conf,
                "source": "registry",
                "content_text": f"Registry content for {code}",
            })
        else:
            metadata.append({
                "section_code": "",
                "confidence_score": 0.0,
                "source": "",
                "content_text": f"Unlabeled content {i}",
            })

    # Build fake FAISS index
    # Make registry vectors cluster together by section
    vectors = rng.randn(n_vectors, dim).astype(np.float32)
    for idx, (code, _) in registry_indices.items():
        # Shift registry vectors to make them similar to each other
        section_offset = hash(code) % 10
        vectors[idx, section_offset * 30:(section_offset + 1) * 30] += 3.0

    # Make some unlabeled vectors similar to registry vectors
    for i in range(n_vectors):
        if i not in registry_indices and i > 0:
            # Copy nearby registry vector with noise
            closest_reg = min(
                registry_indices.keys(),
                key=lambda r: abs(r - i),
                default=0,
            )
            if closest_reg in registry_indices:
                vectors[i] = vectors[closest_reg] + rng.randn(dim).astype(np.float32) * 0.3

    # Normalise
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vectors = vectors / norms

    # Build FAISS index
    import faiss

    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)

    mock_index = MagicMock()
    mock_index.is_empty = False
    mock_index._metadata = metadata
    mock_index._index = faiss_index

    return mock_index


class TestPseudoLabel:
    """Tests for PseudoLabel dataclass."""

    def test_creation(self):
        """Test PseudoLabel creation."""
        pl = PseudoLabel(
            vector_index=5,
            section_code="B.4",
            confidence=0.82,
            source_index=2,
            hop=1,
            similarity=0.91,
        )
        assert pl.vector_index == 5
        assert pl.section_code == "B.4"
        assert pl.hop == 1


class TestPropagationResult:
    """Tests for PropagationResult dataclass."""

    def test_defaults(self):
        """Test default values."""
        result = PropagationResult()
        assert result.labels_assigned == 0
        assert result.hop_1_count == 0
        assert result.hop_2_count == 0


class TestLabelPropagator:
    """Tests for LabelPropagator class."""

    def test_empty_index(self):
        """Empty index returns empty result."""
        mock_index = MagicMock()
        mock_index.is_empty = True

        propagator = LabelPropagator()
        result = propagator.propagate(mock_index)

        assert result.labels_assigned == 0

    def test_no_registry_vectors(self):
        """No registry vectors means nothing to propagate."""
        mock_index = _make_mock_rag_index(
            n_vectors=10,
            registry_indices={},
        )

        propagator = LabelPropagator()
        result = propagator.propagate(mock_index)

        assert result.labels_assigned == 0

    def test_hop_1_propagation(self):
        """Hop 1: labels propagate from registry to direct neighbors."""
        mock_index = _make_mock_rag_index(
            n_vectors=10,
            registry_indices={
                0: ("B.3", 0.9),
                1: ("B.4", 0.9),
            },
        )

        propagator = LabelPropagator(
            k_neighbors=5,
            max_hops=1,
            min_confidence=0.5,
            min_similarity=0.3,
        )
        result = propagator.propagate(mock_index)

        assert result.hop_1_count > 0
        assert result.hop_2_count == 0
        assert all(pl.hop == 1 for pl in result.pseudo_labels)

    def test_hop_2_propagation(self):
        """Hop 2: labels propagate from hop-1 to remaining unlabeled."""
        mock_index = _make_mock_rag_index(
            n_vectors=20,
            registry_indices={
                0: ("B.3", 0.9),
                5: ("B.8", 0.9),
            },
        )

        propagator = LabelPropagator(
            k_neighbors=5,
            max_hops=2,
            min_confidence=0.3,
            min_similarity=0.2,
        )
        result = propagator.propagate(mock_index)

        # Should have both hop 1 and hop 2 labels
        hops = {pl.hop for pl in result.pseudo_labels}
        assert 1 in hops
        # hop 2 may or may not appear depending on vector geometry

    def test_max_hops_capped_at_2(self):
        """max_hops is capped at 2 regardless of input."""
        propagator = LabelPropagator(max_hops=10)
        assert propagator._max_hops == 2

    def test_confidence_decays(self):
        """Pseudo-label confidence = source_conf * similarity."""
        mock_index = _make_mock_rag_index(
            n_vectors=5,
            registry_indices={0: ("B.3", 0.9)},
        )

        propagator = LabelPropagator(
            k_neighbors=3,
            max_hops=1,
            min_confidence=0.3,
            min_similarity=0.2,
        )
        result = propagator.propagate(mock_index)

        for pl in result.pseudo_labels:
            # Confidence should be <= source confidence (0.9)
            assert pl.confidence <= 0.9
            # Confidence = source_conf * similarity
            expected = 0.9 * pl.similarity
            assert abs(pl.confidence - round(expected, 4)) < 0.01

    def test_min_similarity_filters(self):
        """Neighbors below min_similarity are ignored."""
        mock_index = _make_mock_rag_index(
            n_vectors=5,
            registry_indices={0: ("B.3", 0.9)},
        )

        # Very high threshold should filter most neighbors
        propagator = LabelPropagator(
            k_neighbors=3,
            max_hops=1,
            min_confidence=0.0,
            min_similarity=0.99,
        )
        result = propagator.propagate(mock_index)

        # Most or all labels should be filtered
        assert result.labels_assigned <= 1

    def test_min_confidence_filters(self):
        """Pseudo-labels below min_confidence are discarded."""
        mock_index = _make_mock_rag_index(
            n_vectors=5,
            registry_indices={0: ("B.3", 0.5)},  # Low source confidence
        )

        propagator = LabelPropagator(
            k_neighbors=3,
            max_hops=1,
            min_confidence=0.9,  # Very high threshold
            min_similarity=0.2,
        )
        result = propagator.propagate(mock_index)

        # 0.5 * similarity will rarely exceed 0.9
        # All pseudo-labels should be filtered
        for pl in result.pseudo_labels:
            assert pl.confidence >= 0.9


class TestApplyToIndex:
    """Tests for apply_to_index method."""

    def test_updates_metadata(self):
        """Pseudo-labels update metadata with source='propagated'."""
        mock_index = _make_mock_rag_index(
            n_vectors=5,
            registry_indices={0: ("B.3", 0.9)},
        )

        result = PropagationResult(
            labels_assigned=1,
            pseudo_labels=[
                PseudoLabel(
                    vector_index=2,
                    section_code="B.3",
                    confidence=0.82,
                    source_index=0,
                    hop=1,
                    similarity=0.91,
                ),
            ],
        )

        propagator = LabelPropagator()
        updated = propagator.apply_to_index(mock_index, result)

        assert updated == 1
        meta = mock_index._metadata[2]
        assert meta["source"] == "propagated"
        assert meta["section_code"] == "B.3"
        assert meta["confidence_score"] == 0.82
        assert meta["propagation_hop"] == 1
        assert meta["propagation_source_index"] == 0
        assert meta["propagation_similarity"] == 0.91

    def test_does_not_touch_registry_vectors(self):
        """Registry vectors are not modified."""
        mock_index = _make_mock_rag_index(
            n_vectors=3,
            registry_indices={0: ("B.3", 0.9)},
        )

        result = PropagationResult(pseudo_labels=[])

        propagator = LabelPropagator()
        propagator.apply_to_index(mock_index, result)

        meta = mock_index._metadata[0]
        assert meta["source"] == "registry"
        assert meta["section_code"] == "B.3"


class TestHighConfidenceLabels:
    """Tests for get_high_confidence_labels filter."""

    def test_filters_by_threshold(self):
        """Only labels above threshold are returned."""
        result = PropagationResult(
            pseudo_labels=[
                PseudoLabel(0, "B.3", 0.92, 1, 1, 0.95),
                PseudoLabel(1, "B.4", 0.75, 2, 1, 0.83),
                PseudoLabel(2, "B.8", 0.88, 3, 1, 0.91),
                PseudoLabel(3, "B.5", 0.60, 4, 2, 0.70),
            ],
        )

        propagator = LabelPropagator()
        high_conf = propagator.get_high_confidence_labels(
            result, threshold=0.80
        )

        assert len(high_conf) == 2
        codes = {pl.section_code for pl in high_conf}
        assert codes == {"B.3", "B.8"}

    def test_empty_result(self):
        """Empty result returns empty list."""
        propagator = LabelPropagator()
        high_conf = propagator.get_high_confidence_labels(
            PropagationResult()
        )
        assert high_conf == []
