"""Tests for PTCV-234: Centroid Classifier and Integration.

Tests verify centroid computation, classification, high-confidence
detection, RagIndex model auto-detection, and ClassificationRouter
centroid pre-filter integration.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ptcv.ich_parser.centroid_classifier import (
    CentroidClassifier,
    CentroidMatch,
    load_centroid_classifier,
)


def _make_centroids(
    n_sections: int = 4,
    dim: int = 384,
) -> dict[str, tuple[np.ndarray, str]]:
    """Create mock centroids with well-separated directions."""
    rng = np.random.RandomState(42)
    codes = [f"B.{i}" for i in range(1, n_sections + 1)]
    names = [f"Section {c}" for c in codes]
    centroids = {}

    for i, (code, name) in enumerate(zip(codes, names)):
        vec = rng.randn(dim).astype(np.float32)
        # Make each centroid point in a distinct direction
        vec[i * (dim // n_sections):(i + 1) * (dim // n_sections)] += 5.0
        vec = vec / np.linalg.norm(vec)
        centroids[code] = (vec, name)

    return centroids


class TestCentroidMatch:
    """Tests for CentroidMatch dataclass."""

    def test_creation(self):
        """Test CentroidMatch creation."""
        m = CentroidMatch(
            section_code="B.8",
            section_name="Assessment of Efficacy",
            confidence=0.91,
        )
        assert m.section_code == "B.8"
        assert m.confidence == 0.91

    def test_frozen(self):
        """CentroidMatch is immutable."""
        m = CentroidMatch(section_code="B.3", confidence=0.5)
        with pytest.raises(AttributeError):
            m.confidence = 0.9  # type: ignore


class TestCentroidClassifier:
    """Tests for CentroidClassifier class."""

    def test_init_stores_codes(self):
        """Initialisation stores sorted section codes."""
        centroids = _make_centroids(4)
        model = MagicMock()

        clf = CentroidClassifier(centroids=centroids, model=model)

        assert clf.section_count == 4
        assert clf.section_codes == ["B.1", "B.2", "B.3", "B.4"]

    def test_empty_centroids(self):
        """Empty centroids produce empty classification."""
        model = MagicMock()
        clf = CentroidClassifier(centroids={}, model=model)

        assert clf.classify("some text") == []
        assert clf.section_count == 0

    def test_classify_returns_top_k(self):
        """Classification returns top-k sorted by confidence."""
        centroids = _make_centroids(4)
        mock_model = MagicMock()
        # Return an embedding that points toward B.3's centroid
        b3_vec = centroids["B.3"][0]
        mock_model.encode.return_value = b3_vec.reshape(1, -1)

        clf = CentroidClassifier(centroids=centroids, model=mock_model)
        matches = clf.classify("test text", top_k=3)

        assert len(matches) == 3
        # B.3 should be top match (highest similarity to its own centroid)
        assert matches[0].section_code == "B.3"
        assert matches[0].confidence > matches[1].confidence

    def test_classify_confidence_range(self):
        """Confidence values are in valid range."""
        centroids = _make_centroids(4)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        clf = CentroidClassifier(centroids=centroids, model=mock_model)
        matches = clf.classify("test text", top_k=4)

        for m in matches:
            assert -1.0 <= m.confidence <= 1.0

    def test_classify_top_k_capped(self):
        """Top-k capped at number of centroids."""
        centroids = _make_centroids(2)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        clf = CentroidClassifier(centroids=centroids, model=mock_model)
        matches = clf.classify("test", top_k=10)

        assert len(matches) == 2  # Only 2 centroids


class TestHighConfidence:
    """Tests for is_high_confidence static method."""

    def test_above_threshold(self):
        """High confidence when top match exceeds threshold."""
        matches = [
            CentroidMatch("B.8", "Efficacy", 0.91),
            CentroidMatch("B.4", "Design", 0.65),
        ]
        assert CentroidClassifier.is_high_confidence(matches, 0.85)

    def test_below_threshold(self):
        """Not high confidence when top match is below threshold."""
        matches = [
            CentroidMatch("B.8", "Efficacy", 0.70),
            CentroidMatch("B.4", "Design", 0.65),
        ]
        assert not CentroidClassifier.is_high_confidence(matches, 0.85)

    def test_empty_matches(self):
        """Not high confidence with empty matches."""
        assert not CentroidClassifier.is_high_confidence([], 0.85)

    def test_exact_threshold(self):
        """High confidence at exact threshold."""
        matches = [CentroidMatch("B.3", "Objectives", 0.85)]
        assert CentroidClassifier.is_high_confidence(matches, 0.85)


class TestLoadCentroidClassifier:
    """Tests for load_centroid_classifier convenience function."""

    def test_returns_none_on_missing_corpus(self, tmp_path):
        """Returns None when corpus file doesn't exist."""
        result = load_centroid_classifier(
            corpus_path=tmp_path / "nonexistent.parquet",
        )
        assert result is None

    def test_returns_none_on_import_error(self):
        """Returns None when dependencies not available."""
        with patch(
            "ptcv.ich_parser.centroid_classifier.CentroidClassifier.from_alignment_corpus",
            side_effect=ImportError("no pandas"),
        ):
            result = load_centroid_classifier()
            assert result is None


class TestRagIndexModelAutoDetect:
    """Tests for RagIndex fine-tuned model auto-detection."""

    def test_finetuned_path_constant_defined(self):
        """_FINETUNED_MODEL_DIR is defined in rag_index."""
        from ptcv.ich_parser.rag_index import _FINETUNED_MODEL_DIR
        assert _FINETUNED_MODEL_DIR is not None
        assert "ich_classifier_v1" in str(_FINETUNED_MODEL_DIR)


class TestClassificationRouterCentroid:
    """Tests for ClassificationRouter centroid pre-filter."""

    def test_set_centroid_classifier(self):
        """set_centroid_classifier stores the classifier."""
        from ptcv.ich_parser.classification_router import (
            ClassificationRouter,
        )

        router = ClassificationRouter()
        mock_clf = MagicMock()
        router.set_centroid_classifier(mock_clf)

        assert router._centroid_classifier is mock_clf

    def test_set_centroid_classifier_none(self):
        """set_centroid_classifier(None) disables centroid pre-filter."""
        from ptcv.ich_parser.classification_router import (
            ClassificationRouter,
        )

        router = ClassificationRouter()
        router.set_centroid_classifier(None)

        assert router._centroid_classifier is None
