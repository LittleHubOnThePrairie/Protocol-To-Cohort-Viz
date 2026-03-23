"""Centroid Classifier: Fast ICH Section Pre-Filter.

PTCV-234: Computes centroid vectors per ICH section code from the
alignment corpus, then classifies new text blocks by cosine similarity
to centroids.  Acts as a fast first-pass filter before the expensive
Sonnet LLM classification call.

The centroid classifier does NOT replace Sonnet -- it provides candidate
section codes with confidence scores.  When the top centroid confidence
exceeds a threshold (default 0.85), the Sonnet call can be skipped.

Risk tier: LOW -- read-only embedding operations (no API calls).

Usage::

    from ptcv.ich_parser.centroid_classifier import (
        CentroidClassifier,
        CentroidMatch,
    )

    classifier = CentroidClassifier.from_alignment_corpus(
        "data/alignment_corpus/alignments.parquet",
        model_path="data/models/ich_classifier_v1",
    )
    matches = classifier.classify("Primary endpoint: overall response rate")
    # [CentroidMatch(section_code="B.8", confidence=0.91), ...]
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CORPUS_PATH = Path(
    "C:/Dev/PTCV/data/alignment_corpus/alignments.parquet"
)
_DEFAULT_MODEL_PATH = Path(
    "C:/Dev/PTCV/data/models/ich_classifier_v1"
)
_DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HIGH_CONFIDENCE_THRESHOLD = 0.85


@dataclasses.dataclass(frozen=True)
class CentroidMatch:
    """A classification candidate from the centroid classifier.

    Attributes:
        section_code: ICH E6(R3) section code (e.g. ``"B.8"``).
        section_name: Human-readable section name.
        confidence: Cosine similarity to the section centroid (0.0--1.0).
    """

    section_code: str
    section_name: str = ""
    confidence: float = 0.0


class CentroidClassifier:
    """Fast nearest-centroid classifier for ICH section codes.

    Maintains one centroid vector per ICH section code, computed as
    the mean of aligned text span embeddings from the PTCV-232 corpus.
    Classification is a single matrix multiplication (O(N_sections)).

    Args:
        centroids: Dict mapping section_code to (centroid_vector, section_name).
        model: Loaded sentence-transformer model for encoding query text.
    """

    def __init__(
        self,
        centroids: dict[str, tuple[np.ndarray, str]],
        model: Any,
    ) -> None:
        self._centroids = centroids
        self._model = model

        # Pre-build centroid matrix for vectorised classification
        self._codes: list[str] = []
        self._names: list[str] = []
        vectors: list[np.ndarray] = []

        for code in sorted(centroids):
            vec, name = centroids[code]
            self._codes.append(code)
            self._names.append(name)
            vectors.append(vec)

        if vectors:
            self._centroid_matrix = np.stack(vectors)  # (N, dim)
        else:
            self._centroid_matrix = np.empty((0, 0))

        logger.info(
            "CentroidClassifier: %d section centroids loaded",
            len(centroids),
        )

    @classmethod
    def from_alignment_corpus(
        cls,
        corpus_path: str | Path = _DEFAULT_CORPUS_PATH,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        base_model: str = _DEFAULT_BASE_MODEL,
    ) -> "CentroidClassifier":
        """Build classifier from alignment corpus and embedding model.

        Loads the alignment corpus, groups text spans by section_code,
        embeds all spans, and computes mean centroid per section.

        Args:
            corpus_path: Path to alignments.parquet from PTCV-232.
            model_path: Path to fine-tuned model directory.
            base_model: Fallback model if fine-tuned not available.

        Returns:
            Initialised CentroidClassifier.
        """
        try:
            import pandas as pd
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "pandas and sentence_transformers required"
            )

        corpus_path = Path(corpus_path)
        model_path = Path(model_path)

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Alignment corpus not found: {corpus_path}"
            )

        # Load model (prefer fine-tuned)
        if model_path.exists() and (model_path / "config.json").exists():
            model = SentenceTransformer(str(model_path))
            logger.info(
                "CentroidClassifier: using fine-tuned model from %s",
                model_path,
            )
        else:
            model = SentenceTransformer(base_model)
            logger.info(
                "CentroidClassifier: using base model %s",
                base_model,
            )

        # Load corpus and group by section
        df = pd.read_parquet(corpus_path)
        section_groups: dict[str, list[str]] = {}
        section_names: dict[str, str] = {}

        for _, row in df.iterrows():
            code = row["section_code"]
            text = row["pdf_text_span"]
            name = row.get("section_name", code)
            section_groups.setdefault(code, []).append(text)
            section_names[code] = name

        # Embed all texts and compute centroids
        centroids: dict[str, tuple[np.ndarray, str]] = {}

        for code, texts in section_groups.items():
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            # L2 normalise
            norms = np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

            # Centroid = mean of normalised vectors, re-normalised
            centroid = embeddings.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm

            centroids[code] = (centroid.astype(np.float32), section_names[code])

            logger.debug(
                "CentroidClassifier: %s centroid from %d spans",
                code,
                len(texts),
            )

        return cls(centroids=centroids, model=model)

    def classify(
        self,
        text: str,
        top_k: int = 3,
    ) -> list[CentroidMatch]:
        """Classify a text block by cosine similarity to centroids.

        Args:
            text: Text block to classify.
            top_k: Number of top matches to return.

        Returns:
            List of CentroidMatch sorted by confidence descending.
        """
        if not self._codes or len(self._centroid_matrix) == 0:
            return []

        # Encode and normalise
        embedding = self._model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cosine similarity = dot product of normalised vectors
        similarities = (self._centroid_matrix @ embedding.T).flatten()

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            CentroidMatch(
                section_code=self._codes[i],
                section_name=self._names[i],
                confidence=float(similarities[i]),
            )
            for i in top_indices
        ]

    def score_spans(
        self,
        spans: dict[str, str],
        top_k: int = 3,
    ) -> dict[str, list[tuple[str, float]]]:
        """Score all content spans against all section centroids.

        For each ICH section code, returns the top-k most similar
        content spans (by section number) ranked by cosine similarity.
        This enables embedding-based routing where content can be
        found regardless of how Stage 2 classified it.

        Args:
            spans: Dict mapping section_number to text content
                (e.g., ``ProtocolIndex.content_spans``).
            top_k: Number of top spans per section code.

        Returns:
            Dict mapping ICH section code to list of
            ``(section_number, similarity)`` tuples, sorted by
            similarity descending. Only spans with similarity > 0
            are included.

        PTCV-256: Batch similarity for semantic routing Layer 2.
        """
        if not self._codes or not spans:
            return {}

        # Embed all spans in one batch
        span_keys = list(spans.keys())
        span_texts = [spans[k] for k in span_keys]

        embeddings = self._model.encode(
            span_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # Similarity matrix: (N_centroids, N_spans)
        sim_matrix = self._centroid_matrix @ embeddings.T

        result: dict[str, list[tuple[str, float]]] = {}
        for i, code in enumerate(self._codes):
            scored = [
                (span_keys[j], float(sim_matrix[i, j]))
                for j in range(len(span_keys))
                if sim_matrix[i, j] > 0
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            result[code] = scored[:top_k]

        return result

    @property
    def section_codes(self) -> list[str]:
        """List of section codes with centroids."""
        return list(self._codes)

    @property
    def section_count(self) -> int:
        """Number of section centroids."""
        return len(self._codes)

    @staticmethod
    def is_high_confidence(
        matches: list[CentroidMatch],
        threshold: float = _HIGH_CONFIDENCE_THRESHOLD,
    ) -> bool:
        """Check if the top match exceeds the confidence threshold.

        When True, the Sonnet LLM call can be skipped.

        Args:
            matches: Classification results from ``classify()``.
            threshold: Confidence threshold (default 0.85).

        Returns:
            True if top match confidence >= threshold.
        """
        return bool(matches and matches[0].confidence >= threshold)


def load_centroid_classifier(
    corpus_path: str | Path = _DEFAULT_CORPUS_PATH,
    model_path: str | Path = _DEFAULT_MODEL_PATH,
) -> Optional[CentroidClassifier]:
    """Load centroid classifier, returning None if unavailable.

    Convenience function that swallows import/file errors
    for graceful degradation.
    """
    try:
        return CentroidClassifier.from_alignment_corpus(
            corpus_path=corpus_path,
            model_path=model_path,
        )
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        logger.info(
            "CentroidClassifier not available: %s", e
        )
        return None


__all__ = [
    "CentroidClassifier",
    "CentroidMatch",
    "load_centroid_classifier",
]
