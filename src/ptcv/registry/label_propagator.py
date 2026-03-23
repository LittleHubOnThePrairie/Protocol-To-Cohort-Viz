"""Label Propagator: ICH Section Labels via FAISS Similarity Graph.

PTCV-235: Propagates ICH section labels from registry-linked vectors
to unlabeled vectors from non-registry protocols using a FAISS-based
K-nearest-neighbor similarity graph.  Confidence decays with similarity
distance and propagation depth.

This is a batch (offline) operation that enriches the RAG index metadata.
Pseudo-labels are stored with ``source="propagated"`` to distinguish
them from registry ground truth and PDF-extracted classifications.

Risk tier: LOW -- read-only FAISS queries + metadata update (no API calls).

Usage::

    from ptcv.registry.label_propagator import (
        LabelPropagator,
        PropagationResult,
    )

    propagator = LabelPropagator(k_neighbors=5, max_hops=2)
    result = propagator.propagate(rag_index)
    print(f"Propagated {result.labels_assigned} pseudo-labels")
    propagator.apply_to_index(rag_index, result)
    rag_index.save()
"""

import dataclasses
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_K = 5
_DEFAULT_MAX_HOPS = 2
_DEFAULT_MIN_CONFIDENCE = 0.80
_DEFAULT_MIN_SIMILARITY = 0.50


@dataclasses.dataclass
class PseudoLabel:
    """A propagated ICH section label for an unlabeled vector.

    Attributes:
        vector_index: Index position in the FAISS index.
        section_code: Propagated ICH section code.
        confidence: Decayed confidence (source_conf * similarity).
        source_index: Index of the labeled vector that sourced this.
        hop: Propagation depth (1 = direct neighbor, 2 = neighbor's neighbor).
        similarity: Cosine similarity to the source vector.
    """

    vector_index: int
    section_code: str
    confidence: float
    source_index: int
    hop: int
    similarity: float


@dataclasses.dataclass
class PropagationResult:
    """Result of label propagation across the similarity graph.

    Attributes:
        labels_assigned: Total pseudo-labels created.
        vectors_labeled: Unique vectors that received labels.
        vectors_skipped: Vectors already labeled (registry/classified).
        vectors_below_threshold: Vectors whose best neighbor was too dissimilar.
        hop_1_count: Labels assigned at hop 1 (direct neighbors).
        hop_2_count: Labels assigned at hop 2.
        pseudo_labels: All generated pseudo-labels.
        avg_confidence: Mean confidence across all pseudo-labels.
    """

    labels_assigned: int = 0
    vectors_labeled: int = 0
    vectors_skipped: int = 0
    vectors_below_threshold: int = 0
    hop_1_count: int = 0
    hop_2_count: int = 0
    pseudo_labels: list[PseudoLabel] = dataclasses.field(
        default_factory=list
    )
    avg_confidence: float = 0.0


class LabelPropagator:
    """Propagate ICH labels through a FAISS similarity graph.

    For each unlabeled vector in the index, finds K nearest labeled
    neighbors and assigns the majority label with decayed confidence.
    Supports multi-hop propagation (hop 1 uses registry labels,
    hop 2 uses hop-1 pseudo-labels).

    Args:
        k_neighbors: Number of nearest neighbors to query per vector.
        max_hops: Maximum propagation depth (1 or 2).
        min_confidence: Minimum confidence for a pseudo-label to be kept.
        min_similarity: Minimum cosine similarity to accept a neighbor.
    """

    def __init__(
        self,
        k_neighbors: int = _DEFAULT_K,
        max_hops: int = _DEFAULT_MAX_HOPS,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        min_similarity: float = _DEFAULT_MIN_SIMILARITY,
    ) -> None:
        self._k = k_neighbors
        self._max_hops = min(max_hops, 2)  # Cap at 2
        self._min_confidence = min_confidence
        self._min_similarity = min_similarity

    def propagate(self, rag_index: Any) -> PropagationResult:
        """Run label propagation on the RAG index.

        Args:
            rag_index: RagIndex instance with loaded FAISS index and metadata.

        Returns:
            PropagationResult with all generated pseudo-labels.
        """
        if rag_index.is_empty:
            return PropagationResult()

        metadata = rag_index._metadata
        index = rag_index._index
        n_vectors = len(metadata)

        # Identify labeled vs unlabeled vectors
        labeled_indices = set()
        label_map: dict[int, str] = {}  # index -> section_code
        confidence_map: dict[int, float] = {}  # index -> confidence

        for i, meta in enumerate(metadata):
            source = meta.get("source", "")
            section_code = meta.get("section_code", "")
            conf = meta.get("confidence_score", 0.0)

            if source == "registry" and section_code:
                labeled_indices.add(i)
                label_map[i] = section_code
                confidence_map[i] = conf

        unlabeled_indices = set(range(n_vectors)) - labeled_indices

        if not labeled_indices:
            logger.info("No registry-labeled vectors; nothing to propagate")
            return PropagationResult()

        logger.info(
            "LabelPropagator: %d labeled, %d unlabeled vectors",
            len(labeled_indices),
            len(unlabeled_indices),
        )

        # Hop 1: propagate from registry labels to direct neighbors
        hop1_labels = self._propagate_hop(
            index=index,
            metadata=metadata,
            source_indices=labeled_indices,
            target_indices=unlabeled_indices,
            label_map=label_map,
            confidence_map=confidence_map,
            hop=1,
        )

        # Update maps with hop-1 results for hop-2 propagation
        hop1_labeled = set()
        for pl in hop1_labels:
            hop1_labeled.add(pl.vector_index)
            label_map[pl.vector_index] = pl.section_code
            confidence_map[pl.vector_index] = pl.confidence

        # Hop 2: propagate from hop-1 pseudo-labels to remaining unlabeled
        hop2_labels: list[PseudoLabel] = []
        if self._max_hops >= 2:
            remaining_unlabeled = unlabeled_indices - hop1_labeled
            if remaining_unlabeled and hop1_labeled:
                hop2_labels = self._propagate_hop(
                    index=index,
                    metadata=metadata,
                    source_indices=hop1_labeled,
                    target_indices=remaining_unlabeled,
                    label_map=label_map,
                    confidence_map=confidence_map,
                    hop=2,
                )

        all_labels = hop1_labels + hop2_labels
        labeled_vectors = {pl.vector_index for pl in all_labels}

        avg_conf = (
            sum(pl.confidence for pl in all_labels) / len(all_labels)
            if all_labels
            else 0.0
        )

        result = PropagationResult(
            labels_assigned=len(all_labels),
            vectors_labeled=len(labeled_vectors),
            vectors_skipped=len(labeled_indices),
            vectors_below_threshold=(
                len(unlabeled_indices) - len(labeled_vectors)
            ),
            hop_1_count=len(hop1_labels),
            hop_2_count=len(hop2_labels),
            pseudo_labels=all_labels,
            avg_confidence=round(avg_conf, 4),
        )

        logger.info(
            "LabelPropagator: assigned %d labels (%d hop-1, %d hop-2), "
            "avg confidence %.3f",
            result.labels_assigned,
            result.hop_1_count,
            result.hop_2_count,
            result.avg_confidence,
        )

        return result

    def apply_to_index(
        self,
        rag_index: Any,
        result: PropagationResult,
    ) -> int:
        """Apply pseudo-labels to the RAG index metadata.

        Updates metadata entries for labeled vectors with
        ``source="propagated"``, the assigned section_code, and
        propagation metadata.

        Args:
            rag_index: RagIndex instance to update.
            result: PropagationResult from propagate().

        Returns:
            Number of metadata entries updated.
        """
        updated = 0
        for pl in result.pseudo_labels:
            if pl.vector_index < len(rag_index._metadata):
                meta = rag_index._metadata[pl.vector_index]
                meta["source"] = "propagated"
                meta["section_code"] = pl.section_code
                meta["confidence_score"] = pl.confidence
                meta["propagation_hop"] = pl.hop
                meta["propagation_source_index"] = pl.source_index
                meta["propagation_similarity"] = round(pl.similarity, 4)
                updated += 1

        logger.info(
            "LabelPropagator: updated %d metadata entries", updated
        )
        return updated

    def get_high_confidence_labels(
        self,
        result: PropagationResult,
        threshold: float = 0.80,
    ) -> list[PseudoLabel]:
        """Extract high-confidence pseudo-labels for self-training.

        These can be used as additional training pairs for the
        SetFit fine-tuning loop (PTCV-233).

        Args:
            result: PropagationResult from propagate().
            threshold: Minimum confidence to include.

        Returns:
            Filtered list of high-confidence PseudoLabel objects.
        """
        return [
            pl for pl in result.pseudo_labels
            if pl.confidence >= threshold
        ]

    def _propagate_hop(
        self,
        index: Any,
        metadata: list[dict[str, Any]],
        source_indices: set[int],
        target_indices: set[int],
        label_map: dict[int, str],
        confidence_map: dict[int, float],
        hop: int,
    ) -> list[PseudoLabel]:
        """Propagate labels from source to target vectors via KNN.

        For each target vector, finds K nearest neighbors among ALL
        vectors, then checks if any neighbor is a source. Assigns
        the majority source label with decayed confidence.

        Args:
            index: FAISS index for KNN queries.
            metadata: Index metadata list.
            source_indices: Indices of labeled vectors.
            target_indices: Indices of unlabeled vectors to label.
            label_map: Current label assignments.
            confidence_map: Current confidence values.
            hop: Propagation depth for this hop.

        Returns:
            List of PseudoLabel objects for newly labeled vectors.
        """
        if not target_indices or not source_indices:
            return []

        pseudo_labels: list[PseudoLabel] = []

        # Batch KNN query for all target vectors
        target_list = sorted(target_indices)

        # Reconstruct target vectors from the FAISS index
        try:
            target_vectors = np.zeros(
                (len(target_list), index.d), dtype=np.float32
            )
            for i, tidx in enumerate(target_list):
                target_vectors[i] = index.reconstruct(tidx)
        except Exception:
            logger.debug(
                "FAISS reconstruct not available; skipping hop %d", hop
            )
            return []

        # Query K+1 neighbors (first is self)
        k_query = min(self._k + 1, index.ntotal)
        scores, indices = index.search(target_vectors, k_query)

        for i, tidx in enumerate(target_list):
            # Collect labeled neighbors
            neighbor_votes: dict[str, list[tuple[float, float, int]]] = {}

            for j in range(k_query):
                nidx = int(indices[i][j])
                sim = float(scores[i][j])

                # Skip self
                if nidx == tidx:
                    continue

                # Skip if not a labeled source
                if nidx not in source_indices:
                    continue

                # Skip low similarity
                if sim < self._min_similarity:
                    continue

                label = label_map.get(nidx, "")
                conf = confidence_map.get(nidx, 0.0)
                if label:
                    neighbor_votes.setdefault(label, []).append(
                        (conf, sim, nidx)
                    )

            if not neighbor_votes:
                continue

            # Majority vote: pick label with most neighbors
            best_label = max(
                neighbor_votes, key=lambda k: len(neighbor_votes[k])
            )
            votes = neighbor_votes[best_label]

            # Use best (highest similarity) vote for confidence
            best_conf, best_sim, best_source = max(
                votes, key=lambda v: v[1]
            )

            # Decay confidence: source_confidence * similarity
            decayed_conf = best_conf * best_sim

            if decayed_conf < self._min_confidence:
                continue

            pseudo_labels.append(PseudoLabel(
                vector_index=tidx,
                section_code=best_label,
                confidence=round(decayed_conf, 4),
                source_index=best_source,
                hop=hop,
                similarity=round(best_sim, 4),
            ))

        return pseudo_labels


__all__ = [
    "LabelPropagator",
    "PseudoLabel",
    "PropagationResult",
]
