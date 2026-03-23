"""FAISS-backed RAG index for classification context (PTCV-162).

Builds and queries a local embedding index from high-confidence prior
classified protocol sections.  Retrieved exemplars are injected into
the Sonnet classification prompt to improve accuracy on ambiguous
sections.

Research basis: RAG achieves 87.8% accuracy vs 62.6% standalone LLM
on clinical protocol extraction [3].

Citations:
  [3] "AI-assisted Protocol Information Extraction For Clinical
      Research," arXiv:2602.00052, 2025
  [11] Trial2Vec, arXiv:2206.14719 — zero-shot clinical trial
       similarity search

Risk tier: LOW — embedding index of publicly available protocol text.

Regulatory references:
- ALCOA+ Traceable: index tracks source registry_id and confidence.
- ALCOA+ Original: only sections with confidence >= 0.80 are indexed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import IchSection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy dependency loading (follows neobert_classifier._ensure_ml_deps())
# ---------------------------------------------------------------------------

_sentence_transformers: Any = None
_faiss: Any = None


def _ensure_rag_deps() -> tuple[Any, Any]:
    """Import sentence_transformers and faiss on first use.

    Raises:
        ImportError: When either dependency is missing, with an
            actionable install command.
    """
    global _sentence_transformers, _faiss
    if _sentence_transformers is None or _faiss is None:
        try:
            import sentence_transformers as _st
            import faiss as _fa

            _sentence_transformers = _st
            _faiss = _fa
        except ImportError as exc:
            raise ImportError(
                "RagIndex requires 'sentence-transformers' and "
                "'faiss-cpu'.  Install with:\n"
                "  pip install sentence-transformers faiss-cpu"
            ) from exc
    return _sentence_transformers, _faiss


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RagExemplar:
    """A single section retrieved from the RAG index.

    Attributes:
        section_code: ICH section code (e.g. ``"B.5"``).
        section_name: Human-readable section name.
        registry_id: Source protocol registry ID.
        confidence_score: Classification confidence when indexed.
        content_text: Section text (truncated for prompt injection).
        similarity_score: Cosine similarity to the query (0.0–1.0).
    """

    section_code: str
    section_name: str
    registry_id: str
    confidence_score: float
    content_text: str
    similarity_score: float


@dataclasses.dataclass
class RagIndexStats:
    """Metadata about the current RAG index state.

    Attributes:
        total_sections: Number of sections in the index.
        section_code_counts: Dict mapping section codes to counts.
        registry_ids: Set of unique registry IDs represented.
        embedding_model: Name of the embedding model used.
        embedding_dim: Dimensionality of embeddings.
    """

    total_sections: int
    section_code_counts: dict[str, int]
    registry_ids: set[str]
    embedding_model: str
    embedding_dim: int


# ---------------------------------------------------------------------------
# RagIndex
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_FINETUNED_MODEL_DIR = Path("C:/Dev/PTCV/data/models/ich_classifier_v1")
_INDEX_FILENAME = "faiss_index.bin"
_METADATA_FILENAME = "index_metadata.jsonl"
_MIN_CONFIDENCE = 0.80
_MAX_TEXT_CHARS = 2000
_CONTENT_DISPLAY_CHARS = 500


class RagIndex:
    """FAISS-backed embedding index for RAG-augmented classification.

    Provides build, query, save, and load operations over a FAISS
    flat-IP index of section embeddings from prior pipeline runs.
    Only sections with confidence >= ``min_confidence`` are indexed.

    Args:
        index_dir: Directory path for persisted index files.
            Corresponds to ``PTCV_RAG_INDEX`` env var.
        embedding_model: sentence-transformers model name.
        min_confidence: Minimum confidence for indexing.
        max_text_chars: Maximum characters of content_text to embed.
    """

    def __init__(
        self,
        index_dir: str | Path,
        embedding_model: str = _DEFAULT_MODEL,
        min_confidence: float = _MIN_CONFIDENCE,
        max_text_chars: int = _MAX_TEXT_CHARS,
    ) -> None:
        self._index_dir = Path(index_dir)
        self._embedding_model_name = embedding_model
        self._min_confidence = min_confidence
        self._max_text_chars = max_text_chars

        # Lazy-loaded embedding model
        self._model: Any = None

        # In-memory state
        self._index: Any = None  # faiss.IndexFlatIP
        self._metadata: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def build_from_sections(
        self,
        sections: list["IchSection"],
    ) -> RagIndexStats:
        """Build index from IchSection list, filtering by min_confidence.

        Replaces any existing index in memory and on disk.

        Args:
            sections: List of IchSection instances from prior runs.

        Returns:
            RagIndexStats with index metadata.
        """
        st, faiss = _ensure_rag_deps()

        filtered = self._filter_sections(sections)
        if not filtered:
            dim = self._get_embedding_dim()
            self._index = faiss.IndexFlatIP(dim)
            self._metadata = []
            logger.info("RagIndex: built empty index (0 qualifying sections)")
            return self.stats

        texts = [s.content_text[:self._max_text_chars] for s in filtered]
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._metadata = [
            {
                "section_code": s.section_code,
                "section_name": s.section_name,
                "registry_id": s.registry_id,
                "confidence_score": s.confidence_score,
                "content_text": s.content_text[:_CONTENT_DISPLAY_CHARS],
            }
            for s in filtered
        ]

        logger.info(
            "RagIndex: built index with %d sections (%d dim)",
            len(self._metadata),
            dim,
        )
        return self.stats

    def add_sections(
        self,
        sections: list["IchSection"],
    ) -> int:
        """Incrementally add new high-confidence sections.

        Args:
            sections: New IchSection instances to consider.

        Returns:
            Count of sections actually added (after filtering).
        """
        _ensure_rag_deps()

        if self._index is None:
            raise RuntimeError(
                "Index not initialised. Call build_from_sections() "
                "or load() first."
            )

        filtered = self._filter_sections(sections)
        if not filtered:
            return 0

        texts = [s.content_text[:self._max_text_chars] for s in filtered]
        embeddings = self._encode(texts)
        self._index.add(embeddings)

        for s in filtered:
            self._metadata.append({
                "section_code": s.section_code,
                "section_name": s.section_name,
                "registry_id": s.registry_id,
                "confidence_score": s.confidence_score,
                "content_text": s.content_text[:_CONTENT_DISPLAY_CHARS],
            })

        logger.info(
            "RagIndex: added %d sections (total: %d)",
            len(filtered),
            len(self._metadata),
        )
        return len(filtered)

    def query(
        self,
        text: str,
        top_k: int = 3,
    ) -> list[RagExemplar]:
        """Retrieve top-k most similar sections from the index.

        Args:
            text: Query text (section being classified).
            top_k: Number of results to return.

        Returns:
            List of RagExemplar ordered by similarity descending.
            Empty list if index is empty or unavailable.
        """
        if self.is_empty:
            return []

        _ensure_rag_deps()

        query_embedding = self._encode([text[:self._max_text_chars]])
        k = min(top_k, len(self._metadata))
        scores, indices = self._index.search(query_embedding, k)

        results: list[RagExemplar] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append(
                RagExemplar(
                    section_code=meta["section_code"],
                    section_name=meta["section_name"],
                    registry_id=meta["registry_id"],
                    confidence_score=meta["confidence_score"],
                    content_text=meta["content_text"],
                    similarity_score=float(score),
                )
            )

        return results

    def save(self) -> None:
        """Persist index and metadata to index_dir."""
        _, faiss = _ensure_rag_deps()

        if self._index is None:
            raise RuntimeError("No index to save.")

        self._index_dir.mkdir(parents=True, exist_ok=True)

        index_path = self._index_dir / _INDEX_FILENAME
        metadata_path = self._index_dir / _METADATA_FILENAME

        faiss.write_index(self._index, str(index_path))

        with open(metadata_path, "w", encoding="utf-8") as f:
            for meta in self._metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        logger.info(
            "RagIndex: saved %d sections to %s",
            len(self._metadata),
            self._index_dir,
        )

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        embedding_model: str = _DEFAULT_MODEL,
    ) -> "RagIndex":
        """Load a persisted index from disk.

        Args:
            index_dir: Directory containing index files.
            embedding_model: Model name (must match what built the index).

        Returns:
            RagIndex with loaded state.

        Raises:
            FileNotFoundError: If index files don't exist.
        """
        _, faiss = _ensure_rag_deps()

        index_dir = Path(index_dir)
        index_path = index_dir / _INDEX_FILENAME
        metadata_path = index_dir / _METADATA_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Index metadata not found at {metadata_path}"
            )

        instance = cls(
            index_dir=index_dir,
            embedding_model=embedding_model,
        )
        instance._index = faiss.read_index(str(index_path))

        instance._metadata = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    instance._metadata.append(json.loads(line))

        logger.info(
            "RagIndex: loaded %d sections from %s",
            len(instance._metadata),
            index_dir,
        )
        return instance

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        """True if the index has zero vectors."""
        return self._index is None or self._index.ntotal == 0

    @property
    def stats(self) -> RagIndexStats:
        """Current index statistics."""
        code_counts: dict[str, int] = {}
        registry_ids: set[str] = set()
        for meta in self._metadata:
            code = meta["section_code"]
            code_counts[code] = code_counts.get(code, 0) + 1
            registry_ids.add(meta["registry_id"])

        dim = 0
        if self._index is not None:
            dim = self._index.d

        return RagIndexStats(
            total_sections=len(self._metadata),
            section_code_counts=code_counts,
            registry_ids=registry_ids,
            embedding_model=self._embedding_model_name,
            embedding_dim=dim,
        )

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _filter_sections(
        self,
        sections: list["IchSection"],
    ) -> list["IchSection"]:
        """Filter sections by min_confidence and non-empty text."""
        return [
            s
            for s in sections
            if s.confidence_score >= self._min_confidence
            and getattr(s, "content_text", None)
            and s.content_text.strip()
        ]

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model.

        PTCV-234: If the fine-tuned model exists at
        ``_FINETUNED_MODEL_DIR`` and no explicit model override was
        provided, automatically use it instead of the default.
        """
        if self._model is None:
            st, _ = _ensure_rag_deps()

            model_name = self._embedding_model_name

            # PTCV-234: Auto-detect fine-tuned model
            if (
                model_name == _DEFAULT_MODEL
                and _FINETUNED_MODEL_DIR.exists()
                and (_FINETUNED_MODEL_DIR / "config.json").exists()
            ):
                model_name = str(_FINETUNED_MODEL_DIR)
                logger.info(
                    "RagIndex: using fine-tuned model from %s",
                    model_name,
                )

            self._model = st.SentenceTransformer(model_name)
            logger.info(
                "RagIndex: loaded embedding model %s",
                model_name,
            )
        return self._model

    def _encode(self, texts: list[str]) -> "np.ndarray":
        """Encode texts and L2-normalise for cosine similarity."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Normalise to unit vectors so IP == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms
        return embeddings.astype(np.float32)

    def _get_embedding_dim(self) -> int:
        """Get embedding dimensionality from the model."""
        model = self._get_model()
        dim = model.get_sentence_embedding_dimension()
        return int(dim)


# ---------------------------------------------------------------------------
# Utility: build from parquet files
# ---------------------------------------------------------------------------


def build_index_from_parquet_files(
    parquet_paths: list[Path],
    index_dir: Path,
    min_confidence: float = _MIN_CONFIDENCE,
    embedding_model: str = _DEFAULT_MODEL,
) -> RagIndex:
    """Build a RagIndex from multiple sections.parquet files.

    Reads each parquet file via ``parquet_to_sections()``, filters
    by ``min_confidence``, and builds a single unified index.

    Args:
        parquet_paths: Paths to sections.parquet files.
        index_dir: Directory to persist the index.
        min_confidence: Minimum confidence for inclusion.
        embedding_model: Embedding model name.

    Returns:
        Built and saved RagIndex.
    """
    from .parquet_writer import parquet_to_sections

    all_sections: list = []
    for path in parquet_paths:
        try:
            data = path.read_bytes()
            sections = parquet_to_sections(data)
            all_sections.extend(sections)
        except Exception:
            logger.warning(
                "Failed to read parquet %s; skipping",
                path,
                exc_info=True,
            )

    index = RagIndex(
        index_dir=index_dir,
        embedding_model=embedding_model,
        min_confidence=min_confidence,
    )
    index.build_from_sections(all_sections)
    index.save()

    logger.info(
        "Built RAG index: %d sections from %d parquet files -> %s",
        index.stats.total_sections,
        len(parquet_paths),
        index_dir,
    )
    return index
