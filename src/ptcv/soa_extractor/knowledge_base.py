"""FAISS verified SoA knowledge base (PTCV-219).

Stores human-verified SoA table structures as reference documents.
On new extractions, query the index for similar protocols and compare
assessment coverage against verified templates.

Reuses the FAISS + sentence-transformers pattern from
``ptcv.ich_parser.rag_index`` (PTCV-161).

Risk tier: LOW — read-only reference data, no patient data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_INDEX_FILENAME = "soa_faiss_index.bin"
_METADATA_FILENAME = "soa_index_metadata.jsonl"
_CORPUS_DIR_NAME = "corpus"
_MAX_EMBED_CHARS = 2000


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class VerifiedSoaEntry:
    """A human-verified SoA table stored in the knowledge base.

    Attributes:
        registry_id: NCT identifier.
        phase: Trial phase (e.g., "PHASE1/PHASE2").
        condition: Primary condition/disease.
        intervention_type: Type of intervention (e.g., "DRUG").
        assessment_names: Ordered list of assessment/activity names.
        visit_headers: Ordered list of visit column names.
        activity_count: Number of assessments.
        visit_count: Number of visit columns.
        verification_method: How verification was done
            ("human_review", "vision_confirmed", "cross_validated").
        verified_utc: ISO timestamp of verification.
    """

    registry_id: str
    phase: str = ""
    condition: str = ""
    intervention_type: str = ""
    assessment_names: list[str] = field(default_factory=list)
    visit_headers: list[str] = field(default_factory=list)
    activity_count: int = 0
    visit_count: int = 0
    verification_method: str = ""
    verified_utc: str = ""

    def __post_init__(self) -> None:
        if not self.activity_count:
            self.activity_count = len(self.assessment_names)
        if not self.visit_count:
            self.visit_count = len(self.visit_headers)

    @property
    def embed_text(self) -> str:
        """Text to embed for FAISS similarity search.

        Combines protocol metadata + assessment names into a
        single string for embedding.
        """
        parts = []
        if self.phase:
            parts.append(f"Phase: {self.phase}")
        if self.condition:
            parts.append(f"Condition: {self.condition}")
        if self.intervention_type:
            parts.append(f"Intervention: {self.intervention_type}")
        if self.assessment_names:
            parts.append(
                "Assessments: " + ", ".join(self.assessment_names)
            )
        return ". ".join(parts)[:_MAX_EMBED_CHARS]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedSoaEntry:
        """Deserialize from dict."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })


@dataclass
class SimilarProtocol:
    """A nearest-neighbor match from the knowledge base.

    Attributes:
        entry: The matched verified SoA entry.
        similarity: Cosine similarity score (0.0–1.0).
        rank: 1-based rank in the result set.
    """

    entry: VerifiedSoaEntry
    similarity: float
    rank: int = 0


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


class SoaKnowledgeBase:
    """FAISS-backed knowledge base of verified SoA structures.

    Stores verified SoA tables and provides nearest-neighbor
    search for template matching and completeness checking.

    Uses the same FAISS + sentence-transformers pattern as
    ``RagIndex`` (PTCV-161):
    - ``faiss.IndexFlatIP`` for cosine similarity
    - L2-normalized embeddings
    - Parallel metadata list for retrieval

    Args:
        index_dir: Directory for FAISS index + metadata files.
        embedding_model: Sentence-transformers model name.
    """

    def __init__(
        self,
        index_dir: str | Path = "data/soa_knowledge_base",
        embedding_model: str = _DEFAULT_MODEL,
    ) -> None:
        self._index_dir = Path(index_dir)
        self._embedding_model_name = embedding_model
        self._entries: list[VerifiedSoaEntry] = []
        self._index: Any = None  # faiss.IndexFlatIP
        self._model: Any = None  # SentenceTransformer
        self._dim: int = 0

    @property
    def size(self) -> int:
        """Number of entries in the knowledge base."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._embedding_model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def _ensure_index(self) -> None:
        """Ensure FAISS index exists (create empty if needed)."""
        if self._index is not None:
            return
        import faiss
        if self._dim == 0:
            self._get_model()
        self._index = faiss.IndexFlatIP(self._dim)

    def _embed(self, text: str) -> Any:
        """Embed text and L2-normalize for cosine similarity."""
        import numpy as np
        model = self._get_model()
        vec = model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.reshape(1, -1).astype("float32")

    # ------------------------------------------------------------------
    # Add / Remove
    # ------------------------------------------------------------------

    def add(self, entry: VerifiedSoaEntry) -> None:
        """Add a verified SoA entry to the knowledge base.

        Embeds the entry and adds it to the FAISS index.
        Does NOT auto-save — call ``save()`` afterwards.

        Args:
            entry: Verified SoA entry to add.
        """
        self._ensure_index()
        vec = self._embed(entry.embed_text)
        self._index.add(vec)
        self._entries.append(entry)

    def add_batch(self, entries: list[VerifiedSoaEntry]) -> int:
        """Add multiple entries at once.

        Args:
            entries: List of verified entries to add.

        Returns:
            Number of entries added.
        """
        for entry in entries:
            self.add(entry)
        return len(entries)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 3,
    ) -> list[SimilarProtocol]:
        """Find the most similar verified protocols.

        Args:
            text: Query text (protocol metadata + assessment names).
            top_k: Number of results to return.

        Returns:
            List of SimilarProtocol matches, ranked by similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        vec = self._embed(text)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results: list[SimilarProtocol] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), 1
        ):
            if idx < 0 or idx >= len(self._entries):
                continue
            results.append(SimilarProtocol(
                entry=self._entries[idx],
                similarity=float(score),
                rank=rank,
            ))

        return results

    def query_by_entry(
        self,
        entry: VerifiedSoaEntry,
        top_k: int = 3,
    ) -> list[SimilarProtocol]:
        """Find similar protocols using an entry's embed text.

        Args:
            entry: Entry to find neighbors for.
            top_k: Number of results.

        Returns:
            List of SimilarProtocol matches.
        """
        return self.query(entry.embed_text, top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save index and metadata to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            import faiss
            faiss.write_index(
                self._index,
                str(self._index_dir / _INDEX_FILENAME),
            )

        meta_path = self._index_dir / _METADATA_FILENAME
        with open(meta_path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict()) + "\n")

    def load(self) -> bool:
        """Load index and metadata from disk.

        Returns:
            True if loaded successfully, False if files not found.
        """
        index_path = self._index_dir / _INDEX_FILENAME
        meta_path = self._index_dir / _METADATA_FILENAME

        if not index_path.exists() or not meta_path.exists():
            return False

        try:
            import faiss
            self._index = faiss.read_index(str(index_path))
            self._dim = self._index.d

            self._entries = []
            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._entries.append(
                            VerifiedSoaEntry.from_dict(json.loads(line))
                        )

            logger.info(
                "Loaded SoA knowledge base: %d entries",
                len(self._entries),
            )
            return True

        except Exception as exc:
            logger.warning(
                "Failed to load SoA knowledge base: %s", exc,
            )
            return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_all_entries(self) -> list[VerifiedSoaEntry]:
        """Get all entries in the knowledge base."""
        return list(self._entries)

    def get_entry_by_id(
        self, registry_id: str,
    ) -> Optional[VerifiedSoaEntry]:
        """Find an entry by registry ID."""
        for entry in self._entries:
            if entry.registry_id == registry_id:
                return entry
        return None

    def contains(self, registry_id: str) -> bool:
        """Check if a registry ID is already in the knowledge base."""
        return any(
            e.registry_id == registry_id for e in self._entries
        )


__all__ = [
    "SoaKnowledgeBase",
    "SimilarProtocol",
    "VerifiedSoaEntry",
]
