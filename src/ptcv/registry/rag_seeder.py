"""Registry Metadata RAG Seeder (PTCV-196).

Seeds the FAISS RAG index with structured metadata from
ClinicalTrials.gov, providing anchor vectors for downstream
ICH section classification.  Registry vectors carry
``source="registry"`` in their metadata so consumers can
filter or weight by provenance.

Operates on the existing ``RagIndex`` from
``ptcv.ich_parser.rag_index`` — no new index infrastructure needed.

Risk tier: LOW — embedding of publicly available registry text.

Regulatory references:
- ALCOA+ Traceable: each vector tracks source, nct_id, quality_rating.
- ALCOA+ Attributable: source="registry" distinguishes from PDF vectors.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional

from .ich_mapper import MappedRegistrySection

logger = logging.getLogger(__name__)

_CONTENT_DISPLAY_CHARS = 500
_MAX_TEXT_CHARS = 2000
_CONFIDENCE_BOOST = 0.15


@dataclasses.dataclass
class SeedResult:
    """Result of a registry seeding operation.

    Attributes:
        sections_seeded: Number of vectors added to the index.
        sections_skipped: Number skipped (empty content or duplicate).
        nct_id: Trial registry ID that was seeded.
        section_codes: List of ICH section codes that were seeded.
    """

    sections_seeded: int
    sections_skipped: int
    nct_id: str
    section_codes: list[str]


class RegistryRagSeeder:
    """Seed a RagIndex with registry-derived ICH sections.

    Converts ``MappedRegistrySection`` objects into FAISS vectors
    with extended metadata (``source``, ``quality_rating``,
    ``nct_id``).  Handles deduplication to prevent duplicate
    registry entries on index rebuild.

    Args:
        confidence_boost: Amount to boost query confidence when a
            registry vector with quality_rating >= 0.6 is among
            the nearest neighbours.  Default 0.15 (15%).
    """

    def __init__(
        self,
        confidence_boost: float = _CONFIDENCE_BOOST,
    ) -> None:
        self._confidence_boost = confidence_boost

    def seed(
        self,
        rag_index: Any,
        mapped_sections: list[MappedRegistrySection],
        nct_id: str,
    ) -> SeedResult:
        """Add registry sections to the RAG index as seed vectors.

        Each section is embedded and added with extended metadata
        including ``source="registry"``.  Sections already present
        in the index (same nct_id + section_code + source=registry)
        are skipped to prevent duplication on rebuild.

        Args:
            rag_index: A ``RagIndex`` instance (must be initialised).
            mapped_sections: Output from ``MetadataToIchMapper.map()``.
            nct_id: ClinicalTrials.gov NCT ID for provenance.

        Returns:
            SeedResult with counts and seeded codes.
        """
        from ptcv.ich_parser.rag_index import _ensure_rag_deps

        _ensure_rag_deps()

        if rag_index._index is None:
            raise RuntimeError(
                "RagIndex not initialised. Call build_from_sections() "
                "or load() first."
            )

        # Filter out sections already seeded for this trial
        existing_keys = self._existing_registry_keys(
            rag_index, nct_id
        )

        to_seed: list[MappedRegistrySection] = []
        skipped = 0
        for sec in mapped_sections:
            if not sec.content_text.strip():
                skipped += 1
                continue
            key = (nct_id, sec.section_code)
            if key in existing_keys:
                skipped += 1
                continue
            to_seed.append(sec)

        if not to_seed:
            logger.info(
                "RegistryRagSeeder: nothing to seed for %s "
                "(%d skipped)",
                nct_id,
                skipped,
            )
            return SeedResult(
                sections_seeded=0,
                sections_skipped=skipped,
                nct_id=nct_id,
                section_codes=[],
            )

        # Embed and add to index
        texts = [
            s.content_text[:_MAX_TEXT_CHARS] for s in to_seed
        ]
        embeddings = rag_index._encode(texts)
        rag_index._index.add(embeddings)

        # Append metadata with registry-specific fields
        for sec in to_seed:
            rag_index._metadata.append({
                "section_code": sec.section_code,
                "section_name": sec.section_name,
                "registry_id": nct_id,
                "confidence_score": sec.quality_rating,
                "content_text": sec.content_text[
                    :_CONTENT_DISPLAY_CHARS
                ],
                "source": "registry",
                "quality_rating": sec.quality_rating,
                "nct_id": nct_id,
                "content_preview": sec.content_text[:200],
            })

        seeded_codes = [s.section_code for s in to_seed]
        logger.info(
            "RegistryRagSeeder: seeded %d sections for %s (%s)",
            len(to_seed),
            nct_id,
            ", ".join(seeded_codes),
        )

        return SeedResult(
            sections_seeded=len(to_seed),
            sections_skipped=skipped,
            nct_id=nct_id,
            section_codes=seeded_codes,
        )

    def remove_registry_vectors(
        self,
        rag_index: Any,
        nct_id: Optional[str] = None,
    ) -> int:
        """Remove registry-sourced vectors from the index.

        Because FAISS ``IndexFlatIP`` does not support deletion,
        this rebuilds the index without the registry vectors.  Used
        before re-seeding to prevent duplication on full rebuild.

        Args:
            rag_index: A ``RagIndex`` instance.
            nct_id: If given, remove only vectors for this trial.
                If None, remove all registry vectors.

        Returns:
            Number of vectors removed.
        """
        from ptcv.ich_parser.rag_index import _ensure_rag_deps

        _, faiss = _ensure_rag_deps()

        if rag_index._index is None or not rag_index._metadata:
            return 0

        keep_indices: list[int] = []
        remove_count = 0

        for i, meta in enumerate(rag_index._metadata):
            is_registry = meta.get("source") == "registry"
            matches_nct = nct_id is None or meta.get(
                "nct_id"
            ) == nct_id
            if is_registry and matches_nct:
                remove_count += 1
            else:
                keep_indices.append(i)

        if remove_count == 0:
            return 0

        # Rebuild index with kept vectors only
        import numpy as np

        total = rag_index._index.ntotal
        if total == 0:
            return 0

        dim = rag_index._index.d
        all_vectors = np.zeros((total, dim), dtype=np.float32)
        for i in range(total):
            all_vectors[i] = rag_index._index.reconstruct(i)

        kept_vectors = all_vectors[keep_indices]
        new_index = faiss.IndexFlatIP(dim)
        if len(kept_vectors) > 0:
            new_index.add(kept_vectors)

        rag_index._index = new_index
        rag_index._metadata = [
            rag_index._metadata[i] for i in keep_indices
        ]

        logger.info(
            "RegistryRagSeeder: removed %d registry vectors "
            "(nct_id=%s)",
            remove_count,
            nct_id or "all",
        )
        return remove_count

    @staticmethod
    def apply_confidence_boost(
        base_confidence: float,
        exemplars: list[Any],
        boost: float = _CONFIDENCE_BOOST,
    ) -> float:
        """Boost classification confidence if registry exemplars match.

        When nearest-neighbour results include registry vectors with
        ``quality_rating >= 0.6``, the classification confidence is
        boosted by ``boost`` (default 15%).

        Args:
            base_confidence: Original classification confidence.
            exemplars: List of ``RagExemplar`` from a query.
            boost: Confidence increment (0.0–1.0).

        Returns:
            Adjusted confidence, capped at 1.0.
        """
        for ex in exemplars:
            meta = getattr(ex, "_metadata", None)
            # Check if any exemplar's metadata indicates registry
            # source with sufficient quality
            if meta and meta.get("source") == "registry":
                qr = meta.get("quality_rating", 0.0)
                if qr >= 0.6:
                    return min(1.0, base_confidence + boost)

        return base_confidence

    @staticmethod
    def _existing_registry_keys(
        rag_index: Any,
        nct_id: str,
    ) -> set[tuple[str, str]]:
        """Return (nct_id, section_code) pairs already in the index."""
        keys: set[tuple[str, str]] = set()
        for meta in rag_index._metadata:
            if (
                meta.get("source") == "registry"
                and meta.get("nct_id") == nct_id
            ):
                keys.add((nct_id, meta["section_code"]))
        return keys

    @staticmethod
    def get_registry_vectors(
        rag_index: Any,
        nct_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return metadata for all registry vectors in the index.

        Args:
            rag_index: A ``RagIndex`` instance.
            nct_id: If given, filter to this trial only.

        Returns:
            List of metadata dicts with source="registry".
        """
        results: list[dict[str, Any]] = []
        for meta in rag_index._metadata:
            if meta.get("source") != "registry":
                continue
            if nct_id and meta.get("nct_id") != nct_id:
                continue
            results.append(meta)
        return results
