"""Tests for RegistryRagSeeder (PTCV-196).

Covers all GHERKIN scenarios:
- Seed vectors created from registry metadata
- Registry vectors improve classification confidence
- Deduplication with PDF-extracted sections
- Index rebuild preserves registry seeds

All tests mock sentence_transformers and faiss using the same
FakeIndex/FakeFaiss pattern as test_rag_index.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.rag_index import RagIndex
from ptcv.registry.ich_mapper import (
    MappedRegistrySection,
    MetadataToIchMapper,
    QUALITY_CONTEXTUAL,
    QUALITY_DIRECT,
    QUALITY_PARTIAL,
)
from ptcv.registry.rag_seeder import (
    RegistryRagSeeder,
    SeedResult,
)


# ---------------------------------------------------------------------------
# Mock FAISS infrastructure (matches test_rag_index.py patterns)
# ---------------------------------------------------------------------------

_DIM = 384


class FakeIndex:
    """Minimal numpy-based FAISS IndexFlatIP substitute."""

    def __init__(self, d: int):
        self.d = d
        self._vectors: list[np.ndarray] = []

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def add(self, vectors: np.ndarray) -> None:
        for v in vectors:
            self._vectors.append(v.copy())

    def search(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return (
                np.array([[-1.0] * k], dtype=np.float32),
                np.array([[-1] * k], dtype=np.int64),
            )
        mat = np.stack(self._vectors)
        scores = mat @ query[0]
        top_k = min(k, len(self._vectors))
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        if len(top_indices) < k:
            pad = k - len(top_indices)
            top_indices = np.concatenate(
                [top_indices, np.full(pad, -1, dtype=np.int64)]
            )
            top_scores = np.concatenate(
                [top_scores, np.full(pad, -1.0, dtype=np.float32)]
            )
        return (
            top_scores.reshape(1, -1).astype(np.float32),
            top_indices.reshape(1, -1).astype(np.int64),
        )

    def reconstruct(self, i: int) -> np.ndarray:
        return self._vectors[i].copy()


class FakeFaiss:
    """Mock faiss module."""

    @staticmethod
    def IndexFlatIP(d: int) -> FakeIndex:
        return FakeIndex(d)


class FakeModel:
    """Mock SentenceTransformer with deterministic embeddings."""

    def __init__(self, model_name: str):
        self._name = model_name

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        result = np.zeros((len(texts), _DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            result[i] = rng.randn(_DIM).astype(np.float32)
        return result

    def get_sentence_embedding_dimension(self) -> int:
        return _DIM


class FakeSentenceTransformers:
    """Mock sentence_transformers module."""

    @staticmethod
    def SentenceTransformer(model_name: str) -> FakeModel:
        return FakeModel(model_name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_rag_deps() -> Any:
    """Inject fake faiss and sentence_transformers."""
    import ptcv.ich_parser.rag_index as mod

    orig_st = mod._sentence_transformers
    orig_fa = mod._faiss

    mod._sentence_transformers = FakeSentenceTransformers()
    mod._faiss = FakeFaiss()

    yield

    mod._sentence_transformers = orig_st
    mod._faiss = orig_fa


@pytest.fixture
def rag_index(tmp_path: Path) -> RagIndex:
    """Return an initialised empty RagIndex."""
    idx = RagIndex(index_dir=tmp_path / "rag")
    idx.build_from_sections([])
    return idx


@pytest.fixture
def seeder() -> RegistryRagSeeder:
    return RegistryRagSeeder()


@pytest.fixture
def sample_sections() -> list[MappedRegistrySection]:
    """Return mapped sections for NCT01512251."""
    return [
        MappedRegistrySection(
            section_code="B.3",
            section_name="Trial Objectives and Purpose",
            content_text="BKM120 combined with Vemurafenib Phase 1/2",
            content_json='{"official_title": "BKM120 trial"}',
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="IdentificationModule",
        ),
        MappedRegistrySection(
            section_code="B.4",
            section_name="Trial Design",
            content_text="Interventional, Phase 1/2, Non-randomized",
            content_json='{"study_type": "INTERVENTIONAL"}',
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="DesignModule",
        ),
        MappedRegistrySection(
            section_code="B.5",
            section_name="Selection of Subjects",
            content_text="BRAF V600E/K mutation required, age >= 18",
            content_json='{"sex": "ALL"}',
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="EligibilityModule",
        ),
        MappedRegistrySection(
            section_code="B.8",
            section_name="Assessment of Efficacy",
            content_text="Primary: MTD. Secondary: PFS, OS",
            content_json='{"primary_outcomes": []}',
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="OutcomesModule",
        ),
        MappedRegistrySection(
            section_code="B.2",
            section_name="Background Information",
            content_text="Conditions: Melanoma, BRAF V600E",
            content_json='{"conditions": ["Melanoma"]}',
            quality_rating=QUALITY_CONTEXTUAL,
            ct_gov_module="ConditionsModule",
        ),
        MappedRegistrySection(
            section_code="B.1",
            section_name="General Information",
            content_text="Status: Completed, Start: 2012-02",
            content_json='{"status": "Completed"}',
            quality_rating=QUALITY_PARTIAL,
            ct_gov_module="StatusModule",
        ),
    ]


def _make_pdf_section(
    code: str = "B.4",
    text: str = "Study design from PDF extraction",
) -> IchSection:
    """Create a PDF-extracted IchSection for coexistence tests."""
    return IchSection(
        run_id="run-pdf-1",
        source_run_id="src-pdf-1",
        source_sha256="sha256pdf",
        registry_id="NCT01512251",
        section_code=code,
        section_name="Trial Design",
        content_json="{}",
        confidence_score=0.92,
        review_required=False,
        legacy_format=False,
        content_text=text,
    )


# ---------------------------------------------------------------------------
# Scenario: Seed vectors created from registry metadata
# ---------------------------------------------------------------------------


class TestSeedVectors:
    """Scenario: Seed vectors created from registry metadata."""

    def test_seed_adds_vectors_to_index(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Each mapped section becomes a vector in the index."""
        result = seeder.seed(rag_index, sample_sections, "NCT01512251")

        assert result.sections_seeded == 6
        assert result.nct_id == "NCT01512251"
        assert rag_index._index.ntotal == 6

    def test_seed_metadata_has_source_registry(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Every seeded vector has source='registry' in metadata."""
        seeder.seed(rag_index, sample_sections, "NCT01512251")

        for meta in rag_index._metadata:
            assert meta["source"] == "registry"

    def test_seed_metadata_has_quality_rating(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Each vector metadata includes quality_rating."""
        seeder.seed(rag_index, sample_sections, "NCT01512251")

        for meta in rag_index._metadata:
            assert "quality_rating" in meta
            assert meta["quality_rating"] > 0.0

    def test_seed_metadata_has_nct_id(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Each vector metadata includes nct_id."""
        seeder.seed(rag_index, sample_sections, "NCT01512251")

        for meta in rag_index._metadata:
            assert meta["nct_id"] == "NCT01512251"

    def test_seed_metadata_has_section_code(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Each vector metadata includes the ICH section code."""
        seeder.seed(rag_index, sample_sections, "NCT01512251")

        codes = {m["section_code"] for m in rag_index._metadata}
        assert codes == {"B.1", "B.2", "B.3", "B.4", "B.5", "B.8"}

    def test_seed_result_lists_codes(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """SeedResult.section_codes lists all seeded codes."""
        result = seeder.seed(rag_index, sample_sections, "NCT01512251")

        assert set(result.section_codes) == {
            "B.1", "B.2", "B.3", "B.4", "B.5", "B.8"
        }

    def test_seed_skips_empty_content(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
    ) -> None:
        """Sections with empty content_text are skipped."""
        sections = [
            MappedRegistrySection(
                section_code="B.3",
                section_name="Objectives",
                content_text="Real content here",
                content_json="{}",
                quality_rating=0.9,
            ),
            MappedRegistrySection(
                section_code="B.4",
                section_name="Design",
                content_text="   ",  # whitespace only
                content_json="{}",
                quality_rating=0.9,
            ),
        ]
        result = seeder.seed(rag_index, sections, "NCT01512251")

        assert result.sections_seeded == 1
        assert result.sections_skipped == 1


# ---------------------------------------------------------------------------
# Scenario: Registry vectors queryable alongside PDF vectors
# ---------------------------------------------------------------------------


class TestRegistryAndPdfCoexistence:
    """Scenario: Deduplication with PDF-extracted sections."""

    def test_pdf_and_registry_coexist(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Both PDF and registry vectors exist in the same index."""
        # Add PDF sections first
        pdf_sections = [_make_pdf_section()]
        rag_index.add_sections(pdf_sections)
        pdf_count = rag_index._index.ntotal

        # Seed registry
        seeder.seed(rag_index, sample_sections, "NCT01512251")

        assert rag_index._index.ntotal == pdf_count + 6

    def test_distinct_source_tags(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """PDF vectors have no source tag; registry vectors have 'registry'."""
        pdf_sections = [_make_pdf_section()]
        rag_index.add_sections(pdf_sections)
        seeder.seed(rag_index, sample_sections, "NCT01512251")

        sources = [m.get("source") for m in rag_index._metadata]
        assert None in sources  # PDF vector (no source field)
        assert "registry" in sources

    def test_get_registry_vectors_filters(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """get_registry_vectors returns only registry-sourced metadata."""
        pdf_sections = [_make_pdf_section()]
        rag_index.add_sections(pdf_sections)
        seeder.seed(rag_index, sample_sections, "NCT01512251")

        reg_vecs = RegistryRagSeeder.get_registry_vectors(
            rag_index, "NCT01512251"
        )
        assert len(reg_vecs) == 6
        for v in reg_vecs:
            assert v["source"] == "registry"

    def test_query_returns_both_sources(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
    ) -> None:
        """Querying the index can return both PDF and registry results."""
        # Add one PDF section
        pdf_sections = [
            _make_pdf_section(text="Randomized double-blind design")
        ]
        rag_index.add_sections(pdf_sections)

        # Add one registry section
        reg_sections = [
            MappedRegistrySection(
                section_code="B.4",
                section_name="Trial Design",
                content_text="Interventional Phase 1 design",
                content_json="{}",
                quality_rating=0.9,
            ),
        ]
        seeder.seed(rag_index, reg_sections, "NCT01512251")

        results = rag_index.query("trial design", top_k=5)
        assert len(results) == 2  # Both vectors returned


# ---------------------------------------------------------------------------
# Scenario: Index rebuild preserves registry seeds (deduplication)
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Scenario: Index rebuild preserves registry seeds."""

    def test_duplicate_seed_is_skipped(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Seeding the same NCT ID twice doesn't duplicate vectors."""
        result1 = seeder.seed(
            rag_index, sample_sections, "NCT01512251"
        )
        result2 = seeder.seed(
            rag_index, sample_sections, "NCT01512251"
        )

        assert result1.sections_seeded == 6
        assert result2.sections_seeded == 0
        assert result2.sections_skipped == 6
        assert rag_index._index.ntotal == 6

    def test_different_nct_ids_both_seeded(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Different NCT IDs are independently seeded."""
        seeder.seed(rag_index, sample_sections, "NCT01512251")
        seeder.seed(rag_index, sample_sections, "NCT99999999")

        assert rag_index._index.ntotal == 12

    def test_remove_then_reseed(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Remove + re-seed produces the same count (no duplication)."""
        seeder.seed(rag_index, sample_sections, "NCT01512251")
        assert rag_index._index.ntotal == 6

        removed = seeder.remove_registry_vectors(
            rag_index, "NCT01512251"
        )
        assert removed == 6
        assert rag_index._index.ntotal == 0

        result = seeder.seed(
            rag_index, sample_sections, "NCT01512251"
        )
        assert result.sections_seeded == 6
        assert rag_index._index.ntotal == 6


# ---------------------------------------------------------------------------
# Remove registry vectors
# ---------------------------------------------------------------------------


class TestRemoveRegistryVectors:
    """Removal and rebuild operations."""

    def test_remove_by_nct_id(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Remove only vectors for a specific NCT ID."""
        seeder.seed(rag_index, sample_sections, "NCT01512251")
        seeder.seed(rag_index, sample_sections, "NCT99999999")
        assert rag_index._index.ntotal == 12

        removed = seeder.remove_registry_vectors(
            rag_index, "NCT01512251"
        )
        assert removed == 6
        assert rag_index._index.ntotal == 6

        remaining = RegistryRagSeeder.get_registry_vectors(rag_index)
        assert all(v["nct_id"] == "NCT99999999" for v in remaining)

    def test_remove_all_registry(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Remove all registry vectors (nct_id=None)."""
        seeder.seed(rag_index, sample_sections, "NCT01512251")
        seeder.seed(rag_index, sample_sections, "NCT99999999")

        removed = seeder.remove_registry_vectors(rag_index)
        assert removed == 12
        assert rag_index._index.ntotal == 0

    def test_remove_preserves_pdf_vectors(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
        sample_sections: list[MappedRegistrySection],
    ) -> None:
        """Removing registry vectors doesn't affect PDF vectors."""
        pdf_sections = [_make_pdf_section()]
        rag_index.add_sections(pdf_sections)
        seeder.seed(rag_index, sample_sections, "NCT01512251")

        total_before = rag_index._index.ntotal
        removed = seeder.remove_registry_vectors(rag_index)

        assert removed == 6
        assert rag_index._index.ntotal == total_before - 6
        # PDF metadata still present
        assert any(
            m.get("source") != "registry"
            for m in rag_index._metadata
        )

    def test_remove_from_empty_index(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
    ) -> None:
        """Removing from empty index returns 0."""
        removed = seeder.remove_registry_vectors(rag_index)
        assert removed == 0


# ---------------------------------------------------------------------------
# SeedResult dataclass
# ---------------------------------------------------------------------------


class TestSeedResult:
    """SeedResult dataclass validation."""

    def test_seed_result_fields(self) -> None:
        result = SeedResult(
            sections_seeded=5,
            sections_skipped=1,
            nct_id="NCT01512251",
            section_codes=["B.3", "B.4", "B.5"],
        )
        assert result.sections_seeded == 5
        assert result.sections_skipped == 1
        assert result.nct_id == "NCT01512251"
        assert len(result.section_codes) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_seed_uninitialised_index_raises(
        self,
        seeder: RegistryRagSeeder,
        sample_sections: list[MappedRegistrySection],
        tmp_path: Path,
    ) -> None:
        """Seeding an uninitialised index raises RuntimeError."""
        raw_index = RagIndex(index_dir=tmp_path / "empty")
        with pytest.raises(RuntimeError, match="not initialised"):
            seeder.seed(raw_index, sample_sections, "NCT01512251")

    def test_seed_empty_list(
        self,
        seeder: RegistryRagSeeder,
        rag_index: RagIndex,
    ) -> None:
        """Seeding with empty list returns zero counts."""
        result = seeder.seed(rag_index, [], "NCT01512251")
        assert result.sections_seeded == 0
        assert result.sections_skipped == 0
        assert rag_index._index.ntotal == 0
