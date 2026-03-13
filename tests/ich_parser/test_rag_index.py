"""Tests for RagIndex — PTCV-162.

All tests mock sentence_transformers and faiss to avoid real model
downloads.  Uses deterministic numpy arrays for embeddings.

GHERKIN coverage:
  Scenario 1: Prior classified sections provide context
  Scenario 3: RAG skipped when index empty
  Scenario 4: Only high-confidence prior classifications indexed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.rag_index import (
    RagExemplar,
    RagIndex,
    RagIndexStats,
    _CONTENT_DISPLAY_CHARS,
    _MIN_CONFIDENCE,
    _ensure_rag_deps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)
_DIM = 384


def _make_section(
    code: str = "B.4",
    name: str = "Trial Design",
    confidence: float = 0.90,
    text: str = "Study design randomisation blinding placebo",
    registry_id: str = "NCT00000001",
    run_id: str = "run-1",
) -> IchSection:
    return IchSection(
        run_id=run_id,
        source_run_id="src-1",
        source_sha256="abc123",
        registry_id=registry_id,
        section_code=code,
        section_name=name,
        content_json="{}",
        confidence_score=confidence,
        review_required=confidence < 0.70,
        legacy_format=False,
        content_text=text,
    )


def _make_sections(n: int = 10) -> list[IchSection]:
    """Generate n sections with varied codes and high confidence."""
    codes = ["B.1", "B.2", "B.3", "B.4", "B.5", "B.6", "B.7", "B.8"]
    names = [
        "General Information",
        "Trial Objectives",
        "Investigational Plan",
        "Trial Design",
        "Statistical Considerations",
        "Treatment",
        "Assessments",
        "Safety Reporting",
    ]
    sections = []
    for i in range(n):
        idx = i % len(codes)
        sections.append(
            _make_section(
                code=codes[idx],
                name=names[idx],
                confidence=0.85 + 0.01 * i,
                text=f"Section {codes[idx]} content for protocol {i}.",
                registry_id=f"NCT{i:08d}",
            )
        )
    return sections


# ---------------------------------------------------------------------------
# Mock FAISS index (in-memory numpy-based replacement)
# ---------------------------------------------------------------------------

class FakeIndex:
    """Minimal numpy-based FAISS IndexFlatIP substitute for testing."""

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
        # Pad if fewer than k
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


class FakeFaiss:
    """Mock faiss module using FakeIndex."""

    @staticmethod
    def IndexFlatIP(d: int) -> FakeIndex:
        return FakeIndex(d)

    @staticmethod
    def write_index(index: FakeIndex, path: str) -> None:
        """Write index vectors as numpy binary for roundtrip test."""
        if index._vectors:
            mat = np.stack(index._vectors)
        else:
            mat = np.empty((0, index.d), dtype=np.float32)
        # Save with .npy extension alongside the .bin path
        np.save(path + ".npy", mat)
        # Also create the .bin file so existence checks pass
        Path(path).write_bytes(b"FAKE")

    @staticmethod
    def read_index(path: str) -> FakeIndex:
        """Read index from numpy binary."""
        npy_path = path + ".npy"
        mat = np.load(npy_path)
        idx = FakeIndex(mat.shape[1] if mat.ndim == 2 else _DIM)
        if mat.size > 0:
            idx.add(mat)
        return idx


class FakeModel:
    """Mock SentenceTransformer that returns deterministic embeddings."""

    def __init__(self, model_name: str):
        self._name = model_name

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        # Generate deterministic embeddings based on text hash
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
def mock_rag_deps():
    """Inject fake faiss and sentence_transformers for all tests."""
    import ptcv.ich_parser.rag_index as mod

    orig_st = mod._sentence_transformers
    orig_fa = mod._faiss

    mod._sentence_transformers = FakeSentenceTransformers()
    mod._faiss = FakeFaiss()

    yield

    mod._sentence_transformers = orig_st
    mod._faiss = orig_fa


# ---------------------------------------------------------------------------
# TestRagExemplar
# ---------------------------------------------------------------------------

class TestRagExemplar:
    """Frozen dataclass validation."""

    def test_creation(self) -> None:
        ex = RagExemplar(
            section_code="B.5",
            section_name="Statistical Considerations",
            registry_id="NCT00000001",
            confidence_score=0.92,
            content_text="Sample size calculation",
            similarity_score=0.87,
        )
        assert ex.section_code == "B.5"
        assert ex.similarity_score == 0.87

    def test_frozen(self) -> None:
        ex = RagExemplar(
            section_code="B.5",
            section_name="Stats",
            registry_id="NCT1",
            confidence_score=0.9,
            content_text="text",
            similarity_score=0.8,
        )
        with pytest.raises(AttributeError):
            ex.section_code = "B.1"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRagIndexBuild
# ---------------------------------------------------------------------------

class TestRagIndexBuild:
    """Building the index from IchSection lists."""

    def test_build_filters_low_confidence(self, tmp_path: Path) -> None:
        """Sections below min_confidence are excluded."""
        sections = [
            _make_section(confidence=0.60),  # excluded
            _make_section(code="B.5", confidence=0.85),  # included
            _make_section(code="B.6", confidence=0.90),  # included
        ]
        idx = RagIndex(index_dir=tmp_path)
        stats = idx.build_from_sections(sections)
        assert stats.total_sections == 2

    def test_build_excludes_empty_text(self, tmp_path: Path) -> None:
        """Sections with empty content_text are skipped."""
        sections = [
            _make_section(confidence=0.90, text=""),
            _make_section(confidence=0.90, text="   "),
            _make_section(code="B.5", confidence=0.90, text="real text"),
        ]
        idx = RagIndex(index_dir=tmp_path)
        stats = idx.build_from_sections(sections)
        assert stats.total_sections == 1

    def test_build_from_empty_list(self, tmp_path: Path) -> None:
        """Empty input produces empty index."""
        idx = RagIndex(index_dir=tmp_path)
        stats = idx.build_from_sections([])
        assert stats.total_sections == 0
        assert idx.is_empty

    def test_stats_section_code_counts(self, tmp_path: Path) -> None:
        """Stats correctly count sections per code."""
        sections = _make_sections(10)
        idx = RagIndex(index_dir=tmp_path)
        stats = idx.build_from_sections(sections)
        assert stats.total_sections == 10
        # With 8 codes and 10 sections, B.1 and B.2 have 2 each
        assert stats.section_code_counts["B.1"] == 2
        assert stats.section_code_counts["B.2"] == 2
        assert stats.section_code_counts["B.3"] == 1

    def test_stats_registry_ids(self, tmp_path: Path) -> None:
        """Stats contain unique registry IDs."""
        sections = _make_sections(5)
        idx = RagIndex(index_dir=tmp_path)
        stats = idx.build_from_sections(sections)
        assert len(stats.registry_ids) == 5

    def test_stats_embedding_model(self, tmp_path: Path) -> None:
        """Stats report the model name and dimension."""
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections(_make_sections(3))
        stats = idx.stats
        assert stats.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert stats.embedding_dim == _DIM


# ---------------------------------------------------------------------------
# TestRagIndexQuery
# ---------------------------------------------------------------------------

class TestRagIndexQuery:
    """Querying the RAG index."""

    def test_query_returns_top_k(self, tmp_path: Path) -> None:
        """Given 10 indexed sections, query returns exactly 3."""
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections(_make_sections(10))
        results = idx.query("trial design randomisation", top_k=3)
        assert len(results) == 3

    def test_query_returns_rag_exemplars_with_metadata(
        self, tmp_path: Path,
    ) -> None:
        """All RagExemplar fields are populated."""
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections(_make_sections(5))
        results = idx.query("test query text")
        assert len(results) > 0
        for ex in results:
            assert isinstance(ex, RagExemplar)
            assert ex.section_code.startswith("B.")
            assert ex.section_name
            assert ex.registry_id.startswith("NCT")
            assert ex.confidence_score >= _MIN_CONFIDENCE
            assert isinstance(ex.similarity_score, float)

    def test_query_on_empty_index_returns_empty(
        self, tmp_path: Path,
    ) -> None:
        """When is_empty=True, query returns []."""
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections([])
        results = idx.query("anything")
        assert results == []

    def test_query_similarity_scores_descending(
        self, tmp_path: Path,
    ) -> None:
        """Results ordered by similarity descending."""
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections(_make_sections(10))
        results = idx.query("trial design", top_k=5)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_with_fewer_sections_than_k(
        self, tmp_path: Path,
    ) -> None:
        """When index has 2 sections and top_k=5, returns 2."""
        sections = _make_sections(2)
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections(sections)
        results = idx.query("test", top_k=5)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# TestRagIndexPersistence
# ---------------------------------------------------------------------------

class TestRagIndexPersistence:
    """Save and load roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Build, save, load, query produces consistent results."""
        index_dir = tmp_path / "rag_index"

        # Build and save
        idx1 = RagIndex(index_dir=index_dir)
        idx1.build_from_sections(_make_sections(5))
        idx1.save()

        # Load and query
        idx2 = RagIndex.load(index_dir)
        assert idx2.stats.total_sections == 5
        results = idx2.query("trial design")
        assert len(results) > 0

    def test_load_nonexistent_raises_file_not_found(
        self, tmp_path: Path,
    ) -> None:
        """FileNotFoundError when path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            RagIndex.load(tmp_path / "nonexistent")

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Save creates index_dir if it doesn't exist."""
        index_dir = tmp_path / "nested" / "rag"
        idx = RagIndex(index_dir=index_dir)
        idx.build_from_sections(_make_sections(3))
        idx.save()
        assert (index_dir / "index_metadata.jsonl").exists()

    def test_metadata_is_valid_jsonl(self, tmp_path: Path) -> None:
        """Metadata file contains valid JSONL."""
        index_dir = tmp_path / "rag"
        idx = RagIndex(index_dir=index_dir)
        idx.build_from_sections(_make_sections(3))
        idx.save()

        meta_path = index_dir / "index_metadata.jsonl"
        with open(meta_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            assert "section_code" in data
            assert "registry_id" in data
            assert "confidence_score" in data


# ---------------------------------------------------------------------------
# TestRagIndexIncremental
# ---------------------------------------------------------------------------

class TestRagIndexIncremental:
    """Incremental section additions."""

    def test_add_sections_increments_count(
        self, tmp_path: Path,
    ) -> None:
        """Build with 5, add 3, total is 8."""
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections(_make_sections(5))
        assert idx.stats.total_sections == 5

        new_sections = [
            _make_section(
                code="B.1",
                confidence=0.95,
                text=f"New section {i}",
                registry_id=f"NCT9999{i:04d}",
            )
            for i in range(3)
        ]
        added = idx.add_sections(new_sections)
        assert added == 3
        assert idx.stats.total_sections == 8

    def test_add_sections_filters_low_confidence(
        self, tmp_path: Path,
    ) -> None:
        """Only high-confidence sections are added."""
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections(_make_sections(3))

        new_sections = [
            _make_section(confidence=0.45, text="Low conf"),
            _make_section(confidence=0.90, text="High conf",
                          code="B.7"),
        ]
        added = idx.add_sections(new_sections)
        assert added == 1
        assert idx.stats.total_sections == 4

    def test_add_to_uninitialised_raises(
        self, tmp_path: Path,
    ) -> None:
        """add_sections before build/load raises RuntimeError."""
        idx = RagIndex(index_dir=tmp_path)
        with pytest.raises(RuntimeError, match="not initialised"):
            idx.add_sections([_make_section()])


# ---------------------------------------------------------------------------
# TestEnsureRagDeps
# ---------------------------------------------------------------------------

class TestEnsureRagDeps:
    """Dependency import error handling."""

    def test_import_error_message(self) -> None:
        """When deps missing, ImportError mentions the packages."""
        import ptcv.ich_parser.rag_index as mod

        # Temporarily clear cached modules
        orig_st = mod._sentence_transformers
        orig_fa = mod._faiss
        mod._sentence_transformers = None
        mod._faiss = None

        try:
            with patch.dict(
                sys.modules,
                {"sentence_transformers": None, "faiss": None},
            ):
                with pytest.raises(
                    ImportError,
                    match="sentence-transformers",
                ):
                    _ensure_rag_deps()
        finally:
            mod._sentence_transformers = orig_st
            mod._faiss = orig_fa


# ---------------------------------------------------------------------------
# TestContentTruncation
# ---------------------------------------------------------------------------

class TestContentTruncation:
    """Content text is truncated for display."""

    def test_content_text_truncated_in_metadata(
        self, tmp_path: Path,
    ) -> None:
        """Indexed content_text is truncated to display limit."""
        long_text = "A" * 2000
        section = _make_section(confidence=0.95, text=long_text)
        idx = RagIndex(index_dir=tmp_path)
        idx.build_from_sections([section])
        results = idx.query("A" * 100)
        assert len(results) == 1
        assert len(results[0].content_text) == _CONTENT_DISPLAY_CHARS
