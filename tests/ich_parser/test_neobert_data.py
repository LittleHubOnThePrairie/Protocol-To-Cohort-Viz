"""Tests for NeoBERT data preparation — PTCV-164.

IQ/OQ tests for data loading, label mapping, splitting, and
bootstrap augmentation. No GPU required — pure CPU operations.

Qualification phase: IQ/OQ
Risk tier: MEDIUM
Regulatory: ALCOA+ Consistent (deterministic splits).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.neobert_config import LABEL_MAP, INV_LABEL_MAP, NUM_LABELS
from ptcv.ich_parser.neobert_data import (
    ClassificationSample,
    _map_section_code,
    bootstrap_from_markdown,
    dataset_stats,
    load_training_data,
    split_dataset,
)


# ---------------------------------------------------------------------------
# TestLabelMap
# ---------------------------------------------------------------------------


class TestLabelMap:
    """Label map covers all required ICH sections."""

    def test_all_14_mandatory_sections_mapped(self) -> None:
        for i in range(1, 15):
            code = f"B.{i}"
            assert code in LABEL_MAP, f"{code} missing from LABEL_MAP"

    def test_other_class_exists(self) -> None:
        assert "OTHER" in LABEL_MAP

    def test_num_labels_correct(self) -> None:
        assert NUM_LABELS == 15

    def test_inverse_map_roundtrip(self) -> None:
        for code, idx in LABEL_MAP.items():
            assert INV_LABEL_MAP[idx] == code

    def test_no_duplicate_indices(self) -> None:
        indices = list(LABEL_MAP.values())
        assert len(indices) == len(set(indices))

    def test_map_known_code(self) -> None:
        assert _map_section_code("B.3") == LABEL_MAP["B.3"]

    def test_map_unknown_code_to_other(self) -> None:
        assert _map_section_code("B.99") == LABEL_MAP["OTHER"]
        assert _map_section_code("X.1") == LABEL_MAP["OTHER"]


# ---------------------------------------------------------------------------
# TestLoadTrainingData
# ---------------------------------------------------------------------------


class TestLoadTrainingData:
    """Gold-standard corpus loading and filtering."""

    def test_loads_gold_standard_corpus(self) -> None:
        from ptcv.ich_parser.neobert_config import NeoBERTConfig

        config = NeoBERTConfig()
        samples = load_training_data(config)
        assert len(samples) > 0
        assert all(isinstance(s, ClassificationSample) for s in samples)

    def test_all_samples_have_required_fields(self) -> None:
        from ptcv.ich_parser.neobert_config import NeoBERTConfig

        config = NeoBERTConfig()
        samples = load_training_data(config)
        for s in samples:
            assert s.text
            assert 0 <= s.label < NUM_LABELS
            assert s.section_code
            assert s.registry_id
            assert s.source == "gold_standard"
            assert s.confidence >= config.min_confidence

    def test_filters_below_min_confidence(self) -> None:
        from ptcv.ich_parser.neobert_config import NeoBERTConfig

        config = NeoBERTConfig(min_confidence=0.99)
        samples = load_training_data(config)
        for s in samples:
            assert s.confidence >= 0.99

    def test_no_bootstrap_without_dir(self) -> None:
        from ptcv.ich_parser.neobert_config import NeoBERTConfig

        config = NeoBERTConfig()
        samples = load_training_data(config, bootstrap_md_dir=None)
        assert all(s.source == "gold_standard" for s in samples)


# ---------------------------------------------------------------------------
# TestBootstrapFromMarkdown
# ---------------------------------------------------------------------------


class TestBootstrapFromMarkdown:
    """Bootstrap augmentation from benchmark Markdown."""

    def test_bootstrap_from_benchmark_dir(self) -> None:
        md_dir = Path("C:/Dev/PTCV/data/analysis/benchmark_md")
        if not md_dir.exists():
            pytest.skip("Benchmark Markdown directory not available")

        samples = bootstrap_from_markdown(
            sorted(md_dir.glob("*/pymupdf4llm.md")),
            min_confidence=0.80,
        )
        assert len(samples) > 0
        assert all(s.source == "bootstrap" for s in samples)
        assert all(s.confidence >= 0.80 for s in samples)

    def test_bootstrap_empty_on_no_files(self) -> None:
        samples = bootstrap_from_markdown([], min_confidence=0.80)
        assert samples == []


# ---------------------------------------------------------------------------
# TestSplitDataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    """Stratified train/val/test splitting."""

    @pytest.fixture()
    def sample_data(self) -> list[ClassificationSample]:
        """Generate 50 samples across 5 classes."""
        samples = []
        for i in range(5):
            code = f"B.{i + 1}"
            for j in range(10):
                samples.append(
                    ClassificationSample(
                        text=f"Sample text for {code} example {j}",
                        label=LABEL_MAP[code],
                        section_code=code,
                        registry_id=f"NCT{j:08d}",
                        source="test",
                        confidence=1.0,
                    )
                )
        return samples

    def test_split_proportions(
        self, sample_data: list[ClassificationSample]
    ) -> None:
        train, val, test = split_dataset(sample_data, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_data)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_all_labels_in_train_split(
        self, sample_data: list[ClassificationSample]
    ) -> None:
        train, _, _ = split_dataset(sample_data, seed=42)
        train_codes = {s.section_code for s in train}
        all_codes = {s.section_code for s in sample_data}
        assert train_codes == all_codes

    def test_deterministic_with_same_seed(
        self, sample_data: list[ClassificationSample]
    ) -> None:
        t1, v1, te1 = split_dataset(sample_data, seed=42)
        t2, v2, te2 = split_dataset(sample_data, seed=42)
        assert [s.text for s in t1] == [s.text for s in t2]
        assert [s.text for s in v1] == [s.text for s in v2]
        assert [s.text for s in te1] == [s.text for s in te2]

    def test_different_seed_gives_different_split(
        self, sample_data: list[ClassificationSample]
    ) -> None:
        t1, _, _ = split_dataset(sample_data, seed=42)
        t2, _, _ = split_dataset(sample_data, seed=123)
        texts1 = [s.text for s in t1]
        texts2 = [s.text for s in t2]
        assert texts1 != texts2

    def test_no_sample_in_multiple_splits(
        self, sample_data: list[ClassificationSample]
    ) -> None:
        train, val, test = split_dataset(sample_data, seed=42)
        train_texts = {s.text for s in train}
        val_texts = {s.text for s in val}
        test_texts = {s.text for s in test}
        assert not train_texts & val_texts
        assert not train_texts & test_texts
        assert not val_texts & test_texts

    def test_small_class_all_in_train(self) -> None:
        """Classes with < 3 samples go entirely to training."""
        samples = [
            ClassificationSample(
                text=f"Small class sample {i}",
                label=LABEL_MAP["B.14"],
                section_code="B.14",
                registry_id="NCT00000001",
                source="test",
                confidence=1.0,
            )
            for i in range(2)
        ]
        train, val, test = split_dataset(samples, seed=42)
        assert len(train) == 2
        assert len(val) == 0
        assert len(test) == 0


# ---------------------------------------------------------------------------
# TestDatasetStats
# ---------------------------------------------------------------------------


class TestDatasetStats:
    """Dataset statistics computation."""

    def test_stats_correct(self) -> None:
        samples = [
            ClassificationSample(
                text="Hello world",
                label=0,
                section_code="B.1",
                registry_id="NCT1",
                source="gold_standard",
                confidence=1.0,
            ),
            ClassificationSample(
                text="Goodbye",
                label=1,
                section_code="B.2",
                registry_id="NCT2",
                source="bootstrap",
                confidence=0.9,
            ),
        ]
        stats = dataset_stats(samples)
        assert stats["total"] == 2
        assert stats["per_section"]["B.1"] == 1
        assert stats["per_section"]["B.2"] == 1
        assert stats["by_source"]["gold_standard"] == 1
        assert stats["by_source"]["bootstrap"] == 1
        assert stats["label_coverage"] == 2

    def test_empty_dataset(self) -> None:
        stats = dataset_stats([])
        assert stats["total"] == 0
        assert stats["label_coverage"] == 0
