"""Tests for PTCV-233: SetFit Embedding Fine-Tuner.

Tests verify corpus loading, pair generation, class balancing,
train/eval split, and evaluation metrics.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ptcv.registry.embedding_finetuner import (
    EmbeddingFineTuner,
    PairStats,
    EvaluationResult,
)


def _create_test_parquet(tmp_path: Path, n_per_section: dict) -> Path:
    """Create a test alignment corpus Parquet file."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not available")

    rows = []
    for section_code, count in n_per_section.items():
        for i in range(count):
            rows.append({
                "nct_id": f"NCT{10000000 + i:08d}",
                "section_code": section_code,
                "section_name": f"Section {section_code}",
                "registry_field": "TestModule",
                "registry_text": f"Registry text for {section_code} item {i}",
                "pdf_text_span": (
                    f"PDF content for {section_code} "
                    f"protocol item number {i} with enough text"
                ),
                "char_offset": i * 100,
                "span_length": 50,
                "quality_rating": 0.9,
                "similarity_score": 0.92,
            })

    df = pd.DataFrame(rows)
    path = tmp_path / "test_alignments.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


class TestPairStats:
    """Tests for PairStats dataclass."""

    def test_creation(self):
        """Test PairStats defaults."""
        stats = PairStats()
        assert stats.total_positive == 0
        assert stats.total_negative == 0
        assert stats.sections_used == 0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        """Test EvaluationResult defaults."""
        result = EvaluationResult()
        assert result.baseline_same_sim == 0.0
        assert result.separation_improvement == 0.0

    def test_populated(self):
        """Test fully populated result."""
        result = EvaluationResult(
            baseline_same_sim=0.7,
            baseline_cross_sim=0.4,
            finetuned_same_sim=0.85,
            finetuned_cross_sim=0.3,
            same_sim_delta=0.15,
            cross_sim_delta=-0.1,
            separation_improvement=0.25,
        )
        assert result.same_sim_delta == 0.15
        assert result.separation_improvement == 0.25


class TestLoadCorpus:
    """Tests for EmbeddingFineTuner.load_corpus."""

    def test_loads_and_splits(self, tmp_path):
        """Corpus loaded and split into train/eval."""
        path = _create_test_parquet(tmp_path, {
            "B.3": 30, "B.4": 20, "B.5": 50, "B.8": 25,
        })

        finetuner = EmbeddingFineTuner(max_samples_per_class=100)
        total = finetuner.load_corpus(path, eval_fraction=0.2)

        assert total == 125  # Original count
        assert finetuner.train_size > 0
        assert finetuner.eval_size > 0
        assert finetuner.train_size + finetuner.eval_size <= 125

    def test_class_balancing(self, tmp_path):
        """Oversized classes are capped at max_samples_per_class."""
        path = _create_test_parquet(tmp_path, {
            "B.3": 10, "B.5": 500,  # B.5 heavily over-represented
        })

        finetuner = EmbeddingFineTuner(max_samples_per_class=50)
        finetuner.load_corpus(path, eval_fraction=0.2)

        # Total should be capped: 10 (B.3) + 50 (B.5 capped) = 60
        total = finetuner.train_size + finetuner.eval_size
        assert total <= 60

    def test_eval_fraction(self, tmp_path):
        """Eval fraction controls train/eval split ratio."""
        path = _create_test_parquet(tmp_path, {
            "B.3": 50, "B.4": 50,
        })

        finetuner = EmbeddingFineTuner(max_samples_per_class=100)
        finetuner.load_corpus(path, eval_fraction=0.3)

        total = finetuner.train_size + finetuner.eval_size
        eval_ratio = finetuner.eval_size / total
        assert 0.2 <= eval_ratio <= 0.4  # Approximate due to rounding


class TestGeneratePairs:
    """Tests for EmbeddingFineTuner.generate_pairs."""

    def test_produces_positive_and_negative(self, tmp_path):
        """Both positive and negative pairs are generated."""
        path = _create_test_parquet(tmp_path, {
            "B.3": 20, "B.4": 20, "B.8": 20,
        })

        finetuner = EmbeddingFineTuner(max_samples_per_class=100)
        finetuner.load_corpus(path)
        stats = finetuner.generate_pairs()

        assert stats.total_positive > 0
        assert stats.total_negative > 0
        assert stats.sections_used == 3

    def test_pair_stats_per_section(self, tmp_path):
        """Per-section counts are tracked."""
        path = _create_test_parquet(tmp_path, {
            "B.3": 15, "B.5": 25,
        })

        finetuner = EmbeddingFineTuner(max_samples_per_class=100)
        finetuner.load_corpus(path)
        stats = finetuner.generate_pairs()

        assert "B.3" in stats.per_section
        assert "B.5" in stats.per_section
        assert stats.per_section["B.3"] <= 15
        assert stats.per_section["B.5"] <= 25

    def test_minimum_pair_counts(self, tmp_path):
        """At least 200 positive and 200 negative with sufficient data."""
        path = _create_test_parquet(tmp_path, {
            "B.3": 50, "B.4": 50, "B.5": 50,
            "B.8": 50, "B.16": 50, "B.2": 50,
        })

        finetuner = EmbeddingFineTuner(max_samples_per_class=100)
        finetuner.load_corpus(path)
        stats = finetuner.generate_pairs()

        # With 6 sections * 40 train samples each, should have plenty
        assert stats.total_positive >= 200
        assert stats.total_negative >= 200


class TestTrainGuards:
    """Tests for train() precondition checks."""

    def test_train_without_corpus_raises(self):
        """Training without loading corpus raises error."""
        finetuner = EmbeddingFineTuner()
        with pytest.raises(RuntimeError, match="No training data"):
            finetuner.train()

    def test_save_without_training_raises(self):
        """Saving without training raises error."""
        finetuner = EmbeddingFineTuner()
        with pytest.raises(RuntimeError, match="No trained model"):
            finetuner.save()

    def test_evaluate_without_corpus_raises(self):
        """Evaluating without loading corpus raises error."""
        finetuner = EmbeddingFineTuner()
        with pytest.raises(RuntimeError, match="No eval data"):
            finetuner.evaluate()

    def test_evaluate_without_training_raises(self, tmp_path):
        """Evaluating without training raises error."""
        path = _create_test_parquet(tmp_path, {"B.3": 10, "B.4": 10})
        finetuner = EmbeddingFineTuner()
        finetuner.load_corpus(path)
        with pytest.raises(RuntimeError, match="No trained model"):
            finetuner.evaluate()


class TestProperties:
    """Tests for EmbeddingFineTuner properties."""

    def test_pair_stats_none_before_generation(self):
        """pair_stats is None before generate_pairs()."""
        finetuner = EmbeddingFineTuner()
        assert finetuner.pair_stats is None

    def test_sizes_zero_before_load(self):
        """train_size and eval_size are 0 before loading."""
        finetuner = EmbeddingFineTuner()
        assert finetuner.train_size == 0
        assert finetuner.eval_size == 0
