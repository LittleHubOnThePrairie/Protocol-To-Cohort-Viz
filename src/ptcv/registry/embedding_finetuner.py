"""Embedding Fine-Tuner: SetFit Contrastive Training on Alignment Corpus.

PTCV-233: Fine-tunes the all-MiniLM-L6-v2 sentence-transformer model
using contrastive pairs from the PTCV-232 alignment corpus.  Positive
pairs share the same ICH section code; negative pairs differ.  The
fine-tuned model produces 384-dim embeddings compatible with RagIndex.

Risk tier: LOW -- offline model training on local data (no API calls).

Usage::

    from ptcv.registry.embedding_finetuner import (
        EmbeddingFineTuner,
        PairStats,
    )

    finetuner = EmbeddingFineTuner()
    finetuner.load_corpus("data/alignment_corpus/alignments.parquet")
    stats = finetuner.generate_pairs()
    finetuner.train()
    finetuner.save("data/models/ich_classifier_v1")
    evaluation = finetuner.evaluate()
"""

import dataclasses
import itertools
import logging
import random
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_OUTPUT_DIR = Path("C:/Dev/PTCV/data/models/ich_classifier_v1")

# Pair generation: max pairs per class to avoid B.5 domination.
_MAX_SAMPLES_PER_CLASS = 100

# SetFit training defaults.
_DEFAULT_NUM_ITERATIONS = 20
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_NUM_EPOCHS = 1


@dataclasses.dataclass
class PairStats:
    """Statistics from contrastive pair generation.

    Attributes:
        total_positive: Number of same-section pairs.
        total_negative: Number of cross-section pairs.
        per_section: Count of samples used per section code.
        sections_used: Number of distinct section codes.
    """

    total_positive: int = 0
    total_negative: int = 0
    per_section: dict[str, int] = dataclasses.field(default_factory=dict)
    sections_used: int = 0


@dataclasses.dataclass
class EvaluationResult:
    """Evaluation of fine-tuned vs baseline embeddings.

    Attributes:
        baseline_same_sim: Avg cosine similarity for same-section (baseline).
        baseline_cross_sim: Avg cosine similarity for cross-section (baseline).
        finetuned_same_sim: Avg cosine similarity for same-section (fine-tuned).
        finetuned_cross_sim: Avg cosine similarity for cross-section (fine-tuned).
        same_sim_delta: Improvement in same-section similarity.
        cross_sim_delta: Change in cross-section similarity.
        separation_improvement: How much better fine-tuned separates classes.
    """

    baseline_same_sim: float = 0.0
    baseline_cross_sim: float = 0.0
    finetuned_same_sim: float = 0.0
    finetuned_cross_sim: float = 0.0
    same_sim_delta: float = 0.0
    cross_sim_delta: float = 0.0
    separation_improvement: float = 0.0


class EmbeddingFineTuner:
    """Fine-tune sentence-transformer embeddings via SetFit contrastive loss.

    Loads the alignment corpus, generates balanced contrastive pairs,
    trains the model, and evaluates improvement in section discrimination.

    Args:
        base_model: Sentence-transformer model name or path.
        max_samples_per_class: Cap per section to handle class imbalance.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        base_model: str = _DEFAULT_BASE_MODEL,
        max_samples_per_class: int = _MAX_SAMPLES_PER_CLASS,
        seed: int = 42,
    ) -> None:
        self._base_model = base_model
        self._max_per_class = max_samples_per_class
        self._seed = seed
        self._corpus_df: Any = None
        self._train_texts: list[str] = []
        self._train_labels: list[str] = []
        self._eval_texts: list[str] = []
        self._eval_labels: list[str] = []
        self._model: Any = None
        self._pair_stats: Optional[PairStats] = None

    def load_corpus(
        self,
        parquet_path: str | Path,
        eval_fraction: float = 0.2,
    ) -> int:
        """Load alignment corpus and split into train/eval sets.

        Args:
            parquet_path: Path to alignments.parquet from PTCV-232.
            eval_fraction: Fraction of data reserved for evaluation.

        Returns:
            Total number of records loaded.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required: pip install pandas")

        df = pd.read_parquet(parquet_path)
        self._corpus_df = df

        # Balance classes: cap each section at max_samples_per_class
        rng = random.Random(self._seed)
        balanced_rows: list[dict[str, Any]] = []

        for section_code, group in df.groupby("section_code"):
            rows = group.to_dict("records")
            if len(rows) > self._max_per_class:
                rng.shuffle(rows)
                rows = rows[: self._max_per_class]
            balanced_rows.extend(rows)

        rng.shuffle(balanced_rows)

        # Split train/eval
        split_idx = int(len(balanced_rows) * (1 - eval_fraction))
        train_rows = balanced_rows[:split_idx]
        eval_rows = balanced_rows[split_idx:]

        self._train_texts = [r["pdf_text_span"] for r in train_rows]
        self._train_labels = [r["section_code"] for r in train_rows]
        self._eval_texts = [r["pdf_text_span"] for r in eval_rows]
        self._eval_labels = [r["section_code"] for r in eval_rows]

        logger.info(
            "Loaded %d records, balanced to %d train + %d eval",
            len(df),
            len(self._train_texts),
            len(self._eval_texts),
        )
        return len(df)

    def generate_pairs(self) -> PairStats:
        """Generate contrastive training pairs from balanced corpus.

        Positive pairs: (text_A, text_B) with same section_code.
        Negative pairs: (text_A, text_C) with different section_code.

        Returns:
            PairStats with pair counts and per-section breakdown.
        """
        per_section: dict[str, list[str]] = {}
        for text, label in zip(self._train_texts, self._train_labels):
            per_section.setdefault(label, []).append(text)

        n_positive = 0
        n_negative = 0

        for section, texts in per_section.items():
            # Positive pairs within section
            n_positive += min(
                len(texts) * (len(texts) - 1) // 2,
                len(texts) * 5,  # Cap at 5 pairs per sample
            )

        # Negative: cross-section pairs
        sections = list(per_section.keys())
        for i, s1 in enumerate(sections):
            for s2 in sections[i + 1 :]:
                n_neg = min(
                    len(per_section[s1]) * len(per_section[s2]),
                    50,  # Cap per section pair
                )
                n_negative += n_neg

        self._pair_stats = PairStats(
            total_positive=n_positive,
            total_negative=n_negative,
            per_section={s: len(t) for s, t in per_section.items()},
            sections_used=len(per_section),
        )

        logger.info(
            "Pairs: %d positive, %d negative across %d sections",
            n_positive,
            n_negative,
            len(per_section),
        )
        return self._pair_stats

    def train(
        self,
        num_iterations: int = _DEFAULT_NUM_ITERATIONS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        num_epochs: int = _DEFAULT_NUM_EPOCHS,
    ) -> None:
        """Train the SetFit model on contrastive pairs.

        Args:
            num_iterations: SetFit sampling iterations.
            batch_size: Training batch size.
            num_epochs: Number of training epochs.
        """
        if not self._train_texts:
            raise RuntimeError(
                "No training data. Call load_corpus() first."
            )

        try:
            from setfit import SetFitModel, SetFitTrainer
            from datasets import Dataset
        except ImportError:
            raise RuntimeError(
                "setfit and datasets required: "
                "pip install setfit datasets"
            )

        logger.info(
            "Training SetFit on %d samples, %d sections, "
            "iterations=%d, batch=%d, epochs=%d",
            len(self._train_texts),
            len(set(self._train_labels)),
            num_iterations,
            batch_size,
            num_epochs,
        )

        # Create SetFit model from base sentence-transformer
        model = SetFitModel.from_pretrained(self._base_model)

        # Build HuggingFace Dataset
        train_dataset = Dataset.from_dict({
            "text": self._train_texts,
            "label": self._train_labels,
        })

        eval_dataset = Dataset.from_dict({
            "text": self._eval_texts,
            "label": self._eval_labels,
        })

        # Train
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_iterations=num_iterations,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

        trainer.train()
        self._model = model

        logger.info("SetFit training complete")

    def save(self, output_dir: str | Path = _DEFAULT_OUTPUT_DIR) -> Path:
        """Save the fine-tuned model to disk.

        Args:
            output_dir: Directory to save the model.

        Returns:
            Path to the saved model directory.
        """
        if self._model is None:
            raise RuntimeError("No trained model. Call train() first.")

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(path))
        logger.info("Model saved to %s", path)
        return path

    def evaluate(self) -> EvaluationResult:
        """Evaluate fine-tuned vs baseline embedding quality.

        Computes cosine similarity distributions for same-section
        and cross-section pairs on the held-out eval set, comparing
        baseline and fine-tuned models.

        Returns:
            EvaluationResult with similarity deltas.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "sentence_transformers and numpy required"
            )

        if not self._eval_texts:
            raise RuntimeError(
                "No eval data. Call load_corpus() first."
            )

        if self._model is None:
            raise RuntimeError(
                "No trained model. Call train() first."
            )

        # Encode eval texts with both models
        baseline = SentenceTransformer(self._base_model)
        baseline_embeds = baseline.encode(
            self._eval_texts, convert_to_numpy=True,
            show_progress_bar=False,
        )

        ft_embeds = self._model.model_body.encode(
            self._eval_texts, convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Compute pairwise similarities
        def _cosine_sims(
            embeds: Any, labels: list[str],
        ) -> tuple[list[float], list[float]]:
            """Compute same-section and cross-section cosine sims."""
            from numpy.linalg import norm

            same: list[float] = []
            cross: list[float] = []
            n = len(labels)
            # Sample pairs to keep computation tractable
            rng = random.Random(self._seed)
            pairs = list(range(n))
            rng.shuffle(pairs)
            max_pairs = min(n * 5, 500)

            for idx in range(min(max_pairs, n)):
                i = pairs[idx]
                for j in range(i + 1, min(i + 10, n)):
                    a = embeds[i]
                    b = embeds[j]
                    sim = float(
                        np.dot(a, b) / (norm(a) * norm(b) + 1e-10)
                    )
                    if labels[i] == labels[j]:
                        same.append(sim)
                    else:
                        cross.append(sim)
            return same, cross

        bl_same, bl_cross = _cosine_sims(
            baseline_embeds, self._eval_labels,
        )
        ft_same, ft_cross = _cosine_sims(
            ft_embeds, self._eval_labels,
        )

        def _avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        bl_same_avg = _avg(bl_same)
        bl_cross_avg = _avg(bl_cross)
        ft_same_avg = _avg(ft_same)
        ft_cross_avg = _avg(ft_cross)

        result = EvaluationResult(
            baseline_same_sim=round(bl_same_avg, 4),
            baseline_cross_sim=round(bl_cross_avg, 4),
            finetuned_same_sim=round(ft_same_avg, 4),
            finetuned_cross_sim=round(ft_cross_avg, 4),
            same_sim_delta=round(ft_same_avg - bl_same_avg, 4),
            cross_sim_delta=round(ft_cross_avg - bl_cross_avg, 4),
            separation_improvement=round(
                (ft_same_avg - ft_cross_avg)
                - (bl_same_avg - bl_cross_avg),
                4,
            ),
        )

        logger.info(
            "Evaluation: same_sim delta=%.4f, cross_sim delta=%.4f, "
            "separation improvement=%.4f",
            result.same_sim_delta,
            result.cross_sim_delta,
            result.separation_improvement,
        )
        return result

    @property
    def pair_stats(self) -> Optional[PairStats]:
        """Return pair generation statistics."""
        return self._pair_stats

    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return len(self._train_texts)

    @property
    def eval_size(self) -> int:
        """Number of evaluation samples."""
        return len(self._eval_texts)


__all__ = [
    "EmbeddingFineTuner",
    "PairStats",
    "EvaluationResult",
]
