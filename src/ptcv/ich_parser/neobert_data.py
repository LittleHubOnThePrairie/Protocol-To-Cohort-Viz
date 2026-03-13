"""Data preparation for NeoBERT fine-tuning — PTCV-164.

Loads the gold-standard corpus (PTCV-46), augments with
RuleBasedClassifier bootstrap on benchmark protocols, splits
into train/val/test, and tokenizes for HuggingFace Trainer.

Risk tier: MEDIUM — data pipeline component.

Regulatory references:
- ALCOA+ Accurate: confidence filtering at >= 0.80
- ALCOA+ Consistent: deterministic splits via seeded RNG
- ALCOA+ Traceable: registry_id + source preserved through pipeline
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from .corpus_loader import CorpusExemplar, load_corpus
from .neobert_config import LABEL_MAP, NeoBERTConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ClassificationSample:
    """Single training/evaluation sample."""

    text: str
    label: int
    section_code: str
    registry_id: str
    source: str  # "gold_standard" | "bootstrap"
    confidence: float


def load_training_data(
    config: NeoBERTConfig,
    corpus_path: Optional[Path] = None,
    bootstrap_md_dir: Optional[Path] = None,
) -> list[ClassificationSample]:
    """Load and combine all training data sources.

    Sources (in priority order):
    1. Gold-standard corpus (PTCV-46) — always loaded
    2. Bootstrap augmentation — RuleBasedClassifier on benchmark
       Markdown files, filtered to confidence >= config.min_confidence

    Args:
        config: NeoBERT configuration.
        corpus_path: Override path to gold_standard.jsonl.
        bootstrap_md_dir: Directory with pymupdf4llm Markdown files
            (e.g. ``data/analysis/benchmark_md/``).

    Returns:
        List of ClassificationSample instances.
    """
    samples: list[ClassificationSample] = []

    # Source 1: Gold-standard corpus
    corpus = load_corpus(corpus_path)
    samples.extend(_corpus_to_samples(corpus, config.min_confidence))
    logger.info("Gold-standard corpus: %d samples", len(samples))

    # Source 2: Bootstrap from benchmark Markdown
    if bootstrap_md_dir and bootstrap_md_dir.is_dir():
        md_paths = sorted(bootstrap_md_dir.glob("*/pymupdf4llm.md"))
        if md_paths:
            bootstrap = bootstrap_from_markdown(
                md_paths, config.min_confidence
            )
            samples.extend(bootstrap)
            logger.info("Bootstrap augmentation: %d samples", len(bootstrap))

    logger.info(
        "Total training data: %d samples across %d classes",
        len(samples),
        len({s.section_code for s in samples}),
    )
    return samples


def _corpus_to_samples(
    corpus: dict[str, list[CorpusExemplar]],
    min_confidence: float,
) -> list[ClassificationSample]:
    """Convert CorpusExemplar instances to ClassificationSample."""
    samples: list[ClassificationSample] = []
    for code, exemplars in corpus.items():
        label = _map_section_code(code)
        for ex in exemplars:
            if ex.confidence < min_confidence:
                continue
            samples.append(
                ClassificationSample(
                    text=ex.text,
                    label=label,
                    section_code=code,
                    registry_id=ex.registry_id,
                    source="gold_standard",
                    confidence=ex.confidence,
                )
            )
    return samples


def _map_section_code(code: str) -> int:
    """Map section code to label index, falling back to OTHER."""
    return LABEL_MAP.get(code, LABEL_MAP["OTHER"])


def bootstrap_from_markdown(
    md_paths: Sequence[Path],
    min_confidence: float = 0.80,
) -> list[ClassificationSample]:
    """Generate labeled samples by running RuleBasedClassifier on Markdown.

    Uses the existing rule-based classifier as a teacher to label
    benchmark protocol text. Only keeps sections with confidence
    >= min_confidence for training data quality.

    Args:
        md_paths: Paths to pymupdf4llm Markdown files.
        min_confidence: Minimum classifier confidence to include.

    Returns:
        List of ClassificationSample from bootstrap.
    """
    # Lazy import to avoid circular dependency through schema_loader
    from .classifier import RuleBasedClassifier

    classifier = RuleBasedClassifier()
    samples: list[ClassificationSample] = []

    for md_path in md_paths:
        nct_id = md_path.parent.name
        text = md_path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        sections = classifier.classify(
            text=text,
            registry_id=nct_id,
            run_id="bootstrap",
            source_run_id="bootstrap",
            source_sha256="bootstrap",
        )

        for section in sections:
            if section.confidence_score < min_confidence:
                continue
            label = _map_section_code(section.section_code)
            samples.append(
                ClassificationSample(
                    text=section.content_text or section.content_json,
                    label=label,
                    section_code=section.section_code,
                    registry_id=nct_id,
                    source="bootstrap",
                    confidence=section.confidence_score,
                )
            )

    logger.info(
        "Bootstrap: %d samples from %d protocols (min_confidence=%.2f)",
        len(samples),
        len(md_paths),
        min_confidence,
    )
    return samples


def split_dataset(
    samples: list[ClassificationSample],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[
    list[ClassificationSample],
    list[ClassificationSample],
    list[ClassificationSample],
]:
    """Stratified train/val/test split.

    Stratified by section_code to ensure each label appears in all
    splits where possible. Classes with fewer than 3 samples are
    placed entirely in the training set.

    Args:
        samples: All classification samples.
        train_ratio: Fraction for training (default 0.80).
        val_ratio: Fraction for validation (default 0.10).
        test_ratio: Fraction for testing (default 0.10).
        seed: Random seed (ALCOA+ Consistent).

    Returns:
        Tuple of (train, val, test) sample lists.
    """
    rng = random.Random(seed)

    # Group by section_code
    by_code: dict[str, list[ClassificationSample]] = defaultdict(list)
    for s in samples:
        by_code[s.section_code].append(s)

    train: list[ClassificationSample] = []
    val: list[ClassificationSample] = []
    test: list[ClassificationSample] = []

    for code in sorted(by_code.keys()):
        group = list(by_code[code])
        rng.shuffle(group)

        n = len(group)
        if n < 3:
            # Too few for splitting — put all in training
            train.extend(group)
            continue

        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            n_val = max(1, (n - 1) // 2)
            n_test = n - 1 - n_val

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def build_hf_dataset(
    samples: list[ClassificationSample],
    tokenizer: Any,
    max_length: int,
) -> Any:
    """Build a HuggingFace Dataset from classification samples.

    Tokenizes text with the given tokenizer, truncating to
    max_length tokens. Returns a Dataset with columns:
    input_ids, attention_mask, labels.

    Requires: datasets (optional import).

    Args:
        samples: Classification samples to tokenize.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.

    Returns:
        HuggingFace Dataset ready for Trainer.
    """
    from datasets import Dataset

    texts = [s.text for s in samples]
    labels = [s.label for s in samples]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"].tolist(),
            "attention_mask": encodings["attention_mask"].tolist(),
            "labels": labels,
        }
    )


def dataset_stats(
    samples: list[ClassificationSample],
) -> dict[str, Any]:
    """Compute summary statistics for a sample list.

    Args:
        samples: Classification samples.

    Returns:
        Dict with total, per_section counts, source breakdown,
        avg_text_length, and label coverage.
    """
    by_code: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    total_len = 0

    for s in samples:
        by_code[s.section_code] += 1
        by_source[s.source] += 1
        total_len += len(s.text)

    return {
        "total": len(samples),
        "per_section": dict(sorted(by_code.items())),
        "by_source": dict(sorted(by_source.items())),
        "avg_text_length": round(total_len / max(len(samples), 1)),
        "label_coverage": len(by_code),
    }
