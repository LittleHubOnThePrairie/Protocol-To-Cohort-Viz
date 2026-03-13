"""Fine-tune NeoBERT for ICH E6(R3) section classification — PTCV-164.

Uses HuggingFace Trainer with:
- AdamW optimizer, linear warmup + cosine decay
- FP16 mixed precision (6 GB VRAM RTX 3060)
- Early stopping on validation macro F1 (patience=3)
- Class-weight balancing for imbalanced data
- Saves checkpoint + config snapshot + metrics.json

Usage:
    python scripts/train_neobert.py [OPTIONS]

Options:
    --epochs INT          Training epochs (default: 10)
    --lr FLOAT            Learning rate (default: 2e-5)
    --batch-size INT      Training batch size (default: 2)
    --max-length INT      Max token length (default: 2048)
    --corpus PATH         Gold-standard corpus path
    --bootstrap-dir PATH  Directory with pymupdf4llm Markdown files
    --output-dir PATH     Model output directory (default: data/models/neobert)
    --seed INT            Random seed (default: 42)

Risk tier: MEDIUM — ML training pipeline.

Regulatory references:
- ALCOA+ Consistent: seeded RNG, deterministic training
- ALCOA+ Traceable: config snapshot + metrics logged
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Mock fitz (PyMuPDF) if not installed — prevents ImportError when
# ich_parser.__init__ imports toc_extractor.
if "fitz" not in sys.modules:
    try:
        import fitz  # noqa: F401
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["fitz"] = MagicMock()

from ptcv.ich_parser.neobert_config import NUM_LABELS, NeoBERTConfig
from ptcv.ich_parser.neobert_data import (
    build_hf_dataset,
    dataset_stats,
    load_training_data,
    split_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune NeoBERT for ICH section classification",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Training batch size",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Max token length (2048 for 6GB VRAM training)",
    )
    parser.add_argument(
        "--corpus", type=Path, default=None,
        help="Gold-standard corpus path",
    )
    parser.add_argument(
        "--bootstrap-dir", type=Path, default=None,
        help="Directory with pymupdf4llm Markdown files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/models/neobert"),
        help="Model output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed",
    )
    return parser.parse_args()


def compute_class_weights(
    labels: list[int], num_classes: int,
) -> list[float]:
    """Compute inverse-frequency class weights for imbalanced data."""
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total / (num_classes * count))
    return weights


def main() -> None:
    """Fine-tune NeoBERT."""
    args = parse_args()

    # Late imports for optional ML dependencies
    try:
        import numpy as np
        import torch
        from sklearn.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        logger.error(
            "Missing ML dependencies. Install with: "
            "pip install -r requirements-ml.txt"
        )
        raise SystemExit(1) from e

    # Build config from CLI args
    config = NeoBERTConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        output_dir=str(args.output_dir),
    )

    logger.info("NeoBERT fine-tuning config: %s", config)

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

    # Load data
    logger.info("Loading training data...")
    bootstrap_dir = args.bootstrap_dir
    if bootstrap_dir is None:
        # Default bootstrap directory
        candidate = Path("data/analysis/benchmark_md")
        if candidate.is_dir():
            bootstrap_dir = candidate

    samples = load_training_data(
        config, corpus_path=args.corpus, bootstrap_md_dir=bootstrap_dir,
    )

    if len(samples) < 10:
        logger.error(
            "Only %d samples — too few for training. "
            "Provide bootstrap data via --bootstrap-dir.",
            len(samples),
        )
        raise SystemExit(1)

    # Split
    train_samples, val_samples, test_samples = split_dataset(
        samples, seed=config.seed,
    )

    # Stats
    for name, split in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]:
        stats = dataset_stats(split)
        logger.info("%s split: %s", name, json.dumps(stats, indent=2))

    # Tokenizer
    logger.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True,
    )

    # Build HuggingFace datasets
    train_ds = build_hf_dataset(train_samples, tokenizer, config.max_length)
    val_ds = build_hf_dataset(val_samples, tokenizer, config.max_length)

    # Model
    logger.info("Loading model: %s", config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
        trust_remote_code=True,
    )

    # Class weights for loss
    train_labels = [s.label for s in train_samples]
    class_weights = compute_class_weights(train_labels, NUM_LABELS)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logger.info("Class weights: %s", class_weights)

    # Custom Trainer with weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights_tensor.to(logits.device),
            )
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro": f1_score(labels, preds, average="macro"),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0,
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0,
            ),
        }

    # Training arguments
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16 and device == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=10,
        seed=config.seed,
        report_to="none",
    )

    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    logger.info("Starting training...")
    start_time = time.monotonic()
    train_result = trainer.train()
    elapsed = time.monotonic() - start_time
    logger.info("Training completed in %.1f seconds", elapsed)

    # Save best model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model saved to %s", output_dir)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_ds = build_hf_dataset(test_samples, tokenizer, config.max_length)
    test_metrics = trainer.evaluate(test_ds)
    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    # Detailed classification report
    test_preds = trainer.predict(test_ds)
    preds = np.argmax(test_preds.predictions, axis=-1)
    labels = test_preds.label_ids
    from ptcv.ich_parser.neobert_config import INV_LABEL_MAP
    present_labels = sorted(set(labels) | set(preds))
    target_names = [INV_LABEL_MAP[i] for i in present_labels]
    report = classification_report(
        labels, preds, labels=present_labels,
        target_names=target_names, zero_division=0,
    )
    logger.info("Classification report:\n%s", report)

    # Save metrics
    metrics = {
        "config": {
            k: v for k, v in config.__dict__.items()
        },
        "training": {
            "elapsed_seconds": round(elapsed, 1),
            "train_loss": train_result.training_loss,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
        },
        "test_metrics": test_metrics,
        "classification_report": report,
        "device": device,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", metrics_path)

    # GPU memory
    if device == "cuda":
        mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info("GPU memory: %.0f MB", mem_mb)


if __name__ == "__main__":
    main()
