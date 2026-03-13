"""NeoBERT classifier configuration — PTCV-164.

Hyperparameters, label mapping, and model paths for the NeoBERT
ICH section classifier. Separated from classifier code to allow
independent tuning without touching the inference path.

Risk tier: LOW — configuration constants only.
"""

from __future__ import annotations

import dataclasses


# 14 mandatory ICH E6(R3) sections + OTHER for B.15/B.16/unclassifiable
LABEL_MAP: dict[str, int] = {
    "B.1": 0,
    "B.2": 1,
    "B.3": 2,
    "B.4": 3,
    "B.5": 4,
    "B.6": 5,
    "B.7": 6,
    "B.8": 7,
    "B.9": 8,
    "B.10": 9,
    "B.11": 10,
    "B.12": 11,
    "B.13": 12,
    "B.14": 13,
    "OTHER": 14,
}

INV_LABEL_MAP: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

NUM_LABELS: int = len(LABEL_MAP)  # 15


@dataclasses.dataclass(frozen=True)
class NeoBERTConfig:
    """Fine-tuning and inference configuration.

    Attributes:
        model_name: HuggingFace model identifier.
        max_length: Maximum token sequence length for training.
            NeoBERT supports 4096; use 2048 for 6 GB VRAM training.
        inference_max_length: Maximum token length at inference time.
        num_labels: Number of classification labels (15).
        learning_rate: AdamW learning rate for fine-tuning.
        batch_size: Training batch size (2 for 6 GB VRAM).
        eval_batch_size: Evaluation batch size.
        num_epochs: Training epochs.
        gradient_accumulation_steps: Steps before optimizer update.
        warmup_ratio: Fraction of steps for LR warmup.
        weight_decay: AdamW weight decay.
        min_confidence: Minimum corpus confidence for training data.
        seed: Random seed for reproducibility (ALCOA+ Consistent).
        output_dir: Directory for model checkpoints.
        fp16: Use mixed-precision training (saves VRAM).
    """

    model_name: str = "chandar-lab/NeoBERT"
    max_length: int = 2048
    inference_max_length: int = 4096
    num_labels: int = NUM_LABELS
    learning_rate: float = 2e-5
    batch_size: int = 2
    eval_batch_size: int = 4
    num_epochs: int = 10
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    min_confidence: float = 0.80
    seed: int = 42
    output_dir: str = "data/models/neobert"
    fp16: bool = True
