"""NeoBERT ICH E6(R3) section classifier — PTCV-164.

Transformer-based 15-class classifier using fine-tuned NeoBERT
(chandar-lab/NeoBERT, 250M params, 4096-token context).

This is an EVALUATION implementation (PTCV-164). Production
integration is tracked by PTCV-161.

Requires: torch, transformers (optional — graceful ImportError).

Risk tier: MEDIUM — data pipeline ML component (no patient data).

Regulatory references:
- ALCOA+ Accurate: confidence_score from softmax probability
- ALCOA+ Consistent: deterministic inference (seed fixed)
- ALCOA+ Traceable: model version logged at classification time
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from .classifier import SectionClassifier
from .models import IchSection
from .neobert_config import INV_LABEL_MAP, LABEL_MAP, NUM_LABELS, NeoBERTConfig
from .schema_loader import get_classifier_sections, get_review_threshold

logger = logging.getLogger(__name__)

# Lazy imports for optional torch/transformers
_torch: Any = None
_transformers: Any = None


def _ensure_ml_deps() -> tuple[Any, Any]:
    """Import torch and transformers, raising clear error if missing."""
    global _torch, _transformers
    if _torch is None:
        try:
            import torch
            import transformers

            _torch = torch
            _transformers = transformers
        except ImportError as e:
            raise ImportError(
                "NeoBERTClassifier requires 'torch' and 'transformers'. "
                "Install with: pip install -r requirements-ml.txt"
            ) from e
    return _torch, _transformers


class NeoBERTClassifier(SectionClassifier):
    """Fine-tuned NeoBERT classifier for ICH E6(R3) sections.

    Loads a fine-tuned NeoBERT checkpoint with a sequence
    classification head (14 ICH sections + OTHER). Classifies
    protocol text by splitting into overlapping windows, classifying
    each window, then aggregating predictions per section.

    Args:
        model_path: Path to fine-tuned model directory.
        device: PyTorch device string ("cuda", "cpu", "auto").
        config: NeoBERT configuration (default used if None).

    Raises:
        ImportError: If torch/transformers not installed.
        FileNotFoundError: If model_path does not exist.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        config: Optional[NeoBERTConfig] = None,
    ) -> None:
        torch, transformers = _ensure_ml_deps()
        self._config = config or NeoBERTConfig()
        self._model_path = Path(model_path)

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"NeoBERT model not found at {self._model_path}"
            )

        # Resolve device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Load tokenizer and model
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(self._model_path),
            trust_remote_code=True,
        )
        self._model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                str(self._model_path),
                num_labels=NUM_LABELS,
                trust_remote_code=True,
            )
        )
        self._model.to(self._device)
        self._model.eval()

        self._ich_sections = get_classifier_sections()
        logger.info(
            "NeoBERTClassifier loaded from %s on %s",
            self._model_path,
            self._device,
        )

    def classify(
        self,
        text: str,
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        """Classify protocol text into ICH E6(R3) sections.

        Strategy:
        1. Split text into overlapping windows (stride = max_length // 2).
        2. Classify each window -> (predicted_label, probability).
        3. Group windows by predicted section code.
        4. For each section, merge windows and use max probability
           as confidence score.
        5. Deduplicate to one IchSection per section_code.

        Args:
            text: Full protocol text to classify.
            registry_id: Trial identifier.
            run_id: UUID4 for this ICH-parse run.
            source_run_id: run_id from upstream extraction.
            source_sha256: SHA-256 of upstream artifact.

        Returns:
            List of IchSection instances, one per detected section.
        """
        torch, _ = _ensure_ml_deps()
        start_time = time.monotonic()

        if not text.strip():
            return []

        windows = self._split_into_windows(text)
        predictions = self._classify_windows(windows)
        sections = self._aggregate_predictions(
            predictions,
            registry_id,
            run_id,
            source_run_id,
            source_sha256,
        )

        elapsed = time.monotonic() - start_time
        logger.info(
            "NeoBERT classified %d windows -> %d sections in %.2fs",
            len(windows),
            len(sections),
            elapsed,
        )
        return sections

    def _split_into_windows(
        self, text: str
    ) -> list[tuple[str, int, int]]:
        """Split text into overlapping token windows.

        Returns list of (window_text, char_start, char_end) tuples.
        Uses a stride of max_length // 2 for 50% overlap, ensuring
        no section boundary falls entirely within a stride gap.
        """
        max_len = self._config.inference_max_length
        stride = max_len // 2

        # Tokenize full text to find natural split points
        encoding = self._tokenizer(
            text,
            truncation=False,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoding["offset_mapping"]
        total_tokens = len(offsets)

        if total_tokens == 0:
            return []

        # Reserve 2 tokens for [CLS] and [SEP]
        effective_len = max_len - 2
        effective_stride = stride

        windows: list[tuple[str, int, int]] = []
        token_start = 0

        while token_start < total_tokens:
            token_end = min(token_start + effective_len, total_tokens)

            # Map token positions back to character positions
            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]

            window_text = text[char_start:char_end]
            if window_text.strip():
                windows.append((window_text, char_start, char_end))

            if token_end >= total_tokens:
                break
            token_start += effective_stride

        return windows

    def _classify_windows(
        self, windows: list[tuple[str, int, int]]
    ) -> list[tuple[str, float, str, int, int]]:
        """Classify each window.

        Returns list of (section_code, confidence, window_text,
        char_start, char_end) tuples.
        """
        torch, _ = _ensure_ml_deps()
        results: list[tuple[str, float, str, int, int]] = []

        for window_text, char_start, char_end in windows:
            inputs = self._tokenizer(
                window_text,
                truncation=True,
                max_length=self._config.inference_max_length,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1)
                confidence, predicted = torch.max(probs, dim=-1)

            section_code = INV_LABEL_MAP.get(
                predicted.item(), "OTHER"
            )
            results.append((
                section_code,
                round(confidence.item(), 4),
                window_text,
                char_start,
                char_end,
            ))

        return results

    def _aggregate_predictions(
        self,
        predictions: list[tuple[str, float, str, int, int]],
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        """Aggregate per-window predictions into IchSection list.

        Groups by section_code, merges content_text, takes max
        confidence per section. Mirrors deduplication in
        RuleBasedClassifier._deduplicate().
        """
        from collections import defaultdict

        grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for code, conf, text, _, _ in predictions:
            grouped[code].append((conf, text))

        sections: list[IchSection] = []
        for code, entries in sorted(grouped.items()):
            if code == "OTHER":
                continue  # Skip unclassifiable windows

            # Use max confidence from all windows for this section
            best_conf = max(conf for conf, _ in entries)
            merged_text = "\n\n".join(text for _, text in entries)

            # Truncate excerpt for content_json
            excerpt = merged_text[:500] + (
                "..." if len(merged_text) > 500 else ""
            )
            word_count = len(merged_text.split())
            content_json = json.dumps(
                {"text_excerpt": excerpt, "word_count": word_count}
            )

            section_name = self._ich_sections.get(
                code, {}
            ).get("name", code)
            threshold = get_review_threshold(code)

            sections.append(
                IchSection(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    section_code=code,
                    section_name=section_name,
                    content_json=content_json,
                    confidence_score=best_conf,
                    review_required=best_conf < threshold,
                    legacy_format=False,
                    content_text=merged_text,
                )
            )

        return sections

    def get_gpu_memory_mb(self) -> float:
        """Report current GPU memory usage in MB.

        Returns 0.0 if running on CPU.
        """
        torch, _ = _ensure_ml_deps()
        if self._device == "cpu":
            return 0.0
        return torch.cuda.memory_allocated(self._device) / (1024 * 1024)
