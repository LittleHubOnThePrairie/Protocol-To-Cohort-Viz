"""Tests for NeoBERTClassifier — PTCV-164.

IQ/OQ tests for the NeoBERT section classifier. Unit tests use
mocked model weights to run on CPU without a fine-tuned checkpoint.
Integration tests require GPU + checkpoint (auto-skipped).

Qualification phase: IQ/OQ
Risk tier: MEDIUM
Regulatory: ALCOA+ Accurate (confidence score validation).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.classifier import SectionClassifier
from ptcv.ich_parser.neobert_config import LABEL_MAP, NUM_LABELS, NeoBERTConfig


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _has_torch() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_checkpoint() -> bool:
    return Path("data/models/neobert").exists()


requires_torch = pytest.mark.skipif(
    not _has_torch(), reason="torch not installed"
)
requires_cuda = pytest.mark.skipif(
    not _has_cuda(), reason="CUDA GPU required"
)
requires_checkpoint = pytest.mark.skipif(
    not _has_checkpoint(), reason="Fine-tuned checkpoint required"
)


# ---------------------------------------------------------------------------
# TestNeoBERTClassifierABC
# ---------------------------------------------------------------------------


class TestNeoBERTClassifierABC:
    """NeoBERTClassifier implements the SectionClassifier interface."""

    def test_implements_section_classifier_abc(self) -> None:
        from ptcv.ich_parser.neobert_classifier import NeoBERTClassifier
        assert issubclass(NeoBERTClassifier, SectionClassifier)


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Graceful errors when dependencies or files missing."""

    def test_file_not_found_error(self) -> None:
        """FileNotFoundError raised for missing checkpoint."""
        if not _has_torch():
            pytest.skip("torch required for this test")
        from ptcv.ich_parser.neobert_classifier import NeoBERTClassifier
        with pytest.raises(FileNotFoundError, match="not found"):
            NeoBERTClassifier(model_path="/nonexistent/path")


# ---------------------------------------------------------------------------
# TestNeoBERTConfig
# ---------------------------------------------------------------------------


class TestNeoBERTConfig:
    """Configuration dataclass validation."""

    def test_default_config(self) -> None:
        config = NeoBERTConfig()
        assert config.model_name == "chandar-lab/NeoBERT"
        assert config.max_length == 2048
        assert config.inference_max_length == 4096
        assert config.num_labels == 15
        assert config.fp16 is True
        assert config.batch_size == 2
        assert config.seed == 42

    def test_config_immutable(self) -> None:
        config = NeoBERTConfig()
        with pytest.raises(AttributeError):
            config.seed = 99  # type: ignore[misc]

    def test_custom_config(self) -> None:
        config = NeoBERTConfig(
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=4,
        )
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 5
        assert config.batch_size == 4


# ---------------------------------------------------------------------------
# TestLabelMapping
# ---------------------------------------------------------------------------


class TestLabelMapping:
    """Label map consistency checks."""

    def test_contiguous_indices(self) -> None:
        """Label indices are 0..NUM_LABELS-1 without gaps."""
        indices = sorted(LABEL_MAP.values())
        assert indices == list(range(NUM_LABELS))

    def test_b1_through_b14_present(self) -> None:
        for i in range(1, 15):
            assert f"B.{i}" in LABEL_MAP

    def test_other_is_last_index(self) -> None:
        assert LABEL_MAP["OTHER"] == NUM_LABELS - 1


# ---------------------------------------------------------------------------
# Integration tests (require GPU + checkpoint)
# ---------------------------------------------------------------------------


@requires_torch
@requires_cuda
@requires_checkpoint
class TestNeoBERTIntegration:
    """Integration tests requiring fine-tuned checkpoint + GPU."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from ptcv.ich_parser.neobert_classifier import NeoBERTClassifier
        self.classifier = NeoBERTClassifier(
            model_path="data/models/neobert",
            device="cuda",
        )

    def test_classify_returns_ich_sections(self) -> None:
        sections = self.classifier.classify(
            text="1. INTRODUCTION\nThis is a phase III trial.",
            registry_id="NCT00000001",
            run_id="test-run",
            source_run_id="test-source",
            source_sha256="a" * 64,
        )
        from ptcv.ich_parser.models import IchSection
        assert all(isinstance(s, IchSection) for s in sections)

    def test_confidence_between_zero_and_one(self) -> None:
        sections = self.classifier.classify(
            text="5. SELECTION OF SUBJECTS\nInclusion criteria: age 18-75",
            registry_id="NCT00000001",
            run_id="test-run",
            source_run_id="test-source",
            source_sha256="a" * 64,
        )
        for s in sections:
            assert 0.0 <= s.confidence_score <= 1.0

    def test_no_duplicate_section_codes(self) -> None:
        sections = self.classifier.classify(
            text="Large protocol text with multiple sections...",
            registry_id="NCT00000001",
            run_id="test-run",
            source_run_id="test-source",
            source_sha256="a" * 64,
        )
        codes = [s.section_code for s in sections]
        assert len(codes) == len(set(codes))

    def test_gpu_memory_under_6gb(self) -> None:
        mem_mb = self.classifier.get_gpu_memory_mb()
        assert mem_mb < 6144, f"GPU memory {mem_mb:.0f} MB exceeds 6 GB"

    def test_empty_text_returns_empty(self) -> None:
        sections = self.classifier.classify(
            text="",
            registry_id="NCT00000001",
            run_id="test-run",
            source_run_id="test-source",
            source_sha256="a" * 64,
        )
        assert sections == []
