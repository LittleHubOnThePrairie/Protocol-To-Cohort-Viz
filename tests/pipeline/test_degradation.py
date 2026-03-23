"""Tests for graceful degradation chain — PTCV-163.

All tests construct ``PipelineCapabilities`` directly so no real ML
dependencies (torch, anthropic, docling) are required.
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# PTCV-213: Import normally instead of spec_from_file_location.
# The old approach created a duplicate module instance whose enum
# classes were distinct Python objects from the normally-imported
# ones, causing KeyError when _CLASSIFICATION_LABELS was looked up
# by orchestrator tests later in the same pytest session.
from ptcv.pipeline.degradation import (
    ClassificationLevel,
    ExtractionLevel,
    PipelineCapabilities,
    detect_capabilities,
    log_pipeline_strategy,
    select_classification_level,
    select_extraction_level,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _caps(**overrides: bool) -> PipelineCapabilities:
    """Build a PipelineCapabilities with all-False defaults."""
    defaults = {
        "has_docling": False,
        "has_vision_api": False,
        "has_torch": False,
        "has_neobert_model": False,
        "has_anthropic": False,
        "has_rag_index": False,
    }
    defaults.update(overrides)
    return PipelineCapabilities(**defaults)


_ALL_TRUE = _caps(
    has_docling=True,
    has_vision_api=True,
    has_torch=True,
    has_neobert_model=True,
    has_anthropic=True,
    has_rag_index=True,
)

_ALL_FALSE = _caps()


# ------------------------------------------------------------------
# TestDetectCapabilities
# ------------------------------------------------------------------

class TestDetectCapabilities:
    """PipelineCapabilities detection."""

    def test_returns_dataclass(self) -> None:
        result = detect_capabilities()
        assert isinstance(result, PipelineCapabilities)

    def test_all_fields_are_bool(self) -> None:
        result = detect_capabilities()
        for field_name in (
            "has_docling", "has_vision_api", "has_torch",
            "has_neobert_model", "has_anthropic", "has_rag_index",
        ):
            assert isinstance(getattr(result, field_name), bool)

    def test_neobert_requires_torch_and_path(self, tmp_path: Path) -> None:
        """has_neobert_model is False when torch unavailable."""
        model_dir = tmp_path / "neobert"
        model_dir.mkdir()
        # Even with existing path, no torch means no model
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            result = detect_capabilities(
                neobert_model_path=str(model_dir),
            )
        # torch import fails → has_torch=False → has_neobert_model=False
        assert result.has_neobert_model is False

    def test_has_anthropic_requires_key_and_package(self) -> None:
        """has_anthropic is False when ANTHROPIC_API_KEY unset."""
        with patch.dict("os.environ", {}, clear=True):
            result = detect_capabilities()
        assert result.has_anthropic is False

    def test_has_rag_index_requires_importable_deps(
        self, tmp_path: Path,
    ) -> None:
        """has_rag_index is False when sentence-transformers unavailable
        even if index directory exists."""
        index_dir = tmp_path / "rag_index"
        index_dir.mkdir()
        with patch.dict(
            sys.modules,
            {"sentence_transformers": None, "faiss": None},
        ):
            result = detect_capabilities(
                rag_index_path=str(index_dir),
            )
        assert result.has_rag_index is False

    def test_never_raises(self) -> None:
        """detect_capabilities must never raise, even with hostile env."""
        with patch.dict(
            sys.modules,
            {"docling": None, "torch": None, "transformers": None,
             "anthropic": None, "sentence_transformers": None,
             "faiss": None},
        ):
            result = detect_capabilities(
                neobert_model_path="/nonexistent/path",
                rag_index_path="/nonexistent/rag",
            )
        assert result == _ALL_FALSE


# ------------------------------------------------------------------
# TestSelectExtractionLevel
# ------------------------------------------------------------------

class TestSelectExtractionLevel:
    """Extraction axis selection."""

    def test_e1_when_docling_and_vision(self) -> None:
        caps = _caps(has_docling=True, has_vision_api=True)
        assert select_extraction_level(caps) == ExtractionLevel.E1

    def test_e2_when_docling_only(self) -> None:
        caps = _caps(has_docling=True)
        assert select_extraction_level(caps) == ExtractionLevel.E2

    def test_e3_when_nothing(self) -> None:
        assert select_extraction_level(_ALL_FALSE) == ExtractionLevel.E3

    def test_e3_is_always_available(self) -> None:
        """E3 is the universal fallback."""
        assert select_extraction_level(_ALL_FALSE) == ExtractionLevel.E3

    def test_force_overrides_detection(self) -> None:
        """--force-extraction-level bypasses capability detection."""
        assert select_extraction_level(
            _ALL_TRUE, force_level=ExtractionLevel.E3,
        ) == ExtractionLevel.E3


# ------------------------------------------------------------------
# TestSelectClassificationLevel
# ------------------------------------------------------------------

class TestSelectClassificationLevel:
    """Classification axis selection."""

    def test_c1_full_pipeline(self) -> None:
        caps = _caps(
            has_neobert_model=True, has_rag_index=True,
            has_anthropic=True, has_torch=True,
        )
        assert select_classification_level(caps) == ClassificationLevel.C1

    def test_c2_no_rag(self) -> None:
        caps = _caps(
            has_neobert_model=True, has_anthropic=True, has_torch=True,
        )
        assert select_classification_level(caps) == ClassificationLevel.C2

    def test_c3_no_neobert(self) -> None:
        caps = _caps(has_anthropic=True)
        assert select_classification_level(caps) == ClassificationLevel.C3

    def test_c4_no_sonnet(self) -> None:
        caps = _caps(has_neobert_model=True, has_torch=True)
        assert select_classification_level(caps) == ClassificationLevel.C4

    def test_c5_nothing(self) -> None:
        assert (
            select_classification_level(_ALL_FALSE) == ClassificationLevel.C5
        )

    def test_c5_is_always_available(self) -> None:
        """C5 is the universal emergency fallback."""
        assert (
            select_classification_level(_ALL_FALSE) == ClassificationLevel.C5
        )

    def test_force_overrides_detection(self) -> None:
        assert select_classification_level(
            _ALL_TRUE, force_level=ClassificationLevel.C5,
        ) == ClassificationLevel.C5


# ------------------------------------------------------------------
# TestLogPipelineStrategy
# ------------------------------------------------------------------

class TestLogPipelineStrategy:
    """Quality warning log output."""

    def test_info_on_full_pipeline(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            log_pipeline_strategy(
                ExtractionLevel.E1, ClassificationLevel.C1, _ALL_TRUE,
            )
        assert "E1" in caplog.text
        assert "C1" in caplog.text
        # No WARNING for full pipeline
        warnings = [
            r for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warnings) == 0

    def test_warns_on_e3(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            log_pipeline_strategy(
                ExtractionLevel.E3, ClassificationLevel.C1, _ALL_TRUE,
            )
        assert "DEGRADED EXTRACTION" in caplog.text

    def test_warns_on_c4(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            log_pipeline_strategy(
                ExtractionLevel.E1, ClassificationLevel.C4, _ALL_TRUE,
            )
        assert "DEGRADED CLASSIFICATION" in caplog.text

    def test_warns_on_c5(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            log_pipeline_strategy(
                ExtractionLevel.E1, ClassificationLevel.C5, _ALL_TRUE,
            )
        assert "EMERGENCY MODE" in caplog.text

    def test_logs_all_capabilities(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO):
            log_pipeline_strategy(
                ExtractionLevel.E3, ClassificationLevel.C5, _ALL_FALSE,
            )
        assert "docling=False" in caplog.text
        assert "anthropic=False" in caplog.text


# ------------------------------------------------------------------
# TestCombinations
# ------------------------------------------------------------------

class TestCombinations:
    """All E x C combinations are valid states."""

    @pytest.mark.parametrize(
        "ext,cls",
        list(itertools.product(ExtractionLevel, ClassificationLevel)),
        ids=lambda x: x.name if hasattr(x, "name") else str(x),
    )
    def test_all_15_combinations_valid(
        self,
        ext: ExtractionLevel,
        cls: ClassificationLevel,
    ) -> None:
        """Every ExC pair can be logged without error."""
        log_pipeline_strategy(ext, cls, _ALL_TRUE)

    def test_exactly_15_combinations(self) -> None:
        combos = list(
            itertools.product(ExtractionLevel, ClassificationLevel),
        )
        assert len(combos) == 15


# ------------------------------------------------------------------
# TestEnumValues
# ------------------------------------------------------------------

class TestEnumValues:
    """Enum string values for metadata output."""

    def test_extraction_values(self) -> None:
        assert ExtractionLevel.E1.value == "docling_vision"
        assert ExtractionLevel.E2.value == "docling"
        assert ExtractionLevel.E3.value == "pdfplumber"

    def test_classification_values(self) -> None:
        assert ClassificationLevel.C1.value == "neobert_rag_sonnet"
        assert ClassificationLevel.C2.value == "neobert_sonnet"
        assert ClassificationLevel.C3.value == "rulebased_sonnet"
        assert ClassificationLevel.C4.value == "neobert_rulebased"
        assert ClassificationLevel.C5.value == "rulebased_only"

    def test_format_string(self) -> None:
        """Metadata format: 'NAME:value'."""
        level = ExtractionLevel.E3
        formatted = f"{level.name}:{level.value}"
        assert formatted == "E3:pdfplumber"
