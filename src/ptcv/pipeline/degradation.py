"""Graceful degradation chain for PTCV pipeline (PTCV-163).

2D degradation matrix with independent axes:
  - Extraction:      E1 (Docling+Vision) > E2 (Docling) > E3 (pdfplumber)
  - Classification:  C1 (NeoBERT+RAG+Sonnet) > C2 (NeoBERT+Sonnet) >
                     C3 (RuleBased+Sonnet) > C4 (NeoBERT+RuleBased) >
                     C5 (RuleBased only)

Any extraction level works with any classification level.
E3+C5 must ALWAYS work (zero external dependencies).

Research basis: PTCV-156 Section 3.2.
Citations:
  [9] "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing," ICLR 2024
  [10] "Hybrid LLM Routing for Efficient App Feedback Classification,"
       arXiv:2507.08250

Risk tier: LOW — configuration and capability detection (no patient data).
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Degradation level enums
# ------------------------------------------------------------------

class ExtractionLevel(enum.Enum):
    """Extraction axis degradation levels (highest quality first)."""

    E1 = "docling_vision"   # Docling + Vision API
    E2 = "docling"          # Docling only
    E3 = "pdfplumber"       # pdfplumber cascade (always available)


class ClassificationLevel(enum.Enum):
    """Classification axis degradation levels (highest quality first)."""

    C1 = "neobert_rag_sonnet"    # NeoBERT + RAG + Sonnet cascade
    C2 = "neobert_sonnet"        # NeoBERT + Sonnet cascade
    C3 = "rulebased_sonnet"      # RuleBasedClassifier + Sonnet
    C4 = "neobert_rulebased"     # NeoBERT + rule-based fallback
    C5 = "rulebased_only"        # RuleBasedClassifier only (emergency)


# ------------------------------------------------------------------
# Capability detection
# ------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineCapabilities:
    """Available pipeline components detected at startup.

    Attributes:
        has_docling: docling package is importable.
        has_vision_api: Vision API key is set (PTCV-172).
        has_torch: torch and transformers packages are importable.
        has_neobert_model: NeoBERT checkpoint directory exists on disk.
        has_anthropic: anthropic package importable AND API key set.
        has_rag_index: RAG vector index directory exists on disk.
    """

    has_docling: bool
    has_vision_api: bool
    has_torch: bool
    has_neobert_model: bool
    has_anthropic: bool
    has_rag_index: bool


def detect_capabilities(
    neobert_model_path: str = "",
    rag_index_path: str = "",
) -> PipelineCapabilities:
    """Probe the runtime environment for available components.

    Uses lazy try/except imports following the pattern established in
    ``neobert_classifier._ensure_ml_deps()`` and
    ``llm_retemplater._get_client()``.

    Args:
        neobert_model_path: Path to NeoBERT checkpoint directory.
            Falls back to ``PTCV_NEOBERT_MODEL`` env var, then
            ``data/models/neobert`` default.
        rag_index_path: Path to RAG vector index directory.  Falls
            back to ``PTCV_RAG_INDEX`` env var.

    Returns:
        PipelineCapabilities with all checks resolved.  Never raises.
    """
    # -- Docling --
    has_docling = False
    try:
        import docling  # noqa: F401
        has_docling = True
    except ImportError:
        pass

    # -- Vision API (PTCV-172 placeholder) --
    has_vision_api = bool(os.environ.get("PTCV_VISION_API_KEY"))

    # -- torch + transformers --
    has_torch = False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        has_torch = True
    except ImportError:
        pass

    # -- NeoBERT checkpoint --
    model_dir = (
        neobert_model_path
        or os.environ.get("PTCV_NEOBERT_MODEL", "")
        or "data/models/neobert"
    )
    has_neobert_model = has_torch and Path(model_dir).exists()

    # -- Anthropic SDK + API key --
    has_anthropic = False
    try:
        import anthropic  # noqa: F401
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    except ImportError:
        pass

    # -- RAG index (requires path + sentence-transformers + faiss) --
    index_path = rag_index_path or os.environ.get("PTCV_RAG_INDEX", "")
    has_rag_index = bool(index_path) and Path(index_path).exists()
    if has_rag_index:
        try:
            import sentence_transformers  # noqa: F401
            import faiss  # noqa: F401
        except ImportError:
            has_rag_index = False

    return PipelineCapabilities(
        has_docling=has_docling,
        has_vision_api=has_vision_api,
        has_torch=has_torch,
        has_neobert_model=has_neobert_model,
        has_anthropic=has_anthropic,
        has_rag_index=has_rag_index,
    )


# ------------------------------------------------------------------
# Strategy selection
# ------------------------------------------------------------------

def select_extraction_level(
    capabilities: PipelineCapabilities,
    force_level: Optional[ExtractionLevel] = None,
) -> ExtractionLevel:
    """Select the highest available extraction level.

    Args:
        capabilities: Detected runtime capabilities.
        force_level: Override for ``--force-extraction-level`` CLI flag.

    Returns:
        Selected ExtractionLevel.
    """
    if force_level is not None:
        return force_level

    if capabilities.has_docling and capabilities.has_vision_api:
        return ExtractionLevel.E1
    if capabilities.has_docling:
        return ExtractionLevel.E2
    return ExtractionLevel.E3


def select_classification_level(
    capabilities: PipelineCapabilities,
    force_level: Optional[ClassificationLevel] = None,
) -> ClassificationLevel:
    """Select the highest available classification level.

    Args:
        capabilities: Detected runtime capabilities.
        force_level: Override for ``--force-classification-level`` CLI flag.

    Returns:
        Selected ClassificationLevel.
    """
    if force_level is not None:
        return force_level

    has_nb = capabilities.has_neobert_model
    has_sonnet = capabilities.has_anthropic
    has_rag = capabilities.has_rag_index

    if has_nb and has_rag and has_sonnet:
        return ClassificationLevel.C1
    if has_nb and has_sonnet:
        return ClassificationLevel.C2
    if has_sonnet:
        return ClassificationLevel.C3
    if has_nb:
        return ClassificationLevel.C4
    return ClassificationLevel.C5


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

_EXTRACTION_LABELS: dict[ExtractionLevel, str] = {
    ExtractionLevel.E1: "Docling+Vision",
    ExtractionLevel.E2: "Docling",
    ExtractionLevel.E3: "pdfplumber cascade",
}

_CLASSIFICATION_LABELS: dict[ClassificationLevel, str] = {
    ClassificationLevel.C1: "Full cascade (NeoBERT+RAG+Sonnet)",
    ClassificationLevel.C2: "NeoBERT+Sonnet",
    ClassificationLevel.C3: "RuleBased+Sonnet",
    ClassificationLevel.C4: "NeoBERT+RuleBased",
    ClassificationLevel.C5: "RuleBased only",
}


def log_pipeline_strategy(
    extraction_level: ExtractionLevel,
    classification_level: ClassificationLevel,
    capabilities: PipelineCapabilities,
) -> None:
    """Log selected degradation levels prominently at pipeline start.

    Emits INFO for the selected levels and WARNING when operating
    at degraded levels on either axis.
    """
    ext_label = _EXTRACTION_LABELS[extraction_level]
    cls_label = _CLASSIFICATION_LABELS[classification_level]

    logger.info(
        "Pipeline strategy: Extraction: %s (%s), "
        "Classification: %s (%s)",
        extraction_level.name, ext_label,
        classification_level.name, cls_label,
    )

    # Quality warnings for degraded levels
    if extraction_level == ExtractionLevel.E3:
        logger.warning(
            "DEGRADED EXTRACTION (%s): Using pdfplumber cascade. "
            "Table quality may be reduced for complex layouts.",
            extraction_level.name,
        )

    if classification_level in (
        ClassificationLevel.C4,
        ClassificationLevel.C5,
    ):
        logger.warning(
            "DEGRADED CLASSIFICATION (%s): No LLM available. "
            "Section assignment accuracy will be lower.",
            classification_level.name,
        )

    if classification_level == ClassificationLevel.C5:
        logger.warning(
            "EMERGENCY MODE (C5): Rule-based only. "
            "Zero external dependencies. Confidence scores "
            "will be significantly lower.",
        )

    logger.info(
        "Capabilities: docling=%s vision=%s torch=%s "
        "neobert=%s anthropic=%s rag=%s",
        capabilities.has_docling,
        capabilities.has_vision_api,
        capabilities.has_torch,
        capabilities.has_neobert_model,
        capabilities.has_anthropic,
        capabilities.has_rag_index,
    )
