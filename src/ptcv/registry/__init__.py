"""Registry metadata integration for ClinicalTrials.gov and ICH E6(R3).

PTCV-193: Pre-populate RAG index with structured registry metadata
to improve classification confidence for PDF-extracted sections.
"""

from __future__ import annotations

import importlib
from typing import Any

from .fallback import (
    RegistryEnrichmentResult,
    RegistryPipelineMetrics,
    collect_batch_metrics,
    detect_nct_id,
    try_registry_enrichment,
)
from .gdelt_adapter import GdeltAdapter, GdeltSearchResult, NewswireArticle
from .ich_mapper import MappedRegistrySection, MetadataToIchMapper
from .newsdata_adapter import NewsdataAdapter, NewsdataSearchResult
from .metadata_fetcher import RegistryMetadataFetcher
from .rag_seeder import RegistryRagSeeder, SeedResult

__all__ = [
    "CrossValidationReport",
    "CrossValidator",
    "GdeltAdapter",
    "GdeltSearchResult",
    "MappedRegistrySection",
    "MetadataToIchMapper",
    "NewsdataAdapter",
    "NewsdataSearchResult",
    "NewswireArticle",
    "RegistryEnrichmentResult",
    "RegistryMetadataFetcher",
    "RegistryPipelineMetrics",
    "RegistryRagSeeder",
    "SectionStatus",
    "SectionValidation",
    "SeedResult",
    "collect_batch_metrics",
    "detect_nct_id",
    "try_registry_enrichment",
]


def __getattr__(name: str) -> Any:
    """Lazy-load cross_validator to avoid pulling in fitz/PyMuPDF."""
    _cross_validator_names = {
        "CrossValidationReport",
        "CrossValidator",
        "SectionStatus",
        "SectionValidation",
    }
    if name in _cross_validator_names:
        mod = importlib.import_module(".cross_validator", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
