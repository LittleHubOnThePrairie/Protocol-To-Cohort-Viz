"""Registry metadata integration for ClinicalTrials.gov and ICH E6(R3).

PTCV-193: Pre-populate RAG index with structured registry metadata
to improve classification confidence for PDF-extracted sections.
"""

from __future__ import annotations

from .cross_validator import (
    CrossValidationReport,
    CrossValidator,
    SectionStatus,
    SectionValidation,
)
from .fallback import (
    RegistryEnrichmentResult,
    RegistryPipelineMetrics,
    collect_batch_metrics,
    detect_nct_id,
    try_registry_enrichment,
)
from .gdelt_adapter import GdeltAdapter, GdeltSearchResult, NewswireArticle
from .ich_mapper import MappedRegistrySection, MetadataToIchMapper
from .literature_enricher import EnrichmentResult, LiteratureEnricher
from .metadata_fetcher import RegistryMetadataFetcher
from .newsdata_adapter import NewsdataAdapter, NewsdataSearchResult
from .pubmed_adapter import (
    EndpointResult,
    PubmedAdapter,
    PubmedArticle,
    PubmedSearchResult,
    extract_endpoints,
)
from .rag_seeder import RegistryRagSeeder, SeedResult

__all__ = [
    "CrossValidationReport",
    "CrossValidator",
    "EnrichmentResult",
    "EndpointResult",
    "GdeltAdapter",
    "GdeltSearchResult",
    "LiteratureEnricher",
    "MappedRegistrySection",
    "MetadataToIchMapper",
    "NewsdataAdapter",
    "NewsdataSearchResult",
    "NewswireArticle",
    "PubmedAdapter",
    "PubmedArticle",
    "PubmedSearchResult",
    "RegistryEnrichmentResult",
    "RegistryMetadataFetcher",
    "RegistryPipelineMetrics",
    "RegistryRagSeeder",
    "SectionStatus",
    "SectionValidation",
    "SeedResult",
    "collect_batch_metrics",
    "detect_nct_id",
    "extract_endpoints",
    "try_registry_enrichment",
]
