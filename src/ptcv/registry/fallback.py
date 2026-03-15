"""Graceful degradation utilities for registry metadata pipeline.

PTCV-199: Ensures the registry metadata pipeline degrades gracefully
when ClinicalTrials.gov data is unavailable — covering EU-CTR protocols
(no NCT ID), API outages, withdrawn trials, and incomplete metadata.

Provides:
- NCT ID detection from protocol metadata/filenames
- Safe orchestration wrapper that never raises
- Pipeline metrics tracking registry coverage

Risk tier: LOW — utility functions, no API calls or data mutation.
"""

import dataclasses
import logging
import re
from typing import Any, Optional

from .ich_mapper import MappedRegistrySection, MetadataToIchMapper
from .metadata_fetcher import RegistryMetadataFetcher

logger = logging.getLogger(__name__)

_NCT_PATTERN = re.compile(r"NCT\d{8}")


@dataclasses.dataclass
class RegistryEnrichmentResult:
    """Result of attempting registry enrichment for one protocol.

    Attributes:
        registry_id: NCT ID if detected, else empty string.
        registry_available: Whether registry metadata was fetched.
        registry_sections_mapped: Number of ICH sections mapped.
        mapped_sections: List of mapped registry sections (empty
            if unavailable).
        skip_reason: Why registry was skipped (empty if available).
    """

    registry_id: str = ""
    registry_available: bool = False
    registry_sections_mapped: int = 0
    mapped_sections: list[MappedRegistrySection] = dataclasses.field(
        default_factory=list
    )
    skip_reason: str = ""


@dataclasses.dataclass
class RegistryPipelineMetrics:
    """Aggregate metrics for registry coverage across a batch.

    Attributes:
        total_protocols: Total protocols processed.
        registry_enriched: Protocols that received registry data.
        registry_skipped: Protocols without registry enrichment.
        registry_coverage: Fraction with registry data (0.0–1.0).
        per_protocol: Per-protocol enrichment results.
    """

    total_protocols: int = 0
    registry_enriched: int = 0
    registry_skipped: int = 0
    registry_coverage: float = 0.0
    per_protocol: list[RegistryEnrichmentResult] = dataclasses.field(
        default_factory=list
    )

    def update_coverage(self) -> None:
        """Recalculate coverage from per-protocol results."""
        self.total_protocols = len(self.per_protocol)
        self.registry_enriched = sum(
            1 for r in self.per_protocol if r.registry_available
        )
        self.registry_skipped = (
            self.total_protocols - self.registry_enriched
        )
        self.registry_coverage = (
            self.registry_enriched / self.total_protocols
            if self.total_protocols > 0
            else 0.0
        )


def detect_nct_id(
    registry_id: str = "",
    filename: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Extract an NCT ID from available protocol identifiers.

    Checks (in order):
    1. ``registry_id`` if it matches NCT pattern
    2. ``filename`` for an embedded NCT ID
    3. ``metadata`` dict for ``nctId`` or ``registry_id`` keys

    Args:
        registry_id: Protocol registry identifier string.
        filename: Protocol filename (e.g. ``NCT01512251_1.0.pdf``).
        metadata: Optional protocol metadata dict.

    Returns:
        NCT ID string or None if no NCT ID found.
    """
    # Check registry_id directly
    if registry_id:
        match = _NCT_PATTERN.search(registry_id)
        if match:
            return match.group(0)

    # Check filename
    if filename:
        match = _NCT_PATTERN.search(filename)
        if match:
            return match.group(0)

    # Check metadata dict
    if metadata:
        for key in ("nctId", "nct_id", "registry_id"):
            val = metadata.get(key, "")
            if val:
                match = _NCT_PATTERN.search(str(val))
                if match:
                    return match.group(0)

    return None


def try_registry_enrichment(
    registry_id: str = "",
    filename: str = "",
    metadata: Optional[dict[str, Any]] = None,
    fetcher: Optional[RegistryMetadataFetcher] = None,
    mapper: Optional[MetadataToIchMapper] = None,
) -> RegistryEnrichmentResult:
    """Attempt registry enrichment, never raising on failure.

    This is the safe entry point for the registry pipeline. It
    detects the NCT ID, fetches metadata, maps to ICH sections,
    and returns a result. Any failure at any step produces a
    result with ``registry_available=False`` and a ``skip_reason``.

    Args:
        registry_id: Protocol registry identifier.
        filename: Protocol filename.
        metadata: Optional protocol metadata dict.
        fetcher: RegistryMetadataFetcher instance (created if None).
        mapper: MetadataToIchMapper instance (created if None).

    Returns:
        RegistryEnrichmentResult (never raises).
    """
    nct_id = detect_nct_id(registry_id, filename, metadata)
    if nct_id is None:
        logger.info(
            "No NCT ID detected for registry_id=%s, "
            "filename=%s — skipping registry enrichment",
            registry_id,
            filename,
        )
        return RegistryEnrichmentResult(
            skip_reason="No NCT ID detected (non-ClinicalTrials.gov "
            "protocol or missing identifier)"
        )

    if fetcher is None:
        fetcher = RegistryMetadataFetcher()
    if mapper is None:
        mapper = MetadataToIchMapper()

    try:
        ct_metadata = fetcher.fetch(nct_id)
    except Exception:
        logger.warning(
            "Unexpected error fetching registry metadata for %s",
            nct_id,
            exc_info=True,
        )
        return RegistryEnrichmentResult(
            registry_id=nct_id,
            skip_reason="Registry metadata fetch failed",
        )

    if ct_metadata is None:
        logger.info(
            "Registry metadata unavailable for %s — "
            "pipeline continues without enrichment",
            nct_id,
        )
        return RegistryEnrichmentResult(
            registry_id=nct_id,
            skip_reason="API returned no data (404, timeout, or "
            "withdrawn trial)",
        )

    try:
        mapped = mapper.map(ct_metadata)
    except Exception:
        logger.warning(
            "Unexpected error mapping registry metadata for %s",
            nct_id,
            exc_info=True,
        )
        return RegistryEnrichmentResult(
            registry_id=nct_id,
            skip_reason="Registry metadata mapping failed",
        )

    logger.info(
        "Registry enrichment for %s: %d sections mapped",
        nct_id,
        len(mapped),
    )
    return RegistryEnrichmentResult(
        registry_id=nct_id,
        registry_available=True,
        registry_sections_mapped=len(mapped),
        mapped_sections=mapped,
    )


def collect_batch_metrics(
    results: list[RegistryEnrichmentResult],
) -> RegistryPipelineMetrics:
    """Build aggregate metrics from a list of enrichment results.

    Args:
        results: Per-protocol enrichment results.

    Returns:
        RegistryPipelineMetrics with coverage calculation.
    """
    metrics = RegistryPipelineMetrics(per_protocol=list(results))
    metrics.update_coverage()
    return metrics
