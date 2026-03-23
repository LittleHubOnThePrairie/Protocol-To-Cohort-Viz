"""Gap recovery pipeline for failed query extractions (PTCV-258).

When Stage 3 extraction produces no result for a query, this module
retries with progressively broader strategies before recording the gap:

1. **Alternative strategy** — try different extraction types
   (e.g. list/table when text_long failed).
2. **Adjacent section search** — broaden to neighboring ICH sections
   (e.g. check B.3 and B.5 when B.4 route was empty).
3. **Registry fallback** — check CT.gov for a direct answer.
4. **Record exhaustive gap** — mark gap with attempted strategies.

Risk tier: LOW — read-only text analysis + cached registry lookup.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from .query_schema import AppendixBQuery
from .section_matcher import MatchConfidence

logger = logging.getLogger(__name__)

# Alternative strategies to try when the primary strategy fails.
# Ordered by expected yield (table/list are the most common
# format mismatches for text_long queries).
_ALTERNATIVE_STRATEGIES: dict[str, list[str]] = {
    "text_long": ["list", "table"],
    "text_short": ["text_long"],
    "list": ["text_long"],
    "table": ["text_long", "list"],
    "statement": ["text_long"],
}

# ICH section adjacency map: section code → neighboring codes to search.
_ADJACENT_SECTIONS: dict[str, list[str]] = {
    "B.1": ["B.2"],
    "B.2": ["B.1", "B.3"],
    "B.3": ["B.2", "B.4"],
    "B.4": ["B.3", "B.5", "B.7"],
    "B.5": ["B.4", "B.6"],
    "B.6": ["B.5", "B.7"],
    "B.7": ["B.6", "B.4", "B.8"],
    "B.8": ["B.7", "B.9", "B.4"],
    "B.9": ["B.8", "B.7"],
    "B.10": ["B.4", "B.3"],
    "B.11": ["B.12", "B.14"],
    "B.12": ["B.11", "B.14"],
    "B.13": ["B.14"],
    "B.14": ["B.13", "B.12"],
    "B.15": ["B.16"],
    "B.16": ["B.15"],
}


@dataclasses.dataclass
class RecoveryAttempt:
    """Record of a single recovery attempt.

    Attributes:
        strategy: Recovery strategy name.
        success: Whether the attempt produced a result.
        detail: Optional detail (e.g. which section was searched).
    """

    strategy: str
    success: bool
    detail: str = ""


@dataclasses.dataclass
class RecoveryResult:
    """Result of the gap recovery cascade.

    Attributes:
        recovered: True if a result was recovered.
        content: Extracted content (empty if not recovered).
        confidence: Extraction confidence (0.0 if not recovered).
        method: Extraction method tag with recovery suffix.
        source_section: Which section the content was found in.
        attempts: List of all recovery attempts tried.
    """

    recovered: bool
    content: str = ""
    confidence: float = 0.0
    method: str = ""
    source_section: str = ""
    attempts: list[RecoveryAttempt] = dataclasses.field(
        default_factory=list,
    )

    @property
    def exhaustive_search(self) -> bool:
        """True if all recovery strategies were attempted."""
        return not self.recovered and len(self.attempts) > 0

    @property
    def strategies_attempted(self) -> list[str]:
        """Names of all strategies that were attempted."""
        return [a.strategy for a in self.attempts]


# Extraction function type: (text, query) → (content, confidence, method)
ExtractorFn = Callable[
    [str, AppendixBQuery], tuple[str, float, str]
]


def recover_gap(
    query: AppendixBQuery,
    routes: dict[str, tuple[str, str, MatchConfidence]],
    type_dispatch: dict[str, ExtractorFn],
    registry_metadata: Optional[dict[str, Any]] = None,
) -> RecoveryResult:
    """Run gap recovery cascade for a failed query extraction.

    Attempts recovery in order:
    1. Alternative extraction strategy on existing routed content
    2. Adjacent section search with primary + alternative strategies
    3. Registry fallback from CT.gov cached metadata

    Args:
        query: The query that failed extraction.
        routes: Route map from Stage 2 classification
            ``{section_code: (content, source, confidence)}``.
        type_dispatch: The ``_TYPE_DISPATCH`` dict from query_extractor.
        registry_metadata: Optional cached CT.gov JSON for registry
            fallback (from ``RegistryMetadataFetcher``).

    Returns:
        RecoveryResult with recovery outcome and attempt log.
    """
    attempts: list[RecoveryAttempt] = []

    # --- Level 1: Alternative strategy on existing content ---
    result = _try_alternative_strategy(
        query, routes, type_dispatch, attempts,
    )
    if result is not None:
        return RecoveryResult(
            recovered=True,
            content=result[0],
            confidence=result[1],
            method=f"{result[2]}(gap_recovery)",
            source_section=result[3],
            attempts=attempts,
        )

    # --- Level 2: Adjacent section search ---
    result = _try_adjacent_sections(
        query, routes, type_dispatch, attempts,
    )
    if result is not None:
        return RecoveryResult(
            recovered=True,
            content=result[0],
            confidence=result[1],
            method=f"{result[2]}(adjacent_recovery)",
            source_section=result[3],
            attempts=attempts,
        )

    # --- Level 3: Registry fallback ---
    result = _try_registry_fallback(
        query, registry_metadata, attempts,
    )
    if result is not None:
        return RecoveryResult(
            recovered=True,
            content=result[0],
            confidence=result[1],
            method="registry_direct(gap_recovery)",
            source_section="[REGISTRY]",
            attempts=attempts,
        )

    # All recovery levels failed
    return RecoveryResult(
        recovered=False,
        attempts=attempts,
    )


# ------------------------------------------------------------------
# Level 1: Alternative extraction strategy
# ------------------------------------------------------------------


def _try_alternative_strategy(
    query: AppendixBQuery,
    routes: dict[str, tuple[str, str, MatchConfidence]],
    type_dispatch: dict[str, ExtractorFn],
    attempts: list[RecoveryAttempt],
) -> tuple[str, float, str, str] | None:
    """Try alternative extraction types on the routed content."""
    alternatives = _ALTERNATIVE_STRATEGIES.get(
        query.expected_type, [],
    )
    if not alternatives:
        return None

    # Find content from sub-section or parent route
    content, source = _find_routed_content(query, routes)
    if not content:
        return None

    for alt_type in alternatives:
        extractor_fn = type_dispatch.get(alt_type)
        if extractor_fn is None:
            continue

        extracted, confidence, method = extractor_fn(content, query)
        attempt = RecoveryAttempt(
            strategy=f"alt_strategy:{alt_type}",
            success=bool(extracted),
            detail=f"tried {alt_type} on {source}",
        )
        attempts.append(attempt)

        if extracted and confidence > 0.0:
            logger.info(
                "Gap recovery L1: %s recovered via %s strategy "
                "(conf=%.3f, source=%s)",
                query.query_id, alt_type, confidence, source,
            )
            return extracted, confidence, method, source

    return None


# ------------------------------------------------------------------
# Level 2: Adjacent section search
# ------------------------------------------------------------------


def _try_adjacent_sections(
    query: AppendixBQuery,
    routes: dict[str, tuple[str, str, MatchConfidence]],
    type_dispatch: dict[str, ExtractorFn],
    attempts: list[RecoveryAttempt],
) -> tuple[str, float, str, str] | None:
    """Search adjacent ICH sections for query-relevant content."""
    parent_code = query.schema_section  # e.g. "B.4"
    neighbors = _ADJACENT_SECTIONS.get(parent_code, [])
    if not neighbors:
        return None

    # Try primary strategy first, then alternatives
    strategies = [query.expected_type] + _ALTERNATIVE_STRATEGIES.get(
        query.expected_type, [],
    )

    for neighbor in neighbors:
        if neighbor not in routes:
            continue

        content, source, _ = routes[neighbor]
        if not content or not content.strip():
            continue

        for strategy in strategies:
            extractor_fn = type_dispatch.get(strategy)
            if extractor_fn is None:
                continue

            extracted, confidence, method = extractor_fn(
                content, query,
            )
            attempt = RecoveryAttempt(
                strategy=f"adjacent:{neighbor}:{strategy}",
                success=bool(extracted),
                detail=f"tried {strategy} on {neighbor}",
            )
            attempts.append(attempt)

            if extracted and confidence > 0.0:
                # Apply a penalty for cross-section discovery
                confidence = round(confidence * 0.80, 4)
                logger.info(
                    "Gap recovery L2: %s found in adjacent %s "
                    "via %s (conf=%.3f)",
                    query.query_id, neighbor,
                    strategy, confidence,
                )
                return extracted, confidence, method, neighbor

    return None


# ------------------------------------------------------------------
# Level 3: Registry fallback
# ------------------------------------------------------------------


def _try_registry_fallback(
    query: AppendixBQuery,
    registry_metadata: Optional[dict[str, Any]],
    attempts: list[RecoveryAttempt],
) -> tuple[str, float, str] | None:
    """Check CT.gov registry for a direct answer to this query."""
    if not registry_metadata:
        return None

    try:
        from ..registry.query_injector import (
            QUERY_TO_REGISTRY_FIELD,
            navigate_json,
        )
    except ImportError:
        return None

    field_path = QUERY_TO_REGISTRY_FIELD.get(query.query_id)
    if not field_path:
        return None

    value = navigate_json(registry_metadata, field_path)
    attempt = RecoveryAttempt(
        strategy="registry_fallback",
        success=value is not None,
        detail=f"field={field_path}",
    )
    attempts.append(attempt)

    if value is None:
        return None

    # Format the registry value as text
    if isinstance(value, list):
        text = "\n".join(
            str(item) if not isinstance(item, dict)
            else ", ".join(f"{k}: {v}" for k, v in item.items())
            for item in value
        )
    elif isinstance(value, dict):
        text = ", ".join(f"{k}: {v}" for k, v in value.items())
    else:
        text = str(value)

    if not text.strip():
        return None

    logger.info(
        "Gap recovery L3: %s recovered from registry "
        "field %s",
        query.query_id, field_path,
    )
    return text.strip(), 0.85, "registry_direct"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _find_routed_content(
    query: AppendixBQuery,
    routes: dict[str, tuple[str, str, MatchConfidence]],
) -> tuple[str, str]:
    """Find content from sub-section or parent route.

    Returns:
        Tuple of (content, source_description). Empty strings if
        no route found.
    """
    # Sub-section route
    if query.section_id in routes:
        content, source, _ = routes[query.section_id]
        return content, source

    # Parent route
    if query.schema_section in routes:
        content, source, _ = routes[query.schema_section]
        return content, source

    return "", ""
