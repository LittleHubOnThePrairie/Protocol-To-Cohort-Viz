"""Query-level registry injection for low-confidence extractions (PTCV-259).

Replaces the coarse route-level injection (PTCV-230) with fine-grained
query-level injection that provides direct CT.gov answers for specific
subsection queries that have LOW CONFIDENCE extractions.

Each query is mapped to a specific CT.gov JSON field path. When a query's
extraction confidence is below the threshold, the direct registry answer
replaces it — more precise than route-level injection because no
irrelevant content is mixed in.

Risk tier: LOW — read-only data lookup, no API calls, no data mutation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Registry confidence for direct field matches (high — structured data).
_REGISTRY_CONFIDENCE = 0.85

# Queries below this confidence are candidates for registry injection.
_LOW_CONFIDENCE_THRESHOLD = 0.70

# Extraction method tag for registry-injected answers.
_EXTRACTION_METHOD = "registry_direct"

# ---------------------------------------------------------------------------
# Query → CT.gov field path mapping
# ---------------------------------------------------------------------------

# Maps query IDs to dotted paths within the CT.gov protocolSection JSON.
# Each path resolves to a specific value (string, list, or dict) that
# directly answers the query.
QUERY_TO_REGISTRY_FIELD: dict[str, str] = {
    "B.1.1.q1": "protocolSection.identificationModule.officialTitle",
    "B.1.2.q1": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "B.1.3.q1": "protocolSection.contactsLocationsModule.overallOfficials",
    # PTCV-285: B.3 enriched — briefSummary is richer than briefTitle
    # for describing trial purpose/objectives.
    "B.3.q1": "protocolSection.descriptionModule.briefSummary",
    "B.3.q2": "protocolSection.outcomesModule.secondaryOutcomes",
    "B.4.1.q1": "protocolSection.outcomesModule.primaryOutcomes",
    "B.4.1.q2": "protocolSection.outcomesModule.secondaryOutcomes",
    "B.5.1.q1": "protocolSection.eligibilityModule.eligibilityCriteria",
    "B.10.q1": "protocolSection.designModule.enrollmentInfo",
}


# ---------------------------------------------------------------------------
# JSON navigation
# ---------------------------------------------------------------------------


def navigate_json(data: dict[str, Any], path: str) -> Any:
    """Navigate a nested dict by dotted path.

    Args:
        data: Root JSON dict (e.g., full CT.gov response).
        path: Dotted path (e.g., ``"protocolSection.identificationModule.officialTitle"``).

    Returns:
        The value at the path, or None if any key is missing.
    """
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def format_registry_answer(value: Any, field_path: str = "") -> str:
    """Format a registry value as human-readable text.

    Handles strings, lists of dicts (outcomes, officials), and
    simple dicts (enrollment info).

    Args:
        value: Raw value from CT.gov JSON.
        field_path: Original field path (for context-specific formatting).

    Returns:
        Formatted string suitable for query answer content.
    """
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        if not value:
            return ""
        # List of dicts (outcomes, officials, etc.)
        if isinstance(value[0], dict):
            return _format_dict_list(value, field_path)
        # List of strings
        return "\n".join(f"- {item}" for item in value)

    if isinstance(value, dict):
        return _format_single_dict(value, field_path)

    return str(value)


def _format_dict_list(items: list[dict], field_path: str) -> str:
    """Format a list of dicts as numbered items."""
    lines: list[str] = []
    for i, item in enumerate(items, 1):
        if "measure" in item:
            # Outcomes format
            parts = [f"{i}. {item['measure']}"]
            if item.get("timeFrame"):
                parts.append(f"   Time Frame: {item['timeFrame']}")
            if item.get("description"):
                parts.append(f"   Description: {item['description']}")
            lines.append("\n".join(parts))
        elif "name" in item:
            # Officials format
            role = item.get("role", "")
            affiliation = item.get("affiliation", "")
            entry = item["name"]
            if role:
                entry += f" ({role})"
            if affiliation:
                entry += f", {affiliation}"
            lines.append(f"- {entry}")
        else:
            # Generic dict
            parts = [f"{k}: {v}" for k, v in item.items() if v]
            lines.append(f"{i}. " + ", ".join(parts))
    return "\n".join(lines)


def _format_single_dict(item: dict, field_path: str) -> str:
    """Format a single dict as key-value pairs."""
    if "count" in item and "type" in item:
        # Enrollment info
        return f"{item['count']} ({item['type']})"
    parts = [f"{k}: {v}" for k, v in item.items() if v]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main injection function
# ---------------------------------------------------------------------------


def inject_registry_answers(
    extractions: list,
    registry_metadata: dict[str, Any],
    confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
) -> list:
    """Replace low-confidence extractions with direct registry answers.

    For each extraction below the confidence threshold, checks if a
    direct registry field mapping exists. If the registry has a
    non-empty answer, replaces the extraction with a high-confidence
    registry-sourced result.

    High-confidence PDF extractions are never overridden.

    Args:
        extractions: List of ``QueryExtraction`` objects (frozen
            dataclasses — replaced, not mutated).
        registry_metadata: Full CT.gov JSON response dict.
        confidence_threshold: Extractions below this are candidates
            for injection. Defaults to 0.70.

    Returns:
        New list of ``QueryExtraction`` objects with registry
        injections applied. Original list is not mutated.
    """
    import dataclasses

    if not registry_metadata:
        return list(extractions)

    result: list = []
    injected_count = 0

    for extraction in extractions:
        query_id = extraction.query_id
        field_path = QUERY_TO_REGISTRY_FIELD.get(query_id)

        # Skip if no registry mapping for this query
        if not field_path:
            result.append(extraction)
            continue

        # Skip if extraction is already high confidence
        if extraction.confidence >= confidence_threshold:
            result.append(extraction)
            continue

        # Look up the registry value
        value = navigate_json(registry_metadata, field_path)
        if value is None or value == "" or value == []:
            result.append(extraction)
            continue

        # Format the registry answer
        formatted = format_registry_answer(value, field_path)
        if not formatted.strip():
            result.append(extraction)
            continue

        # Replace with registry-sourced extraction
        replacement = dataclasses.replace(
            extraction,
            content=formatted,
            confidence=_REGISTRY_CONFIDENCE,
            extraction_method=_EXTRACTION_METHOD,
            source_section=f"registry:{field_path}",
            verbatim_content=extraction.content,
        )
        result.append(replacement)
        injected_count += 1

        logger.info(
            "PTCV-259: Injected registry answer for %s "
            "(was %.2f → %.2f via %s)",
            query_id,
            extraction.confidence,
            _REGISTRY_CONFIDENCE,
            field_path,
        )

    if injected_count:
        logger.info(
            "PTCV-259: Query-level registry injection: "
            "%d/%d queries replaced",
            injected_count,
            len(extractions),
        )

    return result


__all__ = [
    "QUERY_TO_REGISTRY_FIELD",
    "format_registry_answer",
    "inject_registry_answers",
    "navigate_json",
]
