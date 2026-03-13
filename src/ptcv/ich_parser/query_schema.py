"""ICH E6(R3) Appendix B query schema loader (PTCV-88).

Loads the structured query/checklist decomposition of ICH E6(R3)
Appendix B from ``data/templates/appendix_b_queries.yaml``.  Each
Appendix B sub-section is expressed as one or more queries that
describe what content a compliant protocol should contain.

Used by the query-driven extraction pipeline (PTCV-87) to search
protocol documents for specific content rather than classifying
text blocks.

Risk tier: LOW — static schema, no I/O beyond YAML load.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Sequence

import yaml

logger = logging.getLogger(__name__)

# Default path — relative to project root.
_DEFAULT_QUERY_PATH = (
    Path(__file__).resolve().parents[3]  # src/ptcv/ich_parser -> root
    / "data"
    / "templates"
    / "appendix_b_queries.yaml"
)

# ---------------------------------------------------------------------------
# Natural sort key for section codes
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"(\d+)")


def section_sort_key(code: str) -> list[int]:
    """Return a sort key for an ICH section code like ``"B.10.2"``.

    Splits on ``"."`` and converts numeric parts to ``int`` so that
    ``"B.10"`` sorts after ``"B.9"`` rather than before ``"B.2"``.

    Non-numeric parts (e.g. the leading ``"B"``) are mapped to ``0``.
    """
    return [int(p) if p.isdigit() else 0 for p in code.split(".")]


# Valid expected_type values.
EXPECTED_TYPES = frozenset({
    "text_short",
    "text_long",
    "identifier",
    "date",
    "list",
    "table",
    "numeric",
    "enum",
    "statement",
})


@dataclasses.dataclass(frozen=True)
class AppendixBQuery:
    """A single structured query for an Appendix B section.

    Attributes:
        query_id: Unique query identifier (e.g. ``"B.1.1.q1"``).
        section_id: Appendix B sub-section (e.g. ``"B.1.1"``).
        parent_section: Top-level Appendix B section (e.g. ``"B.1"``).
        schema_section: Corresponding code in ``ich_e6r3_schema.yaml``
            (e.g. ``"B.1"``).  Useful for mapping queries back to the
            existing classifier's section codes.
        query_text: Natural-language question describing what content
            should be found.
        expected_type: Content type hint for the extraction engine.
            One of :data:`EXPECTED_TYPES`.
        required: ``True`` if ICH mandates this content; ``False`` if
            recommended or conditional.
    """

    query_id: str
    section_id: str
    parent_section: str
    schema_section: str
    query_text: str
    expected_type: str
    required: bool


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_QUERY_CACHE: list[AppendixBQuery] | None = None


def load_query_schema(
    path: Path | None = None,
) -> list[AppendixBQuery]:
    """Load and cache the Appendix B query schema from YAML.

    Args:
        path: Override path for testing.  Uses the default
            ``data/templates/appendix_b_queries.yaml`` if *None*.

    Returns:
        List of :class:`AppendixBQuery` instances, one per query.

    Raises:
        FileNotFoundError: If the YAML file is missing.
        ValueError: If a query has an invalid ``expected_type``.
    """
    global _QUERY_CACHE
    if _QUERY_CACHE is not None and path is None:
        return _QUERY_CACHE

    query_path = path or _DEFAULT_QUERY_PATH
    with open(query_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    queries: list[AppendixBQuery] = []
    for entry in raw.get("queries", []):
        expected_type = entry["expected_type"]
        if expected_type not in EXPECTED_TYPES:
            raise ValueError(
                f"Invalid expected_type {expected_type!r} in query "
                f"{entry.get('query_id', '?')}. "
                f"Must be one of {sorted(EXPECTED_TYPES)}"
            )

        queries.append(
            AppendixBQuery(
                query_id=entry["query_id"],
                section_id=str(entry["section_id"]),
                parent_section=str(entry["parent_section"]),
                schema_section=str(entry["schema_section"]),
                query_text=entry["query_text"],
                expected_type=expected_type,
                required=bool(entry.get("required", True)),
            )
        )

    if path is None:
        _QUERY_CACHE = queries

    logger.info(
        "Loaded %d Appendix B queries across %d parent sections",
        len(queries),
        len({q.parent_section for q in queries}),
    )
    return queries


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def get_queries_for_section(
    section_code: str,
    queries: Sequence[AppendixBQuery] | None = None,
) -> list[AppendixBQuery]:
    """Return queries whose ``parent_section`` matches *section_code*.

    Args:
        section_code: Top-level section code (e.g. ``"B.4"``).
        queries: Pre-loaded query list; loads from YAML if *None*.

    Returns:
        Filtered list of queries for the given section.
    """
    if queries is None:
        queries = load_query_schema()
    return [q for q in queries if q.parent_section == section_code]


def get_queries_by_schema_section(
    schema_code: str,
    queries: Sequence[AppendixBQuery] | None = None,
) -> list[AppendixBQuery]:
    """Return queries whose ``schema_section`` matches *schema_code*.

    This maps to the ``ich_e6r3_schema.yaml`` section codes
    (B.1 through B.16), which is useful when bridging the query schema
    to the classifier pipeline.

    Args:
        schema_code: Schema section code (e.g. ``"B.14"``).
        queries: Pre-loaded query list; loads from YAML if *None*.

    Returns:
        Filtered list of queries for the given schema section.
    """
    if queries is None:
        queries = load_query_schema()
    return [q for q in queries if q.schema_section == schema_code]


def get_queries_by_section_id(
    section_id: str,
    queries: Sequence[AppendixBQuery] | None = None,
) -> list[AppendixBQuery]:
    """Return queries whose ``section_id`` matches *section_id*.

    Args:
        section_id: Sub-section code (e.g. ``"B.2.1"``).
        queries: Pre-loaded query list; loads from YAML if *None*.

    Returns:
        Filtered list of queries for the given sub-section.
    """
    if queries is None:
        queries = load_query_schema()
    return [q for q in queries if q.section_id == section_id]


def get_required_queries(
    queries: Sequence[AppendixBQuery] | None = None,
) -> list[AppendixBQuery]:
    """Return only the required (mandatory) queries.

    Args:
        queries: Pre-loaded query list; loads from YAML if *None*.

    Returns:
        Filtered list of required queries.
    """
    if queries is None:
        queries = load_query_schema()
    return [q for q in queries if q.required]


def get_parent_sections(
    queries: Sequence[AppendixBQuery] | None = None,
) -> list[str]:
    """Return sorted list of unique parent section codes.

    Args:
        queries: Pre-loaded query list; loads from YAML if *None*.

    Returns:
        Sorted unique parent section codes (e.g.
        ``["B.1", "B.2", ..., "B.16"]``).
    """
    if queries is None:
        queries = load_query_schema()
    return sorted({q.parent_section for q in queries}, key=section_sort_key)


def _reset_cache() -> None:
    """Clear the module-level query cache (testing only)."""
    global _QUERY_CACHE
    _QUERY_CACHE = None
