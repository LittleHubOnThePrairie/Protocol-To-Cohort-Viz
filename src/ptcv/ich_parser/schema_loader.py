"""ICH E6(R3) schema loader (PTCV-67).

Loads section definitions from ``data/templates/ich_e6r3_schema.yaml``
and exposes drop-in replacements for the hardcoded dicts that previously
lived in ``llm_retemplater.py`` and ``classifier.py``.

The YAML file is the **single source of truth** for section codes,
names, descriptions, regex patterns, and keywords.

Loaded once at module import and cached.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import yaml

# Default schema path — relative to project root
_DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]  # src/ptcv/ich_parser -> project root
    / "data"
    / "templates"
    / "ich_e6r3_schema.yaml"
)


@dataclasses.dataclass(frozen=True)
class IchSectionDef:
    """Single ICH section definition."""

    code: str
    name: str
    description: str
    requirement: str
    render_order: int
    patterns: list[str]
    keywords: list[str]


@dataclasses.dataclass(frozen=True)
class IchSchema:
    """Complete ICH E6(R3) schema."""

    version: str
    effective_date: str
    sections: dict[str, IchSectionDef]


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_SCHEMA_CACHE: IchSchema | None = None


def load_ich_schema(path: Path | None = None) -> IchSchema:
    """Load and cache the ICH schema from YAML.

    Args:
        path: Override path for testing. Uses default if *None*.

    Returns:
        Parsed ``IchSchema`` instance (cached after first call).
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None and path is None:
        return _SCHEMA_CACHE

    schema_path = path or _DEFAULT_SCHEMA_PATH
    with open(schema_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    sections: dict[str, IchSectionDef] = {}
    for code, defn in raw["sections"].items():
        sections[code] = IchSectionDef(
            code=code,
            name=defn["name"],
            description=defn["description"],
            requirement=defn.get("requirement", "mandatory"),
            render_order=defn["render_order"],
            patterns=defn.get("patterns", []),
            keywords=defn.get("keywords", []),
        )

    schema = IchSchema(
        version=raw["version"],
        effective_date=raw["effective_date"],
        sections=sections,
    )

    if path is None:
        _SCHEMA_CACHE = schema
    return schema


# ---------------------------------------------------------------------------
# Drop-in replacement functions
# ---------------------------------------------------------------------------


def get_section_defs() -> dict[str, str]:
    """Return ``{code: description}`` matching old ``_ICH_SECTION_DEFS``.

    Example::

        {"B.1": "General Information (title, sponsor, ...)", ...}
    """
    schema = load_ich_schema()
    return {code: s.description for code, s in schema.sections.items()}


def get_section_order() -> list[tuple[str, str]]:
    """Return ``[(code, name)]`` matching old ``_ICH_SECTION_ORDER``.

    Sorted by ``render_order``.
    """
    schema = load_ich_schema()
    ordered = sorted(schema.sections.values(), key=lambda s: s.render_order)
    return [(s.code, s.name) for s in ordered]


def get_classifier_sections() -> dict[str, dict[str, Any]]:
    """Return ``{code: {name, patterns, keywords}}`` matching old
    ``_ICH_SECTIONS`` in classifier.py.

    Patterns are returned as raw regex strings (no double-escaping).
    """
    schema = load_ich_schema()
    result: dict[str, dict[str, Any]] = {}
    for code, s in schema.sections.items():
        # YAML stores double-escaped patterns (\\b → literal \b).
        # Convert to single-escaped raw regex strings.
        raw_patterns = [p.replace("\\\\", "\\") for p in s.patterns]
        result[code] = {
            "name": s.name,
            "patterns": raw_patterns,
            "keywords": list(s.keywords),
        }
    return result
