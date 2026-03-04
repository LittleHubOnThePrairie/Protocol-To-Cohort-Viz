"""ICH E6(R3) schema loader (PTCV-67, PTCV-68, PTCV-69).

Loads section definitions from ``data/templates/ich_e6r3_schema.yaml``
and exposes drop-in replacements for the hardcoded dicts that previously
lived in ``llm_retemplater.py`` and ``classifier.py``.

The YAML file is the **single source of truth** for section codes,
names, descriptions, regex patterns, and keywords.

Stage-specific prompt rendering (PTCV-68) supports ``full`` and
``compact`` formats, allowing downstream stages to receive only the
ICH sections they need.

Loaded once at module import and cached.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

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
class StagePromptConfig:
    """Per-stage prompt configuration (PTCV-68)."""

    sections: list[str]
    format: str  # "full" | "compact"


@dataclasses.dataclass(frozen=True)
class IchSchema:
    """Complete ICH E6(R3) schema."""

    version: str
    effective_date: str
    sections: dict[str, IchSectionDef]
    stage_prompts: dict[str, StagePromptConfig]
    configuration: dict[str, Any]


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

    stage_prompts: dict[str, StagePromptConfig] = {}
    for stage_name, cfg in raw.get("stage_prompts", {}).items():
        stage_prompts[stage_name] = StagePromptConfig(
            sections=cfg["sections"],
            format=cfg["format"],
        )

    schema = IchSchema(
        version=raw["version"],
        effective_date=raw["effective_date"],
        sections=sections,
        stage_prompts=stage_prompts,
        configuration=raw.get("configuration", {}),
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


# ---------------------------------------------------------------------------
# Stage-specific prompt rendering (PTCV-68)
# ---------------------------------------------------------------------------

_REQ_ABBREV = {"mandatory": "M", "recommended": "R", "conditional": "C"}


def _render_full(sections: list[IchSectionDef]) -> str:
    """Render sections in the legacy prose format.

    Produces the exact same string as the old inline code::

        "  B.1: General Information (title, sponsor, ...)"
    """
    return "\n".join(
        f"  {s.code}: {s.description}"
        for s in sorted(sections, key=lambda s: s.code)
    )


def _render_compact(sections: list[IchSectionDef], version: str) -> str:
    """Render sections as compressed XML.

    Produces::

        <ich_e6r3 version="R3">
          <section code="B.4" req="M">Design: randomisation, blinding, SoA</section>
        </ich_e6r3>
    """
    lines = [f'<ich_e6r3 version="{version}">']
    for s in sorted(sections, key=lambda s: s.render_order):
        req = _REQ_ABBREV.get(s.requirement, s.requirement)
        # Compact description: name + top keywords (comma-separated)
        top_kw = ", ".join(s.keywords[:5])
        lines.append(
            f'  <section code="{s.code}" req="{req}">'
            f"{s.name}: {top_kw}</section>"
        )
    lines.append("</ich_e6r3>")
    return "\n".join(lines)


def get_stage_prompt(stage: str) -> str:
    """Return the rendered ICH section prompt for a named pipeline stage.

    If *stage* is not found in the YAML ``stage_prompts`` block, falls
    back to the ``retemplater`` configuration (all 14 sections, full
    format) and logs a warning.
    """
    schema = load_ich_schema()

    cfg = schema.stage_prompts.get(stage)
    if cfg is None:
        logger.warning(
            "Unknown stage %r — falling back to retemplater prompt",
            stage,
        )
        cfg = schema.stage_prompts.get("retemplater")
        if cfg is None:
            # Absolute fallback: all sections, full format
            all_secs = sorted(
                schema.sections.values(), key=lambda s: s.code,
            )
            return _render_full(list(all_secs))

    selected = [
        schema.sections[code]
        for code in cfg.sections
        if code in schema.sections
    ]

    if cfg.format == "compact":
        return _render_compact(selected, schema.version)
    return _render_full(selected)


# ---------------------------------------------------------------------------
# Configuration accessors (PTCV-69)
# ---------------------------------------------------------------------------
# Compiled-pattern caches — lazily populated on first call.
_SOA_PATTERN_CACHE: re.Pattern[str] | None = None
_BOILERPLATE_PATTERN_CACHE: re.Pattern[str] | None = None


def get_review_threshold(section_code: str | None = None) -> float:
    """Return the review-confidence threshold for a section.

    If *section_code* has a per-section override in the YAML
    ``configuration.section_thresholds`` block, that value is returned.
    Otherwise the ``review_threshold_default`` is returned.

    Drop-in replacement for both ``_REVIEW_THRESHOLD`` (llm_retemplater)
    and ``REVIEW_THRESHOLD`` (classifier).
    """
    cfg = load_ich_schema().configuration
    if section_code:
        overrides = cfg.get("section_thresholds", {})
        if section_code in overrides:
            return float(overrides[section_code])
    return float(cfg.get("review_threshold_default", 0.70))


def get_priority_sections() -> frozenset[str]:
    """Return the set of priority ICH sections.

    Drop-in replacement for ``PRIORITY_SECTIONS`` in priority_sections.py.
    """
    cfg = load_ich_schema().configuration
    return frozenset(cfg.get("priority_sections", ["B.4", "B.5", "B.10", "B.14"]))


def get_soa_pattern() -> re.Pattern[str]:
    """Return compiled regex for SoA column-header detection.

    Drop-in replacement for ``_VISIT_RE`` in priority_sections.py.
    """
    global _SOA_PATTERN_CACHE
    if _SOA_PATTERN_CACHE is not None:
        return _SOA_PATTERN_CACHE

    cfg = load_ich_schema().configuration
    soa = cfg.get("soa_detection", {})
    pattern_str = soa.get(
        "visit_pattern",
        r"\bvisit\b|\bday\s*-?\d+|\bweek\s*-?\d+|\bmonth\s*\d+"
        r"|\bscreening\b|\bbaseline\b|\bfollow[\s-]*up\b"
        r"|\bdischarge\b|\badmission\b|\brandomiz"
        r"|\bend\s+of\s+(?:study|treatment|trial)\b|\beot\b|\beos\b",
    )
    _SOA_PATTERN_CACHE = re.compile(pattern_str, re.IGNORECASE)
    return _SOA_PATTERN_CACHE


def get_soa_min_columns() -> int:
    """Minimum table columns for SoA candidacy."""
    cfg = load_ich_schema().configuration
    return int(cfg.get("soa_detection", {}).get("min_columns", 5))


def get_soa_min_visit_matches() -> int:
    """Minimum visit-keyword column matches for SoA candidacy."""
    cfg = load_ich_schema().configuration
    return int(cfg.get("soa_detection", {}).get("min_visit_matches", 3))


def get_boilerplate_pattern() -> re.Pattern[str]:
    """Return compiled regex for boilerplate line exclusion.

    Builds an anchored ``^\\s*(?:...|...)\\s*$`` pattern from
    individual YAML patterns, matching the original
    ``_BOILERPLATE_RE`` in coverage_reviewer.py.
    """
    global _BOILERPLATE_PATTERN_CACHE
    if _BOILERPLATE_PATTERN_CACHE is not None:
        return _BOILERPLATE_PATTERN_CACHE

    cfg = load_ich_schema().configuration
    cov = cfg.get("coverage", {})
    patterns: list[str] = cov.get("boilerplate_patterns", [])
    if patterns:
        combined = "|".join(patterns)
    else:
        combined = (
            r"page\s+\d+|confidential|proprietary|"
            r"version\s+\d|date\s*:?\s*\d|table\s+of\s+contents|"
            r"protocol\s+number|eudract\s+number|"
            r"list\s+of\s+(?:tables|figures|abbreviations)"
        )
    _BOILERPLATE_PATTERN_CACHE = re.compile(
        rf"^\s*(?:{combined})\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    return _BOILERPLATE_PATTERN_CACHE


def get_min_sentence_length() -> int:
    """Minimum sentence length (chars) for coverage scoring."""
    cfg = load_ich_schema().configuration
    return int(cfg.get("coverage", {}).get("min_sentence_length", 20))
