"""Protocol catalog with therapeutic area classification (PTCV-42).

Pure-Python module for loading protocol metadata, classifying
conditions into therapeutic area buckets, and building a sorted
catalog for the file browser sidebar.

Enriched with ClinicalTrials.gov registry cache (PTCV-207).

No Streamlit dependency — fully testable with ``pytest``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Registry source subdirectories under data/protocols/
_REGISTRY_SOURCES = ("clinicaltrials", "eu-ctr")


# ---------------------------------------------------------------------------
# Therapeutic area classification
# ---------------------------------------------------------------------------

class TherapeuticArea(str, Enum):
    """Normalized therapeutic area buckets for protocol grouping."""

    ONCOLOGY = "Oncology"
    CARDIOVASCULAR = "Cardiovascular"
    NERVOUS_SYSTEM = "Nervous System"
    DIABETES_METABOLIC = "Diabetes/Metabolic"
    OTHER = "Other"


AREA_DISPLAY_ORDER: list[TherapeuticArea] = [
    TherapeuticArea.ONCOLOGY,
    TherapeuticArea.CARDIOVASCULAR,
    TherapeuticArea.NERVOUS_SYSTEM,
    TherapeuticArea.DIABETES_METABOLIC,
    TherapeuticArea.OTHER,
]

# Keywords checked in list order; first match wins.
# Longer/more-specific substrings come first to avoid
# false positives (e.g. "neuroblastoma" → Oncology, not
# Nervous System).
_ONCOLOGY_KEYWORDS: list[str] = [
    "glioblastoma",
    "neuroblastoma",
    "hepatoblastoma",
    "myelodysplastic",
    "myeloproliferative",
    "histiocytosis",
    "waldenstrom",
    "mesothelioma",
    "cancer",
    "carcinoma",
    "lymphoma",
    "leukemia",
    "myeloma",
    "sarcoma",
    "tumor",
    "neoplasm",
    "melanoma",
    "glioma",
    "blastoma",
    "cml",
]

_CARDIOVASCULAR_KEYWORDS: list[str] = [
    "cardiovascular",
    "cardiorenal",
    "heart failure",
    "cardiac",
    "coronary",
    "myocardial",
    "arterial",
    "artery",
    "ventricular",
    "atrial",
    "hypertension",
    "thrombosis",
    "embolism",
    "stroke",
    "ischemic",
    "angina",
]

_NERVOUS_SYSTEM_KEYWORDS: list[str] = [
    "alzheimer",
    "parkinson",
    "epilepsy",
    "seizure",
    "dementia",
    "multiple sclerosis",
    "neuromyelitis",
    "neuropath",
    "migraine",
    "headache",
    "friedreich",
    "tardive",
    "delirium",
    "psychosis",
    "ataxia",
    "neurofibroma",
    "anosmia",
    "olfactory",
]

_DIABETES_METABOLIC_KEYWORDS: list[str] = [
    "prediabetes",
    "pre-diabetes",
    "acromegaly",
    "diabetes",
    "diabetic",
    "insulin",
    "glucose",
    "metabolic",
    "obesity",
    "hypoglycemia",
]

_AREA_KEYWORDS: list[tuple[TherapeuticArea, list[str]]] = [
    (TherapeuticArea.ONCOLOGY, _ONCOLOGY_KEYWORDS),
    (TherapeuticArea.CARDIOVASCULAR, _CARDIOVASCULAR_KEYWORDS),
    (TherapeuticArea.NERVOUS_SYSTEM, _NERVOUS_SYSTEM_KEYWORDS),
    (TherapeuticArea.DIABETES_METABOLIC, _DIABETES_METABOLIC_KEYWORDS),
]


def classify_therapeutic_area(condition: str) -> TherapeuticArea:
    """Classify a condition string into a therapeutic area bucket.

    Uses case-insensitive keyword matching. First match wins
    (checked in order: Oncology, Cardiovascular, Nervous System,
    Diabetes/Metabolic). Unmatched conditions map to Other.

    Args:
        condition: Raw condition string from metadata
            (may be comma-separated).

    Returns:
        TherapeuticArea enum value.
    """
    condition_lower = condition.lower()
    for area, keywords in _AREA_KEYWORDS:
        for kw in keywords:
            if kw in condition_lower:
                return area
    return TherapeuticArea.OTHER


# ---------------------------------------------------------------------------
# Quality scores
# ---------------------------------------------------------------------------

# Verdict tier ordering for sort: ICH_E6R3 best → unscored worst.
_VERDICT_TIER: dict[str, int] = {
    "ICH_E6R3": 0,
    "PARTIAL_ICH": 1,
    "NON_ICH": 2,
    "": 3,
}


@dataclass(frozen=True)
class QualityScores:
    """ICH quality scores from extraction + parse pipeline.

    Attributes:
        format_verdict: ``"ICH_E6R3"``, ``"PARTIAL_ICH"``,
            or ``"NON_ICH"``.  Empty string if unscored.
        format_confidence: 0.0–1.0 confidence score.
        section_count: Number of ICH sections detected (max 11).
    """

    format_verdict: str = ""
    format_confidence: float = 0.0
    section_count: int = 0


# ---------------------------------------------------------------------------
# Protocol entry
# ---------------------------------------------------------------------------

_MAX_TITLE_LEN = 55


@dataclass
class ProtocolEntry:
    """Enriched protocol entry for display in the file browser.

    Attributes:
        registry_id: Trial identifier (e.g., ``"NCT00004088"``).
        title: Full protocol title from metadata.
        condition: Raw condition string from metadata.
        therapeutic_area: Classified therapeutic area bucket.
        registry_source: Subdirectory (``"clinicaltrials"`` or
            ``"eu-ctr"``).
        filename: PDF filename (e.g., ``"NCT00004088_1.0.pdf"``).
        file_path: Absolute path to the PDF file.
        quality: Quality scores (empty if not yet parsed).
    """

    registry_id: str
    title: str
    condition: str
    therapeutic_area: TherapeuticArea
    registry_source: str
    filename: str
    file_path: Path
    quality: QualityScores = field(default_factory=QualityScores)
    sponsor: str = ""
    phase: str = ""

    @property
    def display_label(self) -> str:
        """Truncated title with registry ID parenthetical."""
        if len(self.title) > _MAX_TITLE_LEN:
            short = self.title[:_MAX_TITLE_LEN] + "..."
        else:
            short = self.title
        return f"{short} ({self.registry_id})"

    @property
    def sort_key(self) -> tuple[int, float, str]:
        """Sort key: verdict tier ASC, confidence DESC, id ASC."""
        tier = _VERDICT_TIER.get(self.quality.format_verdict, 3)
        return (tier, -self.quality.format_confidence, self.registry_id)


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

def _find_pdf_path(
    protocols_dir: Path,
    meta: dict[str, Any],
) -> tuple[str, Path] | None:
    """Locate the PDF file for a metadata record.

    Tries each registry source subdirectory for a matching PDF.
    Constructs the expected filename from ``registry_id`` and
    ``amendment_number``.

    Args:
        protocols_dir: Root protocols directory.
        meta: Parsed metadata dict.

    Returns:
        Tuple of (registry_source, pdf_path) or ``None`` if
        not found.
    """
    registry_id = meta.get("registry_id", "")
    amendment = meta.get("amendment_number", meta.get("version", "1.0"))
    expected_name = f"{registry_id}_{amendment}.pdf"

    for source in _REGISTRY_SOURCES:
        candidate = protocols_dir / source / expected_name
        if candidate.is_file():
            return source, candidate

    # Fallback: try file_path from metadata directly
    fp = meta.get("file_path", "")
    if fp:
        p = Path(fp)
        if p.is_file():
            # Infer registry_source from parent directory
            src = p.parent.name
            if src not in _REGISTRY_SOURCES:
                src = "clinicaltrials"
            return src, p

    return None


def _build_entry(
    meta: dict[str, Any],
    registry_source: str,
    pdf_path: Path,
) -> ProtocolEntry:
    """Construct a ProtocolEntry from a metadata dict.

    Args:
        meta: Parsed metadata JSON dict.
        registry_source: Subdirectory name.
        pdf_path: Absolute path to the PDF file.

    Returns:
        ProtocolEntry with therapeutic_area classified and
        quality_scores populated if present in metadata.
    """
    condition = meta.get("condition", "")

    # Use pre-classified area if available, else classify
    area_str = meta.get("therapeutic_area", "")
    try:
        area = TherapeuticArea(area_str)
    except ValueError:
        area = classify_therapeutic_area(condition)

    # Load quality scores if present
    qs_dict = meta.get("quality_scores", {})
    quality = QualityScores(
        format_verdict=qs_dict.get("format_verdict", ""),
        format_confidence=qs_dict.get("format_confidence", 0.0),
        section_count=qs_dict.get("section_count", 0),
    )

    title = meta.get("title", "")
    registry_id = meta.get("registry_id", pdf_path.stem.split("_")[0])
    if not title:
        title = registry_id

    return ProtocolEntry(
        registry_id=registry_id,
        title=title,
        condition=condition,
        therapeutic_area=area,
        registry_source=registry_source,
        filename=pdf_path.name,
        file_path=pdf_path,
        quality=quality,
    )


def _enrich_from_registry_cache(
    entries: list[ProtocolEntry],
    registry_cache_dir: Path,
) -> list[ProtocolEntry]:
    """Enrich catalog entries with CT.gov registry cache data.

    Overwrites title, condition, therapeutic_area, sponsor, and
    phase from the authoritative CT.gov JSON when available.
    Falls back to existing metadata values when cache is missing.

    Args:
        entries: Flat list of ProtocolEntry objects to enrich.
        registry_cache_dir: Path to registry_cache directory.

    Returns:
        Same list with entries replaced by enriched copies where
        registry data was available.
    """
    if not registry_cache_dir.is_dir():
        return entries

    from ptcv.ui.components.registry_panel import (
        load_registry_metadata,
    )

    enriched: list[ProtocolEntry] = []
    for entry in entries:
        meta = load_registry_metadata(
            registry_cache_dir, entry.registry_id,
        )
        if meta is None:
            enriched.append(entry)
            continue

        # Use CT.gov title if available
        title = meta.display_title or entry.title

        # Use CT.gov conditions for classification
        condition = (
            ", ".join(meta.conditions)
            if meta.conditions
            else entry.condition
        )
        area = classify_therapeutic_area(condition)

        enriched.append(
            ProtocolEntry(
                registry_id=entry.registry_id,
                title=title,
                condition=condition,
                therapeutic_area=area,
                registry_source=entry.registry_source,
                filename=entry.filename,
                file_path=entry.file_path,
                quality=entry.quality,
                sponsor=meta.sponsor or entry.sponsor,
                phase=meta.phase_display or entry.phase,
            )
        )

    return enriched


def load_protocol_catalog(
    protocols_dir: Path,
) -> dict[TherapeuticArea, list[ProtocolEntry]]:
    """Load all protocol metadata and build grouped, sorted catalog.

    Scans metadata JSON files under ``protocols_dir/metadata/``,
    matches each to a PDF file, classifies by therapeutic area,
    and sorts by quality within each group.

    If a CT.gov registry cache exists under
    ``protocols_dir/clinicaltrials/registry_cache/``, entries are
    enriched with authoritative trial titles, conditions, sponsor,
    and phase data (PTCV-207).

    Args:
        protocols_dir: Root protocols directory (e.g.,
            ``data/protocols/``).

    Returns:
        Dict mapping TherapeuticArea to sorted list of
        ProtocolEntry objects. Areas with no protocols are
        omitted.
    """
    metadata_dir = protocols_dir / "metadata"
    if not metadata_dir.is_dir():
        return {}

    all_entries: list[ProtocolEntry] = []

    for meta_path in sorted(metadata_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", meta_path.name, exc)
            continue

        result = _find_pdf_path(protocols_dir, meta)
        if result is None:
            continue

        registry_source, pdf_path = result
        entry = _build_entry(meta, registry_source, pdf_path)
        all_entries.append(entry)

    # Enrich from CT.gov registry cache (PTCV-207)
    registry_cache_dir = (
        protocols_dir / "clinicaltrials" / "registry_cache"
    )
    all_entries = _enrich_from_registry_cache(
        all_entries, registry_cache_dir,
    )

    # Group by therapeutic area
    catalog: dict[TherapeuticArea, list[ProtocolEntry]] = {}
    for entry in all_entries:
        catalog.setdefault(entry.therapeutic_area, []).append(entry)

    # Sort each group by quality
    for entries in catalog.values():
        entries.sort(key=lambda e: e.sort_key)

    return catalog
