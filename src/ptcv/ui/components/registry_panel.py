"""Registry metadata panel for Streamlit UI (PTCV-207).

Loads ClinicalTrials.gov cached metadata and renders a summary
panel in the Results tab.  Also provides a catalog enrichment
function for the sidebar file browser.

No Streamlit dependency in the loader — fully testable with pytest.
The ``render_*`` functions use Streamlit.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CTGOV_URL = "https://clinicaltrials.gov/study"


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass(frozen=True)
class RegistryMetadata:
    """Parsed subset of ClinicalTrials.gov metadata for UI display.

    Attributes:
        nct_id: NCT identifier.
        official_title: Full official title (falls back to brief).
        brief_title: Short title.
        sponsor: Lead sponsor name.
        phase: Phase string (e.g., ``"PHASE2"``).
        status: Overall study status.
        conditions: List of condition strings.
        enrollment: Enrollment count (0 if unknown).
        enrollment_type: ``"ACTUAL"`` or ``"ESTIMATED"``.
        study_type: ``"INTERVENTIONAL"`` or ``"OBSERVATIONAL"``.
        start_date: Study start date string.
        completion_date: Primary completion date string.
        primary_outcomes: List of primary outcome measure names.
        secondary_outcomes: List of secondary outcome measure names.
        allocation: Randomisation allocation (e.g., ``"RANDOMIZED"``).
        masking: Masking type (e.g., ``"DOUBLE"``).
        intervention_model: Intervention model (e.g., ``"PARALLEL"``).
    """

    nct_id: str = ""
    official_title: str = ""
    brief_title: str = ""
    sponsor: str = ""
    phase: str = ""
    status: str = ""
    conditions: list[str] = field(default_factory=list)
    enrollment: int = 0
    enrollment_type: str = ""
    study_type: str = ""
    start_date: str = ""
    completion_date: str = ""
    primary_outcomes: list[str] = field(default_factory=list)
    secondary_outcomes: list[str] = field(default_factory=list)
    allocation: str = ""
    masking: str = ""
    intervention_model: str = ""

    @property
    def display_title(self) -> str:
        """Best available title."""
        return self.official_title or self.brief_title or self.nct_id

    @property
    def ctgov_url(self) -> str:
        """ClinicalTrials.gov study page URL."""
        return f"{_CTGOV_URL}/{self.nct_id}"

    @property
    def phase_display(self) -> str:
        """Human-readable phase string."""
        if not self.phase:
            return "N/A"
        return self.phase.replace("PHASE", "Phase ").replace(
            "/", " / "
        )


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------

def _extract_date(date_struct: dict[str, Any] | None) -> str:
    """Extract date string from a CT.gov date struct."""
    if not date_struct:
        return ""
    return date_struct.get("date", "")


def parse_registry_json(raw: dict[str, Any]) -> RegistryMetadata:
    """Parse a CT.gov API JSON response into RegistryMetadata.

    Args:
        raw: Full CT.gov JSON (with ``protocolSection`` key).

    Returns:
        Parsed RegistryMetadata.
    """
    proto = raw.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    cond_mod = proto.get("conditionsModule", {})
    outcomes_mod = proto.get("outcomesModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

    # Design info
    design_info = design_mod.get("designInfo", {})
    masking_info = design_info.get("maskingInfo", {})

    # Phases
    phases = design_mod.get("phases", [])
    phase_str = "/".join(phases) if phases else ""

    # Outcomes
    primary = [
        o.get("measure", "")
        for o in outcomes_mod.get("primaryOutcomes", [])
        if o.get("measure")
    ]
    secondary = [
        o.get("measure", "")
        for o in outcomes_mod.get("secondaryOutcomes", [])
        if o.get("measure")
    ]

    # Enrollment
    enrollment_info = design_mod.get("enrollmentInfo", {})

    return RegistryMetadata(
        nct_id=id_mod.get("nctId", ""),
        official_title=id_mod.get("officialTitle", ""),
        brief_title=id_mod.get("briefTitle", ""),
        sponsor=sponsor_mod.get("leadSponsor", {}).get("name", ""),
        phase=phase_str,
        status=status_mod.get("overallStatus", ""),
        conditions=cond_mod.get("conditions", []),
        enrollment=enrollment_info.get("count", 0),
        enrollment_type=enrollment_info.get("type", ""),
        study_type=design_mod.get("studyType", ""),
        start_date=_extract_date(
            status_mod.get("startDateStruct"),
        ),
        completion_date=_extract_date(
            status_mod.get("primaryCompletionDateStruct"),
        ),
        primary_outcomes=primary,
        secondary_outcomes=secondary,
        allocation=design_info.get("allocation", ""),
        masking=masking_info.get("masking", ""),
        intervention_model=design_info.get(
            "interventionModel", ""
        ),
    )


def load_registry_metadata(
    registry_cache_dir: Path,
    nct_id: str,
) -> RegistryMetadata | None:
    """Load cached CT.gov metadata for a single NCT ID.

    Args:
        registry_cache_dir: Path to registry_cache directory.
        nct_id: NCT identifier (e.g., ``"NCT00004088"``).

    Returns:
        Parsed RegistryMetadata, or ``None`` if not cached.
    """
    cache_file = registry_cache_dir / f"{nct_id}.json"
    if not cache_file.is_file():
        return None
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return parse_registry_json(raw)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Failed to load registry cache for %s: %s",
            nct_id,
            exc,
        )
        return None


def load_all_registry_metadata(
    registry_cache_dir: Path,
) -> dict[str, RegistryMetadata]:
    """Load all cached CT.gov metadata files.

    Args:
        registry_cache_dir: Path to registry_cache directory.

    Returns:
        Dict mapping NCT ID to RegistryMetadata.
    """
    if not registry_cache_dir.is_dir():
        return {}
    result: dict[str, RegistryMetadata] = {}
    for cache_file in registry_cache_dir.glob("NCT*.json"):
        nct_id = cache_file.stem
        try:
            raw = json.loads(
                cache_file.read_text(encoding="utf-8")
            )
            result[nct_id] = parse_registry_json(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Skipping %s: %s", cache_file.name, exc
            )
    return result


# ------------------------------------------------------------------
# Streamlit rendering
# ------------------------------------------------------------------

def render_registry_panel(
    registry_cache_dir: Path,
    registry_id: str,
) -> None:
    """Render a CT.gov metadata summary panel in the Results tab.

    Displays sponsor, phase, status, enrollment, study design,
    endpoints, and a link to ClinicalTrials.gov.

    Args:
        registry_cache_dir: Path to registry_cache directory.
        registry_id: NCT identifier for the selected protocol.
    """
    import streamlit as st

    meta = load_registry_metadata(registry_cache_dir, registry_id)
    if meta is None:
        return  # Graceful fallback — no panel if no cache

    with st.expander(
        "ClinicalTrials.gov Metadata", expanded=True,
    ):
        # Row 1: Key identifiers
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Status", meta.status.title())
        with c2:
            st.metric("Phase", meta.phase_display)
        with c3:
            enrollment_label = (
                f"{meta.enrollment:,}"
                if meta.enrollment
                else "N/A"
            )
            if meta.enrollment_type:
                enrollment_label += (
                    f" ({meta.enrollment_type.lower()})"
                )
            st.metric("Enrollment", enrollment_label)

        # Row 2: Sponsor and dates
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown(f"**Sponsor:** {meta.sponsor or 'N/A'}")
        with c5:
            st.markdown(
                f"**Start:** {meta.start_date or 'N/A'}"
            )
        with c6:
            st.markdown(
                f"**Completion:** "
                f"{meta.completion_date or 'N/A'}"
            )

        # Study design
        design_parts: list[str] = []
        if meta.study_type:
            design_parts.append(meta.study_type.title())
        if meta.allocation and meta.allocation != "NA":
            design_parts.append(meta.allocation.title())
        if meta.masking and meta.masking != "NONE":
            design_parts.append(
                f"{meta.masking.title()} Masked"
            )
        if meta.intervention_model:
            design_parts.append(
                meta.intervention_model.replace("_", " ").title()
            )
        if design_parts:
            st.markdown(
                f"**Design:** {' · '.join(design_parts)}"
            )

        # Conditions
        if meta.conditions:
            st.markdown(
                f"**Conditions:** {', '.join(meta.conditions)}"
            )

        # Endpoints
        if meta.primary_outcomes:
            st.markdown("**Primary Endpoints:**")
            for ep in meta.primary_outcomes:
                st.markdown(f"- {ep}")

        if meta.secondary_outcomes:
            with st.expander(
                f"Secondary Endpoints ({len(meta.secondary_outcomes)})",
                expanded=False,
            ):
                for ep in meta.secondary_outcomes:
                    st.markdown(f"- {ep}")

        # Link
        st.markdown(
            f"[View on ClinicalTrials.gov]({meta.ctgov_url})"
        )
