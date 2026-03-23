"""EX (Exposure) domain builder from intervention structure (PTCV-248).

Generates the EX domain spec from trial arm structure and
intervention details. Extracts treatment names, doses, routes,
and frequencies from CT.gov ArmsInterventionsModule or protocol
text.

Risk tier: MEDIUM — regulatory submission artifacts, no patient data.

References:
- SDTMIG v3.4 Section 6.2: Interventions Observation Class
- CDISC SDTM v1.7 EX domain variables
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class InterventionDetail:
    """Parsed intervention details from protocol or registry.

    Attributes:
        name: Treatment name (e.g., "Drug X").
        intervention_type: "DRUG", "BIOLOGICAL", "PROCEDURE", etc.
        dose: Dose value (e.g., "10", "100").
        dose_unit: Dose unit (e.g., "mg", "mg/kg").
        route: Route of administration (e.g., "ORAL", "IV").
        frequency: Dosing frequency (e.g., "BID", "QD", "WEEKLY").
        arm_labels: Which arms receive this intervention.
        raw_description: Original description text.
    """

    name: str
    intervention_type: str = ""
    dose: str = ""
    dose_unit: str = ""
    route: str = ""
    frequency: str = ""
    arm_labels: list[str] = dataclasses.field(default_factory=list)
    raw_description: str = ""


@dataclasses.dataclass
class ExDomainSpec:
    """EX domain specification.

    Attributes:
        interventions: Parsed intervention details.
        variables: Standard EX variable definitions.
        arm_treatment_map: Maps arm code → list of treatment names.
    """

    interventions: list[InterventionDetail] = dataclasses.field(
        default_factory=list,
    )
    variables: list[dict[str, str]] = dataclasses.field(
        default_factory=list,
    )
    arm_treatment_map: dict[str, list[str]] = dataclasses.field(
        default_factory=dict,
    )

    @property
    def treatment_count(self) -> int:
        return len(self.interventions)


# Standard EX domain variables
_EX_VARIABLES: list[dict[str, str]] = [
    {"name": "STUDYID", "label": "Study Identifier", "type": "Char"},
    {"name": "DOMAIN", "label": "Domain Abbreviation", "type": "Char"},
    {"name": "USUBJID", "label": "Unique Subject Identifier", "type": "Char"},
    {"name": "EXSEQ", "label": "Sequence Number", "type": "Num"},
    {"name": "EXTRT", "label": "Name of Treatment", "type": "Char"},
    {"name": "EXCAT", "label": "Category for Treatment", "type": "Char"},
    {"name": "EXDOSE", "label": "Dose", "type": "Num"},
    {"name": "EXDOSU", "label": "Dose Units", "type": "Char"},
    {"name": "EXDOSFRM", "label": "Dose Form", "type": "Char"},
    {"name": "EXDOSFRQ", "label": "Dosing Frequency per Interval", "type": "Char"},
    {"name": "EXROUTE", "label": "Route of Administration", "type": "Char"},
    {"name": "EXSTDTC", "label": "Start Date/Time of Treatment", "type": "Char"},
    {"name": "EXENDTC", "label": "End Date/Time of Treatment", "type": "Char"},
    {"name": "VISITNUM", "label": "Visit Number", "type": "Num"},
    {"name": "VISIT", "label": "Visit Name", "type": "Char"},
]


# ---------------------------------------------------------------------------
# Dose/route/frequency parsing
# ---------------------------------------------------------------------------

# Dose pattern: number + unit (e.g., "100 mg", "10mg/kg", "5 mL")
_DOSE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg(?:/kg)?|g|mcg|ug|µg|ml|mL|IU|units?)\b",
    re.IGNORECASE,
)

# Route patterns
_ROUTE_MAP: dict[str, str] = {
    "oral": "ORAL",
    "orally": "ORAL",
    "po": "ORAL",
    "intravenous": "INTRAVENOUS",
    "iv": "INTRAVENOUS",
    "subcutaneous": "SUBCUTANEOUS",
    "sc": "SUBCUTANEOUS",
    "sq": "SUBCUTANEOUS",
    "intramuscular": "INTRAMUSCULAR",
    "im": "INTRAMUSCULAR",
    "topical": "TOPICAL",
    "transdermal": "TRANSDERMAL",
    "inhalation": "INHALATION",
    "intranasal": "INTRANASAL",
    "ophthalmic": "OPHTHALMIC",
}

# Frequency patterns
# Ordered longest-first so "twice daily" matches before "daily"
_FREQ_MAP: list[tuple[str, str]] = [
    ("twice daily", "BID"),
    ("twice a day", "BID"),
    ("three times daily", "TID"),
    ("four times daily", "QID"),
    ("once daily", "QD"),
    ("every two weeks", "Q2W"),
    ("every 2 weeks", "Q2W"),
    ("every three weeks", "Q3W"),
    ("every 3 weeks", "Q3W"),
    ("every 4 weeks", "Q4W"),
    ("every week", "QW"),
    ("single dose", "ONCE"),
    ("b.i.d", "BID"),
    ("biweekly", "Q2W"),
    ("monthly", "QM"),
    ("weekly", "QW"),
    ("daily", "QD"),
    ("bid", "BID"),
    ("tid", "TID"),
    ("qid", "QID"),
    ("qd", "QD"),
    ("once", "ONCE"),
]


def _parse_dose(text: str) -> tuple[str, str]:
    """Extract dose and unit from text.

    Args:
        text: Description text containing dose info.

    Returns:
        Tuple of (dose_value, dose_unit). Empty strings if not found.
    """
    match = _DOSE_RE.search(text)
    if match:
        return match.group(1), match.group(2).upper()
    return "", ""


def _parse_route(text: str) -> str:
    """Extract route of administration from text.

    Args:
        text: Description text.

    Returns:
        CDISC route term, or empty string.
    """
    lower = text.lower()
    for keyword, route in _ROUTE_MAP.items():
        if keyword in lower:
            return route
    return ""


def _parse_frequency(text: str) -> str:
    """Extract dosing frequency from text.

    Args:
        text: Description text.

    Returns:
        CDISC frequency term, or empty string.
    """
    lower = text.lower()
    for keyword, freq in _FREQ_MAP:
        if keyword in lower:
            return freq
    return ""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def parse_interventions_from_registry(
    registry_metadata: dict[str, Any],
) -> list[InterventionDetail]:
    """Parse interventions from CT.gov ArmsInterventionsModule.

    Args:
        registry_metadata: Full CT.gov study JSON.

    Returns:
        List of parsed InterventionDetail objects.
    """
    proto = registry_metadata.get("protocolSection", {})
    arms_mod = proto.get("armsInterventionsModule", {})

    interventions_raw = arms_mod.get("interventions", [])
    results: list[InterventionDetail] = []

    for intv in interventions_raw:
        name = intv.get("name", "")
        intv_type = intv.get("type", "")
        description = intv.get("description", "")
        arm_labels = intv.get("armGroupLabels", [])

        dose, dose_unit = _parse_dose(description)
        route = _parse_route(description)
        frequency = _parse_frequency(description)

        # If route not in description, infer from type
        if not route and intv_type == "DRUG":
            route = "ORAL"  # Common default for drugs

        results.append(InterventionDetail(
            name=name,
            intervention_type=intv_type,
            dose=dose,
            dose_unit=dose_unit,
            route=route,
            frequency=frequency,
            arm_labels=arm_labels,
            raw_description=description,
        ))

    return results


def parse_interventions_from_text(
    text: str,
) -> list[InterventionDetail]:
    """Parse interventions from protocol section text (B.4/B.6).

    Heuristic extraction for protocols without registry metadata.

    Args:
        text: Protocol section text describing interventions.

    Returns:
        List of parsed InterventionDetail objects.
    """
    results: list[InterventionDetail] = []

    # Look for "Drug X dose route frequency" patterns
    # Simple heuristic: each line with a dose is a potential intervention
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        dose, dose_unit = _parse_dose(line)
        if not dose:
            continue

        route = _parse_route(line)
        frequency = _parse_frequency(line)

        # Use the line as the treatment name (truncated)
        name = line[:80]

        results.append(InterventionDetail(
            name=name,
            intervention_type="DRUG",
            dose=dose,
            dose_unit=dose_unit,
            route=route,
            frequency=frequency,
            raw_description=line,
        ))

    return results


def build_ex_domain_spec(
    registry_metadata: Optional[dict[str, Any]] = None,
    protocol_text: str = "",
) -> ExDomainSpec:
    """Build EX domain spec from registry metadata or protocol text.

    Args:
        registry_metadata: CT.gov study JSON (preferred source).
        protocol_text: B.4/B.6 section text (fallback).

    Returns:
        ExDomainSpec with interventions, variables, and arm mapping.
    """
    interventions: list[InterventionDetail] = []

    if registry_metadata:
        interventions = parse_interventions_from_registry(
            registry_metadata
        )

    if not interventions and protocol_text:
        interventions = parse_interventions_from_text(protocol_text)

    # Build arm → treatment map
    arm_map: dict[str, list[str]] = {}
    for intv in interventions:
        for arm in intv.arm_labels:
            arm_map.setdefault(arm, []).append(intv.name)

    return ExDomainSpec(
        interventions=interventions,
        variables=list(_EX_VARIABLES),
        arm_treatment_map=arm_map,
    )


__all__ = [
    "ExDomainSpec",
    "InterventionDetail",
    "build_ex_domain_spec",
    "parse_interventions_from_registry",
    "parse_interventions_from_text",
]
