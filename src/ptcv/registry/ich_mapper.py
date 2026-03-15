"""ClinicalTrials.gov → ICH E6(R3) Appendix B Metadata Mapper.

PTCV-195: Maps structured ClinicalTrials.gov API v2 JSON fields to
ICH E6(R3) Appendix B section codes, producing IchSection-compatible
dicts with ``source="registry"`` for downstream RAG seeding (PTCV-196).

Section codes align with the PTCV ICH schema
(``data/templates/ich_e6r3_schema.yaml``):

    B.1  General Information          ← StatusModule, IdentificationModule
    B.2  Background Information       ← ConditionsModule
    B.3  Trial Objectives and Purpose ← IdentificationModule (titles)
    B.4  Trial Design                 ← DesignModule
    B.5  Selection of Subjects        ← EligibilityModule
    B.8  Assessment of Efficacy       ← OutcomesModule (primary + secondary)
    B.10 Statistics                   ← DesignModule (enrollment)
    B.16 Publication Policy           ← ReferencesModule

Quality ratings reflect how directly CT.gov fields map to ICH content:
    Direct (0.9)      — field semantics match closely
    Partial (0.6)     — partial overlap, some content missing
    Contextual (0.3)  — tangential relevance only

Risk tier: LOW — read-only data transformation (no API calls).
"""

import dataclasses
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Quality rating constants
QUALITY_DIRECT = 0.9
QUALITY_PARTIAL = 0.6
QUALITY_CONTEXTUAL = 0.3


@dataclasses.dataclass
class MappedRegistrySection:
    """A single ICH section derived from registry metadata.

    Compatible with ``IchSection`` fields — can be converted via
    :meth:`to_ich_kwargs` for constructing ``IchSection`` objects.

    Attributes:
        section_code: ICH E6(R3) section code (e.g. ``"B.4"``).
        section_name: Human-readable section name.
        content_text: Assembled text content from registry fields.
        content_json: Structured content serialised as JSON string.
        quality_rating: Mapping quality (0.0–1.0).
        source: Always ``"registry"`` for registry-derived sections.
        ct_gov_module: Source CT.gov API module name.
    """

    section_code: str
    section_name: str
    content_text: str
    content_json: str
    quality_rating: float
    source: str = "registry"
    ct_gov_module: str = ""

    def to_ich_kwargs(
        self,
        run_id: str = "",
        source_run_id: str = "",
        source_sha256: str = "",
        registry_id: str = "",
    ) -> dict[str, Any]:
        """Return kwargs suitable for ``IchSection(**kwargs)``."""
        return {
            "run_id": run_id,
            "source_run_id": source_run_id,
            "source_sha256": source_sha256,
            "registry_id": registry_id,
            "section_code": self.section_code,
            "section_name": self.section_name,
            "content_json": self.content_json,
            "confidence_score": self.quality_rating,
            "review_required": self.quality_rating < 0.70,
            "legacy_format": False,
            "content_text": self.content_text,
        }


def _safe_get(data: dict[str, Any], *keys: str) -> Any:
    """Navigate nested dicts safely, returning None on missing keys."""
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_outcomes(
    outcomes: list[dict[str, Any]],
    label: str,
) -> str:
    """Format a list of outcome dicts as readable text."""
    if not outcomes:
        return ""
    lines = [f"{label}:"]
    for i, outcome in enumerate(outcomes, 1):
        measure = outcome.get("measure", "")
        timeframe = outcome.get("timeFrame", "")
        desc = outcome.get("description", "")
        parts = [f"  {i}. {measure}"]
        if timeframe:
            parts.append(f"     Time Frame: {timeframe}")
        if desc:
            parts.append(f"     Description: {desc}")
        lines.extend(parts)
    return "\n".join(lines)


def _format_interventions(
    interventions: list[dict[str, Any]],
) -> str:
    """Format intervention list as readable text."""
    if not interventions:
        return ""
    lines = ["Interventions:"]
    for i, intv in enumerate(interventions, 1):
        name = intv.get("name", "Unknown")
        itype = intv.get("type", "")
        desc = intv.get("description", "")
        line = f"  {i}. {name}"
        if itype:
            line += f" ({itype})"
        lines.append(line)
        if desc:
            lines.append(f"     {desc}")
    return "\n".join(lines)


class MetadataToIchMapper:
    """Map ClinicalTrials.gov metadata to ICH E6(R3) sections.

    Each ``map_*`` method extracts one ICH section from the registry
    JSON.  The main ``map()`` method runs all extractors and returns
    only sections with non-empty content.
    """

    def map(
        self, metadata: dict[str, Any]
    ) -> list[MappedRegistrySection]:
        """Map all available CT.gov fields to ICH sections.

        Args:
            metadata: Full study JSON from CT.gov API v2.

        Returns:
            List of mapped sections with non-empty content.
        """
        proto = metadata.get("protocolSection", {})
        if not proto:
            logger.warning("No protocolSection in metadata")
            return []

        extractors = [
            self._map_b1_general,
            self._map_b2_background,
            self._map_b3_objectives,
            self._map_b4_design,
            self._map_b5_eligibility,
            self._map_b8_efficacy,
            self._map_b10_statistics,
            self._map_b16_references,
        ]

        sections: list[MappedRegistrySection] = []
        for extractor in extractors:
            result = extractor(proto)
            if result is not None:
                sections.append(result)

        logger.info(
            "Mapped %d ICH sections from registry metadata",
            len(sections),
        )
        return sections

    def _map_b1_general(
        self, proto: dict[str, Any]
    ) -> Optional[MappedRegistrySection]:
        """B.1 General Information ← StatusModule + IdentificationModule."""
        status_mod = proto.get("statusModule", {})
        id_mod = proto.get("identificationModule", {})
        if not status_mod and not id_mod:
            return None

        parts: list[str] = []
        content: dict[str, Any] = {}

        nct_id = id_mod.get("nctId", "")
        if nct_id:
            parts.append(f"Registry ID: {nct_id}")
            content["nct_id"] = nct_id

        org = id_mod.get("organization", {})
        if org.get("fullName"):
            parts.append(f"Sponsor: {org['fullName']}")
            content["sponsor"] = org["fullName"]

        status = status_mod.get("overallStatus", "")
        if status:
            parts.append(f"Status: {status}")
            content["status"] = status

        start = _safe_get(status_mod, "startDateStruct", "date")
        if start:
            parts.append(f"Start Date: {start}")
            content["start_date"] = start

        completion = _safe_get(
            status_mod, "completionDateStruct", "date"
        )
        if completion:
            parts.append(f"Completion Date: {completion}")
            content["completion_date"] = completion

        if not parts:
            return None

        return MappedRegistrySection(
            section_code="B.1",
            section_name="General Information",
            content_text="\n".join(parts),
            content_json=json.dumps(content, ensure_ascii=False),
            quality_rating=QUALITY_PARTIAL,
            ct_gov_module="StatusModule+IdentificationModule",
        )

    def _map_b2_background(
        self, proto: dict[str, Any]
    ) -> Optional[MappedRegistrySection]:
        """B.2 Background Information ← ConditionsModule."""
        cond_mod = proto.get("conditionsModule", {})
        conditions = cond_mod.get("conditions", [])
        if not conditions:
            return None

        text = "Conditions: " + ", ".join(conditions)
        keywords = cond_mod.get("keywords", [])
        if keywords:
            text += "\nKeywords: " + ", ".join(keywords)

        return MappedRegistrySection(
            section_code="B.2",
            section_name="Background Information",
            content_text=text,
            content_json=json.dumps(
                {"conditions": conditions, "keywords": keywords},
                ensure_ascii=False,
            ),
            quality_rating=QUALITY_CONTEXTUAL,
            ct_gov_module="ConditionsModule",
        )

    def _map_b3_objectives(
        self, proto: dict[str, Any]
    ) -> Optional[MappedRegistrySection]:
        """B.3 Trial Objectives ← IdentificationModule titles."""
        id_mod = proto.get("identificationModule", {})
        official = id_mod.get("officialTitle", "")
        brief = id_mod.get("briefTitle", "")
        brief_summary = _safe_get(
            proto, "descriptionModule", "briefSummary"
        )

        if not official and not brief:
            return None

        parts: list[str] = []
        content: dict[str, Any] = {}

        if official:
            parts.append(f"Official Title: {official}")
            content["official_title"] = official
        if brief:
            parts.append(f"Brief Title: {brief}")
            content["brief_title"] = brief
        if brief_summary:
            parts.append(f"\nBrief Summary:\n{brief_summary}")
            content["brief_summary"] = brief_summary

        return MappedRegistrySection(
            section_code="B.3",
            section_name="Trial Objectives and Purpose",
            content_text="\n".join(parts),
            content_json=json.dumps(content, ensure_ascii=False),
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="IdentificationModule",
        )

    def _map_b4_design(
        self, proto: dict[str, Any]
    ) -> Optional[MappedRegistrySection]:
        """B.4 Trial Design ← DesignModule."""
        design_mod = proto.get("designModule", {})
        if not design_mod:
            return None

        parts: list[str] = []
        content: dict[str, Any] = {}

        study_type = design_mod.get("studyType", "")
        if study_type:
            parts.append(f"Study Type: {study_type}")
            content["study_type"] = study_type

        phases = design_mod.get("phases", [])
        if phases:
            parts.append(f"Phase: {', '.join(phases)}")
            content["phases"] = phases

        design_info = design_mod.get("designInfo", {})
        if design_info:
            allocation = design_info.get("allocation", "")
            if allocation:
                parts.append(f"Allocation: {allocation}")
                content["allocation"] = allocation

            masking = design_info.get("maskingInfo", {})
            if masking.get("masking"):
                parts.append(f"Masking: {masking['masking']}")
                content["masking"] = masking["masking"]

            model = design_info.get("interventionModel", "")
            if model:
                parts.append(f"Intervention Model: {model}")
                content["intervention_model"] = model

            purpose = design_info.get("primaryPurpose", "")
            if purpose:
                parts.append(f"Primary Purpose: {purpose}")
                content["primary_purpose"] = purpose

        # Arms
        arms_mod = proto.get("armsInterventionsModule", {})
        interventions = arms_mod.get("interventions", [])
        if interventions:
            parts.append("")
            parts.append(_format_interventions(interventions))
            content["interventions"] = interventions

        if not parts:
            return None

        return MappedRegistrySection(
            section_code="B.4",
            section_name="Trial Design",
            content_text="\n".join(parts),
            content_json=json.dumps(content, ensure_ascii=False),
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="DesignModule+ArmsInterventionsModule",
        )

    def _map_b5_eligibility(
        self, proto: dict[str, Any]
    ) -> Optional[MappedRegistrySection]:
        """B.5 Selection of Subjects ← EligibilityModule."""
        elig_mod = proto.get("eligibilityModule", {})
        if not elig_mod:
            return None

        parts: list[str] = []
        content: dict[str, Any] = {}

        criteria = elig_mod.get("eligibilityCriteria", "")
        if criteria:
            parts.append(f"Eligibility Criteria:\n{criteria}")
            content["eligibility_criteria"] = criteria

        sex = elig_mod.get("sex", "")
        if sex:
            parts.append(f"Sex: {sex}")
            content["sex"] = sex

        min_age = elig_mod.get("minimumAge", "")
        if min_age:
            parts.append(f"Minimum Age: {min_age}")
            content["minimum_age"] = min_age

        max_age = elig_mod.get("maximumAge", "")
        if max_age:
            parts.append(f"Maximum Age: {max_age}")
            content["maximum_age"] = max_age

        healthy = elig_mod.get("healthyVolunteers", "")
        if healthy:
            parts.append(f"Healthy Volunteers: {healthy}")
            content["healthy_volunteers"] = healthy

        if not parts:
            return None

        return MappedRegistrySection(
            section_code="B.5",
            section_name="Selection of Subjects",
            content_text="\n".join(parts),
            content_json=json.dumps(content, ensure_ascii=False),
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="EligibilityModule",
        )

    def _map_b8_efficacy(
        self, proto: dict[str, Any]
    ) -> Optional[MappedRegistrySection]:
        """B.8 Assessment of Efficacy ← OutcomesModule.

        Both primary and secondary outcomes are efficacy endpoints.
        Safety assessments (AEs, labs, vitals) are not available from
        CT.gov structured data, so B.9 is not mapped.
        """
        outcomes_mod = proto.get("outcomesModule", {})
        primary = outcomes_mod.get("primaryOutcomes", [])
        secondary = outcomes_mod.get("secondaryOutcomes", [])
        if not primary and not secondary:
            return None

        parts: list[str] = []
        content: dict[str, Any] = {}

        if primary:
            parts.append(_format_outcomes(primary, "Primary Outcomes"))
            content["primary_outcomes"] = primary
        if secondary:
            if parts:
                parts.append("")
            parts.append(
                _format_outcomes(secondary, "Secondary Outcomes")
            )
            content["secondary_outcomes"] = secondary

        return MappedRegistrySection(
            section_code="B.8",
            section_name="Assessment of Efficacy",
            content_text="\n".join(parts),
            content_json=json.dumps(content, ensure_ascii=False),
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="OutcomesModule",
        )

    def _map_b10_statistics(
        self, proto: dict[str, Any]
    ) -> Optional[MappedRegistrySection]:
        """B.10 Statistics ← DesignModule (enrollment)."""
        design_mod = proto.get("designModule", {})
        enrollment = design_mod.get("enrollmentInfo", {})
        if not enrollment:
            return None

        count = enrollment.get("count")
        etype = enrollment.get("type", "")
        if count is None:
            return None

        text = f"Enrollment: {count}"
        if etype:
            text += f" ({etype})"

        return MappedRegistrySection(
            section_code="B.10",
            section_name="Statistics",
            content_text=text,
            content_json=json.dumps(
                {"enrollment_count": count, "enrollment_type": etype},
                ensure_ascii=False,
            ),
            quality_rating=QUALITY_PARTIAL,
            ct_gov_module="DesignModule",
        )

    def _map_b16_references(
        self, proto: dict[str, Any]
    ) -> Optional[MappedRegistrySection]:
        """B.16 Publication Policy ← ReferencesModule."""
        refs_mod = proto.get("referencesModule", {})
        references = refs_mod.get("references", [])
        if not references:
            return None

        parts: list[str] = ["References:"]
        for i, ref in enumerate(references, 1):
            pmid = ref.get("pmid", "")
            citation = ref.get("citation", "")
            line = f"  {i}."
            if citation:
                line += f" {citation}"
            if pmid:
                line += f" [PMID: {pmid}]"
            parts.append(line)

        return MappedRegistrySection(
            section_code="B.16",
            section_name="Publication Policy",
            content_text="\n".join(parts),
            content_json=json.dumps(
                {"references": references}, ensure_ascii=False
            ),
            quality_rating=QUALITY_DIRECT,
            ct_gov_module="ReferencesModule",
        )
