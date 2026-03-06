"""ICH M11 CeSHarP machine-readable protocol parser (PTCV-56).

Parses ICH M11 structured protocol files (JSON format per CeSHarP
specification) and produces IchSection objects with confidence=1.0
plus a RawSoaTable with full cell-status metadata.

ICH M11 (adopted November 2024, effective 2025) defines the Clinical
Electronic Structured Harmonised Protocol (CeSHarP) — a machine-readable
protocol template that mandates structured, parseable SoA data.

SoA Cell Notation:
  X = Required  (maps to status="required")
  C = Conditional (maps to status="conditional", includes condition text)
  O = Optional  (maps to status="optional")
  Empty = Not applicable

Risk tier: LOW — deterministic parsing, no LLM calls, no patient data.

Regulatory references:
- ICH M11: Clinical electronic Structured Harmonised Protocol (CeSHarP)
- ALCOA+ Accurate: confidence_score=1.0 for machine-parsed sections
- ALCOA+ Contemporaneous: extraction_timestamp_utc set at parse time
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ..soa_extractor.models import (
    RawSoaTable,
    UsdmActivity,
    UsdmEpoch,
    UsdmScheduledInstance,
    UsdmTimepoint,
)
from .models import IchSection


# ---------------------------------------------------------------------------
# ICH E6(R3) section name lookup
# ---------------------------------------------------------------------------

_SECTION_NAMES: dict[str, str] = {
    "B.1": "General Information",
    "B.2": "Background Information",
    "B.3": "Trial Objectives and Purpose",
    "B.4": "Trial Design",
    "B.5": "Selection of Subjects",
    "B.6": "Treatment of Subjects",
    "B.7": "Assessment of Efficacy",
    "B.8": "Assessment of Safety",
    "B.9": "Statistics",
    "B.10": "Direct Access to Source Data and Documents",
    "B.11": "Quality Control and Quality Assurance",
}


# ---------------------------------------------------------------------------
# SoA cell status constants
# ---------------------------------------------------------------------------

_CELL_STATUS_MAP: dict[str, str] = {
    "X": "required",
    "x": "required",
    "C": "conditional",
    "c": "conditional",
    "O": "optional",
    "o": "optional",
}


# ---------------------------------------------------------------------------
# Activity type classification
# ---------------------------------------------------------------------------

_ACTIVITY_TYPE_KEYWORDS: dict[str, list[str]] = {
    "Lab": [
        "blood", "haematology", "hematology", "chemistry", "urinalysis",
        "laboratory", "lab", "biomarker", "serology", "coagulation",
        "liver function", "renal function", "lipid panel",
    ],
    "Vital": [
        "vital sign", "blood pressure", "heart rate", "temperature",
        "weight", "height", "bmi", "pulse", "respiratory rate",
    ],
    "ECG": ["ecg", "electrocardiogram", "12-lead"],
    "Procedure": [
        "biopsy", "imaging", "mri", "ct scan", "x-ray", "ultrasound",
        "endoscopy", "surgery", "procedure",
    ],
    "Assessment": [
        "questionnaire", "diary", "score", "scale", "assessment",
        "evaluation", "quality of life", "proms", "patient reported",
    ],
}


def _classify_activity_type(name: str) -> str:
    """Classify an activity name into a USDM activity type."""
    lower = name.lower()
    for act_type, keywords in _ACTIVITY_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return act_type
    return "Other"


# ---------------------------------------------------------------------------
# M11 parse result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class M11ParseResult:
    """Result of parsing an ICH M11 CeSHarP protocol.

    Attributes:
        sections: ICH sections with confidence=1.0.
        soa_table: Parsed SoA table with cell status metadata.
        cell_metadata: Per-cell status/condition info for SoA.
        timepoints: USDM timepoints from M11 visit definitions.
        activities: USDM activities from M11 assessment rows.
        scheduled_instances: Activity × Timepoint instances.
        epochs: USDM epochs from M11 epoch definitions.
        m11_metadata: Raw M11 metadata (version, etc.).
        run_id: UUID4 for this parse run.
        registry_id: Trial registry identifier.
    """

    sections: list[IchSection]
    soa_table: Optional[RawSoaTable]
    cell_metadata: list[dict[str, str]]
    timepoints: list[UsdmTimepoint]
    activities: list[UsdmActivity]
    scheduled_instances: list[UsdmScheduledInstance]
    epochs: list[UsdmEpoch]
    m11_metadata: dict[str, Any]
    run_id: str
    registry_id: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class M11ProtocolParser:
    """Parses ICH M11 CeSHarP structured protocols.

    Takes a JSON string conforming to the CeSHarP specification and
    produces IchSection objects (confidence=1.0), a RawSoaTable with
    cell-status metadata, and USDM entities for direct downstream use.

    This parser bypasses PDF extraction entirely — structured input
    yields lossless, deterministic output.
    """

    def parse(
        self,
        data: str | bytes,
        registry_id: str = "",
        source_run_id: str = "",
        source_sha256: str = "",
    ) -> M11ParseResult:
        """Parse an ICH M11 CeSHarP protocol.

        Args:
            data: JSON string or bytes of the M11 structured protocol.
            registry_id: Trial identifier (NCT/EudraCT).
            source_run_id: Upstream run_id (empty for direct input).
            source_sha256: SHA-256 of the source file.

        Returns:
            M11ParseResult with sections, SoA table, and USDM entities.

        Raises:
            ValueError: If data is not valid M11 JSON.
        """
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        try:
            protocol = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid M11 JSON: {exc}") from exc

        if not isinstance(protocol, dict):
            raise ValueError("M11 protocol must be a JSON object")

        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract registry_id from protocol if not provided
        if not registry_id:
            registry_id = protocol.get("registry_id", "")

        # Parse M11 metadata
        m11_metadata = self._extract_metadata(protocol)

        # Parse ICH sections (confidence=1.0 for structured input)
        sections = self._parse_sections(
            protocol, registry_id, run_id,
            source_run_id, source_sha256, timestamp,
        )

        # Parse SoA table with cell metadata
        soa_table, cell_metadata = self._parse_soa(protocol)

        # Parse USDM entities
        epochs = self._parse_epochs(
            protocol, registry_id, run_id,
            source_run_id, source_sha256, timestamp,
        )
        timepoints = self._parse_timepoints(
            protocol, epochs, registry_id, run_id,
            source_run_id, source_sha256, timestamp,
        )
        activities = self._parse_activities(
            protocol, registry_id, run_id,
            source_run_id, source_sha256, timestamp,
        )
        scheduled_instances = self._parse_instances(
            protocol, timepoints, activities, registry_id, run_id,
            source_run_id, source_sha256, timestamp,
        )

        return M11ParseResult(
            sections=sections,
            soa_table=soa_table,
            cell_metadata=cell_metadata,
            timepoints=timepoints,
            activities=activities,
            scheduled_instances=scheduled_instances,
            epochs=epochs,
            m11_metadata=m11_metadata,
            run_id=run_id,
            registry_id=registry_id,
        )

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(protocol: dict[str, Any]) -> dict[str, Any]:
        """Extract M11-level metadata from the protocol."""
        return {
            "m11_version": protocol.get("m11_version", ""),
            "cesharp_version": protocol.get("cesharp_version", ""),
            "protocol_version": protocol.get("protocol_version", ""),
            "sponsor": protocol.get("sponsor", ""),
            "phase": protocol.get("phase", ""),
            "indication": protocol.get("indication", ""),
        }

    # ------------------------------------------------------------------
    # ICH section parsing
    # ------------------------------------------------------------------

    def _parse_sections(
        self,
        protocol: dict[str, Any],
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        timestamp: str,
    ) -> list[IchSection]:
        """Parse ICH E6(R3) sections from M11 structured data.

        Machine-readable sections get confidence_score=1.0 and
        review_required=False — no human review needed.
        """
        sections_data = protocol.get("sections", {})
        result: list[IchSection] = []

        for code, section_name in _SECTION_NAMES.items():
            content = sections_data.get(code)
            if content is None:
                continue

            # Normalise content to a JSON-serialisable dict
            if isinstance(content, str):
                content_dict = {
                    "text_excerpt": content,
                    "word_count": len(content.split()),
                }
            elif isinstance(content, dict):
                content_dict = content
            else:
                content_dict = {"data": content}

            result.append(IchSection(
                run_id=run_id,
                source_run_id=source_run_id,
                source_sha256=source_sha256,
                registry_id=registry_id,
                section_code=code,
                section_name=section_name,
                content_json=json.dumps(content_dict, ensure_ascii=False),
                confidence_score=1.0,
                review_required=False,
                legacy_format=False,
                extraction_timestamp_utc=timestamp,
            ))

        return result

    # ------------------------------------------------------------------
    # SoA parsing with cell metadata
    # ------------------------------------------------------------------

    def _parse_soa(
        self,
        protocol: dict[str, Any],
    ) -> tuple[Optional[RawSoaTable], list[dict[str, str]]]:
        """Parse SoA from M11 structured data.

        Returns:
            Tuple of (RawSoaTable, cell_metadata_list).
            Cell metadata includes status (required/optional/conditional)
            and condition text for conditional cells.
        """
        soa = protocol.get("schedule_of_activities")
        if not soa:
            return None, []

        visits = soa.get("visits", [])
        assessments = soa.get("assessments", [])

        if not visits or not assessments:
            return None, []

        # Build visit headers and day headers
        visit_headers: list[str] = []
        day_headers: list[str] = []
        for visit in visits:
            visit_headers.append(str(visit.get("name", "")))
            day_offset = visit.get("day_offset")
            if day_offset is not None:
                day_headers.append(f"Day {day_offset}")
            else:
                day_headers.append("")

        # Build activities with scheduled flags and cell metadata
        activities: list[tuple[str, list[bool]]] = []
        cell_metadata: list[dict[str, str]] = []

        for assessment in assessments:
            name = str(assessment.get("name", ""))
            cells = assessment.get("cells", [])

            flags: list[bool] = []
            for i, visit in enumerate(visits):
                cell_value = ""
                if i < len(cells):
                    cell_value = str(cells[i].get("value", "")
                                     if isinstance(cells[i], dict)
                                     else cells[i]).strip()

                status = _CELL_STATUS_MAP.get(cell_value, "not_applicable")
                scheduled = status in ("required", "conditional", "optional")
                flags.append(scheduled)

                # Capture condition text for conditional cells
                condition = ""
                if isinstance(cells[i], dict) if i < len(cells) else False:
                    condition = str(cells[i].get("condition", ""))

                cell_metadata.append({
                    "assessment": name,
                    "visit": visit_headers[i] if i < len(visit_headers) else "",
                    "status": status,
                    "condition": condition,
                    "raw_value": cell_value,
                })

            activities.append((name, flags))

        table = RawSoaTable(
            visit_headers=visit_headers,
            day_headers=day_headers,
            activities=activities,
            section_code="B.4",
        )

        return table, cell_metadata

    # ------------------------------------------------------------------
    # USDM entity parsing
    # ------------------------------------------------------------------

    def _parse_epochs(
        self,
        protocol: dict[str, Any],
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        timestamp: str,
    ) -> list[UsdmEpoch]:
        """Parse USDM epochs from M11 epoch definitions."""
        epochs_data = protocol.get("epochs", [])
        result: list[UsdmEpoch] = []

        for i, ep in enumerate(epochs_data):
            result.append(UsdmEpoch(
                run_id=run_id,
                source_run_id=source_run_id,
                source_sha256=source_sha256,
                registry_id=registry_id,
                epoch_id=f"epoch-{i + 1}",
                epoch_name=str(ep.get("name", f"Epoch {i + 1}")),
                epoch_type=str(ep.get("type", "Treatment")),
                order=i + 1,
                extraction_timestamp_utc=timestamp,
            ))

        # Default epochs if none provided
        if not result:
            soa = protocol.get("schedule_of_activities", {})
            visits = soa.get("visits", [])
            epoch_names = self._infer_epochs_from_visits(visits)
            for i, (name, etype) in enumerate(epoch_names):
                result.append(UsdmEpoch(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    epoch_id=f"epoch-{i + 1}",
                    epoch_name=name,
                    epoch_type=etype,
                    order=i + 1,
                    extraction_timestamp_utc=timestamp,
                ))

        return result

    @staticmethod
    def _infer_epochs_from_visits(
        visits: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """Infer epoch names from visit types when no explicit epochs."""
        seen: dict[str, str] = {}
        for visit in visits:
            vtype = str(visit.get("type", "Treatment"))
            if vtype not in seen:
                seen[vtype] = vtype
        if not seen:
            return [("Treatment", "Treatment")]
        return [(name, etype) for name, etype in seen.items()]

    def _parse_timepoints(
        self,
        protocol: dict[str, Any],
        epochs: list[UsdmEpoch],
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        timestamp: str,
    ) -> list[UsdmTimepoint]:
        """Parse USDM timepoints from M11 visit definitions."""
        soa = protocol.get("schedule_of_activities", {})
        visits = soa.get("visits", [])
        result: list[UsdmTimepoint] = []

        # Build epoch lookup by type
        epoch_by_type: dict[str, str] = {}
        for ep in epochs:
            epoch_by_type[ep.epoch_type] = ep.epoch_id
        default_epoch = epochs[0].epoch_id if epochs else "epoch-1"

        for i, visit in enumerate(visits):
            visit_type = str(visit.get("type", "Treatment"))
            epoch_id = epoch_by_type.get(visit_type, default_epoch)

            result.append(UsdmTimepoint(
                run_id=run_id,
                source_run_id=source_run_id,
                source_sha256=source_sha256,
                registry_id=registry_id,
                timepoint_id=f"tp-{i + 1}",
                epoch_id=epoch_id,
                visit_name=str(visit.get("name", f"Visit {i + 1}")),
                visit_type=visit_type,
                day_offset=int(visit.get("day_offset", i + 1)),
                window_early=int(visit.get("window_early", 0)),
                window_late=int(visit.get("window_late", 0)),
                mandatory=bool(visit.get("mandatory", True)),
                extraction_timestamp_utc=timestamp,
                review_required=False,
            ))

        return result

    def _parse_activities(
        self,
        protocol: dict[str, Any],
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        timestamp: str,
    ) -> list[UsdmActivity]:
        """Parse USDM activities from M11 assessment definitions."""
        soa = protocol.get("schedule_of_activities", {})
        assessments = soa.get("assessments", [])
        result: list[UsdmActivity] = []

        for i, assessment in enumerate(assessments):
            name = str(assessment.get("name", ""))
            act_type = str(assessment.get(
                "activity_type",
                _classify_activity_type(name),
            ))

            result.append(UsdmActivity(
                run_id=run_id,
                source_run_id=source_run_id,
                source_sha256=source_sha256,
                registry_id=registry_id,
                activity_id=f"act-{i + 1}",
                activity_name=name,
                activity_type=act_type,
                extraction_timestamp_utc=timestamp,
            ))

        return result

    def _parse_instances(
        self,
        protocol: dict[str, Any],
        timepoints: list[UsdmTimepoint],
        activities: list[UsdmActivity],
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        timestamp: str,
    ) -> list[UsdmScheduledInstance]:
        """Create ScheduledInstances for every scheduled cell."""
        soa = protocol.get("schedule_of_activities", {})
        assessments = soa.get("assessments", [])
        visits = soa.get("visits", [])
        result: list[UsdmScheduledInstance] = []
        seq = 0

        for act_idx, assessment in enumerate(assessments):
            cells = assessment.get("cells", [])
            activity_id = (
                activities[act_idx].activity_id
                if act_idx < len(activities) else f"act-{act_idx + 1}"
            )

            for visit_idx in range(len(visits)):
                cell_value = ""
                if visit_idx < len(cells):
                    raw = cells[visit_idx]
                    cell_value = str(
                        raw.get("value", "") if isinstance(raw, dict) else raw
                    ).strip()

                status = _CELL_STATUS_MAP.get(cell_value, "not_applicable")
                if status == "not_applicable":
                    continue

                seq += 1
                timepoint_id = (
                    timepoints[visit_idx].timepoint_id
                    if visit_idx < len(timepoints) else f"tp-{visit_idx + 1}"
                )

                result.append(UsdmScheduledInstance(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    instance_id=f"inst-{seq}",
                    activity_id=activity_id,
                    timepoint_id=timepoint_id,
                    scheduled=True,
                    extraction_timestamp_utc=timestamp,
                ))

        return result
