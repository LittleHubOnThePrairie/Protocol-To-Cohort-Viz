"""UsdmMapper — converts RawSoaTable objects to CDISC USDM v4.0 entities.

Maps parsed SoA tables to the full USDM entity graph:
  Epoch → UsdmTimepoint → UsdmActivity → UsdmScheduledInstance

Also produces SynonymMapping audit records for every visit header resolved.

Epoch grouping rules:
  Screening       → "Screening" epoch
  Baseline        → "Treatment" epoch (Baseline is the first Treatment visit)
  Treatment       → "Treatment" epoch
  Follow-up       → "Follow-up" epoch
  Long-term FU    → "Follow-up" epoch
  End of Study    → "End of Study" epoch
  Unscheduled     → "Treatment" epoch (administrative)
  Early Term.     → own "Early Termination" epoch
  Remote          → "Treatment" epoch

Risk tier: MEDIUM — data pipeline mapping (no patient data).

Regulatory references:
- ALCOA+ Accurate: seven required timepoint attributes checked; missing
  ones set review_required=True
- ALCOA+ Traceable: run_id, source_run_id, source_sha256 on every entity
"""

from __future__ import annotations

import logging
import re
import uuid

from .models import (
    ExtractResult,
    RawSoaTable,
    SynonymMapping,
    UsdmActivity,
    UsdmEpoch,
    UsdmScheduledInstance,
    UsdmTimepoint,
)
from .resolver import SynonymResolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Boilerplate header filtering (PTCV-74)
# ---------------------------------------------------------------------------

# Page numbers: "Page 22 of 58", "Page 5"
_PAGE_RE = re.compile(r"(?i)^page\s+\d+")
# Protocol metadata: "Final Protocol Version 1.0", "Protocol No. 123",
# "Amendment 2", "Version 1.0"
_PROTOCOL_META_RE = re.compile(
    r"(?i)^(?:final\s+)?protocol\s+\w"
    r"|^amendment\s+\d"
    r"|^version\s+\d"
)
# Document structure: "Table of Contents", "List of Tables", "Confidential"
_DOC_STRUCTURE_RE = re.compile(
    r"(?i)^(?:table\s+of\s+contents|list\s+of\s+|confidential|proprietary)"
)
# Bare dates — "01 June 2016", "January 15, 2024", "2024-01-15",
# "15-Mar-2022". Must NOT match "Day N" (which is a valid visit).
_BARE_DATE_RE = re.compile(
    r"(?i)"
    r"^\d{1,2}\s+(?:january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\s+\d{4}$"
    r"|^(?:january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\s+\d{1,2},?\s+\d{4}$"
    r"|^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
    r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$"
    r"|^\d{1,2}[-/](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct"
    r"|nov|dec)[-/]\d{2,4}$"
)
# Non-visit content (eligibility criteria, regulatory text)
_NONVISIT_RE = re.compile(
    r"(?i)\b(?:employees?|(?:immediate\s+)?relatives?)\b"
    r"|\b(?:the\s+)?sponsor\b"
    r"|\b(?:inclusion|exclusion)\s+criteria\b"
    r"|\b(?:copyright|trademark|proprietary)\b"
    r"|\b(?:investigator.?s?\s+brochure)\b"
    r"|\b(?:clinical\s+study\s+report)\b"
)
# Parenthetical unit suffixes to strip from visit names (PTCV-180).
# Matches trailing "(cGy)", "(mg/kg)", "(mL)", etc.
_UNIT_SUFFIX_RE = re.compile(
    r"\s*\([^)]*"
    r"(?:cGy|mGy|Gy|mg|mL|\u03bcg|mcg|\xb5g|IU|mmol|\xb5mol|ng|kg)"
    r"[^)]*\)\s*$",
    re.IGNORECASE,
)

# Visit names longer than this are almost certainly boilerplate
_MAX_VISIT_NAME_LEN = 60


def _canonicalize_visit_name(header: str) -> str:
    """Strip parenthetical unit suffixes from visit names (PTCV-180).

    Converts "Day 2 (cGy)" to "Day 2", "Week 4 (mg/kg)" to "Week 4".
    Non-unit parentheticals like "(Day -30 to -1)" are preserved.
    """
    return _UNIT_SUFFIX_RE.sub("", header).strip()


def _is_boilerplate_header(header: str) -> bool:
    """Return True if header is boilerplate, not a real visit name.

    Filters page numbers, bare dates, protocol metadata, eligibility
    criteria text, and other non-visit content (PTCV-74).
    """
    text = header.strip()
    if not text:
        return True
    if len(text) > _MAX_VISIT_NAME_LEN:
        return True
    if _PAGE_RE.search(text):
        return True
    if _BARE_DATE_RE.match(text):
        return True
    if _PROTOCOL_META_RE.search(text):
        return True
    if _DOC_STRUCTURE_RE.search(text):
        return True
    if _NONVISIT_RE.search(text):
        return True
    return False


# Visit type → epoch name mapping
_EPOCH_FOR_TYPE: dict[str, str] = {
    "Screening": "Screening",
    "Baseline": "Treatment",
    "Treatment": "Treatment",
    "Unscheduled": "Treatment",
    "Remote": "Treatment",
    "Follow-up": "Follow-up",
    "Long-term Follow-up": "Follow-up",
    "End of Study": "End of Study",
    "Early Termination": "Early Termination",
}

# Canonical epoch order
_EPOCH_ORDER = [
    "Screening",
    "Treatment",
    "Follow-up",
    "Long-term Follow-up",
    "End of Study",
    "Early Termination",
    "Unscheduled",
]

# Activity type classification by keyword
_ACTIVITY_KEYWORDS: list[tuple[str, str]] = [
    ("informed consent", "Consent"),
    ("ecg", "ECG"),
    ("electrocardiogram", "ECG"),
    ("vital", "Vital Signs"),
    ("blood pressure", "Vital Signs"),
    ("weight", "Vital Signs"),
    ("height", "Vital Signs"),
    ("bmi", "Vital Signs"),
    ("lab", "Lab"),
    ("haematology", "Lab"),
    ("hematology", "Lab"),
    ("biochemistry", "Lab"),
    ("urinalysis", "Lab"),
    ("pk", "Pharmacokinetics"),
    ("pharmacokinetic", "Pharmacokinetics"),
    ("electronic diary", "Other"),
    ("e-diary", "Other"),
    ("ediary", "Other"),
    ("patient diary", "Other"),
    ("diary", "Other"),
    ("biopsy", "Procedure"),
    ("imaging", "Imaging"),
    ("mri", "Imaging"),
    ("ct scan", "Imaging"),
    ("physical exam", "Assessment"),
    ("performance status", "Assessment"),
    ("ecog", "Assessment"),
    ("adverse", "Safety"),
    ("sae", "Safety"),
    ("concomitant", "Safety"),
    ("pregnancy", "Safety"),
]


class UsdmMapper:
    """Map RawSoaTable objects to CDISC USDM v4.0 entity collections.

    Args:
        resolver: SynonymResolver instance. A new default resolver is
            created if None.
    """

    def __init__(self, resolver: SynonymResolver | None = None) -> None:
        self._resolver = resolver if resolver is not None else SynonymResolver()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def map(
        self,
        tables: list[RawSoaTable],
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        registry_id: str,
        timestamp: str,
    ) -> tuple[
        list[UsdmEpoch],
        list[UsdmTimepoint],
        list[UsdmActivity],
        list[UsdmScheduledInstance],
        list[SynonymMapping],
    ]:
        """Map all RawSoaTable objects to USDM entities.

        Args:
            tables: Parsed SoA tables from SoaTableParser.
            run_id: UUID4 for this extraction run.
            source_run_id: run_id from PTCV-20 ICH parse.
            source_sha256: SHA-256 of the PTCV-20 sections.parquet artifact.
            registry_id: Trial identifier.
            timestamp: ISO 8601 UTC write timestamp.

        Returns:
            Tuple of (epochs, timepoints, activities, instances, synonyms).
        [PTCV-21 Scenario: Extract SoA to USDM Parquet with lineage]
        [PTCV-21 Scenario: Unscheduled and early termination visits captured]
        """
        all_timepoints: list[UsdmTimepoint] = []
        all_activities: list[UsdmActivity] = []
        all_instances: list[UsdmScheduledInstance] = []
        all_synonyms: list[SynonymMapping] = []

        # Track activities across tables to avoid duplicates
        activity_name_to_id: dict[str, str] = {}
        # Track timepoints by canonical name for cross-table dedup (PTCV-180)
        visit_name_to_tp_id: dict[str, str] = {}

        for table in tables:
            tps, acts, insts, syns = self._map_table(
                table,
                run_id,
                source_run_id,
                source_sha256,
                registry_id,
                timestamp,
                activity_name_to_id,
            )

            # Deduplicate timepoints across tables (PTCV-180).
            # When a visit_name already exists, remap instances to the
            # existing timepoint_id instead of creating a duplicate.
            for tp in tps:
                existing_id = visit_name_to_tp_id.get(tp.visit_name)
                if existing_id is not None:
                    old_id = tp.timepoint_id
                    for inst in insts:
                        if inst.timepoint_id == old_id:
                            inst.timepoint_id = existing_id
                    logger.debug(
                        "Dedup timepoint %r: reused %s", tp.visit_name, existing_id,
                    )
                else:
                    visit_name_to_tp_id[tp.visit_name] = tp.timepoint_id
                    all_timepoints.append(tp)

            all_activities.extend(acts)
            all_instances.extend(insts)
            all_synonyms.extend(syns)

        # Build epoch list from the visit types seen
        epochs = self._build_epochs(
            all_timepoints, run_id, source_run_id, source_sha256, registry_id, timestamp
        )

        # Assign epoch_id to timepoints
        epoch_name_to_id = {e.epoch_name: e.epoch_id for e in epochs}
        for tp in all_timepoints:
            epoch_name = _EPOCH_FOR_TYPE.get(tp.visit_type, "Treatment")
            tp.epoch_id = epoch_name_to_id.get(epoch_name, "")

        return epochs, all_timepoints, all_activities, all_instances, all_synonyms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_table(
        self,
        table: RawSoaTable,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        registry_id: str,
        timestamp: str,
        activity_name_to_id: dict[str, str],
    ) -> tuple[
        list[UsdmTimepoint],
        list[UsdmActivity],
        list[UsdmScheduledInstance],
        list[SynonymMapping],
    ]:
        """Map a single RawSoaTable to entity lists."""
        # Resolve visit headers → timepoints
        timepoints: list[UsdmTimepoint] = []
        synonyms: list[SynonymMapping] = []
        tp_index: list[str] = []  # timepoint_id per visit column

        for i, header in enumerate(table.visit_headers):
            # Skip boilerplate headers (PTCV-74)
            if _is_boilerplate_header(header):
                tp_index.append("")  # placeholder for column alignment
                logger.debug(
                    "Skipping boilerplate visit header [%d]: %r", i, header,
                )
                continue

            # Prefer day_headers for temporal info if present
            temporal_hint = (
                table.day_headers[i]
                if i < len(table.day_headers) and table.day_headers[i].strip()
                else header
            )
            resolved, mapping = self._resolver.resolve_to_mapping(
                temporal_hint, run_id, timestamp
            )
            # Override canonical name with the original header (more readable)
            synonyms.append(mapping)

            tp_id = f"tp-{uuid.uuid4().hex[:8]}"
            tp_index.append(tp_id)

            # Check 7 required attributes (ALCOA+ Accurate)
            review = resolved.confidence < 0.80 or resolved.day_offset == 0 and (
                resolved.visit_type == "Treatment"
                and resolved.method == "default"
            )

            tp = UsdmTimepoint(
                run_id=run_id,
                source_run_id=source_run_id,
                source_sha256=source_sha256,
                registry_id=registry_id,
                timepoint_id=tp_id,
                epoch_id="",  # filled after epoch build
                visit_name=_canonicalize_visit_name(header) or resolved.visit_type,
                visit_type=resolved.visit_type,
                day_offset=resolved.day_offset,
                window_early=resolved.window_early,
                window_late=resolved.window_late,
                mandatory=resolved.mandatory,
                repeat_cycle=resolved.repeat_cycle,
                conditional_rule=resolved.conditional_rule,
                review_required=review,
                extraction_timestamp_utc=timestamp,
            )
            timepoints.append(tp)

        # Map activities and scheduled instances
        activities: list[UsdmActivity] = []
        instances: list[UsdmScheduledInstance] = []

        for activity_name, scheduled_flags in table.activities:
            name = activity_name.strip()
            if not name:
                continue

            # Reuse activity_id across tables for the same activity
            if name not in activity_name_to_id:
                act_id = f"act-{uuid.uuid4().hex[:8]}"
                activity_name_to_id[name] = act_id
                act = UsdmActivity(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    activity_id=act_id,
                    activity_name=name,
                    activity_type=self._classify_activity(name),
                    extraction_timestamp_utc=timestamp,
                )
                activities.append(act)
            else:
                act_id = activity_name_to_id[name]

            for col_idx, is_scheduled in enumerate(scheduled_flags):
                if not is_scheduled:
                    continue
                if col_idx >= len(tp_index):
                    continue
                if not tp_index[col_idx]:  # PTCV-74: boilerplate column
                    continue
                inst = UsdmScheduledInstance(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    instance_id=f"si-{uuid.uuid4().hex[:8]}",
                    activity_id=act_id,
                    timepoint_id=tp_index[col_idx],
                    scheduled=True,
                    extraction_timestamp_utc=timestamp,
                )
                instances.append(inst)

        return timepoints, activities, instances, synonyms

    @staticmethod
    def _build_epochs(
        timepoints: list[UsdmTimepoint],
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        registry_id: str,
        timestamp: str,
    ) -> list[UsdmEpoch]:
        """Create one UsdmEpoch per distinct epoch group seen in timepoints."""
        seen: dict[str, UsdmEpoch] = {}
        order = 1
        for tp in timepoints:
            epoch_name = _EPOCH_FOR_TYPE.get(tp.visit_type, "Treatment")
            if epoch_name not in seen:
                seen[epoch_name] = UsdmEpoch(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    epoch_id=f"ep-{uuid.uuid4().hex[:8]}",
                    epoch_name=epoch_name,
                    epoch_type=epoch_name,
                    order=order,
                    extraction_timestamp_utc=timestamp,
                )
                order += 1

        # Sort by canonical order
        def _sort_key(name: str) -> int:
            try:
                return _EPOCH_ORDER.index(name)
            except ValueError:
                return 99

        return sorted(seen.values(), key=lambda e: _sort_key(e.epoch_name))

    @staticmethod
    def _classify_activity(name: str) -> str:
        """Classify an activity by keyword matching."""
        lower = name.lower()
        for keyword, activity_type in _ACTIVITY_KEYWORDS:
            if keyword in lower:
                return activity_type
        return "Assessment"
