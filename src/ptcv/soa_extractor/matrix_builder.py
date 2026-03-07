"""Assessment x Visit x Type matrix builder (PTCV-152).

Joins USDM entities (timepoints, activities, scheduled instances)
into a flat DataFrame where each row is one (assessment, visit) pair.

Two entry points:
  build_assessment_matrix()            — end-to-end from AssembledProtocol
  build_assessment_matrix_from_usdm()  — from pre-extracted USDM entities

Target output schema::

    assessment_name  str   "CBC", "Vital Signs", "ECG"
    assessment_type  str   "Specimens", "Clinical Encounter", "At-home"
    visit_label      str   "Screening", "V1", "Week 2"
    day_offset       int   0, 1, 8  (screening = day 0)
    scheduled        bool  True
    frequency        str   "daily" (only when at-home; column dropped
                           if no at-home assessments present)

Risk tier: MEDIUM — data pipeline transformation (no patient data).
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

import pandas as pd

from ..ich_parser.template_assembler import AssembledProtocol
from ..ui.components.schedule_of_visits import classify_swimlane
from .mapper import UsdmMapper, _ACTIVITY_KEYWORDS
from .models import (
    UsdmActivity,
    UsdmScheduledInstance,
    UsdmTimepoint,
)
from .parser import SoaTableParser
from .query_bridge import assembled_to_sections
from .resolver import SynonymResolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

_AT_HOME_RE = re.compile(
    r"\be[-.]?diary\b|\belectronic\s+diary\b|\bpatient\s+diary\b"
    r"|\bdiary\b|\bat[-\s]?home\b|\bself[-\s]?report",
    re.IGNORECASE,
)

_FREQUENCY_RE = re.compile(
    r"\b(daily|weekly|twice\s+daily|bi[-\s]?weekly"
    r"|every\s+\d+\s+(?:days?|weeks?|hours?))\b",
    re.IGNORECASE,
)

# Bare lowercase prose words (no digits, no capitals, no clinical
# abbreviations).  Used to detect garbage visit headers like
# "double-blind,", "randomized,", "placebo-controlled,".
_PROSE_WORD_RE = re.compile(r"^[a-z][a-z-]{2,}[,.]?$")

# Visit-like patterns that should NOT be flagged as prose.
_VISIT_LIKE_RE = re.compile(
    r"(?:screen|baseline|day|week|month|year|cycle|visit|"
    r"unscheduled|termination|follow|end|eot|ltfu|dose)",
    re.IGNORECASE,
)

# Minimum thresholds for valid data.
_MIN_VISITS = 2
_MIN_ASSESSMENTS = 2

# Required output columns (excluding optional ``frequency``).
_COLUMNS = [
    "assessment_name",
    "assessment_type",
    "visit_label",
    "day_offset",
    "scheduled",
]


# ---------------------------------------------------------------------------
# Primary path — from cached USDM entities
# ---------------------------------------------------------------------------


def build_assessment_matrix_from_usdm(
    timepoints: Sequence[UsdmTimepoint],
    activities: Sequence[UsdmActivity],
    instances: Sequence[UsdmScheduledInstance],
    *,
    anchor_screening: bool = True,
) -> pd.DataFrame:
    """Build assessment x visit matrix from USDM entities.

    Joins scheduled instances to their activities and timepoints,
    classifies each assessment by swimlane type, detects at-home
    assessments, and optionally anchors day offsets to screening.

    Args:
        timepoints: USDM timepoints from the SoA pipeline.
        activities: USDM activities from the SoA pipeline.
        instances: Scheduled instances linking activities to
            timepoints.
        anchor_screening: If True (default), rebase day_offset so
            screening = day 0.  If False, retain original offsets.

    Returns:
        DataFrame with columns: assessment_name, assessment_type,
        visit_label, day_offset, scheduled.  Optionally includes
        a ``frequency`` column when at-home assessments are
        detected.  Returns empty DataFrame if validation fails.
    """
    if not instances:
        return _empty_df()

    tp_by_id = {tp.timepoint_id: tp for tp in timepoints}
    act_by_id = {a.activity_id: a for a in activities}

    rows: list[dict] = []
    has_frequency = False

    for inst in instances:
        if not inst.scheduled:
            continue
        tp = tp_by_id.get(inst.timepoint_id)
        act = act_by_id.get(inst.activity_id)
        if tp is None or act is None:
            continue

        # Classify assessment type via swimlane logic.
        atype = classify_swimlane(act.activity_type, act.activity_name)

        # Override to At-home for diary/self-report assessments.
        frequency: Optional[str] = None
        if _AT_HOME_RE.search(act.activity_name):
            atype = "At-home"
            m = _FREQUENCY_RE.search(act.activity_name)
            if m:
                frequency = m.group(1).lower()
                has_frequency = True

        rows.append({
            "assessment_name": act.activity_name,
            "assessment_type": atype,
            "visit_label": tp.visit_name,
            "day_offset": tp.day_offset,
            "scheduled": True,
            "frequency": frequency,
        })

    if not rows:
        return _empty_df()

    df = pd.DataFrame(rows)

    # Anchor screening at day 0.
    if anchor_screening:
        df = _anchor_to_screening(df, timepoints)

    # Validate the matrix.
    if not _validate_matrix(df):
        logger.warning(
            "Matrix validation failed — returning empty DataFrame. "
            "Unique visits: %d, unique assessments: %d",
            df["visit_label"].nunique(),
            df["assessment_name"].nunique(),
        )
        return _empty_df()

    # Drop frequency column if no at-home assessments found.
    if not has_frequency:
        df = df.drop(columns=["frequency"])

    df = df.sort_values(
        ["day_offset", "assessment_name"], ignore_index=True,
    )
    return df


# ---------------------------------------------------------------------------
# End-to-end path — from AssembledProtocol
# ---------------------------------------------------------------------------


def build_assessment_matrix(
    assembled: AssembledProtocol,
    registry_id: str = "",
    *,
    anchor_screening: bool = True,
) -> pd.DataFrame:
    """Build assessment x visit matrix from AssembledProtocol.

    End-to-end pipeline: query_bridge -> parser -> mapper ->
    matrix join.  Falls back to hit-level iteration when the
    parser cannot find a structured SoA table.

    Args:
        assembled: Completed AssembledProtocol from the query
            pipeline.
        registry_id: Trial identifier for lineage.
        anchor_screening: If True, rebase day_offset so
            screening = day 0.

    Returns:
        DataFrame with columns: assessment_name, assessment_type,
        visit_label, day_offset, scheduled.  Empty DataFrame if
        no data extractable.
    """
    sections = assembled_to_sections(
        assembled, registry_id=registry_id,
    )

    # Try structured table parsing first.
    parser = SoaTableParser()
    raw_table = parser.parse(sections)

    if raw_table is not None:
        ts = datetime.now(timezone.utc).isoformat()
        run_id = str(uuid.uuid4())
        mapper = UsdmMapper(resolver=SynonymResolver(use_spacy=False))
        _, timepoints, activities, instances, _ = mapper.map(
            [raw_table],
            run_id=run_id,
            source_run_id="query_pipeline",
            source_sha256="",
            registry_id=registry_id,
            timestamp=ts,
        )
        result = build_assessment_matrix_from_usdm(
            timepoints, activities, instances,
            anchor_screening=anchor_screening,
        )
        if not result.empty:
            return result
        logger.info(
            "Structured table produced invalid matrix; "
            "falling back to hit-level extraction.",
        )

    # Fallback: discover assessments/visits from raw hits.
    return _build_matrix_from_hits(
        assembled, registry_id, anchor_screening,
    )


# ---------------------------------------------------------------------------
# Screening anchor
# ---------------------------------------------------------------------------


def _anchor_to_screening(
    df: pd.DataFrame,
    timepoints: Sequence[UsdmTimepoint],
) -> pd.DataFrame:
    """Rebase day_offset so screening = day 0.

    Finds the minimum day_offset among Screening-type visits.
    If no screening visits exist, uses the minimum day_offset
    overall.

    Args:
        df: Matrix DataFrame with day_offset column.
        timepoints: USDM timepoints (for visit_type lookup).

    Returns:
        DataFrame with rebased day_offset values.
    """
    screening_offsets = [
        tp.day_offset
        for tp in timepoints
        if tp.visit_type == "Screening"
    ]
    if screening_offsets:
        anchor = min(screening_offsets)
    elif not df.empty:
        anchor = int(df["day_offset"].min())
    else:
        return df

    df = df.copy()
    df["day_offset"] = df["day_offset"] - anchor
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_matrix(df: pd.DataFrame) -> bool:
    """Validate that the matrix contains non-garbage data.

    Checks:
      1. At least ``_MIN_VISITS`` unique visit labels.
      2. At least ``_MIN_ASSESSMENTS`` unique assessment names.
      3. No more than 50% of visit labels look like prose words.

    Args:
        df: The candidate assessment matrix.

    Returns:
        True if valid, False if garbage data detected.
    """
    if df.empty:
        return False

    unique_visits = df["visit_label"].unique()
    unique_assessments = df["assessment_name"].unique()

    if len(unique_visits) < _MIN_VISITS:
        return False
    if len(unique_assessments) < _MIN_ASSESSMENTS:
        return False

    # Count prose-word visit labels.
    prose_count = 0
    for v in unique_visits:
        v_str = str(v).strip()
        if _PROSE_WORD_RE.match(v_str) and not _VISIT_LIKE_RE.search(
            v_str
        ):
            prose_count += 1

    if prose_count > len(unique_visits) * 0.5:
        return False

    return True


# ---------------------------------------------------------------------------
# Fallback — hit-level iteration
# ---------------------------------------------------------------------------


def _build_matrix_from_hits(
    assembled: AssembledProtocol,
    registry_id: str,
    anchor_screening: bool,
) -> pd.DataFrame:
    """Fallback: discover assessments and visits from raw hits.

    Scans B.4 and B.7 section hits for assessment keywords and
    visit patterns.  Produces cross-product matrix (lower fidelity
    — all discovered assessments at all discovered visits).

    Args:
        assembled: AssembledProtocol from query pipeline.
        registry_id: Trial identifier.
        anchor_screening: Whether to anchor offsets to screening.

    Returns:
        DataFrame with same schema as primary path.  May have
        all assessments scheduled at all visits (no cell-level
        granularity).
    """
    logger.warning(
        "SoA table not parseable; using fallback hit-level "
        "extraction for %s.",
        registry_id,
    )

    # Collect text from B.4 and B.7 hits.
    texts: list[str] = []
    for code in ("B.4", "B.7"):
        section = assembled.get_section(code)
        if section is None or not section.populated:
            continue
        for hit in section.hits:
            if hit.extracted_content:
                texts.append(hit.extracted_content)

    if not texts:
        return _empty_df()

    combined = "\n".join(texts).lower()

    # Discover assessments via _ACTIVITY_KEYWORDS.
    found_assessments: dict[str, str] = {}  # name → type
    for keyword, atype in _ACTIVITY_KEYWORDS:
        if keyword in combined and keyword not in found_assessments:
            # Use the keyword as-is for the name (title-cased).
            name = keyword.title()
            swimlane = classify_swimlane(atype, name)
            if _AT_HOME_RE.search(name):
                swimlane = "At-home"
            found_assessments[name] = swimlane

    # Discover visits via SynonymResolver patterns.
    resolver = SynonymResolver(use_spacy=False)
    visit_patterns = re.findall(
        r"(?:screening|baseline|day\s*-?\d+|week\s*\d+"
        r"|month\s*\d+|year\s*\d+|cycle\s*\d+\s*day\s*\d+"
        r"|visit\s*\d+|v\d+|end\s+of\s+(?:study|treatment)"
        r"|follow[-\s]?up|unscheduled)",
        combined,
        re.IGNORECASE,
    )

    # Deduplicate and resolve visits.
    seen_labels: set[str] = set()
    visits: list[tuple[str, int]] = []  # (label, day_offset)
    for raw in visit_patterns:
        label = raw.strip().title()
        if label in seen_labels:
            continue
        seen_labels.add(label)
        resolved = resolver.resolve(raw.strip())
        visits.append((label, resolved.day_offset))

    if not found_assessments or not visits:
        return _empty_df()

    # Build cross-product matrix.
    rows: list[dict] = []
    for name, atype in found_assessments.items():
        for vlabel, day in visits:
            rows.append({
                "assessment_name": name,
                "assessment_type": atype,
                "visit_label": vlabel,
                "day_offset": day,
                "scheduled": True,
            })

    df = pd.DataFrame(rows)

    if anchor_screening:
        # Build synthetic timepoints for the anchor function.
        synthetic_tps = [
            UsdmTimepoint(
                run_id="", source_run_id="", source_sha256="",
                registry_id=registry_id, timepoint_id="",
                epoch_id="", visit_name=vlabel,
                visit_type=(
                    "Screening" if "screen" in vlabel.lower()
                    else "Treatment"
                ),
                day_offset=day, window_early=0, window_late=0,
                mandatory=True,
            )
            for vlabel, day in visits
        ]
        df = _anchor_to_screening(df, synthetic_tps)

    if not _validate_matrix(df):
        return _empty_df()

    df = df.sort_values(
        ["day_offset", "assessment_name"], ignore_index=True,
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the target schema."""
    return pd.DataFrame(columns=_COLUMNS)
