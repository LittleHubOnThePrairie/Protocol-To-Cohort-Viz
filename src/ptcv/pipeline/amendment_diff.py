"""Protocol amendment diff engine for versioned SoA tracking (PTCV-59).

Compares two MockSdtmDataset objects (from different protocol versions) and
produces categorised change records covering visit additions/removals, timing
shifts, window changes, and assessment-level status modifications.

Risk tier: MEDIUM — protocol structure comparison (no patient data).

Regulatory references:
- ICH E6(R3) §5.4: Protocol amendments must be documented
- ALCOA+ Traceable: source_sha256 links each version to its source PDF
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..sdtm.models import SoaCellMetadata
    from ..sdtm.soa_mapper import MockSdtmDataset


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AmendmentChange:
    """A single categorised change between two protocol versions.

    Attributes:
        category: Change type — visit_added, visit_removed, window_changed,
            assessment_added, assessment_removed, status_changed, or
            timing_changed.
        domain: SDTM domain affected ("TV", "TE", or "SOA_MATRIX").
        description: Human-readable change summary.
        visitnum: Visit number (if applicable).
        visit_name: Visit name (if applicable).
        assessment: Assessment/activity name (if applicable).
        field: Field that changed (e.g. "VISITDY", "TVSTRL").
        old_value: Previous value as string.
        new_value: New value as string.
    """

    category: str
    domain: str
    description: str
    visitnum: int | None = None
    visit_name: str = ""
    assessment: str = ""
    field: str = ""
    old_value: str = ""
    new_value: str = ""


@dataclasses.dataclass
class AmendmentDiff:
    """Result of comparing two protocol versions.

    Attributes:
        registry_id: Trial registry identifier.
        version_a: Source version label (e.g. "1.0").
        version_b: Target version label (e.g. "1.1").
        changes: List of categorised changes.
        diff_timestamp_utc: ISO 8601 UTC timestamp of the comparison.
        source_sha256_a: SHA-256 of version A source PDF.
        source_sha256_b: SHA-256 of version B source PDF.
    """

    registry_id: str
    version_a: str
    version_b: str
    changes: list[AmendmentChange]
    diff_timestamp_utc: str
    source_sha256_a: str = ""
    source_sha256_b: str = ""

    @property
    def change_count(self) -> int:
        """Total number of changes detected."""
        return len(self.changes)

    @property
    def has_changes(self) -> bool:
        """True when at least one change was detected."""
        return len(self.changes) > 0

    def changes_by_category(self) -> dict[str, list[AmendmentChange]]:
        """Group changes by category.

        Returns:
            Dict mapping category name to list of changes.
        """
        grouped: dict[str, list[AmendmentChange]] = defaultdict(list)
        for change in self.changes:
            grouped[change.category].append(change)
        return dict(grouped)


@dataclasses.dataclass
class ProtocolVersion:
    """A single protocol version in the amendment chain.

    Attributes:
        registry_id: Trial registry identifier.
        version: Version label (e.g. "1.0", "1.1", "2.0").
        amendment_number: Zero-padded amendment number (e.g. "00", "01").
        source_sha256: SHA-256 of the source protocol PDF.
        timestamp_utc: ISO 8601 UTC timestamp of extraction.
    """

    registry_id: str
    version: str
    amendment_number: str
    source_sha256: str
    timestamp_utc: str = ""


@dataclasses.dataclass
class VersionAncestry:
    """Chronologically ordered chain of protocol versions.

    Attributes:
        registry_id: Trial registry identifier.
        versions: Protocol versions in chronological order.
    """

    registry_id: str
    versions: list[ProtocolVersion]

    @property
    def current(self) -> ProtocolVersion | None:
        """Most recent protocol version, or None if empty."""
        return self.versions[-1] if self.versions else None

    @property
    def chain(self) -> list[str]:
        """Ordered version label strings (e.g. ["1.0", "1.1", "2.0"])."""
        return [v.version for v in self.versions]


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------

class AmendmentDiffEngine:
    """Compares two MockSdtmDataset objects to detect protocol amendments.

    Usage::

        engine = AmendmentDiffEngine()
        diff = engine.diff(dataset_v1, dataset_v2, "1.0", "1.1")
        for change in diff.changes:
            print(change.description)
    """

    def diff(
        self,
        dataset_a: "MockSdtmDataset",
        dataset_b: "MockSdtmDataset",
        version_a: str = "1.0",
        version_b: str = "1.1",
        registry_id: str = "",
    ) -> AmendmentDiff:
        """Compare two MockSdtmDatasets and return categorised changes.

        Args:
            dataset_a: Baseline protocol version (earlier).
            dataset_b: Amended protocol version (later).
            version_a: Label for version A.
            version_b: Label for version B.
            registry_id: Trial registry identifier.

        Returns:
            AmendmentDiff with all detected changes.
        """
        rid = registry_id or dataset_a.studyid or dataset_b.studyid
        changes: list[AmendmentChange] = []

        changes.extend(self._diff_visits(dataset_a.tv, dataset_b.tv))
        changes.extend(self._diff_timing(dataset_a.tv, dataset_b.tv))
        changes.extend(self._diff_windows(dataset_a.tv, dataset_b.tv))
        changes.extend(
            self._diff_assessments(
                dataset_a.soa_matrix, dataset_b.soa_matrix
            )
        )
        changes.extend(
            self._diff_status(dataset_a.soa_matrix, dataset_b.soa_matrix)
        )

        return AmendmentDiff(
            registry_id=rid,
            version_a=version_a,
            version_b=version_b,
            changes=changes,
            diff_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def build_ancestry(
        self,
        versions: list[ProtocolVersion],
    ) -> VersionAncestry:
        """Build a chronologically ordered version ancestry chain.

        Sorts by amendment_number then version label.

        Args:
            versions: Unordered list of protocol versions.

        Returns:
            VersionAncestry with versions in chronological order.
        """
        if not versions:
            return VersionAncestry(registry_id="", versions=[])

        sorted_versions = sorted(
            versions,
            key=lambda v: (v.amendment_number, v.version),
        )
        return VersionAncestry(
            registry_id=sorted_versions[0].registry_id,
            versions=sorted_versions,
        )

    # -- Private diff helpers -----------------------------------------------

    @staticmethod
    def _tv_visit_map(
        tv: pd.DataFrame,
    ) -> dict[int, dict[str, object]]:
        """Build a VISITNUM → row-dict mapping from a TV DataFrame."""
        result: dict[int, dict[str, object]] = {}
        for _, row in tv.iterrows():
            vnum = int(row["VISITNUM"])
            result[vnum] = dict(row)
        return result

    def _diff_visits(
        self,
        tv_a: pd.DataFrame,
        tv_b: pd.DataFrame,
    ) -> list[AmendmentChange]:
        """Detect added and removed visits."""
        changes: list[AmendmentChange] = []
        map_a = self._tv_visit_map(tv_a)
        map_b = self._tv_visit_map(tv_b)
        nums_a = set(map_a.keys())
        nums_b = set(map_b.keys())

        for vnum in sorted(nums_b - nums_a):
            row = map_b[vnum]
            visit = str(row.get("VISIT", ""))
            visitdy = row.get("VISITDY", "")
            changes.append(AmendmentChange(
                category="visit_added",
                domain="TV",
                description=(
                    f"Visit '{visit}' (VISITNUM {vnum}) added at Day {visitdy}"
                ),
                visitnum=vnum,
                visit_name=visit,
                field="VISITNUM",
                new_value=str(vnum),
            ))

        for vnum in sorted(nums_a - nums_b):
            row = map_a[vnum]
            visit = str(row.get("VISIT", ""))
            changes.append(AmendmentChange(
                category="visit_removed",
                domain="TV",
                description=(
                    f"Visit '{visit}' (VISITNUM {vnum}) removed"
                ),
                visitnum=vnum,
                visit_name=visit,
                field="VISITNUM",
                old_value=str(vnum),
            ))

        return changes

    def _diff_timing(
        self,
        tv_a: pd.DataFrame,
        tv_b: pd.DataFrame,
    ) -> list[AmendmentChange]:
        """Detect timing changes (VISITDY) for matching visits."""
        changes: list[AmendmentChange] = []
        map_a = self._tv_visit_map(tv_a)
        map_b = self._tv_visit_map(tv_b)
        common = set(map_a.keys()) & set(map_b.keys())

        for vnum in sorted(common):
            dy_a = map_a[vnum].get("VISITDY")
            dy_b = map_b[vnum].get("VISITDY")
            if _safe_ne(dy_a, dy_b):
                visit = str(map_b[vnum].get("VISIT", ""))
                changes.append(AmendmentChange(
                    category="timing_changed",
                    domain="TV",
                    description=(
                        f"Visit '{visit}' (VISITNUM {vnum}) moved from "
                        f"Day {dy_a} to Day {dy_b}"
                    ),
                    visitnum=vnum,
                    visit_name=visit,
                    field="VISITDY",
                    old_value=str(dy_a),
                    new_value=str(dy_b),
                ))

        return changes

    def _diff_windows(
        self,
        tv_a: pd.DataFrame,
        tv_b: pd.DataFrame,
    ) -> list[AmendmentChange]:
        """Detect visit window changes (TVSTRL, TVENRL)."""
        changes: list[AmendmentChange] = []
        map_a = self._tv_visit_map(tv_a)
        map_b = self._tv_visit_map(tv_b)
        common = set(map_a.keys()) & set(map_b.keys())

        for vnum in sorted(common):
            for field in ("TVSTRL", "TVENRL"):
                val_a = map_a[vnum].get(field)
                val_b = map_b[vnum].get(field)
                if _safe_ne(val_a, val_b):
                    visit = str(map_b[vnum].get("VISIT", ""))
                    changes.append(AmendmentChange(
                        category="window_changed",
                        domain="TV",
                        description=(
                            f"Visit '{visit}' (VISITNUM {vnum}) "
                            f"{field} changed from {val_a} to {val_b}"
                        ),
                        visitnum=vnum,
                        visit_name=visit,
                        field=field,
                        old_value=str(val_a),
                        new_value=str(val_b),
                    ))

        return changes

    def _diff_assessments(
        self,
        matrix_a: dict[tuple[int, str], "SoaCellMetadata"],
        matrix_b: dict[tuple[int, str], "SoaCellMetadata"],
    ) -> list[AmendmentChange]:
        """Detect added and removed assessments from soa_matrix."""
        changes: list[AmendmentChange] = []
        if not matrix_a and not matrix_b:
            return changes

        keys_a = set(matrix_a.keys())
        keys_b = set(matrix_b.keys())

        for vnum, assessment in sorted(keys_b - keys_a):
            cell = matrix_b[(vnum, assessment)]
            changes.append(AmendmentChange(
                category="assessment_added",
                domain="SOA_MATRIX",
                description=(
                    f"Assessment '{assessment}' added at Visit {vnum}"
                ),
                visitnum=vnum,
                assessment=assessment,
                field="assessment",
                new_value=cell.status,
            ))

        for vnum, assessment in sorted(keys_a - keys_b):
            cell = matrix_a[(vnum, assessment)]
            changes.append(AmendmentChange(
                category="assessment_removed",
                domain="SOA_MATRIX",
                description=(
                    f"Assessment '{assessment}' removed from Visit {vnum}"
                ),
                visitnum=vnum,
                assessment=assessment,
                field="assessment",
                old_value=cell.status,
            ))

        return changes

    def _diff_status(
        self,
        matrix_a: dict[tuple[int, str], "SoaCellMetadata"],
        matrix_b: dict[tuple[int, str], "SoaCellMetadata"],
    ) -> list[AmendmentChange]:
        """Detect status changes for matching soa_matrix cells."""
        changes: list[AmendmentChange] = []
        if not matrix_a or not matrix_b:
            return changes

        common = set(matrix_a.keys()) & set(matrix_b.keys())
        for vnum, assessment in sorted(common):
            cell_a = matrix_a[(vnum, assessment)]
            cell_b = matrix_b[(vnum, assessment)]
            if cell_a.status != cell_b.status:
                changes.append(AmendmentChange(
                    category="status_changed",
                    domain="SOA_MATRIX",
                    description=(
                        f"Assessment '{assessment}' at Visit {vnum} "
                        f"changed from {cell_a.status} to {cell_b.status}"
                    ),
                    visitnum=vnum,
                    assessment=assessment,
                    field="status",
                    old_value=cell_a.status,
                    new_value=cell_b.status,
                ))

        return changes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ne(a: object, b: object) -> bool:
    """Compare two values for inequality, handling NaN safely."""
    if isinstance(a, float) and isinstance(b, float):
        if pd.isna(a) and pd.isna(b):
            return False
        if pd.isna(a) or pd.isna(b):
            return True
    return a != b
