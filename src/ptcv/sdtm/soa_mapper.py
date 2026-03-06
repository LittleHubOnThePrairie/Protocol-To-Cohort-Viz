"""SoA-to-SDTM mapper — maps RawSoaTable data to mock SDTM domains (PTCV-53).

Bridges the extracted SoA table structure (from PTCV-51) and priority section
context (from PTCV-52) into CDISC SDTM Trial Design domain DataFrames:
  TV — Trial Visits (from SoA visit columns)
  TA — Trial Arms (from B.4 section text)
  TE — Trial Elements (from SoA column groupings + B.4 text)
  SE — Subject Elements (per-subject planned visit schedule)

Also produces an assessment-to-SDTM-domain mapping matrix from SoA row labels.

Risk tier: MEDIUM — regulatory submission artefacts; no patient data.

Regulatory references:
- SDTMIG v3.4 Section 7.4: Trial Design Domains
- CDISC SDTM v1.7 for variable attributes
- CDISC Controlled Terminology for domain codes
"""

from __future__ import annotations

import dataclasses
import re
import uuid
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..soa_extractor.resolver import SynonymResolver
from .domain_generators import _extract_arms, _extract_elements
from .models import SoaCellMetadata

if TYPE_CHECKING:
    from ..ich_parser.models import IchSection
    from ..soa_extractor.models import RawSoaTable


# ---------------------------------------------------------------------------
# Assessment → SDTM domain mapping
# ---------------------------------------------------------------------------

_ASSESSMENT_TO_DOMAIN: list[tuple[str, str, str]] = [
    # (keyword, domain_code, domain_name)
    ("informed consent", "DS", "Disposition"),
    ("consent", "DS", "Disposition"),
    ("ecg", "EG", "ECG Test Results"),
    ("electrocardiogram", "EG", "ECG Test Results"),
    ("12-lead", "EG", "ECG Test Results"),
    ("vital sign", "VS", "Vital Signs"),
    ("vitals", "VS", "Vital Signs"),
    ("blood pressure", "VS", "Vital Signs"),
    ("heart rate", "VS", "Vital Signs"),
    ("pulse", "VS", "Vital Signs"),
    ("temperature", "VS", "Vital Signs"),
    ("weight", "VS", "Vital Signs"),
    ("height", "VS", "Vital Signs"),
    ("bmi", "VS", "Vital Signs"),
    ("lab", "LB", "Laboratory Test Results"),
    ("haematology", "LB", "Laboratory Test Results"),
    ("hematology", "LB", "Laboratory Test Results"),
    ("chemistry", "LB", "Laboratory Test Results"),
    ("biochemistry", "LB", "Laboratory Test Results"),
    ("urinalysis", "LB", "Laboratory Test Results"),
    ("blood count", "LB", "Laboratory Test Results"),
    ("cbc", "LB", "Laboratory Test Results"),
    ("coagulation", "LB", "Laboratory Test Results"),
    ("serology", "LB", "Laboratory Test Results"),
    ("lipid panel", "LB", "Laboratory Test Results"),
    ("liver function", "LB", "Laboratory Test Results"),
    ("renal function", "LB", "Laboratory Test Results"),
    ("thyroid", "LB", "Laboratory Test Results"),
    ("pharmacokinetic", "PC", "Pharmacokinetics Concentrations"),
    ("pk sample", "PC", "Pharmacokinetics Concentrations"),
    ("drug concentration", "PC", "Pharmacokinetics Concentrations"),
    ("physical exam", "PE", "Physical Examination"),
    ("physical examination", "PE", "Physical Examination"),
    ("adverse event", "AE", "Adverse Events"),
    ("adverse", "AE", "Adverse Events"),
    ("sae", "AE", "Adverse Events"),
    ("concomitant", "CM", "Concomitant Medications"),
    ("medication", "CM", "Concomitant Medications"),
    ("prior medication", "CM", "Concomitant Medications"),
    ("imaging", "MI", "Microscopic Findings"),
    ("mri", "MI", "Microscopic Findings"),
    ("ct scan", "MI", "Microscopic Findings"),
    ("x-ray", "MI", "Microscopic Findings"),
    ("tumor", "TU", "Tumor Results"),
    ("tumour", "TU", "Tumor Results"),
    ("recist", "TU", "Tumor Results"),
    ("biopsy", "PR", "Procedures"),
    ("surgery", "PR", "Procedures"),
    ("pregnancy", "SC", "Subject Characteristics"),
    ("questionnaire", "QS", "Questionnaires"),
    ("diary", "QS", "Questionnaires"),
    ("quality of life", "QS", "Questionnaires"),
    ("ecog", "QS", "Questionnaires"),
    ("performance status", "QS", "Questionnaires"),
]

# Visit type → epoch name (SDTM TE terminology)
_VISIT_TYPE_TO_EPOCH: dict[str, str] = {
    "Screening": "SCRN",
    "Baseline": "TRT",
    "Treatment": "TRT",
    "Follow-up": "FUP",
    "Long-term Follow-up": "FUP",
    "End of Study": "EOS",
    "Early Termination": "EOS",
    "Unscheduled": "TRT",
    "Remote": "TRT",
}

_ETCD_TO_ELEMENT: dict[str, str] = {
    "SCRN": "Screening",
    "TRT": "Treatment",
    "FUP": "Follow-up",
    "EOS": "End of Study",
    "RUN": "Run-in",
    "RAND": "Randomization",
}

# SDTM domain → Faro-style assessment category (PTCV-1 research)
_DOMAIN_TO_CATEGORY: dict[str, str] = {
    "VS": "safety",
    "LB": "safety",
    "EG": "safety",
    "AE": "safety",
    "PE": "safety",
    "CM": "safety",
    "TU": "efficacy",
    "QS": "efficacy",
    "PC": "efficacy",
    "MI": "efficacy",
    "DS": "operational",
    "PR": "operational",
    "SC": "operational",
    "FA": "operational",
}


@dataclasses.dataclass
class MockSdtmDataset:
    """Container for mock SDTM trial design domain DataFrames.

    Attributes:
        tv: Trial Visits domain DataFrame.
        ta: Trial Arms domain DataFrame.
        te: Trial Elements domain DataFrame.
        se: Subject Elements domain DataFrame.
        domain_mapping: Assessment-to-SDTM domain mapping matrix.
        studyid: STUDYID used across all domains.
        run_id: Pipeline run UUID4.
        soa_matrix: Cell-level metadata keyed by (visitnum, assessment).
    """

    tv: pd.DataFrame
    ta: pd.DataFrame
    te: pd.DataFrame
    se: pd.DataFrame
    domain_mapping: pd.DataFrame
    studyid: str
    run_id: str
    soa_matrix: dict[tuple[int, str], SoaCellMetadata] = dataclasses.field(
        default_factory=dict
    )

    @property
    def domains(self) -> dict[str, pd.DataFrame]:
        """Return all domain DataFrames as a dict."""
        return {"TV": self.tv, "TA": self.ta, "TE": self.te, "SE": self.se}

    def to_csv(self, prefix: str = "") -> dict[str, str]:
        """Serialize all domains to CSV strings.

        Args:
            prefix: Optional prefix for domain names in the returned dict.

        Returns:
            Dict mapping domain name to CSV string.
        """
        result: dict[str, str] = {}
        for name, df in self.domains.items():
            key = f"{prefix}{name}" if prefix else name
            result[key] = df.to_csv(index=False)
        result[f"{prefix}DOMAIN_MAP" if prefix else "DOMAIN_MAP"] = (
            self.domain_mapping.to_csv(index=False)
        )
        if self.soa_matrix:
            matrix_rows = [
                {
                    "VISITNUM": cell.visitnum,
                    "ASSESSMENT": cell.assessment,
                    "STATUS": cell.status,
                    "CONDITION": cell.condition,
                    "CATEGORY": cell.category,
                    "CDASH_DOMAIN": cell.cdash_domain,
                    "WINDOW_EARLY": cell.timing_window_days[0],
                    "WINDOW_LATE": cell.timing_window_days[1],
                }
                for cell in self.soa_matrix.values()
            ]
            matrix_key = f"{prefix}SOA_MATRIX" if prefix else "SOA_MATRIX"
            result[matrix_key] = pd.DataFrame(matrix_rows).to_csv(
                index=False
            )
        return result


def _extract_text(section: "IchSection") -> str:
    """Return text from IchSection.content_json."""
    import json

    try:
        data = json.loads(section.content_json)
        return data.get("text_excerpt", "") or data.get("text", "") or ""
    except (json.JSONDecodeError, AttributeError):
        return ""


def _classify_assessment(name: str) -> tuple[str, str]:
    """Classify an assessment name to its SDTM domain.

    Args:
        name: Activity/assessment name from SoA row label.

    Returns:
        Tuple of (domain_code, domain_name). Defaults to ("FA", "Findings About")
        for unrecognized assessments.
    """
    lower = name.lower()
    for keyword, domain_code, domain_name in _ASSESSMENT_TO_DOMAIN:
        if keyword in lower:
            return domain_code, domain_name
    return "FA", "Findings About"


class SoaToSdtmMapper:
    """Maps RawSoaTable data to mock SDTM trial design domains.

    Consumes the structured SoA table output from PTCV-51 and optional
    IchSection context from PTCV-52 to produce TV, TA, TE, SE DataFrames
    and an assessment-to-SDTM domain mapping matrix.

    Args:
        resolver: SynonymResolver for visit header resolution.
            A new default resolver is created if None.
    """

    def __init__(self, resolver: SynonymResolver | None = None) -> None:
        self._resolver = resolver or SynonymResolver()

    def map(
        self,
        table: "RawSoaTable",
        sections: list["IchSection"] | None = None,
        studyid: str = "",
        run_id: str | None = None,
    ) -> MockSdtmDataset:
        """Map a RawSoaTable to a MockSdtmDataset.

        Args:
            table: Parsed SoA table from PTCV-51 pipeline.
            sections: Optional IchSection list for B.4/B.5 context.
            studyid: STUDYID value. Defaults to empty string.
            run_id: Pipeline run UUID4. Auto-generated if None.

        Returns:
            MockSdtmDataset containing TV, TA, TE, SE DataFrames
            and domain mapping matrix.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())
        if sections is None:
            sections = []

        tv_df = self._build_tv(table, studyid)
        ta_df = self._build_ta(sections, studyid)
        te_df = self._build_te(table, sections, studyid)
        se_df = self._build_se(tv_df, studyid)
        domain_map_df = self._build_domain_mapping(table)
        soa_matrix = self._build_soa_matrix(table, tv_df)

        return MockSdtmDataset(
            tv=tv_df,
            ta=ta_df,
            te=te_df,
            se=se_df,
            domain_mapping=domain_map_df,
            studyid=studyid,
            run_id=run_id,
            soa_matrix=soa_matrix,
        )

    def _build_tv(
        self, table: "RawSoaTable", studyid: str
    ) -> pd.DataFrame:
        """Build TV (Trial Visits) domain from SoA visit columns.

        Each visit column header is resolved via SynonymResolver to extract
        VISITNUM, VISIT name, VISITDY (day offset), and visit windows.
        Visits are ordered chronologically by day offset.
        """
        timestamp = ""
        resolved_visits: list[dict[str, Any]] = []

        for i, header in enumerate(table.visit_headers):
            temporal_hint = (
                table.day_headers[i]
                if i < len(table.day_headers) and table.day_headers[i].strip()
                else header
            )
            resolved, _ = self._resolver.resolve_to_mapping(
                temporal_hint, "tv-build", timestamp
            )
            resolved_visits.append({
                "visit_name": header.strip() or resolved.visit_type,
                "visit_type": resolved.visit_type,
                "day_offset": resolved.day_offset,
                "window_early": resolved.window_early,
                "window_late": resolved.window_late,
            })

        # Sort by day_offset for chronological ordering
        resolved_visits.sort(key=lambda v: v["day_offset"])

        rows: list[dict[str, Any]] = []
        for visitnum, visit in enumerate(resolved_visits, start=1):
            rows.append({
                "STUDYID": studyid,
                "DOMAIN": "TV",
                "VISITNUM": float(visitnum),
                "VISIT": visit["visit_name"][:40],
                "VISITDY": float(visit["day_offset"]),
                "TVSTRL": float(-visit["window_early"]),
                "TVENRL": float(visit["window_late"]),
            })

        return pd.DataFrame(rows)

    def _build_ta(
        self, sections: list["IchSection"], studyid: str
    ) -> pd.DataFrame:
        """Build TA (Trial Arms) domain from B.4 section text.

        Extracts treatment arm names and element structure from the B.4
        (Trial Design) section, producing ARMCD, ARM, and element ordering.
        """
        b4 = [s for s in sections if s.section_code == "B.4"]
        text = _extract_text(b4[0]) if b4 else ""

        arms = _extract_arms(text)
        elements = _extract_elements(text)

        rows: list[dict[str, Any]] = []
        for arm_idx, arm_label in enumerate(arms):
            armcd = f"ARM{arm_idx + 1:02d}"
            for et_idx, (etcd, element) in enumerate(elements):
                epoch = _ETCD_TO_ELEMENT.get(etcd, element)
                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "TA",
                    "ARMCD": armcd[:20],
                    "ARM": arm_label[:40],
                    "TAETORD": float(et_idx + 1),
                    "ETCD": etcd[:8],
                    "ELEMENT": element[:40],
                    "TABRANCH": "",
                    "TATRANS": "",
                    "EPOCH": epoch[:40],
                })

        return pd.DataFrame(rows)

    def _build_te(
        self,
        table: "RawSoaTable",
        sections: list["IchSection"],
        studyid: str,
    ) -> pd.DataFrame:
        """Build TE (Trial Elements) domain from SoA column groupings.

        Infers study phases from the visit types present in the SoA table
        visit columns, enriched with B.4 section text where available.
        """
        # Resolve visit types from SoA columns to determine epochs
        timestamp = ""
        seen_etcds: list[str] = []

        for i, header in enumerate(table.visit_headers):
            temporal_hint = (
                table.day_headers[i]
                if i < len(table.day_headers) and table.day_headers[i].strip()
                else header
            )
            resolved, _ = self._resolver.resolve_to_mapping(
                temporal_hint, "te-build", timestamp
            )
            etcd = _VISIT_TYPE_TO_EPOCH.get(resolved.visit_type, "TRT")
            if etcd not in seen_etcds:
                seen_etcds.append(etcd)

        # Enrich with B.4 text elements (may add phases not in SoA headers)
        b4 = [s for s in sections if s.section_code == "B.4"]
        text = _extract_text(b4[0]) if b4 else ""
        if text:
            text_elements = _extract_elements(text)
            for etcd, _ in text_elements:
                if etcd not in seen_etcds:
                    seen_etcds.append(etcd)

        # Ensure at least screening, treatment, follow-up
        for default_etcd in ("SCRN", "TRT", "FUP"):
            if default_etcd not in seen_etcds:
                seen_etcds.append(default_etcd)

        # Canonical order
        canonical_order = ["SCRN", "RUN", "RAND", "TRT", "FUP", "EOS"]
        seen_etcds.sort(
            key=lambda e: canonical_order.index(e)
            if e in canonical_order
            else 99
        )

        rows: list[dict[str, Any]] = []
        for etcd in seen_etcds:
            element = _ETCD_TO_ELEMENT.get(etcd, etcd)
            rows.append({
                "STUDYID": studyid,
                "DOMAIN": "TE",
                "ETCD": etcd[:8],
                "ELEMENT": element[:40],
                "TESTRL": f"VISIT='{element[:20]}'",
                "TEENRL": "",
                "TEDUR": "",
            })

        return pd.DataFrame(rows)

    def _build_se(
        self, tv_df: pd.DataFrame, studyid: str
    ) -> pd.DataFrame:
        """Build SE (Subject Elements) domain from TV.

        SE describes the per-subject planned visit schedule. Each SE row
        maps a subject element to its start and end relative day.
        Derived directly from the TV domain visits.
        """
        if tv_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "SESEQ", "ETCD",
                "ELEMENT", "SESTDTC", "SEENDTC", "SESTDY", "SEENDY",
            ])

        rows: list[dict[str, Any]] = []
        for seq, (_, tv_row) in enumerate(tv_df.iterrows(), start=1):
            rows.append({
                "STUDYID": studyid,
                "DOMAIN": "SE",
                "SESEQ": float(seq),
                "ETCD": "",
                "ELEMENT": tv_row["VISIT"][:40],
                "SESTDTC": "",
                "SEENDTC": "",
                "SESTDY": float(tv_row["VISITDY"]),
                "SEENDY": float(tv_row["VISITDY"]),
            })

        return pd.DataFrame(rows)

    def _build_domain_mapping(
        self, table: "RawSoaTable"
    ) -> pd.DataFrame:
        """Build assessment-to-SDTM domain mapping matrix.

        Each SoA row label is classified to its likely SDTM domain code,
        producing a matrix of (assessment, domain, visit_scheduled) tuples.
        """
        rows: list[dict[str, Any]] = []

        for activity_name, scheduled_flags in table.activities:
            name = activity_name.strip()
            if not name:
                continue

            domain_code, domain_name = _classify_assessment(name)

            # Build per-visit scheduled indicators
            visit_schedule: dict[str, bool] = {}
            for i, is_scheduled in enumerate(scheduled_flags):
                if i < len(table.visit_headers):
                    visit_schedule[table.visit_headers[i]] = is_scheduled

            category = _DOMAIN_TO_CATEGORY.get(domain_code, "operational")

            rows.append({
                "ASSESSMENT": name[:200],
                "SDTM_DOMAIN": domain_code,
                "DOMAIN_NAME": domain_name[:40],
                "CATEGORY": category,
                "CDASH_DOMAIN": domain_code,
                "SCHEDULED_VISITS": sum(scheduled_flags),
                "TOTAL_VISITS": len(table.visit_headers),
            })

        return pd.DataFrame(rows)

    def _build_soa_matrix(
        self, table: "RawSoaTable", tv_df: pd.DataFrame
    ) -> dict[tuple[int, str], SoaCellMetadata]:
        """Build cell-level SoA metadata matrix.

        For each (visit, assessment) pair where the activity is scheduled,
        creates a SoaCellMetadata entry with status, category, CDASH domain,
        and timing window derived from the TV domain.

        Args:
            table: Parsed SoA table.
            tv_df: Trial Visits DataFrame (for timing windows).

        Returns:
            Dict keyed by (visitnum, assessment_name) → SoaCellMetadata.
        """
        matrix: dict[tuple[int, str], SoaCellMetadata] = {}

        # Build visitnum → window lookup from TV
        visit_windows: dict[int, tuple[int, int]] = {}
        for _, row in tv_df.iterrows():
            vnum = int(row["VISITNUM"])
            visit_windows[vnum] = (
                int(row["TVSTRL"]),
                int(row["TVENRL"]),
            )

        for activity_name, scheduled_flags in table.activities:
            name = activity_name.strip()
            if not name:
                continue

            domain_code, _ = _classify_assessment(name)
            category = _DOMAIN_TO_CATEGORY.get(
                domain_code, "operational"
            )

            for i, is_scheduled in enumerate(scheduled_flags):
                if not is_scheduled:
                    continue
                if i >= len(table.visit_headers):
                    continue

                visitnum = i + 1
                window = visit_windows.get(visitnum, (0, 0))

                cell = SoaCellMetadata(
                    visitnum=visitnum,
                    assessment=name,
                    status="required",
                    condition="",
                    category=category,
                    cdash_domain=domain_code,
                    timing_window_days=window,
                )
                matrix[(visitnum, name)] = cell

        return matrix
