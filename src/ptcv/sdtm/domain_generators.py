"""CDISC SDTM Trial Design Domain generators.

Produces five SDTM trial design datasets as pandas DataFrames:
  TS — Trial Summary
  TA — Trial Arms
  TE — Trial Elements
  TV — Trial Visits (derived from USDM UsdmTimepoint records)
  TI — Trial Inclusion/Exclusion Criteria

ICH section content_json is parsed with regex to extract structured values;
unmapped CT values are returned for caller routing to the CT review queue.

Risk tier: MEDIUM — regulatory submission artefacts; no patient data.

Regulatory references:
- SDTMIG v3.3 Section 7.4: Trial Design Domains
- CDISC SDTM v1.7 for variable attributes
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from .ct_normalizer import CtNormalizer, CtLookupResult

if TYPE_CHECKING:
    from ..ich_parser.models import IchSection
    from ..soa_extractor.models import UsdmTimepoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(section: "IchSection") -> str:
    """Return text_excerpt from IchSection.content_json."""
    try:
        data = json.loads(section.content_json)
        return data.get("text_excerpt", "") or ""
    except (json.JSONDecodeError, AttributeError):
        return ""


def _first_sentence(text: str) -> str:
    """Return the first non-empty sentence from text (max 200 chars)."""
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 10:
            return line[:200]
    return text[:200].strip()


def _extract_arms(text: str) -> list[str]:
    """Extract treatment arm names from B.4 section text."""
    arms: list[str] = []
    patterns = [
        r"(?:arm|group|cohort|treatment)\s+([A-Z]\b)",        # Arm A, Group B
        r"(?:arm|group|cohort)\s+(\d+)",                       # Arm 1, Group 2
        r"(treatment\s+arm\s+[A-Za-z0-9]+)",                  # Treatment arm X
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            arm = m.group(0).strip().title()
            if arm not in arms:
                arms.append(arm)
    # Fallback: at least one arm
    if not arms:
        arms = ["Treatment"]
    return arms[:6]  # cap at 6 arms


def _extract_elements(text: str) -> list[tuple[str, str]]:
    """Return list of (ETCD, ELEMENT) tuples from B.4 text."""
    elements: list[tuple[str, str]] = []

    # Standard epoch names
    epoch_patterns = [
        ("SCRN", "Screening", r"screen(?:ing)?"),
        ("RUN", "Run-in", r"run.in|lead.in"),
        ("RAND", "Randomization", r"randomi[sz]"),
        ("TRT", "Treatment", r"treatment|dosing"),
        ("FUP", "Follow-up", r"follow.?up"),
        ("EOS", "End of Study", r"end.of.study"),
    ]
    for etcd, element, pattern in epoch_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            elements.append((etcd, element))

    if not elements:
        elements = [("SCRN", "Screening"), ("TRT", "Treatment"), ("FUP", "Follow-up")]
    return elements


def _extract_criteria(text: str) -> list[tuple[str, str, str]]:
    """Return list of (ietestcd, ietest, iecat) from B.5 text.

    Args:
        text: ICH B.5 section text excerpt.

    Returns:
        List of (IETESTCD, IETEST, IECAT) tuples.
    """
    criteria: list[tuple[str, str, str]] = []
    current_cat = "INCLUSION"

    # Split into numbered list items
    item_re = re.compile(r"^\s*\d+[\.\)]\s+(.+)", re.MULTILINE)
    items = item_re.findall(text)

    inc_idx = 0
    exc_idx = 0
    for line in text.splitlines():
        stripped = line.strip()
        # Detect category switch
        if re.search(r"exclu?sion", stripped, re.IGNORECASE):
            current_cat = "EXCLUSION"
        elif re.search(r"inclu?sion", stripped, re.IGNORECASE):
            current_cat = "INCLUSION"
        # Detect numbered criteria
        m = re.match(r"^\d+[\.\)]\s+(.+)", stripped)
        if m:
            criterion_text = m.group(1).strip()[:200]
            if current_cat == "INCLUSION":
                inc_idx += 1
                testcd = f"IE{inc_idx:03d}I"
            else:
                exc_idx += 1
                testcd = f"IE{exc_idx:03d}E"
            criteria.append((testcd, criterion_text, current_cat))

    # Fallback
    if not criteria:
        criteria = [
            ("IE001I", "Subject meets inclusion criteria", "INCLUSION"),
            ("IE001E", "Subject meets exclusion criteria", "EXCLUSION"),
        ]

    return criteria[:20]  # cap at 20 criteria


# ---------------------------------------------------------------------------
# TS Generator
# ---------------------------------------------------------------------------

class TsGenerator:
    """Generates the TS (Trial Summary) domain DataFrame.

    Extracts study-level parameters from the B.1 ICH section and applies
    CDISC CT normalization. CT-unmapped values are collected and returned
    for caller routing.

    Args:
        ct_normalizer: CtNormalizer instance to use.
    """

    def __init__(self, ct_normalizer: CtNormalizer | None = None) -> None:
        self._ct = ct_normalizer or CtNormalizer()

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
        run_id: str,
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build the TS domain DataFrame.

        Args:
            sections: All IchSection records for the protocol.
            studyid: STUDYID for TS.STUDYID.
            run_id: Pipeline run UUID4.

        Returns:
            Tuple of (DataFrame, list of unmapped CtLookupResult).
        [PTCV-22 Scenario: Generate TS domain and write to WORM]
        """
        b1_sections = [s for s in sections if s.section_code == "B.1"]
        b4_sections = [s for s in sections if s.section_code == "B.4"]
        text_b1 = _extract_text(b1_sections[0]) if b1_sections else ""
        text_b4 = _extract_text(b4_sections[0]) if b4_sections else ""
        all_text = text_b1 + " " + text_b4

        unmapped: list[CtLookupResult] = []

        rows: list[dict[str, Any]] = []
        seq = 0

        def _add(
            parmcd: str, parm: str, value: str, normalize_parmcd: str = ""
        ) -> None:
            nonlocal seq
            if not value:
                return
            seq += 1
            if normalize_parmcd:
                ct_result = self._ct.normalize(normalize_parmcd, value)
                tsvalcd = ct_result.tsvalcd
                if not ct_result.mapped:
                    unmapped.append(ct_result)
            else:
                tsvalcd = ""
            rows.append(
                {
                    "STUDYID": studyid,
                    "DOMAIN": "TS",
                    "TSSEQ": float(seq),
                    "TSPARMCD": parmcd[:8],
                    "TSPARM": parm[:40],
                    "TSVAL": value[:200],
                    "TSVALCD": tsvalcd[:20],
                    "TSVALNF": "",
                    "TSVALUNIT": "",
                }
            )

        # TITLE
        title = _first_sentence(text_b1)
        if not title:
            title = f"Protocol {studyid}"
        _add("TITLE", "Trial Title", title[:200])

        # PHASE — from B.1 or B.4 text
        phase_result = self._ct.normalize_phase(all_text)
        if phase_result.tsval:
            _add("PHASE", "Trial Phase", phase_result.tsval, "PHASE")

        # TTYPE — trial type
        ttype_val = "INTERVENTIONAL"
        if re.search(r"observational", all_text, re.IGNORECASE):
            ttype_val = "OBSERVATIONAL"
        _add("TTYPE", "Trial Type", ttype_val, "TTYPE")

        # STYPE — study design (parallel/crossover)
        for pattern, label in [
            (r"parallel", "PARALLEL"),
            (r"crossover", "CROSSOVER"),
            (r"factorial", "FACTORIAL"),
            (r"single.group|single.arm", "SINGLE GROUP"),
        ]:
            if re.search(pattern, all_text, re.IGNORECASE):
                _add("STYPE", "Study Type", label, "STYPE")
                break

        # BLIND — blinding
        for pattern, label in [
            (r"double.blind", "DOUBLE-BLIND"),
            (r"single.blind", "SINGLE-BLIND"),
            (r"open.label", "OPEN-LABEL"),
        ]:
            if re.search(pattern, all_text, re.IGNORECASE):
                _add("BLIND", "Blinding Schema", label, "BLIND")
                break

        # SPONSOR — extract from B.1 if present
        sponsor_m = re.search(
            r"sponsor[:\s]+([A-Za-z][\w\s,]+?)(?:\n|\.)", text_b1, re.IGNORECASE
        )
        if sponsor_m:
            _add("SPONSOR", "Clinical Study Sponsor", sponsor_m.group(1).strip()[:80])

        # INDIC — indication from B.1
        indic_m = re.search(
            r"(?:indication|disease|condition)[:\s]+([A-Za-z][\w\s]+?)(?:\n|;|\.)",
            text_b1,
            re.IGNORECASE,
        )
        if indic_m:
            _add("INDIC", "Studied Indication", indic_m.group(1).strip()[:80])

        if not rows:
            # Minimal fallback — always produce at least a TITLE row
            seq += 1
            rows.append(
                {
                    "STUDYID": studyid,
                    "DOMAIN": "TS",
                    "TSSEQ": float(seq),
                    "TSPARMCD": "TITLE",
                    "TSPARM": "Trial Title",
                    "TSVAL": f"Protocol {studyid}",
                    "TSVALCD": "",
                    "TSVALNF": "",
                    "TSVALUNIT": "",
                }
            )

        return pd.DataFrame(rows), unmapped


# ---------------------------------------------------------------------------
# TA Generator
# ---------------------------------------------------------------------------

class TaGenerator:
    """Generates the TA (Trial Arms) domain DataFrame.

    Extracts arm and element structure from ICH B.4 section text.
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
    ) -> pd.DataFrame:
        """Build the TA domain DataFrame.

        Args:
            sections: All IchSection records for the protocol.
            studyid: STUDYID value.

        Returns:
            TA domain DataFrame.
        """
        b4 = [s for s in sections if s.section_code == "B.4"]
        text = _extract_text(b4[0]) if b4 else ""

        arms = _extract_arms(text)
        elements = _extract_elements(text)

        rows: list[dict[str, Any]] = []
        for arm_idx, arm_label in enumerate(arms):
            armcd = f"ARM{arm_idx + 1:02d}"
            for et_idx, (etcd, element) in enumerate(elements):
                epoch_map = {"SCRN": "Screening", "TRT": "Treatment", "FUP": "Follow-up"}
                epoch = epoch_map.get(etcd, element)
                rows.append(
                    {
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
                    }
                )

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TE Generator
# ---------------------------------------------------------------------------

class TeGenerator:
    """Generates the TE (Trial Elements) domain DataFrame.

    Extracts element definitions from ICH B.4 section text.
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
    ) -> pd.DataFrame:
        """Build the TE domain DataFrame.

        Args:
            sections: All IchSection records for the protocol.
            studyid: STUDYID value.

        Returns:
            TE domain DataFrame.
        """
        b4 = [s for s in sections if s.section_code == "B.4"]
        text = _extract_text(b4[0]) if b4 else ""
        elements = _extract_elements(text)

        rows: list[dict[str, Any]] = []
        for etcd, element in elements:
            rows.append(
                {
                    "STUDYID": studyid,
                    "DOMAIN": "TE",
                    "ETCD": etcd[:8],
                    "ELEMENT": element[:40],
                    "TESTRL": f"VISIT='{element[:20]}'",
                    "TEENRL": "",
                    "TEDUR": "",
                }
            )

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TV Generator
# ---------------------------------------------------------------------------

class TvGenerator:
    """Generates the TV (Trial Visits) domain DataFrame.

    Derives all rows from USDM UsdmTimepoint records (PTCV-21 output).
    No ICH section parsing is needed for TV — all timing information is
    already normalised in the UsdmTimepoint model.
    """

    def generate(
        self,
        timepoints: list["UsdmTimepoint"],
        studyid: str,
    ) -> pd.DataFrame:
        """Build the TV domain DataFrame.

        Args:
            timepoints: UsdmTimepoint records from PTCV-21 SoA extractor.
            studyid: STUDYID value.

        Returns:
            TV domain DataFrame. One row per timepoint, sorted by day_offset.
        [PTCV-22 Scenario: Generate TV domain from USDM Timepoints]
        """
        sorted_tp = sorted(timepoints, key=lambda t: t.day_offset)
        rows: list[dict[str, Any]] = []

        for visitnum, tp in enumerate(sorted_tp, start=1):
            rows.append(
                {
                    "STUDYID": studyid,
                    "DOMAIN": "TV",
                    "VISITNUM": float(visitnum),
                    "VISIT": tp.visit_name[:40],
                    "VISITDY": float(tp.day_offset),
                    "TVSTRL": float(-tp.window_early),   # days before → negative
                    "TVENRL": float(tp.window_late),
                    "TVTIMY": "",
                    "TVTIM": "",
                    "TVENDY": "",
                }
            )

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TI Generator
# ---------------------------------------------------------------------------

class TiGenerator:
    """Generates the TI (Trial Inclusion/Exclusion) domain DataFrame.

    Parses numbered list items from the ICH B.5 section text.
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
    ) -> pd.DataFrame:
        """Build the TI domain DataFrame.

        Args:
            sections: All IchSection records for the protocol.
            studyid: STUDYID value.

        Returns:
            TI domain DataFrame. One row per inclusion/exclusion criterion.
        """
        b5 = [s for s in sections if s.section_code == "B.5"]
        text = _extract_text(b5[0]) if b5 else ""
        criteria = _extract_criteria(text)

        rows: list[dict[str, Any]] = []
        for ietestcd, ietest, iecat in criteria:
            rows.append(
                {
                    "STUDYID": studyid,
                    "DOMAIN": "TI",
                    "IETESTCD": ietestcd[:8],
                    "IETEST": ietest[:200],
                    "IECAT": iecat[:20],
                    "IESCAT": "",
                    "TIRL": "",
                    "TIVERS": "1",
                }
            )

        return pd.DataFrame(rows)
