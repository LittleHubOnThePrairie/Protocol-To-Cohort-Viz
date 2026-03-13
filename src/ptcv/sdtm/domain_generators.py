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
    from ..ich_parser.template_assembler import QueryExtractionHit
    from ..soa_extractor.models import UsdmTimepoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(section: "IchSection") -> str:
    """Return text_excerpt from IchSection.content_json."""
    try:
        data = json.loads(section.content_json)
        return data.get("text_excerpt", "") or data.get("text", "") or ""
    except (json.JSONDecodeError, AttributeError):
        return ""


def _section_text(sections: list["IchSection"], code: str) -> str:
    """Extract text from the first section matching *code*."""
    for s in sections:
        if s.section_code == code:
            return _extract_text(s)
    return ""


def _extract_objective(text: str) -> str:
    """Extract primary objective statement from B.3 text."""
    if not text:
        return ""
    # Look for "primary objective" label
    m = re.search(
        r"(?:primary\s+)?objective[s]?\s*(?:is|are|:)\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")
    # Fallback: first meaningful sentence
    return _first_sentence(text)


def _extract_age_range(text: str) -> tuple[str, str]:
    """Extract minimum and maximum age from B.5 eligibility text.

    Returns:
        (age_min, age_max) as strings, empty if not found.
    """
    if not text:
        return "", ""
    age_min = ""
    age_max = ""
    # "Age 18-65 years" or "18 to 65 years"
    m = re.search(
        r"(?:age[d]?\s*)?(\d+)\s*(?:to|-)\s*(\d+)\s*years",
        text, re.IGNORECASE,
    )
    if m:
        age_min = m.group(1)
        age_max = m.group(2)
        return age_min, age_max
    # "≥ 18 years" or ">= 18 years" or "at least 18 years"
    m = re.search(
        r"(?:[≥>=]+\s*|at\s+least\s+)(\d+)\s*years", text, re.IGNORECASE,
    )
    if m:
        age_min = m.group(1)
    # "≤ 75 years" or "<= 75 years" or "no older than 75"
    m = re.search(
        r"(?:[≤<=]+\s*|no\s+older\s+than\s+)(\d+)\s*years",
        text, re.IGNORECASE,
    )
    if m:
        age_max = m.group(1)
    return age_min, age_max


def _extract_sample_size(text: str) -> str:
    """Extract planned number of subjects from B.10 or protocol text."""
    if not text:
        return ""
    m = re.search(
        r"(\d+)\s*(?:subjects|patients|participants|individuals)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)
    # "sample size of 200" or "enroll 200"
    m = re.search(
        r"(?:sample\s+size|enroll(?:ed|ment)?)\s*(?:of|:)?\s*(\d+)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return ""


def _extract_intervention_type(text: str) -> str:
    """Extract intervention type from B.2/B.7 text."""
    if not text:
        return ""
    for pattern, label in [
        (r"\bdrug\b|\btablet\b|\bcapsule\b|\bdose\b|\bmg\b", "DRUG"),
        (r"\bbiologic(?:al)?\b|\bvaccine\b|\bantibod", "BIOLOGICAL"),
        (r"\bdevice\b|\bimplant\b", "DEVICE"),
        (r"\bradiat(?:ion|otherapy)\b|\bcGy\b|\bGy\b", "RADIATION"),
        (r"\bdietary\b|\bsupplement\b", "DIETARY SUPPLEMENT"),
        (r"\bprocedur(?:e|al)\b|\bsurger", "PROCEDURE/SURGERY"),
    ]:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return ""


def _extract_stop_rule(text: str) -> str:
    """Extract study stop rule from B.6 text."""
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if re.search(
            r"stop|discontinu|terminat|halt",
            line, re.IGNORECASE,
        ) and len(line) > 20:
            return line[:200]
    return ""


def _extract_pharm_class(text: str) -> str:
    """Best-effort pharmacological class extraction from B.2/B.7 text."""
    if not text:
        return ""
    for pattern, label in [
        (r"ACE\s*inhibitor", "ACE Inhibitor"),
        (r"ARB|angiotensin.receptor.blocker", "ARB"),
        (r"beta.blocker|β.blocker", "Beta Blocker"),
        (r"calcium.channel.blocker", "Calcium Channel Blocker"),
        (r"diuretic", "Diuretic"),
        (r"statin|HMG.CoA", "HMG-CoA Reductase Inhibitor"),
        (r"SSRI|serotonin.reuptake", "SSRI"),
        (r"monoclonal.antibod", "Monoclonal Antibody"),
        (r"kinase.inhibitor|TKI", "Kinase Inhibitor"),
        (r"checkpoint.inhibitor|anti.PD|anti.CTLA", "Checkpoint Inhibitor"),
        (r"antineoplastic|chemotherap", "Antineoplastic"),
        (r"antibiotic|antimicrobial", "Antibiotic"),
        (r"antiviral", "Antiviral"),
        (r"analgesic|NSAID", "Analgesic"),
        (r"anticoagulant|heparin|warfarin", "Anticoagulant"),
    ]:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return ""


def _hits_for_section(
    hits: list["QueryExtractionHit"], parent_section: str,
) -> list["QueryExtractionHit"]:
    """Return hits belonging to *parent_section* (e.g. ``"B.1"``)."""
    return [h for h in hits if h.parent_section == parent_section]


def _hit_by_query_id(
    hits: list["QueryExtractionHit"], query_id: str,
) -> str:
    """Return extracted_content for the first hit matching *query_id*."""
    for h in hits:
        if h.query_id == query_id and h.extracted_content:
            return h.extracted_content
    return ""


def _hits_text(
    hits: list["QueryExtractionHit"],
) -> str:
    """Concatenate extracted_content from all hits (fallback helper)."""
    parts = [h.extracted_content for h in hits if h.extracted_content]
    return "\n\n".join(parts)


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

    Extracts study-level parameters using a SoA-first, protocol-second
    strategy (PTCV-181). Structured SoA data and pipeline context are
    used first; protocol section text is parsed secondarily via regex.
    CT-unmapped values are collected and returned for caller routing.

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
        *,
        timepoints: list["UsdmTimepoint"] | None = None,
        registry_id: str = "",
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build the TS domain DataFrame.

        Uses a two-tier extraction strategy (PTCV-181):
          Tier 1 — SoA structured data + pipeline context
          Tier 2 — Protocol section text (regex extraction)

        Args:
            sections: All IchSection records for the protocol.
            studyid: STUDYID for TS.STUDYID.
            run_id: Pipeline run UUID4.
            timepoints: Optional USDM timepoints from SoA extraction.
            registry_id: Trial registry identifier (e.g. "NCT00112827").

        Returns:
            Tuple of (DataFrame, list of unmapped CtLookupResult).
        [PTCV-22 Scenario: Generate TS domain and write to WORM]
        [PTCV-181 Scenario: SoA-first data sourcing for TCG v5.9]
        """
        # Extract text from all relevant ICH sections
        text_b1 = _section_text(sections, "B.1")
        text_b2 = _section_text(sections, "B.2")
        text_b3 = _section_text(sections, "B.3")
        text_b4 = _section_text(sections, "B.4")
        text_b5 = _section_text(sections, "B.5")
        text_b6 = _section_text(sections, "B.6")
        text_b7 = _section_text(sections, "B.7")
        text_b10 = _section_text(sections, "B.10")
        all_text = " ".join(
            [text_b1, text_b2, text_b3, text_b4, text_b5,
             text_b6, text_b7, text_b10]
        )

        unmapped: list[CtLookupResult] = []

        rows: list[dict[str, Any]] = []
        seq = 0

        def _add(
            parmcd: str,
            parm: str,
            value: str,
            normalize_parmcd: str = "",
            tsvalnf: str = "",
        ) -> None:
            nonlocal seq
            if not value and not tsvalnf:
                return
            seq += 1
            tsvalcd = ""
            if value and normalize_parmcd:
                ct_result = self._ct.normalize(normalize_parmcd, value)
                tsvalcd = ct_result.tsvalcd
                if not ct_result.mapped:
                    unmapped.append(ct_result)
            rows.append(
                {
                    "STUDYID": studyid,
                    "DOMAIN": "TS",
                    "TSSEQ": float(seq),
                    "TSPARMCD": parmcd[:8],
                    "TSPARM": parm[:40],
                    "TSVAL": (value or "")[:200],
                    "TSVALCD": tsvalcd[:20],
                    "TSVALNF": tsvalnf,
                    "TSVALUNIT": "",
                }
            )

        # --- Existing parameters (PTCV-22) --------------------------------

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
        stype_val = ""
        for pattern, label in [
            (r"parallel", "PARALLEL"),
            (r"crossover", "CROSSOVER"),
            (r"factorial", "FACTORIAL"),
            (r"single.group|single.arm", "SINGLE GROUP"),
        ]:
            if re.search(pattern, all_text, re.IGNORECASE):
                stype_val = label
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

        # --- New parameters (PTCV-181) ------------------------------------

        # Tier 1: SoA + context
        if registry_id:
            _add("REGID", "Registry Identifier", registry_id)
        _add("SDTMVR", "SDTM Version", "1.7")

        # NARMS — count arms from B.4 text
        arms = _extract_arms(text_b4) if text_b4 else ["Treatment"]
        _add("NARMS", "Planned Number of Arms", str(len(arms)))

        # INTMODEL — map from study type
        _INTMODEL_MAP = {
            "PARALLEL": "PARALLEL GROUP",
            "CROSSOVER": "CROSSOVER",
            "FACTORIAL": "FACTORIAL ASSIGNMENT",
            "SINGLE GROUP": "SINGLE GROUP ASSIGNMENT",
        }
        intmodel = _INTMODEL_MAP.get(stype_val, "")
        if intmodel:
            _add("INTMODEL", "Intervention Model", intmodel)

        # Tier 2: Protocol section text
        # RANDOM — randomization indicator
        random_val = "Y" if re.search(
            r"randomi[sz]", all_text, re.IGNORECASE
        ) else "N"
        _add("RANDOM", "Trial Randomization", random_val)

        # ADDON — add-on to existing treatments
        addon_val = "Y" if re.search(
            r"add.on|adjunct", all_text, re.IGNORECASE
        ) else "N"
        _add("ADDON", "Add-on to Existing Treatments", addon_val)

        # OBJPRIM — primary objective from B.3
        objprim = _extract_objective(text_b3)
        if objprim:
            _add("OBJPRIM", "Primary Objective", objprim[:200])

        # AGEMIN / AGEMAX — from B.5 eligibility criteria
        age_min, age_max = _extract_age_range(text_b5)
        if age_min:
            _add("AGEMIN", "Planned Min Age of Subjects", age_min)
        if age_max:
            _add("AGEMAX", "Planned Max Age of Subjects", age_max)

        # PLANSUB — planned number of subjects from B.10
        plansub = _extract_sample_size(text_b10 or text_b5 or all_text)
        if plansub:
            _add("PLANSUB", "Planned Number of Subjects", plansub)

        # INTTYPE — intervention type from B.2/B.7
        inttype = _extract_intervention_type(text_b7 or text_b2 or all_text)
        if inttype:
            _add("INTTYPE", "Intervention Type", inttype)

        # STOPRULE — study stop rules from B.6
        stoprule = _extract_stop_rule(text_b6)
        if stoprule:
            _add("STOPRULE", "Study Stop Rules", stoprule[:200])

        # PCLAS — pharmacological class (best-effort)
        pclas = _extract_pharm_class(text_b2 or text_b7 or all_text)
        if pclas:
            _add("PCLAS", "Pharmacological Class", pclas[:80])

        # Tier 3: Not available from protocol — emit with TSVALNF
        _add("ACTSUB", "Actual Number of Subjects", "",
             tsvalnf="NOT APPLICABLE")
        _add("DCUTDT", "Data Cutoff Date", "",
             tsvalnf="NOT APPLICABLE")
        _add("DCUTDESC", "Data Cutoff Description", "",
             tsvalnf="NOT APPLICABLE")

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

    def generate_from_hits(
        self,
        hits: list["QueryExtractionHit"],
        studyid: str,
        run_id: str,
        *,
        timepoints: list["UsdmTimepoint"] | None = None,
        registry_id: str = "",
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build TS domain from query pipeline hits (PTCV-140).

        Maps query_ids directly to SDTM parameters where possible,
        falling back to regex parsing for unmapped fields.
        Includes PTCV-181 new parameters.
        """
        b1_hits = _hits_for_section(hits, "B.1")
        b3_hits = _hits_for_section(hits, "B.3")
        b4_hits = _hits_for_section(hits, "B.4")
        b5_hits = _hits_for_section(hits, "B.5")
        b6_hits = _hits_for_section(hits, "B.6")
        b7_hits = _hits_for_section(hits, "B.7")
        b10_hits = _hits_for_section(hits, "B.10")
        text_b1 = _hits_text(b1_hits)
        text_b3 = _hits_text(b3_hits)
        text_b4 = _hits_text(b4_hits)
        text_b5 = _hits_text(b5_hits)
        text_b6 = _hits_text(b6_hits)
        text_b7 = _hits_text(b7_hits)
        text_b10 = _hits_text(b10_hits)
        all_text = " ".join(
            [text_b1, text_b3, text_b4, text_b5, text_b6, text_b7, text_b10]
        )

        unmapped: list[CtLookupResult] = []
        rows: list[dict[str, Any]] = []
        seq = 0

        def _add(
            parmcd: str,
            parm: str,
            value: str,
            normalize_parmcd: str = "",
            tsvalnf: str = "",
        ) -> None:
            nonlocal seq
            if not value and not tsvalnf:
                return
            seq += 1
            tsvalcd = ""
            if value and normalize_parmcd:
                ct_result = self._ct.normalize(normalize_parmcd, value)
                tsvalcd = ct_result.tsvalcd
                if not ct_result.mapped:
                    unmapped.append(ct_result)
            rows.append(
                {
                    "STUDYID": studyid,
                    "DOMAIN": "TS",
                    "TSSEQ": float(seq),
                    "TSPARMCD": parmcd[:8],
                    "TSPARM": parm[:40],
                    "TSVAL": (value or "")[:200],
                    "TSVALCD": tsvalcd[:20],
                    "TSVALNF": tsvalnf,
                    "TSVALUNIT": "",
                }
            )

        # TITLE — direct from query B.1.1.q1
        title = _hit_by_query_id(hits, "B.1.1.q1")
        if not title:
            title = _first_sentence(text_b1)
        if not title:
            title = f"Protocol {studyid}"
        _add("TITLE", "Trial Title", title[:200])

        # PCNTID — protocol number from B.1.1.q2
        pcntid = _hit_by_query_id(hits, "B.1.1.q2")
        if pcntid:
            _add("PCNTID", "Protocol/Study Number", pcntid[:200])

        # PHASE — from B.1 or B.4 text
        phase_result = self._ct.normalize_phase(all_text)
        if phase_result.tsval:
            _add("PHASE", "Trial Phase", phase_result.tsval, "PHASE")

        # TTYPE
        ttype_val = "INTERVENTIONAL"
        if re.search(r"observational", all_text, re.IGNORECASE):
            ttype_val = "OBSERVATIONAL"
        _add("TTYPE", "Trial Type", ttype_val, "TTYPE")

        # STYPE — study design from B.4.1.q1 or regex
        stype_val = ""
        design_text = _hit_by_query_id(hits, "B.4.1.q1") or all_text
        for pattern, label in [
            (r"parallel", "PARALLEL"),
            (r"crossover", "CROSSOVER"),
            (r"factorial", "FACTORIAL"),
            (r"single.group|single.arm", "SINGLE GROUP"),
        ]:
            if re.search(pattern, design_text, re.IGNORECASE):
                stype_val = label
                _add("STYPE", "Study Type", label, "STYPE")
                break

        # BLIND — from B.4.2.q1 or regex
        blind_text = _hit_by_query_id(hits, "B.4.2.q1") or all_text
        for pattern, label in [
            (r"double.blind", "DOUBLE-BLIND"),
            (r"single.blind", "SINGLE-BLIND"),
            (r"open.label", "OPEN-LABEL"),
        ]:
            if re.search(pattern, blind_text, re.IGNORECASE):
                _add("BLIND", "Blinding Schema", label, "BLIND")
                break

        # SPONSOR — direct from B.1.2.q1
        sponsor = _hit_by_query_id(hits, "B.1.2.q1")
        if not sponsor:
            sponsor_m = re.search(
                r"sponsor[:\s]+([A-Za-z][\w\s,]+?)(?:\n|\.)",
                text_b1, re.IGNORECASE,
            )
            if sponsor_m:
                sponsor = sponsor_m.group(1).strip()
        if sponsor:
            _add("SPONSOR", "Clinical Study Sponsor", sponsor[:80])

        # INDIC — indication from B.1
        indic_m = re.search(
            r"(?:indication|disease|condition)[:\s]+([A-Za-z][\w\s]+?)(?:\n|;|\.)",
            text_b1, re.IGNORECASE,
        )
        if indic_m:
            _add("INDIC", "Studied Indication", indic_m.group(1).strip()[:80])

        # --- New parameters (PTCV-181) ------------------------------------

        # Tier 1: SoA + context
        if registry_id:
            _add("REGID", "Registry Identifier", registry_id)
        _add("SDTMVR", "SDTM Version", "1.7")

        # NARMS — from B.7 treatment arms or B.4
        treat_text = _hit_by_query_id(hits, "B.7.1.q1") or text_b4
        arms = _extract_arms(treat_text) if treat_text else ["Treatment"]
        _add("NARMS", "Planned Number of Arms", str(len(arms)))

        # INTMODEL — map from study type
        _INTMODEL_MAP = {
            "PARALLEL": "PARALLEL GROUP",
            "CROSSOVER": "CROSSOVER",
            "FACTORIAL": "FACTORIAL ASSIGNMENT",
            "SINGLE GROUP": "SINGLE GROUP ASSIGNMENT",
        }
        intmodel = _INTMODEL_MAP.get(stype_val, "")
        if intmodel:
            _add("INTMODEL", "Intervention Model", intmodel)

        # Tier 2: Protocol section text
        # RANDOM — randomization indicator
        random_val = "Y" if re.search(
            r"randomi[sz]", all_text, re.IGNORECASE
        ) else "N"
        _add("RANDOM", "Trial Randomization", random_val)

        # ADDON — add-on to existing treatments
        addon_val = "Y" if re.search(
            r"add.on|adjunct", all_text, re.IGNORECASE
        ) else "N"
        _add("ADDON", "Add-on to Existing Treatments", addon_val)

        # OBJPRIM — from B.3.q1 or B.3 text
        objprim_text = _hit_by_query_id(hits, "B.3.q1") or text_b3
        objprim = _extract_objective(objprim_text)
        if objprim:
            _add("OBJPRIM", "Primary Objective", objprim[:200])

        # AGEMIN / AGEMAX — from B.5 hits
        elig_text = (
            _hit_by_query_id(hits, "B.5.1.q1")
            + " " + _hit_by_query_id(hits, "B.5.2.q1")
            + " " + text_b5
        )
        age_min, age_max = _extract_age_range(elig_text)
        if age_min:
            _add("AGEMIN", "Planned Min Age of Subjects", age_min)
        if age_max:
            _add("AGEMAX", "Planned Max Age of Subjects", age_max)

        # PLANSUB — from B.10 hits
        plansub_text = (
            _hit_by_query_id(hits, "B.10.1.q1") or text_b10 or all_text
        )
        plansub = _extract_sample_size(plansub_text)
        if plansub:
            _add("PLANSUB", "Planned Number of Subjects", plansub)

        # INTTYPE — from B.7 treatment hits
        inttype_text = (
            _hit_by_query_id(hits, "B.7.1.q1") or text_b7 or all_text
        )
        inttype = _extract_intervention_type(inttype_text)
        if inttype:
            _add("INTTYPE", "Intervention Type", inttype)

        # STOPRULE — from B.6 hits
        stoprule = _extract_stop_rule(text_b6)
        if stoprule:
            _add("STOPRULE", "Study Stop Rules", stoprule[:200])

        # PCLAS — pharmacological class (best-effort)
        pclas = _extract_pharm_class(text_b7 or all_text)
        if pclas:
            _add("PCLAS", "Pharmacological Class", pclas[:80])

        # Tier 3: Not available from protocol — emit with TSVALNF
        _add("ACTSUB", "Actual Number of Subjects", "",
             tsvalnf="NOT APPLICABLE")
        _add("DCUTDT", "Data Cutoff Date", "",
             tsvalnf="NOT APPLICABLE")
        _add("DCUTDESC", "Data Cutoff Description", "",
             tsvalnf="NOT APPLICABLE")

        if not rows:
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

    def generate_from_hits(
        self,
        hits: list["QueryExtractionHit"],
        studyid: str,
    ) -> pd.DataFrame:
        """Build TA domain from query pipeline hits (PTCV-140).

        Uses B.7.1.q1 (treatments) for arm names when available,
        falling back to regex extraction from concatenated B.4 text.
        """
        b4_hits = _hits_for_section(hits, "B.4")
        b7_hits = _hits_for_section(hits, "B.7")
        text_b4 = _hits_text(b4_hits)

        # Try B.7.1.q1 for arm names; fall back to B.4 text
        treatments = _hit_by_query_id(hits, "B.7.1.q1")
        if treatments:
            arms = _extract_arms(treatments)
        else:
            arms = _extract_arms(text_b4)

        # Also try B.7 text for additional arm names
        if not arms and b7_hits:
            arms = _extract_arms(_hits_text(b7_hits))

        elements = _extract_elements(text_b4)

        rows: list[dict[str, Any]] = []
        for arm_idx, arm_label in enumerate(arms):
            armcd = f"ARM{arm_idx + 1:02d}"
            for et_idx, (etcd, element) in enumerate(elements):
                epoch_map = {
                    "SCRN": "Screening",
                    "TRT": "Treatment",
                    "FUP": "Follow-up",
                }
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

    def generate_from_hits(
        self,
        hits: list["QueryExtractionHit"],
        studyid: str,
    ) -> pd.DataFrame:
        """Build TE domain from query pipeline hits (PTCV-140)."""
        b4_hits = _hits_for_section(hits, "B.4")
        text = _hits_text(b4_hits)
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

    def generate_from_hits(
        self,
        hits: list["QueryExtractionHit"],
        studyid: str,
    ) -> pd.DataFrame:
        """Build TI domain from query pipeline hits (PTCV-140).

        Uses B.5.1.q1 (inclusion) and B.5.2.q1 (exclusion) query IDs
        to assign IECAT directly, eliminating regex category guessing.
        """
        inclusion_text = _hit_by_query_id(hits, "B.5.1.q1")
        exclusion_text = _hit_by_query_id(hits, "B.5.2.q1")

        rows: list[dict[str, Any]] = []

        def _parse_items(
            text: str, cat: str, start_idx: int,
        ) -> int:
            """Parse numbered list items and append to rows."""
            idx = start_idx
            for line in text.splitlines():
                stripped = line.strip()
                m = re.match(r"^\d+[\.\)]\s+(.+)", stripped)
                if m:
                    criterion = m.group(1).strip()[:200]
                    idx += 1
                    suffix = "I" if cat == "INCLUSION" else "E"
                    rows.append(
                        {
                            "STUDYID": studyid,
                            "DOMAIN": "TI",
                            "IETESTCD": f"IE{idx:03d}{suffix}"[:8],
                            "IETEST": criterion,
                            "IECAT": cat,
                            "IESCAT": "",
                            "TIRL": "",
                            "TIVERS": "1",
                        }
                    )
            return idx

        _parse_items(inclusion_text, "INCLUSION", 0)
        _parse_items(exclusion_text, "EXCLUSION", 0)

        # Fallback: if no structured hits, use concatenated B.5 text
        if not rows:
            b5_hits = _hits_for_section(hits, "B.5")
            text = _hits_text(b5_hits)
            criteria = _extract_criteria(text)
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
