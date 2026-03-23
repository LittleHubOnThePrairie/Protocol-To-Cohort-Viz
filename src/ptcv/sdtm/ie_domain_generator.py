"""IE (Inclusion/Exclusion) SDTM Domain Generator.

PTCV-247: Generates the IE domain from protocol B.5 content, mapping
each inclusion/exclusion criterion to its implied data collection
variable.  Unlike TI (which stores the criteria text), IE specifies
*what data must be collected* to verify each criterion at screening.

Example:
    Criterion: "Age >= 18 years"
    → IETESTCD: "AGE_CHK", IETEST: "Age Verification",
      IEORRES: ">= 18 years", IECAT: "INCLUSION"

Risk tier: MEDIUM — regulatory submission artefact; no patient data.

Usage::

    from ptcv.sdtm.ie_domain_generator import IeSpecGenerator

    gen = IeSpecGenerator()
    df, unmapped = gen.generate(sections, "STUDY01", "run-1")
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from .ct_normalizer import CtLookupResult

if TYPE_CHECKING:
    from ..ich_parser.models import IchSection


# ---------------------------------------------------------------------------
# Criterion → Data Collection Variable Mapping
# ---------------------------------------------------------------------------

# Each pattern: (regex, ietestcd, ietest_label, ieorres_template)
# ieorres_template uses {match} placeholder for captured value.
_CRITERION_PATTERNS: list[tuple[str, str, str, str]] = [
    # Age criteria
    (
        r"(?:age[d]?\s*)?([≥>=]+\s*\d+)\s*years",
        "AGE_CHK",
        "Age Verification",
        "{match} years",
    ),
    (
        r"age[d]?\s*(\d+)\s*(?:to|-)\s*(\d+)\s*years",
        "AGE_CHK",
        "Age Verification",
        "{g1} to {g2} years",
    ),
    (
        r"at\s+least\s+(\d+)\s*years\s*(?:of\s*age|old)?",
        "AGE_CHK",
        "Age Verification",
        ">= {match} years",
    ),
    # Sex/Gender
    (
        r"\b(male|female)\s+(?:and\s+(?:male|female)\s+)?(?:subjects?|patients?|participants?)",
        "SEX_CHK",
        "Sex Verification",
        "{match}",
    ),
    # Body weight / BMI
    (
        r"(?:body\s*)?(?:mass\s*index|BMI)\s*(?:[≥>=<≤]+\s*[\d.]+)",
        "BMI_CHK",
        "BMI Verification",
        "{match}",
    ),
    (
        r"body\s*weight\s*(?:[≥>=<≤]+\s*\d+\s*kg)",
        "BW_CHK",
        "Body Weight Verification",
        "{match}",
    ),
    # Diagnosis / disease
    (
        r"(?:confirmed|documented|established)\s+(?:diagnosis\s+of\s+)?(.{10,80}?)(?:\.|$)",
        "DX_CHK",
        "Diagnosis Verification",
        "Confirmed: {match}",
    ),
    (
        r"diagnosed\s+with\s+(.{10,80}?)(?:\.|,|$)",
        "DX_CHK",
        "Diagnosis Verification",
        "Diagnosed with {match}",
    ),
    # Lab values
    (
        r"(?:hemoglobin|hgb|hb)\s*(?:[≥>=<≤]+\s*[\d.]+\s*g/[dL]+)",
        "HGB_CHK",
        "Hemoglobin Verification",
        "{match}",
    ),
    (
        r"(?:creatinine|CrCl|eGFR)\s*(?:[≥>=<≤]+\s*[\d.]+)",
        "RENAL_CHK",
        "Renal Function Verification",
        "{match}",
    ),
    (
        r"(?:ALT|AST|bilirubin)\s*(?:[≤<=]+\s*[\d.]+\s*[xX×]\s*ULN)",
        "LIVER_CHK",
        "Hepatic Function Verification",
        "{match}",
    ),
    (
        r"(?:platelet|PLT)\s*(?:count\s*)?(?:[≥>=]+\s*[\d,]+)",
        "PLT_CHK",
        "Platelet Count Verification",
        "{match}",
    ),
    (
        r"(?:ANC|absolute\s+neutrophil)\s*(?:count\s*)?(?:[≥>=]+\s*[\d.]+)",
        "ANC_CHK",
        "ANC Verification",
        "{match}",
    ),
    # ECOG / Performance status
    (
        r"ECOG\s*(?:performance\s*status\s*)?(?:[≤<=]+\s*\d|of\s*\d)",
        "ECOG_CHK",
        "ECOG Performance Status Verification",
        "{match}",
    ),
    # Informed consent
    (
        r"(?:signed|written|informed)\s+(?:informed\s+)?consent",
        "ICF_CHK",
        "Informed Consent Verification",
        "Signed informed consent obtained",
    ),
    # Pregnancy / contraception
    (
        r"(?:negative\s+)?pregnancy\s+test",
        "PREG_CHK",
        "Pregnancy Test Verification",
        "Negative pregnancy test required",
    ),
    (
        r"(?:adequate|acceptable)\s+(?:method\s+of\s+)?contraception",
        "CONTRA_CHK",
        "Contraception Verification",
        "Adequate contraception required",
    ),
    # Prior therapy
    (
        r"(?:prior|previous)\s+(?:treatment|therapy|regimen)\s+(?:with\s+)?(.{5,60}?)(?:\.|,|$)",
        "PRIOR_CHK",
        "Prior Therapy Verification",
        "Prior therapy: {match}",
    ),
    # Washout period
    (
        r"(?:washout|wash-out)\s+(?:period\s+)?(?:of\s+)?(\d+\s*(?:days?|weeks?|months?))",
        "WASH_CHK",
        "Washout Period Verification",
        "Washout: {match}",
    ),
    # Life expectancy
    (
        r"life\s+expectancy\s*(?:[≥>=]+\s*\d+\s*(?:months?|weeks?))",
        "LIFE_CHK",
        "Life Expectancy Verification",
        "{match}",
    ),
]

# Generic fallback for unmatched criteria
_GENERIC_TESTCD = "CRIT"
_GENERIC_LABEL = "Criterion Verification"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section_text(sections: list["IchSection"], code: str) -> str:
    """Extract text from the first section matching *code*."""
    for s in sections:
        if s.section_code == code:
            try:
                data = json.loads(s.content_json)
                return (
                    data.get("text_excerpt", "")
                    or data.get("text", "")
                    or ""
                )
            except (json.JSONDecodeError, AttributeError):
                return ""
    return ""


def _parse_criteria(text: str) -> list[tuple[str, str]]:
    """Parse numbered criteria from B.5 text.

    Returns:
        List of (criterion_text, category) tuples.
        Category is "INCLUSION" or "EXCLUSION".
    """
    criteria: list[tuple[str, str]] = []
    current_cat = "INCLUSION"

    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"exclu?sion", stripped, re.IGNORECASE):
            current_cat = "EXCLUSION"
        elif re.search(r"inclu?sion", stripped, re.IGNORECASE):
            current_cat = "INCLUSION"

        m = re.match(r"^\d+[\.\)]\s+(.+)", stripped)
        if m:
            criteria.append((m.group(1).strip(), current_cat))

    return criteria


def _match_criterion(
    criterion_text: str,
    seq: int,
    category: str,
) -> dict[str, Any]:
    """Map a single criterion to its data collection variable.

    Tries each pattern in _CRITERION_PATTERNS. If no pattern matches,
    falls back to a generic criterion verification row.

    Returns:
        Dict with IE domain variable values.
    """
    text_lower = criterion_text.lower()

    for pattern, testcd, label, orres_template in _CRITERION_PATTERNS:
        m = re.search(pattern, criterion_text, re.IGNORECASE)
        if m:
            # Build IEORRES from template
            match_text = m.group(0).strip()
            orres = orres_template.replace("{match}", match_text)
            if m.lastindex and m.lastindex >= 1:
                orres = orres.replace("{g1}", m.group(1))
            if m.lastindex and m.lastindex >= 2:
                orres = orres.replace("{g2}", m.group(2))

            suffix = "I" if category == "INCLUSION" else "E"
            return {
                "IETESTCD": f"{testcd}{seq:02d}"[:8],
                "IETEST": label,
                "IEORRES": orres[:200],
                "IESTRESC": "Y" if category == "INCLUSION" else "N",
                "IECAT": category,
                "IESCAT": "",
                "IECRTEXT": criterion_text[:200],
                "MATCHED": True,
            }

    # Generic fallback
    suffix = "I" if category == "INCLUSION" else "E"
    return {
        "IETESTCD": f"{_GENERIC_TESTCD}{seq:02d}{suffix}"[:8],
        "IETEST": _GENERIC_LABEL,
        "IEORRES": criterion_text[:100],
        "IESTRESC": "",
        "IECAT": category,
        "IESCAT": "",
        "IECRTEXT": criterion_text[:200],
        "MATCHED": False,
    }


# ---------------------------------------------------------------------------
# IeSpecGenerator
# ---------------------------------------------------------------------------


class IeSpecGenerator:
    """Generate the IE (Inclusion/Exclusion) SDTM domain.

    Parses each I/E criterion from B.5 and maps it to a data collection
    variable specifying what must be measured/verified at screening.

    The IE domain complements the TI domain:
    - TI: stores the criteria *text* (what the rule says)
    - IE: stores the data collection *spec* (what must be collected)
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
        run_id: str,
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build the IE domain DataFrame.

        Args:
            sections: All IchSection records for the protocol.
            studyid: STUDYID value.
            run_id: Pipeline run UUID4.

        Returns:
            Tuple of (IE DataFrame, unmapped CT terms).
        """
        b5_text = _section_text(sections, "B.5")
        criteria = _parse_criteria(b5_text)

        rows: list[dict[str, Any]] = []
        inc_seq = 0
        exc_seq = 0

        for criterion_text, category in criteria:
            if category == "INCLUSION":
                inc_seq += 1
                seq = inc_seq
            else:
                exc_seq += 1
                seq = exc_seq

            matched = _match_criterion(criterion_text, seq, category)

            rows.append({
                "STUDYID": studyid,
                "DOMAIN": "IE",
                "USUBJID": "",  # Spec only — no subject data
                "IESEQ": inc_seq + exc_seq,
                "IETESTCD": matched["IETESTCD"],
                "IETEST": matched["IETEST"],
                "IEORRES": matched["IEORRES"],
                "IESTRESC": matched["IESTRESC"],
                "IECAT": matched["IECAT"],
                "IESCAT": matched["IESCAT"],
                "VISITNUM": 1,  # Screening visit
                "VISIT": "SCREENING",
                "IEDTC": "",
                "IECRTEXT": matched["IECRTEXT"],
            })

        return pd.DataFrame(rows), []

    @staticmethod
    def get_match_summary(
        sections: list["IchSection"],
    ) -> dict[str, Any]:
        """Summarize how many criteria were pattern-matched vs generic.

        Useful for reporting coverage of the criterion-to-variable mapping.

        Returns:
            Dict with total, matched, unmatched counts and details.
        """
        b5_text = _section_text(sections, "B.5")
        criteria = _parse_criteria(b5_text)

        matched_count = 0
        unmatched: list[str] = []

        for i, (criterion_text, category) in enumerate(criteria):
            result = _match_criterion(criterion_text, i + 1, category)
            if result["MATCHED"]:
                matched_count += 1
            else:
                unmatched.append(criterion_text[:80])

        return {
            "total": len(criteria),
            "matched": matched_count,
            "unmatched": len(criteria) - matched_count,
            "unmatched_criteria": unmatched,
            "match_rate": (
                matched_count / len(criteria) if criteria else 0.0
            ),
        }


__all__ = [
    "IeSpecGenerator",
]
