"""CDISC SDTM Subject-Level Domain Spec Generators.

PTCV-245: Generates domain specification stubs for the 5 always-include
subject-level SDTM domains: DM, DS, AE, CM, MH.

These are *specification* generators — they produce the domain variable
lists, controlled terminology references, and expected data structures
that a clinical trial needs to collect.  They do NOT produce synthetic
patient data (see ``synthetic_generator.py`` for that).

Each generator follows the same pattern as ``domain_generators.py``:
takes IchSection records + studyid, returns (DataFrame, unmapped_ct).

Risk tier: MEDIUM — regulatory submission artefacts; no patient data.

Regulatory references:
- SDTMIG v3.3 Sections 6.1 (DM), 6.2 (AE), 6.3 (CM, MH, DS)
- CDISC SDTM v1.7 for variable attributes
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


def _make_var_row(
    studyid: str,
    domain: str,
    varname: str,
    varlabel: str,
    vartype: str = "Char",
    varlength: int = 200,
    core: str = "Req",
    codelist: str = "",
    origin: str = "CRF",
    comment: str = "",
) -> dict[str, Any]:
    """Build a single variable specification row."""
    return {
        "STUDYID": studyid,
        "DOMAIN": domain,
        "VARNAME": varname,
        "VARLABEL": varlabel,
        "VARTYPE": vartype,
        "VARLENGTH": varlength,
        "CORE": core,
        "CODELIST": codelist,
        "ORIGIN": origin,
        "COMMENT": comment,
    }


# ---------------------------------------------------------------------------
# DM — Demographics
# ---------------------------------------------------------------------------


class DmSpecGenerator:
    """Generates DM (Demographics) domain specification.

    Variables are derived from SDTMIG v3.3 Section 6.1 plus
    protocol-specific arm structure from B.4.

    Source ICH sections: B.4 (arm structure), B.5 (age/sex/race criteria).
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
        run_id: str,
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build DM domain variable specification."""
        rows: list[dict[str, Any]] = []
        domain = "DM"

        # Required identifier variables
        rows.append(_make_var_row(studyid, domain, "STUDYID", "Study Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "DOMAIN", "Domain Abbreviation", varlength=2, core="Req"))
        rows.append(_make_var_row(studyid, domain, "USUBJID", "Unique Subject Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "SUBJID", "Subject Identifier for the Study", core="Req"))
        rows.append(_make_var_row(studyid, domain, "SITEID", "Study Site Identifier", core="Req"))

        # Demographics variables
        rows.append(_make_var_row(studyid, domain, "BRTHDTC", "Date/Time of Birth", origin="CRF", core="Perm"))
        rows.append(_make_var_row(studyid, domain, "AGE", "Age", vartype="Num", varlength=8, core="Exp", codelist="", origin="Derived"))
        rows.append(_make_var_row(studyid, domain, "AGEU", "Age Units", varlength=10, core="Exp", codelist="AGEU", origin="Derived"))
        rows.append(_make_var_row(studyid, domain, "SEX", "Sex", varlength=2, core="Req", codelist="SEX"))
        rows.append(_make_var_row(studyid, domain, "RACE", "Race", core="Exp", codelist="RACE"))
        rows.append(_make_var_row(studyid, domain, "ETHNIC", "Ethnicity", core="Perm", codelist="ETHNIC"))
        rows.append(_make_var_row(studyid, domain, "COUNTRY", "Country", varlength=3, core="Exp", codelist="COUNTRY", origin="Derived"))

        # Arm assignment
        rows.append(_make_var_row(studyid, domain, "ARMCD", "Planned Arm Code", varlength=20, core="Req", codelist="", origin="Assigned"))
        rows.append(_make_var_row(studyid, domain, "ARM", "Description of Planned Arm", core="Req", origin="Assigned"))
        rows.append(_make_var_row(studyid, domain, "ACTARMCD", "Actual Arm Code", varlength=20, core="Req", origin="Assigned"))
        rows.append(_make_var_row(studyid, domain, "ACTARM", "Description of Actual Arm", core="Req", origin="Assigned"))

        # Timing
        rows.append(_make_var_row(studyid, domain, "RFSTDTC", "Subject Reference Start Date/Time", core="Exp", origin="Derived"))
        rows.append(_make_var_row(studyid, domain, "RFENDTC", "Subject Reference End Date/Time", core="Exp", origin="Derived"))
        rows.append(_make_var_row(studyid, domain, "DMDTC", "Date/Time of Collection", core="Perm", origin="CRF"))

        # Extract protocol-specific comments from eligibility
        b5_text = _section_text(sections, "B.5")
        if b5_text:
            age_comment = ""
            m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s*years", b5_text, re.IGNORECASE)
            if m:
                age_comment = f"Protocol specifies age {m.group(1)}-{m.group(2)} years"
            elif re.search(r"[≥>=]+\s*(\d+)\s*years", b5_text, re.IGNORECASE):
                m2 = re.search(r"[≥>=]+\s*(\d+)\s*years", b5_text, re.IGNORECASE)
                if m2:
                    age_comment = f"Protocol specifies age >= {m2.group(1)} years"

            if age_comment:
                for row in rows:
                    if row["VARNAME"] == "AGE":
                        row["COMMENT"] = age_comment

        return pd.DataFrame(rows), []


# ---------------------------------------------------------------------------
# DS — Disposition
# ---------------------------------------------------------------------------


class DsSpecGenerator:
    """Generates DS (Disposition) domain specification.

    Captures subject milestones: informed consent, randomization,
    treatment completion, early termination, study completion.

    Source ICH sections: B.1 (study admin), B.4 (design).
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
        run_id: str,
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build DS domain variable specification."""
        rows: list[dict[str, Any]] = []
        domain = "DS"

        # Identifiers
        rows.append(_make_var_row(studyid, domain, "STUDYID", "Study Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "DOMAIN", "Domain Abbreviation", varlength=2, core="Req"))
        rows.append(_make_var_row(studyid, domain, "USUBJID", "Unique Subject Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "DSSEQ", "Sequence Number", vartype="Num", varlength=8, core="Req"))

        # Topic
        rows.append(_make_var_row(studyid, domain, "DSTERM", "Reported Term for the Disposition Event", core="Req"))
        rows.append(_make_var_row(studyid, domain, "DSDECOD", "Standardized Disposition Term", core="Req", codelist="NCOMPLT"))

        # Qualifiers
        rows.append(_make_var_row(studyid, domain, "DSCAT", "Category for Disposition Event", core="Exp", codelist="DSCAT",
                                  comment="PROTOCOL MILESTONE, DISPOSITION EVENT"))
        rows.append(_make_var_row(studyid, domain, "DSSCAT", "Subcategory for Disposition Event", core="Perm"))
        rows.append(_make_var_row(studyid, domain, "EPOCH", "Epoch", core="Exp", codelist="EPOCH", origin="Derived"))

        # Timing
        rows.append(_make_var_row(studyid, domain, "DSSTDTC", "Start Date/Time of Disposition Event", core="Exp", origin="CRF"))

        return pd.DataFrame(rows), []


# ---------------------------------------------------------------------------
# AE — Adverse Events
# ---------------------------------------------------------------------------


class AeSpecGenerator:
    """Generates AE (Adverse Events) domain specification.

    Standard AE collection variables per SDTMIG v3.3 Section 6.2.
    All interventional trials must collect AE data.

    Source ICH sections: B.9 (safety assessments).
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
        run_id: str,
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build AE domain variable specification."""
        rows: list[dict[str, Any]] = []
        domain = "AE"

        # Identifiers
        rows.append(_make_var_row(studyid, domain, "STUDYID", "Study Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "DOMAIN", "Domain Abbreviation", varlength=2, core="Req"))
        rows.append(_make_var_row(studyid, domain, "USUBJID", "Unique Subject Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "AESEQ", "Sequence Number", vartype="Num", varlength=8, core="Req"))

        # Topic
        rows.append(_make_var_row(studyid, domain, "AETERM", "Reported Term for the Adverse Event", core="Req"))
        rows.append(_make_var_row(studyid, domain, "AEDECOD", "Dictionary-Derived Term", core="Req",
                                  codelist="MedDRA", comment="MedDRA preferred term"))
        rows.append(_make_var_row(studyid, domain, "AEBODSYS", "Body System or Organ Class", core="Exp",
                                  codelist="MedDRA", comment="MedDRA system organ class"))

        # Qualifiers — severity/seriousness
        rows.append(_make_var_row(studyid, domain, "AESEV", "Severity/Intensity", core="Exp", codelist="AESEV",
                                  comment="MILD, MODERATE, SEVERE"))
        rows.append(_make_var_row(studyid, domain, "AESER", "Serious Event", varlength=2, core="Exp", codelist="NY"))
        rows.append(_make_var_row(studyid, domain, "AEREL", "Causality", core="Exp", codelist="AEREL",
                                  comment="Relationship to study drug"))
        rows.append(_make_var_row(studyid, domain, "AEACN", "Action Taken with Study Treatment", core="Exp", codelist="ACN"))
        rows.append(_make_var_row(studyid, domain, "AEOUT", "Outcome of Adverse Event", core="Exp", codelist="OUT"))

        # Timing
        rows.append(_make_var_row(studyid, domain, "AESTDTC", "Start Date/Time of Adverse Event", core="Exp", origin="CRF"))
        rows.append(_make_var_row(studyid, domain, "AEENDTC", "End Date/Time of Adverse Event", core="Exp", origin="CRF"))
        rows.append(_make_var_row(studyid, domain, "EPOCH", "Epoch", core="Exp", codelist="EPOCH", origin="Derived"))

        return pd.DataFrame(rows), []


# ---------------------------------------------------------------------------
# CM — Concomitant Medications
# ---------------------------------------------------------------------------


class CmSpecGenerator:
    """Generates CM (Concomitant Medications) domain specification.

    Variables for recording medications taken before/during the trial.

    Source ICH sections: B.5 (prohibited/allowed medications).
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
        run_id: str,
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build CM domain variable specification."""
        rows: list[dict[str, Any]] = []
        domain = "CM"

        # Identifiers
        rows.append(_make_var_row(studyid, domain, "STUDYID", "Study Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "DOMAIN", "Domain Abbreviation", varlength=2, core="Req"))
        rows.append(_make_var_row(studyid, domain, "USUBJID", "Unique Subject Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "CMSEQ", "Sequence Number", vartype="Num", varlength=8, core="Req"))

        # Topic
        rows.append(_make_var_row(studyid, domain, "CMTRT", "Reported Name of Drug, Med, or Therapy", core="Req"))
        rows.append(_make_var_row(studyid, domain, "CMDECOD", "Standardized Medication Name", core="Exp",
                                  codelist="WHODrug", comment="WHODrug preferred name"))
        rows.append(_make_var_row(studyid, domain, "CMCAT", "Category for Medication", core="Perm",
                                  comment="PRE-STUDY, CONCOMITANT, PROHIBITED"))

        # Qualifiers
        rows.append(_make_var_row(studyid, domain, "CMDOSE", "Dose per Administration", vartype="Num", varlength=8, core="Exp"))
        rows.append(_make_var_row(studyid, domain, "CMDOSU", "Dose Units", core="Exp", codelist="UNIT"))
        rows.append(_make_var_row(studyid, domain, "CMDOSFRQ", "Dosing Frequency per Interval", core="Exp", codelist="FREQ"))
        rows.append(_make_var_row(studyid, domain, "CMROUTE", "Route of Administration", core="Exp", codelist="ROUTE"))
        rows.append(_make_var_row(studyid, domain, "CMINDC", "Indication", core="Perm"))

        # Timing
        rows.append(_make_var_row(studyid, domain, "CMSTDTC", "Start Date/Time of Medication", core="Exp", origin="CRF"))
        rows.append(_make_var_row(studyid, domain, "CMENDTC", "End Date/Time of Medication", core="Exp", origin="CRF"))
        rows.append(_make_var_row(studyid, domain, "CMSTRF", "Start Relative to Reference Period", core="Perm", codelist="STENRF",
                                  comment="BEFORE, DURING, AFTER"))

        return pd.DataFrame(rows), []


# ---------------------------------------------------------------------------
# MH — Medical History
# ---------------------------------------------------------------------------


class MhSpecGenerator:
    """Generates MH (Medical History) domain specification.

    Variables for recording pre-existing conditions. Medical history
    categories are extracted from B.5 eligibility criteria where
    specific conditions are required or excluded.

    Source ICH sections: B.5 (medical history in eligibility).
    """

    def generate(
        self,
        sections: list["IchSection"],
        studyid: str,
        run_id: str,
    ) -> tuple[pd.DataFrame, list[CtLookupResult]]:
        """Build MH domain variable specification."""
        rows: list[dict[str, Any]] = []
        domain = "MH"

        # Identifiers
        rows.append(_make_var_row(studyid, domain, "STUDYID", "Study Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "DOMAIN", "Domain Abbreviation", varlength=2, core="Req"))
        rows.append(_make_var_row(studyid, domain, "USUBJID", "Unique Subject Identifier", core="Req"))
        rows.append(_make_var_row(studyid, domain, "MHSEQ", "Sequence Number", vartype="Num", varlength=8, core="Req"))

        # Topic
        rows.append(_make_var_row(studyid, domain, "MHTERM", "Reported Term for the Medical History", core="Req"))
        rows.append(_make_var_row(studyid, domain, "MHDECOD", "Dictionary-Derived Term", core="Exp",
                                  codelist="MedDRA", comment="MedDRA preferred term"))
        rows.append(_make_var_row(studyid, domain, "MHBODSYS", "Body System or Organ Class", core="Exp",
                                  codelist="MedDRA", comment="MedDRA system organ class"))

        # Qualifiers
        rows.append(_make_var_row(studyid, domain, "MHCAT", "Category for Medical History", core="Perm",
                                  comment="Protocol-defined categories from eligibility"))
        rows.append(_make_var_row(studyid, domain, "MHPRESP", "Pre-Specified", varlength=2, core="Perm", codelist="NY",
                                  comment="Y if condition is pre-specified in eligibility criteria"))
        rows.append(_make_var_row(studyid, domain, "MHOCCUR", "Occurrence", varlength=2, core="Perm", codelist="NY",
                                  comment="Y/N — did this condition occur"))

        # Timing
        rows.append(_make_var_row(studyid, domain, "MHSTDTC", "Start Date/Time of Medical History Event", core="Perm", origin="CRF"))
        rows.append(_make_var_row(studyid, domain, "MHENDTC", "End Date/Time of Medical History Event", core="Perm", origin="CRF"))
        rows.append(_make_var_row(studyid, domain, "MHENRF", "End Relative to Reference Period", core="Perm", codelist="STENRF"))

        # Extract protocol-specific MH categories from eligibility
        b5_text = _section_text(sections, "B.5")
        mh_categories = self._extract_mh_categories(b5_text)
        if mh_categories:
            for row in rows:
                if row["VARNAME"] == "MHCAT":
                    row["COMMENT"] = (
                        "Protocol-defined categories: "
                        + ", ".join(mh_categories)
                    )

        return pd.DataFrame(rows), []

    @staticmethod
    def _extract_mh_categories(b5_text: str) -> list[str]:
        """Extract medical history category keywords from eligibility text.

        Looks for disease/condition mentions that imply MH data collection.
        """
        if not b5_text:
            return []

        categories: list[str] = []
        # Common disease categories in eligibility criteria
        patterns = [
            (r"(?:history\s+of\s+)?cardiovascular\b", "Cardiovascular"),
            (r"(?:history\s+of\s+)?hepatic\b|liver\s+disease", "Hepatic"),
            (r"(?:history\s+of\s+)?renal\b|kidney\s+disease", "Renal"),
            (r"(?:history\s+of\s+)?diabetes\b", "Endocrine/Metabolic"),
            (r"(?:history\s+of\s+)?cancer\b|malignancy\b|oncolog", "Oncology"),
            (r"(?:history\s+of\s+)?pulmonary\b|respiratory\b", "Respiratory"),
            (r"(?:history\s+of\s+)?neurologic\b|seizure\b|epileps", "Neurological"),
            (r"(?:history\s+of\s+)?psychiatric\b|depression\b", "Psychiatric"),
            (r"(?:history\s+of\s+)?autoimmune\b", "Autoimmune"),
            (r"(?:history\s+of\s+)?infection\b|HIV\b|hepatitis\b", "Infectious Disease"),
            (r"surgery\b|surgical\b", "Surgical"),
            (r"allergy\b|allergic\b|hypersensitiv", "Allergy"),
        ]

        for pattern, category in patterns:
            if re.search(pattern, b5_text, re.IGNORECASE):
                categories.append(category)

        return categories


# ---------------------------------------------------------------------------
# Convenience: generate all always-include domains
# ---------------------------------------------------------------------------


def generate_all_subject_domains(
    sections: list["IchSection"],
    studyid: str,
    run_id: str,
) -> dict[str, tuple[pd.DataFrame, list[CtLookupResult]]]:
    """Generate specs for all 5 always-include subject-level domains.

    Returns:
        Dict mapping domain code to (DataFrame, unmapped_ct) tuple.
    """
    generators: dict[str, Any] = {
        "DM": DmSpecGenerator(),
        "DS": DsSpecGenerator(),
        "AE": AeSpecGenerator(),
        "CM": CmSpecGenerator(),
        "MH": MhSpecGenerator(),
    }

    results: dict[str, tuple[pd.DataFrame, list[CtLookupResult]]] = {}
    for domain_code, generator in generators.items():
        results[domain_code] = generator.generate(sections, studyid, run_id)

    return results


__all__ = [
    "DmSpecGenerator",
    "DsSpecGenerator",
    "AeSpecGenerator",
    "CmSpecGenerator",
    "MhSpecGenerator",
    "generate_all_subject_domains",
]
