"""SDTM domain variable metadata registry (PTCV-116).

Defines per-domain SDTM variable specifications used by the Great
Expectations suite builder and the SDV metadata adapter.  Initial
registry covers 6 observation-class domains: DM, VS, LB, AE, EG, CM.

Variable attributes sourced from SDTMIG v3.4 and the existing
``sdtm/define_xml.py`` ``_VAR_ATTRS`` dictionary.

Risk tier: LOW — metadata definitions only; no patient data.

Regulatory references:
- SDTMIG v3.4 domain specifications
- CDISC Controlled Terminology 2023-12-15
"""

from __future__ import annotations

import dataclasses
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SdtmVariableSpec:
    """Specification for a single SDTM variable within a domain.

    Attributes:
        name: Variable name (e.g. ``VSTESTCD``).
        label: Human-readable label from SDTMIG.
        type: ``"Char"`` or ``"Num"``.
        core: ``"Required"``, ``"Expected"``, or ``"Permissible"``.
        length: Maximum character/display length.
        codelist: Allowed values when controlled terminology applies,
            or ``None`` if unrestricted.
        range_min: Minimum plausible numeric value (``None`` if N/A).
        range_max: Maximum plausible numeric value (``None`` if N/A).
    """

    name: str
    label: str
    type: str  # "Char" | "Num"
    core: str  # "Required" | "Expected" | "Permissible"
    length: int = 200
    codelist: Optional[frozenset[str]] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class SdtmDomainSpec:
    """Specification for a complete SDTM domain.

    Attributes:
        domain_code: Two-letter domain abbreviation (e.g. ``"VS"``).
        domain_name: Full domain name (e.g. ``"Vital Signs"``).
        domain_class: SDTM observation class
            (``"Findings"``, ``"Events"``, ``"Interventions"``,
            ``"Special Purpose"``).
        variables: Ordered list of variable specifications.
        keys: Variable names forming the natural key.
    """

    domain_code: str
    domain_name: str
    domain_class: str
    variables: tuple[SdtmVariableSpec, ...]
    keys: tuple[str, ...]


# ---------------------------------------------------------------------------
# Helper to shorten variable definitions
# ---------------------------------------------------------------------------

def _var(
    name: str,
    label: str,
    type_: str = "Char",
    core: str = "Expected",
    length: int = 200,
    codelist: frozenset[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
) -> SdtmVariableSpec:
    return SdtmVariableSpec(
        name=name,
        label=label,
        type=type_,
        core=core,
        length=length,
        codelist=codelist,
        range_min=range_min,
        range_max=range_max,
    )


# ---------------------------------------------------------------------------
# Domain: DM — Demographics (Special Purpose)
# ---------------------------------------------------------------------------

_DM_VARS = (
    _var("STUDYID", "Study Identifier", core="Required", length=20),
    _var("DOMAIN", "Domain Abbreviation", core="Required", length=2,
         codelist=frozenset({"DM"})),
    _var("USUBJID", "Unique Subject Identifier", core="Required", length=40),
    _var("SUBJID", "Subject Identifier for the Study", core="Required",
         length=20),
    _var("RFSTDTC", "Subject Reference Start Date/Time", length=20),
    _var("RFENDTC", "Subject Reference End Date/Time", length=20),
    _var("SITEID", "Study Site Identifier", core="Required", length=20),
    _var("AGE", "Age", type_="Num", length=8, range_min=18, range_max=100),
    _var("AGEU", "Age Units", length=10,
         codelist=frozenset({"YEARS", "MONTHS", "DAYS"})),
    _var("SEX", "Sex", core="Required", length=2,
         codelist=frozenset({"M", "F", "U"})),
    _var("RACE", "Race", length=60, codelist=frozenset({
        "WHITE", "BLACK OR AFRICAN AMERICAN",
        "ASIAN", "AMERICAN INDIAN OR ALASKA NATIVE",
        "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
        "MULTIPLE", "OTHER", "NOT REPORTED", "UNKNOWN",
    })),
    _var("ETHNIC", "Ethnicity", length=60, codelist=frozenset({
        "HISPANIC OR LATINO", "NOT HISPANIC OR LATINO",
        "NOT REPORTED", "UNKNOWN",
    })),
    _var("ARMCD", "Planned Arm Code", length=20),
    _var("ARM", "Description of Planned Arm", length=200),
    _var("COUNTRY", "Country", length=3),
    _var("DMDTC", "Date/Time of Collection", length=20),
    _var("DMDY", "Study Day of Collection", type_="Num", length=8),
)

_DM_SPEC = SdtmDomainSpec(
    domain_code="DM",
    domain_name="Demographics",
    domain_class="Special Purpose",
    variables=_DM_VARS,
    keys=("STUDYID", "USUBJID"),
)


# ---------------------------------------------------------------------------
# Domain: VS — Vital Signs (Findings)
# ---------------------------------------------------------------------------

_VS_TESTCD_CL = frozenset({
    "SYSBP", "DIABP", "PULSE", "TEMP", "HEIGHT", "WEIGHT", "BMI", "RESP",
})

_VS_VARS = (
    _var("STUDYID", "Study Identifier", core="Required", length=20),
    _var("DOMAIN", "Domain Abbreviation", core="Required", length=2,
         codelist=frozenset({"VS"})),
    _var("USUBJID", "Unique Subject Identifier", core="Required", length=40),
    _var("VSSEQ", "Sequence Number", type_="Num", core="Required", length=8,
         range_min=1, range_max=9999),
    _var("VSTESTCD", "Vital Signs Test Short Name", core="Required",
         length=8, codelist=_VS_TESTCD_CL),
    _var("VSTEST", "Vital Signs Test Name", core="Required", length=40),
    _var("VSORRES", "Result or Finding in Original Units", core="Required",
         length=200),
    _var("VSORRESU", "Original Units", core="Required", length=40,
         codelist=frozenset({
             "mmHg", "beats/min", "C", "cm", "kg", "kg/m2", "breaths/min",
         })),
    _var("VSSTRESC", "Character Result/Finding in Std Format", length=200),
    _var("VSSTRESN", "Numeric Result/Finding in Standard Units", type_="Num",
         length=8, range_min=0, range_max=500),
    _var("VSSTRESU", "Standard Units", length=40),
    _var("VSSTAT", "Completion Status", length=8,
         codelist=frozenset({"NOT DONE"})),
    _var("VISITNUM", "Visit Number", type_="Num", length=8,
         range_min=1, range_max=999),
    _var("VISIT", "Visit Name", length=40),
    _var("VSDTC", "Date/Time of Measurements", length=20),
    _var("VSDY", "Study Day of Vital Signs", type_="Num", length=8),
)

_VS_SPEC = SdtmDomainSpec(
    domain_code="VS",
    domain_name="Vital Signs",
    domain_class="Findings",
    variables=_VS_VARS,
    keys=("STUDYID", "USUBJID", "VSTESTCD", "VISITNUM"),
)


# ---------------------------------------------------------------------------
# Domain: LB — Laboratory Test Results (Findings)
# ---------------------------------------------------------------------------

_LB_TESTCD_CL = frozenset({
    "ALB", "ALP", "ALT", "AST", "BILI", "BUN", "CA", "CHOL",
    "CK", "CREAT", "GLUC", "HBA1C", "HCT", "HGB", "K", "LDH",
    "LYMPH", "MCV", "MONO", "NA", "NEUT", "PLAT", "PROT", "RBC",
    "TRIG", "URATE", "WBC",
})

_LB_VARS = (
    _var("STUDYID", "Study Identifier", core="Required", length=20),
    _var("DOMAIN", "Domain Abbreviation", core="Required", length=2,
         codelist=frozenset({"LB"})),
    _var("USUBJID", "Unique Subject Identifier", core="Required", length=40),
    _var("LBSEQ", "Sequence Number", type_="Num", core="Required", length=8,
         range_min=1, range_max=9999),
    _var("LBTESTCD", "Lab Test or Examination Short Name", core="Required",
         length=8, codelist=_LB_TESTCD_CL),
    _var("LBTEST", "Lab Test or Examination Name", core="Required",
         length=40),
    _var("LBCAT", "Category for Lab Test", length=40,
         codelist=frozenset({
             "CHEMISTRY", "HEMATOLOGY", "COAGULATION", "URINALYSIS",
         })),
    _var("LBORRES", "Result or Finding in Original Units", core="Required",
         length=200),
    _var("LBORRESU", "Original Units", core="Required", length=40),
    _var("LBSTRESC", "Character Result/Finding in Std Format", length=200),
    _var("LBSTRESN", "Numeric Result/Finding in Standard Units", type_="Num",
         length=8, range_min=0, range_max=10000),
    _var("LBSTRESU", "Standard Units", length=40),
    _var("LBNRIND", "Reference Range Indicator", length=8,
         codelist=frozenset({"NORMAL", "LOW", "HIGH"})),
    _var("LBORNRLO", "Reference Range Lower Limit in Orig Unit",
         type_="Num", length=8),
    _var("LBORNRHI", "Reference Range Upper Limit in Orig Unit",
         type_="Num", length=8),
    _var("VISITNUM", "Visit Number", type_="Num", length=8,
         range_min=1, range_max=999),
    _var("VISIT", "Visit Name", length=40),
    _var("LBDTC", "Date/Time of Specimen Collection", length=20),
    _var("LBDY", "Study Day of Specimen Collection", type_="Num", length=8),
)

_LB_SPEC = SdtmDomainSpec(
    domain_code="LB",
    domain_name="Laboratory Test Results",
    domain_class="Findings",
    variables=_LB_VARS,
    keys=("STUDYID", "USUBJID", "LBTESTCD", "VISITNUM"),
)


# ---------------------------------------------------------------------------
# Domain: AE — Adverse Events (Events)
# ---------------------------------------------------------------------------

_AE_VARS = (
    _var("STUDYID", "Study Identifier", core="Required", length=20),
    _var("DOMAIN", "Domain Abbreviation", core="Required", length=2,
         codelist=frozenset({"AE"})),
    _var("USUBJID", "Unique Subject Identifier", core="Required", length=40),
    _var("AESEQ", "Sequence Number", type_="Num", core="Required", length=8,
         range_min=1, range_max=9999),
    _var("AETERM", "Reported Term for the Adverse Event", core="Required",
         length=200),
    _var("AEDECOD", "Dictionary-Derived Term", core="Required", length=200),
    _var("AEBODSYS", "Body System or Organ Class", length=200),
    _var("AESEV", "Severity/Intensity", length=20,
         codelist=frozenset({"MILD", "MODERATE", "SEVERE"})),
    _var("AESER", "Serious Event", length=2,
         codelist=frozenset({"Y", "N"})),
    _var("AEREL", "Causality", length=20, codelist=frozenset({
        "RELATED", "POSSIBLY RELATED", "UNLIKELY RELATED",
        "NOT RELATED",
    })),
    _var("AEACN", "Action Taken with Study Treatment", length=40,
         codelist=frozenset({
             "DOSE NOT CHANGED", "DOSE REDUCED", "DRUG INTERRUPTED",
             "DRUG WITHDRAWN", "NOT APPLICABLE",
         })),
    _var("AEOUT", "Outcome of Adverse Event", length=40,
         codelist=frozenset({
             "RECOVERED/RESOLVED",
             "RECOVERING/RESOLVING",
             "NOT RECOVERED/NOT RESOLVED",
             "RECOVERED/RESOLVED WITH SEQUELAE",
             "FATAL",
         })),
    _var("AESTDTC", "Start Date/Time of Adverse Event", length=20),
    _var("AEENDTC", "End Date/Time of Adverse Event", length=20),
    _var("AESTDY", "Study Day of Start of Adverse Event", type_="Num",
         length=8),
    _var("AEENDY", "Study Day of End of Adverse Event", type_="Num",
         length=8),
)

_AE_SPEC = SdtmDomainSpec(
    domain_code="AE",
    domain_name="Adverse Events",
    domain_class="Events",
    variables=_AE_VARS,
    keys=("STUDYID", "USUBJID", "AETERM", "AESTDTC"),
)


# ---------------------------------------------------------------------------
# Domain: EG — ECG Test Results (Findings)
# ---------------------------------------------------------------------------

_EG_TESTCD_CL = frozenset({
    "EGHRMN", "EGPRMN", "EGQRSMN", "EGQTMN", "EGQTCF",
    "EGRRMIN", "EGINTP",
})

_EG_VARS = (
    _var("STUDYID", "Study Identifier", core="Required", length=20),
    _var("DOMAIN", "Domain Abbreviation", core="Required", length=2,
         codelist=frozenset({"EG"})),
    _var("USUBJID", "Unique Subject Identifier", core="Required", length=40),
    _var("EGSEQ", "Sequence Number", type_="Num", core="Required", length=8,
         range_min=1, range_max=9999),
    _var("EGTESTCD", "ECG Test Short Name", core="Required", length=8,
         codelist=_EG_TESTCD_CL),
    _var("EGTEST", "ECG Test Name", core="Required", length=40),
    _var("EGORRES", "Result or Finding in Original Units", core="Required",
         length=200),
    _var("EGORRESU", "Original Units", core="Required", length=40,
         codelist=frozenset({"ms", "beats/min"})),
    _var("EGSTRESC", "Character Result/Finding in Std Format", length=200),
    _var("EGSTRESN", "Numeric Result/Finding in Standard Units", type_="Num",
         length=8, range_min=0, range_max=1000),
    _var("EGSTRESU", "Standard Units", length=40),
    _var("EGSTAT", "Completion Status", length=8,
         codelist=frozenset({"NOT DONE"})),
    _var("VISITNUM", "Visit Number", type_="Num", length=8,
         range_min=1, range_max=999),
    _var("VISIT", "Visit Name", length=40),
    _var("EGDTC", "Date/Time of ECG", length=20),
    _var("EGDY", "Study Day of ECG", type_="Num", length=8),
)

_EG_SPEC = SdtmDomainSpec(
    domain_code="EG",
    domain_name="ECG Test Results",
    domain_class="Findings",
    variables=_EG_VARS,
    keys=("STUDYID", "USUBJID", "EGTESTCD", "VISITNUM"),
)


# ---------------------------------------------------------------------------
# Domain: CM — Concomitant Medications (Interventions)
# ---------------------------------------------------------------------------

_CM_VARS = (
    _var("STUDYID", "Study Identifier", core="Required", length=20),
    _var("DOMAIN", "Domain Abbreviation", core="Required", length=2,
         codelist=frozenset({"CM"})),
    _var("USUBJID", "Unique Subject Identifier", core="Required", length=40),
    _var("CMSEQ", "Sequence Number", type_="Num", core="Required", length=8,
         range_min=1, range_max=9999),
    _var("CMTRT", "Reported Name of Drug, Med, or Therapy", core="Required",
         length=200),
    _var("CMDECOD", "Standardized Medication Name", length=200),
    _var("CMCAT", "Category for Medication", length=40,
         codelist=frozenset({
             "PRIOR", "CONCOMITANT", "PRIOR AND CONCOMITANT",
         })),
    _var("CMDOSE", "Dose per Administration", type_="Num", length=8,
         range_min=0, range_max=10000),
    _var("CMDOSU", "Dose Units", length=40,
         codelist=frozenset({"mg", "g", "mcg", "mL", "IU"})),
    _var("CMROUTE", "Route of Administration", length=40,
         codelist=frozenset({
             "ORAL", "INTRAVENOUS", "SUBCUTANEOUS",
             "INTRAMUSCULAR", "TOPICAL", "INHALATION",
         })),
    _var("CMSTDTC", "Start Date/Time", length=20),
    _var("CMENDTC", "End Date/Time", length=20),
    _var("CMSTDY", "Study Day of Start of Medication", type_="Num",
         length=8),
    _var("CMENDY", "Study Day of End of Medication", type_="Num", length=8),
    _var("CMINDC", "Indication", length=200),
    _var("CMONGO", "Ongoing", length=2,
         codelist=frozenset({"Y", "N"})),
)

_CM_SPEC = SdtmDomainSpec(
    domain_code="CM",
    domain_name="Concomitant Medications",
    domain_class="Interventions",
    variables=_CM_VARS,
    keys=("STUDYID", "USUBJID", "CMTRT", "CMSTDTC"),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, SdtmDomainSpec] = {
    "DM": _DM_SPEC,
    "VS": _VS_SPEC,
    "LB": _LB_SPEC,
    "AE": _AE_SPEC,
    "EG": _EG_SPEC,
    "CM": _CM_SPEC,
}


def get_domain_spec(domain_code: str) -> SdtmDomainSpec:
    """Look up a domain specification by code.

    Args:
        domain_code: Two-letter SDTM domain code (e.g. ``"VS"``).

    Returns:
        The domain specification.

    Raises:
        KeyError: If the domain code is not in the registry.
    """
    return _REGISTRY[domain_code.upper()]


def get_all_domain_specs() -> dict[str, SdtmDomainSpec]:
    """Return all registered domain specifications.

    Returns:
        Dictionary mapping domain code to ``SdtmDomainSpec``.
    """
    return dict(_REGISTRY)
