"""Domain spec builder — maps SoA assessments to SDTM domain specs (PTCV-246).

Generates domain-level specifications with standard CDISC variable
sets for subject-level observation domains: VS (Vital Signs),
LB (Labs), EG (ECG), PE (Physical Exam), and others.

Each domain spec includes:
- Standard variable list (TESTCD, TEST, ORRES, etc.)
- Test codes mapped from SoA assessment names
- Visit schedule from SoA (which visits collect each assessment)

Risk tier: MEDIUM — regulatory submission artifacts, no patient data.

References:
- SDTMIG v3.4 Section 6: Findings Observation Classes
- CDISC SDTM v1.7 variable attributes
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from .soa_mapper import _classify_assessment

if TYPE_CHECKING:
    from ..soa_extractor.models import RawSoaTable


# ---------------------------------------------------------------------------
# CDISC variable definitions per domain
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CdiscVariable:
    """A single CDISC variable definition.

    Attributes:
        name: Variable name (e.g., "VSTESTCD").
        label: Variable label.
        type: "Char" or "Num".
        required: Whether the variable is required by SDTM.
    """

    name: str
    label: str
    type: str = "Char"
    required: bool = True


# Standard variable sets per observation domain
_DOMAIN_VARIABLES: dict[str, list[CdiscVariable]] = {
    "VS": [
        CdiscVariable("STUDYID", "Study Identifier"),
        CdiscVariable("DOMAIN", "Domain Abbreviation"),
        CdiscVariable("USUBJID", "Unique Subject Identifier"),
        CdiscVariable("VSSEQ", "Sequence Number", "Num"),
        CdiscVariable("VSTESTCD", "Vital Signs Test Short Name"),
        CdiscVariable("VSTEST", "Vital Signs Test Name"),
        CdiscVariable("VSORRES", "Result or Finding in Original Units"),
        CdiscVariable("VSORRESU", "Original Units"),
        CdiscVariable("VSSTRESN", "Numeric Result in Standard Units", "Num"),
        CdiscVariable("VSSTRESU", "Standard Units"),
        CdiscVariable("VSSTAT", "Completion Status", required=False),
        CdiscVariable("VISITNUM", "Visit Number", "Num"),
        CdiscVariable("VISIT", "Visit Name"),
        CdiscVariable("VSDTC", "Date/Time of Measurements"),
    ],
    "LB": [
        CdiscVariable("STUDYID", "Study Identifier"),
        CdiscVariable("DOMAIN", "Domain Abbreviation"),
        CdiscVariable("USUBJID", "Unique Subject Identifier"),
        CdiscVariable("LBSEQ", "Sequence Number", "Num"),
        CdiscVariable("LBTESTCD", "Lab Test Short Name"),
        CdiscVariable("LBTEST", "Lab Test Name"),
        CdiscVariable("LBCAT", "Category for Lab Test", required=False),
        CdiscVariable("LBORRES", "Result or Finding in Original Units"),
        CdiscVariable("LBORRESU", "Original Units"),
        CdiscVariable("LBSTRESN", "Numeric Result in Standard Units", "Num"),
        CdiscVariable("LBSTRESU", "Standard Units"),
        CdiscVariable("LBSPEC", "Specimen Type", required=False),
        CdiscVariable("LBNRIND", "Reference Range Indicator", required=False),
        CdiscVariable("VISITNUM", "Visit Number", "Num"),
        CdiscVariable("VISIT", "Visit Name"),
        CdiscVariable("LBDTC", "Date/Time of Specimen Collection"),
    ],
    "EG": [
        CdiscVariable("STUDYID", "Study Identifier"),
        CdiscVariable("DOMAIN", "Domain Abbreviation"),
        CdiscVariable("USUBJID", "Unique Subject Identifier"),
        CdiscVariable("EGSEQ", "Sequence Number", "Num"),
        CdiscVariable("EGTESTCD", "ECG Test Short Name"),
        CdiscVariable("EGTEST", "ECG Test Name"),
        CdiscVariable("EGORRES", "Result or Finding in Original Units"),
        CdiscVariable("EGORRESU", "Original Units"),
        CdiscVariable("EGSTRESN", "Numeric Result in Standard Units", "Num"),
        CdiscVariable("EGSTAT", "Completion Status", required=False),
        CdiscVariable("VISITNUM", "Visit Number", "Num"),
        CdiscVariable("VISIT", "Visit Name"),
        CdiscVariable("EGDTC", "Date/Time of ECG"),
    ],
    "PE": [
        CdiscVariable("STUDYID", "Study Identifier"),
        CdiscVariable("DOMAIN", "Domain Abbreviation"),
        CdiscVariable("USUBJID", "Unique Subject Identifier"),
        CdiscVariable("PESEQ", "Sequence Number", "Num"),
        CdiscVariable("PETESTCD", "Physical Exam Test Short Name"),
        CdiscVariable("PETEST", "Physical Exam Test Name"),
        CdiscVariable("PEORRES", "Result or Finding in Original Units"),
        CdiscVariable("PEBODSYS", "Body System or Organ Class"),
        CdiscVariable("PESTAT", "Completion Status", required=False),
        CdiscVariable("VISITNUM", "Visit Number", "Num"),
        CdiscVariable("VISIT", "Visit Name"),
        CdiscVariable("PEDTC", "Date/Time of Examination"),
    ],
}


# ---------------------------------------------------------------------------
# Assessment → test code mapping
# ---------------------------------------------------------------------------

# Maps common assessment names to CDISC controlled terminology test codes
_ASSESSMENT_TO_TESTCODES: dict[str, list[tuple[str, str]]] = {
    # VS test codes
    "blood pressure": [("SYSBP", "Systolic Blood Pressure"), ("DIABP", "Diastolic Blood Pressure")],
    "systolic": [("SYSBP", "Systolic Blood Pressure")],
    "diastolic": [("DIABP", "Diastolic Blood Pressure")],
    "heart rate": [("HR", "Heart Rate")],
    "pulse": [("PULSE", "Pulse Rate")],
    "temperature": [("TEMP", "Temperature")],
    "weight": [("WEIGHT", "Weight")],
    "height": [("HEIGHT", "Height")],
    "bmi": [("BMI", "Body Mass Index")],
    "respiratory": [("RESP", "Respiratory Rate")],
    "vital": [
        ("SYSBP", "Systolic Blood Pressure"),
        ("DIABP", "Diastolic Blood Pressure"),
        ("HR", "Heart Rate"),
        ("TEMP", "Temperature"),
        ("RESP", "Respiratory Rate"),
        ("WEIGHT", "Weight"),
    ],
    # LB test codes
    "hematology": [("WBC", "White Blood Cells"), ("RBC", "Red Blood Cells"), ("HGB", "Hemoglobin"), ("HCT", "Hematocrit"), ("PLAT", "Platelets")],
    "chemistry": [("ALT", "Alanine Aminotransferase"), ("AST", "Aspartate Aminotransferase"), ("BILI", "Bilirubin"), ("CREAT", "Creatinine"), ("ALB", "Albumin")],
    "coagulation": [("PT", "Prothrombin Time"), ("INR", "International Normalized Ratio"), ("APTT", "Activated Partial Thromboplastin Time")],
    "urinalysis": [("UPROT", "Urine Protein"), ("UGLUC", "Urine Glucose")],
    "lipid": [("CHOL", "Cholesterol"), ("TRIG", "Triglycerides"), ("HDL", "HDL Cholesterol"), ("LDL", "LDL Cholesterol")],
    "thyroid": [("TSH", "Thyroid Stimulating Hormone"), ("T4", "Thyroxine")],
    "liver function": [("ALT", "Alanine Aminotransferase"), ("AST", "Aspartate Aminotransferase"), ("ALP", "Alkaline Phosphatase"), ("BILI", "Bilirubin")],
    "fasting glucose": [("GLUC", "Glucose")],
    "glucose": [("GLUC", "Glucose")],
    "hba1c": [("HBA1C", "Hemoglobin A1C")],
    "ldh": [("LDH", "Lactate Dehydrogenase")],
    "complement": [("C3", "Complement C3"), ("C4", "Complement C4")],
    "reticulocyte": [("RETIC", "Reticulocyte Count")],
    "egfr": [("EGFR", "Estimated Glomerular Filtration Rate")],
    # EG test codes
    "ecg": [("INTP", "Interpretation"), ("QTCF", "QTcF Interval"), ("HR", "Heart Rate")],
    "electrocardiogram": [("INTP", "Interpretation"), ("QTCF", "QTcF Interval")],
    "12-lead": [("INTP", "Interpretation"), ("QTCF", "QTcF Interval"), ("HR", "Heart Rate")],
}

# Specimen type inference for LB domain
_SPECIMEN_MAP: dict[str, str] = {
    "hematology": "BLOOD",
    "chemistry": "SERUM",
    "coagulation": "PLASMA",
    "urinalysis": "URINE",
    "serology": "SERUM",
    "lipid": "SERUM",
    "liver function": "SERUM",
    "thyroid": "SERUM",
    "glucose": "PLASMA",
    "fasting glucose": "PLASMA",
    "hba1c": "BLOOD",
    "ldh": "SERUM",
    "complement": "SERUM",
    "reticulocyte": "BLOOD",
    "egfr": "SERUM",
    "cbc": "BLOOD",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DomainTestCode:
    """A test code within a domain spec.

    Attributes:
        testcd: Test short name (e.g., "SYSBP").
        test: Test full name (e.g., "Systolic Blood Pressure").
        source_assessment: SoA assessment name that mapped here.
        visit_schedule: Visit names where this test is scheduled.
    """

    testcd: str
    test: str
    source_assessment: str = ""
    visit_schedule: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DomainSpec:
    """Specification for one SDTM observation domain.

    Attributes:
        domain_code: SDTM domain code (e.g., "VS").
        domain_name: Domain full name (e.g., "Vital Signs").
        variables: Standard CDISC variable definitions.
        test_codes: Mapped test codes from SoA assessments.
        specimen_type: Specimen type (LB domain only).
        source_assessments: SoA assessment names that mapped here.
    """

    domain_code: str
    domain_name: str
    variables: list[CdiscVariable] = dataclasses.field(default_factory=list)
    test_codes: list[DomainTestCode] = dataclasses.field(default_factory=list)
    specimen_type: str = ""
    source_assessments: list[str] = dataclasses.field(default_factory=list)

    @property
    def variable_count(self) -> int:
        return len(self.variables)

    @property
    def test_count(self) -> int:
        return len(self.test_codes)


@dataclasses.dataclass
class UnmappedAssessment:
    """An SoA assessment with no SDTM domain mapping.

    Attributes:
        assessment_name: Name from SoA table.
        suggested_domain: Best-guess domain (defaults to "FA").
        reason: Why it couldn't be mapped.
    """

    assessment_name: str
    suggested_domain: str = "FA"
    reason: str = "No keyword match"


@dataclasses.dataclass
class DomainSpecResult:
    """Result of building domain specs from SoA.

    Attributes:
        specs: Domain specifications produced.
        unmapped: Assessments that couldn't be mapped.
        total_assessments: Total assessments processed.
        mapped_count: Number successfully mapped.
    """

    specs: list[DomainSpec] = dataclasses.field(default_factory=list)
    unmapped: list[UnmappedAssessment] = dataclasses.field(default_factory=list)
    total_assessments: int = 0
    mapped_count: int = 0

    def get_spec(self, domain_code: str) -> DomainSpec | None:
        """Get spec for a specific domain."""
        for s in self.specs:
            if s.domain_code == domain_code:
                return s
        return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _infer_test_codes(
    assessment_name: str,
) -> list[tuple[str, str]]:
    """Infer CDISC test codes from an assessment name.

    Args:
        assessment_name: SoA assessment name.

    Returns:
        List of (testcd, test_name) tuples. Empty if no match.
    """
    lower = assessment_name.lower()
    for keyword, codes in _ASSESSMENT_TO_TESTCODES.items():
        if keyword in lower:
            return codes
    return []


def _infer_specimen(assessment_name: str) -> str:
    """Infer specimen type for LB domain assessments.

    Args:
        assessment_name: SoA assessment name.

    Returns:
        Specimen type string (e.g., "BLOOD"), or empty string.
    """
    lower = assessment_name.lower()
    for keyword, specimen in _SPECIMEN_MAP.items():
        if keyword in lower:
            return specimen
    return ""


def build_domain_specs(
    table: "RawSoaTable",
    target_domains: set[str] | None = None,
) -> DomainSpecResult:
    """Build SDTM domain specs from SoA table assessments.

    For each assessment in the SoA table, maps it to an SDTM domain
    and generates a domain spec with standard CDISC variables and
    inferred test codes.

    Args:
        table: Parsed SoA table with assessments and visit schedule.
        target_domains: Optional set of domain codes to generate
            (e.g., {"VS", "LB", "EG", "PE"}). If None, all mapped
            domains are included.

    Returns:
        DomainSpecResult with specs and unmapped assessments.
    """
    # Group assessments by domain
    domain_assessments: dict[str, list[tuple[str, list[bool]]]] = {}
    unmapped: list[UnmappedAssessment] = []
    total = 0

    for name, flags in table.activities:
        name = name.strip()
        if not name:
            continue
        total += 1

        domain_code, domain_name = _classify_assessment(name)

        if domain_code == "FA":
            unmapped.append(UnmappedAssessment(
                assessment_name=name,
                suggested_domain="FA",
                reason="No keyword match in _ASSESSMENT_TO_DOMAIN",
            ))
            continue

        if target_domains and domain_code not in target_domains:
            continue

        domain_assessments.setdefault(domain_code, []).append(
            (name, flags)
        )

    # Build specs for each domain
    specs: list[DomainSpec] = []

    for domain_code, assessments in sorted(domain_assessments.items()):
        variables = _DOMAIN_VARIABLES.get(domain_code, [])
        test_codes: list[DomainTestCode] = []
        source_names: list[str] = []
        specimens: set[str] = set()

        for assess_name, flags in assessments:
            source_names.append(assess_name)

            # Build visit schedule
            visit_schedule = [
                table.visit_headers[i]
                for i, scheduled in enumerate(flags)
                if scheduled and i < len(table.visit_headers)
            ]

            # Infer test codes
            codes = _infer_test_codes(assess_name)
            if codes:
                for testcd, test_name in codes:
                    # Avoid duplicate test codes
                    if not any(tc.testcd == testcd for tc in test_codes):
                        test_codes.append(DomainTestCode(
                            testcd=testcd,
                            test=test_name,
                            source_assessment=assess_name,
                            visit_schedule=visit_schedule,
                        ))
            else:
                # No specific test codes — use assessment name
                testcd = assess_name[:8].upper().replace(" ", "")
                test_codes.append(DomainTestCode(
                    testcd=testcd,
                    test=assess_name,
                    source_assessment=assess_name,
                    visit_schedule=visit_schedule,
                ))

            # Infer specimen for LB
            if domain_code == "LB":
                spec = _infer_specimen(assess_name)
                if spec:
                    specimens.add(spec)

        # Determine domain name from first assessment classification
        _, domain_name = _classify_assessment(assessments[0][0])

        specs.append(DomainSpec(
            domain_code=domain_code,
            domain_name=domain_name,
            variables=list(variables),
            test_codes=test_codes,
            specimen_type=", ".join(sorted(specimens)) if specimens else "",
            source_assessments=source_names,
        ))

    mapped = total - len(unmapped)

    return DomainSpecResult(
        specs=specs,
        unmapped=unmapped,
        total_assessments=total,
        mapped_count=mapped,
    )


__all__ = [
    "CdiscVariable",
    "DomainSpec",
    "DomainSpecResult",
    "DomainTestCode",
    "UnmappedAssessment",
    "build_domain_specs",
]
