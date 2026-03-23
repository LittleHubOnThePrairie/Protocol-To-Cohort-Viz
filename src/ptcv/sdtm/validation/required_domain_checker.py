"""Required domain checker for SDTM packages (PTCV-249).

Validates that all CDISC SDTM "always-include" domains are present
in the generated SDTM package. Flags missing required domains as
errors and missing conditional domains as warnings.

Risk tier: MEDIUM — regulatory validation, no patient data.

References:
- SDTMIG v3.4 Section 3.2: Required Domains
- FDA Technical Conformance Guide v5.9
"""

from __future__ import annotations

import dataclasses
from typing import Optional


# ---------------------------------------------------------------------------
# Domain requirements
# ---------------------------------------------------------------------------

# Always required for interventional trials
ALWAYS_REQUIRED: list[tuple[str, str]] = [
    ("DM", "Demographics"),
    ("DS", "Disposition"),
    ("AE", "Adverse Events"),
    ("TA", "Trial Arms"),
    ("TE", "Trial Elements"),
    ("TI", "Trial Inclusion/Exclusion"),
    ("TS", "Trial Summary"),
    ("TV", "Trial Visits"),
]

# Conditionally required — triggered by SoA assessment types
CONDITIONAL_DOMAINS: list[tuple[str, str, list[str]]] = [
    # (domain_code, domain_name, trigger_keywords)
    ("CM", "Concomitant Medications", [
        "concomitant", "medication", "prior medication",
    ]),
    ("MH", "Medical History", [
        "medical history", "history",
    ]),
    ("VS", "Vital Signs", [
        "vital", "blood pressure", "heart rate", "pulse",
        "temperature", "weight",
    ]),
    ("LB", "Laboratory Test Results", [
        "lab", "hematology", "chemistry", "urinalysis",
        "coagulation", "blood count", "cbc",
    ]),
    ("EG", "ECG Test Results", [
        "ecg", "electrocardiogram", "12-lead",
    ]),
    ("PE", "Physical Examination", [
        "physical exam", "physical examination",
    ]),
    ("EX", "Exposure", [
        "drug", "treatment", "dose", "dosing",
    ]),
    ("PC", "Pharmacokinetics Concentrations", [
        "pharmacokinetic", "pk sample", "pk sampling",
        "drug concentration",
    ]),
    ("QS", "Questionnaires", [
        "questionnaire", "quality of life", "ecog",
        "performance status", "diary",
    ]),
    ("TU", "Tumor Results", [
        "tumor", "tumour", "recist", "imaging",
    ]),
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DomainFinding:
    """A single domain completeness validation finding.

    Attributes:
        domain_code: SDTM domain code (e.g., "DM").
        domain_name: Domain full name.
        severity: "Error" for always-required, "Warning" for conditional.
        message: Human-readable finding description.
        trigger: What triggered the check (e.g., "always required"
            or "SoA contains vital signs").
    """

    domain_code: str
    domain_name: str
    severity: str
    message: str
    trigger: str = ""


@dataclasses.dataclass
class DomainCheckResult:
    """Result of required domain validation.

    Attributes:
        findings: List of validation findings.
        domains_present: Domain codes found in the package.
        domains_required: Always-required domain codes.
        domains_conditional: Conditionally-required domain codes
            (triggered by SoA content).
        passed: True if no Error-severity findings.
    """

    findings: list[DomainFinding] = dataclasses.field(
        default_factory=list,
    )
    domains_present: list[str] = dataclasses.field(
        default_factory=list,
    )
    domains_required: list[str] = dataclasses.field(
        default_factory=list,
    )
    domains_conditional: list[str] = dataclasses.field(
        default_factory=list,
    )

    @property
    def passed(self) -> bool:
        return not any(f.severity == "Error" for f in self.findings)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "Error")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "Warning")


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


def check_required_domains(
    domains_present: list[str],
    soa_assessments: Optional[list[str]] = None,
    study_type: str = "INTERVENTIONAL",
) -> DomainCheckResult:
    """Validate that required SDTM domains are present.

    Args:
        domains_present: List of domain codes in the SDTM package
            (e.g., ["DM", "TS", "TV", "TA", "TE"]).
        soa_assessments: Optional list of SoA assessment names
            (used to determine conditional domain requirements).
        study_type: "INTERVENTIONAL" or "OBSERVATIONAL".

    Returns:
        DomainCheckResult with findings and status.
    """
    present_set = set(d.upper() for d in domains_present)
    findings: list[DomainFinding] = []
    required_codes: list[str] = []
    conditional_codes: list[str] = []

    # Check always-required domains
    for code, name in ALWAYS_REQUIRED:
        required_codes.append(code)
        if code not in present_set:
            findings.append(DomainFinding(
                domain_code=code,
                domain_name=name,
                severity="Error",
                message=(
                    f"Required domain {code} ({name}) is missing. "
                    f"All interventional trials must include {code}."
                ),
                trigger="always required (SDTMIG v3.4)",
            ))

    # Check conditional domains based on SoA content
    if soa_assessments:
        assessments_lower = [a.lower() for a in soa_assessments]

        for code, name, triggers in CONDITIONAL_DOMAINS:
            triggered = any(
                any(kw in assess for kw in triggers)
                for assess in assessments_lower
            )

            if triggered:
                conditional_codes.append(code)
                if code not in present_set:
                    trigger_matches = [
                        kw for kw in triggers
                        if any(kw in a for a in assessments_lower)
                    ]
                    findings.append(DomainFinding(
                        domain_code=code,
                        domain_name=name,
                        severity="Warning",
                        message=(
                            f"Domain {code} ({name}) is missing but "
                            f"SoA includes related assessments. "
                            f"Consider adding {code}."
                        ),
                        trigger=(
                            f"SoA contains: "
                            f"{', '.join(trigger_matches[:3])}"
                        ),
                    ))

    return DomainCheckResult(
        findings=findings,
        domains_present=sorted(present_set),
        domains_required=required_codes,
        domains_conditional=conditional_codes,
    )


__all__ = [
    "ALWAYS_REQUIRED",
    "CONDITIONAL_DOMAINS",
    "DomainCheckResult",
    "DomainFinding",
    "check_required_domains",
]
