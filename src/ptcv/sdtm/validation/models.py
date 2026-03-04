"""Data models for SDTM validation and compliance reporting (PTCV-23).

Covers Pinnacle 21, FDA TCG Appendix B, and Define-XML structural checks.

Risk tier: MEDIUM — regulatory submission validation artefacts; no patient data.

Regulatory references:
- ALCOA+ Traceable: sdtm_sha256 links each report to the upstream SDTM run
- 21 CFR 11.10(e): audit trail via StorageGateway LineageRecord chain
- FDA TCG v5.9 (October 2024): required TS parameter list
- SDTMIG v3.3: variable and domain specification
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .schedule_validator import ScheduleIssue


@dataclasses.dataclass
class P21Issue:
    """One Pinnacle 21 validation finding.

    Attributes:
        rule_id: Pinnacle 21 rule identifier (e.g. "P21-TS-001").
        severity: "Error", "Warning", or "Notice".
        domain: SDTM domain name (e.g. "TS").
        variable: Variable within the domain (e.g. "TSVAL").
        description: Human-readable description of the violation.
        remediation_guidance: Reference to the relevant SDTMIG v3.3 section
            and suggested fix.
        row_number: Optional 1-based row index within the domain dataset
            where the issue was found. None if domain-level.
    """

    rule_id: str
    severity: str
    domain: str
    variable: str
    description: str
    remediation_guidance: str
    row_number: Optional[int] = None


@dataclasses.dataclass
class TcgParameter:
    """One FDA TCG v5.9 Appendix B required TS parameter.

    Attributes:
        tsparmcd: Parameter code (max 8 chars, e.g. "STYPE").
        tsparm: Full parameter name (e.g. "Study Type").
        required_by: Regulatory basis, always "FDA TCG v5.9 Appendix B".
        present: True when the parameter is found in the TS dataset.
        missing: True when the parameter is absent (complement of present).
    """

    tsparmcd: str
    tsparm: str
    required_by: str
    present: bool
    missing: bool


@dataclasses.dataclass
class DefineXmlIssue:
    """One structural mismatch or codelist reference error in define.xml.

    Attributes:
        issue_type: "structural_mismatch" (variable absent from XPT) or
            "codelist_ref" (codelist OID not found in CDISC CT).
        domain: SDTM domain affected (e.g. "TS").
        variable: Variable name involved.
        description: Human-readable description.
    """

    issue_type: str
    domain: str
    variable: str
    description: str


@dataclasses.dataclass
class ValidationResult:
    """Full result from ValidationService.validate().

    Attributes:
        run_id: UUID4 for this validation run.
        registry_id: Trial registry identifier validated.
        sdtm_run_id: run_id of the upstream SDTM generation run.
        sdtm_sha256: SHA-256 of the representative SDTM artifact (ts.xpt)
            used as source_hash in the LineageRecord.
        p21_issues: All Pinnacle 21 findings.
        p21_error_count: Count of P21Issue with severity == "Error".
        p21_warning_count: Count of P21Issue with severity == "Warning".
        tcg_parameters: All TCG Appendix B parameters checked.
        tcg_passed: True when no required parameters are missing.
        tcg_missing_params: TSPARMCD values absent from the TS dataset.
        define_xml_issues: All Define-XML structural and codelist issues.
        artifact_keys: Mapping from report type to storage key.
        artifact_sha256s: Mapping from report type to SHA-256 of the report.
        validation_timestamp_utc: ISO 8601 UTC timestamp of the validation.
    """

    run_id: str
    registry_id: str
    sdtm_run_id: str
    sdtm_sha256: str
    p21_issues: list[P21Issue]
    p21_error_count: int
    p21_warning_count: int
    tcg_parameters: list[TcgParameter]
    tcg_passed: bool
    tcg_missing_params: list[str]
    define_xml_issues: list[DefineXmlIssue]
    artifact_keys: dict[str, str]
    artifact_sha256s: dict[str, str]
    validation_timestamp_utc: str

    # PTCV-57: Visit schedule feasibility validation
    schedule_issues: list["ScheduleIssue"] = dataclasses.field(
        default_factory=list,
    )
    schedule_error_count: int = 0
    schedule_warning_count: int = 0
    schedule_feasible: bool = True

    def __repr__(self) -> str:
        return (
            f"ValidationResult(run_id={self.run_id!r}, "
            f"registry_id={self.registry_id!r}, "
            f"p21_errors={self.p21_error_count}, "
            f"tcg_passed={self.tcg_passed})"
        )
