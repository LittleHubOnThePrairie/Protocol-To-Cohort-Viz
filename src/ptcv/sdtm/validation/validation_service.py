"""ValidationService — SDTM validation and compliance reporting (PTCV-23).

Orchestrates the full validation pipeline for one SDTM generation run:
  1. Read XPT artifacts from StorageGateway by key
  2. Deserialize each XPT to a pandas DataFrame via pyreadstat
  3. Read define.xml from StorageGateway
  4. Run P21Validator (Pinnacle 21 rules)
  5. Run TcgChecker (FDA TCG v5.9 Appendix B)
  6. Run DefineXmlValidator (structural + codelist checks)
  7. Build three JSON compliance reports
  8. Write reports to StorageGateway (immutable=False — versioned)
  9. Append one LineageRecord per report (stage="validation")
  10. Return ValidationResult

Storage layout (under gateway root):
  validation-reports/{registry_id}/{run_id}/p21_report.json
  validation-reports/{registry_id}/{run_id}/tcg_completeness.json
  validation-reports/{registry_id}/{run_id}/compliance_summary.json

Risk tier: MEDIUM — regulatory submission validation artefacts; no patient data.

Regulatory references:
- ALCOA+ Traceable: source_hash in LineageRecord links each report to the
  SDTM generation run sha256 (ts.xpt sha256 used as representative)
- ALCOA+ Contemporaneous: validation_timestamp_utc captured at write boundary
- 21 CFR 11.10(e): audit trail via StorageGateway LineageRecord chain
- Retention: 25 years per EU CTR 536/2014 (co-retention with SDTM datasets)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from ...storage import StorageGateway
from .define_xml_validator import DefineXmlValidator
from .models import P21Issue, ValidationResult
from .p21_validator import P21Validator
from .tcg_checker import TcgChecker

if TYPE_CHECKING:
    from ..models import SdtmGenerationResult


logger = logging.getLogger(__name__)

_USER = "ptcv-validation-service"


def _xpt_bytes_to_df(xpt_bytes: bytes) -> pd.DataFrame:
    """Deserialize SAS XPT bytes to a DataFrame via pyreadstat.

    Args:
        xpt_bytes: Raw bytes of a SAS XPT v5 file.

    Returns:
        DataFrame with columns matching the XPT variables.
    """
    import pyreadstat  # type: ignore[import-untyped]

    fd, tmp_path = tempfile.mkstemp(suffix=".xpt")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(xpt_bytes)
        df, _ = pyreadstat.read_xport(tmp_path)
        return df
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _build_p21_report(
    issues: list[P21Issue],
    registry_id: str,
    sdtm_run_id: str,
    timestamp: str,
) -> bytes:
    """Serialize P21 issues and define.xml issues to JSON bytes."""
    payload = {
        "report_type": "p21_validation",
        "registry_id": registry_id,
        "sdtm_run_id": sdtm_run_id,
        "timestamp_utc": timestamp,
        "total_issues": len(issues),
        "error_count": sum(1 for i in issues if i.severity == "Error"),
        "warning_count": sum(1 for i in issues if i.severity == "Warning"),
        "notice_count": sum(1 for i in issues if i.severity == "Notice"),
        "issues": [dataclasses.asdict(i) for i in issues],
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def _build_tcg_report(
    parameters: list,
    passed: bool,
    missing: list[str],
    registry_id: str,
    sdtm_run_id: str,
    timestamp: str,
) -> bytes:
    """Serialize TCG completeness results to JSON bytes."""
    payload = {
        "report_type": "tcg_completeness",
        "regulatory_basis": "FDA TCG v5.9 Appendix B",
        "registry_id": registry_id,
        "sdtm_run_id": sdtm_run_id,
        "timestamp_utc": timestamp,
        "passed": passed,
        "missing_params": missing,
        "total_required": len(parameters),
        "total_present": sum(1 for p in parameters if p.present),
        "parameters": [dataclasses.asdict(p) for p in parameters],
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def _build_compliance_summary(
    p21_issues: list[P21Issue],
    missing_params: list[str],
    define_issues: list,
    registry_id: str,
    sdtm_run_id: str,
    timestamp: str,
    format_verdict: str = "NON_ICH",
    format_confidence: float = 0.0,
    sections_detected: int = 0,
    missing_required_sections: Optional[list[str]] = None,
) -> bytes:
    """Build human-readable compliance summary with remediation guidance."""
    all_issues = []

    for issue in p21_issues:
        all_issues.append({
            "source": "Pinnacle 21",
            "rule_id": issue.rule_id,
            "severity": issue.severity,
            "domain": issue.domain,
            "variable": issue.variable,
            "description": issue.description,
            "remediation_guidance": issue.remediation_guidance,
        })

    for parmcd in missing_params:
        all_issues.append({
            "source": "FDA TCG v5.9",
            "rule_id": f"TCG-TS-{parmcd}",
            "severity": "Error",
            "domain": "TS",
            "variable": "TSPARMCD",
            "description": (
                f"Required TS parameter {parmcd!r} is missing from the "
                "TS dataset."
            ),
            "remediation_guidance": (
                f"Add a row to TS with TSPARMCD='{parmcd}' and an "
                "appropriate TSVAL. See FDA TCG v5.9 Appendix B and "
                "SDTMIG v3.3 §7.4.1."
            ),
        })

    for di in define_issues:
        all_issues.append({
            "source": "Define-XML",
            "rule_id": f"DEFXML-{di.issue_type.upper()[:20]}",
            "severity": "Error",
            "domain": di.domain,
            "variable": di.variable,
            "description": di.description,
            "remediation_guidance": (
                "Reconcile define.xml variable declarations with the "
                "actual XPT column set. See CDISC Define-XML v2.1 §4.2 "
                "and SDTMIG v3.3 §7.4."
            ),
        })

    _verdict_recommendation = {
        "ICH_E6R3": (
            "Protocol conforms to ICH E6(R3) Appendix B structure. "
            "SDTM generation proceeded normally."
        ),
        "PARTIAL_ICH": (
            "Protocol partially conforms to ICH E6(R3) Appendix B. "
            "Some required sections may be missing or low-confidence. "
            "Review SDTM output for completeness."
        ),
        "NON_ICH": (
            "Protocol does not appear to follow ICH E6(R3) Appendix B "
            "structure. SDTM output may be incomplete. Manual review required."
        ),
    }

    payload = {
        "report_type": "compliance_summary",
        "registry_id": registry_id,
        "sdtm_run_id": sdtm_run_id,
        "timestamp_utc": timestamp,
        "total_issues": len(all_issues),
        "error_count": sum(
            1 for i in all_issues if i.get("severity") == "Error"
        ),
        "warning_count": sum(
            1 for i in all_issues if i.get("severity") == "Warning"
        ),
        "issues": all_issues,
        "format_assessment": {
            "verdict": format_verdict,
            "format_confidence": format_confidence,
            "sections_detected": sections_detected,
            "missing_required_sections": (
                missing_required_sections
                if missing_required_sections is not None
                else []
            ),
            "recommendation": _verdict_recommendation.get(
                format_verdict, _verdict_recommendation["NON_ICH"]
            ),
        },
    }
    return json.dumps(payload, indent=2).encode("utf-8")


class ValidationService:
    """Orchestrates SDTM validation and compliance report generation.

    Args:
        gateway: StorageGateway instance to read SDTM artifacts from
            and write validation reports to.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway

    def validate(
        self,
        sdtm_result: "SdtmGenerationResult",
        validation_run_id: Optional[str] = None,
        format_verdict: str = "NON_ICH",
        format_confidence: float = 0.0,
        sections_detected: int = 0,
        missing_required_sections: Optional[list[str]] = None,
    ) -> ValidationResult:
        """Run the full validation pipeline for one SDTM generation run.

        Args:
            sdtm_result: SdtmGenerationResult from SdtmService.generate().
                Provides artifact_keys, artifact_sha256s, run_id,
                registry_id, and source_sha256.
            validation_run_id: Optional UUID4 for this validation run.
                Defaults to a fresh UUID4.
            format_verdict: ICH format verdict from IchParser
                (``"ICH_E6R3"``, ``"PARTIAL_ICH"``, or ``"NON_ICH"``).
            format_confidence: Format confidence score in [0.0, 1.0].
            sections_detected: Number of ICH sections classified.
            missing_required_sections: Required ICH section codes absent
                from the classified sections.

        Returns:
            ValidationResult with all findings and report artifact metadata.
        [PTCV-23 Scenario: Run Pinnacle 21 and archive report with lineage]
        [PTCV-23 Scenario: Report linked to specific SDTM version]
        [PTCV-30 Scenario: Format verdict surfaced in compliance report]
        """
        val_run_id = validation_run_id or str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        registry_id = sdtm_result.registry_id
        sdtm_run_id = sdtm_result.run_id

        # ------------------------------------------------------------------
        # Step 1: Load XPT datasets
        # ------------------------------------------------------------------
        domains_df: dict[str, pd.DataFrame] = {}
        for domain_key in ["ts", "ta", "te", "tv", "ti"]:
            artifact_key = sdtm_result.artifact_keys.get(domain_key)
            if not artifact_key:
                continue
            try:
                xpt_bytes = self._gateway.get_artifact(artifact_key)
                domains_df[domain_key.upper()] = _xpt_bytes_to_df(xpt_bytes)
                logger.debug("Loaded %s.xpt (%d bytes)", domain_key, len(xpt_bytes))
            except Exception as exc:
                logger.warning("Could not read %s: %s", artifact_key, exc)

        # Representative SDTM SHA-256 (ts.xpt; fallback to source_sha256)
        sdtm_sha256 = sdtm_result.artifact_sha256s.get(
            "ts", sdtm_result.source_sha256
        )

        # ------------------------------------------------------------------
        # Step 2: Load define.xml
        # ------------------------------------------------------------------
        define_xml_bytes: bytes = b""
        define_key = sdtm_result.artifact_keys.get("define")
        if define_key:
            try:
                define_xml_bytes = self._gateway.get_artifact(define_key)
            except Exception as exc:
                logger.warning("Could not read define.xml: %s", exc)

        # ------------------------------------------------------------------
        # Step 3: Run P21 validator
        # ------------------------------------------------------------------
        p21_issues = P21Validator(domains_df).validate()

        # ------------------------------------------------------------------
        # Step 4: Run TCG checker
        # ------------------------------------------------------------------
        ts_df = domains_df.get("TS", pd.DataFrame())
        tcg_params, tcg_passed, tcg_missing = TcgChecker(ts_df).check()

        # ------------------------------------------------------------------
        # Step 5: Run Define-XML validator (if define.xml is available)
        # ------------------------------------------------------------------
        define_issues: list = []
        if define_xml_bytes:
            define_issues = DefineXmlValidator(
                define_xml_bytes, domains_df
            ).validate()

            # Merge define_xml_issues into p21_report as per GHERKIN
            from .models import P21Issue as _P21Issue
            for di in define_issues:
                p21_issues.append(
                    _P21Issue(
                        rule_id=f"P21-DEFXML-{di.issue_type.upper()[:12]}",
                        severity="Error",
                        domain=di.domain,
                        variable=di.variable,
                        description=di.description,
                        remediation_guidance=(
                            "Reconcile define.xml declarations with XPT columns. "
                            "See CDISC Define-XML v2.1 §4.2."
                        ),
                    )
                )

        p21_error_count = sum(1 for i in p21_issues if i.severity == "Error")
        p21_warning_count = sum(
            1 for i in p21_issues if i.severity == "Warning"
        )

        # ------------------------------------------------------------------
        # Step 6: Build report JSON bytes
        # ------------------------------------------------------------------
        p21_bytes = _build_p21_report(
            p21_issues, registry_id, sdtm_run_id, timestamp
        )
        tcg_bytes = _build_tcg_report(
            tcg_params, tcg_passed, tcg_missing, registry_id, sdtm_run_id,
            timestamp,
        )
        summary_bytes = _build_compliance_summary(
            p21_issues, tcg_missing, define_issues,
            registry_id, sdtm_run_id, timestamp,
            format_verdict=format_verdict,
            format_confidence=format_confidence,
            sections_detected=sections_detected,
            missing_required_sections=missing_required_sections,
        )

        # ------------------------------------------------------------------
        # Step 7: Write reports to StorageGateway
        # ------------------------------------------------------------------
        key_prefix = (
            f"validation-reports/{registry_id}/{sdtm_run_id}"
        )

        # Source metadata for LineageRecord.source field
        lineage_meta = json.dumps({
            "p21_error_count": p21_error_count,
            "p21_warning_count": p21_warning_count,
            "tcg_passed": tcg_passed,
            "validation_run_id": val_run_id,
        })

        artifact_keys: dict[str, str] = {}
        artifact_sha256s: dict[str, str] = {}

        for report_name, report_bytes, report_tag in [
            ("p21_report.json", p21_bytes, "p21"),
            ("tcg_completeness.json", tcg_bytes, "tcg"),
            ("compliance_summary.json", summary_bytes, "summary"),
        ]:
            key = f"{key_prefix}/{report_name}"
            artifact = self._gateway.put_artifact(
                key=key,
                data=report_bytes,
                content_type="application/json",
                run_id=val_run_id,
                source_hash=sdtm_sha256,
                user=_USER,
                immutable=False,
                stage="validation",
                registry_id=registry_id,
                source=lineage_meta,
            )
            artifact_keys[report_tag] = artifact.key
            artifact_sha256s[report_tag] = artifact.sha256

        return ValidationResult(
            run_id=val_run_id,
            registry_id=registry_id,
            sdtm_run_id=sdtm_run_id,
            sdtm_sha256=sdtm_sha256,
            p21_issues=p21_issues,
            p21_error_count=p21_error_count,
            p21_warning_count=p21_warning_count,
            tcg_parameters=tcg_params,
            tcg_passed=tcg_passed,
            tcg_missing_params=tcg_missing,
            define_xml_issues=define_issues,
            artifact_keys=artifact_keys,
            artifact_sha256s=artifact_sha256s,
            validation_timestamp_utc=timestamp,
        )
