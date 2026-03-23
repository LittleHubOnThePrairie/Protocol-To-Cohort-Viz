"""Tests for validation results panel (PTCV-272).

Tests data structure compatibility for domain checklist and
validation results rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.sdtm.validation.required_domain_checker import (
    DomainCheckResult,
    DomainFinding,
)
from ptcv.sdtm.validation.models import (
    P21Issue,
    TcgParameter,
    ValidationResult,
)


class TestDomainChecklistData:
    """Verify domain checklist data structures."""

    def test_passing_checklist(self):
        result = DomainCheckResult(
            findings=[],
            domains_present=["DM", "DS", "AE", "TA", "TE", "TI", "TS", "TV"],
            domains_required=["DM", "DS", "AE", "TA", "TE", "TI", "TS", "TV"],
        )
        assert result.passed
        assert result.error_count == 0

    def test_failing_checklist(self):
        result = DomainCheckResult(
            findings=[
                DomainFinding("DM", "Demographics", "Error", "Missing DM"),
                DomainFinding("VS", "Vital Signs", "Warning", "Missing VS"),
            ],
        )
        assert not result.passed
        assert result.error_count == 1
        assert result.warning_count == 1

    def test_finding_has_required_fields(self):
        f = DomainFinding(
            domain_code="DM",
            domain_name="Demographics",
            severity="Error",
            message="Required domain DM is missing.",
            trigger="always required (SDTMIG v3.4)",
        )
        assert f.domain_code == "DM"
        assert f.severity == "Error"
        assert f.trigger != ""


class TestValidationResultData:
    """Verify validation result data structures for rendering."""

    def test_p21_issues_by_severity(self):
        issues = [
            P21Issue("P21-TS-001", "Error", "TS", "TSVAL", "desc", "fix"),
            P21Issue("P21-TA-001", "Warning", "TA", "ARMCD", "desc", "fix"),
            P21Issue("P21-TV-001", "Notice", "TV", "VISITNUM", "desc", "fix"),
        ]
        errors = [i for i in issues if i.severity == "Error"]
        warnings = [i for i in issues if i.severity == "Warning"]
        notices = [i for i in issues if i.severity == "Notice"]
        assert len(errors) == 1
        assert len(warnings) == 1
        assert len(notices) == 1

    def test_p21_issue_has_remediation(self):
        issue = P21Issue(
            rule_id="P21-TS-010",
            severity="Error",
            domain="TS",
            variable="TSSEQ",
            description="TSSEQ must be positive",
            remediation_guidance="See SDTMIG v3.3 §7.4.1",
        )
        assert "SDTMIG" in issue.remediation_guidance

    def test_tcg_missing_params(self):
        params = [
            TcgParameter("TITLE", "Trial Title", "FDA TCG v5.9", True, False),
            TcgParameter("PHASE", "Trial Phase", "FDA TCG v5.9", False, True),
        ]
        missing = [p.tsparmcd for p in params if p.missing]
        assert missing == ["PHASE"]

    def test_validation_result_summary(self):
        result = ValidationResult(
            run_id="val-1",
            registry_id="NCT001",
            sdtm_run_id="sdtm-1",
            sdtm_sha256="abc",
            p21_issues=[
                P21Issue("P21-TS-001", "Error", "TS", "TSVAL", "d", "r"),
            ],
            p21_error_count=1,
            p21_warning_count=0,
            tcg_parameters=[],
            tcg_passed=False,
            tcg_missing_params=["PHASE", "STYPE"],
            define_xml_issues=[],
            artifact_keys={},
            artifact_sha256s={},
            validation_timestamp_utc="2026-03-22T00:00:00",
            schedule_feasible=True,
        )
        assert result.p21_error_count == 1
        assert not result.tcg_passed
        assert len(result.tcg_missing_params) == 2
        assert result.schedule_feasible


class TestModuleImport:
    def test_import_render_functions(self):
        from ptcv.ui.components.validation_results_panel import (
            render_domain_checklist,
            render_validation_results,
        )
        assert callable(render_domain_checklist)
        assert callable(render_validation_results)
