"""Tests for required domain checker (PTCV-249).

Tests always-required domain validation, conditional domain
detection from SoA, and complete package passing.
"""

from __future__ import annotations

import pytest

from ptcv.sdtm.validation.required_domain_checker import (
    ALWAYS_REQUIRED,
    CONDITIONAL_DOMAINS,
    DomainCheckResult,
    DomainFinding,
    check_required_domains,
)


class TestAlwaysRequired:
    def test_missing_dm_is_error(self):
        result = check_required_domains(
            domains_present=["TS", "TV", "TA", "TE", "TI", "DS", "AE"],
        )
        errors = [f for f in result.findings if f.severity == "Error"]
        dm_errors = [f for f in errors if f.domain_code == "DM"]
        assert len(dm_errors) == 1
        assert "required" in dm_errors[0].message.lower()

    def test_all_required_present_no_errors(self):
        all_codes = [code for code, _ in ALWAYS_REQUIRED]
        result = check_required_domains(domains_present=all_codes)
        assert result.error_count == 0
        assert result.passed

    def test_empty_package_all_errors(self):
        result = check_required_domains(domains_present=[])
        assert result.error_count == len(ALWAYS_REQUIRED)
        assert not result.passed

    def test_case_insensitive(self):
        all_codes = [code.lower() for code, _ in ALWAYS_REQUIRED]
        result = check_required_domains(domains_present=all_codes)
        assert result.error_count == 0

    def test_each_required_domain_checked(self):
        """Test all 8 always-required domains are checked."""
        expected_codes = {code for code, _ in ALWAYS_REQUIRED}
        assert expected_codes == {"DM", "DS", "AE", "TA", "TE", "TI", "TS", "TV"}


class TestConditionalDomains:
    def test_vs_triggered_by_vitals(self):
        result = check_required_domains(
            domains_present=["DM", "TS", "TV", "TA", "TE", "TI", "DS", "AE"],
            soa_assessments=["Vital Signs", "ECG"],
        )
        warnings = [f for f in result.findings if f.severity == "Warning"]
        vs_warnings = [f for f in warnings if f.domain_code == "VS"]
        assert len(vs_warnings) == 1
        assert "VS" in result.domains_conditional

    def test_lb_triggered_by_hematology(self):
        result = check_required_domains(
            domains_present=[],
            soa_assessments=["Hematology", "Chemistry"],
        )
        lb_findings = [
            f for f in result.findings
            if f.domain_code == "LB" and f.severity == "Warning"
        ]
        assert len(lb_findings) == 1

    def test_eg_triggered_by_ecg(self):
        result = check_required_domains(
            domains_present=[],
            soa_assessments=["12-lead ECG"],
        )
        eg_findings = [
            f for f in result.findings
            if f.domain_code == "EG"
        ]
        assert len(eg_findings) == 1

    def test_no_conditional_without_soa(self):
        result = check_required_domains(
            domains_present=["DM", "TS", "TV", "TA", "TE", "TI", "DS", "AE"],
        )
        warnings = [f for f in result.findings if f.severity == "Warning"]
        assert len(warnings) == 0

    def test_conditional_present_no_warning(self):
        """Test no warning when conditional domain IS present."""
        result = check_required_domains(
            domains_present=["DM", "TS", "TV", "TA", "TE", "TI", "DS", "AE", "VS"],
            soa_assessments=["Vital Signs"],
        )
        vs_findings = [
            f for f in result.findings if f.domain_code == "VS"
        ]
        assert len(vs_findings) == 0

    def test_trigger_text_included(self):
        result = check_required_domains(
            domains_present=[],
            soa_assessments=["Blood Pressure", "Heart Rate"],
        )
        vs_findings = [
            f for f in result.findings if f.domain_code == "VS"
        ]
        assert len(vs_findings) == 1
        assert "blood pressure" in vs_findings[0].trigger.lower()


class TestCompletePackage:
    def test_complete_package_passes(self):
        all_required = [code for code, _ in ALWAYS_REQUIRED]
        extra = ["VS", "LB", "EG", "PE", "EX", "CM"]
        result = check_required_domains(
            domains_present=all_required + extra,
            soa_assessments=[
                "Vital Signs", "Hematology", "ECG",
                "Physical Exam", "Drug X 100mg",
                "Concomitant Medications",
            ],
        )
        assert result.passed
        assert result.error_count == 0
        assert result.warning_count == 0


class TestDomainCheckResult:
    def test_passed_property(self):
        result = DomainCheckResult(findings=[
            DomainFinding("VS", "Vital Signs", "Warning", "msg"),
        ])
        assert result.passed  # Warnings don't block

    def test_not_passed_with_error(self):
        result = DomainCheckResult(findings=[
            DomainFinding("DM", "Demographics", "Error", "msg"),
        ])
        assert not result.passed

    def test_counts(self):
        result = DomainCheckResult(findings=[
            DomainFinding("DM", "Demographics", "Error", "msg"),
            DomainFinding("VS", "Vital Signs", "Warning", "msg"),
            DomainFinding("LB", "Labs", "Warning", "msg"),
        ])
        assert result.error_count == 1
        assert result.warning_count == 2
