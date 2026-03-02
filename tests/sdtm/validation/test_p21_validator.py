"""Tests for P21Validator (PTCV-23)."""

from __future__ import annotations

import pandas as pd
import pytest

from ptcv.sdtm.validation.models import P21Issue
from ptcv.sdtm.validation.p21_validator import P21Validator


class TestP21ValidatorCleanData:
    """Validator produces no errors on valid input."""

    def test_valid_domains_no_errors(self, all_domains):
        issues = P21Validator(all_domains).validate()
        errors = [i for i in issues if i.severity == "Error"]
        assert errors == [], f"Unexpected errors: {errors}"

    def test_returns_list_of_p21issue(self, all_domains):
        issues = P21Validator(all_domains).validate()
        assert isinstance(issues, list)
        assert all(isinstance(i, P21Issue) for i in issues)


class TestRequiredVariableCheck:
    """P21-{DOMAIN}-002: Required variable missing."""

    def test_missing_required_var_ts(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 1.0,
            # TSPARMCD missing
            "TSPARM": "Trial Title",
            "TSVAL": "Phase III Study",
        }])
        issues = P21Validator({"TS": df}).validate()
        errors = [i for i in issues if i.severity == "Error"]
        assert any(i.variable == "TSPARMCD" for i in errors)

    def test_missing_required_var_ti(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TI",
            # IETESTCD missing
            "IETEST": "Age criterion",
            "IECAT": "INCLUSION",
        }])
        issues = P21Validator({"TI": df}).validate()
        errors = [i for i in issues if i.severity == "Error"]
        assert any(i.variable == "IETESTCD" for i in errors)


class TestDomainFieldCheck:
    """P21-{DOMAIN}-003: DOMAIN value must match dataset name."""

    def test_wrong_domain_value(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "XX",  # wrong — should be TS
            "TSSEQ": 1.0,
            "TSPARMCD": "TITLE",
            "TSPARM": "Trial Title",
            "TSVAL": "Test",
            "TSVALCD": "",
            "TSVALNF": "",
            "TSVALUNIT": "",
        }])
        issues = P21Validator({"TS": df}).validate()
        errors = [i for i in issues if i.variable == "DOMAIN" and i.domain == "TS"]
        assert errors, "Expected domain mismatch error"

    def test_correct_domain_value_no_error(self, ts_df):
        issues = P21Validator({"TS": ts_df}).validate()
        domain_errors = [
            i for i in issues
            if i.variable == "DOMAIN" and "does not match" in i.description
        ]
        assert domain_errors == []


class TestStudyIdCheck:
    """P21-{DOMAIN}-005: STUDYID must not exceed 20 characters."""

    def test_studyid_too_long(self, ts_df):
        ts_df = ts_df.copy()
        ts_df["STUDYID"] = "A" * 21
        issues = P21Validator({"TS": ts_df}).validate()
        errors = [i for i in issues if i.variable == "STUDYID"]
        assert errors, "Expected STUDYID length error"
        # At least one finding must be an Error (Warning from var_lengths check is also OK)
        assert any(i.severity == "Error" for i in errors)

    def test_studyid_exactly_20_no_error(self, ts_df):
        ts_df = ts_df.copy()
        ts_df["STUDYID"] = "A" * 20
        issues = P21Validator({"TS": ts_df}).validate()
        studyid_errors = [i for i in issues if i.variable == "STUDYID"]
        assert studyid_errors == []


class TestTsSpecificRules:
    """TS-specific P21 rules."""

    def test_tsseq_positive(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": -1.0,  # invalid
            "TSPARMCD": "TITLE",
            "TSPARM": "Trial Title",
            "TSVAL": "Test",
            "TSVALCD": "",
            "TSVALNF": "",
            "TSVALUNIT": "",
        }])
        issues = P21Validator({"TS": df}).validate()
        errors = [i for i in issues if "TSSEQ" in i.variable and i.severity == "Error"]
        assert errors

    def test_tsparmcd_alphanumeric(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 1.0,
            "TSPARMCD": "TITLE!",  # invalid char
            "TSPARM": "Trial Title",
            "TSVAL": "Test",
            "TSVALCD": "",
            "TSVALNF": "",
            "TSVALUNIT": "",
        }])
        issues = P21Validator({"TS": df}).validate()
        errors = [i for i in issues if i.variable == "TSPARMCD"]
        assert errors

    def test_tsval_blank_no_tsvalnf_is_warning(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 1.0,
            "TSPARMCD": "TITLE",
            "TSPARM": "Trial Title",
            "TSVAL": "",  # blank
            "TSVALCD": "",
            "TSVALNF": "",  # no null-flavor
            "TSVALUNIT": "",
        }])
        issues = P21Validator({"TS": df}).validate()
        warnings = [i for i in issues if i.variable == "TSVAL" and i.severity == "Warning"]
        assert warnings

    def test_tsval_blank_with_tsvalnf_ok(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 1.0,
            "TSPARMCD": "TITLE",
            "TSPARM": "Trial Title",
            "TSVAL": "",
            "TSVALCD": "",
            "TSVALNF": "PINF",  # null-flavor provided
            "TSVALUNIT": "",
        }])
        issues = P21Validator({"TS": df}).validate()
        warnings = [i for i in issues if i.variable == "TSVAL"]
        assert warnings == []


class TestTaSpecificRules:
    """TA-specific P21 rules."""

    def test_taetord_positive(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TA",
            "ARMCD": "DRUG",
            "ARM": "Drug Arm",
            "TAETORD": 0.0,  # invalid — must be > 0
            "ETCD": "SCRN",
            "ELEMENT": "Screening",
        }])
        issues = P21Validator({"TA": df}).validate()
        errors = [i for i in issues if i.variable == "TAETORD"]
        assert errors

    def test_taetord_unique_per_armcd(self):
        df = pd.DataFrame([
            {
                "STUDYID": "NCT001", "DOMAIN": "TA",
                "ARMCD": "DRUG", "ARM": "Drug", "TAETORD": 1.0,
                "ETCD": "SCRN", "ELEMENT": "Screening",
            },
            {
                "STUDYID": "NCT001", "DOMAIN": "TA",
                "ARMCD": "DRUG", "ARM": "Drug", "TAETORD": 1.0,  # dup
                "ETCD": "TRT", "ELEMENT": "Treatment",
            },
        ])
        issues = P21Validator({"TA": df}).validate()
        errors = [i for i in issues if i.variable == "TAETORD"]
        assert errors


class TestTeSpecificRules:
    """TE-specific P21 rules."""

    def test_etcd_unique(self):
        df = pd.DataFrame([
            {
                "STUDYID": "NCT001", "DOMAIN": "TE",
                "ETCD": "SCRN", "ELEMENT": "Screening",
            },
            {
                "STUDYID": "NCT001", "DOMAIN": "TE",
                "ETCD": "SCRN",  # duplicate
                "ELEMENT": "Screening v2",
            },
        ])
        issues = P21Validator({"TE": df}).validate()
        errors = [i for i in issues if i.variable == "ETCD"]
        assert errors


class TestTvSpecificRules:
    """TV-specific P21 rules."""

    def test_visitnum_must_be_positive(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001", "DOMAIN": "TV",
            "VISITNUM": -1.0, "VISIT": "Screening",
        }])
        issues = P21Validator({"TV": df}).validate()
        errors = [i for i in issues if i.variable == "VISITNUM"]
        assert errors

    def test_visitnum_unique(self):
        df = pd.DataFrame([
            {
                "STUDYID": "NCT001", "DOMAIN": "TV",
                "VISITNUM": 1.0, "VISIT": "Screening",
            },
            {
                "STUDYID": "NCT001", "DOMAIN": "TV",
                "VISITNUM": 1.0, "VISIT": "Baseline",  # dup VISITNUM
            },
        ])
        issues = P21Validator({"TV": df}).validate()
        errors = [i for i in issues if i.variable == "VISITNUM"]
        assert errors


class TestTiSpecificRules:
    """TI-specific P21 rules."""

    def test_iecat_invalid_value(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001", "DOMAIN": "TI",
            "IETESTCD": "IE001I", "IETEST": "Age criterion",
            "IECAT": "MAYBE",  # invalid
        }])
        issues = P21Validator({"TI": df}).validate()
        errors = [i for i in issues if i.variable == "IECAT"]
        assert errors

    def test_ietestcd_too_long(self):
        df = pd.DataFrame([{
            "STUDYID": "NCT001", "DOMAIN": "TI",
            "IETESTCD": "TOOLONGCD",  # 9 chars — exceeds 8
            "IETEST": "Age criterion", "IECAT": "INCLUSION",
        }])
        issues = P21Validator({"TI": df}).validate()
        errors = [i for i in issues if i.variable == "IETESTCD"]
        assert errors


class TestEmptyDataset:
    """P21-{DOMAIN}-001: Empty dataset raises Error."""

    def test_empty_ts_is_error(self):
        df = pd.DataFrame(columns=["STUDYID", "DOMAIN", "TSSEQ", "TSPARMCD", "TSPARM", "TSVAL"])
        issues = P21Validator({"TS": df}).validate()
        errors = [i for i in issues if i.severity == "Error" and i.domain == "TS"]
        assert errors

    def test_empty_tv_is_error(self):
        df = pd.DataFrame(columns=["STUDYID", "DOMAIN", "VISITNUM", "VISIT"])
        issues = P21Validator({"TV": df}).validate()
        errors = [i for i in issues if i.severity == "Error" and i.domain == "TV"]
        assert errors


class TestSortOrder:
    """Validator returns Errors before Warnings."""

    def test_errors_before_warnings(self, ts_df):
        # Inject an error by corrupting TSPARMCD
        bad_ts = ts_df.copy()
        bad_ts.at[0, "TSPARMCD"] = "TITLE!"  # error
        bad_ts.at[1, "TSVAL"] = ""            # warning
        issues = P21Validator({"TS": bad_ts}).validate()
        severities = [i.severity for i in issues]
        error_pos = next((i for i, s in enumerate(severities) if s == "Error"), None)
        warning_pos = next((i for i, s in enumerate(severities) if s == "Warning"), None)
        if error_pos is not None and warning_pos is not None:
            assert error_pos < warning_pos, "Errors must come before Warnings"
