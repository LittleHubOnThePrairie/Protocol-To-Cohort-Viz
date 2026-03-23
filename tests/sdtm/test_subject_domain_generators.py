"""Tests for PTCV-245: Subject-Level SDTM Domain Spec Generators.

Tests verify DM, DS, AE, CM, MH domain specification generation,
variable completeness, controlled terminology references, and
protocol-specific enrichment from eligibility criteria.
"""

import json
import pytest
from unittest.mock import MagicMock

from ptcv.sdtm.subject_domain_generators import (
    DmSpecGenerator,
    DsSpecGenerator,
    AeSpecGenerator,
    CmSpecGenerator,
    MhSpecGenerator,
    generate_all_subject_domains,
)


def _make_section(code: str, text: str = "") -> MagicMock:
    """Create a mock IchSection."""
    s = MagicMock()
    s.section_code = code
    s.content_json = json.dumps({"text_excerpt": text})
    return s


def _make_sections(**kwargs: str) -> list[MagicMock]:
    """Create sections from code=text pairs."""
    return [_make_section(code, text) for code, text in kwargs.items()]


class TestDmSpecGenerator:
    """Tests for DM (Demographics) domain spec."""

    def test_produces_dataframe(self):
        gen = DmSpecGenerator()
        df, ct = gen.generate([], "STUDY01", "run-1")
        assert len(df) > 0
        assert "VARNAME" in df.columns
        assert "DOMAIN" in df.columns

    def test_required_variables_present(self):
        gen = DmSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        varnames = set(df["VARNAME"])
        for required in ["STUDYID", "DOMAIN", "USUBJID", "SUBJID", "AGE", "SEX", "RACE", "ARMCD", "ARM"]:
            assert required in varnames, f"Missing required variable: {required}"

    def test_domain_code_is_dm(self):
        gen = DmSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        assert all(df["DOMAIN"] == "DM")

    def test_studyid_populated(self):
        gen = DmSpecGenerator()
        df, _ = gen.generate([], "MY_STUDY", "run-1")
        assert all(df["STUDYID"] == "MY_STUDY")

    def test_sex_has_codelist(self):
        gen = DmSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        sex_row = df[df["VARNAME"] == "SEX"].iloc[0]
        assert sex_row["CODELIST"] == "SEX"

    def test_age_comment_from_eligibility(self):
        sections = _make_sections(**{"B.5": "Patients aged >= 18 years with confirmed diagnosis"})
        gen = DmSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        age_row = df[df["VARNAME"] == "AGE"].iloc[0]
        assert "18" in age_row["COMMENT"]

    def test_age_range_comment(self):
        sections = _make_sections(**{"B.5": "Age 18 to 65 years, male or female"})
        gen = DmSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        age_row = df[df["VARNAME"] == "AGE"].iloc[0]
        assert "18-65" in age_row["COMMENT"]


class TestDsSpecGenerator:
    """Tests for DS (Disposition) domain spec."""

    def test_produces_dataframe(self):
        gen = DsSpecGenerator()
        df, ct = gen.generate([], "STUDY01", "run-1")
        assert len(df) > 0

    def test_required_variables_present(self):
        gen = DsSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        varnames = set(df["VARNAME"])
        for required in ["STUDYID", "DOMAIN", "USUBJID", "DSSEQ", "DSTERM", "DSDECOD"]:
            assert required in varnames

    def test_domain_code_is_ds(self):
        gen = DsSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        assert all(df["DOMAIN"] == "DS")

    def test_has_epoch(self):
        gen = DsSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        assert "EPOCH" in set(df["VARNAME"])


class TestAeSpecGenerator:
    """Tests for AE (Adverse Events) domain spec."""

    def test_produces_dataframe(self):
        gen = AeSpecGenerator()
        df, ct = gen.generate([], "STUDY01", "run-1")
        assert len(df) > 0

    def test_required_variables_present(self):
        gen = AeSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        varnames = set(df["VARNAME"])
        for required in ["STUDYID", "DOMAIN", "USUBJID", "AESEQ", "AETERM", "AEDECOD"]:
            assert required in varnames

    def test_meddra_reference(self):
        gen = AeSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        aedecod_row = df[df["VARNAME"] == "AEDECOD"].iloc[0]
        assert "MedDRA" in aedecod_row["CODELIST"]

    def test_severity_and_seriousness(self):
        gen = AeSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        varnames = set(df["VARNAME"])
        assert "AESEV" in varnames
        assert "AESER" in varnames
        assert "AEREL" in varnames

    def test_timing_variables(self):
        gen = AeSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        varnames = set(df["VARNAME"])
        assert "AESTDTC" in varnames
        assert "AEENDTC" in varnames


class TestCmSpecGenerator:
    """Tests for CM (Concomitant Medications) domain spec."""

    def test_produces_dataframe(self):
        gen = CmSpecGenerator()
        df, ct = gen.generate([], "STUDY01", "run-1")
        assert len(df) > 0

    def test_required_variables_present(self):
        gen = CmSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        varnames = set(df["VARNAME"])
        for required in ["STUDYID", "DOMAIN", "USUBJID", "CMSEQ", "CMTRT"]:
            assert required in varnames

    def test_whodrug_reference(self):
        gen = CmSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        cmdecod_row = df[df["VARNAME"] == "CMDECOD"].iloc[0]
        assert "WHODrug" in cmdecod_row["CODELIST"]

    def test_dosing_variables(self):
        gen = CmSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        varnames = set(df["VARNAME"])
        assert "CMDOSE" in varnames
        assert "CMDOSU" in varnames
        assert "CMDOSFRQ" in varnames
        assert "CMROUTE" in varnames


class TestMhSpecGenerator:
    """Tests for MH (Medical History) domain spec."""

    def test_produces_dataframe(self):
        gen = MhSpecGenerator()
        df, ct = gen.generate([], "STUDY01", "run-1")
        assert len(df) > 0

    def test_required_variables_present(self):
        gen = MhSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        varnames = set(df["VARNAME"])
        for required in ["STUDYID", "DOMAIN", "USUBJID", "MHSEQ", "MHTERM"]:
            assert required in varnames

    def test_meddra_reference(self):
        gen = MhSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        mhdecod_row = df[df["VARNAME"] == "MHDECOD"].iloc[0]
        assert "MedDRA" in mhdecod_row["CODELIST"]

    def test_prespecified_variable(self):
        gen = MhSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        assert "MHPRESP" in set(df["VARNAME"])

    def test_categories_from_eligibility(self):
        sections = _make_sections(**{
            "B.5": "Patients with history of cardiovascular disease, "
                   "diabetes, or hepatic impairment are excluded. "
                   "No history of cancer or malignancy."
        })
        gen = MhSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        mhcat_row = df[df["VARNAME"] == "MHCAT"].iloc[0]
        assert "Cardiovascular" in mhcat_row["COMMENT"]
        assert "Hepatic" in mhcat_row["COMMENT"]
        assert "Endocrine" in mhcat_row["COMMENT"]  # diabetes
        assert "Oncology" in mhcat_row["COMMENT"]  # cancer

    def test_no_categories_without_eligibility(self):
        gen = MhSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        mhcat_row = df[df["VARNAME"] == "MHCAT"].iloc[0]
        assert "Protocol-defined categories:" not in mhcat_row["COMMENT"]


class TestExtractMhCategories:
    """Tests for MhSpecGenerator._extract_mh_categories."""

    def test_cardiovascular(self):
        cats = MhSpecGenerator._extract_mh_categories("history of cardiovascular disease")
        assert "Cardiovascular" in cats

    def test_multiple_categories(self):
        cats = MhSpecGenerator._extract_mh_categories(
            "No history of hepatic disease, renal failure, or psychiatric disorders"
        )
        assert "Hepatic" in cats
        assert "Renal" in cats
        assert "Psychiatric" in cats

    def test_allergy(self):
        cats = MhSpecGenerator._extract_mh_categories("Known allergy or hypersensitivity to study drug")
        assert "Allergy" in cats

    def test_empty_text(self):
        assert MhSpecGenerator._extract_mh_categories("") == []

    def test_no_matches(self):
        assert MhSpecGenerator._extract_mh_categories("Willing to provide informed consent") == []


class TestGenerateAllSubjectDomains:
    """Tests for generate_all_subject_domains convenience function."""

    def test_returns_all_5_domains(self):
        results = generate_all_subject_domains([], "STUDY01", "run-1")
        assert set(results.keys()) == {"DM", "DS", "AE", "CM", "MH"}

    def test_each_domain_has_dataframe(self):
        results = generate_all_subject_domains([], "STUDY01", "run-1")
        for domain_code, (df, ct) in results.items():
            assert len(df) > 0, f"{domain_code} produced empty DataFrame"
            assert all(df["DOMAIN"] == domain_code)

    def test_no_unmapped_ct(self):
        results = generate_all_subject_domains([], "STUDY01", "run-1")
        for domain_code, (df, ct) in results.items():
            assert ct == [], f"{domain_code} has unexpected unmapped CT"
