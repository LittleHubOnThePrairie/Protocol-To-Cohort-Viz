"""Tests for PTCV-247: IE Domain Generator.

Tests verify criterion parsing, pattern matching, data collection
variable mapping, and domain output structure.
"""

import json
import pytest
from unittest.mock import MagicMock

from ptcv.sdtm.ie_domain_generator import (
    IeSpecGenerator,
    _parse_criteria,
    _match_criterion,
)


def _make_section(code: str, text: str = "") -> MagicMock:
    s = MagicMock()
    s.section_code = code
    s.content_json = json.dumps({"text_excerpt": text})
    return s


class TestParseCriteria:
    """Tests for _parse_criteria helper."""

    def test_inclusion_criteria(self):
        text = "Inclusion Criteria:\n1. Age >= 18 years\n2. Signed consent"
        criteria = _parse_criteria(text)
        assert len(criteria) == 2
        assert criteria[0] == ("Age >= 18 years", "INCLUSION")
        assert criteria[1] == ("Signed consent", "INCLUSION")

    def test_exclusion_criteria(self):
        text = "Exclusion Criteria:\n1. Pregnant or lactating\n2. Prior chemotherapy"
        criteria = _parse_criteria(text)
        assert len(criteria) == 2
        assert all(cat == "EXCLUSION" for _, cat in criteria)

    def test_mixed_criteria(self):
        text = (
            "Inclusion Criteria:\n"
            "1. Age >= 18 years\n"
            "2. Confirmed diagnosis\n"
            "Exclusion Criteria:\n"
            "1. Pregnant\n"
            "2. History of cancer\n"
        )
        criteria = _parse_criteria(text)
        assert len(criteria) == 4
        assert criteria[0][1] == "INCLUSION"
        assert criteria[1][1] == "INCLUSION"
        assert criteria[2][1] == "EXCLUSION"
        assert criteria[3][1] == "EXCLUSION"

    def test_empty_text(self):
        assert _parse_criteria("") == []

    def test_no_numbered_items(self):
        assert _parse_criteria("Some plain text about eligibility") == []


class TestMatchCriterion:
    """Tests for _match_criterion pattern matching."""

    def test_age_gte(self):
        result = _match_criterion("Age >= 18 years", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("AGE")
        assert result["IETEST"] == "Age Verification"
        assert "18" in result["IEORRES"]
        assert result["MATCHED"] is True

    def test_age_range(self):
        result = _match_criterion("Age 18 to 65 years", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("AGE")
        assert "18" in result["IEORRES"]
        assert "65" in result["IEORRES"]

    def test_informed_consent(self):
        result = _match_criterion("Signed informed consent form", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("ICF")
        assert result["MATCHED"] is True

    def test_pregnancy_test(self):
        result = _match_criterion("Negative pregnancy test at screening", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("PREG")
        assert result["MATCHED"] is True

    def test_ecog(self):
        result = _match_criterion("ECOG performance status <= 2", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("ECOG")
        assert result["MATCHED"] is True

    def test_diagnosis(self):
        result = _match_criterion("Confirmed diagnosis of Type 2 Diabetes Mellitus.", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("DX")
        assert "Diabetes" in result["IEORRES"]
        assert result["IECAT"] == "INCLUSION"

    def test_lab_value_hemoglobin(self):
        result = _match_criterion("Hemoglobin >= 9.0 g/dL", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("HGB")
        assert result["MATCHED"] is True

    def test_liver_function(self):
        result = _match_criterion("AST <= 3x ULN", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("LIVER")

    def test_renal_function(self):
        result = _match_criterion("eGFR >= 30 mL/min", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("RENAL")

    def test_bmi(self):
        result = _match_criterion("Body mass index >= 18.5", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("BMI")

    def test_washout(self):
        result = _match_criterion("Washout period of 4 weeks from prior therapy", 1, "EXCLUSION")
        assert result["IETESTCD"].startswith("WASH")
        assert "4 weeks" in result["IEORRES"]

    def test_exclusion_category(self):
        result = _match_criterion("Prior chemotherapy within 6 months", 1, "EXCLUSION")
        assert result["IECAT"] == "EXCLUSION"

    def test_generic_fallback(self):
        result = _match_criterion("Willing to comply with study procedures", 1, "INCLUSION")
        assert result["IETESTCD"].startswith("CRIT")
        assert result["IETEST"] == "Criterion Verification"
        assert result["MATCHED"] is False

    def test_testcd_max_8_chars(self):
        result = _match_criterion("Age >= 18 years", 99, "INCLUSION")
        assert len(result["IETESTCD"]) <= 8


class TestIeSpecGenerator:
    """Tests for IeSpecGenerator.generate()."""

    def test_produces_dataframe(self):
        sections = [_make_section("B.5", "Inclusion Criteria:\n1. Age >= 18\n2. Signed consent")]
        gen = IeSpecGenerator()
        df, ct = gen.generate(sections, "STUDY01", "run-1")
        assert len(df) == 2
        assert "IETESTCD" in df.columns

    def test_domain_is_ie(self):
        sections = [_make_section("B.5", "Inclusion:\n1. Age >= 18 years")]
        gen = IeSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        assert all(df["DOMAIN"] == "IE")

    def test_studyid_populated(self):
        sections = [_make_section("B.5", "Inclusion:\n1. Consent signed")]
        gen = IeSpecGenerator()
        df, _ = gen.generate(sections, "MY_STUDY", "run-1")
        assert all(df["STUDYID"] == "MY_STUDY")

    def test_screening_visit(self):
        sections = [_make_section("B.5", "Inclusion:\n1. Age >= 18 years")]
        gen = IeSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        assert all(df["VISIT"] == "SCREENING")
        assert all(df["VISITNUM"] == 1)

    def test_iecat_populated(self):
        sections = [_make_section("B.5",
            "Inclusion Criteria:\n1. Age >= 18\n"
            "Exclusion Criteria:\n1. Pregnant"
        )]
        gen = IeSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        cats = list(df["IECAT"])
        assert "INCLUSION" in cats
        assert "EXCLUSION" in cats

    def test_iecrtext_preserves_original(self):
        sections = [_make_section("B.5", "Inclusion:\n1. Age >= 18 years of age")]
        gen = IeSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        assert "Age >= 18 years" in df.iloc[0]["IECRTEXT"]

    def test_empty_sections(self):
        gen = IeSpecGenerator()
        df, _ = gen.generate([], "STUDY01", "run-1")
        assert len(df) == 0

    def test_no_b5_section(self):
        sections = [_make_section("B.4", "Study design text")]
        gen = IeSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        assert len(df) == 0

    def test_required_columns_present(self):
        sections = [_make_section("B.5", "Inclusion:\n1. Age >= 18")]
        gen = IeSpecGenerator()
        df, _ = gen.generate(sections, "STUDY01", "run-1")
        required = {"STUDYID", "DOMAIN", "USUBJID", "IESEQ", "IETESTCD", "IETEST", "IEORRES", "IECAT"}
        assert required.issubset(set(df.columns))


class TestMatchSummary:
    """Tests for IeSpecGenerator.get_match_summary()."""

    def test_all_matched(self):
        sections = [_make_section("B.5",
            "Inclusion:\n1. Age >= 18 years\n2. Signed informed consent"
        )]
        summary = IeSpecGenerator.get_match_summary(sections)
        assert summary["total"] == 2
        assert summary["matched"] == 2
        assert summary["unmatched"] == 0
        assert summary["match_rate"] == 1.0

    def test_partial_match(self):
        sections = [_make_section("B.5",
            "Inclusion:\n"
            "1. Age >= 18 years\n"
            "2. Willing to follow study procedures\n"
            "3. Signed informed consent"
        )]
        summary = IeSpecGenerator.get_match_summary(sections)
        assert summary["total"] == 3
        assert summary["matched"] == 2  # Age + consent
        assert summary["unmatched"] == 1  # "willing to follow"

    def test_empty(self):
        summary = IeSpecGenerator.get_match_summary([])
        assert summary["total"] == 0
        assert summary["match_rate"] == 0.0
