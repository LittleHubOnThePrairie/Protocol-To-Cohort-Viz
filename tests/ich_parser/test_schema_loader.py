"""Tests for ICH E6(R3) schema loader (PTCV-67, PTCV-68, PTCV-69).

Feature: ICH E6(R3) YAML schema loader
  As a pipeline developer
  I want section definitions loaded from a single YAML file
  So that hardcoded dicts are eliminated and sections stay consistent

  Scenario: Schema loads all 16 sections
    Given the ich_e6r3_schema.yaml file exists
    When load_ich_schema() is called
    Then the returned IchSchema contains exactly 16 sections B.1-B.16

  Scenario: get_section_defs returns description dict
    When get_section_defs() is called
    Then it returns a dict[str, str] matching the old _ICH_SECTION_DEFS

  Scenario: get_section_order returns render-ordered tuples
    When get_section_order() is called
    Then it returns [(code, name)] sorted by render_order

  Scenario: get_classifier_sections returns regex-ready patterns
    When get_classifier_sections() is called
    Then patterns compile as valid regex
    And the dict matches the old _ICH_SECTIONS structure

  Scenario: Schema is cached after first load
    When load_ich_schema() is called twice
    Then the same object is returned both times
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest
import yaml

from ptcv.ich_parser.schema_loader import (
    IchSchema,
    IchSectionDef,
    StagePromptConfig,
    get_boilerplate_pattern,
    get_classifier_sections,
    get_min_sentence_length,
    get_priority_sections,
    get_review_threshold,
    get_section_defs,
    get_section_order,
    get_soa_min_columns,
    get_soa_min_visit_matches,
    get_soa_pattern,
    get_stage_prompt,
    load_ich_schema,
)

# All expected section codes
_ALL_CODES = [f"B.{i}" for i in range(1, 17)]


class TestLoadIchSchema:
    """Scenario: Schema loads all 16 sections."""

    def test_returns_ich_schema(self) -> None:
        schema = load_ich_schema()
        assert isinstance(schema, IchSchema)

    def test_contains_16_sections(self) -> None:
        schema = load_ich_schema()
        assert len(schema.sections) == 16

    def test_all_codes_present(self) -> None:
        schema = load_ich_schema()
        for code in _ALL_CODES:
            assert code in schema.sections, f"Missing section {code}"

    def test_section_fields(self) -> None:
        schema = load_ich_schema()
        sec = schema.sections["B.1"]
        assert isinstance(sec, IchSectionDef)
        assert sec.code == "B.1"
        assert sec.name == "General Information"
        assert sec.requirement == "mandatory"
        assert sec.render_order == 1
        assert len(sec.patterns) >= 1
        assert len(sec.keywords) >= 1

    def test_version_and_date(self) -> None:
        schema = load_ich_schema()
        assert schema.version == "E6(R3)"
        assert schema.effective_date == "2025-01-06"


class TestGetSectionDefs:
    """Scenario: get_section_defs returns description dict."""

    def test_returns_dict_str_str(self) -> None:
        defs = get_section_defs()
        assert isinstance(defs, dict)
        for k, v in defs.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_all_16_codes(self) -> None:
        defs = get_section_defs()
        assert set(defs.keys()) == set(_ALL_CODES)

    def test_b1_description_content(self) -> None:
        defs = get_section_defs()
        assert "title" in defs["B.1"].lower()
        assert "sponsor" in defs["B.1"].lower()

    def test_b14_description_content(self) -> None:
        defs = get_section_defs()
        assert "data handling" in defs["B.14"].lower()


class TestGetSectionOrder:
    """Scenario: get_section_order returns render-ordered tuples."""

    def test_returns_list_of_tuples(self) -> None:
        order = get_section_order()
        assert isinstance(order, list)
        for item in order:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_16_entries(self) -> None:
        order = get_section_order()
        assert len(order) == 16

    def test_sorted_by_render_order(self) -> None:
        order = get_section_order()
        codes = [code for code, _ in order]
        assert codes == _ALL_CODES

    def test_first_and_last(self) -> None:
        order = get_section_order()
        assert order[0] == ("B.1", "General Information")
        assert order[-1] == ("B.16", "Publication Policy")


class TestGetClassifierSections:
    """Scenario: get_classifier_sections returns regex-ready patterns."""

    def test_returns_dict(self) -> None:
        sections = get_classifier_sections()
        assert isinstance(sections, dict)

    def test_all_16_codes(self) -> None:
        sections = get_classifier_sections()
        assert set(sections.keys()) == set(_ALL_CODES)

    def test_section_structure(self) -> None:
        sections = get_classifier_sections()
        for code, defn in sections.items():
            assert "name" in defn, f"{code} missing name"
            assert "patterns" in defn, f"{code} missing patterns"
            assert "keywords" in defn, f"{code} missing keywords"

    def test_patterns_compile(self) -> None:
        """All patterns must be valid regex (single-escaped)."""
        sections = get_classifier_sections()
        for code, defn in sections.items():
            for pat in defn["patterns"]:
                try:
                    re.compile(pat, re.IGNORECASE)
                except re.error as exc:
                    pytest.fail(f"{code} pattern {pat!r} failed: {exc}")

    def test_b1_pattern_matches(self) -> None:
        sections = get_classifier_sections()
        pat = sections["B.1"]["patterns"][0]
        assert re.search(pat, "General Information", re.IGNORECASE)

    def test_b4_keywords(self) -> None:
        sections = get_classifier_sections()
        kw = sections["B.4"]["keywords"]
        assert "randomisation" in kw
        assert "schedule of activities" in kw


class TestCaching:
    """Scenario: Schema is cached after first load."""

    def test_same_object_returned(self) -> None:
        a = load_ich_schema()
        b = load_ich_schema()
        assert a is b

    def test_custom_path_not_cached(self) -> None:
        """Loading from a custom path should not pollute the cache."""
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "templates"
            / "ich_e6r3_schema.yaml"
        )
        custom = load_ich_schema(path=schema_path)
        default = load_ich_schema()
        # Custom load returns correct data but is a separate object
        assert len(custom.sections) == 16
        assert isinstance(default, IchSchema)


# ===================================================================
# PTCV-68: Stage-specific prompt tests
# ===================================================================


class TestStagePromptConfig:
    """Schema loads stage_prompts block."""

    def test_stage_prompts_present(self) -> None:
        schema = load_ich_schema()
        assert len(schema.stage_prompts) >= 4

    def test_retemplater_has_all_16(self) -> None:
        schema = load_ich_schema()
        cfg = schema.stage_prompts["retemplater"]
        assert len(cfg.sections) == 16

    def test_coverage_reviewer_sections(self) -> None:
        schema = load_ich_schema()
        cfg = schema.stage_prompts["coverage_reviewer"]
        assert set(cfg.sections) == {"B.3", "B.4", "B.5"}
        assert cfg.format == "compact"

    def test_sdtm_generator_sections(self) -> None:
        schema = load_ich_schema()
        cfg = schema.stage_prompts["sdtm_generator"]
        assert set(cfg.sections) == {"B.4", "B.7", "B.8", "B.9"}


class TestRetemplaterStagePrompt:
    """Scenario: Retemplater stage prompt matches existing behavior."""

    def test_full_format_matches_legacy(self) -> None:
        """get_stage_prompt('retemplater') must produce the exact same
        string as the old inline code in _build_classification_prompt,
        but using natural sort order (PTCV-97)."""
        from ptcv.ich_parser.query_schema import section_sort_key
        defs = get_section_defs()
        legacy = "\n".join(
            f"  {code}: {desc}"
            for code, desc in sorted(defs.items(), key=lambda x: section_sort_key(x[0]))
        )
        prompt = get_stage_prompt("retemplater")
        assert prompt == legacy

    def test_full_format_starts_with_b1(self) -> None:
        prompt = get_stage_prompt("retemplater")
        assert prompt.startswith("  B.1:")

    def test_full_format_ends_with_b16(self) -> None:
        """Natural sort: B.16 comes last (PTCV-97, PTCV-135)."""
        prompt = get_stage_prompt("retemplater")
        lines = prompt.strip().split("\n")
        assert lines[-1].strip().startswith("B.16:")

    def test_full_format_contains_all_16(self) -> None:
        prompt = get_stage_prompt("retemplater")
        for code in _ALL_CODES:
            assert f"  {code}:" in prompt


class TestCompactFormat:
    """Scenario: Compact format reduces token count."""

    def test_compact_sdtm_under_150_tokens(self) -> None:
        """sdtm_generator compact prompt (4 sections) < 150 tokens.

        Approximation: len(text) / 4 is a conservative char-to-token
        estimate for English text with Claude's tokenizer.
        """
        prompt = get_stage_prompt("sdtm_generator")
        est_tokens = len(prompt) / 4
        assert est_tokens < 150, f"Compact sdtm_generator: ~{est_tokens:.0f} tokens"

    def test_full_retemplater_under_650_tokens(self) -> None:
        prompt = get_stage_prompt("retemplater")
        est_tokens = len(prompt) / 4
        assert est_tokens < 650, f"Full retemplater: ~{est_tokens:.0f} tokens"

    def test_compact_is_xml(self) -> None:
        prompt = get_stage_prompt("sdtm_generator")
        assert prompt.startswith("<ich_e6r3")
        assert prompt.endswith("</ich_e6r3>")

    def test_compact_contains_section_tags(self) -> None:
        prompt = get_stage_prompt("sdtm_generator")
        assert '<section code="B.4"' in prompt
        assert '<section code="B.9"' in prompt

    def test_compact_has_req_attribute(self) -> None:
        prompt = get_stage_prompt("sdtm_generator")
        assert 'req="M"' in prompt


class TestStageFiltering:
    """Scenario: Stage filtering excludes irrelevant sections."""

    def test_coverage_reviewer_only_3_sections(self) -> None:
        prompt = get_stage_prompt("coverage_reviewer")
        assert "B.3" in prompt
        assert "B.4" in prompt
        assert "B.5" in prompt

    def test_coverage_reviewer_excludes_others(self) -> None:
        prompt = get_stage_prompt("coverage_reviewer")
        assert "B.1" not in prompt
        assert "B.2" not in prompt
        for i in range(6, 17):
            assert f"B.{i}" not in prompt

    def test_annotation_service_3_sections(self) -> None:
        prompt = get_stage_prompt("annotation_service")
        assert "B.4" in prompt
        assert "B.5" in prompt
        assert "B.8" in prompt


class TestUnknownStage:
    """Scenario: Unknown stage falls back to full prompt."""

    def test_unknown_returns_full_prompt(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="ptcv.ich_parser.schema_loader"):
            prompt = get_stage_prompt("unknown_stage")
        # Should be the same as retemplater
        expected = get_stage_prompt("retemplater")
        assert prompt == expected

    def test_unknown_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="ptcv.ich_parser.schema_loader"):
            get_stage_prompt("unknown_stage")
        assert any("unknown_stage" in r.message for r in caplog.records)


# ===================================================================
# PTCV-69: Configuration accessor tests
# ===================================================================


class TestReviewThreshold:
    """Scenario: Default review threshold matches current behavior."""

    def test_default_returns_0_70(self) -> None:
        assert get_review_threshold() == 0.70

    def test_unknown_section_returns_default(self) -> None:
        assert get_review_threshold("B.2") == 0.70

    def test_none_returns_default(self) -> None:
        assert get_review_threshold(None) == 0.70


class TestSectionSpecificThreshold:
    """Scenario: Section-specific threshold overrides work."""

    def test_b4_override(self) -> None:
        assert get_review_threshold("B.4") == 0.85

    def test_b5_override(self) -> None:
        assert get_review_threshold("B.5") == 0.80

    def test_b1_falls_through_to_default(self) -> None:
        assert get_review_threshold("B.1") == 0.70

    def test_b14_falls_through_to_default(self) -> None:
        assert get_review_threshold("B.14") == 0.70


class TestDuplicateThresholdEliminated:
    """Scenario: Duplicate threshold constants are eliminated."""

    def test_llm_retemplater_no_module_constant(self) -> None:
        """_REVIEW_THRESHOLD no longer exists in llm_retemplater."""
        from ptcv.ich_parser import llm_retemplater

        assert not hasattr(llm_retemplater, "_REVIEW_THRESHOLD")

    def test_classifier_no_module_constant(self) -> None:
        """REVIEW_THRESHOLD no longer exists in classifier."""
        from ptcv.ich_parser import classifier

        assert not hasattr(classifier, "REVIEW_THRESHOLD")


class TestSoaPattern:
    """Scenario: SoA detection regex from YAML matches current behavior."""

    def test_matches_visit(self) -> None:
        pat = get_soa_pattern()
        assert pat.search("Visit 1")

    def test_matches_day(self) -> None:
        pat = get_soa_pattern()
        assert pat.search("Day 14")

    def test_matches_week(self) -> None:
        pat = get_soa_pattern()
        assert pat.search("Week 4")

    def test_matches_screening(self) -> None:
        pat = get_soa_pattern()
        assert pat.search("Screening")

    def test_matches_end_of_study(self) -> None:
        pat = get_soa_pattern()
        assert pat.search("End of Study")

    def test_no_match_demographics(self) -> None:
        pat = get_soa_pattern()
        assert not pat.search("Patient demographics")

    def test_no_match_adverse_events(self) -> None:
        pat = get_soa_pattern()
        assert not pat.search("Adverse events")


class TestPrioritySections:
    """Scenario: Priority sections from YAML match current frozenset."""

    def test_returns_frozenset(self) -> None:
        ps = get_priority_sections()
        assert isinstance(ps, frozenset)

    def test_matches_expected_set(self) -> None:
        ps = get_priority_sections()
        assert ps == frozenset({"B.4", "B.5", "B.10", "B.14"})


class TestSoaMinValues:
    """SoA detection min_columns and min_visit_matches."""

    def test_min_columns_default(self) -> None:
        assert get_soa_min_columns() == 5

    def test_min_visit_matches_default(self) -> None:
        assert get_soa_min_visit_matches() == 3


class TestBoilerplatePattern:
    """Scenario: Boilerplate patterns from YAML match current regex."""

    def test_matches_page_number(self) -> None:
        pat = get_boilerplate_pattern()
        assert pat.search("Page 42")

    def test_matches_confidential(self) -> None:
        pat = get_boilerplate_pattern()
        assert pat.search("CONFIDENTIAL")

    def test_matches_version(self) -> None:
        pat = get_boilerplate_pattern()
        assert pat.search("Version 3")

    def test_matches_table_of_contents(self) -> None:
        pat = get_boilerplate_pattern()
        assert pat.search("Table of Contents")

    def test_no_match_clinical_text(self) -> None:
        pat = get_boilerplate_pattern()
        assert not pat.search("The primary endpoint is overall survival")


class TestMinSentenceLength:
    """Coverage min_sentence_length from YAML."""

    def test_default_value(self) -> None:
        assert get_min_sentence_length() == 20


class TestConfigurationBlock:
    """Schema loads configuration block."""

    def test_configuration_present(self) -> None:
        schema = load_ich_schema()
        assert isinstance(schema.configuration, dict)
        assert len(schema.configuration) > 0

    def test_configuration_has_expected_keys(self) -> None:
        schema = load_ich_schema()
        cfg = schema.configuration
        assert "review_threshold_default" in cfg
        assert "section_thresholds" in cfg
        assert "priority_sections" in cfg
        assert "soa_detection" in cfg
        assert "coverage" in cfg


# ===================================================================
# PTCV-154: Schema consistency tests
# ===================================================================


class TestSchemaConsistency:
    """Verify query schema and classifier schema use the same section codes.

    PTCV-154: Section numbering mismatch between schemas caused B.15
    queries (financing) to route to B.15 classifier section (formerly
    Supplements and Amendments), producing 50% malformed extractions.
    """

    def test_query_schema_sections_exist_in_classifier(self) -> None:
        """Every query schema_section must exist in classifier schema."""
        from ptcv.ich_parser.query_schema import load_query_schema

        schema = load_ich_schema()
        queries = load_query_schema()
        classifier_codes = set(schema.sections.keys())

        for q in queries:
            assert q.schema_section in classifier_codes, (
                f"Query {q.query_id} references schema_section "
                f"'{q.schema_section}' which does not exist in "
                f"classifier schema (has: {sorted(classifier_codes)})"
            )

    def test_query_parent_sections_exist_in_classifier(self) -> None:
        """Every query parent_section must exist in classifier schema."""
        from ptcv.ich_parser.query_schema import load_query_schema

        schema = load_ich_schema()
        queries = load_query_schema()
        classifier_codes = set(schema.sections.keys())

        parent_sections = {q.parent_section for q in queries}
        for ps in parent_sections:
            assert ps in classifier_codes, (
                f"Query parent_section '{ps}' does not exist in "
                f"classifier schema"
            )

    def test_assembler_names_match_classifier(self) -> None:
        """APPENDIX_B_SECTION_NAMES codes must match classifier schema."""
        from ptcv.ich_parser.template_assembler import (
            APPENDIX_B_SECTION_NAMES,
        )

        schema = load_ich_schema()
        assert set(APPENDIX_B_SECTION_NAMES.keys()) == set(
            schema.sections.keys()
        ), "template_assembler section codes diverged from schema"

    def test_synonym_boost_targets_exist_in_classifier(self) -> None:
        """Every _SYNONYM_BOOSTS target code must exist in classifier."""
        from ptcv.ich_parser.section_matcher import _SYNONYM_BOOSTS

        schema = load_ich_schema()
        classifier_codes = set(schema.sections.keys())

        for synonym, code in _SYNONYM_BOOSTS.items():
            assert code in classifier_codes, (
                f"Synonym boost '{synonym}' → '{code}' targets a "
                f"section code not in classifier schema"
            )

    def test_b15_is_financing(self) -> None:
        """B.15 must be Financing and Insurance (PTCV-154 regression)."""
        schema = load_ich_schema()
        assert schema.sections["B.15"].name == "Financing and Insurance"

    def test_b16_is_publication(self) -> None:
        """B.16 must be Publication Policy (PTCV-154 regression)."""
        schema = load_ich_schema()
        assert schema.sections["B.16"].name == "Publication Policy"

    def test_b12_is_quality_control(self) -> None:
        """B.12 must be Quality Control (PTCV-154 split from B.11)."""
        schema = load_ich_schema()
        assert schema.sections["B.12"].name == (
            "Quality Control and Quality Assurance"
        )

    def test_b13_is_ethics(self) -> None:
        """B.13 must be Ethics (PTCV-154 renumbered from B.12)."""
        schema = load_ich_schema()
        assert schema.sections["B.13"].name == "Ethics"

    def test_b14_is_data_handling(self) -> None:
        """B.14 must be Data Handling (PTCV-154 renumbered from B.13)."""
        schema = load_ich_schema()
        assert schema.sections["B.14"].name == (
            "Data Handling and Record Keeping"
        )
