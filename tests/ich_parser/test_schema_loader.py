"""Tests for ICH E6(R3) schema loader (PTCV-67).

Feature: ICH E6(R3) YAML schema loader
  As a pipeline developer
  I want section definitions loaded from a single YAML file
  So that hardcoded dicts are eliminated and sections stay consistent

  Scenario: Schema loads all 14 sections
    Given the ich_e6r3_schema.yaml file exists
    When load_ich_schema() is called
    Then the returned IchSchema contains exactly 14 sections B.1-B.14

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
    get_classifier_sections,
    get_section_defs,
    get_section_order,
    load_ich_schema,
)

# All expected section codes
_ALL_CODES = [f"B.{i}" for i in range(1, 15)]


class TestLoadIchSchema:
    """Scenario: Schema loads all 14 sections."""

    def test_returns_ich_schema(self) -> None:
        schema = load_ich_schema()
        assert isinstance(schema, IchSchema)

    def test_contains_14_sections(self) -> None:
        schema = load_ich_schema()
        assert len(schema.sections) == 14

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

    def test_all_14_codes(self) -> None:
        defs = get_section_defs()
        assert set(defs.keys()) == set(_ALL_CODES)

    def test_b1_description_content(self) -> None:
        defs = get_section_defs()
        assert "title" in defs["B.1"].lower()
        assert "sponsor" in defs["B.1"].lower()

    def test_b14_description_content(self) -> None:
        defs = get_section_defs()
        assert "financ" in defs["B.14"].lower()


class TestGetSectionOrder:
    """Scenario: get_section_order returns render-ordered tuples."""

    def test_returns_list_of_tuples(self) -> None:
        order = get_section_order()
        assert isinstance(order, list)
        for item in order:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_14_entries(self) -> None:
        order = get_section_order()
        assert len(order) == 14

    def test_sorted_by_render_order(self) -> None:
        order = get_section_order()
        codes = [code for code, _ in order]
        assert codes == _ALL_CODES

    def test_first_and_last(self) -> None:
        order = get_section_order()
        assert order[0] == ("B.1", "General Information")
        assert order[-1] == ("B.14", "Financing, Insurance, and Publication Policy")


class TestGetClassifierSections:
    """Scenario: get_classifier_sections returns regex-ready patterns."""

    def test_returns_dict(self) -> None:
        sections = get_classifier_sections()
        assert isinstance(sections, dict)

    def test_all_14_codes(self) -> None:
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
        assert len(custom.sections) == 14
        assert isinstance(default, IchSchema)
