"""Tests for ICH E6(R3) Appendix B query schema (PTCV-88).

Feature: ICH E6(R3) Appendix B query schema

  Scenario: All Appendix B sections have at least one query
  Scenario: Queries specify expected content type
  Scenario: Schema is loadable as structured data
  Scenario: Required vs recommended queries are distinguished
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from ptcv.ich_parser.query_schema import (
    EXPECTED_TYPES,
    AppendixBQuery,
    _reset_cache,
    get_parent_sections,
    get_queries_by_schema_section,
    get_queries_for_section,
    get_required_queries,
    load_query_schema,
    section_sort_key,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the module-level cache before each test."""
    _reset_cache()
    yield
    _reset_cache()


# -----------------------------------------------------------------------
# Scenario: All Appendix B sections have at least one query
# -----------------------------------------------------------------------
class TestAllSectionsHaveQueries:
    """Given the ICH E6(R3) Appendix B structured template,
    When the query schema is loaded,
    Then sections B.1 through B.16 each have at least one query
    And the total query count is >= 50.
    """

    def test_all_parent_sections_present(self):
        queries = load_query_schema()
        parents = get_parent_sections(queries)
        expected = [
            "B.1", "B.2", "B.3", "B.4", "B.5", "B.6",
            "B.7", "B.8", "B.9", "B.10", "B.11", "B.12",
            "B.13", "B.14", "B.15", "B.16",
        ]
        for section in expected:
            assert section in parents, (
                f"Section {section} missing from query schema"
            )

    def test_each_section_has_at_least_one_query(self):
        queries = load_query_schema()
        parents = get_parent_sections(queries)
        for section in parents:
            section_queries = get_queries_for_section(section, queries)
            assert len(section_queries) >= 1, (
                f"Section {section} has no queries"
            )

    def test_total_query_count_at_least_50(self):
        queries = load_query_schema()
        assert len(queries) >= 50, (
            f"Expected >= 50 queries, got {len(queries)}"
        )


# -----------------------------------------------------------------------
# Scenario: Queries specify expected content type
# -----------------------------------------------------------------------
class TestExpectedContentTypes:
    """Given the query schema,
    When a query for B.5.1 (inclusion criteria) is retrieved,
    Then expected_type is "list" And required is True.
    """

    def test_inclusion_criteria_is_list_and_required(self):
        queries = load_query_schema()
        b5_queries = get_queries_for_section("B.5", queries)
        inclusion = [q for q in b5_queries if "inclusion" in q.query_text.lower()]
        assert len(inclusion) == 1
        assert inclusion[0].expected_type == "list"
        assert inclusion[0].required is True

    def test_exclusion_criteria_is_list_and_required(self):
        queries = load_query_schema()
        b5_queries = get_queries_for_section("B.5", queries)
        exclusion = [q for q in b5_queries if "exclusion" in q.query_text.lower()]
        assert len(exclusion) == 1
        assert exclusion[0].expected_type == "list"
        assert exclusion[0].required is True

    def test_all_expected_types_are_valid(self):
        queries = load_query_schema()
        for q in queries:
            assert q.expected_type in EXPECTED_TYPES, (
                f"Query {q.query_id} has invalid type {q.expected_type!r}"
            )


# -----------------------------------------------------------------------
# Scenario: Queries specify expected content type for tabular sections
# -----------------------------------------------------------------------
class TestTabularSections:
    """Given the query schema,
    When a query for B.4.6 (schedule of events) is retrieved,
    Then expected_type is "table".
    """

    def test_schedule_of_events_is_table(self):
        queries = load_query_schema()
        b4_queries = get_queries_for_section("B.4", queries)
        soe = [q for q in b4_queries if q.section_id == "B.4.6"]
        assert len(soe) >= 1
        assert soe[0].expected_type == "table"


# -----------------------------------------------------------------------
# Scenario: Schema is loadable as structured data
# -----------------------------------------------------------------------
class TestSchemaLoadable:
    """Given the query schema YAML file,
    When load_query_schema() is called,
    Then it returns a list of AppendixBQuery dataclass instances
    And each instance has non-empty fields.
    """

    def test_returns_list_of_dataclasses(self):
        queries = load_query_schema()
        assert isinstance(queries, list)
        assert all(isinstance(q, AppendixBQuery) for q in queries)

    def test_each_query_has_non_empty_fields(self):
        queries = load_query_schema()
        for q in queries:
            assert q.query_id, f"Empty query_id in {q}"
            assert q.section_id, f"Empty section_id in {q}"
            assert q.parent_section, f"Empty parent_section in {q}"
            assert q.schema_section, f"Empty schema_section in {q}"
            assert q.query_text, f"Empty query_text in {q}"
            assert q.expected_type, f"Empty expected_type in {q}"

    def test_query_ids_are_unique(self):
        queries = load_query_schema()
        ids = [q.query_id for q in queries]
        assert len(ids) == len(set(ids)), "Duplicate query_id found"

    def test_frozen_dataclass(self):
        queries = load_query_schema()
        with pytest.raises(AttributeError):
            queries[0].query_id = "modified"  # type: ignore[misc]


# -----------------------------------------------------------------------
# Scenario: Required vs recommended queries are distinguished
# -----------------------------------------------------------------------
class TestRequiredVsRecommended:
    """Given the query schema,
    When queries are filtered by required=True,
    Then all ICH mandatory content sections are represented.
    """

    def test_required_queries_exist(self):
        required = get_required_queries()
        assert len(required) > 0

    def test_required_covers_mandatory_sections(self):
        required = get_required_queries()
        required_parents = {q.parent_section for q in required}
        mandatory_sections = [
            "B.1", "B.2", "B.3", "B.4", "B.5", "B.6",
            "B.7", "B.8", "B.9", "B.10",
        ]
        for section in mandatory_sections:
            assert section in required_parents, (
                f"Mandatory section {section} has no required queries"
            )

    def test_optional_queries_exist(self):
        queries = load_query_schema()
        optional = [q for q in queries if not q.required]
        assert len(optional) > 0, "No optional queries found"


# -----------------------------------------------------------------------
# Accessor function tests
# -----------------------------------------------------------------------
class TestAccessors:
    """Test convenience accessor functions."""

    def test_get_queries_for_section(self):
        b1 = get_queries_for_section("B.1")
        assert len(b1) >= 3  # title, number, date, sponsor, etc.
        assert all(q.parent_section == "B.1" for q in b1)

    def test_get_queries_for_nonexistent_section(self):
        result = get_queries_for_section("B.99")
        assert result == []

    def test_get_queries_by_schema_section(self):
        # B.15 and B.16 in the query schema map to "B.14" in the
        # existing ich_e6r3_schema.yaml
        b14_schema = get_queries_by_schema_section("B.14")
        assert len(b14_schema) >= 3  # B.14 + B.15 + B.16 queries
        parent_sections = {q.parent_section for q in b14_schema}
        assert "B.14" in parent_sections
        assert "B.15" in parent_sections
        assert "B.16" in parent_sections

    def test_get_parent_sections_sorted(self):
        parents = get_parent_sections()
        assert parents == sorted(parents, key=section_sort_key)
        assert len(parents) == 16

    def test_get_parent_sections_natural_order(self):
        """B.9 must come before B.10 (natural, not lexicographic)."""
        parents = get_parent_sections()
        idx_9 = parents.index("B.9")
        idx_10 = parents.index("B.10")
        assert idx_9 < idx_10, (
            f"B.9 (idx {idx_9}) should come before B.10 (idx {idx_10})"
        )


# -----------------------------------------------------------------------
# Edge cases and error handling
# -----------------------------------------------------------------------
class TestErrorHandling:
    """Test error conditions."""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_query_schema(Path("/nonexistent/queries.yaml"))

    def test_invalid_expected_type_raises(self):
        bad_yaml = {
            "queries": [
                {
                    "query_id": "test.q1",
                    "section_id": "B.1",
                    "parent_section": "B.1",
                    "schema_section": "B.1",
                    "query_text": "Test?",
                    "expected_type": "INVALID_TYPE",
                    "required": True,
                }
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            yaml.dump(bad_yaml, f)
            f.flush()
            with pytest.raises(ValueError, match="Invalid expected_type"):
                load_query_schema(Path(f.name))

    def test_caching_returns_same_object(self):
        first = load_query_schema()
        second = load_query_schema()
        assert first is second

    def test_custom_path_bypasses_cache(self):
        # Load default to populate cache
        default = load_query_schema()
        # Load from explicit path — should not reuse cache
        explicit = load_query_schema(
            Path(__file__).resolve().parents[2]
            / "data" / "templates" / "appendix_b_queries.yaml"
        )
        assert len(explicit) == len(default)
        # But they should be different objects
        assert explicit is not default


# -----------------------------------------------------------------------
# section_sort_key tests (PTCV-97)
# -----------------------------------------------------------------------
class TestSectionSortKey:
    """Test natural sort key for ICH section codes."""

    def test_parent_sections_natural_order(self):
        codes = ["B.1", "B.10", "B.2", "B.16", "B.9", "B.3"]
        result = sorted(codes, key=section_sort_key)
        assert result == ["B.1", "B.2", "B.3", "B.9", "B.10", "B.16"]

    def test_sub_sections_natural_order(self):
        codes = ["B.4.10", "B.4.2", "B.4.1", "B.4.9"]
        result = sorted(codes, key=section_sort_key)
        assert result == ["B.4.1", "B.4.2", "B.4.9", "B.4.10"]

    def test_mixed_depth_sections(self):
        codes = ["B.10", "B.1", "B.1.1", "B.1.2", "B.2"]
        result = sorted(codes, key=section_sort_key)
        assert result == ["B.1", "B.1.1", "B.1.2", "B.2", "B.10"]

    def test_single_code(self):
        assert section_sort_key("B.5") == [0, 5]

    def test_deep_code(self):
        assert section_sort_key("B.4.6.1") == [0, 4, 6, 1]

    def test_full_b1_through_b16(self):
        """The canonical Appendix B ordering."""
        codes = [f"B.{i}" for i in range(1, 17)]
        shuffled = sorted(codes)  # lexicographic — wrong order
        assert shuffled != codes, "Sanity: lex sort should differ"
        result = sorted(shuffled, key=section_sort_key)
        assert result == codes
