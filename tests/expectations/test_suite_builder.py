"""Tests for Great Expectations suite builder (PTCV-117).

Feature: GX ExpectationSuite from SDTM Variable Metadata

  Scenario: VS suite contains column existence expectations
    Given the VS domain spec is loaded
    When build_domain_expectations("VS") is called
    Then there is one expect_column_to_exist per variable

  Scenario: Required variables get not-null expectations
    Given the VS domain spec is loaded
    When build_domain_expectations("VS") is called
    Then USUBJID, VSTESTCD, VSORRES have not-null expectations

  Scenario: Codelist variables get set expectations
    Given the VS domain spec is loaded
    When build_domain_expectations("VS") is called
    Then VSTESTCD gets expect_column_values_to_be_in_set

  Scenario: Numeric ranges generate between expectations
    Given the VS domain spec is loaded
    When build_domain_expectations("VS") is called
    Then VSSTRESN gets expect_column_values_to_be_between
"""

from __future__ import annotations

import pytest

from ptcv.expectations.suite_builder import (
    ExpectationDescriptor,
    SuiteConfig,
    build_all_expectation_sets,
    build_domain_expectations,
    build_domain_suite,
)
from ptcv.mock_data.sdtm_metadata import get_all_domain_specs, get_domain_spec


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _by_type(
    expectations: list[ExpectationDescriptor],
    etype: str,
) -> list[ExpectationDescriptor]:
    """Filter expectations by type."""
    return [e for e in expectations if e.expectation_type == etype]


def _by_column(
    expectations: list[ExpectationDescriptor],
    column: str,
) -> list[ExpectationDescriptor]:
    """Filter expectations by column name."""
    return [e for e in expectations if e.kwargs.get("column") == column]


# -------------------------------------------------------------------
# TestExpectationDescriptor
# -------------------------------------------------------------------


class TestExpectationDescriptor:
    """Data model tests."""

    def test_frozen(self) -> None:
        desc = ExpectationDescriptor(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "X"},
        )
        with pytest.raises(AttributeError):
            desc.expectation_type = "bad"  # type: ignore[misc]

    def test_default_kwargs(self) -> None:
        desc = ExpectationDescriptor(expectation_type="test")
        assert desc.kwargs == {}

    def test_equality(self) -> None:
        a = ExpectationDescriptor("t", {"column": "X"})
        b = ExpectationDescriptor("t", {"column": "X"})
        assert a == b


# -------------------------------------------------------------------
# TestSuiteConfig
# -------------------------------------------------------------------


class TestSuiteConfig:
    """Suite configuration tests."""

    def test_defaults(self) -> None:
        cfg = SuiteConfig()
        assert cfg.include_column_existence is True
        assert cfg.include_not_null is True
        assert cfg.include_value_set is True
        assert cfg.include_value_range is True
        assert cfg.mostly == 1.0

    def test_frozen(self) -> None:
        cfg = SuiteConfig()
        with pytest.raises(AttributeError):
            cfg.mostly = 0.5  # type: ignore[misc]


# -------------------------------------------------------------------
# TestVSColumnExistence
# -------------------------------------------------------------------


class TestVSColumnExistence:
    """VS suite column existence expectations."""

    def test_one_existence_per_variable(self) -> None:
        spec = get_domain_spec("VS")
        exps = build_domain_expectations("VS")
        existence = _by_type(exps, "expect_column_to_exist")
        assert len(existence) == len(spec.variables)

    def test_all_vs_columns_present(self) -> None:
        spec = get_domain_spec("VS")
        exps = build_domain_expectations("VS")
        existence = _by_type(exps, "expect_column_to_exist")
        columns = {e.kwargs["column"] for e in existence}
        expected = {v.name for v in spec.variables}
        assert columns == expected

    def test_existence_disabled(self) -> None:
        cfg = SuiteConfig(include_column_existence=False)
        exps = build_domain_expectations("VS", config=cfg)
        existence = _by_type(exps, "expect_column_to_exist")
        assert len(existence) == 0


# -------------------------------------------------------------------
# TestNotNull
# -------------------------------------------------------------------


class TestNotNull:
    """Not-null expectations for Required variables."""

    def test_required_vars_have_not_null(self) -> None:
        exps = build_domain_expectations("VS")
        not_null = _by_type(
            exps, "expect_column_values_to_not_be_null",
        )
        columns = {e.kwargs["column"] for e in not_null}
        for name in ("USUBJID", "VSTESTCD", "VSORRES"):
            assert name in columns, f"{name} missing not-null"

    def test_non_required_vars_skip_not_null(self) -> None:
        exps = build_domain_expectations("VS")
        not_null = _by_type(
            exps, "expect_column_values_to_not_be_null",
        )
        columns = {e.kwargs["column"] for e in not_null}
        # VSSTRESC is Expected, not Required
        assert "VSSTRESC" not in columns

    def test_not_null_has_mostly(self) -> None:
        exps = build_domain_expectations("VS")
        not_null = _by_type(
            exps, "expect_column_values_to_not_be_null",
        )
        assert all(e.kwargs["mostly"] == 1.0 for e in not_null)

    def test_custom_mostly(self) -> None:
        cfg = SuiteConfig(mostly=0.95)
        exps = build_domain_expectations("VS", config=cfg)
        not_null = _by_type(
            exps, "expect_column_values_to_not_be_null",
        )
        assert all(e.kwargs["mostly"] == 0.95 for e in not_null)

    def test_not_null_disabled(self) -> None:
        cfg = SuiteConfig(include_not_null=False)
        exps = build_domain_expectations("VS", config=cfg)
        not_null = _by_type(
            exps, "expect_column_values_to_not_be_null",
        )
        assert len(not_null) == 0


# -------------------------------------------------------------------
# TestValueSet
# -------------------------------------------------------------------


class TestValueSet:
    """Codelist → value set expectations."""

    def test_vstestcd_has_value_set(self) -> None:
        exps = build_domain_expectations("VS")
        vstestcd = [
            e for e in exps
            if (e.expectation_type
                == "expect_column_values_to_be_in_set"
                and e.kwargs.get("column") == "VSTESTCD")
        ]
        assert len(vstestcd) == 1
        assert "SYSBP" in vstestcd[0].kwargs["value_set"]
        assert "DIABP" in vstestcd[0].kwargs["value_set"]

    def test_value_set_is_sorted(self) -> None:
        exps = build_domain_expectations("VS")
        vstestcd = [
            e for e in exps
            if (e.expectation_type
                == "expect_column_values_to_be_in_set"
                and e.kwargs.get("column") == "VSTESTCD")
        ]
        values = vstestcd[0].kwargs["value_set"]
        assert values == sorted(values)

    def test_no_value_set_for_unrestricted(self) -> None:
        exps = build_domain_expectations("VS")
        # VSTEST has no codelist
        vstest_sets = [
            e for e in exps
            if (e.expectation_type
                == "expect_column_values_to_be_in_set"
                and e.kwargs.get("column") == "VSTEST")
        ]
        assert len(vstest_sets) == 0

    def test_ae_severity_codelist(self) -> None:
        exps = build_domain_expectations("AE")
        aesev = [
            e for e in exps
            if (e.expectation_type
                == "expect_column_values_to_be_in_set"
                and e.kwargs.get("column") == "AESEV")
        ]
        assert len(aesev) == 1
        assert set(aesev[0].kwargs["value_set"]) == {
            "MILD", "MODERATE", "SEVERE",
        }

    def test_value_set_disabled(self) -> None:
        cfg = SuiteConfig(include_value_set=False)
        exps = build_domain_expectations("VS", config=cfg)
        sets = _by_type(exps, "expect_column_values_to_be_in_set")
        assert len(sets) == 0


# -------------------------------------------------------------------
# TestValueRange
# -------------------------------------------------------------------


class TestValueRange:
    """Numeric range → between expectations."""

    def test_vsstresn_has_between(self) -> None:
        exps = build_domain_expectations("VS")
        between = [
            e for e in exps
            if (e.expectation_type
                == "expect_column_values_to_be_between"
                and e.kwargs.get("column") == "VSSTRESN")
        ]
        assert len(between) == 1
        assert between[0].kwargs["min_value"] == 0
        assert between[0].kwargs["max_value"] == 500

    def test_dm_age_between(self) -> None:
        exps = build_domain_expectations("DM")
        between = [
            e for e in exps
            if (e.expectation_type
                == "expect_column_values_to_be_between"
                and e.kwargs.get("column") == "AGE")
        ]
        assert len(between) == 1
        assert between[0].kwargs["min_value"] == 18
        assert between[0].kwargs["max_value"] == 100

    def test_no_between_without_range(self) -> None:
        exps = build_domain_expectations("VS")
        # VSTEST is Char with no range
        vstest_between = [
            e for e in exps
            if (e.expectation_type
                == "expect_column_values_to_be_between"
                and e.kwargs.get("column") == "VSTEST")
        ]
        assert len(vstest_between) == 0

    def test_value_range_disabled(self) -> None:
        cfg = SuiteConfig(include_value_range=False)
        exps = build_domain_expectations("VS", config=cfg)
        between = _by_type(
            exps, "expect_column_values_to_be_between",
        )
        assert len(between) == 0


# -------------------------------------------------------------------
# TestBuildAllExpectationSets
# -------------------------------------------------------------------


class TestBuildAllExpectationSets:
    """build_all_expectation_sets() tests."""

    def test_returns_all_domains(self) -> None:
        result = build_all_expectation_sets()
        assert set(result.keys()) == {"DM", "VS", "LB", "AE", "EG", "CM"}

    def test_each_domain_has_expectations(self) -> None:
        result = build_all_expectation_sets()
        for code, exps in result.items():
            assert len(exps) > 0, f"{code} has no expectations"

    def test_custom_config_propagates(self) -> None:
        cfg = SuiteConfig(include_column_existence=False)
        result = build_all_expectation_sets(config=cfg)
        for code, exps in result.items():
            existence = _by_type(exps, "expect_column_to_exist")
            assert len(existence) == 0, (
                f"{code} still has existence expectations"
            )


# -------------------------------------------------------------------
# TestBuildDomainSuite
# -------------------------------------------------------------------


class TestBuildDomainSuite:
    """build_domain_suite() GX conversion tests."""

    def test_raises_import_error_without_gx(self) -> None:
        with pytest.raises(ImportError, match="great_expectations"):
            build_domain_suite("VS")


# -------------------------------------------------------------------
# TestPassSpecDirectly
# -------------------------------------------------------------------


class TestPassSpecDirectly:
    """Using spec parameter bypasses registry lookup."""

    def test_custom_spec(self) -> None:
        from ptcv.mock_data.sdtm_metadata import (
            SdtmDomainSpec,
            SdtmVariableSpec,
        )
        mini_spec = SdtmDomainSpec(
            domain_code="XX",
            domain_name="Test",
            domain_class="Findings",
            variables=(
                SdtmVariableSpec(
                    name="COL1", label="Column 1",
                    type="Char", core="Required",
                ),
            ),
            keys=("COL1",),
        )
        exps = build_domain_expectations("XX", spec=mini_spec)
        columns = {e.kwargs["column"] for e in exps}
        assert columns == {"COL1"}


# -------------------------------------------------------------------
# TestCrossDomainExpectations
# -------------------------------------------------------------------


class TestCrossDomainExpectations:
    """Cross-domain consistency checks."""

    def test_all_domains_have_studyid_existence(self) -> None:
        all_sets = build_all_expectation_sets()
        for code, exps in all_sets.items():
            studyid = [
                e for e in exps
                if (e.expectation_type == "expect_column_to_exist"
                    and e.kwargs.get("column") == "STUDYID")
            ]
            assert len(studyid) == 1, (
                f"{code} missing STUDYID existence"
            )

    def test_all_domains_have_usubjid_not_null(self) -> None:
        all_sets = build_all_expectation_sets()
        for code, exps in all_sets.items():
            usubjid = [
                e for e in exps
                if (e.expectation_type
                    == "expect_column_values_to_not_be_null"
                    and e.kwargs.get("column") == "USUBJID")
            ]
            assert len(usubjid) == 1, (
                f"{code} missing USUBJID not-null"
            )

    def test_all_domains_have_domain_value_set(self) -> None:
        all_sets = build_all_expectation_sets()
        for code, exps in all_sets.items():
            domain_set = [
                e for e in exps
                if (e.expectation_type
                    == "expect_column_values_to_be_in_set"
                    and e.kwargs.get("column") == "DOMAIN")
            ]
            assert len(domain_set) == 1, (
                f"{code} missing DOMAIN value set"
            )
            assert code in domain_set[0].kwargs["value_set"]
