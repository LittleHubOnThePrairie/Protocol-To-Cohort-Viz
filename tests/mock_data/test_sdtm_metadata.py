"""Tests for SDTM domain variable metadata registry (PTCV-116).

Feature: SDTM Domain Variable Metadata Registry

  Scenario: Domain spec for VS retrieved
    Given the SDTM metadata registry is loaded
    When get_domain_spec("VS") is called
    Then it returns a SdtmDomainSpec with domain_code "VS"
    And the variables list includes VSTESTCD, VSORRES, VSORRESU, VSSTRESN,
        VSSTRESU
    And VSTESTCD has codelist containing SYSBP, DIABP, PULSE, TEMP, HEIGHT,
        WEIGHT

  Scenario: All 6 initial domains registered
    Given the SDTM metadata registry is loaded
    When get_all_domain_specs() is called
    Then it returns specs for DM, VS, LB, AE, EG, CM

  Scenario: Required variables marked correctly
    Given the VS domain spec
    When examining variable cores
    Then USUBJID, VSTESTCD, VSORRES have core "Required"

  Scenario: Numeric ranges defined for vital signs
    Given the VS domain spec
    When examining VSSTRESN range
    Then range_min and range_max are physiologically plausible
"""

from __future__ import annotations

import pytest

from ptcv.mock_data.sdtm_metadata import (
    SdtmDomainSpec,
    SdtmVariableSpec,
    get_all_domain_specs,
    get_domain_spec,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _var_by_name(
    spec: SdtmDomainSpec, name: str,
) -> SdtmVariableSpec:
    """Return the variable spec with the given name, or raise."""
    for v in spec.variables:
        if v.name == name:
            return v
    raise ValueError(f"{name} not found in {spec.domain_code}")


# -------------------------------------------------------------------
# TestRegistryLookup
# -------------------------------------------------------------------


class TestRegistryLookup:
    """Registry lookup and completeness tests."""

    def test_get_domain_spec_vs(self) -> None:
        spec = get_domain_spec("VS")
        assert spec.domain_code == "VS"
        assert spec.domain_name == "Vital Signs"

    def test_get_domain_spec_case_insensitive(self) -> None:
        assert get_domain_spec("vs").domain_code == "VS"

    def test_get_domain_spec_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_domain_spec("ZZ")

    def test_all_six_domains_registered(self) -> None:
        specs = get_all_domain_specs()
        assert set(specs.keys()) == {"DM", "VS", "LB", "AE", "EG", "CM"}

    def test_all_returns_copy(self) -> None:
        a = get_all_domain_specs()
        b = get_all_domain_specs()
        assert a is not b


# -------------------------------------------------------------------
# TestDomainSpecStructure
# -------------------------------------------------------------------


class TestDomainSpecStructure:
    """Domain spec data model integrity."""

    def test_domain_spec_is_frozen(self) -> None:
        spec = get_domain_spec("VS")
        with pytest.raises(AttributeError):
            spec.domain_code = "XX"  # type: ignore[misc]

    def test_variable_spec_is_frozen(self) -> None:
        spec = get_domain_spec("VS")
        with pytest.raises(AttributeError):
            spec.variables[0].name = "BAD"  # type: ignore[misc]

    def test_variables_are_tuple(self) -> None:
        spec = get_domain_spec("VS")
        assert isinstance(spec.variables, tuple)

    def test_keys_are_tuple(self) -> None:
        spec = get_domain_spec("VS")
        assert isinstance(spec.keys, tuple)

    def test_domain_class_values(self) -> None:
        valid = {"Findings", "Events", "Interventions", "Special Purpose"}
        for spec in get_all_domain_specs().values():
            assert spec.domain_class in valid, (
                f"{spec.domain_code} has invalid class {spec.domain_class}"
            )


# -------------------------------------------------------------------
# TestVSDomain
# -------------------------------------------------------------------


class TestVSDomain:
    """VS domain spec per GHERKIN scenarios."""

    def test_vs_variables_include_required_names(self) -> None:
        spec = get_domain_spec("VS")
        names = {v.name for v in spec.variables}
        for expected in ("VSTESTCD", "VSORRES", "VSORRESU", "VSSTRESN",
                         "VSSTRESU"):
            assert expected in names, f"{expected} missing from VS"

    def test_vstestcd_codelist(self) -> None:
        v = _var_by_name(get_domain_spec("VS"), "VSTESTCD")
        assert v.codelist is not None
        for code in ("SYSBP", "DIABP", "PULSE", "TEMP", "HEIGHT", "WEIGHT"):
            assert code in v.codelist, f"{code} missing from VSTESTCD codelist"

    def test_required_variables(self) -> None:
        spec = get_domain_spec("VS")
        for name in ("USUBJID", "VSTESTCD", "VSORRES"):
            v = _var_by_name(spec, name)
            assert v.core == "Required", f"{name} should be Required"

    def test_vsstresn_range(self) -> None:
        v = _var_by_name(get_domain_spec("VS"), "VSSTRESN")
        assert v.type == "Num"
        assert v.range_min is not None
        assert v.range_max is not None
        assert v.range_min >= 0
        assert v.range_max <= 500

    def test_vs_domain_class(self) -> None:
        assert get_domain_spec("VS").domain_class == "Findings"

    def test_vs_keys(self) -> None:
        spec = get_domain_spec("VS")
        assert "USUBJID" in spec.keys
        assert "VSTESTCD" in spec.keys


# -------------------------------------------------------------------
# TestDMDomain
# -------------------------------------------------------------------


class TestDMDomain:
    """DM domain spec tests."""

    def test_dm_domain_class(self) -> None:
        assert get_domain_spec("DM").domain_class == "Special Purpose"

    def test_dm_sex_codelist(self) -> None:
        v = _var_by_name(get_domain_spec("DM"), "SEX")
        assert v.codelist is not None
        assert "M" in v.codelist
        assert "F" in v.codelist

    def test_dm_age_range(self) -> None:
        v = _var_by_name(get_domain_spec("DM"), "AGE")
        assert v.type == "Num"
        assert v.range_min == 18
        assert v.range_max == 100

    def test_dm_keys(self) -> None:
        spec = get_domain_spec("DM")
        assert spec.keys == ("STUDYID", "USUBJID")


# -------------------------------------------------------------------
# TestLBDomain
# -------------------------------------------------------------------


class TestLBDomain:
    """LB domain spec tests."""

    def test_lb_domain_class(self) -> None:
        assert get_domain_spec("LB").domain_class == "Findings"

    def test_lb_testcd_codelist_has_common_tests(self) -> None:
        v = _var_by_name(get_domain_spec("LB"), "LBTESTCD")
        assert v.codelist is not None
        for code in ("ALT", "AST", "CREAT", "HGB", "WBC"):
            assert code in v.codelist

    def test_lb_nrind_codelist(self) -> None:
        v = _var_by_name(get_domain_spec("LB"), "LBNRIND")
        assert v.codelist == frozenset({"NORMAL", "LOW", "HIGH"})

    def test_lb_category_codelist(self) -> None:
        v = _var_by_name(get_domain_spec("LB"), "LBCAT")
        assert v.codelist is not None
        assert "CHEMISTRY" in v.codelist
        assert "HEMATOLOGY" in v.codelist


# -------------------------------------------------------------------
# TestAEDomain
# -------------------------------------------------------------------


class TestAEDomain:
    """AE domain spec tests."""

    def test_ae_domain_class(self) -> None:
        assert get_domain_spec("AE").domain_class == "Events"

    def test_ae_severity_codelist(self) -> None:
        v = _var_by_name(get_domain_spec("AE"), "AESEV")
        assert v.codelist == frozenset({"MILD", "MODERATE", "SEVERE"})

    def test_ae_serious_codelist(self) -> None:
        v = _var_by_name(get_domain_spec("AE"), "AESER")
        assert v.codelist == frozenset({"Y", "N"})

    def test_ae_required_variables(self) -> None:
        spec = get_domain_spec("AE")
        for name in ("USUBJID", "AETERM", "AEDECOD"):
            v = _var_by_name(spec, name)
            assert v.core == "Required", f"AE.{name} should be Required"


# -------------------------------------------------------------------
# TestEGDomain
# -------------------------------------------------------------------


class TestEGDomain:
    """EG domain spec tests."""

    def test_eg_domain_class(self) -> None:
        assert get_domain_spec("EG").domain_class == "Findings"

    def test_eg_testcd_codelist(self) -> None:
        v = _var_by_name(get_domain_spec("EG"), "EGTESTCD")
        assert v.codelist is not None
        assert "EGQTCF" in v.codelist


# -------------------------------------------------------------------
# TestCMDomain
# -------------------------------------------------------------------


class TestCMDomain:
    """CM domain spec tests."""

    def test_cm_domain_class(self) -> None:
        assert get_domain_spec("CM").domain_class == "Interventions"

    def test_cm_route_codelist(self) -> None:
        v = _var_by_name(get_domain_spec("CM"), "CMROUTE")
        assert v.codelist is not None
        assert "ORAL" in v.codelist
        assert "INTRAVENOUS" in v.codelist

    def test_cm_required_variables(self) -> None:
        spec = get_domain_spec("CM")
        v = _var_by_name(spec, "CMTRT")
        assert v.core == "Required"


# -------------------------------------------------------------------
# TestCrossdomainConsistency
# -------------------------------------------------------------------


class TestCrossdomainConsistency:
    """Cross-domain consistency checks."""

    def test_all_domains_have_studyid(self) -> None:
        for code, spec in get_all_domain_specs().items():
            names = {v.name for v in spec.variables}
            assert "STUDYID" in names, f"{code} missing STUDYID"

    def test_all_domains_have_domain_var(self) -> None:
        for code, spec in get_all_domain_specs().items():
            names = {v.name for v in spec.variables}
            assert "DOMAIN" in names, f"{code} missing DOMAIN"

    def test_all_domains_have_usubjid(self) -> None:
        for code, spec in get_all_domain_specs().items():
            names = {v.name for v in spec.variables}
            assert "USUBJID" in names, f"{code} missing USUBJID"

    def test_domain_var_codelist_matches_code(self) -> None:
        for code, spec in get_all_domain_specs().items():
            domain_var = _var_by_name(spec, "DOMAIN")
            assert domain_var.codelist is not None
            assert code in domain_var.codelist

    def test_all_key_vars_exist_in_variables(self) -> None:
        for code, spec in get_all_domain_specs().items():
            names = {v.name for v in spec.variables}
            for key in spec.keys:
                assert key in names, (
                    f"{code} key {key} not in variables"
                )
