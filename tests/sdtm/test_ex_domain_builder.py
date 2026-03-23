"""Tests for EX domain builder (PTCV-248).

Tests intervention parsing from registry and text, dose/route/frequency
extraction, and arm-treatment mapping.
"""

from __future__ import annotations

import pytest

from ptcv.sdtm.ex_domain_builder import (
    ExDomainSpec,
    InterventionDetail,
    build_ex_domain_spec,
    parse_interventions_from_registry,
    parse_interventions_from_text,
    _parse_dose,
    _parse_route,
    _parse_frequency,
)


class TestParseDose:
    def test_mg(self):
        assert _parse_dose("100 mg daily") == ("100", "MG")

    def test_mg_no_space(self):
        assert _parse_dose("10mg bid") == ("10", "MG")

    def test_mg_kg(self):
        assert _parse_dose("5 mg/kg IV") == ("5", "MG/KG")

    def test_mcg(self):
        assert _parse_dose("200 mcg subcutaneous") == ("200", "MCG")

    def test_decimal(self):
        assert _parse_dose("0.5 mg daily") == ("0.5", "MG")

    def test_no_dose(self):
        assert _parse_dose("placebo capsule") == ("", "")


class TestParseRoute:
    def test_oral(self):
        assert _parse_route("oral administration") == "ORAL"

    def test_iv(self):
        assert _parse_route("IV infusion over 30 min") == "INTRAVENOUS"

    def test_subcutaneous(self):
        assert _parse_route("subcutaneous injection") == "SUBCUTANEOUS"

    def test_no_route(self):
        assert _parse_route("take medication") == ""


class TestParseFrequency:
    def test_bid(self):
        assert _parse_frequency("100mg bid") == "BID"

    def test_twice_daily(self):
        assert _parse_frequency("twice daily dosing") == "BID"

    def test_daily(self):
        assert _parse_frequency("once daily") == "QD"

    def test_q3w(self):
        assert _parse_frequency("administered every 3 weeks") == "Q3W"

    def test_weekly(self):
        assert _parse_frequency("weekly injection") == "QW"

    def test_no_frequency(self):
        assert _parse_frequency("as needed") == ""


class TestParseFromRegistry:
    def test_drug_intervention(self):
        metadata = {
            "protocolSection": {
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "Drug X",
                            "description": "100 mg oral twice daily",
                            "armGroupLabels": ["Treatment Arm A"],
                        },
                    ],
                },
            },
        }

        result = parse_interventions_from_registry(metadata)

        assert len(result) == 1
        intv = result[0]
        assert intv.name == "Drug X"
        assert intv.intervention_type == "DRUG"
        assert intv.dose == "100"
        assert intv.dose_unit == "MG"
        assert intv.route == "ORAL"
        assert intv.frequency == "BID"
        assert "Treatment Arm A" in intv.arm_labels

    def test_placebo(self):
        metadata = {
            "protocolSection": {
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "Placebo",
                            "description": "Matching placebo capsule",
                            "armGroupLabels": ["Placebo Arm"],
                        },
                    ],
                },
            },
        }

        result = parse_interventions_from_registry(metadata)

        assert len(result) == 1
        assert result[0].name == "Placebo"
        assert result[0].dose == ""  # No dose for placebo

    def test_multiple_interventions(self):
        metadata = {
            "protocolSection": {
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "Drug A",
                            "description": "10 mg IV weekly",
                            "armGroupLabels": ["Arm 1"],
                        },
                        {
                            "type": "DRUG",
                            "name": "Drug B",
                            "description": "200 mg oral daily",
                            "armGroupLabels": ["Arm 1", "Arm 2"],
                        },
                    ],
                },
            },
        }

        result = parse_interventions_from_registry(metadata)
        assert len(result) == 2
        assert result[0].route == "INTRAVENOUS"
        assert result[1].route == "ORAL"

    def test_empty_module(self):
        metadata = {"protocolSection": {}}
        result = parse_interventions_from_registry(metadata)
        assert result == []

    def test_real_nct01512251_structure(self):
        """Test with structure matching NCT01512251."""
        metadata = {
            "protocolSection": {
                "armsInterventionsModule": {
                    "armGroups": [
                        {
                            "label": "No Previous Treatment",
                            "type": "OTHER",
                            "interventionNames": [
                                "Drug: BKM120 Combined with Vemurafenib",
                            ],
                        },
                    ],
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "BKM120 Combined with Vemurafenib",
                            "description": (
                                "BKM120 60 mg daily, "
                                "Vemurafenib 720 mg bid"
                            ),
                            "armGroupLabels": ["No Previous Treatment"],
                        },
                    ],
                },
            },
        }

        result = parse_interventions_from_registry(metadata)

        assert len(result) == 1
        intv = result[0]
        assert intv.dose == "60"
        assert intv.dose_unit == "MG"
        # "daily" appears before "bid" in combined description
        assert intv.frequency in ("QD", "BID")


class TestParseFromText:
    def test_dose_line(self):
        text = "Drug X 100 mg oral twice daily\nPlacebo capsule"
        result = parse_interventions_from_text(text)

        assert len(result) == 1  # Only line with dose
        assert result[0].dose == "100"
        assert result[0].route == "ORAL"
        assert result[0].frequency == "BID"

    def test_multiple_dose_lines(self):
        text = (
            "Arm A: Drug X 50 mg IV every 3 weeks\n"
            "Arm B: Drug Y 200 mg oral daily\n"
        )
        result = parse_interventions_from_text(text)
        assert len(result) == 2

    def test_empty_text(self):
        assert parse_interventions_from_text("") == []


class TestBuildExDomainSpec:
    def test_from_registry(self):
        metadata = {
            "protocolSection": {
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "Drug X",
                            "description": "10 mg oral daily",
                            "armGroupLabels": ["Arm A"],
                        },
                        {
                            "type": "DRUG",
                            "name": "Placebo",
                            "description": "matching placebo",
                            "armGroupLabels": ["Arm B"],
                        },
                    ],
                },
            },
        }

        spec = build_ex_domain_spec(registry_metadata=metadata)

        assert spec.treatment_count == 2
        assert len(spec.variables) > 0
        assert "Arm A" in spec.arm_treatment_map
        assert "Drug X" in spec.arm_treatment_map["Arm A"]

    def test_from_text_fallback(self):
        spec = build_ex_domain_spec(
            protocol_text="Drug Z 50 mg subcutaneous weekly",
        )

        assert spec.treatment_count == 1
        assert spec.interventions[0].dose == "50"
        assert spec.interventions[0].route == "SUBCUTANEOUS"
        assert spec.interventions[0].frequency == "QW"

    def test_registry_preferred_over_text(self):
        metadata = {
            "protocolSection": {
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "Registry Drug",
                            "description": "100 mg oral",
                            "armGroupLabels": [],
                        },
                    ],
                },
            },
        }

        spec = build_ex_domain_spec(
            registry_metadata=metadata,
            protocol_text="Text Drug 50 mg IV",
        )

        # Registry should take precedence
        assert spec.interventions[0].name == "Registry Drug"

    def test_empty_inputs(self):
        spec = build_ex_domain_spec()
        assert spec.treatment_count == 0
        assert spec.variables == list(spec.variables)  # has standard vars

    def test_variables_include_required_fields(self):
        spec = build_ex_domain_spec()
        var_names = [v["name"] for v in spec.variables]
        assert "EXTRT" in var_names
        assert "EXDOSE" in var_names
        assert "EXDOSU" in var_names
        assert "EXROUTE" in var_names
        assert "EXDOSFRQ" in var_names
        assert "EXSTDTC" in var_names
        assert "EXENDTC" in var_names
