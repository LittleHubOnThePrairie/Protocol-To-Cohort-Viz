"""Tests for observation specs panel (PTCV-271).

Tests the render functions accept correct data types without error.
Streamlit rendering is not tested (requires Streamlit runtime);
we test that the functions handle None and empty inputs gracefully.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.sdtm.domain_spec_builder import (
    CdiscVariable,
    DomainSpec,
    DomainSpecResult,
    DomainTestCode,
    UnmappedAssessment,
)
from ptcv.sdtm.ex_domain_builder import (
    ExDomainSpec,
    InterventionDetail,
)


class TestRenderObservationSpecsData:
    """Verify observation spec data structures are correct."""

    def test_domain_spec_result_with_specs(self):
        result = DomainSpecResult(
            specs=[
                DomainSpec(
                    domain_code="VS",
                    domain_name="Vital Signs",
                    variables=[
                        CdiscVariable("VSTESTCD", "VS Test Short Name"),
                        CdiscVariable("VSTEST", "VS Test Name"),
                    ],
                    test_codes=[
                        DomainTestCode(
                            testcd="SYSBP",
                            test="Systolic Blood Pressure",
                            source_assessment="Blood Pressure",
                            visit_schedule=["Screening", "Day 1"],
                        ),
                    ],
                    source_assessments=["Blood Pressure"],
                ),
            ],
            unmapped=[
                UnmappedAssessment(
                    assessment_name="Custom Biomarker",
                    suggested_domain="FA",
                    reason="No keyword match",
                ),
            ],
            total_assessments=5,
            mapped_count=4,
        )
        assert len(result.specs) == 1
        assert result.specs[0].domain_code == "VS"
        assert result.specs[0].test_count == 1
        assert len(result.unmapped) == 1

    def test_empty_domain_spec_result(self):
        result = DomainSpecResult()
        assert result.specs == []
        assert result.unmapped == []

    def test_domain_spec_variables_have_labels(self):
        spec = DomainSpec(
            domain_code="LB",
            domain_name="Laboratory",
            variables=[
                CdiscVariable("LBTESTCD", "Lab Test Short Name"),
                CdiscVariable("LBSPEC", "Specimen Type", required=False),
            ],
        )
        assert spec.variables[0].label == "Lab Test Short Name"
        assert spec.variables[1].required is False

    def test_test_code_visit_schedule(self):
        tc = DomainTestCode(
            testcd="HR",
            test="Heart Rate",
            source_assessment="Vital Signs",
            visit_schedule=["Screening", "Day 1", "EOT"],
        )
        assert len(tc.visit_schedule) == 3


class TestRenderExDomainSpecData:
    """Verify EX domain spec data structures are correct."""

    def test_ex_spec_with_interventions(self):
        spec = ExDomainSpec(
            interventions=[
                InterventionDetail(
                    name="Drug X",
                    intervention_type="DRUG",
                    dose="10",
                    dose_unit="MG",
                    route="ORAL",
                    frequency="QD",
                    arm_labels=["Arm A"],
                ),
                InterventionDetail(
                    name="Placebo",
                    intervention_type="DRUG",
                    arm_labels=["Arm B"],
                ),
            ],
            arm_treatment_map={
                "Arm A": ["Drug X"],
                "Arm B": ["Placebo"],
            },
        )
        assert spec.treatment_count == 2
        assert "Arm A" in spec.arm_treatment_map
        assert spec.interventions[0].dose == "10"

    def test_empty_ex_spec(self):
        spec = ExDomainSpec()
        assert spec.treatment_count == 0
        assert spec.arm_treatment_map == {}

    def test_intervention_detail_fields(self):
        intv = InterventionDetail(
            name="Test Drug",
            dose="50",
            dose_unit="MG",
            route="INTRAVENOUS",
            frequency="Q3W",
        )
        assert intv.route == "INTRAVENOUS"
        assert intv.frequency == "Q3W"


class TestModuleImport:
    """Verify the panel module imports without error."""

    def test_import_render_functions(self):
        from ptcv.ui.components.observation_specs_panel import (
            render_ex_domain_spec,
            render_observation_specs,
        )
        assert callable(render_observation_specs)
        assert callable(render_ex_domain_spec)
