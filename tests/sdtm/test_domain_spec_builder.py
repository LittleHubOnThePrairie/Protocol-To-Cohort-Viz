"""Tests for domain spec builder (PTCV-246).

Tests SoA assessment → SDTM domain mapping, CDISC variable sets,
test code inference, specimen inference, and unmapped handling.
"""

from __future__ import annotations

import pytest

from ptcv.sdtm.domain_spec_builder import (
    CdiscVariable,
    DomainSpec,
    DomainSpecResult,
    DomainTestCode,
    UnmappedAssessment,
    build_domain_specs,
    _infer_test_codes,
    _infer_specimen,
)
from ptcv.soa_extractor.models import RawSoaTable


def _make_table(
    activities: list[tuple[str, list[bool]]],
    visit_headers: list[str] | None = None,
) -> RawSoaTable:
    if visit_headers is None:
        n = max((len(f) for _, f in activities), default=3)
        visit_headers = [f"V{i}" for i in range(1, n + 1)]
    return RawSoaTable(
        visit_headers=visit_headers,
        day_headers=[],
        activities=activities,
        section_code="B.4",
    )


class TestInferTestCodes:
    def test_blood_pressure(self):
        codes = _infer_test_codes("Blood Pressure")
        testcds = [c[0] for c in codes]
        assert "SYSBP" in testcds
        assert "DIABP" in testcds

    def test_vitals_produces_multiple(self):
        codes = _infer_test_codes("Vital Signs")
        assert len(codes) >= 4

    def test_hematology(self):
        codes = _infer_test_codes("Hematology")
        testcds = [c[0] for c in codes]
        assert "WBC" in testcds
        assert "HGB" in testcds

    def test_ecg(self):
        codes = _infer_test_codes("12-lead ECG")
        testcds = [c[0] for c in codes]
        assert "QTCF" in testcds

    def test_unknown_returns_empty(self):
        assert _infer_test_codes("Custom Biomarker Panel") == []


class TestInferSpecimen:
    def test_hematology_is_blood(self):
        assert _infer_specimen("Hematology") == "BLOOD"

    def test_chemistry_is_serum(self):
        assert _infer_specimen("Chemistry") == "SERUM"

    def test_urinalysis_is_urine(self):
        assert _infer_specimen("Urinalysis") == "URINE"

    def test_coagulation_is_plasma(self):
        assert _infer_specimen("Coagulation Profile") == "PLASMA"

    def test_unknown_returns_empty(self):
        assert _infer_specimen("Physical Exam") == ""


class TestBuildDomainSpecs:
    def test_vs_domain_from_vitals(self):
        table = _make_table([
            ("Vital Signs", [True, True, False]),
        ], ["Screening", "Day 1", "EOT"])

        result = build_domain_specs(table)
        vs = result.get_spec("VS")

        assert vs is not None
        assert vs.domain_code == "VS"
        assert vs.variable_count > 0
        assert vs.test_count >= 4  # SYSBP, DIABP, HR, TEMP, RESP, WEIGHT
        assert "Vital Signs" in vs.source_assessments

    def test_lb_domain_with_specimen(self):
        table = _make_table([
            ("Hematology", [True, True]),
            ("Chemistry", [True, False]),
        ], ["Screening", "Day 1"])

        result = build_domain_specs(table)
        lb = result.get_spec("LB")

        assert lb is not None
        assert lb.domain_code == "LB"
        assert "BLOOD" in lb.specimen_type
        assert "SERUM" in lb.specimen_type
        assert lb.test_count >= 5  # WBC, RBC, HGB, HCT, PLAT + chemistry

    def test_eg_domain_from_ecg(self):
        table = _make_table([
            ("12-lead ECG", [True, False, True]),
        ], ["Screening", "Day 1", "EOT"])

        result = build_domain_specs(table)
        eg = result.get_spec("EG")

        assert eg is not None
        testcds = [tc.testcd for tc in eg.test_codes]
        assert "QTCF" in testcds

    def test_pe_domain_from_physical_exam(self):
        table = _make_table([
            ("Physical Examination", [True, True]),
        ], ["Screening", "Day 1"])

        result = build_domain_specs(table)
        pe = result.get_spec("PE")

        assert pe is not None
        assert pe.domain_code == "PE"
        var_names = [v.name for v in pe.variables]
        assert "PETESTCD" in var_names
        assert "PEBODSYS" in var_names

    def test_unmapped_assessments(self):
        table = _make_table([
            ("Physical Exam", [True]),
            ("Custom Biomarker Panel", [True]),
            ("Genomic Sequencing", [True]),
        ], ["V1"])

        result = build_domain_specs(table)

        assert result.total_assessments == 3
        assert result.mapped_count >= 1
        assert len(result.unmapped) >= 1
        unmapped_names = [u.assessment_name for u in result.unmapped]
        assert any("Biomarker" in n or "Genomic" in n for n in unmapped_names)

    def test_visit_schedule_preserved(self):
        table = _make_table([
            ("ECG", [True, False, True]),
        ], ["Screening", "Day 1", "EOT"])

        result = build_domain_specs(table)
        eg = result.get_spec("EG")

        assert eg is not None
        # First test code should have visit schedule
        tc = eg.test_codes[0]
        assert "Screening" in tc.visit_schedule
        assert "Day 1" not in tc.visit_schedule
        assert "EOT" in tc.visit_schedule

    def test_target_domains_filter(self):
        table = _make_table([
            ("Vital Signs", [True]),
            ("Hematology", [True]),
            ("ECG", [True]),
            ("Physical Exam", [True]),
        ], ["V1"])

        result = build_domain_specs(table, target_domains={"VS", "LB"})

        domain_codes = [s.domain_code for s in result.specs]
        assert "VS" in domain_codes
        assert "LB" in domain_codes
        assert "EG" not in domain_codes
        assert "PE" not in domain_codes

    def test_multiple_domains(self):
        table = _make_table([
            ("Vital Signs", [True, True]),
            ("Hematology", [True, True]),
            ("Chemistry", [True, False]),
            ("ECG", [True, False]),
            ("Physical Exam", [True, True]),
            ("Adverse Events", [True, True]),
        ], ["Screening", "Day 1"])

        result = build_domain_specs(table)

        domain_codes = {s.domain_code for s in result.specs}
        assert "VS" in domain_codes
        assert "LB" in domain_codes
        assert "EG" in domain_codes
        assert "PE" in domain_codes
        assert "AE" in domain_codes

    def test_empty_table(self):
        table = _make_table([], ["V1"])
        result = build_domain_specs(table)

        assert result.total_assessments == 0
        assert result.specs == []
        assert result.unmapped == []

    def test_no_duplicate_test_codes(self):
        """Test same test code from multiple assessments isn't duplicated."""
        table = _make_table([
            ("Vital Signs", [True]),
            ("Blood Pressure", [True]),
        ], ["V1"])

        result = build_domain_specs(table)
        vs = result.get_spec("VS")

        assert vs is not None
        testcds = [tc.testcd for tc in vs.test_codes]
        # SYSBP should appear only once even though both vitals
        # and blood pressure map to it
        assert testcds.count("SYSBP") == 1


class TestDomainSpec:
    def test_variable_count(self):
        spec = DomainSpec(
            domain_code="VS", domain_name="Vital Signs",
            variables=[CdiscVariable("A", "A"), CdiscVariable("B", "B")],
        )
        assert spec.variable_count == 2

    def test_test_count(self):
        spec = DomainSpec(
            domain_code="VS", domain_name="Vital Signs",
            test_codes=[
                DomainTestCode("HR", "Heart Rate"),
                DomainTestCode("TEMP", "Temperature"),
            ],
        )
        assert spec.test_count == 2


class TestDomainSpecResult:
    def test_get_spec(self):
        result = DomainSpecResult(specs=[
            DomainSpec("VS", "Vital Signs"),
            DomainSpec("LB", "Labs"),
        ])
        assert result.get_spec("VS") is not None
        assert result.get_spec("LB") is not None
        assert result.get_spec("XX") is None
