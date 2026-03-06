"""Tests for ICH M11 CeSHarP protocol parser (PTCV-56).

Covers all 5 GHERKIN acceptance criteria scenarios:

  Scenario 1: M11 format detected and routed correctly
  Scenario 2: SoA extracted with full cell metadata
  Scenario 3: ICH sections extracted at confidence 1.0
  Scenario 4: USDM models populated from M11 data
  Scenario 5: Backward compatibility with PDF pipeline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest

from ptcv.extraction.format_detector import (
    FormatDetector as ExtractionFormatDetector,
    ProtocolFormat as ExtractionFormat,
)
from ptcv.ich_parser.format_detector import (
    FormatDetector as IchFormatDetector,
    ProtocolFormat as IchFormat,
)
from ptcv.ich_parser.m11_parser import M11ProtocolParser, M11ParseResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def m11_protocol() -> dict:
    """Realistic ICH M11 CeSHarP protocol JSON."""
    return {
        "m11_version": "1.0",
        "cesharp_version": "1.0",
        "protocol_version": "2.0",
        "registry_id": "NCT00112827",
        "sponsor": "Test Pharma Inc",
        "phase": "Phase III",
        "indication": "Hypertension",
        "sections": {
            "B.1": {
                "text_excerpt": (
                    "A Phase III, Double-Blind, Placebo-Controlled Study "
                    "of Drug X in Patients with Hypertension"
                ),
                "word_count": 16,
                "sponsor": "Test Pharma Inc",
                "phase": "Phase III",
            },
            "B.2": "Background: Drug X is a novel antihypertensive agent.",
            "B.3": {
                "text_excerpt": (
                    "Primary objective: Evaluate efficacy of Drug X "
                    "in reducing systolic blood pressure."
                ),
                "primary_objective": "Evaluate efficacy of Drug X",
                "primary_endpoint": "Change in systolic blood pressure",
            },
            "B.4": (
                "Parallel-group, double-blind, placebo-controlled design. "
                "Arm A: Drug X 10 mg QD. Arm B: Placebo QD."
            ),
            "B.5": {
                "text_excerpt": (
                    "Inclusion: Age 18-65, diagnosed hypertension. "
                    "Exclusion: Prior cardiovascular event."
                ),
                "inclusion_criteria": [
                    "Age 18-65 years",
                    "Diagnosed hypertension",
                ],
                "exclusion_criteria": [
                    "Prior cardiovascular event",
                    "Renal impairment (eGFR < 30 mL/min)",
                ],
            },
        },
        "epochs": [
            {"name": "Screening", "type": "Screening"},
            {"name": "Treatment", "type": "Treatment"},
            {"name": "Follow-up", "type": "Follow-up"},
        ],
        "schedule_of_activities": {
            "visits": [
                {
                    "name": "Screening",
                    "type": "Screening",
                    "day_offset": -14,
                    "window_early": 3,
                    "window_late": 0,
                    "mandatory": True,
                },
                {
                    "name": "Baseline",
                    "type": "Treatment",
                    "day_offset": 1,
                    "window_early": 1,
                    "window_late": 1,
                    "mandatory": True,
                },
                {
                    "name": "Week 4",
                    "type": "Treatment",
                    "day_offset": 29,
                    "window_early": 3,
                    "window_late": 3,
                    "mandatory": True,
                },
                {
                    "name": "End of Study",
                    "type": "Follow-up",
                    "day_offset": 85,
                    "window_early": 3,
                    "window_late": 7,
                    "mandatory": True,
                },
            ],
            "assessments": [
                {
                    "name": "Vital Signs",
                    "cells": [
                        {"value": "X"},
                        {"value": "X"},
                        {"value": "X"},
                        {"value": "X"},
                    ],
                },
                {
                    "name": "Complete Blood Count",
                    "cells": [
                        {"value": "X"},
                        {"value": "X"},
                        {"value": "O"},
                        {"value": "X"},
                    ],
                },
                {
                    "name": "12-lead ECG",
                    "cells": [
                        {"value": "X"},
                        {"value": "C", "condition": "If clinically indicated"},
                        {"value": ""},
                        {"value": "X"},
                    ],
                },
                {
                    "name": "Physical Examination",
                    "cells": [
                        {"value": "X"},
                        {"value": "X"},
                        {"value": ""},
                        {"value": "O"},
                    ],
                },
            ],
        },
    }


@pytest.fixture()
def m11_json(m11_protocol: dict) -> str:
    """M11 protocol as JSON string."""
    return json.dumps(m11_protocol)


@pytest.fixture()
def parser() -> M11ProtocolParser:
    return M11ProtocolParser()


@pytest.fixture()
def parsed_result(parser: M11ProtocolParser, m11_json: str) -> M11ParseResult:
    """Pre-parsed M11 result for reuse across tests."""
    return parser.parse(m11_json, registry_id="NCT00112827")


# ---------------------------------------------------------------------------
# Scenario 1: M11 format detected and routed correctly
# ---------------------------------------------------------------------------

class TestM11FormatDetection:
    """Scenario: M11 format detected and routed correctly."""

    def test_extraction_detector_recognises_m11_json_marker(self) -> None:
        """Given a JSON file with m11_version, detect ICH_M11."""
        data = b'{"m11_version": "1.0", "cesharp_version": "1.0"}'
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data, filename="protocol.json")
        assert result == ExtractionFormat.ICH_M11

    def test_extraction_detector_recognises_m11_xml_namespace(self) -> None:
        """Given an XML file with CeSHarP namespace, detect ICH_M11."""
        data = b'<?xml version="1.0"?><M11Protocol xmlns="ich.org/m11">'
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data, filename="protocol.xml")
        assert result == ExtractionFormat.ICH_M11

    def test_extraction_detector_recognises_m11_extension(self) -> None:
        """Given a file with .m11 extension, detect ICH_M11."""
        data = b"some content"
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data, filename="protocol.m11")
        assert result == ExtractionFormat.ICH_M11

    def test_extraction_detector_recognises_cesharp_extension(self) -> None:
        """Given a file with .cesharp extension, detect ICH_M11."""
        data = b"some content"
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data, filename="study.cesharp")
        assert result == ExtractionFormat.ICH_M11

    def test_ich_detector_recognises_m11_text(self) -> None:
        """Given text with ICH M11 and CeSHarP markers, detect ICH_M11."""
        text = (
            "This protocol follows ICH M11 guidelines and "
            "implements the CeSHarP specification for "
            "machine-readable protocol structure."
        )
        detector = IchFormatDetector()
        result = detector.detect(text)
        assert result.format == IchFormat.ICH_M11

    def test_ich_detector_m11_confidence_high(self) -> None:
        """M11 detection should have high confidence when markers present."""
        text = (
            "ICH M11 CeSHarP machine-readable protocol "
            "m11_version 1.0"
        )
        detector = IchFormatDetector()
        result = detector.detect(text)
        assert result.format == IchFormat.ICH_M11
        assert result.confidence >= 0.80

    def test_ich_detector_m11_recommendation_mentions_m11_parser(self) -> None:
        """Recommendation should mention M11ProtocolParser."""
        text = "ICH M11 CeSHarP structured protocol"
        detector = IchFormatDetector()
        result = detector.detect(text)
        assert "M11" in result.recommendation

    def test_pdf_not_detected_as_m11(self) -> None:
        """Existing PDF detection unchanged — backward compatible."""
        data = b"%PDF-1.7 some pdf content"
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data, filename="protocol.pdf")
        assert result == ExtractionFormat.PDF

    def test_ctr_xml_not_detected_as_m11(self) -> None:
        """Existing CTR-XML detection unchanged — backward compatible."""
        data = b'<?xml version="1.0"?><ODM FileType="Snapshot">'
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data, filename="protocol.xml")
        assert result == ExtractionFormat.CTR_XML


# ---------------------------------------------------------------------------
# Scenario 2: SoA extracted with full cell metadata
# ---------------------------------------------------------------------------

class TestSoaCellMetadata:
    """Scenario: SoA extracted with full cell metadata."""

    def test_soa_table_extracted(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """SoA table should be non-None when protocol has SoA."""
        assert parsed_result.soa_table is not None

    def test_visit_headers_populated(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Visit headers should match M11 visit names."""
        assert parsed_result.soa_table is not None
        headers = parsed_result.soa_table.visit_headers
        assert headers == ["Screening", "Baseline", "Week 4", "End of Study"]

    def test_day_headers_from_offsets(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Day headers should be derived from M11 day_offset values."""
        assert parsed_result.soa_table is not None
        days = parsed_result.soa_table.day_headers
        assert days == ["Day -14", "Day 1", "Day 29", "Day 85"]

    def test_required_cells_detected(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """X cells should map to status=required."""
        required = [
            m for m in parsed_result.cell_metadata
            if m["status"] == "required"
        ]
        assert len(required) > 0

    def test_optional_cells_detected(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """O cells should map to status=optional."""
        optional = [
            m for m in parsed_result.cell_metadata
            if m["status"] == "optional"
        ]
        assert len(optional) > 0

    def test_conditional_cells_detected(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """C cells should map to status=conditional."""
        conditional = [
            m for m in parsed_result.cell_metadata
            if m["status"] == "conditional"
        ]
        assert len(conditional) > 0

    def test_conditional_cell_has_condition_text(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Conditional cells should include condition text from M11."""
        conditional = [
            m for m in parsed_result.cell_metadata
            if m["status"] == "conditional"
        ]
        assert any(m["condition"] != "" for m in conditional)
        ecg_cond = [
            m for m in conditional
            if m["assessment"] == "12-lead ECG"
        ]
        assert len(ecg_cond) > 0
        assert "clinically indicated" in ecg_cond[0]["condition"]

    def test_empty_cells_are_not_applicable(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Empty cells should map to status=not_applicable."""
        na = [
            m for m in parsed_result.cell_metadata
            if m["status"] == "not_applicable"
        ]
        assert len(na) > 0

    def test_visit_timing_windows_in_timepoints(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Visit timing windows should be extracted from M11 metadata."""
        screening = [
            tp for tp in parsed_result.timepoints
            if tp.visit_name == "Screening"
        ]
        assert len(screening) == 1
        assert screening[0].window_early == 3
        assert screening[0].window_late == 0

    def test_activities_match_assessments(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Activities should match the SoA assessment count."""
        assert parsed_result.soa_table is not None
        assert len(parsed_result.soa_table.activities) == 4


# ---------------------------------------------------------------------------
# Scenario 3: ICH sections extracted at confidence 1.0
# ---------------------------------------------------------------------------

class TestIchSectionParsing:
    """Scenario: ICH sections extracted at confidence 1.0."""

    def test_sections_populated(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Sections should be populated from M11 structured data."""
        assert len(parsed_result.sections) > 0

    def test_all_sections_confidence_1(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """All sections should have confidence_score=1.0."""
        for sec in parsed_result.sections:
            assert sec.confidence_score == 1.0, (
                f"{sec.section_code} has confidence {sec.confidence_score}"
            )

    def test_all_sections_review_not_required(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """All sections should have review_required=False."""
        for sec in parsed_result.sections:
            assert sec.review_required is False, (
                f"{sec.section_code} has review_required=True"
            )

    def test_all_sections_not_legacy(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """All sections should have legacy_format=False."""
        for sec in parsed_result.sections:
            assert sec.legacy_format is False

    def test_section_codes_match_m11_data(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Section codes should match keys from M11 sections dict."""
        codes = {sec.section_code for sec in parsed_result.sections}
        assert codes == {"B.1", "B.2", "B.3", "B.4", "B.5"}

    def test_section_names_correct(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Section names should match ICH E6(R3) naming."""
        name_map = {s.section_code: s.section_name for s in parsed_result.sections}
        assert name_map["B.1"] == "General Information"
        assert name_map["B.4"] == "Trial Design"
        assert name_map["B.5"] == "Selection of Subjects"

    def test_section_content_json_valid(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """content_json should be valid JSON for all sections."""
        for sec in parsed_result.sections:
            parsed = json.loads(sec.content_json)
            assert isinstance(parsed, dict)

    def test_section_registry_id_set(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Registry ID should be set on all sections."""
        for sec in parsed_result.sections:
            assert sec.registry_id == "NCT00112827"

    def test_section_timestamp_set(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """extraction_timestamp_utc should be set on all sections."""
        for sec in parsed_result.sections:
            assert sec.extraction_timestamp_utc != ""

    def test_dict_section_preserves_structure(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Dict-based section content should preserve original structure."""
        b1 = [s for s in parsed_result.sections if s.section_code == "B.1"]
        assert len(b1) == 1
        content = json.loads(b1[0].content_json)
        assert "sponsor" in content

    def test_string_section_wrapped_in_dict(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """String-based section content should be wrapped with text_excerpt."""
        b2 = [s for s in parsed_result.sections if s.section_code == "B.2"]
        assert len(b2) == 1
        content = json.loads(b2[0].content_json)
        assert "text_excerpt" in content
        assert "word_count" in content


# ---------------------------------------------------------------------------
# Scenario 4: USDM models populated from M11 data
# ---------------------------------------------------------------------------

class TestUsdmPopulation:
    """Scenario: USDM models populated from M11 data."""

    def test_timepoints_created(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """UsdmTimepoints should be created for each visit."""
        assert len(parsed_result.timepoints) == 4

    def test_timepoint_day_offsets(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Timepoints should have correct day_offset values."""
        offsets = [tp.day_offset for tp in parsed_result.timepoints]
        assert offsets == [-14, 1, 29, 85]

    def test_timepoint_visit_types(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Timepoints should have correct visit_type from M11."""
        types = [tp.visit_type for tp in parsed_result.timepoints]
        assert types == ["Screening", "Treatment", "Treatment", "Follow-up"]

    def test_timepoint_mandatory_flag(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """All timepoints should be mandatory in this protocol."""
        assert all(tp.mandatory for tp in parsed_result.timepoints)

    def test_timepoint_no_review_required(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """M11 timepoints should never need review (deterministic)."""
        assert all(not tp.review_required for tp in parsed_result.timepoints)

    def test_activities_created(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """UsdmActivities should be created for each assessment."""
        assert len(parsed_result.activities) == 4

    def test_activity_names(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Activity names should match M11 assessment names."""
        names = [a.activity_name for a in parsed_result.activities]
        assert "Vital Signs" in names
        assert "Complete Blood Count" in names
        assert "12-lead ECG" in names

    def test_activity_types_classified(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Activity types should be classified from names."""
        type_map = {
            a.activity_name: a.activity_type for a in parsed_result.activities
        }
        assert type_map["Vital Signs"] == "Vital"
        assert type_map["Complete Blood Count"] == "Lab"
        assert type_map["12-lead ECG"] == "ECG"

    def test_scheduled_instances_created(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """ScheduledInstances should exist for non-empty cells."""
        assert len(parsed_result.scheduled_instances) > 0

    def test_scheduled_instances_exclude_empty_cells(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Empty cells should not produce ScheduledInstances."""
        # 4 assessments × 4 visits = 16 cells
        # 3 empty cells (ECG week4, PhysExam week4, one more)
        assert len(parsed_result.scheduled_instances) < 16

    def test_epochs_created(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """UsdmEpochs should be created from M11 epoch definitions."""
        assert len(parsed_result.epochs) == 3

    def test_epoch_names_correct(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Epoch names should match M11 data."""
        names = [e.epoch_name for e in parsed_result.epochs]
        assert names == ["Screening", "Treatment", "Follow-up"]

    def test_epoch_order_sequential(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Epoch order should be sequential starting at 1."""
        orders = [e.order for e in parsed_result.epochs]
        assert orders == [1, 2, 3]

    def test_all_usdm_entities_have_timestamps(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """All USDM entities should have extraction_timestamp_utc set."""
        for tp in parsed_result.timepoints:
            assert tp.extraction_timestamp_utc != ""
        for act in parsed_result.activities:
            assert act.extraction_timestamp_utc != ""
        for inst in parsed_result.scheduled_instances:
            assert inst.extraction_timestamp_utc != ""
        for ep in parsed_result.epochs:
            assert ep.extraction_timestamp_utc != ""

    def test_all_usdm_entities_share_run_id(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """All USDM entities should share the same run_id."""
        run_id = parsed_result.run_id
        for tp in parsed_result.timepoints:
            assert tp.run_id == run_id
        for act in parsed_result.activities:
            assert act.run_id == run_id
        for inst in parsed_result.scheduled_instances:
            assert inst.run_id == run_id

    def test_timepoint_epoch_linkage(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """Timepoints should reference valid epoch IDs."""
        epoch_ids = {e.epoch_id for e in parsed_result.epochs}
        for tp in parsed_result.timepoints:
            assert tp.epoch_id in epoch_ids


# ---------------------------------------------------------------------------
# Scenario 5: Backward compatibility with PDF pipeline
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Scenario: Backward compatibility with PDF pipeline."""

    def test_pdf_still_detected(self) -> None:
        """PDF format detection unchanged after M11 additions."""
        data = b"%PDF-1.7 document content"
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data)
        assert result == ExtractionFormat.PDF

    def test_ctr_xml_still_detected(self) -> None:
        """CTR-XML detection unchanged after M11 additions."""
        data = b'<?xml version="1.0"?><ODM FileType="Snapshot">'
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data)
        assert result == ExtractionFormat.CTR_XML

    def test_word_still_detected(self) -> None:
        """Word format detection unchanged after M11 additions."""
        data = b"PK\x03\x04 docx content"
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data)
        assert result == ExtractionFormat.WORD

    def test_unknown_still_detected(self) -> None:
        """Unknown format returns UNKNOWN as before."""
        data = b"random unknown content"
        detector = ExtractionFormatDetector()
        result = detector.detect_from_bytes(data)
        assert result == ExtractionFormat.UNKNOWN

    def test_ich_e6r3_text_still_detected(self) -> None:
        """ICH E6(R3) text detection unchanged."""
        text = (
            "B.1 General Information\n"
            "B.3 Trial Objectives\n"
            "B.4 Trial Design\n"
            "B.5 Selection of Subjects\n"
            "ICH E6(R3) Appendix B\n"
            "EudraCT Number: 2024-001234-56\n"
            "GCP compliance reference"
        )
        detector = IchFormatDetector()
        result = detector.detect(text)
        assert result.format == IchFormat.ICH_E6R3

    def test_extraction_enum_has_all_original_values(self) -> None:
        """Extraction ProtocolFormat enum retains all original members."""
        assert hasattr(ExtractionFormat, "PDF")
        assert hasattr(ExtractionFormat, "CTR_XML")
        assert hasattr(ExtractionFormat, "WORD")
        assert hasattr(ExtractionFormat, "UNKNOWN")
        assert hasattr(ExtractionFormat, "ICH_M11")

    def test_ich_enum_has_all_original_values(self) -> None:
        """ICH ProtocolFormat enum retains all original members."""
        assert hasattr(IchFormat, "ICH_E6R3")
        assert hasattr(IchFormat, "CTD")
        assert hasattr(IchFormat, "FDA_IND")
        assert hasattr(IchFormat, "UNKNOWN")
        assert hasattr(IchFormat, "ICH_M11")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_json_raises_value_error(
        self, parser: M11ProtocolParser,
    ) -> None:
        """Invalid JSON should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid M11 JSON"):
            parser.parse("not valid json {{{")

    def test_non_object_json_raises_value_error(
        self, parser: M11ProtocolParser,
    ) -> None:
        """Non-object JSON should raise ValueError."""
        with pytest.raises(ValueError, match="JSON object"):
            parser.parse("[1, 2, 3]")

    def test_bytes_input_accepted(
        self, parser: M11ProtocolParser,
    ) -> None:
        """Parser should accept bytes input."""
        data = json.dumps({"sections": {}, "m11_version": "1.0"}).encode()
        result = parser.parse(data, registry_id="NCT99999999")
        assert result.registry_id == "NCT99999999"

    def test_empty_protocol_produces_empty_result(
        self, parser: M11ProtocolParser,
    ) -> None:
        """Empty protocol (no sections, no SoA) still returns result."""
        result = parser.parse("{}", registry_id="NCT00000001")
        assert len(result.sections) == 0
        assert result.soa_table is None
        assert len(result.timepoints) == 0

    def test_registry_id_from_protocol(
        self, parser: M11ProtocolParser,
    ) -> None:
        """Registry ID should be extracted from protocol if not provided."""
        data = json.dumps({"registry_id": "NCT12345678"})
        result = parser.parse(data)
        assert result.registry_id == "NCT12345678"

    def test_m11_metadata_extracted(
        self, parsed_result: M11ParseResult,
    ) -> None:
        """M11 metadata should include version and sponsor."""
        meta = parsed_result.m11_metadata
        assert meta["m11_version"] == "1.0"
        assert meta["sponsor"] == "Test Pharma Inc"
        assert meta["phase"] == "Phase III"

    def test_no_epochs_infers_from_visits(
        self, parser: M11ProtocolParser,
    ) -> None:
        """When no epochs defined, infer from visit types."""
        data = json.dumps({
            "schedule_of_activities": {
                "visits": [
                    {"name": "V1", "type": "Screening", "day_offset": -7},
                    {"name": "V2", "type": "Treatment", "day_offset": 1},
                ],
                "assessments": [
                    {"name": "Lab", "cells": ["X", "X"]},
                ],
            },
        })
        result = parser.parse(data)
        assert len(result.epochs) >= 1
        epoch_types = {e.epoch_type for e in result.epochs}
        assert "Screening" in epoch_types

    def test_simple_cell_values_accepted(
        self, parser: M11ProtocolParser,
    ) -> None:
        """Cells can be simple strings instead of dicts."""
        data = json.dumps({
            "schedule_of_activities": {
                "visits": [
                    {"name": "V1", "day_offset": 1},
                    {"name": "V2", "day_offset": 14},
                ],
                "assessments": [
                    {"name": "Lab", "cells": ["X", "O"]},
                    {"name": "ECG", "cells": ["X", ""]},
                ],
            },
        })
        result = parser.parse(data)
        assert result.soa_table is not None
        assert len(result.soa_table.activities) == 2
        # Lab: X, O → both scheduled
        assert result.soa_table.activities[0][1] == [True, True]
        # ECG: X, "" → first scheduled, second not
        assert result.soa_table.activities[1][1] == [True, False]

    def test_activity_type_explicit_override(
        self, parser: M11ProtocolParser,
    ) -> None:
        """Explicit activity_type in M11 data overrides classification."""
        data = json.dumps({
            "schedule_of_activities": {
                "visits": [{"name": "V1", "day_offset": 1}],
                "assessments": [
                    {
                        "name": "Custom Assessment",
                        "activity_type": "Procedure",
                        "cells": ["X"],
                    },
                ],
            },
        })
        result = parser.parse(data)
        assert result.activities[0].activity_type == "Procedure"
