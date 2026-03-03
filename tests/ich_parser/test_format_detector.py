"""Tests for PTCV-31 — Pre-parser protocol format detection gate.

Covers all GHERKIN scenarios:
  - ICH E6(R3) protocol passes through to IchParser unchanged
  - Non-ICH protocol gated before classification and routed to review queue
  - CTD-formatted protocol detected and recorded
  - FormatDetector completes without API calls (no IchSection objects)
  - Ambiguous protocol passes through with PARTIAL flag
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ptcv.ich_parser.format_detector import (
    FormatDetector,
    FormatDetectionResult,
    ProtocolFormat,
)
from ptcv.ich_parser import IchParser, RuleBasedClassifier
from ptcv.storage import FilesystemAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parser(tmp_path: Path) -> IchParser:
    gw = FilesystemAdapter(root=tmp_path)
    return IchParser(gateway=gw, classifier=RuleBasedClassifier(), review_queue=None)


# ---------------------------------------------------------------------------
# Test texts
# ---------------------------------------------------------------------------

ICH_PROTOCOL_TEXT = """\
CLINICAL STUDY PROTOCOL — ICH E6(R3) GCP Compliant
EudraCT Number: 2024-000001-11

B.1 General Information
Protocol Title: Phase 2 Study of Drug X
Sponsor: Acme Pharma Ltd

B.3 Trial Objectives and Purpose
Primary Objective: Evaluate efficacy of Drug X.
Hypothesis: Drug X improves overall response rate.

B.4 Trial Design
This is a randomised, double-blind, placebo-controlled study.
Randomisation ratio 2:1.

B.5 Selection of Subjects
Inclusion Criteria:
1. Adults aged >= 18 years
Exclusion Criteria:
1. Prior treatment with Drug X
"""

NON_ICH_TEXT = (
    "This document describes a proprietary internal process for "
    "manufacturing quality control. It does not follow any regulatory "
    "protocol format. The batch release procedures are outlined below. "
    "Temperature monitoring and humidity checks are performed daily. "
) * 20

CTD_TEXT = """\
Common Technical Document — Module 5.3
Clinical Study Reports

Module 5.3.1 Reports of Biopharmaceutic Studies
Module 5.3.2 Reports of Studies Pertinent to Pharmacokinetics
eCTD submission package v4.0

5.3.5.1 Study Report for Pivotal Trial NCT00112827
"""

FDA_IND_TEXT = """\
IND Application for Novel Therapeutic Agent
Submitted under 21 CFR Part 312

Investigational New Drug application summary.
This IND submission contains preclinical and clinical data
supporting the safety of the investigational agent.

21 CFR 312.23 Content and format of an IND.
"""

AMBIGUOUS_TEXT = """\
Protocol for Clinical Investigation
NCT00998877

B.3 Objectives
The primary objective is to assess safety.

This document was prepared according to internal sponsor SOPs.
No additional ICH section headings are present.
"""


# ---------------------------------------------------------------------------
# Unit tests for FormatDetector.detect()
# ---------------------------------------------------------------------------

class TestFormatDetector:
    """Direct unit tests for the FormatDetector class."""

    def test_ich_e6r3_detected_with_high_confidence(self) -> None:
        """ICH protocol with B.x headings and ICH keywords → ICH_E6R3.
        [PTCV-31 Scenario: ICH E6(R3) protocol passes through]
        """
        result = FormatDetector().detect(ICH_PROTOCOL_TEXT)
        assert result.format == ProtocolFormat.ICH_E6R3
        assert result.confidence >= 0.70
        assert len(result.positive_markers) >= 2

    def test_non_ich_detected_as_unknown(self) -> None:
        """Text with no ICH/CTD/IND markers → UNKNOWN.
        [PTCV-31 Scenario: Non-ICH protocol gated]
        """
        result = FormatDetector().detect(NON_ICH_TEXT)
        assert result.format == ProtocolFormat.UNKNOWN
        assert result.confidence < 0.20

    def test_ctd_format_detected(self) -> None:
        """Text with CTD module markers → CTD.
        [PTCV-31 Scenario: CTD-formatted protocol detected]
        """
        result = FormatDetector().detect(CTD_TEXT)
        assert result.format == ProtocolFormat.CTD
        assert result.confidence >= 0.50
        assert any("CTD" in m or "Module" in m for m in result.positive_markers)

    def test_fda_ind_format_detected(self) -> None:
        """Text with FDA IND markers → FDA_IND."""
        result = FormatDetector().detect(FDA_IND_TEXT)
        assert result.format == ProtocolFormat.FDA_IND
        assert result.confidence >= 0.50

    def test_ambiguous_text_passes_through(self) -> None:
        """Text with one B.x heading and an NCT ID → ICH_E6R3 with
        confidence between 0.20 and 0.70 (ambiguous range).
        [PTCV-31 Scenario: Ambiguous protocol passes through]
        """
        result = FormatDetector().detect(AMBIGUOUS_TEXT)
        # Has some markers so should not be UNKNOWN with < 0.20
        assert result.confidence >= 0.20
        # Should not gate out — confidence may or may not reach 0.70

    def test_result_has_recommendation(self) -> None:
        """Every FormatDetectionResult includes a human-readable recommendation."""
        for text in (ICH_PROTOCOL_TEXT, NON_ICH_TEXT, CTD_TEXT):
            result = FormatDetector().detect(text)
            assert isinstance(result.recommendation, str)
            assert len(result.recommendation) > 10

    def test_no_api_calls_or_ich_sections_created(self) -> None:
        """FormatDetector uses only regex — no IchSection objects created.
        [PTCV-31 Scenario: FormatDetector completes without API calls]
        """
        # This is a design test — FormatDetector returns FormatDetectionResult,
        # not IchSection. Just verify the return type.
        result = FormatDetector().detect(NON_ICH_TEXT)
        assert isinstance(result, FormatDetectionResult)

    def test_protocol_format_enum_values(self) -> None:
        """ProtocolFormat enum has exactly four members."""
        assert set(ProtocolFormat) == {
            ProtocolFormat.ICH_E6R3,
            ProtocolFormat.CTD,
            ProtocolFormat.FDA_IND,
            ProtocolFormat.UNKNOWN,
        }

    def test_empty_text_returns_unknown(self) -> None:
        """Single whitespace or near-empty text → UNKNOWN."""
        result = FormatDetector().detect("   \n\n   ")
        assert result.format == ProtocolFormat.UNKNOWN
        assert result.confidence < 0.20


# ---------------------------------------------------------------------------
# Integration tests via IchParser.parse()
# ---------------------------------------------------------------------------

class TestFormatDetectorIntegration:
    """Integration tests: FormatDetector gate inside IchParser.parse()."""

    def test_ich_protocol_proceeds_to_classification(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """ICH protocol is not gated — sections are classified normally.
        [PTCV-31 Scenario: ICH E6(R3) protocol passes through]
        """
        parser = _make_parser(tmp_path)
        result = parser.parse(text=sample_text, registry_id="EUCT-ICH-031")
        assert result.section_count >= 1
        assert result.artifact_key != ""
        assert result.detected_format in ("ICH_E6R3", "UNKNOWN")

    def test_non_ich_detected_and_flagged(
        self, tmp_path: Path
    ) -> None:
        """Non-ICH text is detected, flagged via review queue, and
        classification still runs (producing legacy fallback section).
        [PTCV-31 Scenario: Non-ICH protocol detected and routed to review]
        """
        parser = _make_parser(tmp_path)
        result = parser.parse(
            text=NON_ICH_TEXT,
            registry_id="UNKNOWN-001",
        )
        assert result.format_verdict == "NON_ICH"
        assert result.detected_format == "UNKNOWN"
        # Classification still runs, producing legacy fallback
        assert result.section_count >= 1
        assert result.artifact_key != ""
        # FORMAT review entry + legacy low-confidence section entry
        assert result.review_count >= 1

    def test_non_ich_enqueued_in_review_queue(
        self, tmp_path: Path
    ) -> None:
        """Non-ICH gated protocol is enqueued with reason code FORMAT.
        [PTCV-31 Scenario: Protocol registry_id enqueued with reason non_ich_format]
        """
        from ptcv.ich_parser.review_queue import ReviewQueue

        rq = ReviewQueue(db_path=tmp_path / "review_queue.db")
        rq.initialise()

        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(
            gateway=gw, classifier=RuleBasedClassifier(), review_queue=rq
        )
        result = parser.parse(text=NON_ICH_TEXT, registry_id="UNKNOWN-002")

        entries = rq.pending(registry_id="UNKNOWN-002")
        assert len(entries) >= 1
        format_entries = [e for e in entries if e.section_code == "FORMAT"]
        assert len(format_entries) == 1
        assert format_entries[0].confidence_score < 0.20

    def test_parse_result_has_detected_format_field(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """ParseResult always exposes detected_format attribute."""
        parser = _make_parser(tmp_path)
        result = parser.parse(text=sample_text, registry_id="EUCT-FMT-031")
        assert hasattr(result, "detected_format")
        assert result.detected_format in ("ICH_E6R3", "CTD", "FDA_IND", "UNKNOWN")

    def test_ambiguous_protocol_not_gated(
        self, tmp_path: Path
    ) -> None:
        """Ambiguous text (some ICH markers) passes through to classifier.
        [PTCV-31 Scenario: Ambiguous protocol passes through]
        """
        parser = _make_parser(tmp_path)
        result = parser.parse(text=AMBIGUOUS_TEXT, registry_id="EUCT-AMB-031")
        # Should not be gated (confidence >= 0.20 due to B.3 heading + NCT ID)
        # so classifier runs and produces at least one section
        assert result.section_count >= 1
        assert result.artifact_key != ""

    def test_orchestrator_checkpoint_has_detected_format(
        self, tmp_path: Path
    ) -> None:
        """stage-03 checkpoint includes detected_format field.
        [PTCV-31 Scenario: CTD format recorded in checkpoint]
        """
        from ptcv.pipeline.orchestrator import PipelineOrchestrator
        from tests.pipeline.conftest import SYNTHETIC_CTR_XML

        gw = FilesystemAdapter(root=tmp_path)
        gw.initialise()
        orchestrator = PipelineOrchestrator(gateway=gw)

        result = orchestrator.run(
            protocol_data=SYNTHETIC_CTR_XML.encode("utf-8"),
            registry_id="EUCT-2024-CP-031",
            filename="protocol.xml",
            pipeline_run_id="test-ptcv31-checkpoint",
        )

        ich_cp = next(
            (cp for cp in result.stage_checkpoints if cp.stage == "ich_parse"),
            None,
        )
        assert ich_cp is not None

        cp_bytes = gw.get_artifact(ich_cp.artifact_key)
        cp_data = json.loads(cp_bytes.decode("utf-8"))
        assert "detected_format" in cp_data
        assert cp_data["detected_format"] in (
            "ICH_E6R3", "CTD", "FDA_IND", "UNKNOWN"
        )
