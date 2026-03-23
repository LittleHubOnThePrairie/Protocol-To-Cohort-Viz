"""Shared fixtures for SDTM tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest

from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    AssembledSection,
    QueryExtractionHit,
    SourceReference,
)
from ptcv.soa_extractor.models import RawSoaTable, UsdmTimepoint
from ptcv.storage.filesystem_adapter import FilesystemAdapter


# ---------------------------------------------------------------------------
# ICH Section fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def b1_section() -> IchSection:
    """ICH B.1 section with study title, phase, sponsor."""
    return IchSection(
        run_id="ich-run-1",
        source_run_id="",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code="B.1",
        section_name="General Information",
        content_json=json.dumps({
            "text_excerpt": (
                "A Phase III, Double-Blind, Placebo-Controlled, Parallel-Group "
                "Study of Drug X in Patients with Hypertension\n"
                "Sponsor: Test Pharma Inc\n"
                "Indication: Hypertension\n"
                "Protocol version 1.0"
            ),
            "word_count": 28,
        }),
        confidence_score=0.92,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
    )


@pytest.fixture()
def b4_section() -> IchSection:
    """ICH B.4 section with trial design and arms."""
    return IchSection(
        run_id="ich-run-1",
        source_run_id="",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code="B.4",
        section_name="Trial Design",
        content_json=json.dumps({
            "text_excerpt": (
                "This is a randomized, parallel-group, double-blind study.\n"
                "Arm A: Drug X 10 mg once daily\n"
                "Arm B: Placebo once daily\n"
                "Screening period followed by treatment period then follow-up."
            ),
            "word_count": 40,
        }),
        confidence_score=0.88,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
    )


@pytest.fixture()
def b5_section() -> IchSection:
    """ICH B.5 section with inclusion/exclusion criteria."""
    return IchSection(
        run_id="ich-run-1",
        source_run_id="",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code="B.5",
        section_name="Selection of Subjects",
        content_json=json.dumps({
            "text_excerpt": (
                "Inclusion criteria:\n"
                "1. Age 18-65 years\n"
                "2. Diagnosis of hypertension\n"
                "3. Written informed consent\n"
                "Exclusion criteria:\n"
                "1. Prior cardiovascular event\n"
                "2. Renal impairment (eGFR < 30 mL/min)"
            ),
            "word_count": 45,
        }),
        confidence_score=0.95,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
    )


@pytest.fixture()
def b3_section() -> IchSection:
    """ICH B.3 section with trial objectives."""
    return IchSection(
        run_id="ich-run-1",
        source_run_id="",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code="B.3",
        section_name="Trial Objectives and Purpose",
        content_json=json.dumps({
            "text_excerpt": (
                "The primary objective is to evaluate the efficacy of "
                "Drug X 10 mg in reducing systolic blood pressure compared "
                "to placebo in patients with essential hypertension.\n"
                "Secondary objectives include assessment of safety and "
                "tolerability of Drug X."
            ),
            "word_count": 35,
        }),
        confidence_score=0.90,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
    )


@pytest.fixture()
def b6_section() -> IchSection:
    """ICH B.6 section with discontinuation/stop rules."""
    return IchSection(
        run_id="ich-run-1",
        source_run_id="",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code="B.6",
        section_name="Discontinuation and Withdrawal",
        content_json=json.dumps({
            "text_excerpt": (
                "Subjects may withdraw at any time.\n"
                "The study will be discontinued if more than 10% of "
                "subjects experience Grade 4 adverse events.\n"
                "Individual stopping rules apply for liver enzymes."
            ),
            "word_count": 30,
        }),
        confidence_score=0.88,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
    )


@pytest.fixture()
def b7_section() -> IchSection:
    """ICH B.7 section with treatment details."""
    return IchSection(
        run_id="ich-run-1",
        source_run_id="",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code="B.7",
        section_name="Treatment of Participants",
        content_json=json.dumps({
            "text_excerpt": (
                "Drug X 10 mg tablet administered orally once daily.\n"
                "Matching placebo tablet administered orally once daily.\n"
                "Treatment duration is 12 weeks."
            ),
            "word_count": 22,
        }),
        confidence_score=0.90,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
    )


@pytest.fixture()
def b10_section() -> IchSection:
    """ICH B.10 section with statistics/sample size."""
    return IchSection(
        run_id="ich-run-1",
        source_run_id="",
        source_sha256="abc123",
        registry_id="NCT00112827",
        section_code="B.10",
        section_name="Statistics",
        content_json=json.dumps({
            "text_excerpt": (
                "A total of 200 subjects will be randomized in a 1:1 ratio.\n"
                "Sample size was determined to provide 90% power to detect "
                "a 5 mmHg difference in systolic blood pressure.\n"
                "The primary analysis will use the ITT population."
            ),
            "word_count": 35,
        }),
        confidence_score=0.92,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
    )


@pytest.fixture()
def all_sections(
    b1_section, b3_section, b4_section, b5_section,
    b6_section, b7_section, b10_section,
) -> list[IchSection]:
    return [
        b1_section, b3_section, b4_section, b5_section,
        b6_section, b7_section, b10_section,
    ]


# ---------------------------------------------------------------------------
# USDM Timepoint fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def timepoints() -> list[UsdmTimepoint]:
    """Realistic set of USDM timepoints for a parallel-group study."""
    return [
        UsdmTimepoint(
            run_id="usdm-run-1",
            source_run_id="ich-run-1",
            source_sha256="def456",
            registry_id="NCT00112827",
            timepoint_id="tp-1",
            epoch_id="epoch-1",
            visit_name="Screening",
            visit_type="Screening",
            day_offset=-14,
            window_early=3,
            window_late=0,
            mandatory=True,
            extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
        ),
        UsdmTimepoint(
            run_id="usdm-run-1",
            source_run_id="ich-run-1",
            source_sha256="def456",
            registry_id="NCT00112827",
            timepoint_id="tp-2",
            epoch_id="epoch-1",
            visit_name="Baseline",
            visit_type="Baseline",
            day_offset=1,
            window_early=1,
            window_late=1,
            mandatory=True,
            extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
        ),
        UsdmTimepoint(
            run_id="usdm-run-1",
            source_run_id="ich-run-1",
            source_sha256="def456",
            registry_id="NCT00112827",
            timepoint_id="tp-3",
            epoch_id="epoch-2",
            visit_name="Week 4",
            visit_type="Treatment",
            day_offset=22,
            window_early=3,
            window_late=3,
            mandatory=True,
            extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
        ),
        UsdmTimepoint(
            run_id="usdm-run-1",
            source_run_id="ich-run-1",
            source_sha256="def456",
            registry_id="NCT00112827",
            timepoint_id="tp-4",
            epoch_id="epoch-2",
            visit_name="End of Study",
            visit_type="End of Study",
            day_offset=85,
            window_early=3,
            window_late=7,
            mandatory=True,
            extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
        ),
    ]


# ---------------------------------------------------------------------------
# Storage fixture
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Query pipeline fixtures (PTCV-140)
# ---------------------------------------------------------------------------

def _make_hit(
    query_id: str,
    section_id: str,
    parent_section: str,
    query_text: str,
    extracted_content: str,
    confidence: float = 0.90,
) -> QueryExtractionHit:
    """Shorthand factory for QueryExtractionHit."""
    return QueryExtractionHit(
        query_id=query_id,
        section_id=section_id,
        parent_section=parent_section,
        query_text=query_text,
        extracted_content=extracted_content,
        confidence=confidence,
    )


@pytest.fixture()
def assembled_protocol() -> AssembledProtocol:
    """Minimal AssembledProtocol matching IchSection fixture content."""
    from ptcv.ich_parser.template_assembler import CoverageReport

    b1_hits = [
        _make_hit(
            "B.1.1.q1", "B.1.1", "B.1", "What is the protocol title?",
            "A Phase III, Double-Blind, Placebo-Controlled, Parallel-Group "
            "Study of Drug X in Patients with Hypertension",
        ),
        _make_hit(
            "B.1.1.q2", "B.1.1", "B.1", "What is the protocol number?",
            "PROTO-2024-001",
        ),
        _make_hit(
            "B.1.2.q1", "B.1.2", "B.1",
            "Who is the sponsor?",
            "Test Pharma Inc",
        ),
    ]
    b4_hits = [
        _make_hit(
            "B.4.1.q1", "B.4.1", "B.4", "What is the study design?",
            "This is a randomized, parallel-group, double-blind study.\n"
            "Screening period followed by treatment period then follow-up.",
        ),
        _make_hit(
            "B.4.2.q1", "B.4.2", "B.4", "What is the blinding?",
            "Double-blind",
        ),
    ]
    b5_hits = [
        _make_hit(
            "B.5.1.q1", "B.5.1", "B.5",
            "What are the inclusion criteria?",
            "1. Age 18-65 years\n"
            "2. Diagnosis of hypertension\n"
            "3. Written informed consent",
        ),
        _make_hit(
            "B.5.2.q1", "B.5.2", "B.5",
            "What are the exclusion criteria?",
            "1. Prior cardiovascular event\n"
            "2. Renal impairment (eGFR < 30 mL/min)",
        ),
    ]
    b3_hits = [
        _make_hit(
            "B.3.q1", "B.3", "B.3",
            "What are the primary objectives?",
            "The primary objective is to evaluate the efficacy of "
            "Drug X 10 mg in reducing systolic blood pressure compared "
            "to placebo in patients with essential hypertension.",
        ),
    ]
    b6_hits = [
        _make_hit(
            "B.6.q1", "B.6", "B.6",
            "What are the discontinuation criteria?",
            "The study will be discontinued if more than 10% of "
            "subjects experience Grade 4 adverse events.",
        ),
    ]
    b7_hits = [
        _make_hit(
            "B.7.1.q1", "B.7.1", "B.7",
            "What treatments are administered?",
            "Arm A: Drug X 10 mg tablet once daily\n"
            "Arm B: Placebo once daily",
        ),
    ]
    b10_hits = [
        _make_hit(
            "B.10.1.q1", "B.10.1", "B.10",
            "What is the planned sample size?",
            "A total of 200 subjects will be randomized in a 1:1 ratio.",
        ),
    ]

    sections = [
        AssembledSection(
            section_code="B.1",
            section_name="General Information",
            populated=True,
            hits=b1_hits,
            average_confidence=0.90,
            is_gap=False,
            has_low_confidence=False,
            required_query_count=3,
            answered_required_count=3,
        ),
        AssembledSection(
            section_code="B.4",
            section_name="Trial Design",
            populated=True,
            hits=b4_hits,
            average_confidence=0.90,
            is_gap=False,
            has_low_confidence=False,
            required_query_count=2,
            answered_required_count=2,
        ),
        AssembledSection(
            section_code="B.5",
            section_name="Selection of Subjects",
            populated=True,
            hits=b5_hits,
            average_confidence=0.90,
            is_gap=False,
            has_low_confidence=False,
            required_query_count=2,
            answered_required_count=2,
        ),
        AssembledSection(
            section_code="B.7",
            section_name="Treatment of Participants",
            populated=True,
            hits=b7_hits,
            average_confidence=0.90,
            is_gap=False,
            has_low_confidence=False,
            required_query_count=1,
            answered_required_count=1,
        ),
        AssembledSection(
            section_code="B.3",
            section_name="Trial Objectives and Purpose",
            populated=True,
            hits=b3_hits,
            average_confidence=0.90,
            is_gap=False,
            has_low_confidence=False,
            required_query_count=1,
            answered_required_count=1,
        ),
        AssembledSection(
            section_code="B.6",
            section_name="Discontinuation and Withdrawal",
            populated=True,
            hits=b6_hits,
            average_confidence=0.88,
            is_gap=False,
            has_low_confidence=False,
            required_query_count=1,
            answered_required_count=1,
        ),
        AssembledSection(
            section_code="B.10",
            section_name="Statistics",
            populated=True,
            hits=b10_hits,
            average_confidence=0.92,
            is_gap=False,
            has_low_confidence=False,
            required_query_count=1,
            answered_required_count=1,
        ),
    ]

    coverage = CoverageReport(
        total_sections=16,
        populated_count=7,
        gap_count=9,
        average_confidence=0.90,
        high_confidence_count=7,
        medium_confidence_count=0,
        low_confidence_count=0,
        total_queries=11,
        answered_queries=11,
        required_queries=11,
        answered_required=11,
        gap_sections=[],
        low_confidence_sections=[],
    )

    return AssembledProtocol(
        sections=sections,
        coverage=coverage,
        source_traceability={},
    )


@pytest.fixture()
def all_hits(assembled_protocol: AssembledProtocol) -> list[QueryExtractionHit]:
    """Flat list of all hits from assembled_protocol."""
    hits: list[QueryExtractionHit] = []
    for section in assembled_protocol.sections:
        hits.extend(section.hits)
    return hits


# ---------------------------------------------------------------------------
# Storage fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def soa_table() -> RawSoaTable:
    """Realistic SoA table with VS, LB, EG, PE, and unmapped assessments."""
    return RawSoaTable(
        visit_headers=["Screening", "Baseline", "Week 4", "End of Study"],
        day_headers=["-14", "1", "22", "85"],
        activities=[
            ("Vital Signs", [True, True, True, True]),
            ("Blood Pressure", [True, True, True, True]),
            ("Heart Rate", [True, True, True, True]),
            ("Hematology", [True, True, False, True]),
            ("Chemistry", [True, True, False, True]),
            ("12-lead ECG", [True, False, True, True]),
            ("Physical Examination", [True, True, False, True]),
            ("Genomic Sequencing", [False, True, False, False]),
        ],
        section_code="B.7",
    )


@pytest.fixture()
def tmp_gateway(tmp_path) -> FilesystemAdapter:
    adapter = FilesystemAdapter(root=tmp_path / "data")
    adapter.initialise()
    return adapter
