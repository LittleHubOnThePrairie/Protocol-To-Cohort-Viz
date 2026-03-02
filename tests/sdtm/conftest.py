"""Shared fixtures for SDTM tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest

from ptcv.ich_parser.models import IchSection
from ptcv.soa_extractor.models import UsdmTimepoint
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
                "This is a parallel-group, double-blind study.\n"
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
def all_sections(b1_section, b4_section, b5_section) -> list[IchSection]:
    return [b1_section, b4_section, b5_section]


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

@pytest.fixture()
def tmp_gateway(tmp_path) -> FilesystemAdapter:
    adapter = FilesystemAdapter(root=tmp_path / "data")
    adapter.initialise()
    return adapter
