"""Shared fixtures for SDTM validation tests (PTCV-23)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import pandas as pd
import pytest

from ptcv.storage.filesystem_adapter import FilesystemAdapter
from ptcv.sdtm.validation.models import P21Issue, DefineXmlIssue


# ---------------------------------------------------------------------------
# Storage fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_gateway(tmp_path) -> FilesystemAdapter:
    adapter = FilesystemAdapter(root=tmp_path / "data")
    adapter.initialise()
    return adapter


# ---------------------------------------------------------------------------
# Minimal valid domain DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture()
def ts_df() -> pd.DataFrame:
    """Minimal valid TS dataset with 3 rows."""
    return pd.DataFrame([
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 1.0,
            "TSPARMCD": "TITLE",
            "TSPARM": "Trial Title",
            "TSVAL": "Phase III Study of Drug X",
            "TSVALCD": "",
            "TSVALNF": "",
            "TSVALUNIT": "",
        },
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 2.0,
            "TSPARMCD": "PHASE",
            "TSPARM": "Trial Phase",
            "TSVAL": "PHASE III",
            "TSVALCD": "C49688",
            "TSVALNF": "",
            "TSVALUNIT": "",
        },
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 3.0,
            "TSPARMCD": "STYPE",
            "TSPARM": "Study Type",
            "TSVAL": "PARALLEL",
            "TSVALCD": "C82587",
            "TSVALNF": "",
            "TSVALUNIT": "",
        },
    ])


@pytest.fixture()
def ta_df() -> pd.DataFrame:
    """Minimal valid TA dataset."""
    return pd.DataFrame([
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TA",
            "ARMCD": "DRUG",
            "ARM": "Drug X Arm",
            "TAETORD": 1.0,
            "ETCD": "SCRN",
            "ELEMENT": "Screening",
            "TABRANCH": "",
            "TATRANS": "",
            "EPOCH": "Screening",
        },
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TA",
            "ARMCD": "DRUG",
            "ARM": "Drug X Arm",
            "TAETORD": 2.0,
            "ETCD": "TRT",
            "ELEMENT": "Treatment",
            "TABRANCH": "",
            "TATRANS": "",
            "EPOCH": "Treatment",
        },
    ])


@pytest.fixture()
def te_df() -> pd.DataFrame:
    """Minimal valid TE dataset."""
    return pd.DataFrame([
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TE",
            "ETCD": "SCRN",
            "ELEMENT": "Screening",
            "TESTRL": "After informed consent",
            "TEENRL": "Day 0",
            "TEDUR": "P14D",
        },
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TE",
            "ETCD": "TRT",
            "ELEMENT": "Treatment",
            "TESTRL": "Day 1",
            "TEENRL": "Day 84",
            "TEDUR": "P84D",
        },
    ])


@pytest.fixture()
def tv_df() -> pd.DataFrame:
    """Minimal valid TV dataset."""
    return pd.DataFrame([
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TV",
            "VISITNUM": 1.0,
            "VISIT": "Screening",
            "VISITDY": -14.0,
            "TVSTRL": -3.0,
            "TVENRL": 0.0,
            "TVTIMY": "",
            "TVTIM": "",
            "TVENDY": "",
        },
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TV",
            "VISITNUM": 2.0,
            "VISIT": "Baseline",
            "VISITDY": 1.0,
            "TVSTRL": -1.0,
            "TVENRL": 1.0,
            "TVTIMY": "",
            "TVTIM": "",
            "TVENDY": "",
        },
    ])


@pytest.fixture()
def ti_df() -> pd.DataFrame:
    """Minimal valid TI dataset."""
    return pd.DataFrame([
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TI",
            "IETESTCD": "IE001I",
            "IETEST": "Age 18-65 years",
            "IECAT": "INCLUSION",
            "IESCAT": "",
            "TIRL": "",
            "TIVERS": "1.0",
        },
        {
            "STUDYID": "NCT001",
            "DOMAIN": "TI",
            "IETESTCD": "IE001E",
            "IETEST": "Prior cardiovascular event",
            "IECAT": "EXCLUSION",
            "IESCAT": "",
            "TIRL": "",
            "TIVERS": "1.0",
        },
    ])


@pytest.fixture()
def all_domains(ts_df, ta_df, te_df, tv_df, ti_df) -> dict:
    return {"TS": ts_df, "TA": ta_df, "TE": te_df, "TV": tv_df, "TI": ti_df}


# ---------------------------------------------------------------------------
# Sample P21 issues
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_p21_issues() -> list[P21Issue]:
    return [
        P21Issue(
            rule_id="P21-TS-011",
            severity="Error",
            domain="TS",
            variable="TSPARMCD",
            description="TSPARMCD contains non-alphanumeric characters.",
            remediation_guidance="SDTMIG v3.3 §7.4.1 — TSPARMCD must be alphanumeric",
        ),
        P21Issue(
            rule_id="P21-TS-012",
            severity="Warning",
            domain="TS",
            variable="TSVAL",
            description="TSVAL is blank for 1 row(s) without TSVALNF.",
            remediation_guidance="SDTMIG v3.3 §7.4.1 — TSVAL must not be blank",
        ),
    ]
