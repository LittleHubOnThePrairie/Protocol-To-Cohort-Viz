"""Shared fixtures for PTCV protocol search tests.

Qualification phase: IQ/OQ (unit and integration)
Regulatory requirement: PTCV-18 — EU-CTR and ClinicalTrials.gov
  protocol ingestion (21 CFR 11.10(e), ALCOA+)
Risk tier: MEDIUM
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure src/ is on the path for non-installed package
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


@pytest.fixture()
def tmp_filestore(tmp_path: Path):
    """FilestoreManager backed by a temporary directory."""
    from ptcv.protocol_search.filestore import FilestoreManager

    fs = FilestoreManager(root=tmp_path / "protocols")
    fs.ensure_directories()
    return fs


@pytest.fixture()
def tmp_audit(tmp_path: Path):
    """AuditLogger writing to a temporary file."""
    from ptcv.compliance.audit import AuditLogger

    return AuditLogger(
        module="test",
        log_path=tmp_path / "audit.jsonl",
    )


@pytest.fixture()
def ctis_service(tmp_filestore, tmp_audit):
    """CTISService with isolated filestore and audit logger."""
    from ptcv.protocol_search.eu_ctr_service import CTISService

    return CTISService(filestore=tmp_filestore, audit_logger=tmp_audit)


@pytest.fixture()
def ct_service(tmp_filestore, tmp_audit):
    """ClinicalTrialsService with isolated filestore and audit logger."""
    from ptcv.protocol_search.clinicaltrials_service import ClinicalTrialsService

    return ClinicalTrialsService(
        filestore=tmp_filestore, audit_logger=tmp_audit
    )
