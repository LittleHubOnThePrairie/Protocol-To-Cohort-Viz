"""Shared fixtures for PTCV protocol search tests.

Qualification phase: IQ/OQ (unit and integration)
Regulatory requirement: PTCV-18 — EU-CTR and ClinicalTrials.gov
  protocol ingestion (21 CFR 11.10(e), ALCOA+)
Risk tier: MEDIUM
"""

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path for non-installed package
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


@pytest.fixture()
def tmp_gateway(tmp_path: Path):
    """FilesystemAdapter backed by a temporary directory (PTCV-29)."""
    from ptcv.storage import FilesystemAdapter

    adapter = FilesystemAdapter(root=tmp_path / "protocols")
    adapter.initialise()
    return adapter


@pytest.fixture()
def tmp_filestore(tmp_path: Path):
    """FilestoreManager shim backed by a temporary directory.

    Kept for backward compatibility with test_filestore.py tests.
    New tests should use tmp_gateway instead.
    """
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
def ctis_service(tmp_gateway, tmp_audit):
    """CTISService with isolated gateway and audit logger (PTCV-29)."""
    from ptcv.protocol_search.eu_ctr_service import CTISService

    return CTISService(gateway=tmp_gateway, audit_logger=tmp_audit)


@pytest.fixture()
def ct_service(tmp_gateway, tmp_audit):
    """ClinicalTrialsService with isolated gateway and audit logger (PTCV-29)."""
    from ptcv.protocol_search.clinicaltrials_service import ClinicalTrialsService

    return ClinicalTrialsService(gateway=tmp_gateway, audit_logger=tmp_audit)
