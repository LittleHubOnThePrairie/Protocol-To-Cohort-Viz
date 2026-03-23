"""Shared fixtures for PTCV extraction tests.

Qualification phase: IQ/OQ
Risk tier: MEDIUM
"""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src/ is on the path for non-installed package
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# ---------------------------------------------------------------------------
# Stub fitz (PyMuPDF) and pymupdf4llm for environments where they
# are not installed.  This prevents the toc_extractor → fitz import
# chain failure that blocks any import touching ich_parser.__init__.
#
# PTCV-213: Try importing the real package first.  Only install the
# stub when the real package is genuinely unavailable, preventing
# the stub from shadowing an installed fitz and leaking a broken
# _StubFitzDoc into sys.modules for the rest of the test session.
# ---------------------------------------------------------------------------
try:
    import fitz as _real_fitz  # noqa: F401 — trigger real import
except ImportError:
    _fitz = types.ModuleType("fitz")

    class _StubFitzDoc:
        """Minimal fitz.Document stand-in (zero pages)."""

        def __init__(self, **kw: object) -> None:
            self._pages: list[object] = []

        def __len__(self) -> int:
            return len(self._pages)

        def new_page(self, **kw: object) -> MagicMock:
            """Create a stub page (needed by landscape PDF tests)."""
            page = MagicMock()
            self._pages.append(page)
            return page

        def close(self) -> None:
            pass

        def save(self, *a: object, **kw: object) -> None:
            pass

        def tobytes(self) -> bytes:
            return b"%PDF-1.4 stub"

    _fitz.open = lambda **kw: _StubFitzDoc(**kw)  # type: ignore[attr-defined]
    sys.modules["fitz"] = _fitz

try:
    import pymupdf4llm as _real_pymupdf  # noqa: F401
except ImportError:
    _pymupdf = types.ModuleType("pymupdf4llm")
    _pymupdf.to_markdown = lambda doc, **kw: []  # type: ignore[attr-defined]
    sys.modules["pymupdf4llm"] = _pymupdf


# ---------------------------------------------------------------------------
# Minimal synthetic PDF factory
# ---------------------------------------------------------------------------


def make_minimal_pdf(text: str = "Clinical Trial Protocol\nIntroduction text here.") -> bytes:
    """Return a minimal valid PDF with one text page.

    Sufficient for pdfplumber to extract text; not guaranteed to contain
    lattice tables (Camelot/tabula tests use real PDFs or mocks).
    """
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET"
    stream_bytes = stream.encode()
    length = len(stream_bytes)

    resources = "<</Font <</F1 <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>>>>>"
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        + f"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        f"/Contents 4 0 R /Resources {resources}>> endobj\n".encode()
        + f"4 0 obj <</Length {length}>>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000052 00000 n \n"
        b"0000000103 00000 n \n"
        b"0000000250 00000 n \n"
        b"trailer <</Size 5 /Root 1 0 R>>\n"
        b"startxref\n400\n%%EOF\n"
    )
    return content


def make_ctr_xml(
    nct_id: str = "NCT00112827",
    study_name: str = "Test Protocol Study",
) -> bytes:
    """Return a minimal CTR-XML (CDISC ODM) document."""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     FileOID="EudraCT-{nct_id}" FileType="Snapshot" CreationDateTime="2025-01-01">
  <Study OID="{nct_id}">
    <GlobalVariables>
      <StudyName>{study_name}</StudyName>
      <StudyDescription>A phase III randomised controlled trial.</StudyDescription>
      <ProtocolName>Protocol v1.0</ProtocolName>
    </GlobalVariables>
    <MetaDataVersion OID="v1" Name="v1">
      <StudyEventDef OID="VISIT1" Name="Screening Visit" Type="Scheduled">
        <Description>Initial eligibility screening.</Description>
      </StudyEventDef>
      <StudyEventDef OID="VISIT2" Name="Week 4 Visit" Type="Scheduled"/>
      <ArmDef OID="ARM1" Name="Treatment A">
        <Description>Active drug arm.</Description>
      </ArmDef>
      <ArmDef OID="ARM2" Name="Placebo"/>
    </MetaDataVersion>
  </Study>
</ODM>"""
    return xml.encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_pdf_bytes() -> bytes:
    """Minimal valid PDF bytes for format detection and text extraction."""
    return make_minimal_pdf()


@pytest.fixture()
def ctr_xml_bytes() -> bytes:
    """Minimal valid CTR-XML bytes for format detection and XML extraction."""
    return make_ctr_xml()


@pytest.fixture()
def tmp_gateway(tmp_path: Path):
    """FilesystemAdapter backed by a temporary directory."""
    from ptcv.storage import FilesystemAdapter

    adapter = FilesystemAdapter(root=tmp_path / "protocols")
    adapter.initialise()
    return adapter


@pytest.fixture()
def sample_registry_id() -> str:
    return "NCT00112827"


@pytest.fixture()
def sample_source_sha256() -> str:
    """Fake SHA-256 representing a source protocol file from PTCV-18."""
    return "a" * 64


@pytest.fixture()
def real_pdf_bytes() -> bytes:
    """Read one real ClinicalTrials.gov PDF from the local datastore."""
    pdf_path = Path(
        "C:/Dev/PTCV/data/protocols/clinicaltrials/NCT00112827_1.0.pdf"
    )
    if not pdf_path.exists():
        pytest.skip("Real protocol PDF not available at expected path")
    return pdf_path.read_bytes()
