"""Shared fixtures for SoA extractor tests."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# ---------------------------------------------------------------------------
# Stub fitz (PyMuPDF) and pymupdf4llm to prevent toc_extractor import chain
# failure when these packages are not installed.
# ---------------------------------------------------------------------------
if "fitz" not in sys.modules:
    _fitz = types.ModuleType("fitz")

    class _StubFitzDoc:
        """Minimal fitz.Document stand-in (zero pages)."""

        def __init__(self, **kw: object) -> None:
            pass

        def __len__(self) -> int:
            return 0

        def close(self) -> None:
            pass

    _fitz.open = lambda **kw: _StubFitzDoc(**kw)  # type: ignore[attr-defined]
    sys.modules["fitz"] = _fitz

if "pymupdf4llm" not in sys.modules:
    _pymupdf = types.ModuleType("pymupdf4llm")
    _pymupdf.to_markdown = lambda doc, **kw: []  # type: ignore[attr-defined]
    sys.modules["pymupdf4llm"] = _pymupdf


# ---------------------------------------------------------------------------
# Minimal SoA table text (markdown pipe format)
# ---------------------------------------------------------------------------

SAMPLE_SOA_TEXT = """\
Schedule of Activities

| Assessment | Screening | Baseline | Week 2 | Week 4 | Unscheduled | Early Termination |
|------------|-----------|----------|--------|--------|-------------|-------------------|
| Day        | -14 to -1 | 1        | 15 ± 3 | 29 ± 3 |             |                   |
| Informed Consent | X  |          |        |        |             |                   |
| Physical Exam    | X  | X        | X      | X      | X           | X                 |
| ECG              | X  |          | X      |        |             | X                 |
| Laboratory Tests | X  | X        | X      | X      |             | X                 |
| Vital Signs      | X  | X        | X      | X      | X           | X                 |
| ECOG Performance | X  | X        |        |        |             |                   |
"""

SAMPLE_SOA_SECTION_JSON = json.dumps({"text": SAMPLE_SOA_TEXT})

# Content with "Visit 0" and "Pre-treatment" to exercise synonym resolution
SYNONYM_SOA_TEXT = """\
Schedule of Activities

| Assessment | Visit 0 | Pre-treatment | Day 1 | Week 2 |
|------------|---------|---------------|-------|--------|
| ECG        | X       | X             | X     | X      |
| Labs       |         | X             | X     |        |
"""

SYNONYM_SOA_SECTION_JSON = json.dumps({"text": SYNONYM_SOA_TEXT})


@pytest.fixture
def sample_ich_sections():
    """IchSection list containing a B.4 section with a markdown SoA table."""
    from ptcv.ich_parser.models import IchSection

    return [
        IchSection(
            run_id="ich-run-001",
            source_run_id="extract-run-001",
            source_sha256="a" * 64,
            registry_id="NCT00112827",
            section_code="B.4",
            section_name="Trial Design",
            content_json=SAMPLE_SOA_SECTION_JSON,
            confidence_score=0.90,
            review_required=False,
            legacy_format=False,
            extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
        ),
    ]


@pytest.fixture
def synonym_ich_sections():
    """IchSection list with Visit 0 / Pre-treatment synonyms."""
    from ptcv.ich_parser.models import IchSection

    return [
        IchSection(
            run_id="ich-run-002",
            source_run_id="extract-run-001",
            source_sha256="b" * 64,
            registry_id="NCT00112828",
            section_code="B.4",
            section_name="Trial Design",
            content_json=SYNONYM_SOA_SECTION_JSON,
            confidence_score=0.88,
            review_required=False,
            legacy_format=False,
            extraction_timestamp_utc="2024-01-15T10:00:00+00:00",
        ),
    ]


@pytest.fixture
def tmp_gateway(tmp_path: Path):
    """FilesystemAdapter pointed at a temporary directory."""
    from ptcv.storage import FilesystemAdapter

    gw = FilesystemAdapter(root=tmp_path)
    gw.initialise()
    return gw


@pytest.fixture
def tmp_review_queue(tmp_path: Path):
    """ReviewQueue backed by a temporary SQLite file."""
    from ptcv.ich_parser import ReviewQueue

    rq = ReviewQueue(db_path=tmp_path / "review_queue.db")
    rq.initialise()
    return rq
