"""Shared fixtures for PTCV-24 pipeline integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


# ---------------------------------------------------------------------------
# Synthetic 2-page protocol text (used as the pipeline's input "PDF")
# ---------------------------------------------------------------------------
# This text is designed so that:
# - IchParser can extract B.4 and B.1 sections
# - SoaTableParser can parse the SoA table in B.4
# - SdtmService can generate TS, TA, TE, TV, TI domains
SYNTHETIC_PROTOCOL_TEXT = """\
CLINICAL STUDY PROTOCOL
NCT00112827

B.1 Background and Rationale
A phase 2 oncology trial evaluating the efficacy and safety of a novel
kinase inhibitor in patients with advanced solid tumours. The study is
randomised, double-blind, placebo-controlled, and multicentre.

B.4 Trial Design and Schedule of Activities

Schedule of Activities

| Assessment         | Screening | Baseline | Week 2 | Week 4 | Unscheduled | Early Termination |
|--------------------|-----------|----------|--------|--------|-------------|-------------------|
| Day                | -14 to -1 | 1        | 15 ± 3 | 29 ± 3 |             |                   |
| Informed Consent   | X         |          |        |        |             |                   |
| Physical Exam      | X         | X        | X      | X      | X           | X                 |
| ECG                | X         |          | X      |        |             | X                 |
| Laboratory Tests   | X         | X        | X      | X      |             | X                 |
| Vital Signs        | X         | X        | X      | X      | X           | X                 |
| ECOG Performance   | X         | X        |        |        |             |                   |
| Study Drug Admin   |           | X        | X      | X      |             |                   |

B.5 Statistical Considerations
The primary endpoint is progression-free survival at 16 weeks.
Sample size: 120 patients per arm (alpha 0.05, power 80%).
"""

# The synthetic protocol is plain text, not PDF bytes. We wrap it in a
# minimal structure that PdfExtractor will handle gracefully. However,
# since we don't have a real PDF for tests, we use a CTR-XML-like encoding
# or rely on the ExtractionService's fallback path.
#
# To avoid a dependency on pdfplumber for the integration tests, we
# bypass ExtractionService and inject the text directly via a thin
# helper fixture (fake_protocol_bytes) that the orchestrator's test
# stub uses.

SYNTHETIC_CTR_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<ClinicalStudy xmlns="urn:hl7-org:greenlightGreen">
  <StudyProtocol>
    <ProtocolID>NCT00112827</ProtocolID>
    <BriefTitle>Phase 2 Oncology Kinase Inhibitor Trial</BriefTitle>
    <Section code="B.1">
      <Title>Background and Rationale</Title>
      <Paragraph>A phase 2 oncology trial evaluating a novel kinase inhibitor.</Paragraph>
    </Section>
    <Section code="B.4">
      <Title>Trial Design and Schedule of Activities</Title>
      <Paragraph>Schedule of Activities</Paragraph>
      <Paragraph>| Assessment | Screening | Baseline | Week 2 | Week 4 | Unscheduled | Early Termination |</Paragraph>
      <Paragraph>|------------|-----------|----------|--------|--------|-------------|-------------------|</Paragraph>
      <Paragraph>| ECG        | X         |          | X      |        |             | X                 |</Paragraph>
      <Paragraph>| Labs       | X         | X        | X      | X      |             | X                 |</Paragraph>
      <Paragraph>| Vital Signs| X         | X        | X      | X      | X           | X                 |</Paragraph>
    </Section>
  </StudyProtocol>
</ClinicalStudy>
"""


@pytest.fixture
def tmp_gateway(tmp_path: Path):
    """FilesystemAdapter pointed at a fresh temporary directory."""
    from ptcv.storage import FilesystemAdapter

    gw = FilesystemAdapter(root=tmp_path)
    gw.initialise()
    return gw


@pytest.fixture
def tmp_review_queue(tmp_path: Path):
    """ReviewQueue backed by a temporary SQLite file."""
    from ptcv.ich_parser.review_queue import ReviewQueue

    rq = ReviewQueue(db_path=tmp_path / "review_queue.db")
    rq.initialise()
    return rq


@pytest.fixture
def tmp_ct_review_queue(tmp_path: Path):
    """CtReviewQueue backed by a temporary SQLite file."""
    from ptcv.sdtm.review_queue import CtReviewQueue

    rq = CtReviewQueue(db_path=tmp_path / "ct_review_queue.db")
    rq.initialise()
    return rq


@pytest.fixture
def orchestrator(tmp_gateway, tmp_review_queue, tmp_ct_review_queue):
    """PipelineOrchestrator backed by filesystem fixtures."""
    from ptcv.pipeline import PipelineOrchestrator

    return PipelineOrchestrator(
        gateway=tmp_gateway,
        review_queue=tmp_review_queue,
        ct_review_queue=tmp_ct_review_queue,
    )


@pytest.fixture
def ctr_xml_bytes():
    """Synthetic CTR-XML protocol as bytes for pipeline input."""
    return SYNTHETIC_CTR_XML.encode("utf-8")


@pytest.fixture
def registry_id():
    return "NCT00112827"


@pytest.fixture
def amendment_number():
    return "00"
