"""Shared fixtures for PTCV-24/60 pipeline integration tests."""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


# ---------------------------------------------------------------------------
# Synthetic 2-page protocol text (used as the pipeline's input "PDF")
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Mock LlmRetemplater for integration tests (no Anthropic API needed)
# ---------------------------------------------------------------------------


class MockLlmRetemplater:
    """Fake retemplater that produces a deterministic result without LLM calls.

    Writes a real sections.parquet via the gateway so downstream stages
    (coverage review, SDTM) can read it.
    """

    def __init__(self, gateway, review_queue=None):
        self._gateway = gateway
        self._review_queue = review_queue

    def retemplate(
        self,
        text_blocks: list[dict],
        registry_id: str,
        source_run_id: str = "",
        source_sha256: str = "",
        soa_summary: Optional[dict] = None,
        who: str = "mock-retemplater",
    ):
        from ptcv.ich_parser.llm_retemplater import RetemplatingResult
        from ptcv.ich_parser.models import IchSection
        from ptcv.ich_parser.parquet_writer import sections_to_parquet
        from ptcv.ich_parser.parser import _compute_format_verdict

        if not text_blocks:
            raise ValueError("text_blocks must not be empty")

        run_id = str(uuid.uuid4())
        timestamp = "2024-01-15T10:00:00+00:00"

        # Build one B.1 section from the text
        full_text = " ".join(b.get("text", "") for b in text_blocks)
        sections = [
            IchSection(
                run_id=run_id,
                source_run_id=source_run_id,
                source_sha256=source_sha256,
                registry_id=registry_id,
                section_code="B.1",
                section_name="General Information",
                content_json=json.dumps({
                    "text_excerpt": full_text[:500],
                    "key_concepts": ["mock"],
                }),
                confidence_score=0.85,
                review_required=False,
                legacy_format=True,
                extraction_timestamp_utc=timestamp,
            ),
        ]

        # Write Parquet
        parquet_bytes = sections_to_parquet(sections)
        artifact_key = f"ich-json/{run_id}/sections.parquet"
        self._gateway.put_artifact(
            key=artifact_key,
            data=parquet_bytes,
            content_type="application/vnd.apache.parquet",
            run_id=run_id,
            source_hash=source_sha256,
            user=who,
            immutable=False,
            stage="retemplating",
            registry_id=registry_id,
        )

        artifact_sha256 = hashlib.sha256(parquet_bytes).hexdigest()
        verdict, confidence, missing = _compute_format_verdict(sections)

        return RetemplatingResult(
            run_id=run_id,
            registry_id=registry_id,
            artifact_key=artifact_key,
            artifact_sha256=artifact_sha256,
            section_count=len(sections),
            review_count=0,
            source_sha256=source_sha256,
            format_verdict=verdict,
            format_confidence=confidence,
            missing_required_sections=missing,
            input_tokens=0,
            output_tokens=0,
            chunk_count=1,
        )


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
    """PipelineOrchestrator backed by filesystem fixtures with mock retemplater."""
    from ptcv.pipeline import PipelineOrchestrator

    mock_retemplater = MockLlmRetemplater(
        gateway=tmp_gateway,
        review_queue=tmp_review_queue,
    )

    return PipelineOrchestrator(
        gateway=tmp_gateway,
        review_queue=tmp_review_queue,
        ct_review_queue=tmp_ct_review_queue,
        retemplater=mock_retemplater,
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
