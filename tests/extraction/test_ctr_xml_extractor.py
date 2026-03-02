"""Tests for ptcv.extraction.ctr_xml_extractor.

PTCV-19 Scenario 4: CTR-XML parsed natively, no PDF fallback.

Qualification phase: IQ/OQ
Risk tier: MEDIUM
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.extraction.ctr_xml_extractor import CtrXmlExtractor


_RUN = "run-003"
_REG = "EU-CTR-2023-001"
_SHA = "c" * 64


class TestCtrXmlExtractor:
    def setup_method(self):
        self.extractor = CtrXmlExtractor()

    def test_extract_returns_text_blocks_and_zero_page_count(
        self, ctr_xml_bytes
    ):
        blocks, page_count = self.extractor.extract(
            ctr_xml_bytes, _RUN, _REG, _SHA
        )
        assert page_count == 0
        assert len(blocks) > 0

    def test_study_name_extracted_as_heading(self, ctr_xml_bytes):
        blocks, _ = self.extractor.extract(ctr_xml_bytes, _RUN, _REG, _SHA)
        headings = [b for b in blocks if b.block_type == "heading"]
        assert any("Test Protocol Study" in h.text for h in headings)

    def test_study_description_extracted_as_paragraph(self, ctr_xml_bytes):
        blocks, _ = self.extractor.extract(ctr_xml_bytes, _RUN, _REG, _SHA)
        paragraphs = [b for b in blocks if b.block_type == "paragraph"]
        assert any("randomised" in p.text for p in paragraphs)

    def test_visit_schedule_extracted(self, ctr_xml_bytes):
        blocks, _ = self.extractor.extract(ctr_xml_bytes, _RUN, _REG, _SHA)
        visit_blocks = [b for b in blocks if "Visit:" in b.text]
        assert len(visit_blocks) >= 2  # "Screening Visit" and "Week 4 Visit"

    def test_arms_extracted(self, ctr_xml_bytes):
        blocks, _ = self.extractor.extract(ctr_xml_bytes, _RUN, _REG, _SHA)
        arm_blocks = [b for b in blocks if "Arm:" in b.text]
        assert len(arm_blocks) >= 2  # Treatment A and Placebo

    def test_block_fields_set_correctly(self, ctr_xml_bytes):
        blocks, _ = self.extractor.extract(ctr_xml_bytes, _RUN, _REG, _SHA)
        for b in blocks:
            assert b.run_id == _RUN
            assert b.source_registry_id == _REG
            assert b.source_sha256 == _SHA
            assert b.page_number == 0  # XML has no pages
            assert b.extraction_timestamp_utc == ""  # set by service
            assert b.block_type in ("heading", "paragraph", "list_item", "other")

    def test_invalid_xml_raises_value_error(self):
        with pytest.raises(ValueError, match="CTR-XML parse error"):
            self.extractor.extract(b"not xml <<<", _RUN, _REG, _SHA)

    def test_minimal_odm_no_study_events(self):
        xml = b"""<?xml version="1.0"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3">
  <Study OID="STUDY1">
    <GlobalVariables>
      <StudyName>Minimal Study</StudyName>
    </GlobalVariables>
  </Study>
</ODM>"""
        blocks, page_count = self.extractor.extract(xml, _RUN, _REG, _SHA)
        assert page_count == 0
        assert any("Minimal Study" in b.text for b in blocks)

    def test_source_sha256_in_lineage(self, ctr_xml_bytes):
        """Lineage: every TextBlock carries the source file SHA-256."""
        blocks, _ = self.extractor.extract(
            ctr_xml_bytes, _RUN, _REG, _SHA
        )
        for b in blocks:
            assert b.source_sha256 == _SHA
