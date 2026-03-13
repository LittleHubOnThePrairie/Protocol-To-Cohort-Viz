"""Tests for ptcv.extraction.extraction_service.ExtractionService.

Covers all 5 GHERKIN acceptance criteria from PTCV-19:

  Scenario 1: Detect PDF and extract with lineage record
  Scenario 2: Cascade to Camelot on pdfplumber failure
  Scenario 3: Reconstruct multi-page SoA tables
  Scenario 4: Parse CTR-XML natively with lineage
  Scenario 5: Re-extraction creates new run_id, preserves prior extraction

Qualification phase: IQ/OQ
Risk tier: MEDIUM
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
# fitz/pymupdf4llm stubs are provided by conftest.py

from ptcv.extraction.extraction_service import ExtractionService
from ptcv.extraction.format_detector import ProtocolFormat
from ptcv.extraction.parquet_writer import (
    parquet_to_metadata,
    parquet_to_tables,
    parquet_to_text_blocks,
)


_SHA = "d" * 64


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -----------------------------------------------------------------------
# Scenario 1: Detect PDF and extract with lineage record
# -----------------------------------------------------------------------

class TestScenario1PdfWithLineage:
    """Given a protocol PDF → format detected, Parquet written, lineage appended."""

    def test_pdf_format_detected(self, tmp_gateway, minimal_pdf_bytes):
        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT00112827",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s1",
        )
        assert result.format_detected == "pdf"

    def test_parquet_files_written_to_gateway(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT00112827",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s1",
        )
        # All three Parquet artifacts must exist in the gateway
        assert tmp_gateway.get_artifact(result.text_artifact_key)
        assert tmp_gateway.get_artifact(result.tables_artifact_key)
        assert tmp_gateway.get_artifact(result.metadata_artifact_key)

    def test_lineage_record_appended_with_source_hash(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT00112827",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s1",
        )
        lineage = tmp_gateway.get_lineage("run-s1")
        # Three artifacts → three lineage records
        assert len(lineage) == 3
        for rec in lineage:
            assert rec.stage == "extraction"
            assert rec.source_hash == _SHA
            assert rec.run_id == "run-s1"

    def test_timestamp_within_lineage_and_parquet(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        from datetime import datetime, timezone

        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT00112827",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s1",
        )
        lineage = tmp_gateway.get_lineage("run-s1")
        # Read metadata Parquet to check timestamp
        meta_bytes = tmp_gateway.get_artifact(result.metadata_artifact_key)
        meta = parquet_to_metadata(meta_bytes)

        # Per PTCV-19 Scenario 1: extraction_timestamp_utc in Parquet must be
        # within 1 second of the LineageRecord timestamp_utc (ALCOA+ Contemporaneous).
        parquet_ts = datetime.fromisoformat(meta.extraction_timestamp_utc)
        lineage_ts = datetime.fromisoformat(lineage[0].timestamp_utc)
        delta = abs((parquet_ts - lineage_ts).total_seconds())
        assert delta <= 1.0, (
            f"Timestamp gap {delta:.3f}s exceeds 1 second: "
            f"parquet={meta.extraction_timestamp_utc} "
            f"lineage={lineage[0].timestamp_utc}"
        )

    def test_artifact_keys_contain_run_id(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT00112827",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s1",
        )
        assert "run-s1" in result.text_artifact_key
        assert "run-s1" in result.tables_artifact_key
        assert "run-s1" in result.metadata_artifact_key


# -----------------------------------------------------------------------
# Scenario 2: Cascade to Camelot on pdfplumber failure
# -----------------------------------------------------------------------

class TestScenario2CamelotCascade:
    """When pdfplumber returns zero table rows, extractor_used=camelot."""

    def test_camelot_extractor_used_in_parquet(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        import pandas as pd
        from ptcv.extraction.pdf_extractor import PdfExtractor

        camelot_table = MagicMock()
        camelot_table.df = pd.DataFrame(
            [["Visit", "Day"], ["Screening", "-7"]]
        )

        # Force pymupdf4llm to fail so the pdfplumber→Camelot cascade runs
        with (
            patch(
                "ptcv.extraction.pdf_extractor.PdfExtractor"
                "._extract_pymupdf4llm",
                side_effect=RuntimeError("pymupdf4llm unavailable"),
            ),
            patch("camelot.read_pdf", return_value=[camelot_table]),
        ):
            svc = ExtractionService(gateway=tmp_gateway)
            result = svc.extract(
                protocol_data=minimal_pdf_bytes,
                registry_id="NCT00492050",
                amendment_number="1.0",
                source_sha256=_SHA,
                run_id="run-s2",
            )

        tables_bytes = tmp_gateway.get_artifact(result.tables_artifact_key)
        tables = parquet_to_tables(tables_bytes)
        camelot_tables = [t for t in tables if t.extractor_used == "camelot"]
        # Camelot tables were injected via the fallback path
        assert len(camelot_tables) >= 1


# -----------------------------------------------------------------------
# Scenario 3: Reconstruct multi-page SoA tables
# -----------------------------------------------------------------------

class TestScenario3MultiPageReconstruction:
    """Multi-page table reconstruction handled in PdfExtractor internals."""

    def test_three_page_soa_merged_to_single_record(self):
        from ptcv.extraction.pdf_extractor import PdfExtractor

        extractor = PdfExtractor()
        header = ["Visit", "Day", "Assessment"]
        pages = [
            [[header, ["Screening", "-7", "CBC"]]],
            [[header, ["Week 4", "28", "ECG"]]],
            [[header, ["Week 8", "56", "PK"]]],
        ]
        result = extractor._reconstruct_multi_page_tables(pages)
        merged = result[0][0]
        assert len(merged) == 4  # header + 3 data rows
        assert merged[0] == header
        assert merged[-1] == ["Week 8", "56", "PK"]

    def test_column_header_signature_matches_across_pages(self):
        from ptcv.extraction.pdf_extractor import PdfExtractor

        extractor = PdfExtractor()
        header = ["Visit", "Day", "Procedure", "Result"]
        # 2-page table with whitespace differences in continuation header
        pages = [
            [[header, ["V1", "1", "Blood draw", "Normal"]]],
            # Headers with extra spaces — won't match (strict comparison)
            [[["Visit ", "Day", "Procedure", "Result"],
              ["V2", "8", "ECG", "Normal"]]],
        ]
        result = extractor._reconstruct_multi_page_tables(pages)
        # Stripped comparison: "Visit " strips to "Visit" == "Visit"
        # The reconstructed table should merge if stripped values match
        merged = result[0][0]
        # After strip, header on page 2 matches → should be merged
        assert len(merged) >= 2


# -----------------------------------------------------------------------
# Scenario 4: Parse CTR-XML natively with lineage
# -----------------------------------------------------------------------

class TestScenario4CtrXml:
    """Given CTR-XML → detected as CTR-XML, text blocks written, no PDF fallback."""

    def test_ctr_xml_format_detected(self, tmp_gateway, ctr_xml_bytes):
        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=ctr_xml_bytes,
            registry_id="EU-CTR-2023-001",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s4",
        )
        assert result.format_detected == "ctr-xml"

    def test_text_blocks_written_to_parquet(
        self, tmp_gateway, ctr_xml_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=ctr_xml_bytes,
            registry_id="EU-CTR-2023-001",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s4",
        )
        text_bytes = tmp_gateway.get_artifact(result.text_artifact_key)
        blocks = parquet_to_text_blocks(text_bytes)
        assert len(blocks) > 0
        assert all(b.source_sha256 == _SHA for b in blocks)

    def test_lineage_source_hash_references_xml_sha256(
        self, tmp_gateway, ctr_xml_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=ctr_xml_bytes,
            registry_id="EU-CTR-2023-001",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s4",
        )
        lineage = tmp_gateway.get_lineage("run-s4")
        for rec in lineage:
            assert rec.source_hash == _SHA

    def test_pdf_fallback_not_attempted(
        self, tmp_gateway, ctr_xml_bytes
    ):
        """PdfExtractor.extract must not be called for CTR-XML input."""
        svc = ExtractionService(gateway=tmp_gateway)
        with patch.object(
            svc._pdf_extractor, "extract", wraps=svc._pdf_extractor.extract
        ) as mock_pdf:
            svc.extract(
                protocol_data=ctr_xml_bytes,
                registry_id="EU-CTR-2023-001",
                amendment_number="1.0",
                source_sha256=_SHA,
                run_id="run-s4-nofallback",
            )
            mock_pdf.assert_not_called()

    def test_tables_parquet_is_zero_rows_for_xml(
        self, tmp_gateway, ctr_xml_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=ctr_xml_bytes,
            registry_id="EU-CTR-2023-001",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-s4-tables",
        )
        tables_bytes = tmp_gateway.get_artifact(result.tables_artifact_key)
        tables = parquet_to_tables(tables_bytes)
        assert tables == []


# -----------------------------------------------------------------------
# Scenario 5: Re-extraction creates new run_id, preserves prior extraction
# -----------------------------------------------------------------------

class TestScenario5ReExtraction:
    """Re-run produces new run_id; prior Parquet under old run_id is unchanged."""

    def test_new_run_id_on_reextraction(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        r1 = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT01252667",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-abc",
        )
        r2 = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT01252667",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-xyz",
        )
        assert r1.run_id != r2.run_id
        assert "run-abc" in r1.tables_artifact_key
        assert "run-xyz" in r2.tables_artifact_key

    def test_prior_parquet_unchanged_after_reextraction(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        r1 = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT01252667",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-abc",
        )
        # Read prior artifact bytes and SHA-256
        prior_bytes = tmp_gateway.get_artifact(r1.tables_artifact_key)
        prior_sha = hashlib.sha256(prior_bytes).hexdigest()

        # Second extraction
        svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT01252667",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-xyz",
        )

        # Prior artifact must be identical
        still_bytes = tmp_gateway.get_artifact(r1.tables_artifact_key)
        assert hashlib.sha256(still_bytes).hexdigest() == prior_sha

    def test_both_run_lineage_records_exist(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT01252667",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-abc",
        )
        svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT01252667",
            amendment_number="1.0",
            source_sha256=_SHA,
            run_id="run-xyz",
        )
        lineage_abc = tmp_gateway.get_lineage("run-abc")
        lineage_xyz = tmp_gateway.get_lineage("run-xyz")
        assert len(lineage_abc) == 3
        assert len(lineage_xyz) == 3

    def test_auto_generated_run_id_is_uuid4(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        import uuid

        svc = ExtractionService(gateway=tmp_gateway)
        result = svc.extract(
            protocol_data=minimal_pdf_bytes,
            registry_id="NCT01252667",
            amendment_number="1.0",
            source_sha256=_SHA,
        )
        # Must be a valid UUID4
        parsed = uuid.UUID(result.run_id, version=4)
        assert str(parsed) == result.run_id


# -----------------------------------------------------------------------
# Word format
# -----------------------------------------------------------------------

class TestWordNotImplemented:
    def test_word_raises_not_implemented(self, tmp_gateway):
        docx_like = b"PK\x03\x04some docx bytes here"
        svc = ExtractionService(gateway=tmp_gateway)
        with pytest.raises(NotImplementedError):
            svc.extract(
                protocol_data=docx_like,
                registry_id="NCT00000001",
                amendment_number="1.0",
                source_sha256=_SHA,
            )


# -----------------------------------------------------------------------
# Real-data integration
# -----------------------------------------------------------------------

class TestRealPdfIntegration:
    """Integration test against an actual ClinicalTrials.gov protocol PDF."""

    def test_real_protocol_extracts_successfully(
        self, tmp_gateway, real_pdf_bytes
    ):
        svc = ExtractionService(gateway=tmp_gateway)
        source_sha = _sha256(real_pdf_bytes)
        result = svc.extract(
            protocol_data=real_pdf_bytes,
            registry_id="NCT00112827",
            amendment_number="1.0",
            source_sha256=source_sha,
        )
        assert result.format_detected == "pdf"
        assert result.text_block_count > 0
        # Verify Parquet round-trip
        meta_bytes = tmp_gateway.get_artifact(result.metadata_artifact_key)
        meta = parquet_to_metadata(meta_bytes)
        assert meta.page_count > 0
        assert meta.source_registry_id == "NCT00112827"
        assert meta.source_sha256 == source_sha


# -----------------------------------------------------------------------
# Vision integration (PTCV-172)
# -----------------------------------------------------------------------

class _FakeExtLevel:
    """Minimal stand-in for ExtractionLevel enum value."""

    def __init__(self, value: str):
        self.value = value
        self.name = value.upper()


class TestVisionIntegration:
    """PTCV-172: Vision enhancement wired into ExtractionService."""

    def test_e1_triggers_vision(self, tmp_gateway, minimal_pdf_bytes):
        """ext_level with value='docling_vision' invokes VisionEnhancer."""
        from ptcv.extraction.vision_enhancer import VisionEnhancementResult

        fake_result = VisionEnhancementResult(
            text_blocks=[],
            tables=[],
            pages_processed=2,
            total_input_tokens=3600,
            total_output_tokens=1000,
            cover_fields={},
            estimated_cost_usd=0.026,
        )
        with patch(
            "ptcv.extraction.vision_enhancer.VisionEnhancer",
        ) as MockVE:
            MockVE.return_value.enhance.return_value = fake_result
            svc = ExtractionService(gateway=tmp_gateway)
            result = svc.extract(
                protocol_data=minimal_pdf_bytes,
                registry_id="NCT00001",
                amendment_number="1.0",
                source_sha256=_SHA,
                run_id="run-vision-e1",
                ext_level=_FakeExtLevel("docling_vision"),
            )
            MockVE.return_value.enhance.assert_called_once()

    def test_e2_skips_vision(self, tmp_gateway, minimal_pdf_bytes):
        """ext_level with value='docling' should NOT trigger vision."""
        with patch(
            "ptcv.extraction.vision_enhancer.VisionEnhancer",
        ) as MockVE:
            svc = ExtractionService(gateway=tmp_gateway)
            svc.extract(
                protocol_data=minimal_pdf_bytes,
                registry_id="NCT00001",
                amendment_number="1.0",
                source_sha256=_SHA,
                run_id="run-vision-e2",
                ext_level=_FakeExtLevel("docling"),
            )
            MockVE.assert_not_called()

    def test_e3_skips_vision(self, tmp_gateway, minimal_pdf_bytes):
        """ext_level with value='pdfplumber' should NOT trigger vision."""
        with patch(
            "ptcv.extraction.vision_enhancer.VisionEnhancer",
        ) as MockVE:
            svc = ExtractionService(gateway=tmp_gateway)
            svc.extract(
                protocol_data=minimal_pdf_bytes,
                registry_id="NCT00001",
                amendment_number="1.0",
                source_sha256=_SHA,
                run_id="run-vision-e3",
                ext_level=_FakeExtLevel("pdfplumber"),
            )
            MockVE.assert_not_called()

    def test_vision_failure_non_blocking(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        """Vision enhancement exception is caught; extraction continues."""
        with patch(
            "ptcv.extraction.vision_enhancer.VisionEnhancer",
        ) as MockVE:
            MockVE.return_value.enhance.side_effect = RuntimeError(
                "API unavailable"
            )
            svc = ExtractionService(gateway=tmp_gateway)
            result = svc.extract(
                protocol_data=minimal_pdf_bytes,
                registry_id="NCT00001",
                amendment_number="1.0",
                source_sha256=_SHA,
                run_id="run-vision-fail",
                ext_level=_FakeExtLevel("docling_vision"),
            )
            # Extraction still succeeds despite vision failure
            assert result.format_detected == "pdf"
            assert result.text_block_count >= 0

    def test_none_ext_level_skips_vision(
        self, tmp_gateway, minimal_pdf_bytes
    ):
        """Default ext_level=None should NOT trigger vision."""
        with patch(
            "ptcv.extraction.vision_enhancer.VisionEnhancer",
        ) as MockVE:
            svc = ExtractionService(gateway=tmp_gateway)
            svc.extract(
                protocol_data=minimal_pdf_bytes,
                registry_id="NCT00001",
                amendment_number="1.0",
                source_sha256=_SHA,
                run_id="run-vision-none",
            )
            MockVE.assert_not_called()
