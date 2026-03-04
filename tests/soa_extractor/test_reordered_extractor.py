"""Tests for SoaExtractor document-first fallback order (PTCV-60).

Validates that SoaExtractor works without ICH sections and that the
table bridge → PDF discovery → text parsing fallback chain is correct.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.soa_extractor import SoaExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(text: str):
    """Build a minimal IchSection with the given text."""
    from ptcv.ich_parser.models import IchSection

    return IchSection(
        run_id="r1",
        source_run_id="",
        source_sha256="a" * 64,
        registry_id="NCT0001",
        section_code="B.4",
        section_name="Trial Design",
        content_json=json.dumps({"text": text}),
        confidence_score=0.90,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
    )


SOA_TABLE_TEXT = """\
Schedule of Activities

| Assessment | Screening | Baseline | Week 2 | Week 4 |
|------------|-----------|----------|--------|--------|
| ECG        | X         |          | X      |        |
| Labs       | X         | X        | X      | X      |
| Vitals     | X         | X        | X      | X      |
"""


# ---------------------------------------------------------------------------
# No input raises
# ---------------------------------------------------------------------------


class TestNoInputRaises:
    def test_no_sources_raises_value_error(
        self, tmp_gateway, tmp_review_queue,
    ):
        extractor = SoaExtractor(
            gateway=tmp_gateway, review_queue=tmp_review_queue,
        )
        with pytest.raises(ValueError, match="At least one input source"):
            extractor.extract(registry_id="NCT0001")

    def test_empty_sections_no_other_sources_raises(
        self, tmp_gateway, tmp_review_queue,
    ):
        extractor = SoaExtractor(
            gateway=tmp_gateway, review_queue=tmp_review_queue,
        )
        with pytest.raises(ValueError):
            extractor.extract(sections=[], registry_id="NCT0001")


# ---------------------------------------------------------------------------
# Sections still work (backward compat)
# ---------------------------------------------------------------------------


class TestSectionsBackwardCompat:
    def test_extract_with_sections_only(
        self, tmp_gateway, tmp_review_queue,
    ):
        """ICH sections alone still work as a valid input source."""
        section = _make_section(SOA_TABLE_TEXT)
        extractor = SoaExtractor(
            gateway=tmp_gateway, review_queue=tmp_review_queue,
        )
        result = extractor.extract(
            sections=[section],
            registry_id="NCT0001",
            source_sha256="a" * 64,
        )
        assert result is not None
        assert result.registry_id == "NCT0001"
        assert result.timepoint_count >= 0

    def test_extract_sections_keyword_arg(
        self, tmp_gateway, tmp_review_queue,
    ):
        """Sections can be passed as keyword argument."""
        section = _make_section(SOA_TABLE_TEXT)
        extractor = SoaExtractor(
            gateway=tmp_gateway, review_queue=tmp_review_queue,
        )
        result = extractor.extract(
            registry_id="NCT0001",
            sections=[section],
            source_sha256="a" * 64,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Extract without sections
# ---------------------------------------------------------------------------


class TestExtractWithoutSections:
    def test_extract_with_pdf_bytes_only(
        self, tmp_gateway, tmp_review_queue,
    ):
        """PDF bytes alone are accepted (no sections needed)."""
        extractor = SoaExtractor(
            gateway=tmp_gateway, review_queue=tmp_review_queue,
        )
        # Minimal bytes — table discovery will find nothing, but no error
        result = extractor.extract(
            registry_id="NCT0001",
            pdf_bytes=b"fake-pdf-bytes",
            text_blocks=[{"page_number": 1, "text": "no SoA here"}],
            source_sha256="a" * 64,
        )
        assert result is not None
        # Zero entities is fine — the point is no ValueError
        assert result.timepoint_count >= 0

    def test_extract_with_extracted_tables_empty_raises(
        self, tmp_gateway, tmp_review_queue,
    ):
        """Empty extracted_tables list with no other inputs raises."""
        extractor = SoaExtractor(
            gateway=tmp_gateway, review_queue=tmp_review_queue,
        )
        with pytest.raises(ValueError):
            extractor.extract(
                registry_id="NCT0001",
                extracted_tables=[],
                source_sha256="a" * 64,
            )

    def test_extract_with_extracted_tables_and_pdf_no_error(
        self, tmp_gateway, tmp_review_queue,
    ):
        """Extracted tables + pdf_bytes together accepted."""
        extractor = SoaExtractor(
            gateway=tmp_gateway, review_queue=tmp_review_queue,
        )
        result = extractor.extract(
            registry_id="NCT0001",
            extracted_tables=[],
            pdf_bytes=b"fake-pdf",
            text_blocks=[{"page_number": 1, "text": "content"}],
            source_sha256="a" * 64,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Zero tables returns empty
# ---------------------------------------------------------------------------


class TestZeroTablesEmpty:
    def test_no_soa_tables_returns_zero_entities(
        self, tmp_gateway, tmp_review_queue,
    ):
        """When no SoA tables found, all entity counts are 0."""
        extractor = SoaExtractor(
            gateway=tmp_gateway, review_queue=tmp_review_queue,
        )
        result = extractor.extract(
            registry_id="NCT0001",
            pdf_bytes=b"no-soa",
            text_blocks=[{"page_number": 1, "text": "intro"}],
            source_sha256="a" * 64,
        )
        assert result.timepoint_count == 0
        assert result.activity_count == 0
        assert result.epoch_count == 0
