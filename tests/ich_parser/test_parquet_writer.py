"""Tests for ParquetWriter (PTCV-20).

Verifies Parquet round-trip fidelity, schema enforcement, and the
confidence_score non-null constraint (ALCOA+ Accurate).
"""

from __future__ import annotations

import pytest

from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.parquet_writer import sections_to_parquet, parquet_to_sections


def _section(
    section_code: str = "B.3",
    confidence: float = 0.85,
    legacy: bool = False,
) -> IchSection:
    return IchSection(
        run_id="run-parquet-test",
        source_run_id="source-run",
        source_sha256="deadbeef",
        registry_id="EUCT-2024-001",
        section_code=section_code,
        section_name="Test Section",
        content_json='{"text_excerpt": "objectives text"}',
        confidence_score=confidence,
        review_required=confidence < 0.70,
        legacy_format=legacy,
        extraction_timestamp_utc="2026-03-02T12:00:00+00:00",
    )


class TestParquetWriter:
    def test_round_trip_single_section(self) -> None:
        sections = [_section("B.3", 0.85)]
        data = sections_to_parquet(sections)
        result = parquet_to_sections(data)
        assert len(result) == 1
        r = result[0]
        assert r.section_code == "B.3"
        assert abs(r.confidence_score - 0.85) < 1e-4
        assert r.review_required is False
        assert r.legacy_format is False

    def test_round_trip_multiple_sections(self) -> None:
        sections = [_section("B.1"), _section("B.3"), _section("B.9", 0.60)]
        data = sections_to_parquet(sections)
        result = parquet_to_sections(data)
        assert len(result) == 3
        codes = [r.section_code for r in result]
        assert codes == ["B.1", "B.3", "B.9"]

    def test_confidence_score_preserved_as_float32(self) -> None:
        sections = [_section("B.5", 0.7777)]
        data = sections_to_parquet(sections)
        result = parquet_to_sections(data)
        # float32 precision: within 1e-4
        assert abs(result[0].confidence_score - 0.7777) < 1e-4

    def test_review_required_preserved(self) -> None:
        sections = [
            _section("B.1", 0.85),   # review_required=False
            _section("B.7", 0.50),   # review_required=True
        ]
        data = sections_to_parquet(sections)
        result = parquet_to_sections(data)
        assert result[0].review_required is False
        assert result[1].review_required is True

    def test_legacy_format_preserved(self) -> None:
        sections = [_section("B.1", 0.30, legacy=True)]
        data = sections_to_parquet(sections)
        result = parquet_to_sections(data)
        assert result[0].legacy_format is True

    def test_raises_on_empty_list(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sections_to_parquet([])

    def test_raises_when_confidence_score_missing(self) -> None:
        """ALCOA+ Accurate: confidence_score must not be None."""
        sec = _section("B.3", 0.8)
        sec.confidence_score = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="confidence_score is None"):
            sections_to_parquet([sec])

    def test_raises_when_timestamp_empty(self) -> None:
        sec = _section("B.3", 0.8)
        sec.extraction_timestamp_utc = ""
        with pytest.raises(ValueError, match="extraction_timestamp_utc is empty"):
            sections_to_parquet([sec])

    def test_output_is_bytes(self) -> None:
        data = sections_to_parquet([_section()])
        assert isinstance(data, bytes)
        # Parquet magic bytes: PAR1
        assert data[:4] == b"PAR1"

    def test_all_schema_fields_round_trip(self) -> None:
        sec = IchSection(
            run_id="r1",
            source_run_id="s1",
            source_sha256="sha256hex",
            registry_id="NCT-00000000",
            section_code="B.11",
            section_name="Quality Assurance",
            content_json='{"note": "GCP compliance"}',
            confidence_score=0.92,
            review_required=False,
            legacy_format=False,
            extraction_timestamp_utc="2026-01-01T00:00:00+00:00",
        )
        data = sections_to_parquet([sec])
        result = parquet_to_sections(data)
        r = result[0]
        assert r.run_id == "r1"
        assert r.source_run_id == "s1"
        assert r.source_sha256 == "sha256hex"
        assert r.registry_id == "NCT-00000000"
        assert r.section_name == "Quality Assurance"
        assert r.extraction_timestamp_utc == "2026-01-01T00:00:00+00:00"
