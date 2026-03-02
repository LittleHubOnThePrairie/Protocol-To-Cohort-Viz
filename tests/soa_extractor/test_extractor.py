"""Tests for SoaExtractor — full pipeline GHERKIN scenarios (PTCV-21)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
from ptcv.soa_extractor import SoaExtractor
from ptcv.soa_extractor.writer import UsdmParquetWriter


class TestExtractParquetAndLineage:
    """GHERKIN: Extract SoA to USDM Parquet with lineage."""

    def test_extract_returns_result(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_run_id="ich-run-001",
            source_sha256="a" * 64,
        )
        assert result is not None
        assert result.registry_id == "NCT00112827"
        assert result.run_id  # non-empty UUID

    def test_parquet_files_written(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        """Epochs, timepoints, activities, instances, synonyms all written."""
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_run_id="ich-run-001",
            source_sha256="a" * 64,
        )
        expected_artifacts = {
            "epochs", "timepoints", "activities",
            "scheduled_instances", "synonym_mappings",
        }
        assert set(result.artifact_keys.keys()) >= expected_artifacts

    def test_lineage_stage_is_soa_extraction(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        """LineageRecord has stage='soa_extraction'."""
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        lineage = tmp_gateway.get_lineage(result.run_id)
        assert lineage, "No lineage records written"
        stages = [rec.stage for rec in lineage]
        assert "soa_extraction" in stages

    def test_lineage_source_hash_matches(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        """source_hash in lineage record matches the SHA-256 passed in."""
        sha = "c" * 64
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256=sha,
        )
        lineage = tmp_gateway.get_lineage(result.run_id)
        source_hashes = [rec.source_hash for rec in lineage]
        assert sha in source_hashes

    def test_timepoints_count_positive(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        assert result.timepoint_count > 0

    def test_activities_count_positive(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        assert result.activity_count > 0


class TestSevenRequiredAttributes:
    """GHERKIN: All 7 required SoA attributes present in timepoints.parquet."""

    def test_timepoints_have_all_required_attributes(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        key = result.artifact_keys["timepoints"]
        data = tmp_gateway.get_artifact(key)
        writer = UsdmParquetWriter()
        timepoints = writer.parquet_to_timepoints(data)

        for tp in timepoints:
            assert tp.visit_name, f"visit_name empty for {tp.timepoint_id}"
            assert tp.visit_type, f"visit_type empty for {tp.timepoint_id}"
            assert tp.day_offset is not None
            assert tp.window_early is not None
            assert tp.window_late is not None
            assert tp.mandatory is not None
            assert tp.extraction_timestamp_utc, "timestamp missing"

    def test_missing_attribute_sets_review_required(
        self, tmp_gateway, tmp_review_queue
    ):
        """Timepoint with unknown temporal → review_required=True."""
        from ptcv.ich_parser.models import IchSection

        text = """\
| Assessment | Some Unknown Visit |
|------------|--------------------|
| ECG        | X                  |
"""
        section = IchSection(
            run_id="r1", source_run_id="", source_sha256="b" * 64,
            registry_id="NCT0002",
            section_code="B.4", section_name="Design",
            content_json=json.dumps({"text": text}),
            confidence_score=0.9, review_required=False, legacy_format=False,
            extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
        )
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=[section], registry_id="NCT0002", source_sha256="b" * 64
        )
        if result.timepoint_count > 0:
            key = result.artifact_keys["timepoints"]
            data = tmp_gateway.get_artifact(key)
            writer = UsdmParquetWriter()
            timepoints = writer.parquet_to_timepoints(data)
            # Default fallback confidence < 0.80 → review_required=True
            review_tps = [tp for tp in timepoints if tp.review_required]
            # At least one should be flagged (default method produces 0.60)
            assert len(review_tps) >= 1


class TestSynonymResolutionLogged:
    """GHERKIN: Synonym resolution logged in synonym_mappings.parquet."""

    def test_synonym_mappings_written(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        assert result.synonym_mapping_count > 0
        assert "synonym_mappings" in result.artifact_keys

    def test_visit_zero_and_pre_treatment_mapped_to_screening(
        self, synonym_ich_sections, tmp_gateway, tmp_review_queue
    ):
        """GHERKIN: 'Visit 0' and 'Pre-treatment' both map to 'Screening'."""
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=synonym_ich_sections,
            registry_id="NCT00112828",
            source_sha256="b" * 64,
        )
        key = result.artifact_keys["synonym_mappings"]
        data = tmp_gateway.get_artifact(key)
        writer = UsdmParquetWriter()
        mappings = writer.parquet_to_synonyms(data)

        originals = {m.original_text: m.canonical_label for m in mappings}
        assert originals.get("Visit 0") == "Screening"
        assert originals.get("Pre-treatment") == "Screening"

    def test_mapping_fields_all_present(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        """Each SynonymMapping has original_text, canonical_label, method, confidence."""
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        key = result.artifact_keys["synonym_mappings"]
        data = tmp_gateway.get_artifact(key)
        writer = UsdmParquetWriter()
        mappings = writer.parquet_to_synonyms(data)
        for m in mappings:
            assert m.original_text
            assert m.canonical_label
            assert m.method
            assert 0.0 <= m.confidence <= 1.0
            assert m.run_id == result.run_id


class TestUnscheduledAndEarlyTermination:
    """GHERKIN: Unscheduled and early termination visits captured."""

    def test_unscheduled_visit_conditional_rule_prn(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        key = result.artifact_keys["timepoints"]
        data = tmp_gateway.get_artifact(key)
        writer = UsdmParquetWriter()
        timepoints = writer.parquet_to_timepoints(data)

        unscheduled = [tp for tp in timepoints if tp.visit_type == "Unscheduled"]
        assert unscheduled, "No Unscheduled timepoints found"
        for tp in unscheduled:
            assert tp.conditional_rule == "PRN"

    def test_early_termination_conditional_rule(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        key = result.artifact_keys["timepoints"]
        data = tmp_gateway.get_artifact(key)
        writer = UsdmParquetWriter()
        timepoints = writer.parquet_to_timepoints(data)

        early_term = [tp for tp in timepoints if tp.visit_type == "Early Termination"]
        assert early_term, "No Early Termination timepoints found"
        for tp in early_term:
            assert tp.conditional_rule == "EARLY_TERM"

    def test_visit_types_match_9_type_taxonomy(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        from ptcv.soa_extractor.resolver import VISIT_TYPES

        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        key = result.artifact_keys["timepoints"]
        data = tmp_gateway.get_artifact(key)
        writer = UsdmParquetWriter()
        timepoints = writer.parquet_to_timepoints(data)

        for tp in timepoints:
            assert tp.visit_type in VISIT_TYPES, (
                f"Unknown visit_type: {tp.visit_type!r}"
            )


class TestLineageChainVerifiable:
    """GHERKIN: Lineage chain verifiable from USDM back to download."""

    def test_lineage_records_present(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        lineage = tmp_gateway.get_lineage(result.run_id)
        assert len(lineage) >= 1

    def test_run_id_is_uuid4_format(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        import uuid

        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        result = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        # Should parse as UUID4 without error
        parsed = uuid.UUID(result.run_id)
        assert str(parsed) == result.run_id

    def test_rerun_creates_new_run_id(
        self, sample_ich_sections, tmp_gateway, tmp_review_queue
    ):
        """ALCOA+ Original: re-run produces a different run_id."""
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        r1 = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        r2 = extractor.extract(
            sections=sample_ich_sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        assert r1.run_id != r2.run_id

    def test_empty_sections_raises(self, tmp_gateway, tmp_review_queue):
        extractor = SoaExtractor(gateway=tmp_gateway, review_queue=tmp_review_queue)
        with pytest.raises(ValueError):
            extractor.extract(sections=[], registry_id="NCT0001")
