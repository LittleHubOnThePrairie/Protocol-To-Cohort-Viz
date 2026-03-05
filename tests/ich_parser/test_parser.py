"""Tests for IchParser — main ICH E6(R3) parsing service (PTCV-20).

All scenarios from the PTCV-20 GHERKIN acceptance criteria are covered.
Uses FilesystemAdapter + RuleBasedClassifier (no API keys required).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser import IchParser, IchSection, RuleBasedClassifier
from ptcv.ich_parser.schema_loader import get_review_threshold
from ptcv.ich_parser.parquet_writer import parquet_to_sections
from ptcv.storage import FilesystemAdapter


REGISTRY_ID = "EUCT-2024-000001-01"
SOURCE_RUN_ID = "source-run-abc"
SOURCE_SHA = "aabbccdd1122"


def _make_parser(tmp_path: Path) -> IchParser:
    gw = FilesystemAdapter(root=tmp_path)
    return IchParser(gateway=gw, classifier=RuleBasedClassifier(), review_queue=None)


class TestIchParserScenarios:
    """GHERKIN scenario coverage (PTCV-20 acceptance criteria)."""

    # ------------------------------------------------------------------
    # Scenario: Classify and write to Parquet with lineage
    # ------------------------------------------------------------------

    def test_parse_writes_parquet_to_gateway(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """Scenario: Classify protocol sections and write to Parquet with lineage."""
        parser = _make_parser(tmp_path)
        result = parser.parse(
            text=sample_text,
            registry_id=REGISTRY_ID,
            source_run_id=SOURCE_RUN_ID,
            source_sha256=SOURCE_SHA,
        )
        assert result.section_count > 0
        parquet_path = tmp_path / result.artifact_key
        assert parquet_path.exists(), f"Parquet not written at {result.artifact_key}"

    def test_parse_lineage_record_written_with_ich_parse_stage(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """LineageRecord stage must be 'ich_parse'."""
        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(gateway=gw, classifier=RuleBasedClassifier())
        result = parser.parse(
            text=sample_text,
            registry_id=REGISTRY_ID,
            source_run_id=SOURCE_RUN_ID,
            source_sha256=SOURCE_SHA,
        )
        lineage = gw.get_lineage(result.run_id)
        assert len(lineage) == 1
        assert lineage[0].stage == "ich_parse"

    def test_parse_lineage_source_hash_matches_input(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """source_hash in LineageRecord == source_sha256 passed in."""
        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(gateway=gw, classifier=RuleBasedClassifier())
        result = parser.parse(
            text=sample_text,
            registry_id=REGISTRY_ID,
            source_run_id=SOURCE_RUN_ID,
            source_sha256=SOURCE_SHA,
        )
        lineage = gw.get_lineage(result.run_id)
        assert lineage[0].source_hash == SOURCE_SHA

    # ------------------------------------------------------------------
    # Scenario: Confidence score persisted as required column
    # ------------------------------------------------------------------

    def test_all_parquet_rows_have_non_null_confidence(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """ALCOA+ Accurate: every row has a non-null confidence_score."""
        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(gateway=gw, classifier=RuleBasedClassifier())
        result = parser.parse(
            text=sample_text, registry_id=REGISTRY_ID, source_sha256=SOURCE_SHA
        )
        parquet_bytes = gw.get_artifact(result.artifact_key)
        sections = parquet_to_sections(parquet_bytes)
        for sec in sections:
            assert sec.confidence_score is not None
            assert 0.0 <= sec.confidence_score <= 1.0

    # ------------------------------------------------------------------
    # Scenario: Low-confidence sections routed to review queue
    # ------------------------------------------------------------------

    def test_low_confidence_sections_enqueued(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """Sections below threshold appear in review_queue.db."""
        from ptcv.ich_parser import ReviewQueue
        rq = ReviewQueue(db_path=tmp_path / "review_queue.db")
        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(
            gateway=gw, classifier=RuleBasedClassifier(), review_queue=rq
        )
        result = parser.parse(
            text=sample_text, registry_id=REGISTRY_ID, source_sha256=SOURCE_SHA
        )
        pending = rq.pending()
        # Every pending entry should have confidence below REVIEW_THRESHOLD
        for entry in pending:
            assert entry.confidence_score < get_review_threshold()

        assert result.review_count == len(pending)

    def test_review_queue_timestamp_matches_parquet(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """Queue timestamp should match extraction_timestamp_utc in Parquet."""
        from ptcv.ich_parser import ReviewQueue
        rq = ReviewQueue(db_path=tmp_path / "review_queue.db")
        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(
            gateway=gw, classifier=RuleBasedClassifier(), review_queue=rq
        )
        result = parser.parse(
            text=sample_text, registry_id=REGISTRY_ID, source_sha256=SOURCE_SHA
        )
        parquet_bytes = gw.get_artifact(result.artifact_key)
        sections = parquet_to_sections(parquet_bytes)
        parquet_low = {
            s.section_code: s.extraction_timestamp_utc
            for s in sections
            if s.review_required
        }
        for entry in rq.pending():
            if entry.section_code in parquet_low:
                assert entry.queue_timestamp_utc == parquet_low[entry.section_code]

    # ------------------------------------------------------------------
    # Scenario: Legacy protocol handled with flag
    # ------------------------------------------------------------------

    def test_legacy_fallback_for_unclassifiable_text(
        self, tmp_path: Path
    ) -> None:
        """When no sections detected, legacy fallback produces B.1 with legacy_format=True."""
        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(gateway=gw, classifier=RuleBasedClassifier())
        result = parser.parse(
            text="Lorem ipsum dolor sit amet " * 50,
            registry_id="EUCT-LEGACY-001",
        )
        parquet_bytes = gw.get_artifact(result.artifact_key)
        sections = parquet_to_sections(parquet_bytes)
        # Must have at least one section
        assert len(sections) >= 1
        # If only the fallback was triggered, all sections should be legacy
        legacy_sections = [s for s in sections if s.legacy_format]
        assert len(legacy_sections) >= 1
        # Fallback section must be review_required
        fallback = next(s for s in sections if s.legacy_format)
        assert fallback.review_required is True

    def test_legacy_lineage_record_written(
        self, tmp_path: Path
    ) -> None:
        """Legacy fallback still writes a LineageRecord with stage=ich_parse."""
        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(gateway=gw, classifier=RuleBasedClassifier())
        result = parser.parse(
            text="Lorem ipsum " * 50, registry_id="EUCT-LEGACY-002"
        )
        lineage = gw.get_lineage(result.run_id)
        assert any(lr.stage == "ich_parse" for lr in lineage)

    # ------------------------------------------------------------------
    # Scenario: Re-run creates new run_id, prior parse preserved
    # ------------------------------------------------------------------

    def test_rerun_creates_new_run_id(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """Each parse() call creates a distinct run_id."""
        parser = _make_parser(tmp_path)
        r1 = parser.parse(text=sample_text, registry_id=REGISTRY_ID)
        r2 = parser.parse(text=sample_text, registry_id=REGISTRY_ID)
        assert r1.run_id != r2.run_id

    def test_rerun_preserves_prior_parquet(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        """Prior Parquet under old run_id must remain after re-run."""
        gw = FilesystemAdapter(root=tmp_path)
        parser = IchParser(gateway=gw, classifier=RuleBasedClassifier())
        r1 = parser.parse(text=sample_text, registry_id=REGISTRY_ID)
        r2 = parser.parse(text=sample_text, registry_id=REGISTRY_ID)
        # Both keys must exist
        assert (tmp_path / r1.artifact_key).exists()
        assert (tmp_path / r2.artifact_key).exists()
        assert r1.artifact_key != r2.artifact_key

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_raises_on_empty_text(self, tmp_path: Path) -> None:
        parser = _make_parser(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            parser.parse(text="   ", registry_id=REGISTRY_ID)

    def test_parse_result_artifact_sha256_non_empty(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        parser = _make_parser(tmp_path)
        result = parser.parse(text=sample_text, registry_id=REGISTRY_ID)
        assert result.artifact_sha256
        assert len(result.artifact_sha256) == 64  # SHA-256 hex

    def test_parse_artifact_key_contains_run_id(
        self, tmp_path: Path, sample_text: str
    ) -> None:
        parser = _make_parser(tmp_path)
        result = parser.parse(text=sample_text, registry_id=REGISTRY_ID)
        assert result.run_id in result.artifact_key
        assert result.artifact_key.endswith("sections.parquet")
