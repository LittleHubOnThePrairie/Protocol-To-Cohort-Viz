"""Tests for query pipeline artifact persistence (PTCV-126)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    AssembledSection,
    CoverageReport,
    QueryExtractionHit,
    SourceReference,
    assemble_template,
)
from ptcv.ui.query_persistence import (
    load_assembled_protocol,
    save_query_artifacts,
)


FILE_SHA = "abc123def456"


def _make_assembled() -> AssembledProtocol:
    """Create a minimal AssembledProtocol for testing."""
    hits = [
        QueryExtractionHit(
            query_id="B.1.1.q1",
            section_id="B.1.1",
            parent_section="B.1",
            query_text="What is the trial title?",
            extracted_content="A Phase 3 Study",
            confidence=0.92,
            source=SourceReference(
                pdf_page=3,
                section_header="1.0 Title",
            ),
        ),
    ]
    return assemble_template(hits)


class TestAssembledProtocolRoundTrip:
    """Test to_dict / from_dict round-trip."""

    def test_round_trip_preserves_sections(self) -> None:
        original = _make_assembled()
        d = original.to_dict()
        restored = AssembledProtocol.from_dict(d)
        assert len(restored.sections) == len(original.sections)

    def test_round_trip_preserves_coverage(self) -> None:
        original = _make_assembled()
        d = original.to_dict()
        restored = AssembledProtocol.from_dict(d)
        assert (
            restored.coverage.populated_count
            == original.coverage.populated_count
        )
        assert (
            restored.coverage.gap_count
            == original.coverage.gap_count
        )

    def test_round_trip_preserves_hit_content(self) -> None:
        original = _make_assembled()
        d = original.to_dict()
        restored = AssembledProtocol.from_dict(d)
        b1 = restored.get_section("B.1")
        assert b1 is not None
        assert b1.populated
        assert b1.hits[0].extracted_content == "A Phase 3 Study"

    def test_round_trip_preserves_traceability(self) -> None:
        original = _make_assembled()
        d = original.to_dict()
        restored = AssembledProtocol.from_dict(d)
        assert "B.1" in restored.source_traceability
        ref = restored.source_traceability["B.1"][0]
        assert ref.pdf_page == 3
        assert ref.section_header == "1.0 Title"

    def test_round_trip_preserves_parent_section(self) -> None:
        original = _make_assembled()
        d = original.to_dict()
        restored = AssembledProtocol.from_dict(d)
        b1 = restored.get_section("B.1")
        assert b1 is not None
        assert b1.hits[0].parent_section == "B.1"

    def test_json_serializable(self) -> None:
        """to_dict output can survive json.dumps/loads."""
        original = _make_assembled()
        d = original.to_dict()
        text = json.dumps(d)
        d2 = json.loads(text)
        restored = AssembledProtocol.from_dict(d2)
        assert restored.coverage.total_sections == 16


class TestSaveAndLoad:
    """Test save_query_artifacts + load_assembled_protocol."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        assembled = _make_assembled()
        result = {
            "assembled": assembled,
            "assembled_markdown": assembled.to_markdown(),
            "coverage": assembled.coverage,
            "stage_timings": {"doc_assembly": 1.2},
        }

        # Build a mock gateway that writes to tmp_path
        stored: dict[str, bytes] = {}

        def mock_put(key, data, **kwargs):
            stored[key] = data
            return MagicMock(sha256="fakehash")

        def mock_get(key, version_id=""):
            if key not in stored:
                raise FileNotFoundError(key)
            return stored[key]

        gw = MagicMock()
        gw.put_artifact = mock_put
        gw.get_artifact = mock_get

        run_id = save_query_artifacts(
            gw, FILE_SHA, result, registry_id="NCT12345",
        )
        assert run_id  # non-empty UUID

        # Verify manifest was stored
        manifest_key = f"query-pipeline/{FILE_SHA}/manifest.json"
        assert manifest_key in stored

        # Load back
        loaded = load_assembled_protocol(gw, FILE_SHA)
        assert loaded is not None
        assert loaded.coverage.total_sections == 16
        b1 = loaded.get_section("B.1")
        assert b1 is not None
        assert b1.hits[0].extracted_content == "A Phase 3 Study"

    def test_load_returns_none_when_no_manifest(self) -> None:
        gw = MagicMock()
        gw.get_artifact.side_effect = FileNotFoundError

        result = load_assembled_protocol(gw, FILE_SHA)
        assert result is None
