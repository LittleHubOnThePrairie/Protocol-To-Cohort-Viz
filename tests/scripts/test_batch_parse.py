"""Tests for scripts/batch_parse.py (PTCV-42)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.batch_parse import (
    _build_parser,
    _needs_scoring,
    _resolve_pdf_path,
    run_batch,
)


# ---------------------------------------------------------------------------
# _needs_scoring
# ---------------------------------------------------------------------------

class TestNeedsScoring:
    """Tests for the _needs_scoring helper."""

    def test_empty_meta(self) -> None:
        assert _needs_scoring({}) is True

    def test_no_quality_scores(self) -> None:
        assert _needs_scoring({"title": "Test"}) is True

    def test_empty_quality_scores(self) -> None:
        assert _needs_scoring({"quality_scores": {}}) is True

    def test_null_quality_scores(self) -> None:
        assert _needs_scoring({"quality_scores": None}) is True

    def test_no_format_verdict(self) -> None:
        meta = {"quality_scores": {"format_confidence": 0.5}}
        assert _needs_scoring(meta) is True

    def test_empty_format_verdict(self) -> None:
        meta = {"quality_scores": {"format_verdict": ""}}
        assert _needs_scoring(meta) is True

    def test_scored(self) -> None:
        meta = {
            "quality_scores": {
                "format_verdict": "PARTIAL_ICH",
                "format_confidence": 0.42,
            }
        }
        assert _needs_scoring(meta) is False

    def test_scored_ich(self) -> None:
        meta = {
            "quality_scores": {
                "format_verdict": "ICH_E6R3",
                "format_confidence": 0.95,
            }
        }
        assert _needs_scoring(meta) is False

    def test_scored_non_ich(self) -> None:
        meta = {
            "quality_scores": {
                "format_verdict": "NON_ICH",
                "format_confidence": 0.0,
            }
        }
        assert _needs_scoring(meta) is False


# ---------------------------------------------------------------------------
# _resolve_pdf_path
# ---------------------------------------------------------------------------

class TestResolvePdfPath:
    """Tests for the _resolve_pdf_path helper."""

    def test_finds_clinicaltrials_pdf(self, tmp_path: Path) -> None:
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        pdf = ct_dir / "NCT00001234_1.0.pdf"
        pdf.write_bytes(b"%PDF-fake")

        meta = {
            "registry_id": "NCT00001234",
            "amendment_number": "1.0",
        }

        with patch("scripts.batch_parse._PROTOCOLS_DIR", tmp_path):
            result = _resolve_pdf_path(meta)
        assert result == pdf

    def test_finds_euctr_pdf(self, tmp_path: Path) -> None:
        eu_dir = tmp_path / "eu-ctr"
        eu_dir.mkdir()
        pdf = eu_dir / "NCT99999999_00.pdf"
        pdf.write_bytes(b"%PDF-fake")

        meta = {
            "registry_id": "NCT99999999",
            "amendment_number": "00",
        }

        with patch("scripts.batch_parse._PROTOCOLS_DIR", tmp_path):
            result = _resolve_pdf_path(meta)
        assert result == pdf

    def test_returns_none_no_pdf(self, tmp_path: Path) -> None:
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()

        meta = {"registry_id": "NCT00000000", "version": "1.0"}

        with patch("scripts.batch_parse._PROTOCOLS_DIR", tmp_path):
            result = _resolve_pdf_path(meta)
        assert result is None

    def test_fallback_to_file_path(self, tmp_path: Path) -> None:
        pdf = tmp_path / "custom.pdf"
        pdf.write_bytes(b"%PDF-fake")

        meta = {
            "registry_id": "NCT00000000",
            "file_path": str(pdf),
        }

        with patch("scripts.batch_parse._PROTOCOLS_DIR", tmp_path):
            result = _resolve_pdf_path(meta)
        assert result == pdf

    def test_version_fallback(self, tmp_path: Path) -> None:
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        pdf = ct_dir / "NCT12345678_2.0.pdf"
        pdf.write_bytes(b"%PDF-fake")

        meta = {
            "registry_id": "NCT12345678",
            "version": "2.0",
        }

        with patch("scripts.batch_parse._PROTOCOLS_DIR", tmp_path):
            result = _resolve_pdf_path(meta)
        assert result == pdf


# ---------------------------------------------------------------------------
# _build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    """Tests for the CLI argument parser."""

    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.dry_run is False
        assert args.limit is None

    def test_dry_run(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_limit(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--limit", "10"])
        assert args.limit == 10


# ---------------------------------------------------------------------------
# run_batch — integration with tmp_path
# ---------------------------------------------------------------------------

class TestRunBatch:
    """Tests for the run_batch function with mocked metadata dir."""

    def test_empty_metadata_dir(self, tmp_path: Path) -> None:
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        with patch(
            "scripts.batch_parse._METADATA_DIR", meta_dir
        ):
            result = run_batch(dry_run=True)

        assert result["total"] == 0
        assert result["scored"] == 0

    def test_missing_metadata_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"

        with patch(
            "scripts.batch_parse._METADATA_DIR", missing
        ):
            result = run_batch(dry_run=True)

        assert result == {
            "scored": 0,
            "skipped": 0,
            "failed": 0,
            "total": 0,
        }

    def test_skips_already_scored(self, tmp_path: Path) -> None:
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()

        meta = {
            "registry_id": "NCT11111111",
            "amendment_number": "1.0",
            "quality_scores": {
                "format_verdict": "PARTIAL_ICH",
                "format_confidence": 0.5,
            },
        }
        (meta_dir / "NCT11111111.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        (ct_dir / "NCT11111111_1.0.pdf").write_bytes(
            b"%PDF-fake"
        )

        with (
            patch("scripts.batch_parse._METADATA_DIR", meta_dir),
            patch("scripts.batch_parse._PROTOCOLS_DIR", tmp_path),
        ):
            result = run_batch(dry_run=True)

        assert result["skipped"] == 1
        assert result["scored"] == 0

    def test_dry_run_counts_unscored(self, tmp_path: Path) -> None:
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()

        meta = {
            "registry_id": "NCT22222222",
            "amendment_number": "1.0",
            "condition": "Cancer",
        }
        (meta_dir / "NCT22222222.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        (ct_dir / "NCT22222222_1.0.pdf").write_bytes(
            b"%PDF-fake"
        )

        with (
            patch("scripts.batch_parse._METADATA_DIR", meta_dir),
            patch("scripts.batch_parse._PROTOCOLS_DIR", tmp_path),
        ):
            result = run_batch(dry_run=True)

        assert result["scored"] == 1
        assert result["skipped"] == 0

    def test_limit_respected(self, tmp_path: Path) -> None:
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()

        for i in range(5):
            nct = f"NCT{i:08d}"
            meta = {
                "registry_id": nct,
                "amendment_number": "1.0",
            }
            (meta_dir / f"{nct}.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            (ct_dir / f"{nct}_1.0.pdf").write_bytes(
                b"%PDF-fake"
            )

        with (
            patch("scripts.batch_parse._METADATA_DIR", meta_dir),
            patch("scripts.batch_parse._PROTOCOLS_DIR", tmp_path),
        ):
            result = run_batch(dry_run=True, limit=3)

        assert result["scored"] == 3

    def test_bad_json_increments_failed(
        self, tmp_path: Path
    ) -> None:
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        (meta_dir / "bad.json").write_text(
            "not json at all", encoding="utf-8"
        )

        with patch(
            "scripts.batch_parse._METADATA_DIR", meta_dir
        ):
            result = run_batch(dry_run=True)

        assert result["failed"] == 1
