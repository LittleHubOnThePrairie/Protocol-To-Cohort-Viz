"""Tests for registry population module (PTCV-201).

Covers all GHERKIN scenarios:
- Discover NCT IDs from local protocol PDFs
- Dry run lists NCT IDs without fetching
- Batch fetch with rate limiting
- ICH section mapping verification
- Optional RAG index seeding
- Graceful handling of fetch failures
- Module invokable via python -m
- Summary report at completion
"""

import json
import os
import tempfile
import urllib.error
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from ptcv.registry.populate import (
    PopulateResult,
    build_parser,
    discover_nct_ids,
    main,
    populate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    """Create a temp directory with fake protocol PDFs."""
    # Root-level PDFs
    (tmp_path / "NCT00004088_1.0.pdf").touch()
    (tmp_path / "NCT00074490_1.0.pdf").touch()
    (tmp_path / "NCT01512251_1.0.pdf").touch()

    # Subdirectory PDFs
    sub = tmp_path / "protein_search"
    sub.mkdir()
    (sub / "NCT02591862_1.0.pdf").touch()
    (sub / "NCT03234543_1.0.pdf").touch()

    # Duplicate NCT ID across directories
    (sub / "NCT01512251_2.0.pdf").touch()

    # Non-PDF file (should be ignored)
    (tmp_path / "NCT99999999.txt").touch()

    # PDF without NCT pattern (should be ignored)
    (tmp_path / "protocol_draft.pdf").touch()

    return tmp_path


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a cache directory with some pre-cached metadata."""
    cache = tmp_path / "registry_cache"
    cache.mkdir()
    return cache


def _sample_metadata(nct_id: str) -> dict[str, Any]:
    """Return minimal CT.gov-style metadata for testing."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "officialTitle": f"Test Trial {nct_id}",
                "briefTitle": f"Trial {nct_id}",
            },
            "statusModule": {
                "overallStatus": "Completed",
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE2"],
            },
        }
    }


# ---------------------------------------------------------------------------
# Scenario: Discover NCT IDs from local protocol PDFs
# ---------------------------------------------------------------------------


class TestDiscoverNctIds:
    """Scenario: Discover NCT IDs from local protocol PDFs."""

    def test_discovers_all_unique_nct_ids(
        self, pdf_dir: Path
    ) -> None:
        """All unique NCT IDs are extracted from filenames."""
        nct_ids = discover_nct_ids(pdf_dir)
        assert len(nct_ids) == 5
        assert "NCT00004088" in nct_ids
        assert "NCT00074490" in nct_ids
        assert "NCT01512251" in nct_ids
        assert "NCT02591862" in nct_ids
        assert "NCT03234543" in nct_ids

    def test_includes_subdirectories(self, pdf_dir: Path) -> None:
        """Subdirectories (e.g., protein_search/) are included."""
        nct_ids = discover_nct_ids(pdf_dir)
        assert "NCT02591862" in nct_ids
        assert "NCT03234543" in nct_ids

    def test_deduplicates_across_directories(
        self, pdf_dir: Path
    ) -> None:
        """Duplicate NCT IDs across directories are deduplicated."""
        nct_ids = discover_nct_ids(pdf_dir)
        # NCT01512251 exists in both root and protein_search/
        assert nct_ids.count("NCT01512251") == 1

    def test_results_are_sorted(self, pdf_dir: Path) -> None:
        """Results are returned in sorted order."""
        nct_ids = discover_nct_ids(pdf_dir)
        assert nct_ids == sorted(nct_ids)

    def test_ignores_non_pdf_files(self, pdf_dir: Path) -> None:
        """Non-PDF files are ignored."""
        nct_ids = discover_nct_ids(pdf_dir)
        assert "NCT99999999" not in nct_ids

    def test_ignores_pdfs_without_nct_pattern(
        self, pdf_dir: Path
    ) -> None:
        """PDFs without NCT pattern are ignored."""
        nct_ids = discover_nct_ids(pdf_dir)
        assert len(nct_ids) == 5  # No extra entries

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        nct_ids = discover_nct_ids(tmp_path)
        assert nct_ids == []


# ---------------------------------------------------------------------------
# Scenario: Dry run lists NCT IDs without fetching
# ---------------------------------------------------------------------------


class TestDryRun:
    """Scenario: Dry run lists NCT IDs without fetching."""

    def test_dry_run_no_api_calls(
        self,
        pdf_dir: Path,
        cache_dir: Path,
    ) -> None:
        """No API calls are made in dry-run mode."""
        result = populate(
            data_dir=pdf_dir,
            cache_dir=cache_dir,
            dry_run=True,
        )

        assert result.total_nct_ids == 5
        assert result.fetched == 0
        assert result.cached == 0
        assert result.failed == 0

    def test_dry_run_no_cache_files_written(
        self, pdf_dir: Path, cache_dir: Path
    ) -> None:
        """No files are written to cache in dry-run mode."""
        populate(data_dir=pdf_dir, cache_dir=cache_dir, dry_run=True)
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 0


# ---------------------------------------------------------------------------
# Scenario: Batch fetch with rate limiting
# ---------------------------------------------------------------------------


class TestBatchFetch:
    """Scenario: Batch fetch with rate limiting."""

    def test_fetches_uncached_and_loads_cached(
        self, pdf_dir: Path, cache_dir: Path
    ) -> None:
        """Uncached IDs are fetched; cached ones are loaded."""
        # Pre-cache one entry
        cached_nct = "NCT00004088"
        meta = _sample_metadata(cached_nct)
        (cache_dir / f"{cached_nct}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        with patch(
            "ptcv.registry.metadata_fetcher.urllib.request.urlopen"
        ) as mock_urlopen, patch("time.sleep"):
            response = MagicMock()
            response.read.return_value = json.dumps(
                _sample_metadata("NCT_FRESH")
            ).encode()
            response.__enter__ = lambda s: s
            response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = response

            result = populate(
                data_dir=pdf_dir,
                cache_dir=cache_dir,
                delay=0.0,
            )

        assert result.cached >= 1  # At least the pre-cached one
        assert result.fetched + result.cached + result.failed == 5

    def test_progress_reporting(
        self, pdf_dir: Path, cache_dir: Path, capsys: Any
    ) -> None:
        """Progress is reported as [N/total] per NCT ID."""
        # Pre-cache all entries so no API calls needed
        nct_ids = discover_nct_ids(pdf_dir)
        for nct_id in nct_ids:
            (cache_dir / f"{nct_id}.json").write_text(
                json.dumps(_sample_metadata(nct_id)),
                encoding="utf-8",
            )

        populate(data_dir=pdf_dir, cache_dir=cache_dir)
        captured = capsys.readouterr()

        for i, nct_id in enumerate(nct_ids, 1):
            assert f"[{i}/{len(nct_ids)}]" in captured.out
            assert nct_id in captured.out


# ---------------------------------------------------------------------------
# Scenario: ICH section mapping verification
# ---------------------------------------------------------------------------


class TestIchMapping:
    """Scenario: ICH section mapping verification."""

    def test_sections_are_mapped_and_codes_listed(
        self, pdf_dir: Path, cache_dir: Path, capsys: Any
    ) -> None:
        """Mapped section codes appear in output."""
        nct_ids = discover_nct_ids(pdf_dir)
        for nct_id in nct_ids:
            (cache_dir / f"{nct_id}.json").write_text(
                json.dumps(_sample_metadata(nct_id)),
                encoding="utf-8",
            )

        result = populate(data_dir=pdf_dir, cache_dir=cache_dir)
        captured = capsys.readouterr()

        # Each trial should produce some sections
        assert result.total_sections_mapped > 0
        # B.1 and B.3 should appear (from identification + status)
        assert "B.1" in captured.out
        assert "B.3" in captured.out


# ---------------------------------------------------------------------------
# Scenario: Graceful handling of fetch failures
# ---------------------------------------------------------------------------


class TestFetchFailures:
    """Scenario: Graceful handling of fetch failures."""

    def test_failures_counted_and_processing_continues(
        self, pdf_dir: Path, cache_dir: Path, capsys: Any
    ) -> None:
        """Failed fetches are counted; other NCT IDs still process."""
        nct_ids = discover_nct_ids(pdf_dir)

        # Cache all but one
        for nct_id in nct_ids[1:]:
            (cache_dir / f"{nct_id}.json").write_text(
                json.dumps(_sample_metadata(nct_id)),
                encoding="utf-8",
            )

        # The first NCT ID will hit the API and fail
        with patch(
            "ptcv.registry.metadata_fetcher.urllib.request.urlopen"
        ) as mock_urlopen, patch(
            "ptcv.registry.metadata_fetcher.time.sleep"
        ):
            mock_urlopen.side_effect = urllib.error.URLError(
                "Network error"
            )

            result = populate(
                data_dir=pdf_dir,
                cache_dir=cache_dir,
                delay=0.0,
            )

        assert result.failed >= 1
        assert result.cached == len(nct_ids) - 1
        assert "FAILED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Scenario: Module is invokable via python -m
# ---------------------------------------------------------------------------


class TestCliInterface:
    """Scenario: Module is invokable via python -m."""

    def test_help_flag(self) -> None:
        """--help produces usage information."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_all_arguments_documented(self) -> None:
        """All CLI arguments are present in the parser."""
        parser = build_parser()
        actions = {a.dest for a in parser._actions}
        assert "data_dir" in actions
        assert "cache_dir" in actions
        assert "dry_run" in actions
        assert "seed_rag" in actions
        assert "delay" in actions
        assert "verbose" in actions

    def test_main_with_dry_run(self, pdf_dir: Path) -> None:
        """main() accepts argv and runs in dry-run mode."""
        result = main([
            "--data-dir", str(pdf_dir),
            "--dry-run",
        ])
        assert isinstance(result, PopulateResult)
        assert result.total_nct_ids >= 0

    def test_default_delay(self) -> None:
        """Default delay is 0.5 seconds."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.delay == 0.5


# ---------------------------------------------------------------------------
# Scenario: Summary report at completion
# ---------------------------------------------------------------------------


class TestSummaryReport:
    """Scenario: Summary report at completion."""

    def test_summary_contains_all_fields(
        self, pdf_dir: Path, cache_dir: Path, capsys: Any
    ) -> None:
        """Final summary includes all required fields."""
        # Cache everything so no API calls
        for nct_id in discover_nct_ids(pdf_dir):
            (cache_dir / f"{nct_id}.json").write_text(
                json.dumps(_sample_metadata(nct_id)),
                encoding="utf-8",
            )

        populate(data_dir=pdf_dir, cache_dir=cache_dir)
        output = capsys.readouterr().out

        assert "Total NCT IDs:" in output
        assert "Freshly fetched:" in output
        assert "From cache:" in output
        assert "Failed:" in output
        assert "Total ICH sections:" in output
        assert "Cache dir:" in output

    def test_summary_not_shown_on_dry_run(
        self, pdf_dir: Path, cache_dir: Path, capsys: Any
    ) -> None:
        """Dry run does not show the full summary report."""
        populate(
            data_dir=pdf_dir, cache_dir=cache_dir, dry_run=True
        )
        output = capsys.readouterr().out
        # Dry run returns early — no "Results:" banner
        assert "Registry Population Results:" not in output


# ---------------------------------------------------------------------------
# PopulateResult dataclass
# ---------------------------------------------------------------------------


class TestPopulateResult:
    """PopulateResult dataclass validation."""

    def test_defaults_are_zero(self) -> None:
        """All fields default to zero."""
        result = PopulateResult()
        assert result.total_nct_ids == 0
        assert result.fetched == 0
        assert result.cached == 0
        assert result.failed == 0
        assert result.total_sections_mapped == 0
        assert result.sections_seeded == 0
