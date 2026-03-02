"""Tests for PTCV-27 download_sample script.

Qualification phase: OQ (operational qualification)
Regulatory requirement: ALCOA+ Audit trail, ALCOA+ Original (idempotent re-run),
  PTCV-27 Scenarios: Diverse sample, per-bucket download, idempotency, report.
Risk tier: MEDIUM
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ptcv.protocol_search.models import DownloadResult, SearchResult

# Import the script module under test
sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.download_sample import (
    BucketResult,
    SampleReport,
    _download_ct_gov_bucket,
    _download_eu_ctr_bucket,
    run_sample_download,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_results(count: int, source: str = "EU-CTR") -> list[SearchResult]:
    """Build a list of dummy SearchResult objects."""
    return [
        SearchResult(
            registry_id=f"2024-{i:06d}-10-00" if source == "EU-CTR" else f"NCT{i:08d}",
            title=f"Trial {i}",
            source=source,
            sponsor="Sponsor",
            phase="Phase 2",
            condition="oncology",
            status="Authorised",
            url=f"https://example.com/{i}",
        )
        for i in range(count)
    ]


def _success_download(registry_id: str) -> DownloadResult:
    return DownloadResult(
        success=True,
        registry_id=registry_id,
        file_path=f"/tmp/{registry_id}.pdf",
        metadata_path=f"/tmp/{registry_id}.json",
        file_hash_sha256="a" * 64,
    )


def _already_stored_download(registry_id: str) -> DownloadResult:
    return DownloadResult(
        success=False,
        registry_id=registry_id,
        error=f"Artifact already exists at /some/path. ALCOA+ Original: do not overwrite immutable artifact.",
    )


def _failed_download(registry_id: str) -> DownloadResult:
    return DownloadResult(
        success=False,
        registry_id=registry_id,
        error="HTTP 404 Not Found",
    )


# ---------------------------------------------------------------------------
# BucketResult
# ---------------------------------------------------------------------------

class TestBucketResult:
    """IQ: BucketResult dataclass."""

    def test_defaults_zero(self):
        """All counters default to zero."""
        b = BucketResult(condition="oncology", registry="EU-CTR")
        assert b.downloaded == 0
        assert b.skipped == 0
        assert b.failed == 0
        assert b.errors == []

    def test_search_found_set(self):
        """search_found can be set directly."""
        b = BucketResult(condition="cardiology", registry="ClinicalTrials.gov")
        b.search_found = 42
        assert b.search_found == 42


# ---------------------------------------------------------------------------
# SampleReport
# ---------------------------------------------------------------------------

class TestSampleReport:
    """IQ: SampleReport serialisation."""

    def test_to_dict_contains_required_fields(self):
        """to_dict() includes all required audit fields."""
        report = SampleReport(
            run_timestamp="2026-03-02T14:00:00Z",
            downloaded=80,
            skipped_already_stored=15,
            failed=5,
            total_attempted=100,
            therapeutic_areas=["oncology", "cardiovascular"],
            bucket_results=[],
        )
        d = report.to_dict()
        assert d["downloaded"] == 80
        assert d["skipped_already_stored"] == 15
        assert d["failed"] == 5
        assert d["total_attempted"] == 100
        assert d["therapeutic_areas"] == ["oncology", "cardiovascular"]
        assert "run_timestamp" in d

    def test_dry_run_flag_serialised(self):
        """dry_run=True is preserved in to_dict()."""
        report = SampleReport(
            run_timestamp="2026-03-02T14:00:00Z",
            downloaded=0,
            skipped_already_stored=0,
            failed=0,
            total_attempted=0,
            therapeutic_areas=[],
            bucket_results=[],
            dry_run=True,
        )
        assert report.to_dict()["dry_run"] is True


# ---------------------------------------------------------------------------
# _download_eu_ctr_bucket
# ---------------------------------------------------------------------------

class TestDownloadEuCtrBucket:
    """OQ: PTCV-27 Scenario: EU-CTR bucket downloaded per therapeutic area."""

    def test_downloads_up_to_target(self):
        """Downloads min(target, found) protocols."""
        mock_ctis = MagicMock()
        mock_ctis.search.return_value = _make_search_results(20)
        mock_ctis.download.side_effect = lambda euct_code, **kw: _success_download(euct_code)

        result = _download_eu_ctr_bucket(mock_ctis, "oncology", "Phase 2", target=5, dry_run=False)

        assert result.downloaded == 5
        assert result.skipped == 0
        assert result.failed == 0
        assert result.search_found == 20

    def test_counts_already_stored_as_skipped(self):
        """FileExistsError-flavoured errors count as skipped, not failed."""
        mock_ctis = MagicMock()
        mock_ctis.search.return_value = _make_search_results(3)
        mock_ctis.download.side_effect = lambda euct_code, **kw: _already_stored_download(euct_code)

        result = _download_eu_ctr_bucket(mock_ctis, "oncology", "Phase 2", target=3, dry_run=False)

        assert result.skipped == 3
        assert result.downloaded == 0
        assert result.failed == 0

    def test_counts_http_error_as_failed(self):
        """HTTP errors count as failed and are added to errors list."""
        mock_ctis = MagicMock()
        mock_ctis.search.return_value = _make_search_results(2)
        mock_ctis.download.side_effect = lambda euct_code, **kw: _failed_download(euct_code)

        result = _download_eu_ctr_bucket(mock_ctis, "oncology", "Phase 2", target=2, dry_run=False)

        assert result.failed == 2
        assert len(result.errors) == 2

    def test_empty_search_returns_zero_counts(self):
        """No search results produces a zero-count bucket."""
        mock_ctis = MagicMock()
        mock_ctis.search.return_value = []

        result = _download_eu_ctr_bucket(mock_ctis, "oncology", "Phase 2", target=5, dry_run=False)

        assert result.search_found == 0
        assert result.downloaded == 0
        mock_ctis.download.assert_not_called()

    def test_dry_run_does_not_call_download(self):
        """Dry-run mode counts planned downloads but never calls ctis.download()."""
        mock_ctis = MagicMock()
        mock_ctis.search.return_value = _make_search_results(5)

        result = _download_eu_ctr_bucket(mock_ctis, "oncology", "Phase 2", target=3, dry_run=True)

        mock_ctis.download.assert_not_called()
        assert result.downloaded == 3  # dry-run counts planned


# ---------------------------------------------------------------------------
# _download_ct_gov_bucket
# ---------------------------------------------------------------------------

class TestDownloadCtGovBucket:
    """OQ: PTCV-27 Scenario: ClinicalTrials.gov bucket downloaded per therapeutic area."""

    def test_downloads_up_to_target(self):
        """Downloads min(target, found) protocols."""
        mock_ct = MagicMock()
        mock_ct.search_pdf_available.return_value = [
            f"NCT{i:08d}" for i in range(20)
        ]
        mock_ct.download.side_effect = lambda nct_id, **kw: _success_download(nct_id)

        result = _download_ct_gov_bucket(mock_ct, "cardiovascular", "PHASE2", target=6, dry_run=False)

        assert result.downloaded == 6
        assert result.failed == 0

    def test_dry_run_does_not_call_download(self):
        """Dry-run mode never calls ct.download()."""
        mock_ct = MagicMock()
        mock_ct.search_pdf_available.return_value = [
            f"NCT{i:08d}" for i in range(10)
        ]

        result = _download_ct_gov_bucket(mock_ct, "neurology", "PHASE2", target=4, dry_run=True)

        mock_ct.download.assert_not_called()
        assert result.downloaded == 4


# ---------------------------------------------------------------------------
# run_sample_download
# ---------------------------------------------------------------------------

class TestRunSampleDownload:
    """OQ: PTCV-27 full pipeline scenarios."""

    def _patch_services(self, search_count: int = 20, all_success: bool = True):
        """Return a context manager that patches both services."""
        mock_ctis = MagicMock()
        mock_ct = MagicMock()

        mock_ctis.search.return_value = _make_search_results(search_count, "EU-CTR")
        mock_ct.search_pdf_available.return_value = [
            f"NCT{i:08d}" for i in range(search_count)
        ]

        if all_success:
            mock_ctis.download.side_effect = lambda euct_code, **kw: _success_download(euct_code)
            mock_ct.download.side_effect = lambda nct_id, **kw: _success_download(nct_id)
        else:
            mock_ctis.download.side_effect = lambda euct_code, **kw: _already_stored_download(euct_code)
            mock_ct.download.side_effect = lambda nct_id, **kw: _already_stored_download(nct_id)

        return mock_ctis, mock_ct

    def test_report_total_attempted_equals_downloaded_plus_skipped_plus_failed(self):
        """SampleReport total_attempted is the sum of all outcomes."""
        mock_ctis, mock_ct = self._patch_services(search_count=20)

        with patch("scripts.download_sample.CTISService", return_value=mock_ctis), \
             patch("scripts.download_sample.ClinicalTrialsService", return_value=mock_ct):
            report = run_sample_download(target_per_bucket=5, seed=42)

        assert report.total_attempted == report.downloaded + report.skipped_already_stored + report.failed

    def test_dry_run_downloaded_positive_no_actual_files(self):
        """PTCV-27 Scenario: Dry-run counts planned without calling download."""
        mock_ctis, mock_ct = self._patch_services(search_count=20)

        with patch("scripts.download_sample.CTISService", return_value=mock_ctis), \
             patch("scripts.download_sample.ClinicalTrialsService", return_value=mock_ct):
            report = run_sample_download(target_per_bucket=5, dry_run=True, seed=42)

        assert report.dry_run is True
        assert report.downloaded > 0
        mock_ctis.download.assert_not_called()
        mock_ct.download.assert_not_called()

    def test_idempotent_rerun_counts_as_skipped(self):
        """PTCV-27 Scenario: Download script is idempotent — re-run skips stored files."""
        mock_ctis, mock_ct = self._patch_services(search_count=20, all_success=False)

        with patch("scripts.download_sample.CTISService", return_value=mock_ctis), \
             patch("scripts.download_sample.ClinicalTrialsService", return_value=mock_ct):
            report = run_sample_download(target_per_bucket=5, seed=42)

        assert report.skipped_already_stored > 0
        assert report.downloaded == 0

    def test_report_includes_therapeutic_areas(self):
        """PTCV-27 Scenario: Report includes list of therapeutic areas sampled."""
        mock_ctis, mock_ct = self._patch_services(search_count=20)

        with patch("scripts.download_sample.CTISService", return_value=mock_ctis), \
             patch("scripts.download_sample.ClinicalTrialsService", return_value=mock_ct):
            report = run_sample_download(target_per_bucket=3, seed=42)

        assert len(report.therapeutic_areas) == 4
        assert "oncology" in report.therapeutic_areas
        assert "cardiovascular" in report.therapeutic_areas
        assert "nervous system diseases" in report.therapeutic_areas
        assert "diabetes" in report.therapeutic_areas

    def test_report_has_bucket_results_for_each_area_and_registry(self):
        """One bucket per (condition, registry) pair — 4 areas × 2 registries = 8 buckets."""
        mock_ctis, mock_ct = self._patch_services(search_count=10)

        with patch("scripts.download_sample.CTISService", return_value=mock_ctis), \
             patch("scripts.download_sample.ClinicalTrialsService", return_value=mock_ct):
            report = run_sample_download(target_per_bucket=3, seed=42)

        assert len(report.bucket_results) == 8  # 4 areas × 2 registries

    def test_no_single_area_dominates_sample(self):
        """PTCV-27 Scenario: No single area accounts for more than 40% of total."""
        mock_ctis, mock_ct = self._patch_services(search_count=20)

        with patch("scripts.download_sample.CTISService", return_value=mock_ctis), \
             patch("scripts.download_sample.ClinicalTrialsService", return_value=mock_ct):
            report = run_sample_download(target_per_bucket=5, seed=42)

        # Aggregate per condition across both registries
        by_condition: dict[str, int] = {}
        for bucket in report.bucket_results:
            cond = bucket["condition"]
            by_condition[cond] = by_condition.get(cond, 0) + bucket["downloaded"]

        total = sum(by_condition.values())
        if total > 0:
            for cond, count in by_condition.items():
                fraction = count / total
                assert fraction <= 0.40, (
                    f"{cond} accounts for {fraction:.0%} of sample (limit 40%)"
                )

    def test_seed_produces_reproducible_results(self):
        """Same seed → same download plan (deterministic sampling)."""
        mock_ctis, mock_ct = self._patch_services(search_count=20)

        with patch("scripts.download_sample.CTISService", return_value=mock_ctis), \
             patch("scripts.download_sample.ClinicalTrialsService", return_value=mock_ct):
            r1 = run_sample_download(target_per_bucket=5, seed=99)

        mock_ctis.reset_mock()
        mock_ct.reset_mock()
        mock_ctis.search.return_value = _make_search_results(20, "EU-CTR")
        mock_ct.search_pdf_available.return_value = [f"NCT{i:08d}" for i in range(20)]
        mock_ctis.download.side_effect = lambda euct_code, **kw: _success_download(euct_code)
        mock_ct.download.side_effect = lambda nct_id, **kw: _success_download(nct_id)

        with patch("scripts.download_sample.CTISService", return_value=mock_ctis), \
             patch("scripts.download_sample.ClinicalTrialsService", return_value=mock_ct):
            r2 = run_sample_download(target_per_bucket=5, seed=99)

        assert r1.downloaded == r2.downloaded
        assert r1.total_attempted == r2.total_attempted


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------

class TestWriteReport:
    """OQ: PTCV-27 Scenario: Download summary report generated."""

    def test_report_written_to_audit_dir(self, tmp_path, monkeypatch):
        """Report JSON is written to data/audit/sample_download_report.json."""
        import scripts.download_sample as ds_module

        report_path = tmp_path / "audit" / "sample_download_report.json"
        monkeypatch.setattr(ds_module, "_AUDIT_REPORT_PATH", report_path)

        report = SampleReport(
            run_timestamp="2026-03-02T14:00:00Z",
            downloaded=80,
            skipped_already_stored=15,
            failed=5,
            total_attempted=100,
            therapeutic_areas=["oncology", "cardiovascular", "neurology", "diabetes"],
            bucket_results=[],
        )
        ds_module.write_report(report)

        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["downloaded"] == 80
        assert data["therapeutic_areas"] == ["oncology", "cardiovascular", "neurology", "diabetes"]

    def test_report_parent_dir_created_if_missing(self, tmp_path, monkeypatch):
        """write_report creates data/audit/ if it does not exist."""
        import scripts.download_sample as ds_module

        report_path = tmp_path / "new" / "sub" / "report.json"
        monkeypatch.setattr(ds_module, "_AUDIT_REPORT_PATH", report_path)
        assert not report_path.parent.exists()

        report = SampleReport(
            run_timestamp="2026-03-02T14:00:00Z",
            downloaded=0,
            skipped_already_stored=0,
            failed=0,
            total_attempted=0,
            therapeutic_areas=[],
            bucket_results=[],
        )
        ds_module.write_report(report)

        assert report_path.exists()
