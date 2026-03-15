"""Tests for RegistryMetadataFetcher (PTCV-194).

Covers all GHERKIN scenarios:
- Fetch full protocol metadata by NCT ID
- Handle missing or withdrawn trials
- Rate limiting and retry
- Batch fetch for multiple NCT IDs
"""

import json
import urllib.error
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from ptcv.registry.metadata_fetcher import (
    RegistryMetadataFetcher,
    _MAX_RETRIES,
    _RETRYABLE_STATUS_CODES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_STUDY_JSON: dict[str, Any] = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT01512251",
            "officialTitle": "A Phase 1/2 Trial of BKM120",
            "briefTitle": "BKM120 + Vemurafenib in Melanoma",
        },
        "statusModule": {
            "overallStatus": "Completed",
            "startDateStruct": {"date": "2012-02"},
        },
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE1", "PHASE2"],
        },
        "armsInterventionsModule": {
            "interventions": [
                {
                    "name": "BKM120",
                    "type": "DRUG",
                    "description": "Pan-class I PI3K inhibitor",
                },
                {
                    "name": "Vemurafenib",
                    "type": "DRUG",
                    "description": "BRAF V600E inhibitor",
                },
            ],
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {
                    "measure": "Maximum Tolerated Dose",
                    "timeFrame": "Cycle 1 (28 days)",
                }
            ],
        },
        "eligibilityModule": {
            "eligibilityCriteria": "BRAF V600E/K mutation",
            "sex": "ALL",
            "minimumAge": "18 Years",
        },
    },
}


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Return a temporary cache directory."""
    d = tmp_path / "registry_cache"
    d.mkdir()
    return d


@pytest.fixture
def fetcher(cache_dir: Path) -> RegistryMetadataFetcher:
    """Return a fetcher with temp cache and no-op audit logger."""
    audit = MagicMock()
    return RegistryMetadataFetcher(
        cache_dir=cache_dir,
        audit_logger=audit,
        timeout=5,
        batch_delay=0.0,  # no delay in tests
    )


# ---------------------------------------------------------------------------
# Scenario: Fetch full protocol metadata by NCT ID
# ---------------------------------------------------------------------------


class TestFetchMetadata:
    """Scenario: Fetch full protocol metadata by NCT ID."""

    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_fetch_returns_protocol_section(
        self,
        mock_urlopen: MagicMock,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Given a valid NCT ID, returns structured JSON with
        ProtocolSection fields."""
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _SAMPLE_STUDY_JSON
        ).encode("utf-8")
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = fetcher.fetch("NCT01512251")

        assert result is not None
        proto = result["protocolSection"]
        assert "identificationModule" in proto
        assert "statusModule" in proto
        assert "designModule" in proto
        assert "armsInterventionsModule" in proto
        assert "outcomesModule" in proto
        assert "eligibilityModule" in proto

    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_fetch_caches_response(
        self,
        mock_urlopen: MagicMock,
        fetcher: RegistryMetadataFetcher,
        cache_dir: Path,
    ) -> None:
        """Fetched metadata is cached to disk."""
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _SAMPLE_STUDY_JSON
        ).encode("utf-8")
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        fetcher.fetch("NCT01512251")

        cache_file = cache_dir / "NCT01512251.json"
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        assert cached["protocolSection"]["identificationModule"]["nctId"] == "NCT01512251"

    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_fetch_uses_cache_on_second_call(
        self,
        mock_urlopen: MagicMock,
        fetcher: RegistryMetadataFetcher,
        cache_dir: Path,
    ) -> None:
        """Second fetch for same NCT ID reads from cache, no API call."""
        # Pre-populate cache
        cache_file = cache_dir / "NCT01512251.json"
        cache_file.write_text(
            json.dumps(_SAMPLE_STUDY_JSON), encoding="utf-8"
        )

        result = fetcher.fetch("NCT01512251")

        assert result is not None
        assert result["protocolSection"]["identificationModule"]["nctId"] == "NCT01512251"
        mock_urlopen.assert_not_called()

    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_fetch_logs_audit_entry(
        self,
        mock_urlopen: MagicMock,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Successful fetch logs an audit trail entry."""
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _SAMPLE_STUDY_JSON
        ).encode("utf-8")
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        fetcher.fetch("NCT01512251")

        fetcher._audit.log.assert_called_once()  # type: ignore[union-attr]
        call_kwargs = fetcher._audit.log.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["record_id"] == "NCT01512251"


# ---------------------------------------------------------------------------
# Scenario: Handle missing or withdrawn trials
# ---------------------------------------------------------------------------


class TestMissingTrials:
    """Scenario: Handle missing or withdrawn trials."""

    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_fetch_returns_none_for_404(
        self,
        mock_urlopen: MagicMock,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Given a non-existent NCT ID, returns None."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://clinicaltrials.gov/api/v2/studies/NCT99999999",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

        result = fetcher.fetch("NCT99999999")

        assert result is None

    def test_fetch_returns_none_for_invalid_format(
        self,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Given an invalid NCT ID format, returns None without API call."""
        result = fetcher.fetch("INVALID123")
        assert result is None

    def test_fetch_returns_none_for_empty_string(
        self,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Empty string returns None."""
        result = fetcher.fetch("")
        assert result is None


# ---------------------------------------------------------------------------
# Scenario: Rate limiting and retry
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    """Scenario: Rate limiting and retry."""

    @patch("ptcv.registry.metadata_fetcher.time.sleep")
    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_retries_on_429(
        self,
        mock_urlopen: MagicMock,
        mock_sleep: MagicMock,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Retries with backoff on HTTP 429, succeeds on third attempt."""
        error_429 = urllib.error.HTTPError(
            url="", code=429, msg="Too Many Requests",
            hdrs=None, fp=None,  # type: ignore[arg-type]
        )
        resp_ok = MagicMock()
        resp_ok.read.return_value = json.dumps(
            _SAMPLE_STUDY_JSON
        ).encode("utf-8")
        resp_ok.__enter__ = lambda s: s
        resp_ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [error_429, error_429, resp_ok]

        result = fetcher.fetch("NCT01512251")

        assert result is not None
        assert mock_sleep.call_count == 2
        # Exponential backoff: 1s, 2s
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0

    @patch("ptcv.registry.metadata_fetcher.time.sleep")
    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_retries_on_503(
        self,
        mock_urlopen: MagicMock,
        mock_sleep: MagicMock,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Retries on HTTP 503 Service Unavailable."""
        error_503 = urllib.error.HTTPError(
            url="", code=503, msg="Service Unavailable",
            hdrs=None, fp=None,  # type: ignore[arg-type]
        )
        mock_urlopen.side_effect = [error_503] * _MAX_RETRIES

        result = fetcher.fetch("NCT01512251")

        assert result is None
        assert mock_sleep.call_count == _MAX_RETRIES

    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_no_retry_on_400(
        self,
        mock_urlopen: MagicMock,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Non-retryable errors (400) propagate immediately."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=400, msg="Bad Request",
            hdrs=None, fp=None,  # type: ignore[arg-type]
        )

        result = fetcher.fetch("NCT01512251")

        # 400 is not 404, so it raises through _get_json but is caught
        # by fetch()'s except block — returns None
        assert result is None


# ---------------------------------------------------------------------------
# Scenario: Batch fetch for multiple NCT IDs
# ---------------------------------------------------------------------------


class TestBatchFetch:
    """Scenario: Batch fetch for multiple NCT IDs."""

    @patch("ptcv.registry.metadata_fetcher.time.sleep")
    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_batch_fetch_returns_dict(
        self,
        mock_urlopen: MagicMock,
        mock_sleep: MagicMock,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """batch_fetch returns a dict mapping NCT ID to metadata."""
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _SAMPLE_STUDY_JSON
        ).encode("utf-8")
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        ids = ["NCT01512251", "NCT02000000", "NCT03000000"]
        results = fetcher.batch_fetch(ids)

        assert len(results) == 3
        for nct_id in ids:
            assert nct_id in results
            assert results[nct_id] is not None

    @patch("ptcv.registry.metadata_fetcher.time.sleep")
    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_batch_fetch_handles_partial_failures(
        self,
        mock_urlopen: MagicMock,
        mock_sleep: MagicMock,
        fetcher: RegistryMetadataFetcher,
    ) -> None:
        """Batch fetch returns None for failed IDs, data for others."""
        resp_ok = MagicMock()
        resp_ok.read.return_value = json.dumps(
            _SAMPLE_STUDY_JSON
        ).encode("utf-8")
        resp_ok.__enter__ = lambda s: s
        resp_ok.__exit__ = MagicMock(return_value=False)

        error_404 = urllib.error.HTTPError(
            url="", code=404, msg="Not Found",
            hdrs=None, fp=None,  # type: ignore[arg-type]
        )

        mock_urlopen.side_effect = [resp_ok, error_404, resp_ok]

        ids = ["NCT01512251", "NCT99999999", "NCT03000000"]
        results = fetcher.batch_fetch(ids)

        assert results["NCT01512251"] is not None
        assert results["NCT99999999"] is None
        assert results["NCT03000000"] is not None

    @patch("ptcv.registry.metadata_fetcher.time.sleep")
    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_batch_fetch_respects_delay(
        self,
        mock_urlopen: MagicMock,
        mock_sleep: MagicMock,
        cache_dir: Path,
    ) -> None:
        """Batch fetch sleeps between requests."""
        fetcher = RegistryMetadataFetcher(
            cache_dir=cache_dir,
            audit_logger=MagicMock(),
            batch_delay=0.5,
        )
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _SAMPLE_STUDY_JSON
        ).encode("utf-8")
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        fetcher.batch_fetch(["NCT01512251", "NCT02000000", "NCT03000000"])

        # Sleep called between requests (not after last one)
        assert mock_sleep.call_count == 2
        for call in mock_sleep.call_args_list:
            assert call[0][0] == 0.5


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestNctIdValidation:
    """NCT ID format validation and extraction."""

    @pytest.mark.parametrize(
        "nct_id,expected",
        [
            ("NCT01512251", True),
            ("NCT00000000", True),
            ("NCT99999999", True),
            ("nct01512251", False),  # lowercase
            ("NCT0151225", False),  # too short
            ("NCT015122510", False),  # too long
            ("NCTABCDEFGH", False),  # non-digits
            ("", False),
            ("EUCTR2020-001234-56", False),
        ],
    )
    def test_is_valid_nct_id(
        self, nct_id: str, expected: bool
    ) -> None:
        assert RegistryMetadataFetcher.is_valid_nct_id(nct_id) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("NCT01512251", "NCT01512251"),
            ("Protocol NCT01512251 v2.0", "NCT01512251"),
            ("NCT01512251_schedule.pdf", "NCT01512251"),
            ("no nct here", None),
            ("", None),
        ],
    )
    def test_extract_nct_id(
        self, text: str, expected: Optional[str]
    ) -> None:
        assert RegistryMetadataFetcher.extract_nct_id(text) == expected


class TestCorruptCache:
    """Cache corruption handling."""

    @patch("ptcv.registry.metadata_fetcher.urllib.request.urlopen")
    def test_corrupt_cache_triggers_refetch(
        self,
        mock_urlopen: MagicMock,
        fetcher: RegistryMetadataFetcher,
        cache_dir: Path,
    ) -> None:
        """Corrupt cache file causes a fresh API call."""
        cache_file = cache_dir / "NCT01512251.json"
        cache_file.write_text("{invalid json", encoding="utf-8")

        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _SAMPLE_STUDY_JSON
        ).encode("utf-8")
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = fetcher.fetch("NCT01512251")

        assert result is not None
        mock_urlopen.assert_called_once()
