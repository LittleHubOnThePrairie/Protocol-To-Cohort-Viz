"""Tests for GdeltAdapter (PTCV-203).

Covers all GHERKIN scenarios:
- Fetch press releases by NCT ID keyword
- Fetch press releases by trial name keywords
- Handle rate limits and empty results gracefully
- Cache fetched articles per NCT ID
"""

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptcv.registry.gdelt_adapter import (
    GdeltAdapter,
    GdeltSearchResult,
    NewswireArticle,
    _DEFAULT_REQUEST_DELAY,
)


# ---------------------------------------------------------------------------
# Sample GDELT response data
# ---------------------------------------------------------------------------

_SAMPLE_GDELT_RESPONSE: dict = {
    "articles": [
        {
            "url": "https://www.businesswire.com/news/trial-results-2024",
            "title": "Phase 2 Trial NCT01512251 Shows Promising Results",
            "domain": "businesswire.com",
            "seendate": "20240315T120000Z",
            "tone": "-1.5,2.3,0.8",
            "themes": "HEALTH_PANDEMIC;MEDICAL;TAX_FNCACT",
        },
        {
            "url": "https://www.prnewswire.com/melanoma-trial-update",
            "title": "BKM120 Melanoma Combination Trial Update",
            "domain": "prnewswire.com",
            "seendate": "20240310T080000Z",
            "tone": "3.2",
            "themes": "MEDICAL",
        },
    ]
}

_EMPTY_GDELT_RESPONSE: dict = {"articles": []}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Return a temporary newswire cache directory."""
    d = tmp_path / "newswire"
    d.mkdir()
    return d


@pytest.fixture
def adapter(cache_dir: Path) -> GdeltAdapter:
    """Return a GdeltAdapter with temp cache and no-op audit logger."""
    audit = MagicMock()
    return GdeltAdapter(
        cache_dir=cache_dir,
        audit_logger=audit,
        timeout=5,
        request_delay=0.0,  # no delay in tests
    )


def _mock_response(data: dict) -> MagicMock:
    """Create a mock urllib response returning JSON data."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Scenario 1: Fetch press releases by NCT ID keyword
# ---------------------------------------------------------------------------


class TestSearchByNctId:
    """Scenario: Fetch press releases by NCT ID keyword."""

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_returns_articles_with_metadata(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Given a valid NCT ID, returns articles with URL, title,
        date, and tone score."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        result = adapter.search_by_nct_id("NCT01512251")

        assert isinstance(result, GdeltSearchResult)
        assert result.nct_id == "NCT01512251"
        assert len(result.articles) == 2
        assert result.total_found == 2

        article = result.articles[0]
        assert article.url == (
            "https://www.businesswire.com/news/trial-results-2024"
        )
        assert "NCT01512251" in article.title
        assert article.source_domain == "businesswire.com"
        assert article.published_date == "20240315T120000Z"
        assert article.tone == -1.5
        assert article.query_type == "nct_id"

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_parses_tone_correctly(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Tone string with commas is parsed to first float value."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        result = adapter.search_by_nct_id("NCT01512251")

        # First article has tone "-1.5,2.3,0.8" → -1.5
        assert result.articles[0].tone == -1.5
        # Second article has tone "3.2" → 3.2
        assert result.articles[1].tone == 3.2

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_parses_themes_correctly(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Theme string is split on semicolons."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        result = adapter.search_by_nct_id("NCT01512251")

        assert "HEALTH_PANDEMIC" in result.articles[0].themes
        assert "MEDICAL" in result.articles[0].themes
        assert result.articles[1].themes == ["MEDICAL"]

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_query_url_includes_nct_id(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """The API query URL contains the NCT ID."""
        mock_urlopen.return_value = _mock_response(
            _EMPTY_GDELT_RESPONSE
        )

        adapter.search_by_nct_id("NCT01512251")

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert "NCT01512251" in request.full_url

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_logs_audit_entry(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Successful search logs an audit trail entry."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        adapter.search_by_nct_id("NCT01512251")

        adapter._audit.log.assert_called_once()
        call_kwargs = adapter._audit.log.call_args
        assert call_kwargs.kwargs["record_id"] == "NCT01512251"
        assert call_kwargs.kwargs["after"]["articles_found"] == 2


# ---------------------------------------------------------------------------
# Scenario 2: Fetch press releases by trial name keywords
# ---------------------------------------------------------------------------


class TestSearchByKeywords:
    """Scenario: Fetch press releases by trial name keywords."""

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_keyword_search_returns_articles(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Given keywords, returns matching articles with keyword
        query_type."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        result = adapter.search_by_keywords(
            "NCT01512251",
            ["BKM120", "melanoma", "clinical trial"],
        )

        assert result.nct_id == "NCT01512251"
        assert len(result.articles) == 2
        for article in result.articles:
            assert article.query_type == "keyword"
        assert result.query_terms == [
            "BKM120", "melanoma", "clinical trial"
        ]

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_keyword_search_merges_with_cached_nct_results(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
        cache_dir: Path,
    ) -> None:
        """Keyword results merge with existing cached NCT ID results,
        deduplicating by URL."""
        # Pre-populate cache with NCT ID search result
        cached_data = {
            "nct_id": "NCT01512251",
            "articles": [
                {
                    "url": "https://www.businesswire.com/news/trial-results-2024",
                    "title": "Phase 2 Trial NCT01512251 Shows Promising Results",
                    "source_domain": "businesswire.com",
                    "published_date": "20240315T120000Z",
                    "tone": -1.5,
                    "themes": ["MEDICAL"],
                    "query_type": "nct_id",
                    "fetched_utc": "2024-03-15T12:00:00+00:00",
                }
            ],
            "query_terms": ["NCT01512251"],
            "total_found": 1,
            "fetched_utc": "2024-03-15T12:00:00+00:00",
        }
        cache_file = cache_dir / "NCT01512251_gdelt.json"
        cache_file.write_text(
            json.dumps(cached_data), encoding="utf-8"
        )

        # Keyword search returns one overlapping + one new article
        keyword_response = {
            "articles": [
                {
                    "url": "https://www.businesswire.com/news/trial-results-2024",
                    "title": "Phase 2 Trial NCT01512251 (duplicate)",
                    "domain": "businesswire.com",
                    "seendate": "20240315T120000Z",
                    "tone": "-1.5",
                    "themes": "MEDICAL",
                },
                {
                    "url": "https://www.reuters.com/bkm120-update",
                    "title": "New BKM120 Data Presented at ASCO",
                    "domain": "reuters.com",
                    "seendate": "20240312T090000Z",
                    "tone": "2.0",
                    "themes": "MEDICAL;HEALTH_PANDEMIC",
                },
            ]
        }
        mock_urlopen.return_value = _mock_response(keyword_response)

        result = adapter.search_by_keywords(
            "NCT01512251", ["BKM120", "melanoma"]
        )

        # Should have 2 unique articles (deduplicated by URL)
        assert len(result.articles) == 2
        urls = {a.url for a in result.articles}
        assert (
            "https://www.businesswire.com/news/trial-results-2024"
            in urls
        )
        assert "https://www.reuters.com/bkm120-update" in urls

    def test_empty_keywords_returns_empty_result(
        self, adapter: GdeltAdapter
    ) -> None:
        """Empty keywords list returns empty result without API call."""
        result = adapter.search_by_keywords("NCT01512251", [])

        assert result.nct_id == "NCT01512251"
        assert len(result.articles) == 0


# ---------------------------------------------------------------------------
# Scenario 3: Handle rate limits and empty results gracefully
# ---------------------------------------------------------------------------


class TestGracefulErrorHandling:
    """Scenario: Handle rate limits and empty results gracefully."""

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_empty_response_returns_empty_result(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Given no matching articles, returns empty result without
        error."""
        mock_urlopen.return_value = _mock_response(
            _EMPTY_GDELT_RESPONSE
        )

        result = adapter.search_by_nct_id("NCT01512251")

        assert result.nct_id == "NCT01512251"
        assert len(result.articles) == 0
        assert result.total_found == 0

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_network_error_returns_empty_result(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Network errors return empty result without raising."""
        mock_urlopen.side_effect = urllib.error.URLError(
            "Connection refused"
        )

        result = adapter.search_by_nct_id("NCT01512251")

        assert result.nct_id == "NCT01512251"
        assert len(result.articles) == 0

    @patch("ptcv.registry.gdelt_adapter.time.sleep")
    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_retries_on_429_then_returns_empty(
        self,
        mock_urlopen: MagicMock,
        mock_sleep: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Rate limit (429) retries with backoff, then returns empty
        on exhaustion."""
        error_429 = urllib.error.HTTPError(
            url="", code=429, msg="Too Many Requests",
            hdrs=None, fp=None,  # type: ignore[arg-type]
        )
        mock_urlopen.side_effect = [error_429, error_429, error_429]

        result = adapter.search_by_nct_id("NCT01512251")

        assert len(result.articles) == 0
        assert mock_sleep.call_count >= 2

    @patch("ptcv.registry.gdelt_adapter.time.sleep")
    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_retries_on_503_then_succeeds(
        self,
        mock_urlopen: MagicMock,
        mock_sleep: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Service unavailable (503) retries, then succeeds on third
        attempt."""
        error_503 = urllib.error.HTTPError(
            url="", code=503, msg="Service Unavailable",
            hdrs=None, fp=None,  # type: ignore[arg-type]
        )
        mock_urlopen.side_effect = [
            error_503,
            error_503,
            _mock_response(_SAMPLE_GDELT_RESPONSE),
        ]

        result = adapter.search_by_nct_id("NCT01512251")

        assert len(result.articles) == 2
        assert mock_sleep.call_count == 2

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_http_400_returns_empty_result(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """Non-retryable HTTP errors (400) return empty result."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=400, msg="Bad Request",
            hdrs=None, fp=None,  # type: ignore[arg-type]
        )

        result = adapter.search_by_nct_id("NCT01512251")

        assert len(result.articles) == 0

    @patch("ptcv.registry.gdelt_adapter.time.monotonic")
    @patch("ptcv.registry.gdelt_adapter.time.sleep")
    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_rate_limiting_between_requests(
        self,
        mock_urlopen: MagicMock,
        mock_sleep: MagicMock,
        mock_monotonic: MagicMock,
        cache_dir: Path,
    ) -> None:
        """Rate limiter enforces minimum delay between requests."""
        adapter = GdeltAdapter(
            cache_dir=cache_dir,
            audit_logger=MagicMock(),
            request_delay=1.0,
        )
        mock_urlopen.return_value = _mock_response(
            _EMPTY_GDELT_RESPONSE
        )
        # Simulate time: first call at t=0, second at t=0.3
        mock_monotonic.side_effect = [0.0, 0.0, 0.3, 1.0]

        adapter.search_by_nct_id("NCT01512251")
        adapter.search_by_nct_id("NCT02000000")

        # Should sleep for the remaining 0.7s
        assert any(
            abs(call[0][0] - 0.7) < 0.01
            for call in mock_sleep.call_args_list
        )


# ---------------------------------------------------------------------------
# Scenario 4: Cache fetched articles per NCT ID
# ---------------------------------------------------------------------------


class TestCaching:
    """Scenario: Cache fetched articles per NCT ID."""

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_results_cached_to_disk(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
        cache_dir: Path,
    ) -> None:
        """Fetched results are cached to newswire subdirectory."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        adapter.search_by_nct_id("NCT01512251")

        cache_file = cache_dir / "NCT01512251_gdelt.json"
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        assert cached["nct_id"] == "NCT01512251"
        assert len(cached["articles"]) == 2

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_cache_hit_skips_api_call(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
        cache_dir: Path,
    ) -> None:
        """Second search for same NCT ID reads from cache."""
        cached_data = {
            "nct_id": "NCT01512251",
            "articles": [
                {
                    "url": "https://example.com/cached-article",
                    "title": "Cached Article",
                    "source_domain": "example.com",
                    "published_date": "20240301T000000Z",
                    "tone": 0.0,
                    "themes": [],
                    "query_type": "nct_id",
                    "fetched_utc": "2024-03-01T00:00:00+00:00",
                }
            ],
            "query_terms": ["NCT01512251"],
            "total_found": 1,
            "fetched_utc": "2024-03-01T00:00:00+00:00",
        }
        cache_file = cache_dir / "NCT01512251_gdelt.json"
        cache_file.write_text(
            json.dumps(cached_data), encoding="utf-8"
        )

        result = adapter.search_by_nct_id("NCT01512251")

        assert result.from_cache is True
        assert len(result.articles) == 1
        assert result.articles[0].url == (
            "https://example.com/cached-article"
        )
        mock_urlopen.assert_not_called()

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_skip_cache_forces_fresh_query(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
        cache_dir: Path,
    ) -> None:
        """skip_cache=True bypasses cache and queries API."""
        # Pre-populate cache
        cached_data = {
            "nct_id": "NCT01512251",
            "articles": [],
            "query_terms": ["NCT01512251"],
            "total_found": 0,
            "fetched_utc": "2024-03-01T00:00:00+00:00",
        }
        cache_file = cache_dir / "NCT01512251_gdelt.json"
        cache_file.write_text(
            json.dumps(cached_data), encoding="utf-8"
        )

        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        result = adapter.search_by_nct_id(
            "NCT01512251", skip_cache=True
        )

        assert result.from_cache is False
        assert len(result.articles) == 2
        mock_urlopen.assert_called_once()

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_corrupt_cache_triggers_refetch(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
        cache_dir: Path,
    ) -> None:
        """Corrupt cache file causes a fresh API call."""
        cache_file = cache_dir / "NCT01512251_gdelt.json"
        cache_file.write_text("{invalid json", encoding="utf-8")

        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        result = adapter.search_by_nct_id("NCT01512251")

        assert len(result.articles) == 2
        mock_urlopen.assert_called_once()

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_cache_includes_alcoa_provenance(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
        cache_dir: Path,
    ) -> None:
        """Cached results include ALCOA+ provenance metadata."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        adapter.search_by_nct_id("NCT01512251")

        cache_file = cache_dir / "NCT01512251_gdelt.json"
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "fetched_utc" in cached
        assert cached["fetched_utc"]  # Non-empty timestamp
        assert "query_terms" in cached
        for article in cached["articles"]:
            assert "fetched_utc" in article


# ---------------------------------------------------------------------------
# Batch search
# ---------------------------------------------------------------------------


class TestBatchSearch:
    """Batch search for multiple NCT IDs."""

    @patch("ptcv.registry.gdelt_adapter.urllib.request.urlopen")
    def test_batch_search_returns_dict(
        self,
        mock_urlopen: MagicMock,
        adapter: GdeltAdapter,
    ) -> None:
        """batch_search returns a dict mapping NCT ID to results."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_GDELT_RESPONSE
        )

        ids = ["NCT01512251", "NCT02000000"]
        results = adapter.batch_search(ids)

        assert len(results) == 2
        for nct_id in ids:
            assert nct_id in results
            assert isinstance(results[nct_id], GdeltSearchResult)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestDataModels:
    """NewswireArticle and GdeltSearchResult dataclass tests."""

    def test_newswire_article_auto_sets_fetched_utc(self) -> None:
        """NewswireArticle auto-populates fetched_utc if not set."""
        article = NewswireArticle(
            url="https://example.com",
            title="Test",
            source_domain="example.com",
            published_date="20240315T120000Z",
            tone=0.0,
        )
        assert article.fetched_utc  # Non-empty

    def test_newswire_article_preserves_explicit_fetched_utc(
        self,
    ) -> None:
        """NewswireArticle keeps explicitly provided fetched_utc."""
        article = NewswireArticle(
            url="https://example.com",
            title="Test",
            source_domain="example.com",
            published_date="20240315T120000Z",
            tone=0.0,
            fetched_utc="2024-01-01T00:00:00+00:00",
        )
        assert article.fetched_utc == "2024-01-01T00:00:00+00:00"

    def test_gdelt_search_result_auto_sets_fetched_utc(self) -> None:
        """GdeltSearchResult auto-populates fetched_utc if not set."""
        result = GdeltSearchResult(nct_id="NCT01512251")
        assert result.fetched_utc  # Non-empty

    def test_merge_results_deduplicates_by_url(self) -> None:
        """_merge_results removes duplicate articles by URL."""
        existing = GdeltSearchResult(
            nct_id="NCT01512251",
            articles=[
                NewswireArticle(
                    url="https://a.com/1",
                    title="A",
                    source_domain="a.com",
                    published_date="20240315",
                    tone=0.0,
                ),
                NewswireArticle(
                    url="https://b.com/2",
                    title="B",
                    source_domain="b.com",
                    published_date="20240314",
                    tone=1.0,
                ),
            ],
            query_terms=["NCT01512251"],
        )
        new = GdeltSearchResult(
            nct_id="NCT01512251",
            articles=[
                NewswireArticle(
                    url="https://a.com/1",  # duplicate
                    title="A (dup)",
                    source_domain="a.com",
                    published_date="20240315",
                    tone=0.0,
                ),
                NewswireArticle(
                    url="https://c.com/3",
                    title="C",
                    source_domain="c.com",
                    published_date="20240313",
                    tone=2.0,
                ),
            ],
            query_terms=["BKM120"],
        )

        merged = GdeltAdapter._merge_results(existing, new)

        assert len(merged.articles) == 3
        urls = [a.url for a in merged.articles]
        assert urls == [
            "https://a.com/1",
            "https://b.com/2",
            "https://c.com/3",
        ]
        # Query terms merged without duplicates
        assert merged.query_terms == ["NCT01512251", "BKM120"]
