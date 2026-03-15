"""Tests for NewsdataAdapter (PTCV-204).

Covers all GHERKIN scenarios:
- Search news articles by NCT ID or trial keywords
- Respect free tier rate limits (200 req/day)
- Filter results to clinical trial relevance
- Deduplicate against GDELT results
"""

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptcv.registry.gdelt_adapter import NewswireArticle
from ptcv.registry.newsdata_adapter import (
    NewsdataAdapter,
    NewsdataSearchResult,
    _DEFAULT_DAILY_LIMIT,
    _RELEVANCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Sample NewsData.io response data
# ---------------------------------------------------------------------------

_SAMPLE_NEWSDATA_RESPONSE: dict = {
    "status": "success",
    "totalResults": 3,
    "results": [
        {
            "title": "Phase 2 Clinical Trial NCT01512251 Shows Efficacy",
            "link": "https://www.reuters.com/trial-results-2024",
            "description": (
                "A randomized double-blind phase 2 clinical trial "
                "of BKM120 for melanoma showed primary endpoint "
                "efficacy results."
            ),
            "source_id": "reuters",
            "pubDate": "2024-03-15 12:00:00",
            "category": ["health", "science"],
            "language": "english",
        },
        {
            "title": "FDA Approves New Cancer Treatment",
            "link": "https://www.fda.gov/news/approval-2024",
            "description": (
                "The FDA has granted approval for a new pivotal "
                "therapy following successful clinical trial results."
            ),
            "source_id": "fda_gov",
            "pubDate": "2024-03-14 08:00:00",
            "category": ["health"],
            "language": "english",
        },
        {
            "title": "Local Weather Report for Tuesday",
            "link": "https://www.weather.com/tuesday",
            "description": "Sunny skies expected across the midwest.",
            "source_id": "weather_com",
            "pubDate": "2024-03-13 06:00:00",
            "category": ["top"],
            "language": "english",
        },
    ],
}

_EMPTY_NEWSDATA_RESPONSE: dict = {
    "status": "success",
    "totalResults": 0,
    "results": [],
}


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
def adapter(cache_dir: Path) -> NewsdataAdapter:
    """Return a NewsdataAdapter with temp cache and no-op audit."""
    audit = MagicMock()
    return NewsdataAdapter(
        api_key="test_key_123",
        cache_dir=cache_dir,
        audit_logger=audit,
        timeout=5,
    )


def _mock_response(data: dict) -> MagicMock:
    """Create a mock urllib response returning JSON data."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Scenario 1: Search news articles by NCT ID or trial keywords
# ---------------------------------------------------------------------------


class TestSearchArticles:
    """Scenario: Search news articles by NCT ID or trial keywords."""

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_returns_articles_with_metadata(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
    ) -> None:
        """Given keywords, returns articles with title, description,
        source, published date, and category metadata."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_NEWSDATA_RESPONSE
        )

        result = adapter.search(
            "NCT01512251", ["BKM120", "melanoma"]
        )

        assert isinstance(result, NewsdataSearchResult)
        assert result.nct_id == "NCT01512251"
        # Weather article filtered out by relevance
        assert result.relevant_count <= result.total_found

        # Check first relevant article has all fields
        article = result.articles[0]
        assert article.url
        assert article.title
        assert article.source_domain
        assert article.published_date
        assert isinstance(article.themes, list)
        assert article.query_type == "newsdata"

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_query_url_includes_api_key(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
    ) -> None:
        """The API query URL contains the API key."""
        mock_urlopen.return_value = _mock_response(
            _EMPTY_NEWSDATA_RESPONSE
        )

        adapter.search("NCT01512251", [])

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert "test_key_123" in request.full_url

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_query_url_includes_category_filter(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
    ) -> None:
        """The API query URL includes health category filter."""
        mock_urlopen.return_value = _mock_response(
            _EMPTY_NEWSDATA_RESPONSE
        )

        adapter.search("NCT01512251", [])

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert "category=health" in request.full_url

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_query_terms_include_nct_id_and_keywords(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
    ) -> None:
        """Query terms include both NCT ID and provided keywords."""
        mock_urlopen.return_value = _mock_response(
            _EMPTY_NEWSDATA_RESPONSE
        )

        result = adapter.search(
            "NCT01512251", ["BKM120", "melanoma"]
        )

        assert result.query_terms == [
            "NCT01512251", "BKM120", "melanoma"
        ]

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_logs_audit_entry(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
    ) -> None:
        """Successful search logs an audit trail entry."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_NEWSDATA_RESPONSE
        )

        adapter.search("NCT01512251", [])

        adapter._audit.log.assert_called_once()
        call_kwargs = adapter._audit.log.call_args
        assert call_kwargs.kwargs["record_id"] == "NCT01512251"
        assert "remaining_quota" in call_kwargs.kwargs["after"]

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_network_error_returns_empty(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
    ) -> None:
        """Network errors return empty result without raising."""
        mock_urlopen.side_effect = urllib.error.URLError(
            "Connection refused"
        )

        result = adapter.search("NCT01512251", [])

        assert result.nct_id == "NCT01512251"
        assert len(result.articles) == 0


# ---------------------------------------------------------------------------
# Scenario 2: Respect free tier rate limits
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Scenario: Respect free tier rate limits (200 req/day)."""

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_tracks_daily_request_count(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
    ) -> None:
        """Adapter tracks requests made today."""
        mock_urlopen.return_value = _mock_response(
            _EMPTY_NEWSDATA_RESPONSE
        )

        adapter.search("NCT01512251", [])
        adapter.search("NCT02000000", [])

        assert adapter._request_count == 2
        assert adapter.remaining_quota == _DEFAULT_DAILY_LIMIT - 2

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_blocks_when_limit_reached(
        self,
        mock_urlopen: MagicMock,
        cache_dir: Path,
    ) -> None:
        """Returns empty result when daily limit is reached."""
        adapter = NewsdataAdapter(
            api_key="test_key",
            cache_dir=cache_dir,
            audit_logger=MagicMock(),
            daily_limit=2,
        )
        mock_urlopen.return_value = _mock_response(
            _EMPTY_NEWSDATA_RESPONSE
        )

        adapter.search("NCT01512251", [])
        adapter.search("NCT02000000", [])
        # Third request should be blocked
        result = adapter.search("NCT03000000", [])

        assert len(result.articles) == 0
        # Only 2 API calls made
        assert mock_urlopen.call_count == 2

    def test_remaining_quota_resets_on_new_day(
        self,
        cache_dir: Path,
    ) -> None:
        """Quota resets when the date changes."""
        adapter = NewsdataAdapter(
            api_key="test_key",
            cache_dir=cache_dir,
            audit_logger=MagicMock(),
            daily_limit=200,
        )
        adapter._request_count = 150
        adapter._request_date = "2024-01-01"  # Yesterday

        # remaining_quota checks today's date, which differs
        assert adapter.remaining_quota == 200

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_daily_limit_resets_on_new_day(
        self,
        mock_urlopen: MagicMock,
        cache_dir: Path,
    ) -> None:
        """Request count resets when _check_daily_limit detects
        a new day."""
        adapter = NewsdataAdapter(
            api_key="test_key",
            cache_dir=cache_dir,
            audit_logger=MagicMock(),
            daily_limit=200,
        )
        # Simulate yesterday's exhausted quota
        adapter._request_count = 200
        adapter._request_date = "2024-01-01"

        mock_urlopen.return_value = _mock_response(
            _EMPTY_NEWSDATA_RESPONSE
        )

        # Today is different, so limit should reset
        result = adapter.search("NCT01512251", [])

        assert adapter._request_count == 1
        mock_urlopen.assert_called_once()


# ---------------------------------------------------------------------------
# Scenario 3: Filter results to clinical trial relevance
# ---------------------------------------------------------------------------


class TestRelevanceFiltering:
    """Scenario: Filter results to clinical trial relevance."""

    def test_high_relevance_article(self) -> None:
        """Article with many trial keywords scores high."""
        score = NewsdataAdapter.score_relevance(
            title="Phase 2 Clinical Trial Results",
            description=(
                "A randomized double-blind placebo-controlled "
                "clinical trial showed primary outcome efficacy."
            ),
        )
        assert score >= 0.6

    def test_low_relevance_article(self) -> None:
        """Article with no trial keywords scores zero."""
        score = NewsdataAdapter.score_relevance(
            title="Local Weather Report",
            description="Sunny skies expected across the midwest.",
        )
        assert score == 0.0

    def test_moderate_relevance_article(self) -> None:
        """Article with some trial keywords scores moderate."""
        score = NewsdataAdapter.score_relevance(
            title="FDA Approves New Treatment",
            description="Regulatory approval following trial.",
        )
        assert 0.0 < score < 1.0

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_irrelevant_articles_filtered_out(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
    ) -> None:
        """Articles below relevance threshold are excluded."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_NEWSDATA_RESPONSE
        )

        result = adapter.search("NCT01512251", [])

        # Weather article (score=0.0) should be filtered out
        assert result.total_found == 3
        assert result.relevant_count < result.total_found
        titles = [a.title for a in result.articles]
        assert "Local Weather Report for Tuesday" not in titles

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_custom_relevance_threshold(
        self,
        mock_urlopen: MagicMock,
        cache_dir: Path,
    ) -> None:
        """Custom threshold changes which articles pass filter."""
        # Set very high threshold — only very relevant articles pass
        adapter = NewsdataAdapter(
            api_key="test_key",
            cache_dir=cache_dir,
            audit_logger=MagicMock(),
            relevance_threshold=0.8,
        )
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_NEWSDATA_RESPONSE
        )

        result = adapter.search("NCT01512251", [])

        # Higher threshold should filter more aggressively
        assert result.relevant_count <= 3

    def test_relevance_score_capped_at_one(self) -> None:
        """Relevance score never exceeds 1.0."""
        score = NewsdataAdapter.score_relevance(
            title="clinical trial phase 2 randomized placebo",
            description=(
                "double-blind fda approval endpoint efficacy "
                "safety enrollment primary outcome pivotal "
                "regulatory nct clinicaltrials.gov"
            ),
        )
        assert score <= 1.0


# ---------------------------------------------------------------------------
# Scenario 4: Deduplicate against GDELT results
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Scenario: Deduplicate against GDELT results."""

    def test_removes_url_duplicates(self) -> None:
        """Articles with same URL as GDELT are removed."""
        newsdata_articles = [
            NewswireArticle(
                url="https://reuters.com/shared-article",
                title="Shared Article",
                source_domain="reuters.com",
                published_date="20240315",
                tone=0.5,
                query_type="newsdata",
            ),
            NewswireArticle(
                url="https://unique.com/only-newsdata",
                title="Unique NewsData Article",
                source_domain="unique.com",
                published_date="20240314",
                tone=0.6,
                query_type="newsdata",
            ),
        ]
        gdelt_articles = [
            NewswireArticle(
                url="https://reuters.com/shared-article",
                title="Shared Article (GDELT)",
                source_domain="reuters.com",
                published_date="20240315",
                tone=-1.0,
                query_type="nct_id",
            ),
        ]

        unique = NewsdataAdapter.deduplicate_with_gdelt(
            newsdata_articles, gdelt_articles
        )

        assert len(unique) == 1
        assert unique[0].url == "https://unique.com/only-newsdata"

    def test_removes_title_duplicates(self) -> None:
        """Articles with >80% title overlap with GDELT are removed."""
        newsdata_articles = [
            NewswireArticle(
                url="https://newsdata.com/article-1",
                title="Phase 2 Trial BKM120 Shows Results in Melanoma",
                source_domain="newsdata.com",
                published_date="20240315",
                tone=0.5,
                query_type="newsdata",
            ),
            NewswireArticle(
                url="https://newsdata.com/article-2",
                title="Completely Different Article About Something",
                source_domain="newsdata.com",
                published_date="20240314",
                tone=0.3,
                query_type="newsdata",
            ),
        ]
        gdelt_articles = [
            NewswireArticle(
                url="https://gdelt.com/different-url",
                title="Phase 2 Trial BKM120 Shows Results in Melanoma",
                source_domain="gdelt.com",
                published_date="20240315",
                tone=-1.0,
                query_type="nct_id",
            ),
        ]

        unique = NewsdataAdapter.deduplicate_with_gdelt(
            newsdata_articles, gdelt_articles
        )

        assert len(unique) == 1
        assert "Completely Different" in unique[0].title

    def test_preserves_all_when_no_overlap(self) -> None:
        """All articles preserved when no GDELT overlap."""
        newsdata_articles = [
            NewswireArticle(
                url="https://a.com/1",
                title="Article A",
                source_domain="a.com",
                published_date="20240315",
                tone=0.5,
                query_type="newsdata",
            ),
            NewswireArticle(
                url="https://b.com/2",
                title="Article B",
                source_domain="b.com",
                published_date="20240314",
                tone=0.6,
                query_type="newsdata",
            ),
        ]

        unique = NewsdataAdapter.deduplicate_with_gdelt(
            newsdata_articles, gdelt_articles=[]
        )

        assert len(unique) == 2

    def test_preserves_source_provenance(self) -> None:
        """Deduplicated results keep newsdata query_type."""
        newsdata_articles = [
            NewswireArticle(
                url="https://unique.com/article",
                title="Unique Article",
                source_domain="unique.com",
                published_date="20240315",
                tone=0.5,
                query_type="newsdata",
            ),
        ]

        unique = NewsdataAdapter.deduplicate_with_gdelt(
            newsdata_articles, gdelt_articles=[]
        )

        assert unique[0].query_type == "newsdata"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Cache behavior for NewsData results."""

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_results_cached_to_disk(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
        cache_dir: Path,
    ) -> None:
        """Fetched results are cached with _newsdata suffix."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_NEWSDATA_RESPONSE
        )

        adapter.search("NCT01512251", [])

        cache_file = cache_dir / "NCT01512251_newsdata.json"
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        assert cached["nct_id"] == "NCT01512251"

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_cache_hit_skips_api_call(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
        cache_dir: Path,
    ) -> None:
        """Second search reads from cache, no API call."""
        cached_data = {
            "nct_id": "NCT01512251",
            "articles": [
                {
                    "url": "https://example.com/cached",
                    "title": "Cached Article",
                    "source_domain": "example.com",
                    "published_date": "2024-03-01",
                    "tone": 0.5,
                    "themes": ["health"],
                    "query_type": "newsdata",
                    "fetched_utc": "2024-03-01T00:00:00+00:00",
                }
            ],
            "query_terms": ["NCT01512251"],
            "total_found": 1,
            "relevant_count": 1,
            "fetched_utc": "2024-03-01T00:00:00+00:00",
        }
        cache_file = cache_dir / "NCT01512251_newsdata.json"
        cache_file.write_text(
            json.dumps(cached_data), encoding="utf-8"
        )

        result = adapter.search("NCT01512251", [])

        assert result.from_cache is True
        assert len(result.articles) == 1
        mock_urlopen.assert_not_called()

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_skip_cache_forces_api_call(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
        cache_dir: Path,
    ) -> None:
        """skip_cache=True bypasses cache."""
        cached_data = {
            "nct_id": "NCT01512251",
            "articles": [],
            "query_terms": ["NCT01512251"],
            "total_found": 0,
            "relevant_count": 0,
            "fetched_utc": "2024-03-01T00:00:00+00:00",
        }
        cache_file = cache_dir / "NCT01512251_newsdata.json"
        cache_file.write_text(
            json.dumps(cached_data), encoding="utf-8"
        )

        mock_urlopen.return_value = _mock_response(
            _SAMPLE_NEWSDATA_RESPONSE
        )

        result = adapter.search(
            "NCT01512251", [], skip_cache=True
        )

        assert result.from_cache is False
        mock_urlopen.assert_called_once()

    @patch("ptcv.registry.newsdata_adapter.urllib.request.urlopen")
    def test_cache_includes_alcoa_provenance(
        self,
        mock_urlopen: MagicMock,
        adapter: NewsdataAdapter,
        cache_dir: Path,
    ) -> None:
        """Cached results include ALCOA+ provenance metadata."""
        mock_urlopen.return_value = _mock_response(
            _SAMPLE_NEWSDATA_RESPONSE
        )

        adapter.search("NCT01512251", [])

        cache_file = cache_dir / "NCT01512251_newsdata.json"
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "fetched_utc" in cached
        assert cached["fetched_utc"]
        assert "query_terms" in cached


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestDataModels:
    """NewsdataSearchResult dataclass tests."""

    def test_auto_sets_fetched_utc(self) -> None:
        """Auto-populates fetched_utc if not set."""
        result = NewsdataSearchResult(nct_id="NCT01512251")
        assert result.fetched_utc

    def test_batch_search(self) -> None:
        """batch_search returns dict mapping NCT ID to results."""
        adapter = NewsdataAdapter(
            api_key="test_key",
            audit_logger=MagicMock(),
            daily_limit=0,  # Block all requests
        )

        results = adapter.batch_search([
            ("NCT01512251", ["BKM120"]),
            ("NCT02000000", ["drug2"]),
        ])

        assert len(results) == 2
        assert "NCT01512251" in results
        assert "NCT02000000" in results
