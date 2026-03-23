"""Tests for LiteratureEnricher (PTCV-206).

Covers all GHERKIN scenarios:
- Orchestrate enrichment across PubMed, GDELT, and NewsData.io
- Deduplicate articles across sources
- Cache enrichment results per NCT ID
- Support selective source queries
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptcv.registry.gdelt_adapter import (
    GdeltAdapter,
    GdeltSearchResult,
    NewswireArticle,
)
from ptcv.registry.literature_enricher import (
    VALID_SOURCES,
    EnrichmentResult,
    LiteratureEnricher,
    parse_sources,
)
from ptcv.registry.newsdata_adapter import (
    NewsdataAdapter,
    NewsdataSearchResult,
)
from ptcv.registry.pubmed_adapter import PubmedAdapter, PubmedArticle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_newswire(
    url: str = "https://example.com/article",
    title: str = "Trial Article",
    query_type: str = "nct_id",
) -> NewswireArticle:
    return NewswireArticle(
        url=url,
        title=title,
        source_domain="example.com",
        published_date="20240315",
        tone=-1.0,
        query_type=query_type,
    )


def _make_pubmed(
    pmid: str = "12345678",
    title: str = "PubMed Article",
) -> PubmedArticle:
    return PubmedArticle(
        pmid=pmid,
        title=title,
        authors=["Author A"],
        journal="J Clin Oncol",
        pub_date="2024-03",
        abstract="Abstract text.",
        doi="10.1000/test",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "registry_cache"
    d.mkdir()
    return d


@pytest.fixture
def mock_gdelt() -> MagicMock:
    adapter = MagicMock(spec=GdeltAdapter)
    adapter.search_by_nct_id.return_value = GdeltSearchResult(
        nct_id="NCT01512251",
        articles=[
            _make_newswire(
                url="https://reuters.com/gdelt-article",
                title="GDELT Article on NCT01512251",
            ),
        ],
        total_found=1,
    )
    adapter.search_by_keywords.return_value = GdeltSearchResult(
        nct_id="NCT01512251",
        articles=[],
        total_found=0,
    )
    return adapter


@pytest.fixture
def mock_newsdata() -> MagicMock:
    adapter = MagicMock(spec=NewsdataAdapter)
    adapter.search.return_value = NewsdataSearchResult(
        nct_id="NCT01512251",
        articles=[
            _make_newswire(
                url="https://newsdata.com/unique",
                title="Unique NewsData Article",
                query_type="newsdata",
            ),
            _make_newswire(
                url="https://reuters.com/gdelt-article",
                title="Duplicate of GDELT",
                query_type="newsdata",
            ),
        ],
        total_found=2,
        relevant_count=2,
    )
    # Deduplication is a static method so we need to allow it
    adapter.deduplicate_with_gdelt = NewsdataAdapter.deduplicate_with_gdelt
    return adapter


@pytest.fixture
def mock_pubmed() -> MagicMock:
    adapter = MagicMock(spec=PubmedAdapter)
    from ptcv.registry.pubmed_adapter import PubmedSearchResult

    adapter.search_by_nct_id.return_value = PubmedSearchResult(
        nct_id="NCT01512251",
        articles=[_make_pubmed()],
        pmid_count=1,
    )
    return adapter


@pytest.fixture
def enricher(
    cache_dir: Path,
    mock_gdelt: MagicMock,
    mock_newsdata: MagicMock,
    mock_pubmed: MagicMock,
) -> LiteratureEnricher:
    return LiteratureEnricher(
        gdelt_adapter=mock_gdelt,
        newsdata_adapter=mock_newsdata,
        pubmed_adapter=mock_pubmed,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Orchestrate enrichment across all sources
# ---------------------------------------------------------------------------


class TestOrchestration:
    """Scenario: Orchestrate enrichment across PubMed, GDELT, NewsData."""

    def test_queries_all_sources_by_default(
        self,
        enricher: LiteratureEnricher,
        mock_gdelt: MagicMock,
        mock_newsdata: MagicMock,
        mock_pubmed: MagicMock,
    ) -> None:
        """All three sources are queried when no source filter."""
        result = enricher.enrich_trial("NCT01512251")

        mock_pubmed.search_by_nct_id.assert_called_once()
        mock_gdelt.search_by_nct_id.assert_called_once()
        mock_newsdata.search.assert_called_once()
        assert "pubmed" in result.sources_queried
        assert "gdelt" in result.sources_queried
        assert "newsdata" in result.sources_queried

    def test_priority_order_pubmed_first(
        self,
        enricher: LiteratureEnricher,
    ) -> None:
        """PubMed is queried first (highest signal)."""
        result = enricher.enrich_trial("NCT01512251")

        assert result.sources_queried[0] == "pubmed"

    def test_returns_enrichment_result(
        self,
        enricher: LiteratureEnricher,
    ) -> None:
        """Returns EnrichmentResult with articles from all sources."""
        result = enricher.enrich_trial("NCT01512251")

        assert isinstance(result, EnrichmentResult)
        assert result.nct_id == "NCT01512251"
        assert len(result.pubmed_articles) == 1
        assert len(result.gdelt_articles) == 1
        assert result.total_articles > 0

    def test_enrichment_result_summary(
        self,
        enricher: LiteratureEnricher,
    ) -> None:
        """Summary includes counts per source."""
        result = enricher.enrich_trial("NCT01512251")
        summary = result.summary()

        assert "NCT01512251" in summary
        assert "PubMed" in summary
        assert "GDELT" in summary


# ---------------------------------------------------------------------------
# Scenario 2: Deduplicate articles across sources
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Scenario: Deduplicate articles across sources."""

    def test_newsdata_deduped_against_gdelt(
        self,
        enricher: LiteratureEnricher,
    ) -> None:
        """NewsData articles sharing URL with GDELT are removed."""
        result = enricher.enrich_trial("NCT01512251")

        # mock_newsdata returns 2 articles, one duplicates GDELT URL
        # After dedup, only the unique one should remain
        newsdata_urls = [a.url for a in result.newsdata_articles]
        assert "https://reuters.com/gdelt-article" not in newsdata_urls
        assert "https://newsdata.com/unique" in newsdata_urls

    def test_total_articles_reflects_dedup(
        self,
        enricher: LiteratureEnricher,
    ) -> None:
        """total_articles counts unique articles after dedup."""
        result = enricher.enrich_trial("NCT01512251")

        expected = (
            len(result.gdelt_articles)
            + len(result.newsdata_articles)
            + len(result.pubmed_articles)
        )
        assert result.total_articles == expected


# ---------------------------------------------------------------------------
# Scenario 3: Cache enrichment results
# ---------------------------------------------------------------------------


class TestCaching:
    """Scenario: Cache enrichment results per NCT ID."""

    def test_result_cached_to_disk(
        self,
        enricher: LiteratureEnricher,
        cache_dir: Path,
    ) -> None:
        """Enrichment result is written to cache file."""
        enricher.enrich_trial("NCT01512251")

        cache_file = (
            cache_dir / "enrichment" / "NCT01512251_enrichment.json"
        )
        assert cache_file.exists()
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert data["nct_id"] == "NCT01512251"
        assert "enriched_utc" in data

    def test_cache_hit_skips_api_calls(
        self,
        enricher: LiteratureEnricher,
        mock_gdelt: MagicMock,
        mock_pubmed: MagicMock,
    ) -> None:
        """Second call reads from cache, no adapter calls."""
        enricher.enrich_trial("NCT01512251")
        mock_gdelt.reset_mock()
        mock_pubmed.reset_mock()

        result = enricher.enrich_trial("NCT01512251")

        mock_gdelt.search_by_nct_id.assert_not_called()
        mock_pubmed.search_by_nct_id.assert_not_called()

    def test_skip_cache_forces_fresh_query(
        self,
        enricher: LiteratureEnricher,
        mock_gdelt: MagicMock,
    ) -> None:
        """skip_cache=True bypasses cache."""
        enricher.enrich_trial("NCT01512251")
        mock_gdelt.reset_mock()

        enricher.enrich_trial("NCT01512251", skip_cache=True)

        mock_gdelt.search_by_nct_id.assert_called_once()

    def test_cache_includes_provenance(
        self,
        enricher: LiteratureEnricher,
        cache_dir: Path,
    ) -> None:
        """Cached data includes ALCOA+ provenance fields."""
        enricher.enrich_trial("NCT01512251")

        cache_file = (
            cache_dir / "enrichment" / "NCT01512251_enrichment.json"
        )
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert data["enriched_utc"]
        assert "sources_queried" in data
        assert len(data["sources_queried"]) > 0


# ---------------------------------------------------------------------------
# Scenario 4: Support selective source queries
# ---------------------------------------------------------------------------


class TestSelectiveSources:
    """Scenario: Support selective source queries."""

    def test_pubmed_only(
        self,
        enricher: LiteratureEnricher,
        mock_gdelt: MagicMock,
        mock_newsdata: MagicMock,
    ) -> None:
        """Only PubMed is queried when sources={'pubmed'}."""
        result = enricher.enrich_trial(
            "NCT01512251", sources={"pubmed"}
        )

        mock_gdelt.search_by_nct_id.assert_not_called()
        mock_newsdata.search.assert_not_called()
        assert result.sources_queried == ["pubmed"]

    def test_gdelt_only(
        self,
        enricher: LiteratureEnricher,
        mock_pubmed: MagicMock,
        mock_newsdata: MagicMock,
    ) -> None:
        """Only GDELT is queried when sources={'gdelt'}."""
        result = enricher.enrich_trial(
            "NCT01512251", sources={"gdelt"}
        )

        mock_pubmed.search_by_nct_id.assert_not_called()
        mock_newsdata.search.assert_not_called()
        assert result.sources_queried == ["gdelt"]

    def test_gdelt_and_newsdata(
        self,
        enricher: LiteratureEnricher,
        mock_pubmed: MagicMock,
    ) -> None:
        """GDELT + NewsData queried without PubMed."""
        result = enricher.enrich_trial(
            "NCT01512251", sources={"gdelt", "newsdata"}
        )

        mock_pubmed.search_by_nct_id.assert_not_called()
        assert "gdelt" in result.sources_queried
        assert "newsdata" in result.sources_queried

    def test_missing_adapter_logged_gracefully(
        self,
        cache_dir: Path,
    ) -> None:
        """Missing adapter does not raise, source still listed."""
        enricher = LiteratureEnricher(
            gdelt_adapter=None,
            newsdata_adapter=None,
            pubmed_adapter=None,
            cache_dir=cache_dir,
        )

        result = enricher.enrich_trial("NCT01512251")

        assert result.total_articles == 0
        assert result.nct_id == "NCT01512251"


# ---------------------------------------------------------------------------
# parse_sources()
# ---------------------------------------------------------------------------


class TestParseSources:
    """Tests for parse_sources() helper."""

    def test_all_returns_all_sources(self) -> None:
        assert parse_sources("all") == VALID_SOURCES

    def test_single_source(self) -> None:
        assert parse_sources("pubmed") == {"pubmed"}

    def test_multiple_sources(self) -> None:
        assert parse_sources("gdelt,newsdata") == {
            "gdelt", "newsdata"
        }

    def test_whitespace_handling(self) -> None:
        assert parse_sources(" gdelt , pubmed ") == {
            "gdelt", "pubmed"
        }

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid source"):
            parse_sources("gdelt,cision")

    def test_case_insensitive(self) -> None:
        assert parse_sources("PubMed,GDELT") == {
            "pubmed", "gdelt"
        }


# ---------------------------------------------------------------------------
# EnrichmentResult dataclass
# ---------------------------------------------------------------------------


class TestEnrichmentResult:
    """EnrichmentResult dataclass tests."""

    def test_auto_sets_enriched_utc(self) -> None:
        result = EnrichmentResult(nct_id="NCT01512251")
        assert result.enriched_utc

    def test_auto_computes_total_articles(self) -> None:
        result = EnrichmentResult(
            nct_id="NCT01512251",
            gdelt_articles=[_make_newswire()],
            pubmed_articles=[_make_pubmed()],
        )
        assert result.total_articles == 2

    def test_summary_no_articles(self) -> None:
        result = EnrichmentResult(nct_id="NCT01512251")
        assert "no articles" in result.summary()

    def test_summary_with_articles(self) -> None:
        result = EnrichmentResult(
            nct_id="NCT01512251",
            gdelt_articles=[_make_newswire()],
            pubmed_articles=[_make_pubmed()],
        )
        summary = result.summary()
        assert "2 articles" in summary
        assert "GDELT" in summary
        assert "PubMed" in summary


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------


class TestBatchEnrichment:
    """Tests for enrich_batch()."""

    def test_enriches_multiple_trials(
        self,
        enricher: LiteratureEnricher,
        mock_gdelt: MagicMock,
        mock_pubmed: MagicMock,
    ) -> None:
        """enrich_batch returns dict mapping NCT IDs to results."""
        # Set up mock to work for any NCT ID
        mock_gdelt.search_by_nct_id.return_value = GdeltSearchResult(
            nct_id="any", articles=[], total_found=0,
        )
        from ptcv.registry.pubmed_adapter import PubmedSearchResult

        mock_pubmed.search_by_nct_id.return_value = PubmedSearchResult(
            nct_id="any", articles=[], pmid_count=0,
        )

        results = enricher.enrich_batch(
            ["NCT01512251", "NCT02000000"]
        )

        assert len(results) == 2
        assert "NCT01512251" in results
        assert "NCT02000000" in results
