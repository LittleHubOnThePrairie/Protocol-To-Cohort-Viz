"""NewsData.io Clinical Trial News Adapter.

PTCV-204: Queries the NewsData.io API to retrieve structured clinical
trial news articles. Returns article metadata (title, description,
source, date, category) and caches results per NCT ID.

NewsData.io API:
- Endpoint: https://newsdata.io/api/1/news
- Auth: API key (free tier)
- Free tier: 200 requests/day, 10 results per page
- Features: Category filtering, language filtering, keyword search

Risk tier: LOW — read-only news queries (no patient data).

Regulatory requirements:
- Audit trail: every fetch logged (21 CFR 11.10(e), ALCOA+)
- No PHI: public news data only (no participant identifiers)
"""

import dataclasses
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..compliance.audit import AuditAction, AuditLogger
from .gdelt_adapter import NewswireArticle

logger = logging.getLogger(__name__)

_NEWSDATA_API_URL = "https://newsdata.io/api/1/news"
_DEFAULT_CACHE_DIR = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials/registry_cache/newswire"
)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRYABLE_STATUS_CODES = {429, 503}
_DEFAULT_DAILY_LIMIT = 200
_DEFAULT_CATEGORY = "health"
_DEFAULT_LANGUAGE = "en"

# Clinical trial relevance keywords for scoring
_RELEVANCE_KEYWORDS = frozenset({
    "clinical trial", "phase 1", "phase 2", "phase 3", "phase 4",
    "phase i", "phase ii", "phase iii", "phase iv",
    "randomized", "randomised", "placebo", "double-blind",
    "open-label", "fda", "ema", "approval", "endpoint",
    "efficacy", "safety", "enrollment", "enrolment",
    "primary outcome", "secondary outcome", "interim analysis",
    "topline results", "pivotal", "regulatory",
    "nct", "clinicaltrials.gov",
})

# Minimum relevance score to include an article
_RELEVANCE_THRESHOLD = 0.1


@dataclasses.dataclass
class NewsdataSearchResult:
    """Aggregate result from a NewsData.io search for one NCT ID.

    Attributes:
        nct_id: The trial identifier queried.
        articles: Matching articles sorted by date descending.
        query_terms: Search terms used.
        total_found: Total articles returned before relevance filter.
        relevant_count: Articles retained after relevance filter.
        fetched_utc: Timestamp of the search.
        from_cache: Whether result was loaded from disk cache.
    """

    nct_id: str
    articles: list[NewswireArticle] = dataclasses.field(
        default_factory=list
    )
    query_terms: list[str] = dataclasses.field(default_factory=list)
    total_found: int = 0
    relevant_count: int = 0
    fetched_utc: str = ""
    from_cache: bool = False

    def __post_init__(self) -> None:
        if not self.fetched_utc:
            self.fetched_utc = datetime.now(timezone.utc).isoformat()


class NewsdataAdapter:
    """Query NewsData.io for clinical trial news articles.

    Follows the same fetch/cache pattern as ``GdeltAdapter``
    (PTCV-203) and ``RegistryMetadataFetcher`` (PTCV-194).

    Usage::

        adapter = NewsdataAdapter(api_key="your_key")
        result = adapter.search("NCT01512251", ["BKM120", "melanoma"])
        for article in result.articles:
            print(article.title, article.url)

    Args:
        api_key: NewsData.io API key.
        cache_dir: Directory for cached JSON responses.
        audit_logger: AuditLogger instance.  Uses default if None.
        timeout: HTTP request timeout in seconds.
        daily_limit: Maximum API requests per day.
        category: NewsData category filter (default "health").
        language: Language filter (default "en").
        relevance_threshold: Minimum relevance score to keep
            an article (0.0–1.0).
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
        timeout: int = 30,
        daily_limit: int = _DEFAULT_DAILY_LIMIT,
        category: str = _DEFAULT_CATEGORY,
        language: str = _DEFAULT_LANGUAGE,
        relevance_threshold: float = _RELEVANCE_THRESHOLD,
    ) -> None:
        self._api_key = api_key
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._audit = audit_logger or AuditLogger(
            module="newsdata_adapter"
        )
        self._timeout = timeout
        self._daily_limit = daily_limit
        self._category = category
        self._language = language
        self._relevance_threshold = relevance_threshold
        self._request_count = 0
        self._request_date: Optional[str] = None  # YYYY-MM-DD

    @property
    def remaining_quota(self) -> int:
        """Return remaining API requests for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._request_date != today:
            return self._daily_limit
        return max(0, self._daily_limit - self._request_count)

    def _check_daily_limit(self) -> bool:
        """Check and enforce the daily request limit.

        Returns:
            True if a request can be made, False if limit reached.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._request_date != today:
            self._request_date = today
            self._request_count = 0

        if self._request_count >= self._daily_limit:
            logger.warning(
                "NewsData.io daily limit reached (%d/%d). "
                "Skipping request.",
                self._request_count,
                self._daily_limit,
            )
            return False

        self._request_count += 1
        remaining = self._daily_limit - self._request_count
        if remaining <= 20:
            logger.info(
                "NewsData.io quota low: %d requests remaining",
                remaining,
            )
        return True

    def _cache_path(self, nct_id: str) -> Path:
        """Return the cache file path for an NCT ID."""
        return self._cache_dir / f"{nct_id}_newsdata.json"

    def _read_cache(
        self, nct_id: str
    ) -> Optional[NewsdataSearchResult]:
        """Return cached search result if present, else None."""
        path = self._cache_path(nct_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                articles = [
                    NewswireArticle(**a)
                    for a in data.get("articles", [])
                ]
                return NewsdataSearchResult(
                    nct_id=data["nct_id"],
                    articles=articles,
                    query_terms=data.get("query_terms", []),
                    total_found=data.get(
                        "total_found", len(articles)
                    ),
                    relevant_count=data.get(
                        "relevant_count", len(articles)
                    ),
                    fetched_utc=data.get("fetched_utc", ""),
                    from_cache=True,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "Corrupt NewsData cache for %s (%s), "
                    "will re-fetch",
                    nct_id,
                    exc,
                )
        return None

    def _write_cache(
        self, nct_id: str, result: NewsdataSearchResult
    ) -> None:
        """Persist search result to disk cache."""
        path = self._cache_path(nct_id)
        data = {
            "nct_id": result.nct_id,
            "articles": [
                dataclasses.asdict(a) for a in result.articles
            ],
            "query_terms": result.query_terms,
            "total_found": result.total_found,
            "relevant_count": result.relevant_count,
            "fetched_utc": result.fetched_utc,
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _get_json(self, url: str) -> dict[str, Any]:
        """GET a URL and return parsed JSON with retry on 429/503."""
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(
                    req, timeout=self._timeout
                ) as resp:
                    return json.loads(  # type: ignore[no-any-return]
                        resp.read().decode("utf-8")
                    )
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUS_CODES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "HTTP %d for %s (attempt %d/%d), "
                        "retrying in %.1fs",
                        exc.code,
                        url,
                        attempt,
                        _MAX_RETRIES,
                        delay,
                    )
                    last_exc = exc
                    time.sleep(delay)
                    continue
                raise
            except urllib.error.URLError as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "URLError for %s (attempt %d/%d): %s, "
                    "retrying in %.1fs",
                    url,
                    attempt,
                    _MAX_RETRIES,
                    exc.reason,
                    delay,
                )
                last_exc = exc
                time.sleep(delay)
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "Unreachable: retry loop exited without result"
        )

    def _build_query_url(self, query: str) -> str:
        """Build a NewsData.io query URL.

        Args:
            query: Search terms (e.g. ``"NCT01512251"`` or
                ``"pembrolizumab melanoma"``).

        Returns:
            Full URL with query parameters and API key.
        """
        params = {
            "apikey": self._api_key,
            "q": query,
            "language": self._language,
            "category": self._category,
        }
        return f"{_NEWSDATA_API_URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def score_relevance(title: str, description: str) -> float:
        """Score an article's relevance to clinical trials.

        Returns a score between 0.0 and 1.0 based on keyword
        matches in the title and description.

        Args:
            title: Article headline.
            description: Article body text or summary.

        Returns:
            Relevance score (0.0–1.0).
        """
        text = f"{title} {description}".lower()
        matches = sum(
            1 for kw in _RELEVANCE_KEYWORDS if kw in text
        )
        # Normalize: max score at 5+ keyword matches
        return min(matches / 5.0, 1.0)

    def _parse_articles(
        self, data: dict[str, Any]
    ) -> tuple[list[NewswireArticle], int]:
        """Parse NewsData.io response into NewswireArticle list.

        Returns:
            Tuple of (relevant_articles, total_before_filter).
        """
        all_articles: list[NewswireArticle] = []
        now_utc = datetime.now(timezone.utc).isoformat()

        results = data.get("results") or []
        total_before = len(results)

        for item in results:
            title = item.get("title") or ""
            description = item.get("description") or ""
            relevance = self.score_relevance(title, description)

            if relevance < self._relevance_threshold:
                continue

            categories = item.get("category") or []
            if isinstance(categories, list):
                themes = categories
            else:
                themes = [str(categories)]

            all_articles.append(
                NewswireArticle(
                    url=item.get("link") or "",
                    title=title,
                    source_domain=item.get("source_id") or "",
                    published_date=item.get("pubDate") or "",
                    tone=relevance,  # Use relevance as tone proxy
                    themes=themes,
                    query_type="newsdata",
                    fetched_utc=now_utc,
                )
            )

        return all_articles, total_before

    def search(
        self,
        nct_id: str,
        keywords: list[str],
        *,
        skip_cache: bool = False,
    ) -> NewsdataSearchResult:
        """Search NewsData.io for clinical trial news.

        Queries by NCT ID and/or keywords. Applies relevance
        filtering and caches results per NCT ID.

        Args:
            nct_id: NCT ID to associate results with.
            keywords: Search terms (e.g. ``["pembrolizumab",
                "melanoma"]``). Combined with NCT ID for query.
            skip_cache: Force a fresh API query.

        Returns:
            NewsdataSearchResult with relevant articles.
        """
        if not skip_cache:
            cached = self._read_cache(nct_id)
            if cached is not None:
                logger.debug("NewsData cache hit for %s", nct_id)
                return cached

        query_terms = [nct_id] + keywords if keywords else [nct_id]
        query = " ".join(query_terms)

        if not self._check_daily_limit():
            return NewsdataSearchResult(
                nct_id=nct_id, query_terms=query_terms
            )

        url = self._build_query_url(query)

        try:
            data = self._get_json(url)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            OSError,
        ) as exc:
            logger.warning(
                "NewsData.io query failed for %s: %s", nct_id, exc
            )
            return NewsdataSearchResult(
                nct_id=nct_id, query_terms=query_terms
            )

        articles, total_before = self._parse_articles(data)
        result = NewsdataSearchResult(
            nct_id=nct_id,
            articles=articles,
            query_terms=query_terms,
            total_found=total_before,
            relevant_count=len(articles),
        )

        self._write_cache(nct_id, result)
        self._audit.log(
            action=AuditAction.SEARCH,
            record_id=nct_id,
            user_id="ptcv_pipeline",
            reason=(
                "NewsData.io clinical trial news search "
                "(PTCV-204)"
            ),
            after={
                "articles_found": total_before,
                "articles_relevant": len(articles),
                "query_terms": query_terms,
                "remaining_quota": self.remaining_quota,
            },
        )
        logger.info(
            "NewsData.io search for %s: %d/%d articles relevant "
            "(quota: %d remaining)",
            nct_id,
            len(articles),
            total_before,
            self.remaining_quota,
        )
        return result

    @staticmethod
    def deduplicate_with_gdelt(
        newsdata_articles: list[NewswireArticle],
        gdelt_articles: list[NewswireArticle],
    ) -> list[NewswireArticle]:
        """Remove NewsData articles that duplicate GDELT articles.

        Deduplication is by exact URL match or high title similarity
        (>80% token overlap).

        Args:
            newsdata_articles: Articles from NewsData.io.
            gdelt_articles: Articles from GDELT DOC 2.0.

        Returns:
            NewsData articles not present in GDELT results.
        """
        gdelt_urls = {a.url for a in gdelt_articles if a.url}
        gdelt_title_tokens = {
            frozenset(
                re.findall(r"[a-z0-9]+", a.title.lower())
            )
            for a in gdelt_articles
            if a.title
        }

        unique: list[NewswireArticle] = []
        for article in newsdata_articles:
            # Exact URL match
            if article.url and article.url in gdelt_urls:
                continue

            # Title similarity check
            if article.title:
                tokens = set(
                    re.findall(r"[a-z0-9]+", article.title.lower())
                )
                if tokens and any(
                    len(tokens & gt) / len(tokens | gt) > 0.8
                    for gt in gdelt_title_tokens
                    if gt
                ):
                    continue

            unique.append(article)

        return unique

    def batch_search(
        self,
        trials: list[tuple[str, list[str]]],
    ) -> dict[str, NewsdataSearchResult]:
        """Search NewsData.io for multiple trials.

        Args:
            trials: List of (nct_id, keywords) tuples.

        Returns:
            Dict mapping each NCT ID to its search result.
        """
        results: dict[str, NewsdataSearchResult] = {}
        for nct_id, keywords in trials:
            results[nct_id] = self.search(nct_id, keywords)
        return results
