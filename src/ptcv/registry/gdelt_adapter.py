"""GDELT DOC 2.0 Newswire Adapter for clinical trial press releases.

PTCV-203: Queries the GDELT DOC 2.0 API to discover press releases and
news articles mentioning clinical trials by NCT ID or trial keywords.
Returns structured article metadata (URL, title, date, tone score) and
caches results per NCT ID for downstream enrichment.

GDELT DOC 2.0 API:
- Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
- Auth: None required
- Rate limits: No documented limit (conservative 1 req/sec)
- Cost: Free

Risk tier: LOW — read-only news queries (no patient data).

Regulatory requirements:
- Audit trail: every fetch logged (21 CFR 11.10(e), ALCOA+)
- No PHI: public news data only (no participant identifiers)
"""

import dataclasses
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..compliance.audit import AuditAction, AuditLogger

logger = logging.getLogger(__name__)

_GDELT_DOC_V2_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_DEFAULT_CACHE_DIR = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials/registry_cache/newswire"
)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRYABLE_STATUS_CODES = {429, 503}
_DEFAULT_REQUEST_DELAY = 1.0  # Conservative 1 req/sec
_DEFAULT_TIMESPAN = "12m"  # Search last 12 months by default
_DEFAULT_MAX_RECORDS = 50


@dataclasses.dataclass
class NewswireArticle:
    """A single news article from GDELT.

    Attributes:
        url: Article URL.
        title: Article headline.
        source_domain: Publishing domain (e.g. "businesswire.com").
        published_date: ISO 8601 publication date string.
        tone: GDELT tone score (-100 to +100; negative = negative
            sentiment).
        themes: GDELT theme tags (e.g. "HEALTH_PANDEMIC",
            "MEDICAL").
        query_type: Whether found via "nct_id" or "keyword" search.
        fetched_utc: ISO 8601 timestamp of when this article was
            fetched.
    """

    url: str
    title: str
    source_domain: str
    published_date: str
    tone: float
    themes: list[str] = dataclasses.field(default_factory=list)
    query_type: str = "nct_id"
    fetched_utc: str = ""

    def __post_init__(self) -> None:
        if not self.fetched_utc:
            self.fetched_utc = datetime.now(timezone.utc).isoformat()


@dataclasses.dataclass
class GdeltSearchResult:
    """Aggregate result from a GDELT search for one NCT ID.

    Attributes:
        nct_id: The trial identifier queried.
        articles: Matching articles sorted by date descending.
        query_terms: Search terms used.
        total_found: Total articles returned by GDELT.
        fetched_utc: Timestamp of the search.
        from_cache: Whether result was loaded from disk cache.
    """

    nct_id: str
    articles: list[NewswireArticle] = dataclasses.field(
        default_factory=list
    )
    query_terms: list[str] = dataclasses.field(default_factory=list)
    total_found: int = 0
    fetched_utc: str = ""
    from_cache: bool = False

    def __post_init__(self) -> None:
        if not self.fetched_utc:
            self.fetched_utc = datetime.now(timezone.utc).isoformat()


class GdeltAdapter:
    """Query GDELT DOC 2.0 for clinical trial press releases.

    Follows the same fetch/cache pattern as ``RegistryMetadataFetcher``
    (PTCV-194).

    Usage::

        adapter = GdeltAdapter()
        result = adapter.search_by_nct_id("NCT01512251")
        for article in result.articles:
            print(article.title, article.url)

    Args:
        cache_dir: Directory for cached JSON responses.
        audit_logger: AuditLogger instance.  Uses default if None.
        timeout: HTTP request timeout in seconds.
        request_delay: Delay between API requests in seconds.
        timespan: GDELT timespan parameter (e.g. "12m", "3m").
        max_records: Maximum articles to return per query.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
        timeout: int = 30,
        request_delay: float = _DEFAULT_REQUEST_DELAY,
        timespan: str = _DEFAULT_TIMESPAN,
        max_records: int = _DEFAULT_MAX_RECORDS,
    ) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._audit = audit_logger or AuditLogger(
            module="gdelt_adapter"
        )
        self._timeout = timeout
        self._request_delay = request_delay
        self._timespan = timespan
        self._max_records = max_records
        self._last_request_time: float = 0.0

    def _cache_path(self, nct_id: str) -> Path:
        """Return the cache file path for an NCT ID."""
        return self._cache_dir / f"{nct_id}_gdelt.json"

    def _read_cache(self, nct_id: str) -> Optional[GdeltSearchResult]:
        """Return cached search result if present, else None."""
        path = self._cache_path(nct_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                articles = [
                    NewswireArticle(**a) for a in data.get("articles", [])
                ]
                return GdeltSearchResult(
                    nct_id=data["nct_id"],
                    articles=articles,
                    query_terms=data.get("query_terms", []),
                    total_found=data.get("total_found", len(articles)),
                    fetched_utc=data.get("fetched_utc", ""),
                    from_cache=True,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "Corrupt GDELT cache for %s (%s), will re-fetch",
                    nct_id,
                    exc,
                )
        return None

    def _write_cache(
        self, nct_id: str, result: GdeltSearchResult
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
            "fetched_utc": result.fetched_utc,
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _rate_limit(self) -> None:
        """Enforce minimum delay between API requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.monotonic()

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
        """Build a GDELT DOC 2.0 query URL.

        Args:
            query: Search terms (e.g. ``"NCT01512251"`` or
                ``"pembrolizumab melanoma clinical trial"``).

        Returns:
            Full URL with query parameters.
        """
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "timespan": self._timespan,
            "maxrecords": str(self._max_records),
            "sort": "DateDesc",
        }
        return f"{_GDELT_DOC_V2_URL}?{urllib.parse.urlencode(params)}"

    def _parse_articles(
        self, data: dict[str, Any], query_type: str
    ) -> list[NewswireArticle]:
        """Parse GDELT JSON response into NewswireArticle list."""
        articles: list[NewswireArticle] = []
        now_utc = datetime.now(timezone.utc).isoformat()

        for item in data.get("articles", []):
            try:
                tone_str = item.get("tone", "0")
                tone = float(str(tone_str).split(",")[0])
            except (ValueError, IndexError):
                tone = 0.0

            themes_raw = item.get("themes", "")
            themes = (
                [t.strip() for t in themes_raw.split(";") if t.strip()]
                if isinstance(themes_raw, str) and themes_raw
                else []
            )

            articles.append(
                NewswireArticle(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    source_domain=item.get("domain", ""),
                    published_date=item.get("seendate", ""),
                    tone=tone,
                    themes=themes,
                    query_type=query_type,
                    fetched_utc=now_utc,
                )
            )

        return articles

    def search_by_nct_id(
        self, nct_id: str, *, skip_cache: bool = False
    ) -> GdeltSearchResult:
        """Search GDELT for articles mentioning an NCT ID.

        Args:
            nct_id: ClinicalTrials.gov identifier (e.g.
                ``NCT01512251``).
            skip_cache: Force a fresh API query.

        Returns:
            GdeltSearchResult with matching articles.
        """
        if not skip_cache:
            cached = self._read_cache(nct_id)
            if cached is not None:
                logger.debug("GDELT cache hit for %s", nct_id)
                return cached

        url = self._build_query_url(nct_id)
        query_terms = [nct_id]

        try:
            self._rate_limit()
            data = self._get_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            logger.warning(
                "GDELT query failed for %s: %s", nct_id, exc
            )
            return GdeltSearchResult(
                nct_id=nct_id, query_terms=query_terms
            )

        articles = self._parse_articles(data, query_type="nct_id")
        result = GdeltSearchResult(
            nct_id=nct_id,
            articles=articles,
            query_terms=query_terms,
            total_found=len(articles),
        )

        self._write_cache(nct_id, result)
        self._audit.log(
            action=AuditAction.SEARCH,
            record_id=nct_id,
            user_id="ptcv_pipeline",
            reason=(
                "GDELT DOC 2.0 newswire search for clinical trial "
                "press releases (PTCV-203)"
            ),
            after={
                "articles_found": len(articles),
                "query_terms": query_terms,
            },
        )
        logger.info(
            "GDELT search for %s: %d articles found",
            nct_id,
            len(articles),
        )
        return result

    def search_by_keywords(
        self,
        nct_id: str,
        keywords: list[str],
        *,
        skip_cache: bool = False,
    ) -> GdeltSearchResult:
        """Search GDELT by trial name keywords.

        Queries with compound name, indication, or other keywords.
        Results are merged with any existing NCT ID search results
        in the cache.

        Args:
            nct_id: NCT ID to associate results with.
            keywords: Search terms (e.g. ``["pembrolizumab",
                "melanoma", "clinical trial"]``).
            skip_cache: Force a fresh API query.

        Returns:
            GdeltSearchResult with matching articles.
        """
        if not keywords:
            return GdeltSearchResult(nct_id=nct_id)

        query = " ".join(keywords)
        url = self._build_query_url(query)

        try:
            self._rate_limit()
            data = self._get_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            logger.warning(
                "GDELT keyword query failed for %s (%s): %s",
                nct_id,
                query,
                exc,
            )
            return GdeltSearchResult(
                nct_id=nct_id, query_terms=keywords
            )

        articles = self._parse_articles(data, query_type="keyword")
        result = GdeltSearchResult(
            nct_id=nct_id,
            articles=articles,
            query_terms=keywords,
            total_found=len(articles),
        )

        # Merge with existing cached NCT ID results if present
        if not skip_cache:
            cached = self._read_cache(nct_id)
            if cached is not None:
                result = self._merge_results(cached, result)

        self._write_cache(nct_id, result)
        self._audit.log(
            action=AuditAction.SEARCH,
            record_id=nct_id,
            user_id="ptcv_pipeline",
            reason=(
                "GDELT DOC 2.0 keyword search for clinical trial "
                "press releases (PTCV-203)"
            ),
            after={
                "articles_found": len(result.articles),
                "query_terms": keywords,
            },
        )
        logger.info(
            "GDELT keyword search for %s (%s): %d articles found",
            nct_id,
            query,
            len(articles),
        )
        return result

    @staticmethod
    def _merge_results(
        existing: GdeltSearchResult,
        new: GdeltSearchResult,
    ) -> GdeltSearchResult:
        """Merge two search results, deduplicating by URL."""
        seen_urls: set[str] = set()
        merged: list[NewswireArticle] = []

        for article in existing.articles + new.articles:
            if article.url and article.url not in seen_urls:
                seen_urls.add(article.url)
                merged.append(article)

        return GdeltSearchResult(
            nct_id=new.nct_id,
            articles=merged,
            query_terms=list(
                dict.fromkeys(
                    existing.query_terms + new.query_terms
                )
            ),
            total_found=len(merged),
        )

    def batch_search(
        self, nct_ids: list[str]
    ) -> dict[str, GdeltSearchResult]:
        """Search GDELT for multiple NCT IDs.

        Args:
            nct_ids: List of NCT IDs to search.

        Returns:
            Dict mapping each NCT ID to its search result.
        """
        results: dict[str, GdeltSearchResult] = {}
        for nct_id in nct_ids:
            results[nct_id] = self.search_by_nct_id(nct_id)
        return results
