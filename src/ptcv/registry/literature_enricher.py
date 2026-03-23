"""Literature and newswire enrichment for registry metadata.

PTCV-206: Orchestrates PubMed, GDELT, and NewsData.io adapters to
enrich ClinicalTrials.gov metadata with journal articles and press
release data. Called by ``populate()`` when enrichment is enabled, or
directly via ``enrich_trial()``.

Priority order: PubMed (highest signal) → GDELT → NewsData.io

Risk tier: LOW — read-only queries across public data sources.

Regulatory requirements:
- Audit trail: every enrichment logged (21 CFR 11.10(e), ALCOA+)
- No PHI: public news/literature data only
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .gdelt_adapter import GdeltAdapter, GdeltSearchResult, NewswireArticle
from .newsdata_adapter import NewsdataAdapter, NewsdataSearchResult
from .pubmed_adapter import PubmedAdapter, PubmedArticle

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials/registry_cache"
)

# Valid source names for --sources flag
VALID_SOURCES = frozenset({"pubmed", "gdelt", "newsdata"})


@dataclasses.dataclass
class EnrichmentResult:
    """Aggregated enrichment result for one NCT ID.

    Attributes:
        nct_id: Trial identifier.
        gdelt_articles: Articles from GDELT DOC 2.0.
        newsdata_articles: Articles from NewsData.io
            (after deduplication with GDELT).
        pubmed_articles: Articles from PubMed (PTCV-205).
        total_articles: Total unique articles across all sources.
        sources_queried: Which sources were actually queried.
        enriched_utc: Timestamp of enrichment.
    """

    nct_id: str
    gdelt_articles: list[NewswireArticle] = dataclasses.field(
        default_factory=list
    )
    newsdata_articles: list[NewswireArticle] = dataclasses.field(
        default_factory=list
    )
    pubmed_articles: list[PubmedArticle] = dataclasses.field(
        default_factory=list
    )
    total_articles: int = 0
    sources_queried: list[str] = dataclasses.field(
        default_factory=list
    )
    enriched_utc: str = ""

    def __post_init__(self) -> None:
        if not self.enriched_utc:
            self.enriched_utc = datetime.now(timezone.utc).isoformat()
        if not self.total_articles:
            self.total_articles = (
                len(self.gdelt_articles)
                + len(self.newsdata_articles)
                + len(self.pubmed_articles)
            )

    def summary(self) -> str:
        """One-line summary of enrichment result."""
        parts = []
        if self.gdelt_articles:
            parts.append(f"{len(self.gdelt_articles)} GDELT")
        if self.newsdata_articles:
            parts.append(f"{len(self.newsdata_articles)} NewsData")
        if self.pubmed_articles:
            parts.append(f"{len(self.pubmed_articles)} PubMed")
        if not parts:
            return f"{self.nct_id}: no articles found"
        return (
            f"{self.nct_id}: {self.total_articles} articles "
            f"({', '.join(parts)})"
        )


def parse_sources(sources_str: str) -> set[str]:
    """Parse a comma-separated sources string.

    Args:
        sources_str: e.g. ``"gdelt,newsdata"`` or ``"all"``.

    Returns:
        Set of valid source names.

    Raises:
        ValueError: If any source name is invalid.
    """
    if sources_str.strip().lower() == "all":
        return set(VALID_SOURCES)

    requested = {
        s.strip().lower() for s in sources_str.split(",") if s.strip()
    }
    invalid = requested - VALID_SOURCES
    if invalid:
        raise ValueError(
            f"Invalid source(s): {', '.join(sorted(invalid))}. "
            f"Valid sources: {', '.join(sorted(VALID_SOURCES))}"
        )
    return requested


class LiteratureEnricher:
    """Orchestrate literature and newswire enrichment.

    Calls adapters in priority order and aggregates results with
    cross-source deduplication.

    Args:
        gdelt_adapter: GdeltAdapter instance (or None to skip).
        newsdata_adapter: NewsdataAdapter instance (or None to skip).
        pubmed_adapter: PubmedAdapter instance (or None to skip).
        cache_dir: Directory for enrichment cache.
    """

    def __init__(
        self,
        gdelt_adapter: Optional[GdeltAdapter] = None,
        newsdata_adapter: Optional[NewsdataAdapter] = None,
        pubmed_adapter: Optional[PubmedAdapter] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._gdelt = gdelt_adapter
        self._newsdata = newsdata_adapter
        self._pubmed = pubmed_adapter
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _enrichment_cache_path(self, nct_id: str) -> Path:
        """Return the enrichment cache file path."""
        enrichment_dir = self._cache_dir / "enrichment"
        enrichment_dir.mkdir(parents=True, exist_ok=True)
        return enrichment_dir / f"{nct_id}_enrichment.json"

    def _read_enrichment_cache(
        self, nct_id: str
    ) -> Optional[EnrichmentResult]:
        """Return cached enrichment if present."""
        path = self._enrichment_cache_path(nct_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return EnrichmentResult(
                    nct_id=data["nct_id"],
                    gdelt_articles=[
                        NewswireArticle(**a)
                        for a in data.get("gdelt_articles", [])
                    ],
                    newsdata_articles=[
                        NewswireArticle(**a)
                        for a in data.get("newsdata_articles", [])
                    ],
                    pubmed_articles=[
                        PubmedArticle(**a)
                        for a in data.get("pubmed_articles", [])
                    ],
                    total_articles=data.get("total_articles", 0),
                    sources_queried=data.get("sources_queried", []),
                    enriched_utc=data.get("enriched_utc", ""),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "Corrupt enrichment cache for %s (%s)",
                    nct_id,
                    exc,
                )
        return None

    def _write_enrichment_cache(
        self, result: EnrichmentResult
    ) -> None:
        """Persist enrichment result to cache."""
        path = self._enrichment_cache_path(result.nct_id)
        data = {
            "nct_id": result.nct_id,
            "gdelt_articles": [
                dataclasses.asdict(a) for a in result.gdelt_articles
            ],
            "newsdata_articles": [
                dataclasses.asdict(a) for a in result.newsdata_articles
            ],
            "pubmed_articles": [
                dataclasses.asdict(a) for a in result.pubmed_articles
            ],
            "total_articles": result.total_articles,
            "sources_queried": result.sources_queried,
            "enriched_utc": result.enriched_utc,
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def enrich_trial(
        self,
        nct_id: str,
        keywords: Optional[list[str]] = None,
        sources: Optional[set[str]] = None,
        *,
        skip_cache: bool = False,
    ) -> EnrichmentResult:
        """Enrich a single trial with literature and newswire data.

        Args:
            nct_id: ClinicalTrials.gov identifier.
            keywords: Optional search keywords (drug name, indication).
            sources: Set of sources to query. Defaults to all
                available adapters.
            skip_cache: Force fresh queries.

        Returns:
            EnrichmentResult with aggregated articles.
        """
        if not skip_cache:
            cached = self._read_enrichment_cache(nct_id)
            if cached is not None:
                logger.debug("Enrichment cache hit for %s", nct_id)
                return cached

        if sources is None:
            sources = set(VALID_SOURCES)

        kw = keywords or []
        sources_queried: list[str] = []
        gdelt_articles: list[NewswireArticle] = []
        newsdata_articles: list[NewswireArticle] = []
        pubmed_articles: list[PubmedArticle] = []

        # Priority 1: PubMed (highest signal — journal articles)
        if "pubmed" in sources and self._pubmed is not None:
            sources_queried.append("pubmed")
            pm_result = self._pubmed.search_by_nct_id(
                nct_id, skip_cache=skip_cache
            )
            pubmed_articles = list(pm_result.articles)
        elif "pubmed" in sources:
            sources_queried.append("pubmed")
            logger.debug(
                "PubMed adapter not configured, skipping for %s",
                nct_id,
            )

        # Priority 2: GDELT
        if "gdelt" in sources and self._gdelt is not None:
            sources_queried.append("gdelt")
            gdelt_result = self._gdelt.search_by_nct_id(
                nct_id, skip_cache=skip_cache
            )
            gdelt_articles = list(gdelt_result.articles)

            # Also search by keywords if provided
            if kw:
                kw_result = self._gdelt.search_by_keywords(
                    nct_id, kw, skip_cache=skip_cache
                )
                # Merge deduplicated
                seen_urls = {a.url for a in gdelt_articles}
                for a in kw_result.articles:
                    if a.url not in seen_urls:
                        gdelt_articles.append(a)
                        seen_urls.add(a.url)

        # Priority 3: NewsData.io
        if "newsdata" in sources and self._newsdata is not None:
            sources_queried.append("newsdata")
            nd_result = self._newsdata.search(
                nct_id, kw, skip_cache=skip_cache
            )
            # Deduplicate against GDELT results
            newsdata_articles = (
                NewsdataAdapter.deduplicate_with_gdelt(
                    nd_result.articles, gdelt_articles
                )
            )

        result = EnrichmentResult(
            nct_id=nct_id,
            gdelt_articles=gdelt_articles,
            newsdata_articles=newsdata_articles,
            pubmed_articles=pubmed_articles,
            sources_queried=sources_queried,
        )

        self._write_enrichment_cache(result)
        logger.info("Enrichment for %s: %s", nct_id, result.summary())
        return result

    def enrich_batch(
        self,
        nct_ids: list[str],
        sources: Optional[set[str]] = None,
    ) -> dict[str, EnrichmentResult]:
        """Enrich multiple trials.

        Args:
            nct_ids: List of NCT IDs.
            sources: Set of sources to query.

        Returns:
            Dict mapping NCT ID to EnrichmentResult.
        """
        results: dict[str, EnrichmentResult] = {}
        for nct_id in nct_ids:
            results[nct_id] = self.enrich_trial(
                nct_id, sources=sources
            )
        return results
