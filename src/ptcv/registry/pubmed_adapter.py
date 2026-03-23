"""PubMed E-utilities Journal Article Adapter.

PTCV-205: Queries PubMed E-utilities to cross-reference clinical trials
with published journal articles via the NCT ID secondary identifier [SI]
field.  Returns structured article metadata (title, authors, journal,
abstract, DOI, PMCID, MeSH terms) and extracts endpoint results from
abstracts where present.

PubMed E-utilities API:
- esearch: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
- efetch:  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
- Auth: API key (free, 10 req/sec with key, 3/sec without)
- Output: XML with full article metadata

Risk tier: LOW — read-only PubMed queries (no patient data).

Regulatory requirements:
- Audit trail: every fetch logged (21 CFR 11.10(e), ALCOA+)
- No PHI: public journal article data only
"""

import dataclasses
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..compliance.audit import AuditAction, AuditLogger

logger = logging.getLogger(__name__)

_ESEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
_EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)
_DEFAULT_CACHE_DIR = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials/registry_cache/pubmed"
)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRYABLE_STATUS_CODES = {429, 503}
_DEFAULT_RATE_LIMIT = 10  # requests per second with API key
_DEFAULT_RATE_LIMIT_NO_KEY = 3  # without API key
_NCBI_TOOL = "ptcv_pipeline"
_NCBI_EMAIL = "ptcv@littlehub.io"

# Regex patterns for endpoint extraction from abstracts
_ENDPOINT_PATTERNS = [
    # "ORR was 45%" or "overall response rate (ORR) was 45%"
    re.compile(
        r"(?:overall\s+response\s+rate\s*(?:\(ORR\)\s*)?|ORR\s+)"
        r"(?:was|of|=)\s*(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    # "median PFS was 12.5 months"
    re.compile(
        r"median\s+(?:progression[- ]free\s+survival"
        r"\s*(?:\(PFS\)\s*)?|PFS\s+)"
        r"(?:was|of|=)\s*(\d+(?:\.\d+)?)\s*months?",
        re.IGNORECASE,
    ),
    # "median OS was 24 months"
    re.compile(
        r"median\s+(?:overall\s+survival"
        r"\s*(?:\(OS\)\s*)?|OS\s+)"
        r"(?:was|of|=)\s*(\d+(?:\.\d+)?)\s*months?",
        re.IGNORECASE,
    ),
    # "CR was 15%" or "complete response rate of 15%"
    re.compile(
        r"(?:complete\s+response(?:\s+rate)?"
        r"\s*(?:\(CR\)\s*)?|CR\s+)"
        r"(?:was|of|=)\s*(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    # "hazard ratio 0.75"
    re.compile(
        r"hazard\s+ratio\s*(?:\(HR\))?\s*"
        r"(?:was|of|=|,)?\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    # "p = 0.001" or "p < 0.05"
    re.compile(
        r"(?:p\s*(?:value)?\s*[=<>]\s*(\d+(?:\.\d+)?))",
        re.IGNORECASE,
    ),
]


@dataclasses.dataclass
class EndpointResult:
    """Structured endpoint data extracted from an abstract.

    Attributes:
        endpoint_type: Type of endpoint (e.g. "ORR", "PFS", "OS").
        value: Numeric value extracted.
        unit: Unit of measurement (e.g. "%", "months").
        raw_text: Original text snippet the extraction came from.
    """

    endpoint_type: str
    value: str
    unit: str
    raw_text: str


@dataclasses.dataclass
class PubmedArticle:
    """A single PubMed journal article.

    Attributes:
        pmid: PubMed identifier.
        title: Article title.
        authors: Author list (formatted as "Last FM").
        journal: Journal name.
        pub_date: Publication date string (YYYY-MM or YYYY).
        abstract: Full abstract text.
        doi: Digital Object Identifier (empty if unavailable).
        pmcid: PubMed Central ID (empty if unavailable).
        mesh_terms: MeSH descriptor terms.
        endpoint_results: Endpoint data extracted from abstract.
        fetched_utc: ISO 8601 timestamp of when fetched.
    """

    pmid: str
    title: str
    authors: list[str] = dataclasses.field(default_factory=list)
    journal: str = ""
    pub_date: str = ""
    abstract: str = ""
    doi: str = ""
    pmcid: str = ""
    mesh_terms: list[str] = dataclasses.field(default_factory=list)
    publication_types: list[str] = dataclasses.field(
        default_factory=list
    )
    accession_numbers: list[str] = dataclasses.field(
        default_factory=list
    )
    endpoint_results: list[EndpointResult] = dataclasses.field(
        default_factory=list
    )
    fetched_utc: str = ""

    def __post_init__(self) -> None:
        if not self.fetched_utc:
            self.fetched_utc = datetime.now(timezone.utc).isoformat()

    @property
    def pubmed_url(self) -> str:
        """Return the PubMed URL for this article."""
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@dataclasses.dataclass
class PubmedSearchResult:
    """Aggregate result from a PubMed search for one NCT ID.

    Attributes:
        nct_id: The trial identifier queried.
        articles: Matching PubMed articles.
        pmid_count: Number of PubMed IDs found via esearch.
        fetched_utc: Timestamp of the search.
        from_cache: Whether result was loaded from disk cache.
    """

    nct_id: str
    articles: list[PubmedArticle] = dataclasses.field(
        default_factory=list
    )
    pmid_count: int = 0
    fetched_utc: str = ""
    from_cache: bool = False

    def __post_init__(self) -> None:
        if not self.fetched_utc:
            self.fetched_utc = datetime.now(timezone.utc).isoformat()


def extract_endpoints(abstract: str) -> list[EndpointResult]:
    """Extract structured endpoint results from abstract text.

    Uses regex patterns to identify reported clinical outcomes
    such as ORR, PFS, OS, CR, hazard ratios, and p-values.

    Args:
        abstract: Full abstract text.

    Returns:
        List of extracted endpoint results.
    """
    if not abstract:
        return []

    results: list[EndpointResult] = []
    type_map = {
        0: ("ORR", "%"),
        1: ("PFS", "months"),
        2: ("OS", "months"),
        3: ("CR", "%"),
        4: ("HR", "ratio"),
        5: ("p-value", ""),
    }

    for i, pattern in enumerate(_ENDPOINT_PATTERNS):
        for match in pattern.finditer(abstract):
            etype, unit = type_map[i]
            # Get surrounding context (up to 80 chars around match)
            start = max(0, match.start() - 20)
            end = min(len(abstract), match.end() + 20)
            raw = abstract[start:end].strip()

            results.append(
                EndpointResult(
                    endpoint_type=etype,
                    value=match.group(1),
                    unit=unit,
                    raw_text=raw,
                )
            )

    return results


class PubmedAdapter:
    """Query PubMed E-utilities for journal articles linked to trials.

    Follows the same fetch/cache pattern as ``GdeltAdapter``
    (PTCV-203) and ``RegistryMetadataFetcher`` (PTCV-194).

    Usage::

        adapter = PubmedAdapter(api_key="your_ncbi_key")
        result = adapter.search_by_nct_id("NCT01512251")
        for article in result.articles:
            print(article.title, article.doi)

    Args:
        api_key: NCBI API key (free, increases rate to 10 req/sec).
            If None, rate limit is 3 req/sec.
        cache_dir: Directory for cached JSON responses.
        audit_logger: AuditLogger instance.  Uses default if None.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._audit = audit_logger or AuditLogger(
            module="pubmed_adapter"
        )
        self._timeout = timeout
        self._last_request_time: float = 0.0
        self._rate_limit = (
            _DEFAULT_RATE_LIMIT
            if api_key
            else _DEFAULT_RATE_LIMIT_NO_KEY
        )

    def _enforce_rate_limit(self) -> None:
        """Enforce minimum delay between API requests."""
        min_interval = 1.0 / self._rate_limit
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _cache_path(self, nct_id: str) -> Path:
        """Return the cache file path for an NCT ID."""
        return self._cache_dir / f"{nct_id}_pubmed.json"

    def _read_cache(
        self, nct_id: str
    ) -> Optional[PubmedSearchResult]:
        """Return cached search result if present, else None."""
        path = self._cache_path(nct_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                articles = []
                for a in data.get("articles", []):
                    endpoints = [
                        EndpointResult(**ep)
                        for ep in a.pop("endpoint_results", [])
                    ]
                    articles.append(
                        PubmedArticle(
                            **a, endpoint_results=endpoints
                        )
                    )
                return PubmedSearchResult(
                    nct_id=data["nct_id"],
                    articles=articles,
                    pmid_count=data.get(
                        "pmid_count", len(articles)
                    ),
                    fetched_utc=data.get("fetched_utc", ""),
                    from_cache=True,
                )
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ) as exc:
                logger.warning(
                    "Corrupt PubMed cache for %s (%s), "
                    "will re-fetch",
                    nct_id,
                    exc,
                )
        return None

    def _write_cache(
        self, nct_id: str, result: PubmedSearchResult
    ) -> None:
        """Persist search result to disk cache."""
        path = self._cache_path(nct_id)
        data = {
            "nct_id": result.nct_id,
            "articles": [
                dataclasses.asdict(a) for a in result.articles
            ],
            "pmid_count": result.pmid_count,
            "fetched_utc": result.fetched_utc,
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _build_params(
        self, extra: dict[str, str]
    ) -> dict[str, str]:
        """Build common NCBI E-utilities parameters."""
        params: dict[str, str] = {
            "tool": _NCBI_TOOL,
            "email": _NCBI_EMAIL,
        }
        if self._api_key:
            params["api_key"] = self._api_key
        params.update(extra)
        return params

    def _get_xml(self, url: str) -> ET.Element:
        """GET a URL and return parsed XML root with retry."""
        req = urllib.request.Request(url)
        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(
                    req, timeout=self._timeout
                ) as resp:
                    xml_bytes = resp.read()
                    return ET.fromstring(xml_bytes)
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUS_CODES:
                    delay = _RETRY_BASE_DELAY * (
                        2 ** (attempt - 1)
                    )
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

    def _esearch(self, nct_id: str) -> list[str]:
        """Query esearch for PubMed IDs linked to an NCT ID.

        Uses the [SI] (secondary identifier) field to find
        articles that cite the NCT ID.

        Args:
            nct_id: ClinicalTrials.gov identifier.

        Returns:
            List of PubMed ID strings.
        """
        params = self._build_params({
            "db": "pubmed",
            "term": f"{nct_id}[si]",
            "retmode": "xml",
            "retmax": "100",
        })
        url = f"{_ESEARCH_URL}?{urllib.parse.urlencode(params)}"

        self._enforce_rate_limit()
        root = self._get_xml(url)

        pmids: list[str] = []
        id_list = root.find("IdList")
        if id_list is not None:
            for id_elem in id_list.findall("Id"):
                if id_elem.text:
                    pmids.append(id_elem.text.strip())

        return pmids

    def _efetch(self, pmids: list[str]) -> list[PubmedArticle]:
        """Fetch article details for a list of PubMed IDs.

        Args:
            pmids: PubMed ID strings.

        Returns:
            List of PubmedArticle objects.
        """
        if not pmids:
            return []

        params = self._build_params({
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        })
        url = f"{_EFETCH_URL}?{urllib.parse.urlencode(params)}"

        self._enforce_rate_limit()
        root = self._get_xml(url)

        articles: list[PubmedArticle] = []
        for article_elem in root.findall(
            ".//PubmedArticle"
        ):
            parsed = self._parse_article_xml(article_elem)
            if parsed:
                articles.append(parsed)

        return articles

    @staticmethod
    def _parse_article_xml(
        elem: ET.Element,
    ) -> Optional[PubmedArticle]:
        """Parse a single PubmedArticle XML element."""
        medline = elem.find("MedlineCitation")
        if medline is None:
            return None

        # PMID
        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else ""
        if not pmid:
            return None

        article = medline.find("Article")
        if article is None:
            return PubmedArticle(pmid=pmid, title="")

        # Title
        title_elem = article.find("ArticleTitle")
        title = title_elem.text or "" if title_elem is not None else ""

        # Journal
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text or "" if journal_elem is not None else ""

        # Authors
        authors: list[str] = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")
                if last:
                    name = f"{last} {initials}" if initials else last
                    authors.append(name)

        # Abstract
        abstract_parts: list[str] = []
        abstract_elem = article.find("Abstract")
        if abstract_elem is not None:
            for text_elem in abstract_elem.findall(
                "AbstractText"
            ):
                label = text_elem.get("Label", "")
                text = (
                    "".join(text_elem.itertext()).strip()
                )
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        # Publication date
        pub_date = ""
        date_elem = article.find(
            ".//Journal/JournalIssue/PubDate"
        )
        if date_elem is not None:
            year = date_elem.findtext("Year", "")
            month = date_elem.findtext("Month", "")
            if year:
                pub_date = f"{year}-{month}" if month else year

        # DOI
        doi = ""
        pubmed_data = elem.find("PubmedData")
        if pubmed_data is not None:
            for aid in pubmed_data.findall(
                ".//ArticleId"
            ):
                if aid.get("IdType") == "doi" and aid.text:
                    doi = aid.text.strip()
                    break

        # PMCID
        pmcid = ""
        if pubmed_data is not None:
            for aid in pubmed_data.findall(
                ".//ArticleId"
            ):
                if aid.get("IdType") == "pmc" and aid.text:
                    pmcid = aid.text.strip()
                    break

        # MeSH terms
        mesh_terms: list[str] = []
        mesh_list = medline.find("MeshHeadingList")
        if mesh_list is not None:
            for heading in mesh_list.findall("MeshHeading"):
                descriptor = heading.find("DescriptorName")
                if (
                    descriptor is not None
                    and descriptor.text
                ):
                    mesh_terms.append(descriptor.text)

        # Publication types (PTCV-282)
        publication_types: list[str] = []
        pub_type_list = article.find("PublicationTypeList")
        if pub_type_list is not None:
            for pt in pub_type_list.findall("PublicationType"):
                if pt.text:
                    publication_types.append(pt.text.strip())

        # Accession numbers / DataBank references (PTCV-282)
        accession_numbers: list[str] = []
        databank_list = article.find("DataBankList")
        if databank_list is not None:
            for databank in databank_list.findall("DataBank"):
                acc_list = databank.find(
                    "AccessionNumberList"
                )
                if acc_list is not None:
                    for acc in acc_list.findall(
                        "AccessionNumber"
                    ):
                        if acc.text:
                            accession_numbers.append(
                                acc.text.strip()
                            )

        # Extract endpoints from abstract
        endpoints = extract_endpoints(abstract)

        return PubmedArticle(
            pmid=pmid,
            title=title,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            abstract=abstract,
            doi=doi,
            pmcid=pmcid,
            mesh_terms=mesh_terms,
            publication_types=publication_types,
            accession_numbers=accession_numbers,
            endpoint_results=endpoints,
        )

    def search_by_nct_id(
        self,
        nct_id: str,
        *,
        skip_cache: bool = False,
    ) -> PubmedSearchResult:
        """Search PubMed for articles linked to an NCT ID.

        Queries esearch with ``NCT_ID[si]`` to find articles that
        registered the NCT ID as a secondary identifier, then
        fetches full article details via efetch.

        Args:
            nct_id: ClinicalTrials.gov identifier (e.g.
                ``NCT01512251``).
            skip_cache: Force a fresh API query.

        Returns:
            PubmedSearchResult with matched articles.
        """
        if not skip_cache:
            cached = self._read_cache(nct_id)
            if cached is not None:
                logger.debug("PubMed cache hit for %s", nct_id)
                return cached

        try:
            pmids = self._esearch(nct_id)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            OSError,
            ET.ParseError,
        ) as exc:
            logger.warning(
                "PubMed esearch failed for %s: %s",
                nct_id,
                exc,
            )
            return PubmedSearchResult(nct_id=nct_id)

        if not pmids:
            logger.info(
                "No PubMed articles linked to %s via [SI] — "
                "candidate for NLP-based linking",
                nct_id,
            )
            result = PubmedSearchResult(
                nct_id=nct_id, pmid_count=0
            )
            self._write_cache(nct_id, result)
            return result

        try:
            articles = self._efetch(pmids)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            OSError,
            ET.ParseError,
        ) as exc:
            logger.warning(
                "PubMed efetch failed for %s (PMIDs: %s): %s",
                nct_id,
                pmids,
                exc,
            )
            return PubmedSearchResult(
                nct_id=nct_id, pmid_count=len(pmids)
            )

        result = PubmedSearchResult(
            nct_id=nct_id,
            articles=articles,
            pmid_count=len(pmids),
        )

        self._write_cache(nct_id, result)
        self._audit.log(
            action=AuditAction.SEARCH,
            record_id=nct_id,
            user_id="ptcv_pipeline",
            reason=(
                "PubMed E-utilities journal article "
                "cross-reference (PTCV-205)"
            ),
            after={
                "pmids_found": len(pmids),
                "articles_fetched": len(articles),
                "pmids": pmids,
            },
        )
        logger.info(
            "PubMed search for %s: %d articles found",
            nct_id,
            len(articles),
        )
        return result

    def batch_search(
        self, nct_ids: list[str]
    ) -> dict[str, PubmedSearchResult]:
        """Search PubMed for multiple NCT IDs.

        Args:
            nct_ids: List of NCT IDs to search.

        Returns:
            Dict mapping each NCT ID to its search result.
        """
        results: dict[str, PubmedSearchResult] = {}
        for nct_id in nct_ids:
            results[nct_id] = self.search_by_nct_id(nct_id)
        return results
