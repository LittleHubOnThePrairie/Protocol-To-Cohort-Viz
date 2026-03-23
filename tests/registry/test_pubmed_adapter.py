"""Tests for PubMed E-utilities adapter (PTCV-205).

Covers all GHERKIN scenarios:
- Fetch journal articles by NCT ID secondary identifier
- Extract structured endpoint data from abstracts
- Handle trials with no linked publications
- Respect API rate limits with key

Qualification phase: OQ (operational qualification)
Risk tier: LOW
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ptcv.registry.pubmed_adapter import (
    EndpointResult,
    PubmedAdapter,
    PubmedArticle,
    PubmedSearchResult,
    extract_endpoints,
)


# ---------------------------------------------------------------------------
# XML fixtures
# ---------------------------------------------------------------------------

_ESEARCH_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<eSearchResult>
  <Count>2</Count>
  <RetMax>2</RetMax>
  <IdList>
    <Id>25678901</Id>
    <Id>26789012</Id>
  </IdList>
</eSearchResult>
"""

_ESEARCH_EMPTY_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<eSearchResult>
  <Count>0</Count>
  <RetMax>0</RetMax>
  <IdList/>
</eSearchResult>
"""

_EFETCH_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>25678901</PMID>
      <Article>
        <ArticleTitle>BKM120 plus vemurafenib in melanoma</ArticleTitle>
        <Journal>
          <Title>Journal of Clinical Oncology</Title>
          <JournalIssue>
            <PubDate>
              <Year>2015</Year>
              <Month>Jun</Month>
            </PubDate>
          </JournalIssue>
        </Journal>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <Initials>AB</Initials>
          </Author>
          <Author>
            <LastName>Jones</LastName>
            <Initials>CD</Initials>
          </Author>
        </AuthorList>
        <Abstract>
          <AbstractText Label="RESULTS">The overall response rate (ORR) was 45.2% and median progression-free survival (PFS) was 12.5 months. The hazard ratio was 0.72 (p = 0.003).</AbstractText>
        </Abstract>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName>Melanoma</DescriptorName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName>PI3K Inhibitors</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1200/JCO.2015.12345</ArticleId>
        <ArticleId IdType="pmc">PMC4567890</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>26789012</PMID>
      <Article>
        <ArticleTitle>Phase 2 results of BKM120 in BRAF mutant melanoma</ArticleTitle>
        <Journal>
          <Title>New England Journal of Medicine</Title>
          <JournalIssue>
            <PubDate>
              <Year>2016</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <AuthorList>
          <Author>
            <LastName>Johnson</LastName>
            <Initials>EF</Initials>
          </Author>
        </AuthorList>
        <Abstract>
          <AbstractText>Median overall survival (OS) was 24 months with a complete response rate of 15%.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1056/NEJMoa2016789</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


@pytest.fixture
def adapter(tmp_path: Path) -> PubmedAdapter:
    """PubmedAdapter with temp cache and API key."""
    return PubmedAdapter(
        api_key="test_key_123",
        cache_dir=tmp_path / "pubmed_cache",
    )


@pytest.fixture
def adapter_no_key(tmp_path: Path) -> PubmedAdapter:
    """PubmedAdapter without API key."""
    return PubmedAdapter(
        api_key=None,
        cache_dir=tmp_path / "pubmed_cache",
    )


# ---------------------------------------------------------------------------
# Scenario 1: Fetch journal articles by NCT ID secondary identifier
# ---------------------------------------------------------------------------


class TestFetchByNctId:
    """Scenario: Fetch journal articles by NCT ID [SI] field."""

    def test_esearch_returns_pmids(
        self, adapter: PubmedAdapter
    ) -> None:
        """esearch with NCT ID in [SI] returns matching PMIDs."""
        esearch_root = ET.fromstring(_ESEARCH_XML)
        efetch_root = ET.fromstring(_EFETCH_XML)

        with patch.object(
            adapter, "_get_xml", side_effect=[esearch_root, efetch_root]
        ):
            result = adapter.search_by_nct_id("NCT01512251")

        assert result.pmid_count == 2
        assert len(result.articles) == 2

    def test_article_metadata_extracted(
        self, adapter: PubmedAdapter
    ) -> None:
        """efetch retrieves title, authors, journal, DOI, PMCID."""
        esearch_root = ET.fromstring(_ESEARCH_XML)
        efetch_root = ET.fromstring(_EFETCH_XML)

        with patch.object(
            adapter, "_get_xml", side_effect=[esearch_root, efetch_root]
        ):
            result = adapter.search_by_nct_id("NCT01512251")

        art = result.articles[0]
        assert art.pmid == "25678901"
        assert "BKM120" in art.title
        assert art.journal == "Journal of Clinical Oncology"
        assert art.authors == ["Smith AB", "Jones CD"]
        assert art.doi == "10.1200/JCO.2015.12345"
        assert art.pmcid == "PMC4567890"
        assert art.pub_date == "2015-Jun"

    def test_mesh_terms_extracted(
        self, adapter: PubmedAdapter
    ) -> None:
        """MeSH descriptor terms are parsed from XML."""
        esearch_root = ET.fromstring(_ESEARCH_XML)
        efetch_root = ET.fromstring(_EFETCH_XML)

        with patch.object(
            adapter, "_get_xml", side_effect=[esearch_root, efetch_root]
        ):
            result = adapter.search_by_nct_id("NCT01512251")

        art = result.articles[0]
        assert "Melanoma" in art.mesh_terms
        assert "PI3K Inhibitors" in art.mesh_terms

    def test_pubmed_url_property(self) -> None:
        """PubmedArticle.pubmed_url returns correct URL."""
        art = PubmedArticle(pmid="12345678", title="Test")
        assert art.pubmed_url == (
            "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        )

    def test_result_cached_on_disk(
        self, adapter: PubmedAdapter
    ) -> None:
        """Search result is written to cache and read on repeat."""
        esearch_root = ET.fromstring(_ESEARCH_XML)
        efetch_root = ET.fromstring(_EFETCH_XML)

        with patch.object(
            adapter, "_get_xml", side_effect=[esearch_root, efetch_root]
        ) as mock_get:
            result1 = adapter.search_by_nct_id("NCT01512251")

        # Second call should hit cache
        result2 = adapter.search_by_nct_id("NCT01512251")

        assert result2.from_cache is True
        assert result2.pmid_count == result1.pmid_count
        assert len(result2.articles) == len(result1.articles)

    def test_second_article_parsed(
        self, adapter: PubmedAdapter
    ) -> None:
        """Both articles in the efetch response are parsed."""
        esearch_root = ET.fromstring(_ESEARCH_XML)
        efetch_root = ET.fromstring(_EFETCH_XML)

        with patch.object(
            adapter, "_get_xml", side_effect=[esearch_root, efetch_root]
        ):
            result = adapter.search_by_nct_id("NCT01512251")

        art2 = result.articles[1]
        assert art2.pmid == "26789012"
        assert art2.journal == "New England Journal of Medicine"
        assert art2.authors == ["Johnson EF"]
        assert art2.doi == "10.1056/NEJMoa2016789"
        assert art2.pmcid == ""  # No PMC ID in second article


# ---------------------------------------------------------------------------
# Scenario 2: Extract structured endpoint data from abstracts
# ---------------------------------------------------------------------------


class TestExtractEndpoints:
    """Scenario: Extract structured endpoint data from abstracts."""

    def test_orr_extraction(self) -> None:
        """ORR percentage extracted from abstract."""
        text = "The overall response rate (ORR) was 45.2%."
        endpoints = extract_endpoints(text)
        orr = [e for e in endpoints if e.endpoint_type == "ORR"]
        assert len(orr) == 1
        assert orr[0].value == "45.2"
        assert orr[0].unit == "%"

    def test_pfs_extraction(self) -> None:
        """Median PFS in months extracted."""
        text = "Median progression-free survival was 12.5 months."
        endpoints = extract_endpoints(text)
        pfs = [e for e in endpoints if e.endpoint_type == "PFS"]
        assert len(pfs) == 1
        assert pfs[0].value == "12.5"
        assert pfs[0].unit == "months"

    def test_os_extraction(self) -> None:
        """Median OS in months extracted."""
        text = "Median overall survival was 24 months."
        endpoints = extract_endpoints(text)
        os_ep = [e for e in endpoints if e.endpoint_type == "OS"]
        assert len(os_ep) == 1
        assert os_ep[0].value == "24"

    def test_hazard_ratio_extraction(self) -> None:
        """Hazard ratio extracted."""
        text = "The hazard ratio was 0.72."
        endpoints = extract_endpoints(text)
        hr = [e for e in endpoints if e.endpoint_type == "HR"]
        assert len(hr) == 1
        assert hr[0].value == "0.72"
        assert hr[0].unit == "ratio"

    def test_p_value_extraction(self) -> None:
        """P-value extracted."""
        text = "The difference was significant (p = 0.003)."
        endpoints = extract_endpoints(text)
        pv = [e for e in endpoints if e.endpoint_type == "p-value"]
        assert len(pv) == 1
        assert pv[0].value == "0.003"

    def test_complete_response_extraction(self) -> None:
        """Complete response rate extracted."""
        text = "Complete response rate of 15%."
        endpoints = extract_endpoints(text)
        cr = [e for e in endpoints if e.endpoint_type == "CR"]
        assert len(cr) == 1
        assert cr[0].value == "15"

    def test_multiple_endpoints_from_abstract(self) -> None:
        """Multiple endpoints extracted from a single abstract."""
        text = (
            "ORR was 45% and median PFS was 12 months. "
            "Hazard ratio was 0.65 (p = 0.001)."
        )
        endpoints = extract_endpoints(text)
        types = {e.endpoint_type for e in endpoints}
        assert types >= {"ORR", "PFS", "HR", "p-value"}

    def test_empty_abstract_returns_empty(self) -> None:
        """Empty abstract returns no endpoints."""
        assert extract_endpoints("") == []

    def test_no_endpoints_returns_empty(self) -> None:
        """Abstract without endpoint data returns empty list."""
        text = "This study enrolled patients with melanoma."
        assert extract_endpoints(text) == []

    def test_endpoints_stored_in_article(
        self, adapter: PubmedAdapter
    ) -> None:
        """Endpoint extraction runs during article parsing."""
        esearch_root = ET.fromstring(_ESEARCH_XML)
        efetch_root = ET.fromstring(_EFETCH_XML)

        with patch.object(
            adapter, "_get_xml", side_effect=[esearch_root, efetch_root]
        ):
            result = adapter.search_by_nct_id("NCT01512251")

        art = result.articles[0]
        types = {e.endpoint_type for e in art.endpoint_results}
        assert "ORR" in types
        assert "PFS" in types
        assert "HR" in types


# ---------------------------------------------------------------------------
# Scenario 3: Handle trials with no linked publications
# ---------------------------------------------------------------------------


class TestNoLinkedPublications:
    """Scenario: Handle trials with no linked PubMed articles."""

    def test_empty_esearch_returns_empty_result(
        self, adapter: PubmedAdapter
    ) -> None:
        """No PMIDs found → empty result without error."""
        esearch_root = ET.fromstring(_ESEARCH_EMPTY_XML)

        with patch.object(
            adapter, "_get_xml", return_value=esearch_root
        ):
            result = adapter.search_by_nct_id("NCT99999999")

        assert result.pmid_count == 0
        assert result.articles == []
        assert result.nct_id == "NCT99999999"

    def test_no_error_raised_on_empty(
        self, adapter: PubmedAdapter
    ) -> None:
        """Empty result does not raise exceptions."""
        esearch_root = ET.fromstring(_ESEARCH_EMPTY_XML)

        with patch.object(
            adapter, "_get_xml", return_value=esearch_root
        ):
            # Should not raise
            result = adapter.search_by_nct_id("NCT99999999")
            assert isinstance(result, PubmedSearchResult)

    def test_empty_result_cached(
        self, adapter: PubmedAdapter
    ) -> None:
        """Empty results are cached to avoid repeated API calls."""
        esearch_root = ET.fromstring(_ESEARCH_EMPTY_XML)

        with patch.object(
            adapter, "_get_xml", return_value=esearch_root
        ) as mock_get:
            adapter.search_by_nct_id("NCT99999999")

        # Second call should hit cache
        result2 = adapter.search_by_nct_id("NCT99999999")
        assert result2.from_cache is True

    def test_esearch_failure_returns_empty(
        self, adapter: PubmedAdapter
    ) -> None:
        """API error during esearch → empty result, no crash."""
        import urllib.error

        with patch.object(
            adapter,
            "_get_xml",
            side_effect=urllib.error.URLError("timeout"),
        ):
            result = adapter.search_by_nct_id("NCT01512251")

        assert result.pmid_count == 0
        assert result.articles == []

    def test_efetch_failure_returns_pmid_count(
        self, adapter: PubmedAdapter
    ) -> None:
        """API error during efetch → result with pmid_count but
        no articles."""
        esearch_root = ET.fromstring(_ESEARCH_XML)

        import urllib.error

        with patch.object(
            adapter,
            "_get_xml",
            side_effect=[
                esearch_root,
                urllib.error.URLError("timeout"),
            ],
        ):
            result = adapter.search_by_nct_id("NCT01512251")

        assert result.pmid_count == 2
        assert result.articles == []


# ---------------------------------------------------------------------------
# Scenario 4: Respect API rate limits with key
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Scenario: Respect API rate limits with key."""

    def test_rate_limit_10_with_key(
        self, adapter: PubmedAdapter
    ) -> None:
        """Adapter with API key has 10 req/sec rate limit."""
        assert adapter._rate_limit == 10

    def test_rate_limit_3_without_key(
        self, adapter_no_key: PubmedAdapter
    ) -> None:
        """Adapter without API key has 3 req/sec rate limit."""
        assert adapter_no_key._rate_limit == 3

    def test_api_key_included_in_params(
        self, adapter: PubmedAdapter
    ) -> None:
        """API key is included in request parameters."""
        params = adapter._build_params({"db": "pubmed"})
        assert params["api_key"] == "test_key_123"
        assert params["tool"] == "ptcv_pipeline"
        assert params["email"] == "ptcv@littlehub.io"

    def test_no_api_key_excluded_from_params(
        self, adapter_no_key: PubmedAdapter
    ) -> None:
        """Without API key, api_key param is absent."""
        params = adapter_no_key._build_params({"db": "pubmed"})
        assert "api_key" not in params
        assert params["tool"] == "ptcv_pipeline"

    def test_rate_limit_enforced(
        self, adapter: PubmedAdapter
    ) -> None:
        """Rate limiting enforces minimum interval between calls."""
        # Set last request to now
        adapter._last_request_time = time.monotonic()

        start = time.monotonic()
        adapter._enforce_rate_limit()
        elapsed = time.monotonic() - start

        # With 10 req/sec, min interval is 0.1s
        # Allow some tolerance
        assert elapsed >= 0.05


# ---------------------------------------------------------------------------
# Batch search
# ---------------------------------------------------------------------------


class TestBatchSearch:
    """Batch search for multiple NCT IDs."""

    def test_batch_returns_all_ids(
        self, adapter: PubmedAdapter
    ) -> None:
        """batch_search returns a result for every NCT ID."""
        esearch_root = ET.fromstring(_ESEARCH_EMPTY_XML)

        with patch.object(
            adapter, "_get_xml", return_value=esearch_root
        ):
            results = adapter.batch_search(
                ["NCT11111111", "NCT22222222"]
            )

        assert "NCT11111111" in results
        assert "NCT22222222" in results
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Cache serialization roundtrip
# ---------------------------------------------------------------------------


class TestCacheSerialization:
    """Cache write/read preserves all fields."""

    def test_roundtrip_with_endpoints(
        self, adapter: PubmedAdapter
    ) -> None:
        """Articles with endpoints survive cache roundtrip."""
        article = PubmedArticle(
            pmid="12345",
            title="Test Article",
            authors=["Smith AB"],
            journal="Test Journal",
            pub_date="2024",
            abstract="ORR was 50%.",
            doi="10.1234/test",
            pmcid="PMC999",
            mesh_terms=["Oncology"],
            endpoint_results=[
                EndpointResult(
                    endpoint_type="ORR",
                    value="50",
                    unit="%",
                    raw_text="ORR was 50%.",
                ),
            ],
        )
        result = PubmedSearchResult(
            nct_id="NCT00000001",
            articles=[article],
            pmid_count=1,
        )

        adapter._write_cache("NCT00000001", result)
        loaded = adapter._read_cache("NCT00000001")

        assert loaded is not None
        assert loaded.from_cache is True
        assert loaded.pmid_count == 1
        assert len(loaded.articles) == 1
        art = loaded.articles[0]
        assert art.pmid == "12345"
        assert art.doi == "10.1234/test"
        assert len(art.endpoint_results) == 1
        assert art.endpoint_results[0].endpoint_type == "ORR"

    def test_corrupt_cache_returns_none(
        self, adapter: PubmedAdapter
    ) -> None:
        """Corrupt cache file returns None (re-fetches)."""
        cache_path = adapter._cache_path("NCT00000001")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("not valid json", encoding="utf-8")

        result = adapter._read_cache("NCT00000001")
        assert result is None
