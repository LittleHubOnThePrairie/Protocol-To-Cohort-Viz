"""ClinicalTrials.gov Registry Metadata Fetcher.

PTCV-194: Fetches structured protocol metadata from the ClinicalTrials.gov
API v2 by NCT ID.  Returns raw JSON dicts containing ProtocolSection
fields (IdentificationModule, StatusModule, DesignModule, etc.) for
downstream mapping to ICH E6(R3) sections (PTCV-195).

Responses are cached to ``data/protocols/clinicaltrials/registry_cache/``
to avoid redundant API calls.

Uses urllib (matching ClinicalTrialsService conventions) with exponential
backoff retry for 429/503 responses.

Risk tier: LOW — read-only registry queries (no patient data).

Regulatory requirements:
- Audit trail: every fetch logged (21 CFR 11.10(e), ALCOA+)
- No PHI: trial registry data only (no participant identifiers)
"""

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from ..compliance.audit import AuditAction, AuditLogger

logger = logging.getLogger(__name__)

_CT_GOV_V2_BASE = "https://clinicaltrials.gov/api/v2/studies"
_NCT_PATTERN = re.compile(r"^NCT\d{8}$")
_DEFAULT_CACHE_DIR = Path(
    "C:/Dev/PTCV/data/protocols/clinicaltrials/registry_cache"
)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_RETRYABLE_STATUS_CODES = {429, 503}
_DEFAULT_BATCH_DELAY = 0.5  # seconds between batch requests


class RegistryMetadataFetcher:
    """Fetch structured metadata from ClinicalTrials.gov API v2 by NCT ID.

    Queries the ``/studies/{nctId}`` endpoint and returns the full
    ProtocolSection JSON.  Caches responses to disk so repeated
    fetches for the same NCT ID skip the API call.

    Args:
        cache_dir: Directory for cached JSON responses.
        audit_logger: AuditLogger instance.  Uses default if None.
        timeout: HTTP request timeout in seconds.
        batch_delay: Delay in seconds between batch requests.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
        timeout: int = 30,
        batch_delay: float = _DEFAULT_BATCH_DELAY,
    ) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._audit = audit_logger or AuditLogger(
            module="registry_metadata_fetcher"
        )
        self._timeout = timeout
        self._batch_delay = batch_delay

    @staticmethod
    def is_valid_nct_id(nct_id: str) -> bool:
        """Check whether *nct_id* matches the NCT\\d{8} pattern."""
        return bool(_NCT_PATTERN.match(nct_id))

    def _cache_path(self, nct_id: str) -> Path:
        """Return the cache file path for an NCT ID."""
        return self._cache_dir / f"{nct_id}.json"

    def _read_cache(self, nct_id: str) -> Optional[dict[str, Any]]:
        """Return cached metadata if present, else None."""
        path = self._cache_path(nct_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Corrupt cache for %s, will re-fetch", nct_id
                )
        return None

    def _write_cache(
        self, nct_id: str, data: dict[str, Any]
    ) -> None:
        """Persist metadata JSON to the cache directory."""
        path = self._cache_path(nct_id)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _get_json(self, url: str) -> dict[str, Any]:
        """GET a URL and return parsed JSON with retry on 429/503.

        Retries up to ``_MAX_RETRIES`` times with exponential backoff
        for rate-limit (429) and service-unavailable (503) responses.

        Raises:
            urllib.error.HTTPError: For non-retryable HTTP errors.
            urllib.error.URLError: For network connectivity failures
                after retry exhaustion.
        """
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(
                    req, timeout=self._timeout
                ) as resp:
                    return json.loads(resp.read().decode("utf-8"))  # type: ignore[no-any-return]
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
        raise RuntimeError("Unreachable: retry loop exited without result")

    def fetch(self, nct_id: str) -> Optional[dict[str, Any]]:
        """Fetch protocol metadata for a single NCT ID.

        Returns the full study JSON from the ClinicalTrials.gov v2
        ``/studies/{nctId}`` endpoint, or ``None`` if the trial is
        not found, withdrawn, or the API is unreachable after retries.

        Cached responses are returned without an API call.

        Args:
            nct_id: ClinicalTrials.gov identifier (e.g. ``NCT01512251``).

        Returns:
            Parsed JSON dict or None on failure.
        """
        if not self.is_valid_nct_id(nct_id):
            logger.warning("Invalid NCT ID format: %s", nct_id)
            return None

        cached = self._read_cache(nct_id)
        if cached is not None:
            logger.debug("Cache hit for %s", nct_id)
            return cached

        url = f"{_CT_GOV_V2_BASE}/{nct_id}"
        try:
            data = self._get_json(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                logger.warning(
                    "NCT ID %s not found (HTTP 404)", nct_id
                )
            else:
                logger.warning(
                    "HTTP %d fetching %s: %s",
                    exc.code,
                    nct_id,
                    exc.reason,
                )
            return None
        except (urllib.error.URLError, OSError) as exc:
            logger.warning(
                "Network error fetching %s: %s", nct_id, exc
            )
            return None

        self._write_cache(nct_id, data)
        self._audit.log(
            action=AuditAction.DOWNLOAD,
            record_id=nct_id,
            user_id="ptcv_pipeline",
            reason=(
                "Registry metadata fetch from ClinicalTrials.gov "
                "API v2 for RAG pre-population (PTCV-194)"
            ),
            after={"modules": list(data.get("protocolSection", {}).keys())},
        )
        logger.info("Fetched registry metadata for %s", nct_id)
        return data

    def batch_fetch(
        self,
        nct_ids: list[str],
    ) -> dict[str, Optional[dict[str, Any]]]:
        """Fetch metadata for multiple NCT IDs with rate-limit delay.

        Args:
            nct_ids: List of NCT IDs to fetch.

        Returns:
            Dict mapping each NCT ID to its metadata (or None on
            failure).  Order matches input.
        """
        results: dict[str, Optional[dict[str, Any]]] = {}
        for i, nct_id in enumerate(nct_ids):
            results[nct_id] = self.fetch(nct_id)
            if i < len(nct_ids) - 1:
                time.sleep(self._batch_delay)
        return results

    @staticmethod
    def extract_nct_id(text: str) -> Optional[str]:
        """Extract the first NCT ID from arbitrary text.

        Useful for pulling the NCT ID from filenames or protocol
        metadata strings.

        Args:
            text: String that may contain an NCT ID.

        Returns:
            The first NCT ID found, or None.
        """
        match = re.search(r"NCT\d{8}", text)
        return match.group(0) if match else None
