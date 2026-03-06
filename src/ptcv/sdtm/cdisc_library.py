"""CDISC Library REST API client for controlled terminology validation (PTCV-58).

Provides authoritative CT validation, biomedical concept lookup, and
codelist term retrieval from the CDISC Library API. Falls back to
the local CtNormalizer when the API is unavailable or unconfigured.

API base: https://library.cdisc.org/api
Auth: API key via environment variable CDISC_LIBRARY_API_KEY

Caching: Local SQLite cache with configurable TTL. Controlled terminology
changes infrequently (quarterly releases), so a 7-day default TTL is safe.

Risk tier: MEDIUM — data pipeline component; affects regulatory submission.

Regulatory references:
- SDTM v1.7 Section 4.1.2.4: Controlled Terminology requirements
- CDISC NCI EVS subset for SDTM Trial Design domains
- ALCOA+ Traceable: API version/date linked to each CT result
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from .ct_normalizer import CtLookupResult, CtNormalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE = "https://library.cdisc.org/api"
_ENV_KEY = "CDISC_LIBRARY_API_KEY"
_DEFAULT_CACHE_DB = Path("C:/Dev/PTCV/data/sqlite/cdisc_library_cache.db")
_DEFAULT_TTL_HOURS = 168  # 7 days
_REQUEST_TIMEOUT_SECONDS = 15


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CodelistTerm:
    """One term from a CDISC CT codelist.

    Attributes:
        code: NCI C-code (e.g. "C15962").
        submission_value: CDISC submission value (e.g. "PHASE III TRIAL").
        preferred_term: NCI preferred term text.
        codelist_code: Parent codelist C-code.
        synonyms: Pipe-separated synonym string from NCI EVS.
    """

    code: str
    submission_value: str
    preferred_term: str
    codelist_code: str
    synonyms: str = ""


@dataclasses.dataclass
class CtValidationResult:
    """Result of validating a CT value against the CDISC Library.

    Attributes:
        original_value: Input value before validation.
        valid: True when the value matches an official codelist term.
        matched_term: The matching CodelistTerm, or None.
        suggestions: Up to 3 closest official terms when invalid.
        source: "api", "cache", or "fallback".
        codelist_name: Name of the codelist checked.
        package_version: CT package version used for validation.
    """

    original_value: str
    valid: bool
    matched_term: Optional[CodelistTerm]
    suggestions: list[CodelistTerm]
    source: str
    codelist_name: str = ""
    package_version: str = ""


@dataclasses.dataclass
class DomainMapping:
    """Result of mapping an assessment to a CDASH/SDTM domain.

    Attributes:
        assessment_name: Original assessment label.
        domain_code: SDTM domain code (e.g. "LB", "VS").
        domain_name: Human-readable domain name.
        testcd_suggestion: Suggested TESTCD value, if available.
        source: "api", "cache", or "fallback".
        confidence: Mapping confidence (1.0 for API, varies for fallback).
    """

    assessment_name: str
    domain_code: str
    domain_name: str
    testcd_suggestion: str = ""
    source: str = "fallback"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

class _CdiscLibraryCache:
    """SQLite-backed cache for CDISC Library API responses."""

    def __init__(
        self, db_path: Path, ttl_hours: int = _DEFAULT_TTL_HOURS,
    ) -> None:
        self._db_path = db_path
        self._ttl_seconds = ttl_hours * 3600

    def initialise(self) -> None:
        """Create cache database and table (idempotent)."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key   TEXT PRIMARY KEY,
                    response    TEXT NOT NULL,
                    fetched_at  REAL NOT NULL
                )
            """)

    def get(self, key: str) -> Optional[Any]:
        """Return cached response if within TTL, else None."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    "SELECT response, fetched_at FROM api_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
        except sqlite3.Error:
            return None

        if row is None:
            return None

        response_json, fetched_at = row
        if time.time() - fetched_at > self._ttl_seconds:
            return None

        try:
            return json.loads(response_json)
        except json.JSONDecodeError:
            return None

    def put(self, key: str, response: Any) -> None:
        """Store API response in cache."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO api_cache (cache_key, response, fetched_at) "
                    "VALUES (?, ?, ?)",
                    (key, json.dumps(response, ensure_ascii=False), time.time()),
                )
        except sqlite3.Error:
            logger.warning("Failed to write CDISC Library cache entry: %s", key)


# ---------------------------------------------------------------------------
# Codelist registry — maps TSPARMCD to CDISC CT codelist identifiers
# ---------------------------------------------------------------------------

_PARMCD_TO_CODELIST: dict[str, str] = {
    "PHASE": "C66737",      # Trial Phase codelist
    "STYPE": "C66739",      # Trial Design codelist
    "TTYPE": "C66738",      # Trial Type codelist
    "BLIND": "C66735",      # Trial Blinding Schema codelist
    "INDIC": "C66770",      # Indication codelist
    "ADDON": "C99074",      # Trial Add-On Design codelist
    "RANDOM": "C66736",     # Randomization Scheme codelist
}

# Assessment keyword → (domain_code, domain_name, testcd)
_BIOMEDICAL_CONCEPTS: dict[str, tuple[str, str, str]] = {
    "complete blood count": ("LB", "Laboratory Test Results", "CBC"),
    "cbc": ("LB", "Laboratory Test Results", "CBC"),
    "chemistry panel": ("LB", "Laboratory Test Results", "CHEM"),
    "urinalysis": ("LB", "Laboratory Test Results", "URIN"),
    "liver function": ("LB", "Laboratory Test Results", "LFT"),
    "renal function": ("LB", "Laboratory Test Results", "RFT"),
    "lipid panel": ("LB", "Laboratory Test Results", "LIPID"),
    "hba1c": ("LB", "Laboratory Test Results", "HBA1C"),
    "hemoglobin": ("LB", "Laboratory Test Results", "HGB"),
    "creatinine": ("LB", "Laboratory Test Results", "CREAT"),
    "glucose": ("LB", "Laboratory Test Results", "GLUC"),
    "vital signs": ("VS", "Vital Signs", ""),
    "blood pressure": ("VS", "Vital Signs", "SYSBP"),
    "heart rate": ("VS", "Vital Signs", "HR"),
    "pulse": ("VS", "Vital Signs", "PULSE"),
    "temperature": ("VS", "Vital Signs", "TEMP"),
    "weight": ("VS", "Vital Signs", "WEIGHT"),
    "height": ("VS", "Vital Signs", "HEIGHT"),
    "bmi": ("VS", "Vital Signs", "BMI"),
    "ecg": ("EG", "ECG Test Results", ""),
    "electrocardiogram": ("EG", "ECG Test Results", ""),
    "12-lead ecg": ("EG", "ECG Test Results", ""),
    "physical examination": ("PE", "Physical Examination", ""),
    "adverse event": ("AE", "Adverse Events", ""),
    "concomitant medication": ("CM", "Concomitant Medications", ""),
    "medical history": ("MH", "Medical History", ""),
    "questionnaire": ("QS", "Questionnaire", ""),
    "pharmacokinetic": ("PC", "Pharmacokinetics Concentrations", ""),
    "pk sampling": ("PC", "Pharmacokinetics Concentrations", ""),
    "tumour assessment": ("TU", "Tumor/Lesion Identification", ""),
    "recist": ("TU", "Tumor/Lesion Identification", ""),
    "informed consent": ("DS", "Disposition", ""),
    "biopsy": ("PR", "Procedures", ""),
    "imaging": ("PR", "Procedures", ""),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class CdiscLibraryClient:
    """Client for the CDISC Library REST API.

    Provides CT validation, codelist term retrieval, and biomedical
    concept lookup. Caches API responses locally in SQLite.

    Args:
        api_key: CDISC Library API key. Falls back to env var
            CDISC_LIBRARY_API_KEY if empty.
        cache_db: Path to the SQLite cache database.
        cache_ttl_hours: Cache entry TTL in hours (default 168 = 7 days).
    """

    def __init__(
        self,
        api_key: str = "",
        cache_db: Optional[Path] = None,
        cache_ttl_hours: int = _DEFAULT_TTL_HOURS,
    ) -> None:
        self._api_key = api_key or os.environ.get(_ENV_KEY, "")
        self._cache = _CdiscLibraryCache(
            db_path=cache_db or _DEFAULT_CACHE_DB,
            ttl_hours=cache_ttl_hours,
        )
        self._cache.initialise()

    @property
    def configured(self) -> bool:
        """True when an API key is available."""
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _api_get(self, path: str) -> Optional[Any]:
        """Make authenticated GET request. Returns None on failure."""
        if not self._api_key:
            return None

        url = f"{_API_BASE}{path}"

        # Check cache first
        cached = self._cache.get(url)
        if cached is not None:
            return cached

        req = urllib.request.Request(
            url,
            headers={
                "api-key": self._api_key,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                req, timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._cache.put(url, data)
                return data
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("CDISC Library API request failed: %s — %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # CT package listing
    # ------------------------------------------------------------------

    def list_ct_packages(self) -> list[str]:
        """List available CT package identifiers.

        Returns:
            List of package IDs (e.g. ["sdtmct-2024-03-29"]).
            Empty list if API unavailable.
        """
        data = self._api_get("/mdr/ct/packages")
        if not data or "_links" not in data:
            return []

        packages = data.get("_links", {}).get("packages", [])
        return [p.get("title", "") for p in packages if p.get("title")]

    # ------------------------------------------------------------------
    # Codelist term retrieval
    # ------------------------------------------------------------------

    def get_codelist_terms(
        self, codelist_code: str, package: str = "sdtmct-2024-03-29",
    ) -> list[CodelistTerm]:
        """Retrieve all terms for a codelist from a CT package.

        Args:
            codelist_code: NCI C-code for the codelist (e.g. "C66737").
            package: CT package identifier.

        Returns:
            List of CodelistTerm objects. Empty if API unavailable.
        """
        data = self._api_get(
            f"/mdr/ct/packages/{package}/codelists/{codelist_code}",
        )
        if not data:
            return []

        terms_data = data.get("terms", [])
        result: list[CodelistTerm] = []
        for term in terms_data:
            result.append(CodelistTerm(
                code=str(term.get("conceptId", "")),
                submission_value=str(term.get("submissionValue", "")),
                preferred_term=str(term.get("preferredTerm", "")),
                codelist_code=codelist_code,
                synonyms=str(term.get("synonyms", "")),
            ))
        return result

    # ------------------------------------------------------------------
    # CT validation
    # ------------------------------------------------------------------

    def validate_ct_value(
        self,
        parmcd: str,
        value: str,
        package: str = "sdtmct-2024-03-29",
    ) -> CtValidationResult:
        """Validate a CT value against the official CDISC codelist.

        Args:
            parmcd: SDTM TSPARMCD (e.g. "PHASE").
            value: Value to validate.
            package: CT package identifier.

        Returns:
            CtValidationResult with validation status and suggestions.
        """
        codelist_code = _PARMCD_TO_CODELIST.get(parmcd.upper(), "")
        if not codelist_code:
            return CtValidationResult(
                original_value=value,
                valid=False,
                matched_term=None,
                suggestions=[],
                source="fallback",
                codelist_name=parmcd,
            )

        terms = self.get_codelist_terms(codelist_code, package)
        source = "api" if terms else "fallback"

        if not terms:
            # API unavailable — cannot validate
            return CtValidationResult(
                original_value=value,
                valid=False,
                matched_term=None,
                suggestions=[],
                source="fallback",
                codelist_name=parmcd,
                package_version=package,
            )

        # Check cache hit (terms came from cache)
        cached = self._cache.get(
            f"{_API_BASE}/mdr/ct/packages/{package}/codelists/{codelist_code}",
        )
        if cached is not None:
            source = "cache"

        # Exact match by submission value or preferred term
        value_upper = value.strip().upper()
        for term in terms:
            if (term.submission_value.upper() == value_upper
                    or term.preferred_term.upper() == value_upper):
                return CtValidationResult(
                    original_value=value,
                    valid=True,
                    matched_term=term,
                    suggestions=[],
                    source=source,
                    codelist_name=parmcd,
                    package_version=package,
                )

        # No exact match — find closest suggestions
        suggestions = self._find_suggestions(value, terms, max_results=3)
        return CtValidationResult(
            original_value=value,
            valid=False,
            matched_term=None,
            suggestions=suggestions,
            source=source,
            codelist_name=parmcd,
            package_version=package,
        )

    @staticmethod
    def _find_suggestions(
        value: str,
        terms: list[CodelistTerm],
        max_results: int = 3,
    ) -> list[CodelistTerm]:
        """Find closest matching terms using sequence matching."""
        value_upper = value.strip().upper()
        scored: list[tuple[float, CodelistTerm]] = []

        for term in terms:
            sv_ratio = SequenceMatcher(
                None, value_upper, term.submission_value.upper(),
            ).ratio()
            pt_ratio = SequenceMatcher(
                None, value_upper, term.preferred_term.upper(),
            ).ratio()
            best = max(sv_ratio, pt_ratio)
            scored.append((best, term))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [term for _, term in scored[:max_results]]

    # ------------------------------------------------------------------
    # Biomedical concept lookup
    # ------------------------------------------------------------------

    def lookup_domain(self, assessment_name: str) -> DomainMapping:
        """Map an assessment name to a CDASH/SDTM domain.

        Uses the local biomedical concept registry first, then
        the CDISC Library API for unrecognised assessments.

        Args:
            assessment_name: SoA assessment label (e.g. "Complete Blood Count").

        Returns:
            DomainMapping with domain code, name, and optional testcd.
        """
        lower = assessment_name.lower()

        # Check local biomedical concept registry
        for keyword, (domain, name, testcd) in _BIOMEDICAL_CONCEPTS.items():
            if keyword in lower:
                return DomainMapping(
                    assessment_name=assessment_name,
                    domain_code=domain,
                    domain_name=name,
                    testcd_suggestion=testcd,
                    source="local",
                    confidence=1.0,
                )

        # Try API for unrecognised assessments
        if self.configured:
            data = self._api_get(
                f"/mdr/sdtmig/3-4/domains",
            )
            if data and "domains" in data:
                # Search domain definitions for matching assessments
                for domain_info in data["domains"]:
                    domain_name = str(domain_info.get("name", ""))
                    if lower in domain_name.lower():
                        return DomainMapping(
                            assessment_name=assessment_name,
                            domain_code=str(domain_info.get("name", "")[:2]),
                            domain_name=domain_name,
                            source="api",
                            confidence=0.8,
                        )

        # Default fallback
        return DomainMapping(
            assessment_name=assessment_name,
            domain_code="FA",
            domain_name="Findings About",
            source="fallback",
            confidence=0.5,
        )


# ---------------------------------------------------------------------------
# Enhanced normalizer
# ---------------------------------------------------------------------------

class CdiscLibraryNormalizer:
    """Enhanced CT normalizer backed by CDISC Library API.

    Validates CT values against the official CDISC codelist when the API
    is available. Falls back to CtNormalizer when offline.

    Args:
        api_key: CDISC Library API key (or set CDISC_LIBRARY_API_KEY env).
        cache_db: Path to SQLite cache database.
        cache_ttl_hours: Cache TTL in hours.
        fallback: CtNormalizer instance for offline fallback.
    """

    def __init__(
        self,
        api_key: str = "",
        cache_db: Optional[Path] = None,
        cache_ttl_hours: int = _DEFAULT_TTL_HOURS,
        fallback: Optional[CtNormalizer] = None,
    ) -> None:
        self._client = CdiscLibraryClient(
            api_key=api_key,
            cache_db=cache_db,
            cache_ttl_hours=cache_ttl_hours,
        )
        self._fallback = fallback or CtNormalizer()

    @property
    def api_available(self) -> bool:
        """True when the CDISC Library API key is configured."""
        return self._client.configured

    def normalize(
        self,
        parmcd: str,
        value: str,
        package: str = "sdtmct-2024-03-29",
    ) -> CtLookupResult:
        """Normalise a CT value with API validation and offline fallback.

        Strategy:
        1. Try CDISC Library API validation
        2. If API returns a valid match, return it
        3. If API unavailable or no match, fall back to local CtNormalizer
        4. Log warning when using fallback

        Args:
            parmcd: SDTM TSPARMCD (e.g. "PHASE").
            value: Raw value to normalise.
            package: CT package version.

        Returns:
            CtLookupResult with mapped status.
        """
        # Try API first
        if self._client.configured:
            validation = self._client.validate_ct_value(parmcd, value, package)
            if validation.valid and validation.matched_term is not None:
                term = validation.matched_term
                return CtLookupResult(
                    original_value=value,
                    tsval=term.submission_value,
                    tsvalcd=term.submission_value,
                    nci_code=term.code,
                    mapped=True,
                )

            if validation.source != "fallback":
                # API was reachable but value wasn't found — still use
                # fallback for backward compatibility
                fallback_result = self._fallback.normalize(parmcd, value)
                if fallback_result.mapped:
                    return fallback_result

                # Neither API nor fallback matched
                return CtLookupResult(
                    original_value=value,
                    tsval=value.strip(),
                    tsvalcd="",
                    nci_code="",
                    mapped=False,
                )

        # API not configured or not reachable — use fallback
        logger.warning(
            "CDISC Library API unavailable for %s=%s; using local fallback",
            parmcd, value,
        )
        return self._fallback.normalize(parmcd, value)

    def validate_with_suggestions(
        self,
        parmcd: str,
        value: str,
        package: str = "sdtmct-2024-03-29",
    ) -> CtValidationResult:
        """Validate a CT value and return suggestions for unknown terms.

        Args:
            parmcd: SDTM TSPARMCD.
            value: Value to validate.
            package: CT package version.

        Returns:
            CtValidationResult with suggestions for unmatched values.
        """
        return self._client.validate_ct_value(parmcd, value, package)

    def map_assessment(self, assessment_name: str) -> DomainMapping:
        """Map an assessment name to a CDASH/SDTM domain.

        Args:
            assessment_name: SoA assessment label.

        Returns:
            DomainMapping with domain code and optional testcd.
        """
        return self._client.lookup_domain(assessment_name)
