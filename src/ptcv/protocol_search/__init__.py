"""Protocol Search and Download — EU-CTR and ClinicalTrials.gov.

Implements PTCV-18: search the CTIS public API and ClinicalTrials.gov
v2 API for clinical trial protocols, download them, and store files
and metadata in the PTCV local filestore.

Risk tier: MEDIUM — data pipeline (protocol ingestion). Requires
audit trail and SHA-256 integrity checks at download boundaries.

References:
- PTCV-12: Methods for Searching and Downloading Clinical Trial Protocols
- PTCV-11: EU-CTR Registry Access and Format Standards
"""

from .clinicaltrials_service import ClinicalTrialsService
from .eu_ctr_service import CTISService
from .filestore import FilestoreManager
from .industry_sponsors import (
    ALIAS_TO_CANONICAL,
    INDUSTRY_SPONSORS,
    canonicalise_sponsor,
    is_known_industry_sponsor,
)
from .models import DownloadResult, ProtocolMetadata, SearchResult
from .trial_curator import QualifyingTrial, curate_trials, write_manifest

__all__ = [
    "ALIAS_TO_CANONICAL",
    "CTISService",
    "ClinicalTrialsService",
    "DownloadResult",
    "FilestoreManager",
    "INDUSTRY_SPONSORS",
    "ProtocolMetadata",
    "QualifyingTrial",
    "SearchResult",
    "canonicalise_sponsor",
    "curate_trials",
    "is_known_industry_sponsor",
    "write_manifest",
]
