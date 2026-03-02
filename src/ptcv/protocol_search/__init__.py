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
from .models import DownloadResult, ProtocolMetadata, SearchResult

__all__ = [
    "CTISService",
    "ClinicalTrialsService",
    "DownloadResult",
    "FilestoreManager",
    "ProtocolMetadata",
    "SearchResult",
]
