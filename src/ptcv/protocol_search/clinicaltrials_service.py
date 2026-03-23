"""ClinicalTrials.gov Protocol Search and Download Service.

Implements PTCV-18 search via GET https://clinicaltrials.gov/api/v2/studies
with automatic pagination and protocol download to the PTCV filestore.

PTCV-28: download() now checks largeDocumentModule first to retrieve actual
protocol PDFs and Statistical Analysis Plans uploaded by sponsors.  Falls
back to storing the registration JSON when no document is available.

PTCV-29: Storage delegated to StorageGateway (FilesystemAdapter or
LocalStorageAdapter). FilestoreManager accepted as deprecated alias.

ClinicalTrials.gov v2 API requires no authentication. Max page size
is 1,000 records; this service defaults to 100 per page and follows
the nextPageToken cursor for automatic pagination.

Risk tier: MEDIUM — data pipeline ingestion (no patient data).

Regulatory requirements:
- Audit trail: every download logged (21 CFR 11.10(e), ALCOA+)
- SHA-256: computed at download boundary (ALCOA+ Consistent)
- No PHI: trial registry data only (no participant identifiers)
"""

import dataclasses
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Optional

from ..compliance.audit import AuditAction, AuditLogger
from ..compliance.integrity import DataIntegrityGuard
from .filestore import _DEFAULT_ROOT
from .models import (
    DownloadResult,
    ProteinSearchResult,
    ProtocolMetadata,
    SearchResult,
)


_CT_GOV_BASE = "https://clinicaltrials.gov/api/v2"
_CT_GOV_DOCS_BASE = "https://clinicaltrials.gov/ProvidedDocs"
_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 1000

# Maps the caller's fmt string to the typeAbbrev values that satisfy it,
# in priority order (most specific first).
# Prot_SAP contains both protocol and SAP content; it satisfies either request
# but is checked last so that a dedicated Prot or SAP entry takes precedence.
_LARGE_DOC_ABBREVS: dict[str, tuple[str, ...]] = {
    "PDF": ("Prot", "Prot_SAP"),
    "SAP": ("SAP", "Prot_SAP"),
}

_FORMAT_EXT: dict[str, str] = {
    "PDF": "pdf",
    "SAP": "pdf",
    "JSON": "json",
    "CTR-XML": "xml",
}

_CONTENT_TYPES: dict[str, str] = {
    "PDF": "application/pdf",
    "SAP": "application/pdf",
    "JSON": "application/json",
    "CTR-XML": "application/xml",
}


def _make_key(
    registry_id: str,
    amendment_number: str,
    fmt: str,
    source: str,
) -> str:
    """Return the relative storage key for a protocol file."""
    ext = _FORMAT_EXT.get(fmt, "pdf")
    subdir = "eu-ctr" if source == "EU-CTR" else "clinicaltrials"
    return f"{subdir}/{registry_id}_{amendment_number}.{ext}"


def _content_type(fmt: str) -> str:
    """Return MIME type for the given format key."""
    return _CONTENT_TYPES.get(fmt, "application/octet-stream")


def _artifact_path(gateway: Any, key: str) -> str:
    """Return absolute path string for a gateway artifact key."""
    from ptcv.storage import FilesystemAdapter
    if isinstance(gateway, FilesystemAdapter):
        return str(gateway.root / key)
    return key


class ClinicalTrialsService:
    """Search and download protocols from ClinicalTrials.gov v2 API.

    Uses the /studies endpoint with query parameters for condition and
    phase filtering. Handles pagination automatically via nextPageToken.
    Downloads are stored in the PTCV filestore with audit trail entries.

    For download(), the service checks largeDocumentModule first so that
    actual sponsor-uploaded protocol PDFs are retrieved when available.
    Registration JSON is stored as a fallback only.
    [PTCV-28: largeDocumentModule pathway]

    Args:
        gateway: StorageGateway instance. Uses FilesystemAdapter if None.
        filestore: Deprecated — FilestoreManager accepted for backward
            compatibility. Use gateway= in new code.
        audit_logger: AuditLogger instance. Uses default log if None.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        gateway: Any = None,
        filestore: Any = None,
        audit_logger: Optional[AuditLogger] = None,
        timeout: int = 30,
    ) -> None:
        from ptcv.storage import FilesystemAdapter
        from .filestore import FilestoreManager

        if gateway is not None:
            self._gateway = gateway
        elif filestore is not None and isinstance(filestore, FilestoreManager):
            # FilestoreManager wraps FilesystemAdapter — extract the adapter
            self._gateway = filestore._adapter
        elif filestore is not None:
            # Assume a StorageGateway passed via filestore= (future compat)
            self._gateway = filestore
        else:
            self._gateway = FilesystemAdapter(root=_DEFAULT_ROOT)

        self._gateway.initialise()
        self._audit = audit_logger or AuditLogger(
            module="clinicaltrials_service"
        )
        self._integrity = DataIntegrityGuard()
        self._timeout = timeout

    @staticmethod
    def _build_advanced_filter(
        sponsor_class: str = "",
        date_from: str = "",
        has_protocol_doc: bool = False,
    ) -> str:
        """Build ``filter.advanced`` AREA[] clauses.

        Constructs the ClinicalTrials.gov v2 advanced filter string
        from individual filter components, joining with ``AND``.

        [PTCV-39: ICH E6(R3)-aligned sampling strategy]

        Args:
            sponsor_class: Sponsor classification filter.  Valid values:
                ``INDUSTRY``, ``NIH``, ``FED``, ``OTHER``, ``NETWORK``,
                ``INDIV``.  Empty string disables the filter.
            date_from: Minimum ``StudyFirstPostDate`` in ``MM/DD/YYYY``
                format.  Only studies posted on or after this date are
                returned.  Empty string disables the filter.
            has_protocol_doc: When True, restrict to studies that have
                a sponsor-uploaded protocol document
                (``LargeDocHasProtocol``).

        Returns:
            Combined advanced filter string, or empty string if no
            filters are active.
        """
        clauses: list[str] = []
        if sponsor_class:
            clauses.append(f"AREA[LeadSponsorClass]{sponsor_class}")
        if date_from:
            clauses.append(
                f"AREA[StudyFirstPostDate]RANGE[{date_from}, MAX]"
            )
        if has_protocol_doc:
            clauses.append("AREA[LargeDocHasProtocol]true")
        return " AND ".join(clauses)

    def search(
        self,
        condition: str = "",
        phase: str = "",
        status: str = "",
        sponsor_class: str = "",
        date_from: str = "",
        has_protocol_doc: bool = False,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_results: int = 0,
        page_token: str = "",
        who: str = "ptcv-service",
    ) -> list[SearchResult]:
        """Search ClinicalTrials.gov for matching trials.

        Calls GET /api/v2/studies with condition/phase/status filters.
        Follows pagination automatically until max_results is reached
        or no more pages remain.
        [PTCV-18 Scenario: Search ClinicalTrials.gov protocols]
        [PTCV-39 Scenario: Sponsor-filtered and temporal filtering]

        Args:
            condition: Condition or disease filter (e.g., "oncology").
            phase: Phase filter (e.g., "PHASE2").
            status: Overall status filter (e.g., "RECRUITING").
            sponsor_class: Lead sponsor classification.  Use
                ``"INDUSTRY"`` to restrict to industry-sponsored trials.
                Other values: ``"NIH"``, ``"FED"``, ``"OTHER"``,
                ``"NETWORK"``, ``"INDIV"``.
            date_from: Minimum ``StudyFirstPostDate`` in ``MM/DD/YYYY``
                format (e.g., ``"10/01/2024"``).  Only studies posted on
                or after this date are returned.
            has_protocol_doc: When True, restrict to studies with a
                sponsor-uploaded protocol PDF.
            page_size: Results per page (max 1,000).
            max_results: Stop after collecting this many results.
                         0 means fetch all available pages.
            page_token: Resume pagination from this cursor token.
            who: User/service identifier for audit trail.

        Returns:
            List of SearchResult instances. Empty if no matches.
        """
        effective_page_size = min(page_size, _MAX_PAGE_SIZE)

        self._audit.log(
            action=AuditAction.SEARCH,
            record_id="ClinicalTrials.gov",
            user_id=who,
            reason=(
                f"Protocol search: condition={condition!r} "
                f"phase={phase!r} status={status!r} "
                f"sponsor_class={sponsor_class!r} "
                f"date_from={date_from!r}"
            ),
        )

        advanced_filter = self._build_advanced_filter(
            sponsor_class=sponsor_class,
            date_from=date_from,
            has_protocol_doc=has_protocol_doc,
        )

        results: list[SearchResult] = []
        current_token = page_token

        while True:
            params: dict[str, str] = {
                "pageSize": str(effective_page_size),
                "format": "json",
                "fields": (
                    "NCTId,BriefTitle,OfficialTitle,OverallStatus,"
                    "Phase,LeadSponsorName,LeadSponsorClass,"
                    "Condition,ProtocolSection"
                ),
            }
            if condition:
                params["query.cond"] = condition
            if phase:
                params["query.term"] = phase
            if status:
                params["filter.overallStatus"] = status
            if advanced_filter:
                params["filter.advanced"] = advanced_filter
            if current_token:
                params["pageToken"] = current_token

            try:
                data = self._get_json(
                    f"{_CT_GOV_BASE}/studies", params
                )
            except Exception:
                break

            for study in data.get("studies", []):
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})
                sponsor_module = protocol.get("sponsorCollaboratorsModule", {})

                nct_id = id_module.get("nctId", "")
                results.append(
                    SearchResult(
                        registry_id=nct_id,
                        title=(
                            id_module.get("officialTitle")
                            or id_module.get("briefTitle", "")
                        ),
                        source="ClinicalTrials.gov",
                        sponsor=sponsor_module.get("leadSponsor", {}).get(
                            "name", ""
                        ),
                        phase=", ".join(
                            design_module.get("phases", [])
                        ),
                        condition=", ".join(
                            protocol.get("conditionsModule", {}).get(
                                "conditions", []
                            )
                        ),
                        status=status_module.get("overallStatus", ""),
                        url=f"https://clinicaltrials.gov/study/{nct_id}",
                    )
                )

                if max_results and len(results) >= max_results:
                    return results

            current_token = data.get("nextPageToken", "")
            if not current_token:
                break

        return results

    def search_pdf_available(
        self,
        condition: str = "",
        phase: str = "",
        fmt: str = "PDF",
        sponsor_class: str = "",
        date_from: str = "",
        max_results: int = 500,
        who: str = "ptcv-service",
    ) -> list[str]:
        """Return NCT IDs for trials that have a sponsor-uploaded document.

        Includes DocumentSection in the search fields so that large-document
        availability can be determined without a separate per-study HTTP
        request.  Only studies where a largeDocs entry matching ``fmt`` is
        present are returned.

        [PTCV-28 Scenario: Pre-filter search to PDF-available trials]
        [PTCV-39 Scenario: Sponsor/date filtering for PDF-available trials]

        Args:
            condition: Condition or disease filter.
            phase: Phase filter (e.g., "PHASE2").
            fmt: Document format to look for ("PDF" or "SAP").
            sponsor_class: Lead sponsor classification (e.g.,
                ``"INDUSTRY"``).
            date_from: Minimum ``StudyFirstPostDate`` in ``MM/DD/YYYY``
                format.
            max_results: Upper bound on studies to scan.
            who: User/service identifier for audit trail.

        Returns:
            List of NCT IDs that have a matching large document.
        """
        abbrevs = _LARGE_DOC_ABBREVS.get(fmt, ())
        if not abbrevs:
            return []

        self._audit.log(
            action=AuditAction.SEARCH,
            record_id="ClinicalTrials.gov",
            user_id=who,
            reason=(
                f"PDF-filtered search: condition={condition!r} "
                f"phase={phase!r} fmt={fmt!r} "
                f"sponsor_class={sponsor_class!r} "
                f"date_from={date_from!r}"
            ),
        )

        advanced_filter = self._build_advanced_filter(
            sponsor_class=sponsor_class,
            date_from=date_from,
        )

        matching_ids: list[str] = []
        current_token = ""
        fetched = 0

        while fetched < max_results:
            page_size = min(_MAX_PAGE_SIZE, max_results - fetched)
            params: dict[str, str] = {
                "pageSize": str(page_size),
                "format": "json",
                "fields": "NCTId,ProtocolSection,DocumentSection",
            }
            if condition:
                params["query.cond"] = condition
            if phase:
                params["query.term"] = phase
            if advanced_filter:
                params["filter.advanced"] = advanced_filter
            if current_token:
                params["pageToken"] = current_token

            try:
                data = self._get_json(f"{_CT_GOV_BASE}/studies", params)
            except Exception:
                break

            studies = data.get("studies", [])
            for study in studies:
                nct_id = (
                    study.get("protocolSection", {})
                    .get("identificationModule", {})
                    .get("nctId", "")
                )
                if nct_id and self._find_large_doc(study, abbrevs) is not None:
                    matching_ids.append(nct_id)

            fetched += len(studies)
            current_token = data.get("nextPageToken", "")
            if not current_token:
                break

        return matching_ids

    def search_industry_protocols(
        self,
        date_from: str = "10/01/2024",
        condition: str = "",
        phase: str = "",
        max_results: int = 100,
        who: str = "ptcv-service",
    ) -> list[SearchResult]:
        """Search for industry-sponsored trials with protocol PDFs.

        Convenience method combining ``LeadSponsorClass=INDUSTRY``,
        ``LargeDocHasProtocol=true``, and a temporal floor into a single
        call.  Designed for ICH E6(R3)-aligned sampling (PTCV-39).

        Args:
            date_from: Minimum ``StudyFirstPostDate`` in ``MM/DD/YYYY``
                format.  Defaults to ``"10/01/2024"`` to target the
                post-ICH-E6(R3) adoption window.
            condition: Condition or disease filter.
            phase: Phase filter (e.g., "PHASE3").
            max_results: Maximum results to return.
            who: User/service identifier for audit trail.

        Returns:
            List of SearchResult instances for industry-sponsored
            trials that have uploaded protocol documents.
        """
        return self.search(
            condition=condition,
            phase=phase,
            sponsor_class="INDUSTRY",
            date_from=date_from,
            has_protocol_doc=True,
            max_results=max_results,
            who=who,
        )

    # ------------------------------------------------------------------
    # Protein-based search (PTCV-190)
    # ------------------------------------------------------------------

    def search_by_proteins(
        self,
        proteins: list[str],
        pdf_only: bool = True,
        max_results_per_protein: int = 200,
        who: str = "ptcv-service",
    ) -> list[ProteinSearchResult]:
        """Search ClinicalTrials.gov by a list of protein names.

        Uses the v2 API ``query.term`` parameter for free-text matching
        across all indexed fields (interventions, descriptions, titles,
        keywords).  Deduplicates across proteins so each NCT ID appears
        once, with all matched proteins recorded.

        [PTCV-190 Scenario: Search by protein list]
        [PTCV-190 Scenario: Filter to PDF-available trials]

        Args:
            proteins: List of protein names to search for (e.g.,
                ``["VEGF", "PD-1", "HER2"]``).
            pdf_only: When True, only return trials that have a
                sponsor-uploaded protocol PDF.
            max_results_per_protein: Max studies to scan per protein.
            who: User/service identifier for audit trail.

        Returns:
            List of ProteinSearchResult instances, deduplicated by
            NCT ID with matched_proteins aggregated.
        """
        self._audit.log(
            action=AuditAction.SEARCH,
            record_id="ClinicalTrials.gov",
            user_id=who,
            reason=(
                f"Protein search: proteins={proteins!r} "
                f"pdf_only={pdf_only}"
            ),
        )

        abbrevs = _LARGE_DOC_ABBREVS.get("PDF", ())
        # NCT ID → ProteinSearchResult (for deduplication)
        seen: dict[str, ProteinSearchResult] = {}

        for protein in proteins:
            current_token = ""
            fetched = 0

            while fetched < max_results_per_protein:
                page_size = min(
                    _MAX_PAGE_SIZE,
                    max_results_per_protein - fetched,
                )
                params: dict[str, str] = {
                    "pageSize": str(page_size),
                    "format": "json",
                    "query.term": protein,
                    "fields": (
                        "NCTId,BriefTitle,OfficialTitle,"
                        "OverallStatus,Phase,LeadSponsorName,"
                        "Condition,ProtocolSection,"
                        "DocumentSection,ResultsSection,"
                        "ReferencesModule,StatusModule"
                    ),
                }
                if current_token:
                    params["pageToken"] = current_token

                try:
                    data = self._get_json(
                        f"{_CT_GOV_BASE}/studies", params
                    )
                except Exception:
                    break

                studies = data.get("studies", [])
                for study in studies:
                    result = self._extract_protein_result(
                        study, protein, abbrevs, pdf_only,
                    )
                    if result is None:
                        continue

                    nct_id = result.registry_id
                    if nct_id in seen:
                        # Merge protein match into existing entry
                        if protein not in seen[nct_id].matched_proteins:
                            seen[nct_id].matched_proteins.append(protein)
                    else:
                        seen[nct_id] = result

                fetched += len(studies)
                current_token = data.get("nextPageToken", "")
                if not current_token:
                    break

        return list(seen.values())

    def _extract_protein_result(
        self,
        study: dict,
        protein: str,
        abbrevs: tuple[str, ...],
        pdf_only: bool,
    ) -> ProteinSearchResult | None:
        """Extract a ProteinSearchResult from a v2 API study record.

        Returns None if pdf_only is True and no protocol PDF exists.
        """
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        nct_id = id_module.get("nctId", "")
        if not nct_id:
            return None

        has_pdf = self._find_large_doc(study, abbrevs) is not None
        if pdf_only and not has_pdf:
            return None

        status_module = protocol.get("statusModule", {})
        design_module = protocol.get("designModule", {})
        sponsor_module = protocol.get(
            "sponsorCollaboratorsModule", {}
        )
        conditions_module = protocol.get("conditionsModule", {})

        # Year: prefer study start date, fall back to first post date
        year = ""
        start_date = status_module.get(
            "startDateStruct", {}
        ).get("date", "")
        if start_date:
            # Format: "YYYY-MM-DD" or "YYYY-MM" or "Month YYYY"
            year = start_date[:4]
        if not year:
            first_post = status_module.get(
                "studyFirstPostDateStruct", {}
            ).get("date", "")
            if first_post:
                year = first_post[:4]

        # Outcome: check for results, otherwise overall status
        overall_status = status_module.get("overallStatus", "")
        results_date = status_module.get(
            "resultsFirstPostDateStruct", {}
        ).get("date", "")
        outcome = "Has Results" if results_date else overall_status

        # Publications from referencesModule
        refs_module = protocol.get("referencesModule", {})
        references = refs_module.get("references", [])
        pub_links: list[str] = []
        for ref in references:
            pmid = ref.get("pmid", "")
            if pmid:
                pub_links.append(
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                )
            elif ref.get("citation", ""):
                pub_links.append(ref["citation"][:120])
        publications = "; ".join(pub_links)

        return ProteinSearchResult(
            registry_id=nct_id,
            title=(
                id_module.get("officialTitle")
                or id_module.get("briefTitle", "")
            ),
            source="ClinicalTrials.gov",
            sponsor=sponsor_module.get(
                "leadSponsor", {}
            ).get("name", ""),
            phase=", ".join(design_module.get("phases", [])),
            condition=", ".join(
                conditions_module.get("conditions", [])
            ),
            status=overall_status,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
            matched_proteins=[protein],
            year=year,
            outcome=outcome,
            publications=publications,
            has_protocol_pdf=has_pdf,
        )

    def download(
        self,
        nct_id: str,
        version: str = "1.0",
        fmt: str = "PDF",
        who: str = "ptcv-service",
        why: str = "protocol_ingestion",
    ) -> DownloadResult:
        """Download a ClinicalTrials.gov protocol to the PTCV filestore.

        Checks largeDocumentModule first: if the sponsor has uploaded a
        document matching the requested format (Prot/Prot_SAP for PDF,
        SAP/Prot_SAP for SAP), the binary PDF is downloaded from
        https://clinicaltrials.gov/ProvidedDocs/{nct[-2:]}/{nct}/{filename}.

        Falls back to storing the registration JSON when no matching
        large document is found or the PDF fetch fails.

        SHA-256 is computed at the download boundary in all cases.
        Every outcome is written to the audit log (21 CFR 11.10(e)).

        [PTCV-18 Scenario: Download ClinicalTrials.gov protocol to PTCV filestore]
        [PTCV-28 Scenario: Download protocol PDF when largeDocumentModule is available]
        [PTCV-28 Scenario: Fall back to registration JSON when no protocol document]
        [PTCV-28 Scenario: Download Statistical Analysis Plan when available]

        Args:
            nct_id: NCT identifier (e.g., "NCT12345678").
            version: Protocol version string (default "1.0").
            fmt: Requested format — "PDF" (protocol), "SAP" (statistical
                 analysis plan), or "CTR-XML". Falls back to JSON when
                 no matching largeDoc is available.
            who: User/service identifier for audit trail.
            why: Reason for download (mandatory audit field).

        Returns:
            DownloadResult with success status, paths, hash, and
            actual_format ("PDF", "SAP", or "JSON" for fallback).
        """
        # Fetch the full study record (includes documentSection)
        url = f"{_CT_GOV_BASE}/studies/{urllib.parse.quote(nct_id, safe='')}"
        params: dict[str, str] = {"format": "json"}

        try:
            data = self._get_json(url, params)
        except urllib.error.HTTPError as exc:
            error_msg = f"HTTP {exc.code} retrieving {nct_id}: {exc.reason}"
            self._audit.log(
                action=AuditAction.DOWNLOAD,
                record_id=nct_id,
                user_id=who,
                reason=why,
                after={"success": False, "error": error_msg},
            )
            return DownloadResult(
                success=False, registry_id=nct_id, error=error_msg
            )
        except Exception as exc:
            error_msg = f"Download failed for {nct_id}: {exc}"
            self._audit.log(
                action=AuditAction.DOWNLOAD,
                record_id=nct_id,
                user_id=who,
                reason=why,
                after={"success": False, "error": error_msg},
            )
            return DownloadResult(
                success=False, registry_id=nct_id, error=error_msg
            )

        # Attempt to download the actual document from largeDocumentModule
        content, effective_fmt = self._fetch_large_doc(data, nct_id, fmt)

        # Build metadata from the study record
        protocol = data.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        design_module = protocol.get("designModule", {})
        conditions_module = protocol.get("conditionsModule", {})

        return self._store_and_audit(
            content=content,
            registry_id=nct_id,
            amendment_number=version,
            fmt=effective_fmt,
            source="ClinicalTrials.gov",
            title=(
                id_module.get("officialTitle")
                or id_module.get("briefTitle", "")
            ),
            sponsor=sponsor_module.get("leadSponsor", {}).get("name", ""),
            phase=", ".join(design_module.get("phases", [])),
            condition=", ".join(
                conditions_module.get("conditions", [])
            ),
            who=who,
            why=why,
        )

    def _store_and_audit(
        self,
        content: bytes,
        registry_id: str,
        amendment_number: str,
        fmt: str,
        source: str,
        title: str,
        sponsor: str,
        phase: str,
        condition: str,
        who: str,
        why: str,
    ) -> DownloadResult:
        """Store protocol + metadata and write audit entry."""
        run_id = str(uuid.uuid4())
        protocol_key = _make_key(registry_id, amendment_number, fmt, source)

        try:
            artifact = self._gateway.put_artifact(
                key=protocol_key,
                data=content,
                content_type=_content_type(fmt),
                run_id=run_id,
                source_hash="",
                user=who,
                immutable=True,
                registry_id=registry_id,
                amendment_number=amendment_number,
                source=source,
            )
        except FileExistsError as exc:
            return DownloadResult(
                success=False, registry_id=registry_id, error=str(exc)
            )

        file_path_str = _artifact_path(self._gateway, artifact.key)

        metadata = ProtocolMetadata(
            source=source,
            registry_id=registry_id,
            version=amendment_number,
            amendment_number=amendment_number,
            title=title,
            sponsor=sponsor,
            phase=phase,
            condition=condition,
            file_path=file_path_str,
            format=fmt,
            file_hash_sha256=artifact.sha256,
            legacy_source=False,
        )
        meta_bytes = json.dumps(
            dataclasses.asdict(metadata), indent=2, ensure_ascii=False
        ).encode("utf-8")
        metadata_key = f"metadata/{registry_id}_{amendment_number}.json"
        self._gateway.put_artifact(
            key=metadata_key,
            data=meta_bytes,
            content_type="application/json",
            run_id=run_id,
            source_hash=artifact.sha256,
            user=who,
            immutable=False,
        )
        metadata_path_str = _artifact_path(self._gateway, metadata_key)

        self._audit.log(
            action=AuditAction.DOWNLOAD,
            record_id=registry_id,
            user_id=who,
            reason=why,
            after={
                "file_path": file_path_str,
                "file_hash_sha256": artifact.sha256,
                "format": fmt,
            },
        )

        return DownloadResult(
            success=True,
            registry_id=registry_id,
            file_path=file_path_str,
            metadata_path=metadata_path_str,
            file_hash_sha256=artifact.sha256,
            actual_format=fmt,
        )

    # ------------------------------------------------------------------
    # Large document helpers (PTCV-28)
    # ------------------------------------------------------------------

    def _fetch_large_doc(
        self, study_data: dict, nct_id: str, fmt: str
    ) -> tuple[bytes, str]:
        """Fetch the best available document for the given format.

        Checks largeDocumentModule for a matching typeAbbrev entry and
        downloads the binary PDF from the ProvidedDocs URL.  Falls back
        to registration JSON if no entry is found or the fetch fails.

        Args:
            study_data: Full study record from the v2 API.
            nct_id: NCT identifier (used to build the ProvidedDocs URL).
            fmt: Requested format ("PDF" or "SAP").

        Returns:
            Tuple of (content_bytes, effective_format_string).
        """
        abbrevs = _LARGE_DOC_ABBREVS.get(fmt, ())
        large_doc = self._find_large_doc(study_data, abbrevs)

        if large_doc is not None:
            filename = large_doc.get("filename", "")
            pdf_url = self._build_provided_docs_url(nct_id, filename)
            try:
                content = self._get_bytes(pdf_url)
                return content, fmt
            except Exception:
                pass  # fall through to registration JSON fallback

        registration_json = json.dumps(
            study_data, indent=2, ensure_ascii=False
        ).encode("utf-8")
        return registration_json, "JSON"

    def _find_large_doc(
        self, study_data: dict, abbrevs: tuple[str, ...]
    ) -> Optional[dict]:
        """Return the first largeDocs entry matching any of the given typeAbbrevs.

        Priority follows the order of abbrevs — more specific entries
        (e.g., "Prot") are checked before combined entries ("Prot_SAP").

        Args:
            study_data: Full study record from the v2 API.
            abbrevs: typeAbbrev values to search for, in priority order.

        Returns:
            Matching largeDocs dict, or None if not found.
        """
        large_docs: list[dict] = (
            study_data
            .get("documentSection", {})
            .get("largeDocumentModule", {})
            .get("largeDocs", [])
        )
        for abbrev in abbrevs:
            for doc in large_docs:
                if doc.get("typeAbbrev") == abbrev:
                    return doc
        return None

    def _build_provided_docs_url(self, nct_id: str, filename: str) -> str:
        """Build the ProvidedDocs URL for a large document.

        URL pattern: https://clinicaltrials.gov/ProvidedDocs/{nct[-2:]}/{nct}/{file}

        Args:
            nct_id: NCT identifier (e.g., "NCT12345678").
            filename: Filename from the largeDocs entry (e.g., "Prot_000.pdf").

        Returns:
            Full URL string.
        """
        return (
            f"{_CT_GOV_DOCS_BASE}/{nct_id[-2:]}/"
            f"{urllib.parse.quote(nct_id, safe='')}/"
            f"{urllib.parse.quote(filename, safe='')}"
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_json(
        self, url: str, params: dict[str, str]
    ) -> dict[str, Any]:
        """GET with query params and return parsed JSON response."""
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))  # type: ignore[no-any-return]

    def _get_bytes(self, url: str) -> bytes:
        """GET a URL and return raw bytes."""
        req = urllib.request.Request(
            url, headers={"Accept": "application/pdf, */*"}
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return resp.read()  # type: ignore[no-any-return]
