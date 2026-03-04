"""Trial curator for identifying qualifying protocols (PTCV-85).

Searches ClinicalTrials.gov for industry-sponsored, multi-indication,
dose-ranging trials with full-text protocol PDFs available for download.

Filtering pipeline:
  1. API search: sponsor_class=INDUSTRY + LargeDocHasProtocol=true
  2. Multi-indication: conditionsModule lists 2+ distinct conditions
  3. Dose-ranging: arms share interventions at different doses, OR
     title/description contains dose-ranging keywords
  4. Download qualifying PDFs via ClinicalTrialsService.download()
  5. Produce manifest JSON with metadata

Risk tier: MEDIUM — data pipeline (no PHI).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .clinicaltrials_service import (
    ClinicalTrialsService,
    _CT_GOV_BASE,
    _LARGE_DOC_ABBREVS,
)

# Dose-ranging keywords to match in title, brief_summary, or design
_DOSE_KEYWORDS = re.compile(
    r"dose[- ]?(rang|find|escal|response|titrat|explor)",
    re.IGNORECASE,
)

# Regex to extract numeric dose from arm labels/descriptions
# Matches patterns like "10 mg", "0.5mg/kg", "100mcg", "2.5 mg/m2"
_DOSE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mg|mcg|µg|ug|g|ml|iu|units?)"
    r"(?:/(?:kg|m2|day|dose))?",
    re.IGNORECASE,
)


@dataclass
class QualifyingTrial:
    """A trial that meets all curation criteria."""

    nct_id: str
    title: str
    sponsor: str
    conditions: list[str]
    dose_arms: list[str]
    dose_ranging_signal: str
    phase: str
    status: str
    url: str
    file_path: str = ""
    download_error: str = ""


def is_multi_indication(conditions: list[str]) -> bool:
    """Return True if the trial covers 2+ distinct conditions.

    Args:
        conditions: List of condition strings from conditionsModule.

    Returns:
        True if multi-indication.
    """
    return len(set(conditions)) >= 2


def is_dose_ranging(study: dict) -> tuple[bool, str, list[str]]:
    """Detect dose-ranging design from study data.

    Checks two signals:
      1. **Keyword**: title, brief_summary, or detailed_description
         contains dose-ranging terminology.
      2. **Arms**: 2+ experimental arms share an intervention name
         but have distinct numeric dose values in their labels.

    Args:
        study: Raw study dict from ClinicalTrials.gov v2 API.

    Returns:
        Tuple of (is_dose_ranging, signal_type, dose_arm_labels).
        signal_type is "keyword", "arms", or "" if not dose-ranging.
    """
    protocol = study.get("protocolSection", {})

    # --- Keyword check ---
    title = (
        protocol.get("identificationModule", {}).get("officialTitle", "")
        or protocol.get("identificationModule", {}).get("briefTitle", "")
    )
    brief = (
        protocol.get("descriptionModule", {}).get("briefSummary", "")
    )
    detailed = (
        protocol.get("descriptionModule", {}).get("detailedDescription", "")
    )
    text_blob = f"{title} {brief} {detailed}"
    if _DOSE_KEYWORDS.search(text_blob):
        return True, "keyword", []

    # --- Arms-based check ---
    arms_module = protocol.get("armsInterventionsModule", {})
    arm_groups = arms_module.get("armGroups", [])
    interventions = arms_module.get("interventions", [])

    # Build map: intervention_name -> list of (arm_label, dose_values)
    intervention_arms: dict[str, list[tuple[str, set[str]]]] = {}
    for arm in arm_groups:
        arm_type = arm.get("type", "")
        if arm_type in ("PLACEBO_COMPARATOR", "NO_INTERVENTION", "SHAM_COMPARATOR"):
            continue
        label = arm.get("label", "")
        description = arm.get("description", "")
        arm_text = f"{label} {description}"
        doses = set(_DOSE_PATTERN.findall(arm_text))

        for iname in arm.get("interventionNames", []):
            intervention_arms.setdefault(iname, []).append(
                (label, doses)
            )

    # Check if any intervention has 2+ arms with different doses
    for iname, arms_data in intervention_arms.items():
        if len(arms_data) < 2:
            continue
        all_doses: set[str] = set()
        arm_labels: list[str] = []
        for label, doses in arms_data:
            all_doses |= doses
            arm_labels.append(label)
        if len(all_doses) >= 2:
            return True, "arms", arm_labels

    return False, "", []


def _search_raw(
    timeout: int = 30,
    date_from: str = "",
    condition: str = "",
    phase: str = "",
    max_results: int = 500,
) -> list[dict]:
    """Search ClinicalTrials.gov for industry trials with protocol PDFs.

    Returns raw study dicts (not SearchResult) so that arms/conditions
    data is preserved for filtering.

    Args:
        timeout: HTTP timeout in seconds.
        date_from: Minimum StudyFirstPostDate (MM/DD/YYYY).
        condition: Condition filter.
        phase: Phase filter (e.g. "PHASE2").
        max_results: Maximum studies to fetch.

    Returns:
        List of raw study dicts from the v2 API.
    """
    advanced = ClinicalTrialsService._build_advanced_filter(
        sponsor_class="INDUSTRY",
        date_from=date_from,
        has_protocol_doc=True,
    )

    studies: list[dict] = []
    current_token = ""

    while len(studies) < max_results:
        page_size = min(1000, max_results - len(studies))
        params: dict[str, str] = {
            "pageSize": str(page_size),
            "format": "json",
            "fields": (
                "NCTId,BriefTitle,OfficialTitle,OverallStatus,"
                "Phase,LeadSponsorName,LeadSponsorClass,"
                "Condition,ProtocolSection,DocumentSection"
            ),
        }
        if condition:
            params["query.cond"] = condition
        if phase:
            params["query.term"] = phase
        if advanced:
            params["filter.advanced"] = advanced
        if current_token:
            params["pageToken"] = current_token

        url = f"{_CT_GOV_BASE}/studies?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            break

        batch = data.get("studies", [])
        studies.extend(batch)

        current_token = data.get("nextPageToken", "")
        if not current_token:
            break

    return studies[:max_results]


def _extract_conditions(study: dict) -> list[str]:
    """Extract conditions list from a raw study dict."""
    return (
        study.get("protocolSection", {})
        .get("conditionsModule", {})
        .get("conditions", [])
    )


def _extract_nct_id(study: dict) -> str:
    """Extract NCT ID from a raw study dict."""
    return (
        study.get("protocolSection", {})
        .get("identificationModule", {})
        .get("nctId", "")
    )


def _extract_title(study: dict) -> str:
    """Extract trial title from a raw study dict."""
    id_mod = study.get("protocolSection", {}).get(
        "identificationModule", {}
    )
    return id_mod.get("officialTitle") or id_mod.get("briefTitle", "")


def _extract_sponsor(study: dict) -> str:
    """Extract lead sponsor name from a raw study dict."""
    return (
        study.get("protocolSection", {})
        .get("sponsorCollaboratorsModule", {})
        .get("leadSponsor", {})
        .get("name", "")
    )


def _extract_phase(study: dict) -> str:
    """Extract phase string from a raw study dict."""
    phases = (
        study.get("protocolSection", {})
        .get("designModule", {})
        .get("phases", [])
    )
    return ", ".join(phases)


def _extract_status(study: dict) -> str:
    """Extract overall status from a raw study dict."""
    return (
        study.get("protocolSection", {})
        .get("statusModule", {})
        .get("overallStatus", "")
    )


def _has_protocol_pdf(study: dict) -> bool:
    """Check if study has a sponsor-uploaded protocol PDF."""
    large_docs: list[dict] = (
        study.get("documentSection", {})
        .get("largeDocumentModule", {})
        .get("largeDocs", [])
    )
    abbrevs = _LARGE_DOC_ABBREVS.get("PDF", ())
    for abbrev in abbrevs:
        for doc in large_docs:
            if doc.get("typeAbbrev") == abbrev:
                return True
    return False


def curate_trials(
    service: ClinicalTrialsService,
    date_from: str = "",
    condition: str = "",
    phase: str = "",
    max_search: int = 500,
    max_download: int = 20,
    timeout: int = 30,
    who: str = "ptcv-curator",
    why: str = "PTCV-85: dose-ranging multi-indication curation",
) -> list[QualifyingTrial]:
    """Search, filter, and download qualifying trial protocols.

    Pipeline:
      1. Search CT.gov: industry sponsor + protocol PDF available
      2. Filter: multi-indication (2+ conditions)
      3. Filter: dose-ranging (keyword or arms-based)
      4. Download protocol PDFs via service.download()

    Args:
        service: ClinicalTrialsService instance for downloads.
        date_from: Minimum StudyFirstPostDate (MM/DD/YYYY).
        condition: Condition filter for initial search.
        phase: Phase filter for initial search.
        max_search: Maximum studies to scan from API.
        max_download: Maximum protocols to download.
        timeout: HTTP timeout in seconds.
        who: Audit trail user identifier.
        why: Audit trail reason.

    Returns:
        List of QualifyingTrial with download results.
    """
    raw_studies = _search_raw(
        timeout=timeout,
        date_from=date_from,
        condition=condition,
        phase=phase,
        max_results=max_search,
    )

    qualifying: list[QualifyingTrial] = []

    for study in raw_studies:
        nct_id = _extract_nct_id(study)
        if not nct_id:
            continue

        # Filter 1: multi-indication
        conditions = _extract_conditions(study)
        if not is_multi_indication(conditions):
            continue

        # Filter 2: dose-ranging
        dr_flag, dr_signal, dr_arms = is_dose_ranging(study)
        if not dr_flag:
            continue

        # Filter 3: confirm protocol PDF available
        if not _has_protocol_pdf(study):
            continue

        trial = QualifyingTrial(
            nct_id=nct_id,
            title=_extract_title(study),
            sponsor=_extract_sponsor(study),
            conditions=conditions,
            dose_arms=dr_arms,
            dose_ranging_signal=dr_signal,
            phase=_extract_phase(study),
            status=_extract_status(study),
            url=f"https://clinicaltrials.gov/study/{nct_id}",
        )

        # Download protocol PDF
        if len(qualifying) < max_download:
            result = service.download(
                nct_id=nct_id, who=who, why=why
            )
            if result.success:
                trial.file_path = result.file_path
            else:
                trial.download_error = result.error or "Unknown error"

        qualifying.append(trial)

    return qualifying


def write_manifest(
    trials: list[QualifyingTrial],
    path: str,
) -> str:
    """Write manifest JSON listing all qualifying trials.

    Args:
        trials: List of QualifyingTrial from curate_trials().
        path: Output file path for the manifest.

    Returns:
        The path written to.
    """
    records = [asdict(t) for t in trials]
    manifest = {
        "total_qualifying": len(trials),
        "downloaded": sum(1 for t in trials if t.file_path),
        "download_errors": sum(1 for t in trials if t.download_error),
        "trials": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path
