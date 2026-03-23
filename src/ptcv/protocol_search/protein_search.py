"""Protein-based protocol search, download, and CSV index (PTCV-190).

Orchestrates a full protein-search workflow:
1. Search ClinicalTrials.gov by protein names
2. Download protocol PDFs for matching trials
3. Write a CSV index with trial metadata

Risk tier: MEDIUM — data pipeline ingestion (no patient data).
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from .clinicaltrials_service import ClinicalTrialsService
from .filestore import _DEFAULT_ROOT
from .models import DownloadResult, ProteinSearchResult

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "nct_id",
    "trial_name",
    "condition",
    "year",
    "phase",
    "sponsor",
    "outcome",
    "publications",
    "matched_proteins",
    "url",
    "has_pdf",
    "file_path",
]


def write_protein_csv_index(
    results: list[ProteinSearchResult],
    output_path: Path,
    download_paths: Optional[dict[str, str]] = None,
) -> Path:
    """Write a CSV index of protein search results.

    [PTCV-190 Scenario: Generate CSV index]

    Args:
        results: List of ProteinSearchResult from search_by_proteins().
        output_path: Path for the output CSV file.
        download_paths: Optional mapping of NCT ID → local file path
            for downloaded protocols.

    Returns:
        Path to the written CSV file.
    """
    paths = download_paths or {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "nct_id": r.registry_id,
                "trial_name": r.title,
                "condition": r.condition,
                "year": r.year,
                "phase": r.phase,
                "sponsor": r.sponsor,
                "outcome": r.outcome,
                "publications": r.publications,
                "matched_proteins": "; ".join(r.matched_proteins),
                "url": r.url,
                "has_pdf": "yes" if r.has_protocol_pdf else "no",
                "file_path": paths.get(r.registry_id, ""),
            })

    logger.info(
        "PTCV-190: Wrote CSV index with %d entries to %s",
        len(results), output_path,
    )
    return output_path


def run_protein_search(
    proteins: list[str],
    output_dir: Optional[Path] = None,
    pdf_only: bool = True,
    download: bool = True,
    max_results_per_protein: int = 200,
    who: str = "protein-search",
) -> tuple[list[ProteinSearchResult], Path]:
    """End-to-end protein search, download, and CSV index.

    [PTCV-190 Scenario: Search by protein list]
    [PTCV-190 Scenario: Download full protocol PDFs]
    [PTCV-190 Scenario: Generate CSV index]

    Args:
        proteins: List of protein names to search.
        output_dir: Directory for downloads and CSV. Defaults to
            ``data/protocols/clinicaltrials/protein_search/``.
        pdf_only: Only include trials with protocol PDFs.
        download: Whether to download protocol PDFs.
        max_results_per_protein: Max studies to scan per protein.
        who: Audit trail user identifier.

    Returns:
        Tuple of (search results, path to CSV index).
    """
    if output_dir is None:
        output_dir = _DEFAULT_ROOT / "clinicaltrials" / "protein_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    service = ClinicalTrialsService(timeout=60)

    # Step 1: Search
    logger.info(
        "PTCV-190: Searching for proteins: %s", proteins,
    )
    results = service.search_by_proteins(
        proteins=proteins,
        pdf_only=pdf_only,
        max_results_per_protein=max_results_per_protein,
        who=who,
    )
    logger.info(
        "PTCV-190: Found %d unique trials with protocol PDFs",
        len(results),
    )

    # Step 2: Download
    download_paths: dict[str, str] = {}
    if download:
        import shutil

        for i, r in enumerate(results, 1):
            target = output_dir / f"{r.registry_id}_1.0.pdf"
            if target.exists():
                logger.info(
                    "  [%d/%d] SKIP %s (exists)",
                    i, len(results), r.registry_id,
                )
                download_paths[r.registry_id] = str(target)
                continue

            logger.info(
                "  [%d/%d] Downloading %s...",
                i, len(results), r.registry_id,
            )
            dl: DownloadResult = service.download(
                nct_id=r.registry_id,
                version="1.0",
                fmt="PDF",
                who=who,
                why="protein_search_download",
            )
            if dl.success and dl.actual_format == "PDF":
                src = Path(dl.file_path)
                if src.exists():
                    shutil.move(str(src), str(target))
                    download_paths[r.registry_id] = str(target)
            elif dl.success and dl.actual_format == "JSON":
                # Clean up JSON fallback — not a real protocol PDF
                src = Path(dl.file_path)
                if src.exists():
                    src.unlink()
            else:
                logger.warning(
                    "  Download failed for %s: %s",
                    r.registry_id, dl.error,
                )

    # Step 3: CSV index
    csv_path = output_dir / "protein_search_index.csv"
    write_protein_csv_index(results, csv_path, download_paths)

    return results, csv_path
