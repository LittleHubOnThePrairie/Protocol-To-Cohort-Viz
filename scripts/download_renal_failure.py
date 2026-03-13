"""Download protocol PDFs for renal failure clinical trials.

Searches ClinicalTrials.gov for trials targeting renal failure that have
sponsor-uploaded protocol PDFs, then downloads them to:
    data/protocols/clinicaltrials/renal_failure/

Usage:
    cd C:/Dev/PTCV
    python -m scripts.download_renal_failure [--dry-run] [--max N]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ptcv.protocol_search.clinicaltrials_service import ClinicalTrialsService


_TARGET_DIR = Path("C:/Dev/PTCV/data/protocols/clinicaltrials/renal_failure")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download renal failure protocol PDFs"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search only, don't download",
    )
    parser.add_argument(
        "--max", type=int, default=200,
        help="Max results to scan (default 200)",
    )
    args = parser.parse_args()

    service = ClinicalTrialsService(timeout=60)

    # Search for renal failure trials with sponsor-uploaded PDFs
    search_terms = [
        "renal failure",
        "acute kidney injury",
        "chronic kidney disease",
    ]

    all_nct_ids: list[str] = []
    seen: set[str] = set()

    for term in search_terms:
        print(f"\n--- Searching: {term!r} ---")
        nct_ids = service.search_pdf_available(
            condition=term,
            max_results=args.max,
            who="renal-failure-download",
        )
        new = [nid for nid in nct_ids if nid not in seen]
        seen.update(new)
        all_nct_ids.extend(new)
        print(f"  Found {len(nct_ids)} total, {len(new)} new")

    print(f"\nTotal unique trials with PDFs: {len(all_nct_ids)}")

    if args.dry_run:
        for nid in all_nct_ids:
            print(f"  [dry-run] Would download: {nid}")
        return

    # Create target directory
    _TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Download each protocol PDF directly to the subfolder
    success = 0
    skipped = 0
    failed = 0

    for i, nct_id in enumerate(all_nct_ids, 1):
        target_path = _TARGET_DIR / f"{nct_id}_1.0.pdf"
        if target_path.exists():
            print(f"  [{i}/{len(all_nct_ids)}] SKIP {nct_id} (already exists)")
            skipped += 1
            continue

        print(f"  [{i}/{len(all_nct_ids)}] Downloading {nct_id}...", end=" ")
        result = service.download(
            nct_id=nct_id,
            version="1.0",
            fmt="PDF",
            who="renal-failure-download",
            why="renal_failure_corpus_build",
        )

        if result.success and result.actual_format == "PDF":
            # Move from standard location to renal_failure subfolder
            src = Path(result.file_path)
            if src.exists():
                import shutil
                shutil.move(str(src), str(target_path))
                print(f"OK ({target_path.name})")
                success += 1
            else:
                print(f"WARN: downloaded but file not at {src}")
                failed += 1
        elif result.success and result.actual_format == "JSON":
            # JSON fallback — not a PDF, skip
            # Clean up the JSON that was saved
            src = Path(result.file_path)
            if src.exists():
                src.unlink()
            print("SKIP (no PDF, only JSON fallback)")
            skipped += 1
        else:
            print(f"FAIL: {result.error}")
            failed += 1

    print(f"\nDone: {success} downloaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
