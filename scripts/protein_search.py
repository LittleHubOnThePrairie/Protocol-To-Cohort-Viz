"""Search ClinicalTrials.gov by protein names and download protocols.

Usage:
    cd C:/Dev/PTCV
    python -m scripts.protein_search VEGF PD-1 HER2 --dry-run
    python -m scripts.protein_search VEGF PD-1 HER2
    python -m scripts.protein_search --file proteins.txt

PTCV-190: Protein-based ClinicalTrials.gov search with protocol
download and CSV index.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ptcv.protocol_search.protein_search import run_protein_search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search ClinicalTrials.gov by protein names",
    )
    parser.add_argument(
        "proteins", nargs="*",
        help="Protein names to search for (e.g., VEGF PD-1 HER2)",
    )
    parser.add_argument(
        "--file", type=Path, default=None,
        help="Read protein names from a file (one per line)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search only, don't download",
    )
    parser.add_argument(
        "--max", type=int, default=200,
        help="Max results per protein (default 200)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for downloads and CSV",
    )
    parser.add_argument(
        "--include-no-pdf", action="store_true",
        help="Include trials without protocol PDFs in results",
    )
    args = parser.parse_args()

    # Collect protein names
    proteins: list[str] = list(args.proteins or [])
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            for line in fh:
                name = line.strip()
                if name and not name.startswith("#"):
                    proteins.append(name)

    if not proteins:
        parser.error(
            "Provide protein names as arguments or via --file"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    print(f"Searching for proteins: {proteins}")
    results, csv_path = run_protein_search(
        proteins=proteins,
        output_dir=args.output_dir,
        pdf_only=not args.include_no_pdf,
        download=not args.dry_run,
        max_results_per_protein=args.max,
    )

    print(f"\nResults: {len(results)} trials found")
    if args.dry_run:
        for r in results:
            print(
                f"  {r.registry_id}  "
                f"{r.matched_proteins}  "
                f"{r.phase}  "
                f"{r.sponsor[:40]}"
            )
    print(f"CSV index: {csv_path}")


if __name__ == "__main__":
    main()
