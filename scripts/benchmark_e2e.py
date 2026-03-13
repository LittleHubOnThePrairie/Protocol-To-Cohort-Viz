"""End-to-end pipeline benchmark (PTCV-178).

Usage:
    cd C:/Dev/PTCV
    python scripts/benchmark_e2e.py
    python scripts/benchmark_e2e.py --protocols NCT00112827 NCT01678794
    python scripts/benchmark_e2e.py --limit 3 --force-classification-level C5
    python scripts/benchmark_e2e.py --output-dir data/analysis/my_run -v
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from ptcv.analysis.benchmark_e2e import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    PROTOCOL_DIR,
    resolve_protocol_paths,
    run_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline benchmark (PTCV-178).",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=None,
        metavar="NCT_OR_PATH",
        help="NCT IDs or PDF paths (default: 8 curated protocols)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force-extraction-level",
        choices=["E1", "E2", "E3"],
        default=None,
        help="Override extraction degradation level",
    )
    parser.add_argument(
        "--force-classification-level",
        choices=["C1", "C2", "C3", "C4", "C5"],
        default=None,
        help="Override classification degradation level",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of protocols",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-stage timings",
    )
    args = parser.parse_args()

    paths = resolve_protocol_paths(
        protocols=args.protocols,
        protocol_dir=PROTOCOL_DIR,
        limit=args.limit,
    )
    if not paths:
        print("ERROR: No protocol PDFs resolved.", file=sys.stderr)
        return 1

    print(
        f"Benchmarking {len(paths)} protocols...",
        file=sys.stderr,
        flush=True,
    )

    report = run_benchmark(
        protocol_paths=paths,
        output_dir=args.output_dir,
        extraction_level=args.force_extraction_level,
        classification_level=args.force_classification_level,
        verbose=args.verbose,
    )

    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "protocol_count": report.protocol_count,
                "pass_count": report.pass_count,
                "error_count": report.error_count,
                "total_elapsed_seconds": report.total_elapsed_seconds,
                "failures_by_stage": report.failures_by_stage,
            },
            indent=2,
        ),
    )

    return 0 if report.error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
