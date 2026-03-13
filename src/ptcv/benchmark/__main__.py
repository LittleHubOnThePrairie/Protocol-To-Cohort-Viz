"""CLI entry point for the benchmark suite (PTCV-185).

Usage::

    python -m ptcv.benchmark.run [OPTIONS]

Automatically loads secrets from ``.secrets`` before running.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path


def _load_secrets() -> None:
    """Load environment variables from .secrets file.

    Mirrors ``load-secrets.sh`` behaviour so the benchmark can run
    without needing ``source ./load-secrets.sh`` first.
    """
    # Walk up from CWD to find .secrets
    candidates = [
        Path.cwd() / ".secrets",
        Path(__file__).resolve().parents[3] / ".secrets",  # repo root
    ]
    for secrets_path in candidates:
        if secrets_path.exists():
            loaded = 0
            with open(secrets_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    name, _, value = line.partition("=")
                    name = name.strip()
                    value = value.strip()
                    os.environ[name] = value
                    loaded += 1
            print(f"[benchmark] Loaded {loaded} secrets from {secrets_path}")
            return
    print("[benchmark] No .secrets file found — LLM features may be disabled")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PTCV Pipeline Benchmark Suite (PTCV-185)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="JSON file listing protocol filenames to benchmark",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Directory containing protocol PDFs "
        "(default: data/protocols/clinicaltrials)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/benchmark/{timestamp}/)",
    )
    parser.add_argument(
        "--enable-transformation",
        action="store_true",
        default=False,
        help="Enable LLM content transformation",
    )
    parser.add_argument(
        "--no-summarization",
        action="store_true",
        default=False,
        help="Disable LLM sub-section scoring",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        default=False,
        help="Save results as a new baseline",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        metavar="PATH_OR_LATEST",
        help="Compare against a baseline (path or 'latest')",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load secrets before importing pipeline modules
    _load_secrets()

    # Lazy imports after secrets are loaded
    from ptcv.benchmark.runner import run_benchmark
    from ptcv.benchmark.report import (
        format_benchmark_report,
        format_comparison_report,
    )
    from ptcv.benchmark.comparator import (
        compare_runs,
        find_latest_baseline,
        load_baseline,
    )

    print("=" * 60)
    print("  PTCV Pipeline Benchmark Suite")
    print("=" * 60)
    print()

    result = run_benchmark(
        corpus_dir=args.corpus_dir,
        corpus_file=args.corpus,
        enable_transformation=args.enable_transformation,
        enable_summarization=not args.no_summarization,
        output_dir=args.output_dir,
    )

    # Generate and print report
    report = format_benchmark_report(result)
    print()
    print(report)

    # Write report to output dir
    if args.output_dir:
        out = args.output_dir
    else:
        out = Path("data/benchmark") / result.timestamp
    report_path = out / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}")

    # Save as baseline if requested
    if args.save_baseline:
        baselines_dir = Path("data/benchmark/baselines")
        baselines_dir.mkdir(parents=True, exist_ok=True)
        baseline_dir = baselines_dir / result.timestamp
        baseline_dir.mkdir(parents=True, exist_ok=True)
        src = out / "benchmark_results.json"
        dst = baseline_dir / "benchmark_results.json"
        shutil.copy2(src, dst)
        print(f"Baseline saved to {dst}")

    # Compare against baseline if requested
    if args.compare_baseline:
        baselines_dir = Path("data/benchmark/baselines")
        if args.compare_baseline == "latest":
            baseline_path = find_latest_baseline(baselines_dir)
            if not baseline_path:
                print("No baselines found in data/benchmark/baselines/")
                sys.exit(1)
        else:
            baseline_path = Path(args.compare_baseline)

        baseline_data = load_baseline(baseline_path)
        current_data = result.to_dict()
        comparison = compare_runs(baseline_data, current_data)

        comparison_report = format_comparison_report(comparison)
        print()
        print(comparison_report)

        # Write comparison to output dir
        comp_path = out / "comparison_report.md"
        with open(comp_path, "w") as f:
            f.write(comparison_report)

        comp_json = out / "comparison.json"
        with open(comp_json, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2, default=str)

        if comparison.has_regressions:
            print(
                f"\nWARNING: {len(comparison.regressions)} "
                f"regressions detected!"
            )
            sys.exit(2)

    print()
    print(
        f"Benchmark complete: {result.successful_runs}/"
        f"{result.corpus_size} protocols succeeded"
    )


if __name__ == "__main__":
    main()
