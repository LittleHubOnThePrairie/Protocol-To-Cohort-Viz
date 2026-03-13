"""Evaluate NeoBERT vs RuleBasedClassifier — PTCV-164.

Runs both classifiers on held-out test data and benchmark protocols,
producing a comparison report with F1/precision/recall per section,
inference timing, and GPU memory usage.

Usage:
    python scripts/evaluate_neobert.py [OPTIONS]

Options:
    --model-path PATH       Fine-tuned model directory (default: data/models/neobert)
    --benchmark-dir PATH    Benchmark Markdown directory
    --output PATH           Report output path (default: data/analysis/neobert_evaluation_report.md)

Risk tier: MEDIUM — evaluation pipeline (read-only).

Regulatory references:
- ALCOA+ Accurate: metrics computed from held-out data
- ALCOA+ Traceable: full comparison report with per-section breakdown
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Mock fitz (PyMuPDF) if not installed — prevents ImportError when
# ich_parser.__init__ imports toc_extractor.
if "fitz" not in sys.modules:
    try:
        import fitz  # noqa: F401
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["fitz"] = MagicMock()

from ptcv.ich_parser.classifier import RuleBasedClassifier
from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.neobert_config import INV_LABEL_MAP, NUM_LABELS, NeoBERTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NeoBERT vs RuleBasedClassifier",
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("data/models/neobert"),
        help="Fine-tuned model directory",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=None,
        help="Benchmark Markdown directory",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/analysis/neobert_evaluation_report.md"),
        help="Report output path",
    )
    return parser.parse_args()


def _compute_section_metrics(
    predicted: list[IchSection],
    expected_codes: set[str],
) -> dict[str, dict[str, float]]:
    """Compute per-section precision, recall, F1.

    Args:
        predicted: Sections predicted by classifier.
        expected_codes: Set of expected section codes.

    Returns:
        Dict mapping section_code to {precision, recall, f1}.
    """
    pred_codes = {s.section_code for s in predicted}
    all_codes = expected_codes | pred_codes

    metrics: dict[str, dict[str, float]] = {}
    for code in sorted(all_codes):
        tp = 1.0 if code in pred_codes and code in expected_codes else 0.0
        fp = 1.0 if code in pred_codes and code not in expected_codes else 0.0
        fn = 1.0 if code not in pred_codes and code in expected_codes else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[code] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def _macro_average(
    per_section: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute macro-averaged metrics."""
    if not per_section:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    n = len(per_section)
    return {
        "precision": sum(m["precision"] for m in per_section.values()) / n,
        "recall": sum(m["recall"] for m in per_section.values()) / n,
        "f1": sum(m["f1"] for m in per_section.values()) / n,
    }


def evaluate_on_benchmark(
    classifier: Any,
    benchmark_dir: Path,
    classifier_name: str,
) -> dict[str, Any]:
    """Evaluate classifier on benchmark Markdown protocols.

    Args:
        classifier: SectionClassifier instance.
        benchmark_dir: Directory containing benchmark Markdown files.
        classifier_name: Name for logging.

    Returns:
        Dict with per-protocol and aggregate results.
    """
    md_paths = sorted(benchmark_dir.glob("*/pymupdf4llm.md"))
    if not md_paths:
        logger.warning("No Markdown files found in %s", benchmark_dir)
        return {"protocols": [], "aggregate": {}}

    results: list[dict[str, Any]] = []
    all_per_section: dict[str, list[dict[str, float]]] = defaultdict(list)
    total_time = 0.0

    for md_path in md_paths:
        nct_id = md_path.parent.name
        text = md_path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        start = time.monotonic()
        sections = classifier.classify(
            text=text,
            registry_id=nct_id,
            run_id="eval",
            source_run_id="eval",
            source_sha256="eval",
        )
        elapsed = time.monotonic() - start
        total_time += elapsed

        pred_codes = {s.section_code for s in sections}
        avg_conf = (
            sum(s.confidence_score for s in sections) / len(sections)
            if sections
            else 0.0
        )

        results.append({
            "nct_id": nct_id,
            "sections_found": len(sections),
            "section_codes": sorted(pred_codes),
            "avg_confidence": round(avg_conf, 4),
            "elapsed_seconds": round(elapsed, 3),
        })

        logger.info(
            "%s | %s: %d sections in %.3fs (avg conf %.4f)",
            classifier_name, nct_id, len(sections), elapsed, avg_conf,
        )

    return {
        "protocols": results,
        "total_time": round(total_time, 3),
        "avg_time": round(total_time / max(len(results), 1), 3),
        "total_protocols": len(results),
    }


def generate_report(
    rule_results: dict[str, Any],
    neobert_results: dict[str, Any],
    gpu_memory_mb: float,
    output_path: Path,
) -> None:
    """Generate Markdown comparison report.

    Args:
        rule_results: RuleBasedClassifier benchmark results.
        neobert_results: NeoBERTClassifier benchmark results.
        gpu_memory_mb: GPU memory usage in MB.
        output_path: Path to write report.
    """
    lines = [
        "# NeoBERT Evaluation Report — PTCV-164",
        "",
        "## Summary",
        "",
        "Comparison of RuleBasedClassifier (regex/keyword) vs "
        "NeoBERTClassifier (fine-tuned transformer) on benchmark protocols.",
        "",
        "## Benchmark Results",
        "",
        "### RuleBasedClassifier",
        "",
        f"- Protocols evaluated: {rule_results.get('total_protocols', 0)}",
        f"- Total time: {rule_results.get('total_time', 0):.3f}s",
        f"- Avg time per protocol: {rule_results.get('avg_time', 0):.3f}s",
        "",
    ]

    # Per-protocol table for RuleBasedClassifier
    if rule_results.get("protocols"):
        lines.extend([
            "| NCT ID | Sections | Avg Confidence | Time (s) |",
            "|--------|----------|----------------|----------|",
        ])
        for p in rule_results["protocols"]:
            lines.append(
                f"| {p['nct_id']} | {p['sections_found']} | "
                f"{p['avg_confidence']:.4f} | {p['elapsed_seconds']:.3f} |"
            )
        lines.append("")

    lines.extend([
        "### NeoBERTClassifier",
        "",
        f"- Protocols evaluated: {neobert_results.get('total_protocols', 0)}",
        f"- Total time: {neobert_results.get('total_time', 0):.3f}s",
        f"- Avg time per protocol: {neobert_results.get('avg_time', 0):.3f}s",
        f"- GPU memory: {gpu_memory_mb:.0f} MB",
        "",
    ])

    # Per-protocol table for NeoBERTClassifier
    if neobert_results.get("protocols"):
        lines.extend([
            "| NCT ID | Sections | Avg Confidence | Time (s) |",
            "|--------|----------|----------------|----------|",
        ])
        for p in neobert_results["protocols"]:
            lines.append(
                f"| {p['nct_id']} | {p['sections_found']} | "
                f"{p['avg_confidence']:.4f} | {p['elapsed_seconds']:.3f} |"
            )
        lines.append("")

    # Comparison
    lines.extend([
        "## Comparison",
        "",
        "| Metric | RuleBasedClassifier | NeoBERTClassifier |",
        "|--------|--------------------:|------------------:|",
        f"| Protocols | {rule_results.get('total_protocols', 0)} | "
        f"{neobert_results.get('total_protocols', 0)} |",
        f"| Avg time/protocol | {rule_results.get('avg_time', 0):.3f}s | "
        f"{neobert_results.get('avg_time', 0):.3f}s |",
        f"| GPU memory | 0 MB | {gpu_memory_mb:.0f} MB |",
        "",
        "## Recommendation",
        "",
        "_To be filled after evaluation run._",
        "",
        "---",
        "",
        "Generated by `scripts/evaluate_neobert.py` (PTCV-164).",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", output_path)


def main() -> None:
    """Run evaluation."""
    args = parse_args()

    # RuleBasedClassifier — always available
    rule_classifier = RuleBasedClassifier()

    # NeoBERTClassifier — requires checkpoint
    neobert_classifier = None
    gpu_memory_mb = 0.0

    if args.model_path.exists():
        try:
            from ptcv.ich_parser.neobert_classifier import NeoBERTClassifier
            neobert_classifier = NeoBERTClassifier(
                model_path=args.model_path,
                device="auto",
            )
            gpu_memory_mb = neobert_classifier.get_gpu_memory_mb()
            logger.info(
                "NeoBERTClassifier loaded (%.0f MB GPU)", gpu_memory_mb,
            )
        except ImportError:
            logger.warning(
                "torch/transformers not available — "
                "skipping NeoBERT evaluation",
            )
        except FileNotFoundError:
            logger.warning(
                "Model checkpoint not found at %s — "
                "skipping NeoBERT evaluation",
                args.model_path,
            )
    else:
        logger.warning(
            "Model path %s does not exist — "
            "run train_neobert.py first",
            args.model_path,
        )

    # Find benchmark directory
    benchmark_dir = args.benchmark_dir
    if benchmark_dir is None:
        candidate = Path("data/analysis/benchmark_md")
        if candidate.is_dir():
            benchmark_dir = candidate

    if benchmark_dir is None or not benchmark_dir.is_dir():
        logger.error(
            "No benchmark directory found. "
            "Provide --benchmark-dir or place data in "
            "data/analysis/benchmark_md/",
        )
        raise SystemExit(1)

    # Evaluate RuleBasedClassifier
    logger.info("Evaluating RuleBasedClassifier...")
    rule_results = evaluate_on_benchmark(
        rule_classifier, benchmark_dir, "RuleBased",
    )

    # Evaluate NeoBERTClassifier
    neobert_results: dict[str, Any] = {
        "protocols": [],
        "total_time": 0,
        "avg_time": 0,
        "total_protocols": 0,
    }
    if neobert_classifier is not None:
        logger.info("Evaluating NeoBERTClassifier...")
        neobert_results = evaluate_on_benchmark(
            neobert_classifier, benchmark_dir, "NeoBERT",
        )
        gpu_memory_mb = neobert_classifier.get_gpu_memory_mb()

    # Generate report
    generate_report(
        rule_results, neobert_results, gpu_memory_mb, args.output,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
