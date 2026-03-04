#!/usr/bin/env python
"""Benchmark ICH stage-specific prompt token counts (PTCV-68).

Compares full (retemplater) vs compact (stage-filtered) prompts and
prints a comparison table showing token savings per stage.

Usage::

    python scripts/benchmark_ich_prompt_tokens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so ptcv package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ptcv.ich_parser.schema_loader import get_stage_prompt, load_ich_schema


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4 (conservative for English)."""
    return len(text) // 4


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    schema = load_ich_schema()

    # Baseline: retemplater (full, all 14 sections)
    baseline = get_stage_prompt("retemplater")
    baseline_tokens = estimate_tokens(baseline)

    print("=" * 72)
    print("ICH E6(R3) Stage-Specific Prompt Token Benchmark (PTCV-68)")
    print("=" * 72)
    print()
    print(
        f"{'Stage':<22} {'Sections':>8} {'Format':<8} "
        f"{'Words':>6} {'~Tokens':>8} {'Savings':>8}"
    )
    print("-" * 72)

    for stage_name, cfg in schema.stage_prompts.items():
        prompt = get_stage_prompt(stage_name)
        tokens = estimate_tokens(prompt)
        words = word_count(prompt)
        savings = (
            f"{(1 - tokens / baseline_tokens) * 100:.0f}%"
            if baseline_tokens > 0
            else "N/A"
        )
        if stage_name == "retemplater":
            savings = "(baseline)"

        print(
            f"{stage_name:<22} {len(cfg.sections):>8} {cfg.format:<8} "
            f"{words:>6} {tokens:>8} {savings:>8}"
        )

    print("-" * 72)
    print()

    # Show sample compact output
    print("Sample compact prompt (sdtm_generator):")
    print("-" * 40)
    print(get_stage_prompt("sdtm_generator"))
    print()


if __name__ == "__main__":
    main()
