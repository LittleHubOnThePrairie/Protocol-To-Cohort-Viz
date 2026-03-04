"""Tests for benchmark_ich_prompt_tokens.py (PTCV-68).

Scenario: Benchmark script produces comparison table
  Given scripts/benchmark_ich_prompt_tokens.py is run
  When it processes stage prompts
  Then it outputs a table with columns: stage, sections, format, token_count
  And compact-format rows show measurably fewer tokens than retemplater
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _PROJECT_ROOT / "scripts" / "benchmark_ich_prompt_tokens.py"


def test_benchmark_runs_without_error() -> None:
    """Script exits 0 and produces output."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "retemplater" in result.stdout
    assert "Savings" in result.stdout


def test_benchmark_shows_all_stages() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    for stage in ("retemplater", "coverage_reviewer", "sdtm_generator", "annotation_service"):
        assert stage in result.stdout, f"Missing stage: {stage}"


def test_benchmark_shows_compact_xml_sample() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "<ich_e6r3" in result.stdout
    assert "</ich_e6r3>" in result.stdout
