"""Tests for FidelityChecker — two-tier validation (PTCV-65).

Covers all 6 GHERKIN acceptance scenarios:
  1. Hallucination detected in retemplated section
  2. Content omission detected
  3. Faithful retemplating passes fidelity check
  4. Large protocol checked iteratively
  5. Cost warning displayed before running checker
  6. Checker runs without API key using deterministic mode only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest

from ptcv.ich_parser.fidelity_checker import (
    FidelityChecker,
    FidelityResult,
    SectionFidelity,
    _FIDELITY_THRESHOLD,
)
from ptcv.ich_parser.models import IchSection


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_section(
    code: str,
    text: str,
    confidence: float = 0.90,
) -> IchSection:
    """Build a minimal IchSection with content_text."""
    return IchSection(
        run_id="r1",
        source_run_id="",
        source_sha256="a" * 64,
        registry_id="NCT0001",
        section_code=code,
        section_name=f"Section {code}",
        content_json=json.dumps(
            {"text_excerpt": text[:200]},
        ),
        confidence_score=confidence,
        review_required=False,
        legacy_format=False,
        content_text=text,
    )


def _make_blocks(
    texts: list[tuple[int, str]],
) -> list[dict]:
    """Build text block dicts from (page, text) tuples."""
    return [
        {"page_number": p, "text": t} for p, t in texts
    ]


def _make_llm_response(
    score: float,
    hallucinations: list[str] | None = None,
    omissions: list[str] | None = None,
    drift_flags: list[str] | None = None,
) -> MagicMock:
    """Build a mock Anthropic response for fidelity check."""
    result = {
        "fidelity_score": score,
        "hallucinations": hallucinations or [],
        "omissions": omissions or [],
        "drift_flags": drift_flags or [],
    }
    content_block = MagicMock()
    content_block.text = json.dumps(result)
    response = MagicMock()
    response.content = [content_block]
    response.usage = MagicMock()
    response.usage.input_tokens = 5000
    response.usage.output_tokens = 500
    return response


# ------------------------------------------------------------------
# Scenario 1: Hallucination detected
# ------------------------------------------------------------------


class TestHallucinationDetected:
    """Scenario: Hallucination detected in retemplated section."""

    def test_non_overlapping_text_scores_low(self) -> None:
        """Deterministic: retemplated text not in original scores low."""
        original = _make_blocks([
            (1, "The primary endpoint is overall survival "
                "measured over a period of 24 months."),
        ])
        sections = [_make_section(
            "B.3",
            "The primary endpoint is progression-free survival "
            "measured at 12 months with a completely different "
            "assessment methodology.",
        )]
        checker = FidelityChecker(enable_llm=False)
        result = checker.check(
            original, sections, registry_id="NCT0001",
        )
        b3 = result.section_results[0]
        assert b3.fidelity_score < _FIDELITY_THRESHOLD

    def test_llm_hallucination_flagged(self) -> None:
        """LLM mock returns hallucination → flag populated."""
        original = _make_blocks([
            (1, "The study evaluates drug X for condition Y."),
        ])
        sections = [_make_section(
            "B.3", "The study evaluates drug X for condition Y.",
        )]

        mock_resp = _make_llm_response(
            score=0.60,
            hallucinations=[
                "Fabricated secondary endpoint: PFS at 6 months",
            ],
        )

        checker = FidelityChecker(enable_llm=True)
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"},
        ):
            checker._claude = MagicMock()
            checker._claude.messages.create.return_value = (
                mock_resp
            )
            result = checker.check(
                original, sections, registry_id="NCT0001",
            )

        assert result.total_hallucinations == 1
        b3 = result.section_results[0]
        assert len(b3.hallucinations) == 1
        assert "PFS" in b3.hallucinations[0]


# ------------------------------------------------------------------
# Scenario 2: Content omission detected
# ------------------------------------------------------------------


class TestOmissionDetected:
    """Scenario: Content omission detected."""

    def test_llm_omission_flagged(self) -> None:
        """LLM mock returns omissions → flag populated."""
        original = _make_blocks([
            (1, "Exclusion criteria: 1. Pregnant women. "
                "2. Age under 18. 3. Prior chemotherapy. "
                "4. Active infection. 5. Liver disease."),
        ])
        sections = [_make_section(
            "B.5",
            "Exclusion criteria: 1. Pregnant women. "
            "2. Age under 18. 3. Prior chemotherapy.",
        )]

        mock_resp = _make_llm_response(
            score=0.70,
            omissions=[
                "Missing: Active infection exclusion",
                "Missing: Liver disease exclusion",
            ],
        )

        checker = FidelityChecker(enable_llm=True)
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"},
        ):
            checker._claude = MagicMock()
            checker._claude.messages.create.return_value = (
                mock_resp
            )
            result = checker.check(
                original, sections, registry_id="NCT0001",
            )

        assert result.total_omissions == 2
        b5 = result.section_results[0]
        assert b5.fidelity_score < _FIDELITY_THRESHOLD


# ------------------------------------------------------------------
# Scenario 3: Faithful retemplating passes
# ------------------------------------------------------------------


class TestFaithfulRetemplating:
    """Scenario: Faithful retemplating passes fidelity check."""

    def test_identical_text_passes_deterministic(self) -> None:
        """Same text in original and retemplated → PASS."""
        text = (
            "This is a sufficiently long sentence that "
            "describes the study design in full detail. "
            "The randomization uses a 2:1 ratio with "
            "stratification by disease stage."
        )
        original = _make_blocks([(1, text)])
        sections = [_make_section("B.4", text)]
        checker = FidelityChecker(enable_llm=False)
        result = checker.check(
            original, sections, registry_id="NCT0001",
        )
        assert result.fidelity_passed is True
        assert result.overall_score >= _FIDELITY_THRESHOLD
        assert result.total_hallucinations == 0
        assert result.total_omissions == 0

    def test_all_sections_pass_with_llm(self) -> None:
        """LLM returns high scores → all sections pass."""
        text = "Study design includes dose escalation protocol."
        original = _make_blocks([(1, text)])
        sections = [
            _make_section("B.3", text),
            _make_section("B.4", text),
        ]

        mock_resp = _make_llm_response(score=0.95)

        checker = FidelityChecker(enable_llm=True)
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"},
        ):
            checker._claude = MagicMock()
            checker._claude.messages.create.return_value = (
                mock_resp
            )
            result = checker.check(
                original, sections, registry_id="NCT0001",
            )

        assert result.fidelity_passed is True
        assert result.method == "llm"
        for sr in result.section_results:
            assert sr.hallucinations == []
            assert sr.omissions == []


# ------------------------------------------------------------------
# Scenario 4: Large protocol checked iteratively
# ------------------------------------------------------------------


class TestIterativeSectionCheck:
    """Scenario: Large protocol checked iteratively."""

    def test_11_sections_11_llm_calls(self) -> None:
        """11 sections → 11 LLM calls, tokens aggregated."""
        text = "Content for section evaluation purposes."
        original = _make_blocks([(1, text)])
        sections = [
            _make_section(f"B.{i}", text) for i in range(1, 12)
        ]

        mock_resp = _make_llm_response(score=0.90)

        checker = FidelityChecker(enable_llm=True)
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"},
        ):
            checker._claude = MagicMock()
            checker._claude.messages.create.return_value = (
                mock_resp
            )
            result = checker.check(
                original, sections, registry_id="NCT0001",
            )

        assert len(result.section_results) == 11
        assert (
            checker._claude.messages.create.call_count == 11
        )
        assert result.input_tokens == 5000 * 11
        assert result.output_tokens == 500 * 11


# ------------------------------------------------------------------
# Scenario 5: Cost warning
# ------------------------------------------------------------------


class TestCostEstimate:
    """Scenario: Cost warning displayed before running checker."""

    def test_estimate_cost_returns_positive(self) -> None:
        """Cost estimate is positive for non-empty inputs."""
        sections = [_make_section("B.1", "x" * 5000)]
        blocks = _make_blocks([(1, "y" * 5000)])
        cost, in_tok, out_tok = FidelityChecker.estimate_cost(
            sections, blocks,
        )
        assert cost > 0
        assert in_tok > 0
        assert out_tok > 0

    def test_estimate_scales_with_sections(self) -> None:
        """More sections → higher cost estimate."""
        blocks = _make_blocks([(1, "y" * 5000)])
        sections_1 = [_make_section("B.1", "x" * 1000)]
        sections_11 = [
            _make_section(f"B.{i}", "x" * 1000)
            for i in range(1, 12)
        ]

        cost_1, _, _ = FidelityChecker.estimate_cost(
            sections_1, blocks,
        )
        cost_11, _, _ = FidelityChecker.estimate_cost(
            sections_11, blocks,
        )
        assert cost_11 > cost_1


# ------------------------------------------------------------------
# Scenario 6: Deterministic-only mode
# ------------------------------------------------------------------


class TestDeterministicOnlyMode:
    """Scenario: Checker runs without API key."""

    def test_no_api_key_deterministic_only(self) -> None:
        """No ANTHROPIC_API_KEY → deterministic method, 0 tokens."""
        text = (
            "Sufficient text for the deterministic check "
            "that needs to be long enough to pass the "
            "minimum sentence length filter."
        )
        original = _make_blocks([(1, text)])
        sections = [_make_section("B.1", text)]
        checker = FidelityChecker(enable_llm=True)
        with patch.dict("os.environ", {}, clear=True):
            result = checker.check(
                original, sections, registry_id="NCT0001",
            )
        assert result.method == "deterministic"
        assert result.input_tokens == 0

    def test_explicit_disable_llm(self) -> None:
        """enable_llm=False → deterministic regardless of key."""
        text = (
            "Test sentence for deterministic mode check "
            "that also has to be sufficiently long."
        )
        checker = FidelityChecker(enable_llm=False)
        result = checker.check(
            _make_blocks([(1, text)]),
            [_make_section("B.1", text)],
            registry_id="NCT0001",
        )
        assert result.method == "deterministic"


# ------------------------------------------------------------------
# Result type tests
# ------------------------------------------------------------------


class TestFidelityResultDataclass:
    """Test FidelityResult and SectionFidelity types."""

    def test_result_type(self) -> None:
        """check() returns FidelityResult."""
        checker = FidelityChecker(enable_llm=False)
        result = checker.check([], [], registry_id="NCT0001")
        assert isinstance(result, FidelityResult)

    def test_section_fidelity_type(self) -> None:
        """Section results are SectionFidelity instances."""
        text = (
            "Enough text for a proper section fidelity test "
            "with sufficient length to pass filters."
        )
        checker = FidelityChecker(enable_llm=False)
        result = checker.check(
            _make_blocks([(1, text)]),
            [_make_section("B.1", text)],
            registry_id="NCT0001",
        )
        assert all(
            isinstance(sr, SectionFidelity)
            for sr in result.section_results
        )


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty sections, empty original, etc."""

    def test_empty_sections_passes(self) -> None:
        """No sections → fidelity_passed=True."""
        checker = FidelityChecker(enable_llm=False)
        result = checker.check(
            _make_blocks([(1, "original text here")]),
            [],
            registry_id="NCT0001",
        )
        assert result.fidelity_passed is True
        assert result.overall_score == 1.0

    def test_empty_original_with_sections(self) -> None:
        """Empty original blocks → still returns valid result."""
        checker = FidelityChecker(enable_llm=False)
        result = checker.check(
            [],
            [_make_section("B.1", "retemplated text")],
            registry_id="NCT0001",
        )
        assert isinstance(result, FidelityResult)

    def test_custom_threshold(self) -> None:
        """Custom threshold is respected."""
        checker = FidelityChecker(
            pass_threshold=0.95, enable_llm=False,
        )
        assert checker._threshold == 0.95

        # Identical text → score 1.0 → passes even at 0.95
        text = (
            "A long sentence about clinical trial design "
            "that is present in both original and retemplated."
        )
        result = checker.check(
            _make_blocks([(1, text)]),
            [_make_section("B.1", text)],
            registry_id="NCT0001",
        )
        assert result.fidelity_passed is True
        assert result.pass_threshold == 0.95
