"""Tests for model cost-tier selector (PTCV-78).

Pure-Python tests — no Streamlit dependency.

Covers GHERKIN scenarios:
  - User selects Opus tier → LLM method, claude-opus-4-6
  - User selects Sonnet tier → RAG method, claude-sonnet-4-6
  - Cost estimate updates with file size / page count
  - Default is Opus when no selection made
"""

from __future__ import annotations

from ptcv.ui.app import (
    _DEFAULT_TIER,
    _MODEL_TIERS,
    estimate_cost,
)


# ---------------------------------------------------------------------------
# Scenario: Tier definitions
# ---------------------------------------------------------------------------

class TestTierDefinitions:
    """Model tiers are well-formed and complete."""

    def test_two_tiers_defined(self) -> None:
        assert len(_MODEL_TIERS) == 2

    def test_opus_tier_present(self) -> None:
        assert "Best Quality (Opus)" in _MODEL_TIERS

    def test_sonnet_tier_present(self) -> None:
        assert "Balanced (Sonnet)" in _MODEL_TIERS

    def test_opus_model_id(self) -> None:
        assert _MODEL_TIERS["Best Quality (Opus)"]["model_id"] == (
            "claude-opus-4-6"
        )

    def test_sonnet_model_id(self) -> None:
        assert _MODEL_TIERS["Balanced (Sonnet)"]["model_id"] == (
            "claude-sonnet-4-6"
        )

    def test_opus_method_is_llm(self) -> None:
        assert _MODEL_TIERS["Best Quality (Opus)"]["method"] == "llm"

    def test_sonnet_method_is_rag(self) -> None:
        assert _MODEL_TIERS["Balanced (Sonnet)"]["method"] == "rag"


# ---------------------------------------------------------------------------
# Scenario: Default is Opus
# ---------------------------------------------------------------------------

class TestDefaultTier:
    """Default tier is Opus when no selection made."""

    def test_default_tier_is_opus(self) -> None:
        assert _DEFAULT_TIER == "Best Quality (Opus)"

    def test_default_tier_in_model_tiers(self) -> None:
        assert _DEFAULT_TIER in _MODEL_TIERS


# ---------------------------------------------------------------------------
# Scenario: Cost estimate updates with page count
# ---------------------------------------------------------------------------

class TestEstimateCost:
    """estimate_cost() produces correct low/high ranges."""

    def test_opus_60_pages(self) -> None:
        """GHERKIN: 60-page protocol → Opus estimate ~$0.90-1.20."""
        low, high = estimate_cost(60, "Best Quality (Opus)")
        assert 0.80 <= low <= 1.00
        assert 1.10 <= high <= 1.30

    def test_sonnet_60_pages(self) -> None:
        """GHERKIN: 60-page protocol → Sonnet estimate ~$0.05-0.12."""
        low, high = estimate_cost(60, "Balanced (Sonnet)")
        assert 0.03 <= low <= 0.08
        assert 0.08 <= high <= 0.15

    def test_zero_pages(self) -> None:
        low, high = estimate_cost(0, "Best Quality (Opus)")
        assert low == 0.0
        assert high == 0.0

    def test_single_page(self) -> None:
        low, high = estimate_cost(1, "Best Quality (Opus)")
        assert low > 0
        assert high > low

    def test_opus_more_expensive_than_sonnet(self) -> None:
        opus_low, opus_high = estimate_cost(50, "Best Quality (Opus)")
        sonnet_low, sonnet_high = estimate_cost(50, "Balanced (Sonnet)")
        assert opus_low > sonnet_low
        assert opus_high > sonnet_high

    def test_cost_scales_linearly(self) -> None:
        low_10, high_10 = estimate_cost(10, "Best Quality (Opus)")
        low_20, high_20 = estimate_cost(20, "Best Quality (Opus)")
        assert abs(low_20 - 2 * low_10) < 1e-9
        assert abs(high_20 - 2 * high_10) < 1e-9

    def test_low_always_lte_high(self) -> None:
        for tier in _MODEL_TIERS:
            for pages in (1, 10, 60, 200):
                low, high = estimate_cost(pages, tier)
                assert low <= high, f"{tier} {pages}p: {low} > {high}"


# ---------------------------------------------------------------------------
# Scenario: Cost estimate ranges match ticket spec
# ---------------------------------------------------------------------------

class TestCostSpecCompliance:
    """Verify cost ranges match PTCV-78 acceptance criteria."""

    def test_opus_typical_protocol_range(self) -> None:
        """Typical protocol (60-80 pages): ~$0.90-1.20 for Opus."""
        low_60, _ = estimate_cost(60, "Best Quality (Opus)")
        _, high_80 = estimate_cost(80, "Best Quality (Opus)")
        assert low_60 >= 0.80
        assert high_80 <= 1.80

    def test_sonnet_typical_protocol_range(self) -> None:
        """Typical protocol (60-80 pages): ~$0.05-0.10 for Sonnet."""
        low_60, _ = estimate_cost(60, "Balanced (Sonnet)")
        _, high_80 = estimate_cost(80, "Balanced (Sonnet)")
        assert low_60 >= 0.03
        assert high_80 <= 0.20
