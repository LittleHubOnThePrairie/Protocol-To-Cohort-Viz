"""Tests for tab layout reorganization (PTCV-268, formerly PTCV-142).

Verifies that _build_tab_names produces pipeline-aligned tab order:
  Query Pipeline → SoA & Observations → SDTM & Validation → Advanced
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ui.app import _build_tab_names


# ---------------------------------------------------------------------------
# _build_tab_names tests (PTCV-268)
# ---------------------------------------------------------------------------


class TestBuildTabNames:
    """Tests for the _build_tab_names helper (PTCV-268)."""

    def test_first_tab_is_query_pipeline(self) -> None:
        names = _build_tab_names("Review")
        assert names[0] == "Query Pipeline"

    def test_second_tab_is_soa(self) -> None:
        names = _build_tab_names("Review")
        assert names[1] == "SoA & Observations"

    def test_third_tab_is_sdtm(self) -> None:
        names = _build_tab_names("Review")
        assert names[2] == "SDTM & Validation"

    def test_advanced_tab_present(self) -> None:
        names = _build_tab_names("Review")
        assert "Advanced" in names

    def test_tab_count(self) -> None:
        names = _build_tab_names("Review")
        assert len(names) == 4

    def test_pipeline_order(self) -> None:
        """Tabs read left-to-right matching pipeline flow."""
        names = _build_tab_names("Review")
        expected = [
            "Query Pipeline",
            "SoA & Observations",
            "SDTM & Validation",
            "Advanced",
        ]
        assert names == expected

    def test_legacy_tabs_not_at_top_level(self) -> None:
        """Legacy tabs (Process, Results, etc.) are not top-level."""
        names = _build_tab_names("Review")
        for legacy in ["Process", "Results", "Quality",
                        "Mock Data", "Benchmark", "Refinement"]:
            assert legacy not in names


# ---------------------------------------------------------------------------
# Dict-based tab lookup tests
# ---------------------------------------------------------------------------


class TestTabLookup:
    """Verify dict(zip(...)) pattern for tab references (PTCV-268)."""

    def test_all_primary_tabs_accessible(self) -> None:
        names = _build_tab_names("Review")
        tabs_dict = {n: f"widget_{n}" for n in names}
        assert tabs_dict["Query Pipeline"] is not None
        assert tabs_dict["SoA & Observations"] is not None
        assert tabs_dict["SDTM & Validation"] is not None
        assert tabs_dict["Advanced"] is not None

    def test_legacy_tabs_not_in_dict(self) -> None:
        names = _build_tab_names("Review")
        tabs_dict = {n: f"widget_{n}" for n in names}
        assert tabs_dict.get("Process") is None
        assert tabs_dict.get("Results") is None
        assert tabs_dict.get("SoA & SDTM") is None
