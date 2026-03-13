"""Tests for Process tab deprecation feature flag (PTCV-142).

Verifies that the _build_tab_names helper and PTCV_DISABLE_PROCESS_TAB
feature flag correctly control Process tab visibility.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ui.app import _build_tab_names


# ---------------------------------------------------------------------------
# _build_tab_names tests
# ---------------------------------------------------------------------------


class TestBuildTabNames:
    """Tests for the _build_tab_names helper (PTCV-142)."""

    def test_includes_process_by_default(self) -> None:
        names = _build_tab_names("Review")
        assert "Process" in names
        assert names[0] == "Process"

    def test_excludes_process_when_disabled(self) -> None:
        names = _build_tab_names("Review", include_process=False)
        assert "Process" not in names

    def test_first_tab_is_results_when_process_disabled(self) -> None:
        names = _build_tab_names("Review", include_process=False)
        assert names[0] == "Results"

    def test_all_non_process_tabs_present(self) -> None:
        expected = [
            "Results", "Quality", "SoA & SDTM",
            "Mock Data", "Query Pipeline", "Benchmark",
            "Refinement", "Review",
        ]
        names = _build_tab_names("Review", include_process=False)
        assert names == expected

    def test_review_label_with_count(self) -> None:
        names = _build_tab_names("Review (3)")
        assert names[-1] == "Review (3)"

    def test_tab_count_with_process(self) -> None:
        names = _build_tab_names("Review")
        assert len(names) == 9

    def test_tab_count_without_process(self) -> None:
        names = _build_tab_names("Review", include_process=False)
        assert len(names) == 8

    def test_order_preserved_with_process(self) -> None:
        names = _build_tab_names("Review")
        expected = [
            "Process", "Results", "Quality", "SoA & SDTM",
            "Mock Data", "Query Pipeline", "Benchmark",
            "Refinement", "Review",
        ]
        assert names == expected


# ---------------------------------------------------------------------------
# Feature flag tests
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    """Tests for the PTCV_DISABLE_PROCESS_TAB env var (PTCV-142)."""

    def test_flag_false_by_default(self) -> None:
        """Without env var, flag should be False."""
        env_backup = os.environ.pop(
            "PTCV_DISABLE_PROCESS_TAB", None,
        )
        try:
            mod = importlib.import_module("ptcv.ui.app")
            importlib.reload(mod)
            assert mod._PROCESS_TAB_DISABLED is False
        finally:
            if env_backup is not None:
                os.environ["PTCV_DISABLE_PROCESS_TAB"] = (
                    env_backup
                )
            else:
                os.environ.pop(
                    "PTCV_DISABLE_PROCESS_TAB", None,
                )
            # Restore module state
            importlib.reload(mod)

    def test_flag_true_when_env_set(self) -> None:
        """With env var set, flag should be True."""
        env_backup = os.environ.get(
            "PTCV_DISABLE_PROCESS_TAB",
        )
        os.environ["PTCV_DISABLE_PROCESS_TAB"] = "1"
        try:
            mod = importlib.import_module("ptcv.ui.app")
            importlib.reload(mod)
            assert mod._PROCESS_TAB_DISABLED is True
        finally:
            if env_backup is not None:
                os.environ["PTCV_DISABLE_PROCESS_TAB"] = (
                    env_backup
                )
            else:
                os.environ.pop(
                    "PTCV_DISABLE_PROCESS_TAB", None,
                )
            # Restore module state
            importlib.reload(mod)


# ---------------------------------------------------------------------------
# Dict-based tab lookup tests
# ---------------------------------------------------------------------------


class TestTabLookup:
    """Verify dict(zip(...)) pattern for tab references (PTCV-142)."""

    def test_process_tab_none_when_excluded(self) -> None:
        names = _build_tab_names("Review", include_process=False)
        # Simulate dict creation (real st.tabs returns widgets)
        tabs_dict = {n: f"widget_{n}" for n in names}
        assert tabs_dict.get("Process") is None

    def test_process_tab_present_when_included(self) -> None:
        names = _build_tab_names("Review", include_process=True)
        tabs_dict = {n: f"widget_{n}" for n in names}
        assert tabs_dict.get("Process") == "widget_Process"

    def test_all_tabs_accessible_without_process(self) -> None:
        names = _build_tab_names("Review", include_process=False)
        tabs_dict = {n: f"widget_{n}" for n in names}
        # All non-Process tabs must be accessible by key
        assert tabs_dict["Results"] is not None
        assert tabs_dict["Quality"] is not None
        assert tabs_dict["SoA & SDTM"] is not None
        assert tabs_dict["Mock Data"] is not None
        assert tabs_dict["Query Pipeline"] is not None
        assert tabs_dict["Benchmark"] is not None
        assert tabs_dict["Refinement"] is not None
        assert tabs_dict["Review"] is not None
