"""Tests for pipeline stage DAG and dependency resolution (PTCV-77).

Pure-Python tests — no Streamlit dependency.

Covers GHERKIN scenarios:
  - User selects downstream stage → prerequisites auto-enabled
  - User deselects prerequisite → dependents cascaded off
  - Cached results skip re-execution (DAG ordering)
  - Multiple stages run in correct dependency order
"""

from __future__ import annotations

from ptcv.ui.pipeline_stages import (
    PIPELINE_STAGES,
    STAGE_BY_KEY,
    compute_active_stages,
    get_all_dependents,
    get_all_prerequisites,
    get_execution_order,
)


# ---------------------------------------------------------------------------
# Stage structure
# ---------------------------------------------------------------------------

class TestStageStructure:
    """Pipeline DAG is well-formed."""

    def test_seven_stages_defined(self) -> None:
        assert len(PIPELINE_STAGES) == 7

    def test_all_stage_keys_unique(self) -> None:
        keys = [s.key for s in PIPELINE_STAGES]
        assert len(keys) == len(set(keys))

    def test_all_prerequisites_reference_valid_stages(self) -> None:
        valid_keys = {s.key for s in PIPELINE_STAGES}
        for stage in PIPELINE_STAGES:
            for prereq in stage.prerequisites:
                assert prereq in valid_keys, (
                    f"{stage.key} has unknown prerequisite {prereq}"
                )

    def test_extraction_has_no_prerequisites(self) -> None:
        assert STAGE_BY_KEY["extraction"].prerequisites == ()

    def test_sdtm_depends_on_extraction_soa_retemplating(self) -> None:
        prereqs = set(STAGE_BY_KEY["sdtm"].prerequisites)
        assert prereqs == {"extraction", "soa", "retemplating"}


# ---------------------------------------------------------------------------
# Scenario: User selects single downstream stage
# ---------------------------------------------------------------------------

class TestAutoEnablePrerequisites:
    """Checking a downstream stage auto-enables prerequisites."""

    def test_sdtm_enables_three_prerequisites(self) -> None:
        """GHERKIN: When user checks 'SDTM Generation', then
        Extraction, SoA Extraction, ICH Retemplating are auto-checked.
        """
        active = compute_active_stages({"sdtm"})
        assert "extraction" in active
        assert "soa" in active
        assert "retemplating" in active
        assert "sdtm" in active

    def test_fidelity_enables_retemplating_and_extraction(self) -> None:
        active = compute_active_stages({"fidelity"})
        assert "retemplating" in active
        assert "extraction" in active
        assert "fidelity" in active

    def test_sov_enables_soa_and_extraction(self) -> None:
        active = compute_active_stages({"sov"})
        assert "soa" in active
        assert "extraction" in active
        assert "sov" in active

    def test_extraction_has_no_prerequisites(self) -> None:
        active = compute_active_stages({"extraction"})
        assert active == {"extraction"}

    def test_get_all_prerequisites_sdtm(self) -> None:
        prereqs = get_all_prerequisites("sdtm")
        assert prereqs == {"extraction", "soa", "retemplating"}

    def test_get_all_prerequisites_transitive(self) -> None:
        """Fidelity → retemplating → extraction."""
        prereqs = get_all_prerequisites("fidelity")
        assert prereqs == {"retemplating", "extraction"}


# ---------------------------------------------------------------------------
# Scenario: User deselects prerequisite
# ---------------------------------------------------------------------------

class TestCascadeDisable:
    """Unchecking a prerequisite disables all dependents."""

    def test_extraction_dependents_include_all_downstream(self) -> None:
        """GHERKIN: When user unchecks Extraction, then SoA,
        Retemplating, Coverage, Fidelity, SDTM, etc. are unchecked.
        """
        deps = get_all_dependents("extraction")
        assert "retemplating" in deps
        assert "soa" in deps
        assert "fidelity" in deps
        assert "sdtm" in deps
        assert "sov" in deps
        assert "annotations" in deps

    def test_soa_dependents(self) -> None:
        deps = get_all_dependents("soa")
        assert "sdtm" in deps
        assert "sov" in deps
        # Retemplating does NOT depend on SoA
        assert "retemplating" not in deps

    def test_retemplating_dependents(self) -> None:
        deps = get_all_dependents("retemplating")
        assert "fidelity" in deps
        assert "sdtm" in deps
        # SoA does NOT depend on retemplating
        assert "soa" not in deps

    def test_leaf_stage_has_no_dependents(self) -> None:
        assert get_all_dependents("annotations") == set()
        assert get_all_dependents("sov") == set()


# ---------------------------------------------------------------------------
# Scenario: Multiple stages run in correct order
# ---------------------------------------------------------------------------

class TestExecutionOrder:
    """Stages execute in topological dependency order."""

    def test_sdtm_pipeline_order(self) -> None:
        """GHERKIN: Extraction runs first, then SoA and
        Retemplating, then SDTM.
        """
        active = compute_active_stages({"sdtm"})
        order = get_execution_order(active)
        # Extraction must come before everything
        assert order.index("extraction") < order.index("retemplating")
        assert order.index("extraction") < order.index("soa")
        # Both SoA and retemplating before SDTM
        assert order.index("soa") < order.index("sdtm")
        assert order.index("retemplating") < order.index("sdtm")

    def test_fidelity_after_retemplating(self) -> None:
        active = compute_active_stages({"fidelity"})
        order = get_execution_order(active)
        assert order.index("extraction") < order.index("retemplating")
        assert order.index("retemplating") < order.index("fidelity")

    def test_only_selected_stages_in_order(self) -> None:
        """Unrelated stages excluded from execution order."""
        active = compute_active_stages({"soa"})
        order = get_execution_order(active)
        assert "sdtm" not in order
        assert "fidelity" not in order
        assert set(order) == {"extraction", "soa"}

    def test_full_pipeline_order(self) -> None:
        """All 7 stages selected — verify global order."""
        all_keys = {s.key for s in PIPELINE_STAGES}
        order = get_execution_order(all_keys)
        assert len(order) == 7
        # Extraction always first
        assert order[0] == "extraction"


# ---------------------------------------------------------------------------
# Scenario: Cached results skip re-execution
# ---------------------------------------------------------------------------

class TestCacheMapping:
    """Stages map to correct session state cache keys."""

    def test_extraction_and_retemplating_share_parse_cache(self) -> None:
        assert STAGE_BY_KEY["extraction"].cache_key == "parse_cache"
        assert STAGE_BY_KEY["retemplating"].cache_key == "parse_cache"

    def test_soa_uses_soa_cache(self) -> None:
        assert STAGE_BY_KEY["soa"].cache_key == "soa_cache"

    def test_fidelity_uses_fidelity_cache(self) -> None:
        assert STAGE_BY_KEY["fidelity"].cache_key == "fidelity_cache"

    def test_sdtm_uses_sdtm_cache(self) -> None:
        assert STAGE_BY_KEY["sdtm"].cache_key == "sdtm_cache"

    def test_display_only_stages_have_empty_cache_key(self) -> None:
        assert STAGE_BY_KEY["sov"].cache_key == ""
        assert STAGE_BY_KEY["annotations"].cache_key == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases in DAG resolution."""

    def test_empty_selection(self) -> None:
        active = compute_active_stages(set())
        assert active == set()

    def test_execution_order_empty(self) -> None:
        order = get_execution_order(set())
        assert order == []

    def test_duplicate_prerequisites_not_doubled(self) -> None:
        """SDTM has extraction as both direct and transitive prereq."""
        prereqs = get_all_prerequisites("sdtm")
        assert prereqs == {"extraction", "soa", "retemplating"}

    def test_multiple_user_stages_merge_prerequisites(self) -> None:
        active = compute_active_stages({"fidelity", "sov"})
        assert active == {
            "fidelity", "retemplating", "extraction",
            "sov", "soa",
        }
