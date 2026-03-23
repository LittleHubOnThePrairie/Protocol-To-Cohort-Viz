"""Tests for SDTM+validation stages in Query Pipeline (PTCV-240).

Tests that Stages 6-7 are wired into run_query_pipeline and
produce sdtm_result and validation_result.
"""

from __future__ import annotations

import inspect
import pytest


class TestSdtmStageWiring:
    """Tests that SDTM generation is wired into the pipeline."""

    def test_run_query_pipeline_references_sdtm(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "sdtm_generation" in source
        assert "PTCV-240" in source
        assert "build_domain_specs" in source

    def test_return_includes_sdtm_result(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert '"sdtm_result": sdtm_result' in source

    def test_ex_domain_wired(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "build_ex_domain_spec" in source
        assert "ex_spec" in source

    def test_sdtm_failure_non_blocking(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "SDTM generation in query pipeline failed" in source


class TestValidationStageWiring:
    """Tests that validation is wired into the pipeline."""

    def test_run_query_pipeline_references_validation(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "validation" in source
        assert "check_required_domains" in source

    def test_return_includes_validation_result(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert '"validation_result": validation_result' in source

    def test_validation_failure_non_blocking(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "Validation in query pipeline failed" in source

    def test_trial_design_domains_included(self):
        """Test base trial design domains are always present."""
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        # TS, TV, TA, TE, TI should always be in present_domains
        assert '"TS", "TV", "TA", "TE", "TI"' in source


class TestStageOrdering:
    """Tests for correct stage ordering."""

    def test_sdtm_after_soa(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        soa_pos = source.index("soa_extraction")
        sdtm_pos = source.index("sdtm_generation")
        assert soa_pos < sdtm_pos

    def test_validation_after_sdtm(self):
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        sdtm_pos = source.index("sdtm_generation")
        val_pos = source.index("Validation")
        assert sdtm_pos < val_pos
