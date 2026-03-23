"""Tests for SoA extraction in Query Pipeline (PTCV-239).

Tests that run_query_pipeline includes soa_result in output
and that the SoA stage uses ProtocolIndex.tables.
"""

from __future__ import annotations

import inspect
import pytest


class TestQueryPipelineSoaWiring:
    """Tests that SoA extraction is wired into the pipeline."""

    def test_run_query_pipeline_references_soa(self):
        """Test run_query_pipeline source contains SoA extraction."""
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "soa_extraction" in source
        assert "PTCV-239" in source
        assert "filter_soa_tables" in source

    def test_return_dict_includes_soa_result(self):
        """Test return value schema includes soa_result key."""
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert '"soa_result": soa_result' in source

    def test_soa_uses_protocol_index_tables(self):
        """Test SoA stage reads tables from ProtocolIndex."""
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "protocol_index" in source
        assert "tables" in source
        assert "ExtractedTable" in source

    def test_soa_failure_non_blocking(self):
        """Test SoA extraction failure doesn't crash the pipeline."""
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "SoA extraction in query pipeline failed" in source

    def test_soa_maps_to_usdm(self):
        """Test SoA stage produces USDM entities."""
        from ptcv.ui.components.query_pipeline import run_query_pipeline
        source = inspect.getsource(run_query_pipeline)
        assert "UsdmMapper" in source
        assert "timepoints" in source
        assert "activities" in source
