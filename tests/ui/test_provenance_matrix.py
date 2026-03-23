"""Tests for PTCV-251: Source Provenance Matrix.

Tests verify matrix building from pipeline results, extraction method
detection, registry fallback detection, and source page formatting.
"""

import pytest
from unittest.mock import MagicMock

from ptcv.ui.components.provenance_matrix import (
    ProvenanceRow,
    build_provenance_matrix,
    matrix_to_dataframe,
)


def _make_extraction(
    section_id: str,
    content: str = "Some content",
    method: str = "heuristic",
    source_section: str = "5.1",
) -> MagicMock:
    """Create a mock QueryExtraction."""
    ext = MagicMock()
    ext.section_id = section_id
    ext.query_id = f"{section_id}.q1"
    ext.content = content
    ext.confidence = 0.85
    ext.extraction_method = method
    ext.source_section = source_section
    return ext


def _make_mapping(
    protocol_section: str,
    ich_code: str,
    confidence: str = "high",
) -> MagicMock:
    """Create a mock SectionMapping."""
    match = MagicMock()
    match.ich_section_code = ich_code
    conf = MagicMock()
    conf.value = confidence
    match.confidence = conf

    mapping = MagicMock()
    mapping.protocol_section_number = protocol_section
    mapping.protocol_section_title = f"Section {protocol_section}"
    mapping.matches = [match]
    return mapping


def _make_assembled(populated_sections: list[str]) -> MagicMock:
    """Create a mock AssembledProtocol."""
    assembled = MagicMock()

    def _get_section(code):
        sec = MagicMock()
        sec.populated = code in populated_sections
        sec.section_name = f"Section {code}"
        return sec

    assembled.get_section = _get_section
    return assembled


class TestProvenanceRow:
    """Tests for ProvenanceRow dataclass."""

    def test_creation(self):
        row = ProvenanceRow(
            section_code="B.4",
            section_name="Trial Design",
            has_text=True,
            has_table=True,
            source_pages="pp. 12-14",
            populated=True,
        )
        assert row.section_code == "B.4"
        assert row.has_text
        assert row.has_table
        assert not row.has_diagram
        assert row.populated

    def test_defaults(self):
        row = ProvenanceRow("B.1", "General")
        assert not row.has_text
        assert not row.has_table
        assert not row.has_diagram
        assert not row.has_registry
        assert not row.populated
        assert row.source_link == ""


class TestBuildProvenanceMatrix:
    """Tests for build_provenance_matrix function."""

    def test_empty_result(self):
        rows = build_provenance_matrix({})
        assert len(rows) > 0  # All sections present
        assert all(not r.populated for r in rows)

    def test_all_sections_present(self):
        rows = build_provenance_matrix({})
        codes = {r.section_code for r in rows}
        assert "B.1" in codes
        assert "B.4" in codes
        assert "B.14" in codes

    def test_text_extraction_detected(self):
        match_result = MagicMock()
        match_result.mappings = [
            _make_mapping("5.1", "B.5"),
        ]

        extraction_result = MagicMock()
        extraction_result.extractions = [
            _make_extraction("B.5", method="heuristic"),
        ]

        assembled = _make_assembled(["B.5"])

        result = {
            "match_result": match_result,
            "extraction_result": extraction_result,
            "assembled": assembled,
        }

        rows = build_provenance_matrix(result)
        b5_row = next(r for r in rows if r.section_code == "B.5")

        assert b5_row.has_text
        assert b5_row.populated
        assert "5.1" in b5_row.source_pages

    def test_registry_fallback_detected(self):
        extraction_result = MagicMock()
        extraction_result.extractions = [
            _make_extraction(
                "B.1",
                content="[REGISTRY — ClinicalTrials.gov NCT12345678]\nSponsor: Test Corp",
            ),
        ]

        assembled = _make_assembled(["B.1"])

        result = {
            "extraction_result": extraction_result,
            "assembled": assembled,
        }

        rows = build_provenance_matrix(result, nct_id="NCT12345678")
        b1_row = next(r for r in rows if r.section_code == "B.1")

        assert b1_row.has_registry
        assert b1_row.populated
        assert "CT.gov" in b1_row.source_link
        assert "NCT12345678" in b1_row.source_link

    def test_table_extraction_detected(self):
        extraction_result = MagicMock()
        extraction_result.extractions = [
            _make_extraction("B.7", method="table"),
        ]

        assembled = _make_assembled(["B.7"])

        result = {
            "extraction_result": extraction_result,
            "assembled": assembled,
        }

        rows = build_provenance_matrix(result)
        b7_row = next(r for r in rows if r.section_code == "B.7")

        assert b7_row.has_table
        assert "(table)" in b7_row.source_link

    def test_unfilled_section_marked(self):
        rows = build_provenance_matrix({})
        b11_row = next(r for r in rows if r.section_code == "B.11")

        assert not b11_row.populated
        assert b11_row.source_link == "Not found"

    def test_nct_id_from_protocol_index(self):
        protocol_index = MagicMock()
        protocol_index.source_path = "/data/NCT05376319_1.0.pdf"

        result = {
            "protocol_index": protocol_index,
            "extraction_result": MagicMock(extractions=[
                _make_extraction(
                    "B.1",
                    content="[REGISTRY — ClinicalTrials.gov NCT05376319]\nData",
                ),
            ]),
            "assembled": _make_assembled(["B.1"]),
        }

        rows = build_provenance_matrix(result)
        b1_row = next(r for r in rows if r.section_code == "B.1")
        assert "NCT05376319" in b1_row.source_link

    def test_multiple_pages(self):
        match_result = MagicMock()
        match_result.mappings = [
            _make_mapping("3.1", "B.3"),
            _make_mapping("3.2", "B.3"),
        ]

        extraction_result = MagicMock()
        extraction_result.extractions = [
            _make_extraction("B.3"),
        ]

        assembled = _make_assembled(["B.3"])

        result = {
            "match_result": match_result,
            "extraction_result": extraction_result,
            "assembled": assembled,
        }

        rows = build_provenance_matrix(result)
        b3_row = next(r for r in rows if r.section_code == "B.3")

        assert "pp." in b3_row.source_pages
        assert "3.1" in b3_row.source_pages
        assert "3.2" in b3_row.source_pages


class TestMatrixToDataframe:
    """Tests for DataFrame conversion."""

    def test_produces_dataframe(self):
        rows = [
            ProvenanceRow("B.3", "Objectives", has_text=True, populated=True,
                          source_pages="pp. 8-9", source_link="pp. 8-9"),
            ProvenanceRow("B.11", "Direct Access", populated=False,
                          source_link="Not found"),
        ]

        df = matrix_to_dataframe(rows)

        assert len(df) == 2
        assert "Section" in df.columns
        assert "Text" in df.columns
        assert "Tables" in df.columns
        assert "Diagrams" in df.columns
        assert "CT.gov" in df.columns
        assert "Source" in df.columns

    def test_checkmarks(self):
        rows = [
            ProvenanceRow("B.4", "Design", has_text=True, has_table=True,
                          populated=True, source_link="pp. 12-14 (table)"),
        ]

        df = matrix_to_dataframe(rows)
        row = df.iloc[0]

        assert row["Text"] == "✓"
        assert row["Tables"] == "✓"
        assert row["Diagrams"] == ""
        assert row["CT.gov"] == ""

    def test_unfilled_warning(self):
        rows = [
            ProvenanceRow("B.11", "Direct Access", populated=False),
        ]

        df = matrix_to_dataframe(rows)
        assert "Not found" in df.iloc[0]["Source"]

    def test_registry_checkmark(self):
        rows = [
            ProvenanceRow("B.1", "General", has_registry=True,
                          populated=True, source_link="CT.gov"),
        ]

        df = matrix_to_dataframe(rows)
        assert df.iloc[0]["CT.gov"] == "✓"
