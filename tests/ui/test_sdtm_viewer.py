"""Unit tests for SDTM viewer component (PTCV-36).

Tests the pure-Python helpers (build_lineage_records, build_lineage_figure)
without Streamlit or pyreadstat.

Covers GHERKIN scenarios:
  - Sankey diagram shows flows from ICH section to SDTM domain
  - Lineage includes B.4 → TA, B.5 → TI, SoA → TV, B.1 → TS
  - Non-ICH / empty domains produce appropriate output
  - Domain ordering and labels are correct
"""

from __future__ import annotations

import pytest

try:
    import plotly  # noqa: F401
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ptcv.ui.components.sdtm_viewer import (
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    LINEAGE_LINKS,
    build_lineage_figure,
    build_lineage_records,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for module-level constants."""

    def test_domain_order_has_five_entries(self) -> None:
        assert len(DOMAIN_ORDER) == 5
        assert set(DOMAIN_ORDER) == {"TS", "TA", "TE", "TV", "TI"}

    def test_domain_labels_covers_all_domains(self) -> None:
        for domain in DOMAIN_ORDER:
            assert domain in DOMAIN_LABELS

    def test_lineage_links_covers_all_domains(self) -> None:
        target_domains = {link[1] for link in LINEAGE_LINKS}
        assert target_domains == {"TS", "TA", "TE", "TV", "TI"}

    def test_lineage_links_sources_are_valid(self) -> None:
        valid_sources = {
            "B.1 General Information",
            "B.4 Trial Design",
            "B.5 Selection of Subjects",
            "SoA Visit Schedule",
        }
        for source, _ in LINEAGE_LINKS:
            assert source in valid_sources


# ---------------------------------------------------------------------------
# build_lineage_records
# ---------------------------------------------------------------------------

# Standard domain row counts for tests
STANDARD_COUNTS: dict[str, int] = {
    "TS": 7,
    "TA": 6,
    "TE": 3,
    "TV": 4,
    "TI": 5,
}


class TestBuildLineageRecords:
    """Tests for build_lineage_records()."""

    def test_full_protocol_all_links(self) -> None:
        """[PTCV-36 Scenario: Sankey diagram shows all mappings]"""
        records = build_lineage_records(
            section_codes=["B.1", "B.4", "B.5"],
            domain_row_counts=STANDARD_COUNTS,
            has_soa=True,
        )
        # All 6 links should be present
        assert len(records) == 6

    def test_b1_to_ts_link(self) -> None:
        """[PTCV-36 Scenario: B.1 General Information → TS]"""
        records = build_lineage_records(
            section_codes=["B.1"],
            domain_row_counts={"TS": 7},
        )
        ts_links = [r for r in records if r["target_domain"] == "TS"]
        assert len(ts_links) == 1
        assert ts_links[0]["source"] == "B.1 General Information"

    def test_b4_to_ta_te_ts_links(self) -> None:
        """[PTCV-36 Scenario: B.4 Trial Design → TA, TE, TS]"""
        records = build_lineage_records(
            section_codes=["B.4"],
            domain_row_counts=STANDARD_COUNTS,
        )
        sources = {r["target_domain"] for r in records}
        assert {"TS", "TA", "TE"} == sources

    def test_b5_to_ti_link(self) -> None:
        """[PTCV-36 Scenario: B.5 Selection of Subjects → TI]"""
        records = build_lineage_records(
            section_codes=["B.5"],
            domain_row_counts={"TI": 5},
        )
        assert len(records) == 1
        assert records[0]["target_domain"] == "TI"
        assert records[0]["source"] == "B.5 Selection of Subjects"

    def test_soa_to_tv_link(self) -> None:
        """[PTCV-36 Scenario: SoA Visit Schedule → TV]"""
        records = build_lineage_records(
            section_codes=[],
            domain_row_counts={"TV": 4},
            has_soa=True,
        )
        assert len(records) == 1
        assert records[0]["target_domain"] == "TV"
        assert records[0]["source"] == "SoA Visit Schedule"

    def test_soa_false_excludes_tv(self) -> None:
        records = build_lineage_records(
            section_codes=[],
            domain_row_counts={"TV": 4},
            has_soa=False,
        )
        assert len(records) == 0

    def test_missing_section_excludes_link(self) -> None:
        """Only B.1 present — no TA/TE/TI links."""
        records = build_lineage_records(
            section_codes=["B.1"],
            domain_row_counts=STANDARD_COUNTS,
        )
        target_domains = {r["target_domain"] for r in records}
        assert "TA" not in target_domains
        assert "TI" not in target_domains

    def test_zero_row_count_excludes_link(self) -> None:
        records = build_lineage_records(
            section_codes=["B.1", "B.4", "B.5"],
            domain_row_counts={"TS": 0, "TA": 6, "TE": 3, "TV": 0, "TI": 5},
            has_soa=True,
        )
        target_domains = {r["target_domain"] for r in records}
        assert "TS" not in target_domains
        assert "TV" not in target_domains

    def test_empty_inputs_empty_records(self) -> None:
        records = build_lineage_records(
            section_codes=[],
            domain_row_counts={},
        )
        assert records == []

    def test_record_value_equals_row_count(self) -> None:
        records = build_lineage_records(
            section_codes=["B.5"],
            domain_row_counts={"TI": 8},
        )
        assert records[0]["value"] == 8

    def test_record_keys(self) -> None:
        records = build_lineage_records(
            section_codes=["B.1"],
            domain_row_counts={"TS": 5},
        )
        rec = records[0]
        assert "source" in rec
        assert "target" in rec
        assert "value" in rec
        assert "target_domain" in rec

    def test_unknown_section_code_ignored(self) -> None:
        records = build_lineage_records(
            section_codes=["B.7", "B.9"],
            domain_row_counts=STANDARD_COUNTS,
        )
        assert records == []


# ---------------------------------------------------------------------------
# build_lineage_figure
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestBuildLineageFigure:
    """Tests for build_lineage_figure() — Plotly Sankey builder."""

    def test_figure_has_sankey_trace(self) -> None:
        records = build_lineage_records(
            section_codes=["B.1", "B.4", "B.5"],
            domain_row_counts=STANDARD_COUNTS,
            has_soa=True,
        )
        fig = build_lineage_figure(records, registry_id="NCT001")
        assert len(fig.data) == 1
        assert fig.data[0].type == "sankey"

    def test_figure_title_contains_registry_id(self) -> None:
        records = build_lineage_records(
            section_codes=["B.1"],
            domain_row_counts={"TS": 5},
        )
        fig = build_lineage_figure(records, registry_id="NCT00112827")
        assert "NCT00112827" in fig.layout.title.text

    def test_empty_records_shows_annotation(self) -> None:
        """[PTCV-36 Scenario: Non-ICH produces empty lineage]"""
        fig = build_lineage_figure([], registry_id="NCT001")
        assert len(fig.layout.annotations) > 0
        assert "No lineage" in fig.layout.annotations[0].text

    def test_sankey_node_labels(self) -> None:
        records = build_lineage_records(
            section_codes=["B.5"],
            domain_row_counts={"TI": 5},
        )
        fig = build_lineage_figure(records)
        labels = list(fig.data[0].node.label)
        assert "B.5 Selection of Subjects" in labels
        assert DOMAIN_LABELS["TI"] in labels

    def test_sankey_link_count(self) -> None:
        records = build_lineage_records(
            section_codes=["B.1", "B.4", "B.5"],
            domain_row_counts=STANDARD_COUNTS,
            has_soa=True,
        )
        fig = build_lineage_figure(records)
        assert len(fig.data[0].link.source) == 6

    def test_node_colors_source_vs_target(self) -> None:
        records = build_lineage_records(
            section_codes=["B.1", "B.4"],
            domain_row_counts={"TS": 5, "TA": 6, "TE": 3},
        )
        fig = build_lineage_figure(records)
        colors = list(fig.data[0].node.color)
        # Source nodes (B.1, B.4) get blue, targets get green
        assert colors[0] == "#636EFA"  # source
        assert colors[-1] == "#00CC96"  # target

    def test_figure_without_registry_id(self) -> None:
        records = build_lineage_records(
            section_codes=["B.1"],
            domain_row_counts={"TS": 5},
        )
        fig = build_lineage_figure(records)
        assert "Lineage" in fig.layout.title.text
