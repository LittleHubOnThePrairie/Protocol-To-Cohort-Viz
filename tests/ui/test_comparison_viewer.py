"""Tests for comparison viewer pure-Python helpers (PTCV-148).

No Streamlit dependency — only tests the data layer functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ui.pages.comparison_viewer import (
    CONF_HIGH,
    CONF_MODERATE,
    build_comparison_html,
    compute_corpus_summary,
    enrich_comparison_pairs,
    filter_pairs,
    format_run_comparison,
    get_next_protocol,
    get_prev_protocol,
    get_protocol_index,
    viewer_color,
    viewer_label,
)


# ===================================================================
# TestViewerColor
# ===================================================================

class TestViewerColor:
    """Confidence color coding with PTCV-148 thresholds."""

    def test_green_at_threshold(self) -> None:
        assert viewer_color(0.80) == "green"

    def test_green_above_threshold(self) -> None:
        assert viewer_color(0.95) == "green"

    def test_yellow_at_threshold(self) -> None:
        assert viewer_color(0.60) == "goldenrod"

    def test_yellow_in_range(self) -> None:
        assert viewer_color(0.70) == "goldenrod"

    def test_red_below_moderate(self) -> None:
        assert viewer_color(0.59) == "red"

    def test_red_at_zero(self) -> None:
        assert viewer_color(0.0) == "red"

    def test_grey_for_none(self) -> None:
        assert viewer_color(None) == "grey"

    def test_label_high(self) -> None:
        assert viewer_label(0.85) == "HIGH"

    def test_label_moderate(self) -> None:
        assert viewer_label(0.70) == "MODERATE"

    def test_label_low(self) -> None:
        assert viewer_label(0.40) == "LOW"

    def test_label_missing(self) -> None:
        assert viewer_label(None) == "MISSING"

    def test_threshold_constants(self) -> None:
        assert CONF_HIGH == 0.80
        assert CONF_MODERATE == 0.60


# ===================================================================
# TestComputeCorpusSummary
# ===================================================================

class TestComputeCorpusSummary:
    """Cross-protocol corpus summary computation."""

    def test_formats_section_stats(self) -> None:
        stats = [{
            "ich_section_code": "B.1",
            "ich_section_name": "Title Page",
            "avg_boosted": 0.90,
            "min_boosted": 0.85,
            "max_boosted": 0.95,
            "hit_rate": 0.95,
            "protocol_count": 190,
            "total_protocols": 200,
        }]
        result = compute_corpus_summary(stats)
        assert len(result) == 1
        r = result[0]
        assert r["ich_section_code"] == "B.1"
        assert r["avg_confidence"] == 0.9
        assert r["color"] == "green"
        assert r["label"] == "HIGH"
        assert r["hit_rate_pct"] == 95.0
        assert r["variance"] == 0.1

    def test_computes_variance(self) -> None:
        stats = [{
            "ich_section_code": "B.5",
            "ich_section_name": "Selection",
            "avg_boosted": 0.70,
            "min_boosted": 0.40,
            "max_boosted": 0.90,
            "hit_rate": 0.80,
            "protocol_count": 160,
            "total_protocols": 200,
        }]
        result = compute_corpus_summary(stats)
        assert result[0]["variance"] == 0.5

    def test_empty_stats(self) -> None:
        assert compute_corpus_summary([]) == []

    def test_low_confidence_gets_red(self) -> None:
        stats = [{
            "ich_section_code": "B.9",
            "ich_section_name": "Safety",
            "avg_boosted": 0.45,
            "min_boosted": 0.30,
            "max_boosted": 0.60,
            "hit_rate": 0.50,
            "protocol_count": 100,
            "total_protocols": 200,
        }]
        result = compute_corpus_summary(stats)
        assert result[0]["color"] == "red"
        assert result[0]["label"] == "LOW"


# ===================================================================
# TestEnrichComparisonPairs
# ===================================================================

class TestEnrichComparisonPairs:
    """Enrichment of comparison pairs with section match data."""

    def test_adds_match_metadata(self) -> None:
        pairs = [{
            "ich_section_code": "B.1",
            "original_text": "orig",
            "extracted_text": "extr",
            "confidence": 0.85,
            "match_quality": "good",
        }]
        matches = [{
            "ich_section_code": "B.1",
            "similarity_score": 0.82,
            "boosted_score": 0.87,
            "match_method": "embedding",
            "auto_mapped": 1,
        }]
        result = enrich_comparison_pairs(pairs, matches)
        assert len(result) == 1
        assert result[0]["similarity_score"] == 0.82
        assert result[0]["match_method"] == "embedding"
        assert result[0]["auto_mapped"] is True
        assert result[0]["color"] == "green"

    def test_unmatched_section_gets_defaults(self) -> None:
        pairs = [{
            "ich_section_code": "B.9",
            "confidence": 0.50,
            "match_quality": "poor",
        }]
        result = enrich_comparison_pairs(pairs, [])
        assert result[0]["similarity_score"] == 0.0
        assert result[0]["match_method"] == ""
        assert result[0]["auto_mapped"] is False

    def test_empty_inputs(self) -> None:
        assert enrich_comparison_pairs([], []) == []

    def test_best_match_selected(self) -> None:
        """When multiple matches exist, pick highest boosted."""
        pairs = [{
            "ich_section_code": "B.5",
            "confidence": 0.70,
        }]
        matches = [
            {
                "ich_section_code": "B.5",
                "similarity_score": 0.60,
                "boosted_score": 0.65,
                "match_method": "keyword",
                "auto_mapped": 0,
            },
            {
                "ich_section_code": "B.5",
                "similarity_score": 0.75,
                "boosted_score": 0.80,
                "match_method": "embedding",
                "auto_mapped": 1,
            },
        ]
        result = enrich_comparison_pairs(pairs, matches)
        assert result[0]["boosted_score"] == 0.80
        assert result[0]["match_method"] == "embedding"


# ===================================================================
# TestFilterPairs
# ===================================================================

class TestFilterPairs:
    """Filtering logic for comparison pairs."""

    @pytest.fixture()
    def sample_pairs(self) -> list[dict]:
        return [
            {
                "ich_section_code": "B.1",
                "confidence": 0.90,
                "match_quality": "good",
                "extracted_text": "content here",
            },
            {
                "ich_section_code": "B.5",
                "confidence": 0.50,
                "match_quality": "poor",
                "extracted_text": "some text",
            },
            {
                "ich_section_code": "B.9",
                "confidence": 0.70,
                "match_quality": "partial",
                "extracted_text": "",
            },
        ]

    def test_no_filters_returns_all(self, sample_pairs: list) -> None:
        assert len(filter_pairs(sample_pairs)) == 3

    def test_confidence_range(self, sample_pairs: list) -> None:
        result = filter_pairs(
            sample_pairs, confidence_min=0.60, confidence_max=0.95,
        )
        codes = [p["ich_section_code"] for p in result]
        assert "B.1" in codes
        assert "B.9" in codes
        assert "B.5" not in codes

    def test_ich_section_filter(self, sample_pairs: list) -> None:
        result = filter_pairs(sample_pairs, ich_section="B.5")
        assert len(result) == 1
        assert result[0]["ich_section_code"] == "B.5"

    def test_match_quality_filter(self, sample_pairs: list) -> None:
        result = filter_pairs(sample_pairs, match_quality="poor")
        assert len(result) == 1
        assert result[0]["match_quality"] == "poor"

    def test_populated_only(self, sample_pairs: list) -> None:
        result = filter_pairs(sample_pairs, populated_only=True)
        assert len(result) == 2
        assert all(p.get("extracted_text", "").strip() for p in result)

    def test_unpopulated_only(self, sample_pairs: list) -> None:
        result = filter_pairs(sample_pairs, unpopulated_only=True)
        assert len(result) == 1
        assert result[0]["ich_section_code"] == "B.9"

    def test_combined_filters(self, sample_pairs: list) -> None:
        result = filter_pairs(
            sample_pairs,
            confidence_min=0.80,
            populated_only=True,
        )
        assert len(result) == 1
        assert result[0]["ich_section_code"] == "B.1"

    def test_none_confidence_passes_when_no_range(self) -> None:
        pairs = [{"confidence": None, "extracted_text": "x"}]
        result = filter_pairs(pairs)
        assert len(result) == 1


# ===================================================================
# TestFormatRunComparison
# ===================================================================

class TestFormatRunComparison:
    """Run diff formatting with color coding."""

    def test_improved_gets_green(self) -> None:
        comparison = {"sections": [{
            "ich_section_code": "B.5",
            "run_a_avg_score": 0.50,
            "run_b_avg_score": 0.75,
            "delta": 0.25,
            "status": "improved",
        }]}
        rows = format_run_comparison(comparison)
        assert rows[0]["color"] == "green"
        assert rows[0]["status"] == "improved"

    def test_regressed_gets_red(self) -> None:
        comparison = {"sections": [{
            "ich_section_code": "B.9",
            "run_a_avg_score": 0.80,
            "run_b_avg_score": 0.60,
            "delta": -0.20,
            "status": "regressed",
        }]}
        rows = format_run_comparison(comparison)
        assert rows[0]["color"] == "red"

    def test_unchanged_gets_grey(self) -> None:
        comparison = {"sections": [{
            "ich_section_code": "B.1",
            "run_a_avg_score": 0.85,
            "run_b_avg_score": 0.855,
            "delta": 0.005,
            "status": "unchanged",
        }]}
        rows = format_run_comparison(comparison)
        assert rows[0]["color"] == "grey"

    def test_empty_sections(self) -> None:
        assert format_run_comparison({"sections": []}) == []
        assert format_run_comparison({}) == []


# ===================================================================
# TestNavigation
# ===================================================================

class TestNavigation:
    """Protocol navigation helpers."""

    @pytest.fixture()
    def protocols(self) -> list[dict]:
        return [
            {"nct_id": "NCT001"},
            {"nct_id": "NCT002"},
            {"nct_id": "NCT003"},
        ]

    def test_get_next_protocol(self, protocols: list) -> None:
        assert get_next_protocol(protocols, "NCT001") == "NCT002"
        assert get_next_protocol(protocols, "NCT002") == "NCT003"

    def test_next_at_end_returns_none(self, protocols: list) -> None:
        assert get_next_protocol(protocols, "NCT003") is None

    def test_get_prev_protocol(self, protocols: list) -> None:
        assert get_prev_protocol(protocols, "NCT002") == "NCT001"
        assert get_prev_protocol(protocols, "NCT003") == "NCT002"

    def test_prev_at_start_returns_none(self, protocols: list) -> None:
        assert get_prev_protocol(protocols, "NCT001") is None

    def test_get_protocol_index(self, protocols: list) -> None:
        assert get_protocol_index(protocols, "NCT002") == 1

    def test_index_not_found_returns_zero(self, protocols: list) -> None:
        assert get_protocol_index(protocols, "NCT999") == 0

    def test_next_unknown_nct(self, protocols: list) -> None:
        # Unknown NCT → index 0 → next is NCT002
        assert get_next_protocol(protocols, "NCT999") == "NCT002"

    def test_empty_list(self) -> None:
        assert get_next_protocol([], "NCT001") is None
        assert get_prev_protocol([], "NCT001") is None
        assert get_protocol_index([], "NCT001") == 0


# ===================================================================
# TestBuildComparisonHtml
# ===================================================================

class TestBuildComparisonHtml:
    """HTML comparison builder."""

    def test_empty_pairs_returns_message(self) -> None:
        html = build_comparison_html([])
        assert "No comparison pairs" in html

    def test_contains_section_code(self) -> None:
        pairs = [{
            "ich_section_code": "B.5",
            "confidence": 0.85,
            "match_quality": "good",
            "match_method": "embedding",
            "similarity_score": 0.80,
            "boosted_score": 0.85,
            "original_text": "Original content",
            "extracted_text": "Extracted content",
        }]
        html = build_comparison_html(pairs)
        assert "B.5" in html
        assert "Original content" in html
        assert "Extracted content" in html

    def test_missing_content_gets_placeholder(self) -> None:
        pairs = [{
            "ich_section_code": "B.9",
            "confidence": None,
            "match_quality": "missing",
            "match_method": "",
            "similarity_score": 0.0,
            "boosted_score": 0.0,
            "original_text": "",
            "extracted_text": "",
        }]
        html = build_comparison_html(pairs)
        assert "(no original content)" in html
        assert "(no extracted content)" in html
        assert "MISSING" in html

    def test_contains_sync_scroll_js(self) -> None:
        pairs = [{
            "ich_section_code": "B.1",
            "confidence": 0.90,
            "match_quality": "good",
            "match_method": "keyword",
            "similarity_score": 0.88,
            "boosted_score": 0.90,
            "original_text": "text",
            "extracted_text": "text",
        }]
        html = build_comparison_html(pairs)
        assert "cvSync" in html
        assert "jumpTo" in html

    def test_escapes_html_content(self) -> None:
        pairs = [{
            "ich_section_code": "B.1",
            "confidence": 0.80,
            "match_quality": "good",
            "match_method": "regex",
            "similarity_score": 0.78,
            "boosted_score": 0.80,
            "original_text": "<script>alert('xss')</script>",
            "extracted_text": "safe text",
        }]
        html = build_comparison_html(pairs)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
