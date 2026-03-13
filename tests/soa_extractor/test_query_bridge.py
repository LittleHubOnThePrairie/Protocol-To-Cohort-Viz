"""Tests for query_bridge — PTCV-112.

Verifies the adapter that converts AssembledProtocol sections into
IchSection objects for the SoA extraction pipeline.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    AssembledSection,
    CoverageReport,
    QueryExtractionHit,
    SourceReference,
)
from ptcv.soa_extractor.query_bridge import (
    assembled_to_sections,
    get_assembled_protocol,
    get_classified_assembled_protocol,
    has_classified_results,
    has_query_pipeline_results,
    store_classified_result,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_hit(
    query_id: str,
    section_id: str,
    parent: str,
    content: str,
    confidence: float = 0.85,
) -> QueryExtractionHit:
    return QueryExtractionHit(
        query_id=query_id,
        section_id=section_id,
        parent_section=parent,
        query_text=f"Query for {section_id}",
        extracted_content=content,
        confidence=confidence,
    )


def _make_section(
    code: str,
    name: str,
    hits: list[QueryExtractionHit] | None = None,
    populated: bool = True,
) -> AssembledSection:
    hits = hits or []
    avg = (
        sum(h.confidence for h in hits) / len(hits) if hits else 0.0
    )
    return AssembledSection(
        section_code=code,
        section_name=name,
        populated=populated,
        hits=hits,
        average_confidence=avg,
        is_gap=not populated,
        has_low_confidence=any(h.confidence < 0.70 for h in hits),
        required_query_count=len(hits),
        answered_required_count=len(hits),
    )


def _make_protocol(
    sections: list[AssembledSection],
) -> AssembledProtocol:
    return AssembledProtocol(
        sections=sections,
        coverage=CoverageReport(
            total_sections=16,
            populated_count=len(sections),
            gap_count=16 - len(sections),
            average_confidence=0.80,
            high_confidence_count=0,
            medium_confidence_count=len(sections),
            low_confidence_count=0,
            total_queries=10,
            answered_queries=5,
            required_queries=10,
            answered_required=5,
            gap_sections=[],
            low_confidence_sections=[],
        ),
        source_traceability={},
    )


SOA_TABLE = (
    "| Assessment | Screening | Week 2 |\n"
    "|------------|-----------|--------|\n"
    "| ECG        | X         | X      |\n"
    "| Labs       | X         |        |\n"
)


# -----------------------------------------------------------------------
# TestAssembledToSections
# -----------------------------------------------------------------------


class TestAssembledToSections:
    """assembled_to_sections() converts relevant sections."""

    def test_extracts_all_populated_sections(self) -> None:
        """All populated sections (B.1, B.4, B.7) are converted."""
        protocol = _make_protocol([
            _make_section("B.1", "General", [
                _make_hit("B.1.q1", "B.1", "B.1", "Title"),
            ]),
            _make_section("B.4", "Trial Design", [
                _make_hit("B.4.q1", "B.4", "B.4", SOA_TABLE),
            ]),
            _make_section("B.7", "Schedule of Activities", [
                _make_hit("B.7.q1", "B.7", "B.7", "Visit schedule"),
            ]),
        ])

        result = assembled_to_sections(
            protocol, registry_id="NCT001",
        )

        assert len(result) == 3
        codes = {s.section_code for s in result}
        assert codes == {"B.1", "B.4", "B.7"}

    def test_content_json_has_text_field(self) -> None:
        """content_json wraps content as {"text": "..."}."""
        protocol = _make_protocol([
            _make_section("B.4", "Trial Design", [
                _make_hit("B.4.q1", "B.4", "B.4", SOA_TABLE),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 1

        data = json.loads(result[0].content_json)
        assert "text" in data
        assert "ECG" in data["text"]

    def test_skips_unpopulated_sections(self) -> None:
        """Sections with populated=False are not included."""
        protocol = _make_protocol([
            _make_section(
                "B.4", "Trial Design", populated=False,
            ),
            _make_section("B.7", "Schedule", [
                _make_hit("B.7.q1", "B.7", "B.7", "data"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 1
        assert result[0].section_code == "B.7"

    def test_skips_empty_content(self) -> None:
        """Sections where all hits have empty content are skipped."""
        protocol = _make_protocol([
            _make_section("B.4", "Trial Design", [
                _make_hit("B.4.q1", "B.4", "B.4", ""),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 0

    def test_concatenates_multiple_hits(self) -> None:
        """Multiple hits in one section are joined."""
        protocol = _make_protocol([
            _make_section("B.7", "Schedule", [
                _make_hit("B.7.q1", "B.7", "B.7", "Part A"),
                _make_hit("B.7.q2", "B.7", "B.7", "Part B"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert len(result) == 1
        data = json.loads(result[0].content_json)
        assert "Part A" in data["text"]
        assert "Part B" in data["text"]

    def test_preserves_registry_id(self) -> None:
        """registry_id is passed through to IchSection."""
        protocol = _make_protocol([
            _make_section("B.4", "Design", [
                _make_hit("B.4.q1", "B.4", "B.4", "content"),
            ]),
        ])

        result = assembled_to_sections(
            protocol, registry_id="NCT02856802",
        )
        assert result[0].registry_id == "NCT02856802"

    def test_source_run_id_is_query_pipeline(self) -> None:
        """source_run_id is set to 'query_pipeline' for traceability."""
        protocol = _make_protocol([
            _make_section("B.4", "Design", [
                _make_hit("B.4.q1", "B.4", "B.4", "content"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert result[0].source_run_id == "query_pipeline"

    def test_confidence_from_section_average(self) -> None:
        """confidence_score reflects the section average."""
        protocol = _make_protocol([
            _make_section("B.4", "Design", [
                _make_hit("B.4.q1", "B.4", "B.4", "text", 0.90),
                _make_hit("B.4.q2", "B.4", "B.4", "more", 0.80),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert result[0].confidence_score == pytest.approx(0.85)

    def test_returns_empty_when_no_populated_sections(self) -> None:
        """Protocol with no populated sections returns empty list."""
        protocol = _make_protocol([
            _make_section("B.1", "General", populated=False),
            _make_section("B.4", "Design", populated=False),
        ])

        result = assembled_to_sections(protocol)
        assert result == []


# -----------------------------------------------------------------------
# TestFullSectionCoverage (PTCV-136)
# -----------------------------------------------------------------------


class TestFullSectionCoverage:
    """All 14 ICH sections (B.1–B.14) are bridged when populated."""

    ALL_CODES = [f"B.{i}" for i in range(1, 15)]
    ALL_NAMES = {
        "B.1": "General Information",
        "B.2": "Background Information",
        "B.3": "Trial Objectives and Purpose",
        "B.4": "Trial Design",
        "B.5": "Selection of Subjects",
        "B.6": "Discontinuation",
        "B.7": "Treatment of Participants",
        "B.8": "Assessment of Efficacy",
        "B.9": "Assessment of Safety",
        "B.10": "Statistics",
        "B.11": "Direct Access",
        "B.12": "Ethics",
        "B.13": "Data Handling",
        "B.14": "Financing and Insurance",
    }

    def test_all_14_sections_bridged(self) -> None:
        """Given all 14 sections populated, all are converted."""
        sections = [
            _make_section(code, self.ALL_NAMES[code], [
                _make_hit(
                    f"{code}.q1", code, code,
                    f"Content for {code}",
                ),
            ])
            for code in self.ALL_CODES
        ]
        protocol = _make_protocol(sections)

        result = assembled_to_sections(protocol)
        result_codes = {s.section_code for s in result}
        assert result_codes == set(self.ALL_CODES)
        assert len(result) == 14

    def test_fallback_names_used_when_section_name_missing(
        self,
    ) -> None:
        """Given section_name is None, fallback from _CODE_NAMES."""
        section = _make_section("B.12", "", [
            _make_hit("B.12.q1", "B.12", "B.12", "Ethics text"),
        ])
        # Clear section_name to trigger fallback.
        section = dataclasses.replace(section, section_name="")
        protocol = _make_protocol([section])

        result = assembled_to_sections(protocol)
        assert len(result) == 1
        assert result[0].section_name == "Ethics"

    def test_mixed_populated_and_unpopulated(self) -> None:
        """Only populated sections appear in output."""
        protocol = _make_protocol([
            _make_section("B.1", "General", [
                _make_hit("B.1.q1", "B.1", "B.1", "Title"),
            ]),
            _make_section("B.2", "Background", populated=False),
            _make_section("B.3", "Objectives", [
                _make_hit("B.3.q1", "B.3", "B.3", "Objective"),
            ]),
            _make_section("B.5", "Subjects", populated=False),
            _make_section("B.9", "Safety", [
                _make_hit("B.9.q1", "B.9", "B.9", "AE data"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        codes = {s.section_code for s in result}
        assert codes == {"B.1", "B.3", "B.9"}

    def test_unknown_section_code_ignored(self) -> None:
        """Sections outside B.1–B.14 are not bridged."""
        protocol = _make_protocol([
            _make_section("B.99", "Unknown", [
                _make_hit("B.99.q1", "B.99", "B.99", "Mystery"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        assert result == []

    def test_section_order_follows_ich(self) -> None:
        """Sections appear in ICH order (B.1 before B.14)."""
        protocol = _make_protocol([
            _make_section("B.14", "Financing", [
                _make_hit("B.14.q1", "B.14", "B.14", "Budget"),
            ]),
            _make_section("B.3", "Objectives", [
                _make_hit("B.3.q1", "B.3", "B.3", "Goals"),
            ]),
            _make_section("B.1", "General", [
                _make_hit("B.1.q1", "B.1", "B.1", "Title"),
            ]),
        ])

        result = assembled_to_sections(protocol)
        result_codes = [s.section_code for s in result]
        assert result_codes == ["B.1", "B.3", "B.14"]


# -----------------------------------------------------------------------
# TestSessionStateHelpers
# -----------------------------------------------------------------------


class TestSessionStateHelpers:
    """has_query_pipeline_results and get_assembled_protocol."""

    def test_has_results_empty(self) -> None:
        assert has_query_pipeline_results({}) is False
        assert has_query_pipeline_results(
            {"query_cache": {}}
        ) is False

    def test_has_results_populated(self) -> None:
        assert has_query_pipeline_results(
            {"query_cache": {"abc_v1_t1": {"assembled": "x"}}}
        ) is True

    def test_get_assembled_found(self) -> None:
        assembled = _make_protocol([])
        state = {
            "query_cache": {
                "sha123_v2_t1": {"assembled": assembled},
            },
        }
        result = get_assembled_protocol(state, "sha123")
        assert result is assembled

    def test_get_assembled_not_found(self) -> None:
        state = {
            "query_cache": {
                "sha999_v2_t1": {"assembled": "proto"},
            },
        }
        result = get_assembled_protocol(state, "sha123")
        assert result is None

    def test_get_assembled_no_cache(self) -> None:
        result = get_assembled_protocol({}, "sha123")
        assert result is None


# -----------------------------------------------------------------------
# TestClassifiedCacheHelpers (PTCV-179)
# -----------------------------------------------------------------------


class TestClassifiedCacheHelpers:
    """Classified pipeline session state helpers."""

    def test_has_classified_results_empty(self) -> None:
        assert has_classified_results({}, "sha123") is False

    def test_has_classified_results_no_cache_key(self) -> None:
        assert has_classified_results(
            {"classified_cache": {}}, "sha123",
        ) is False

    def test_has_classified_results_populated(self) -> None:
        state = {
            "classified_cache": {
                "sha123": {"assembled": _make_protocol([])},
            },
        }
        assert has_classified_results(state, "sha123") is True

    def test_has_classified_results_wrong_sha(self) -> None:
        state = {
            "classified_cache": {
                "sha999": {"assembled": _make_protocol([])},
            },
        }
        assert has_classified_results(state, "sha123") is False

    def test_get_classified_found(self) -> None:
        assembled = _make_protocol([])
        state = {
            "classified_cache": {
                "sha123": {"assembled": assembled},
            },
        }
        result = get_classified_assembled_protocol(state, "sha123")
        assert result is assembled

    def test_get_classified_not_found(self) -> None:
        result = get_classified_assembled_protocol({}, "sha123")
        assert result is None

    def test_get_classified_wrong_sha(self) -> None:
        state = {
            "classified_cache": {
                "sha999": {"assembled": _make_protocol([])},
            },
        }
        result = get_classified_assembled_protocol(state, "sha123")
        assert result is None

    def test_store_and_retrieve(self) -> None:
        state: dict = {}
        assembled = _make_protocol([
            _make_section("B.4", "Design", [
                _make_hit("B.4.q1", "B.4", "B.4", "content"),
            ]),
        ])
        store_classified_result(state, "sha123", assembled)
        result = get_classified_assembled_protocol(state, "sha123")
        assert result is assembled

    def test_store_with_cascade_stats(self) -> None:
        state: dict = {}
        assembled = _make_protocol([])
        stats = {"local_count": 8, "sonnet_count": 2}
        store_classified_result(
            state, "sha123", assembled,
            cascade_stats=stats,
        )
        entry = state["classified_cache"]["sha123"]
        assert entry["cascade_stats"] == stats
        assert entry["assembled"] is assembled

    def test_store_with_stage_timings(self) -> None:
        state: dict = {}
        assembled = _make_protocol([])
        timings = {"extraction": 2.0, "classification": 5.0}
        store_classified_result(
            state, "sha123", assembled,
            stage_timings=timings,
        )
        entry = state["classified_cache"]["sha123"]
        assert entry["stage_timings"] == timings

    def test_store_creates_cache_key(self) -> None:
        """store_classified_result auto-creates classified_cache."""
        state: dict = {}
        store_classified_result(
            state, "sha123", _make_protocol([]),
        )
        assert "classified_cache" in state
        assert "sha123" in state["classified_cache"]

    def test_store_overwrites_existing(self) -> None:
        state: dict = {}
        proto1 = _make_protocol([])
        proto2 = _make_protocol([
            _make_section("B.1", "General", [
                _make_hit("B.1.q1", "B.1", "B.1", "content"),
            ]),
        ])
        store_classified_result(state, "sha123", proto1)
        store_classified_result(state, "sha123", proto2)
        result = get_classified_assembled_protocol(state, "sha123")
        assert result is proto2
