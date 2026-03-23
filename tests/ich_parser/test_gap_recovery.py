"""Tests for gap recovery pipeline (PTCV-258).

Feature: Gap recovery for failed query extractions

  Scenario: Alternative strategy recovers extraction
  Scenario: Adjacent section search finds content
  Scenario: Registry provides query-specific answer
  Scenario: Exhaustive failure recorded
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from ptcv.ich_parser.gap_recovery import (
    RecoveryResult,
    recover_gap,
    _try_alternative_strategy,
    _try_adjacent_sections,
    _try_registry_fallback,
)
from ptcv.ich_parser.query_schema import AppendixBQuery
from ptcv.ich_parser.section_matcher import MatchConfidence


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_query(
    query_id: str = "B.4.1.q1",
    section_id: str = "B.4.1",
    parent_section: str = "B.4",
    expected_type: str = "text_long",
) -> AppendixBQuery:
    return AppendixBQuery(
        query_id=query_id,
        section_id=section_id,
        parent_section=parent_section,
        schema_section=parent_section,
        query_text="What are the primary endpoints?",
        expected_type=expected_type,
        required=True,
    )


def _simple_extractor(
    text: str, query: AppendixBQuery,
) -> tuple[str, float, str]:
    """A mock extractor that returns the text if non-empty."""
    if text and text.strip():
        return text.strip()[:200], 0.75, "mock_extract"
    return "", 0.0, "mock_extract"


def _failing_extractor(
    text: str, query: AppendixBQuery,
) -> tuple[str, float, str]:
    """An extractor that always fails."""
    return "", 0.0, "mock_fail"


def _list_extractor(
    text: str, query: AppendixBQuery,
) -> tuple[str, float, str]:
    """Extractor that succeeds only if text has list markers."""
    lines = [
        l for l in text.splitlines()
        if l.strip().startswith(("-", "•", "1."))
    ]
    if len(lines) >= 2:
        return "\n".join(lines), 0.80, "list"
    return "", 0.0, "list"


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestAlternativeStrategy:
    """Scenario: Alternative strategy recovers extraction."""

    def test_list_strategy_recovers_text_long_failure(self) -> None:
        """Given query B.4.1 failed with text_long strategy
        And the content contains endpoints in a bulleted list
        When gap recovery tries the list strategy
        Then the endpoints are extracted successfully."""
        query = _make_query(expected_type="text_long")
        routes = {
            "B.4.1": (
                "Endpoints:\n- Overall survival\n- PFS\n- ORR\n",
                "B.4", MatchConfidence.HIGH,
            ),
        }
        dispatch = {
            "text_long": _failing_extractor,
            "list": _list_extractor,
            "table": _failing_extractor,
        }

        result = recover_gap(query, routes, dispatch)

        assert result.recovered is True
        assert "gap_recovery" in result.method
        assert "survival" in result.content.lower() or "PFS" in result.content

    def test_no_alternatives_for_unknown_type(self) -> None:
        query = _make_query(expected_type="unknown_type")
        routes = {"B.4.1": ("content", "B.4", MatchConfidence.HIGH)}
        dispatch = {"text_long": _simple_extractor}

        result = recover_gap(query, routes, dispatch)
        # May or may not recover — but should not crash
        assert isinstance(result, RecoveryResult)


class TestAdjacentSectionSearch:
    """Scenario: Adjacent section search finds content."""

    def test_finds_content_in_neighbor(self) -> None:
        """Given B.10 (Statistics) route is empty
        And B.4 (Trial Design) contains sample size information
        When gap recovery broadens to adjacent sections
        Then the sample size is found in B.4 content."""
        query = _make_query(
            query_id="B.10.q1",
            section_id="B.10.1",
            parent_section="B.10",
            expected_type="text_long",
        )
        # B.10 route is missing, but B.4 has relevant content
        routes = {
            "B.4": (
                "Sample size: 500 participants will be enrolled.",
                "B.4", MatchConfidence.HIGH,
            ),
        }
        dispatch = {
            "text_long": _simple_extractor,
            "list": _failing_extractor,
            "table": _failing_extractor,
        }

        result = recover_gap(query, routes, dispatch)

        assert result.recovered is True
        assert "adjacent_recovery" in result.method
        assert "500" in result.content
        # Cross-section penalty applied
        assert result.confidence < 0.75

    def test_no_neighbors_available(self) -> None:
        query = _make_query(
            query_id="B.10.q1",
            section_id="B.10.1",
            parent_section="B.10",
        )
        # No routes at all
        routes: dict = {}
        dispatch = {"text_long": _simple_extractor}

        result = recover_gap(query, routes, dispatch)
        assert result.recovered is False


class TestRegistryFallback:
    """Scenario: Registry provides query-specific answer."""

    def test_registry_provides_protocol_title(self) -> None:
        """Given query B.1.1 (protocol title) has no PDF match
        And CT.gov registry has the officialTitle field
        When gap recovery falls back to registry
        Then the title is extracted from registry metadata."""
        query = _make_query(
            query_id="B.1.1.q1",
            section_id="B.1.1",
            parent_section="B.1",
            expected_type="text_short",
        )
        routes: dict = {}
        dispatch = {"text_short": _failing_extractor}

        registry = {
            "protocolSection": {
                "identificationModule": {
                    "officialTitle": (
                        "A Phase 3 Study of Drug X in Condition Y"
                    ),
                },
            },
        }

        result = recover_gap(
            query, routes, dispatch,
            registry_metadata=registry,
        )

        assert result.recovered is True
        assert "registry" in result.method.lower()
        assert "Drug X" in result.content
        assert result.source_section == "[REGISTRY]"

    def test_no_registry_metadata(self) -> None:
        query = _make_query(query_id="B.1.1.q1")
        routes: dict = {}
        dispatch = {"text_long": _failing_extractor}

        result = recover_gap(query, routes, dispatch, registry_metadata=None)
        assert result.recovered is False

    def test_query_not_in_registry_map(self) -> None:
        query = _make_query(
            query_id="B.99.q1",  # Not in QUERY_TO_REGISTRY_FIELD
            section_id="B.99",
            parent_section="B.99",
        )
        routes: dict = {}
        dispatch = {"text_long": _failing_extractor}
        registry = {"protocolSection": {}}

        result = recover_gap(
            query, routes, dispatch, registry_metadata=registry,
        )
        assert result.recovered is False


class TestExhaustiveFailure:
    """Scenario: Exhaustive failure recorded."""

    def test_all_levels_fail(self) -> None:
        """Given all 4 recovery levels fail for a query
        When the gap is recorded
        Then it includes an exhaustive_search flag
        And lists which recovery strategies were attempted."""
        query = _make_query(
            query_id="B.4.1.q1",
            expected_type="text_long",
        )
        # Route exists but all extractors fail
        routes = {
            "B.4.1": ("some content", "B.4", MatchConfidence.HIGH),
            "B.3": ("other content", "B.3", MatchConfidence.HIGH),
        }
        dispatch = {
            "text_long": _failing_extractor,
            "list": _failing_extractor,
            "table": _failing_extractor,
        }

        result = recover_gap(query, routes, dispatch)

        assert result.recovered is False
        assert result.exhaustive_search is True
        assert len(result.attempts) > 0
        assert len(result.strategies_attempted) > 0

    def test_no_attempts_when_no_routes(self) -> None:
        """If there's nothing to try, attempts list reflects that."""
        query = _make_query(
            query_id="B.99.q1",
            section_id="B.99",
            parent_section="B.99",
            expected_type="text_long",
        )
        routes: dict = {}
        dispatch = {"text_long": _failing_extractor}

        result = recover_gap(query, routes, dispatch)
        assert result.recovered is False
        # No routes → no alternative/adjacent attempts possible
        # (but not exhaustive since nothing was tried)


class TestRecoveryResult:
    """Tests for RecoveryResult properties."""

    def test_exhaustive_search_flag(self) -> None:
        result = RecoveryResult(recovered=False, attempts=[
            MagicMock(strategy="alt_strategy:list"),
        ])
        assert result.exhaustive_search is True

    def test_not_exhaustive_when_recovered(self) -> None:
        result = RecoveryResult(recovered=True, attempts=[
            MagicMock(strategy="alt_strategy:list"),
        ])
        assert result.exhaustive_search is False

    def test_strategies_attempted(self) -> None:
        result = RecoveryResult(recovered=False, attempts=[
            MagicMock(strategy="alt_strategy:list"),
            MagicMock(strategy="adjacent:B.3:text_long"),
            MagicMock(strategy="registry_fallback"),
        ])
        assert result.strategies_attempted == [
            "alt_strategy:list",
            "adjacent:B.3:text_long",
            "registry_fallback",
        ]
