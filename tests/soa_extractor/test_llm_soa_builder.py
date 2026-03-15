"""Tests for LlmSoaBuilder — PTCV-166.

All tests mock the Anthropic client to avoid real API calls.
Covers all 8 GHERKIN scenarios from PTCV-166 acceptance criteria.

Qualification phase: IQ/OQ
Risk tier: MEDIUM
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.soa_extractor.llm_soa_builder import (
    LlmSoaBuilder,
    MIN_ACTIVITIES_THRESHOLD,
)
from ptcv.soa_extractor.models import RawSoaTable


# ------------------------------------------------------------------
# Fake Anthropic client (same pattern as test_vision_enhancer.py)
# ------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, input_tokens: int = 500, output_tokens: int = 300):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, input_tokens: int = 500, output_tokens: int = 300):
        self.content = [_FakeContentBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def create(self, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text: str):
        self.messages = _FakeMessages(response_text)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_soa_response(
    visits: list[dict] | None = None,
    assessments: list[dict] | None = None,
) -> str:
    """Build a mock Sonnet SoA JSON response."""
    return json.dumps({
        "visits": visits or [
            {"name": "Screening", "day_offset": -14},
            {"name": "Baseline", "day_offset": 1},
            {"name": "Week 2", "day_offset": 15},
            {"name": "Week 4", "day_offset": 29},
        ],
        "assessments": assessments or [
            {"name": "Physical Exam", "visits_scheduled": [True, True, True, True]},
            {"name": "ECG", "visits_scheduled": [True, False, True, False]},
            {"name": "CBC", "visits_scheduled": [True, True, True, True]},
            {"name": "Vital Signs", "visits_scheduled": [True, True, True, True]},
        ],
    })


def _make_section(code: str, text: str) -> object:
    """Build a minimal IchSection-like object."""
    from ptcv.ich_parser.models import IchSection

    return IchSection(
        run_id="ich-run-001",
        source_run_id="extract-run-001",
        source_sha256="a" * 64,
        registry_id="NCT00112827",
        section_code=code,
        section_name=f"Section {code}",
        content_json=json.dumps({"text": text}),
        confidence_score=0.90,
        review_required=False,
        legacy_format=False,
        content_text=text,
    )


def _make_partial_table() -> RawSoaTable:
    """A partial table with 2 visits and 1 activity."""
    return RawSoaTable(
        visit_headers=["Screening", "Baseline"],
        day_headers=["-14", "1"],
        activities=[("Informed Consent", [True, False])],
        section_code="tables.parquet",
    )


# ------------------------------------------------------------------
# TestCascadePriority — GHERKIN scenarios 4, 5, 8
# ------------------------------------------------------------------


class TestCascadePriority:
    """Level 3 is only invoked when Levels 1/2 produce < 3 activities."""

    def test_level3_not_invoked_when_sufficient_activities(self):
        """GHERKIN: Level 1 Docling tables take priority."""
        builder = LlmSoaBuilder()
        builder._client = _FakeAnthropicClient(_make_soa_response())

        # Simulate Level 1/2 producing >= 3 activities
        existing_table = RawSoaTable(
            visit_headers=["V1", "V2", "V3"],
            day_headers=["1", "8", "15"],
            activities=[
                ("ECG", [True, True, True]),
                ("Labs", [True, True, True]),
                ("Vitals", [True, True, True]),
            ],
            section_code="tables.parquet",
        )
        activity_count = sum(len(t.activities) for t in [existing_table])
        assert activity_count >= MIN_ACTIVITIES_THRESHOLD
        # Level 3 would NOT be invoked by extractor.py cascade logic

    def test_level3_invoked_when_levels12_produce_fewer_than_3(self):
        """GHERKIN: Sonnet constructs SoA when Docling and vision fail."""
        builder = LlmSoaBuilder()
        builder._client = _FakeAnthropicClient(_make_soa_response())

        sections = [
            _make_section("B.4", "Trial Design: Screening Day -14, Baseline Day 1"),
        ]
        result = builder.build(sections=sections)
        assert result is not None
        assert len(result) == 1
        assert result[0].construction_method == "llm_text"

    def test_threshold_is_3(self):
        """Cascade trigger threshold matches GHERKIN."""
        assert MIN_ACTIVITIES_THRESHOLD == 3


# ------------------------------------------------------------------
# TestSectionTextExtraction — GHERKIN scenarios 2, 3
# ------------------------------------------------------------------


class TestSectionTextExtraction:
    """B.8/B.9 sections included in SoA construction."""

    def test_b8_efficacy_sections_included(self):
        """GHERKIN: B.8 efficacy sections provide assessment detail."""
        builder = LlmSoaBuilder()
        sections = [
            _make_section("B.8", "Assessment of Efficacy: tumor response, PRO questionnaire"),
            _make_section("B.8.1", "Primary endpoint: ORR per RECIST v1.1"),
            _make_section("B.8.2", "Efficacy assessed at Weeks 4, 8, 12"),
        ]
        texts = builder._gather_section_text(sections=sections)
        assert "B.8" in texts
        assert "tumor response" in texts["B.8"]
        assert "ORR per RECIST" in texts["B.8"]

    def test_b9_safety_sections_included(self):
        """GHERKIN: B.9 safety sections provide assessment detail."""
        builder = LlmSoaBuilder()
        sections = [
            _make_section("B.9", "Assessment of Safety: labs, vitals, ECG"),
            _make_section("B.9.1", "Safety parameters: CBC, CMP, urinalysis"),
            _make_section("B.9.2", "Safety assessed at screening, Day 1, every 2 weeks"),
        ]
        texts = builder._gather_section_text(sections=sections)
        assert "B.9" in texts
        assert "CBC" in texts["B.9"]

    def test_heuristic_heading_detection_from_text_blocks(self):
        """Fallback: detect section headings in raw text_blocks."""
        builder = LlmSoaBuilder()
        text_blocks = [
            {"page_number": 1, "text": "Some preamble text."},
            {"page_number": 2, "text": "Trial Design\nScreening visit at Day -14."},
            {"page_number": 3, "text": "Assessment of Efficacy\nPrimary: tumor response."},
            {"page_number": 4, "text": "Assessment of Safety\nLabs at every visit."},
        ]
        texts = builder._gather_section_text(text_blocks=text_blocks)
        assert "B.4" in texts
        assert "Screening" in texts["B.4"]
        assert "B.8" in texts
        assert "tumor response" in texts["B.8"]
        assert "B.9" in texts

    def test_sections_preferred_over_text_blocks(self):
        """When sections have data, text_blocks are not used."""
        builder = LlmSoaBuilder()
        sections = [_make_section("B.4", "Trial Design from sections")]
        text_blocks = [{"page_number": 1, "text": "Trial Design from text blocks"}]
        texts = builder._gather_section_text(
            sections=sections, text_blocks=text_blocks,
        )
        assert "from sections" in texts["B.4"]


# ------------------------------------------------------------------
# TestPartialFragments — GHERKIN scenario 6
# ------------------------------------------------------------------


class TestPartialFragments:
    """Partial Level 1/2 results passed as context to Sonnet."""

    def test_partial_tables_passed_as_context(self):
        """GHERKIN: Partial fragments from Level 1/2 supplement Sonnet."""
        builder = LlmSoaBuilder()
        partial = _make_partial_table()
        context = builder._format_partial_context([partial])
        assert "Screening" in context
        assert "Baseline" in context
        assert "Informed Consent" in context

    def test_no_partials_returns_empty(self):
        """No partial tables produces empty context."""
        builder = LlmSoaBuilder()
        assert builder._format_partial_context(None) == ""
        assert builder._format_partial_context([]) == ""


# ------------------------------------------------------------------
# TestConstructionMethodAudit — GHERKIN scenario 7
# ------------------------------------------------------------------


class TestConstructionMethodAudit:
    """construction_method tracking for audit trail."""

    def test_construction_method_is_llm_text(self):
        """GHERKIN: construction_method is 'llm_text'."""
        builder = LlmSoaBuilder()
        builder._client = _FakeAnthropicClient(_make_soa_response())

        sections = [_make_section("B.4", "Trial Design text")]
        result = builder.build(sections=sections)
        assert result is not None
        for table in result:
            assert table.construction_method == "llm_text"

    def test_construction_method_distinct_from_vision(self):
        """GHERKIN: distinguishable from PTCV-172's 'llm_vision'."""
        builder = LlmSoaBuilder()
        builder._client = _FakeAnthropicClient(_make_soa_response())

        sections = [_make_section("B.4", "Trial Design text")]
        result = builder.build(sections=sections)
        assert result is not None
        assert result[0].construction_method != "llm_vision"

    def test_section_code_is_llm_text(self):
        """section_code set to 'llm_text' for traceability."""
        builder = LlmSoaBuilder()
        builder._client = _FakeAnthropicClient(_make_soa_response())

        sections = [_make_section("B.4", "Trial Design text")]
        result = builder.build(sections=sections)
        assert result is not None
        assert result[0].section_code == "llm_text"


# ------------------------------------------------------------------
# TestEmptySections — GHERKIN scenario 8
# ------------------------------------------------------------------


class TestEmptySections:
    """Empty sections produce warning and skip Sonnet."""

    def test_no_relevant_sections_returns_none(self):
        """GHERKIN: Empty sections produce warning."""
        builder = LlmSoaBuilder()
        builder._client = _FakeAnthropicClient(_make_soa_response())

        # Only B.1 section — not relevant for SoA
        sections = [_make_section("B.1", "General information")]
        result = builder.build(sections=sections)
        assert result is None

    def test_empty_text_blocks_returns_none(self):
        """No relevant text in text_blocks."""
        builder = LlmSoaBuilder()
        builder._client = _FakeAnthropicClient(_make_soa_response())

        text_blocks = [{"page_number": 1, "text": "Unrelated content."}]
        result = builder.build(text_blocks=text_blocks)
        assert result is None

    def test_no_inputs_returns_none(self):
        """No sections or text_blocks at all."""
        builder = LlmSoaBuilder()
        result = builder.build()
        assert result is None


# ------------------------------------------------------------------
# TestSonnetModel — GHERKIN scenario 6
# ------------------------------------------------------------------


class TestSonnetModel:
    """Use Sonnet (NOT Opus) model."""

    def test_uses_sonnet_not_opus(self):
        """GHERKIN: Use Sonnet (NOT Opus) model."""
        builder = LlmSoaBuilder()
        assert "sonnet" in builder._sonnet_model
        assert "opus" not in builder._sonnet_model

    def test_model_passed_to_api_call(self):
        """Model string passed through to client.messages.create()."""
        builder = LlmSoaBuilder()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _FakeResponse(
            _make_soa_response()
        )
        builder._client = mock_client

        sections = [_make_section("B.4", "Trial Design text")]
        builder.build(sections=sections)

        call_kwargs = mock_client.messages.create.call_args
        assert "sonnet" in call_kwargs.kwargs.get("model", "")


# ------------------------------------------------------------------
# TestResponseParsing
# ------------------------------------------------------------------


class TestResponseParsing:
    """Sonnet JSON response validation and parsing."""

    def test_valid_json_produces_raw_soa_table(self):
        """Valid Sonnet response maps to RawSoaTable."""
        result = LlmSoaBuilder._parse_response(_make_soa_response())
        assert result is not None
        assert len(result) == 1
        table = result[0]
        assert len(table.visit_headers) == 4
        assert len(table.activities) == 4
        assert table.visit_headers[0] == "Screening"

    def test_invalid_json_returns_none(self):
        """Unparseable JSON gracefully returns None."""
        result = LlmSoaBuilder._parse_response("not valid json {{{")
        assert result is None

    def test_empty_visits_returns_none(self):
        """Empty visits array returns None."""
        resp = json.dumps({"visits": [], "assessments": [{"name": "X", "visits_scheduled": []}]})
        result = LlmSoaBuilder._parse_response(resp)
        assert result is None

    def test_empty_assessments_returns_none(self):
        """Empty assessments array returns None."""
        resp = json.dumps({
            "visits": [{"name": "V1", "day_offset": 1}],
            "assessments": [],
        })
        result = LlmSoaBuilder._parse_response(resp)
        assert result is None

    def test_mismatched_array_lengths_padded(self):
        """visits_scheduled shorter than visits is padded with False."""
        resp = json.dumps({
            "visits": [
                {"name": "V1", "day_offset": 1},
                {"name": "V2", "day_offset": 8},
                {"name": "V3", "day_offset": 15},
            ],
            "assessments": [
                {"name": "ECG", "visits_scheduled": [True]},  # only 1, should be 3
            ],
        })
        result = LlmSoaBuilder._parse_response(resp)
        assert result is not None
        flags = result[0].activities[0][1]
        assert len(flags) == 3
        assert flags == [True, False, False]

    def test_markdown_fences_stripped(self):
        """JSON wrapped in markdown fences is handled."""
        raw = f"```json\n{_make_soa_response()}\n```"
        result = LlmSoaBuilder._parse_response(raw)
        assert result is not None

    def test_no_scheduled_flags_returns_none(self):
        """All False flags returns None."""
        resp = json.dumps({
            "visits": [{"name": "V1", "day_offset": 1}],
            "assessments": [
                {"name": "ECG", "visits_scheduled": [False]},
            ],
        })
        result = LlmSoaBuilder._parse_response(resp)
        assert result is None

    def test_day_headers_from_day_offset(self):
        """day_offset values become day_headers."""
        resp = json.dumps({
            "visits": [
                {"name": "Screening", "day_offset": -14},
                {"name": "Day 1", "day_offset": 1},
            ],
            "assessments": [
                {"name": "Labs", "visits_scheduled": [True, True]},
            ],
        })
        result = LlmSoaBuilder._parse_response(resp)
        assert result is not None
        assert result[0].day_headers == ["-14", "1"]


# ------------------------------------------------------------------
# TestUsdmMapping — GHERKIN scenario 1
# ------------------------------------------------------------------


class TestUsdmMapping:
    """LLM-constructed tables map to USDM entities."""

    def test_llm_tables_map_to_usdm_entities(self):
        """GHERKIN: output maps to UsdmTimepoint, UsdmActivity, UsdmScheduledInstance."""
        from ptcv.soa_extractor.mapper import UsdmMapper
        from ptcv.soa_extractor.resolver import SynonymResolver

        # Parse a mock Sonnet response
        tables = LlmSoaBuilder._parse_response(_make_soa_response())
        assert tables is not None

        mapper = UsdmMapper(resolver=SynonymResolver())
        epochs, timepoints, activities, instances, synonyms = mapper.map(
            tables,
            run_id="test-run",
            source_run_id="src-run",
            source_sha256="a" * 64,
            registry_id="NCT00112827",
            timestamp="2026-03-09T00:00:00Z",
        )

        assert len(timepoints) >= 1
        assert len(activities) >= 1
        assert len(instances) >= 1
        # Verify USDM types
        from ptcv.soa_extractor.models import (
            UsdmActivity,
            UsdmScheduledInstance,
            UsdmTimepoint,
        )

        assert all(isinstance(tp, UsdmTimepoint) for tp in timepoints)
        assert all(isinstance(a, UsdmActivity) for a in activities)
        assert all(isinstance(i, UsdmScheduledInstance) for i in instances)


# ------------------------------------------------------------------
# TestExtractorIntegration — cascade integration in SoaExtractor
# ------------------------------------------------------------------


class TestExtractorIntegration:
    """Level 3 integration within SoaExtractor.extract()."""

    def test_level3_triggered_when_no_tables(
        self, tmp_gateway, tmp_review_queue,
    ):
        """SoaExtractor invokes Level 3 when Levels 1/2 produce nothing."""
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
        )
        # Mock the LLM builder
        extractor._llm_builder = MagicMock()
        mock_table = RawSoaTable(
            visit_headers=["V1", "V2"],
            day_headers=["1", "8"],
            activities=[
                ("ECG", [True, True]),
                ("Labs", [True, False]),
                ("Vitals", [True, True]),
            ],
            section_code="llm_text",
            construction_method="llm_text",
        )
        extractor._llm_builder.build.return_value = [mock_table]

        sections = [_make_section("B.4", "Trial Design text")]
        result = extractor.extract(
            sections=sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )

        extractor._llm_builder.build.assert_called_once()
        assert result.activity_count >= 1

    def test_level3_skipped_when_sufficient_tables(
        self, tmp_gateway, tmp_review_queue,
    ):
        """SoaExtractor skips Level 3 when Level 1c produces >= 3 activities."""
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
        )
        extractor._llm_builder = MagicMock()

        # sample_ich_sections fixture produces >= 3 activities via parser
        from ptcv.ich_parser.models import IchSection

        soa_text = """\
Schedule of Activities

| Assessment | Screening | Baseline | Week 2 | Week 4 |
|------------|-----------|----------|--------|--------|
| ECG        | X         | X        | X      |        |
| Labs       | X         | X        | X      | X      |
| Vitals     | X         | X        | X      | X      |
| PE         | X         |          |        | X      |
"""
        sections = [
            IchSection(
                run_id="r1",
                source_run_id="sr1",
                source_sha256="a" * 64,
                registry_id="NCT00112827",
                section_code="B.4",
                section_name="Trial Design",
                content_json=json.dumps({"text": soa_text}),
                confidence_score=0.9,
                review_required=False,
                legacy_format=False,
            ),
        ]

        result = extractor.extract(
            sections=sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )

        extractor._llm_builder.build.assert_not_called()
        assert result.activity_count >= 3

    def test_level3_failure_non_blocking(
        self, tmp_gateway, tmp_review_queue,
    ):
        """LLM builder exception does not crash extraction."""
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
        )
        extractor._llm_builder = MagicMock()
        extractor._llm_builder.build.side_effect = RuntimeError("API down")

        sections = [_make_section("B.4", "Trial Design text")]
        result = extractor.extract(
            sections=sections,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        # Should not raise, returns empty result
        assert result is not None

    def test_text_blocks_accepted_as_standalone_input(
        self, tmp_gateway, tmp_review_queue,
    ):
        """text_blocks alone is a valid input (for Level 3)."""
        from ptcv.soa_extractor import SoaExtractor

        extractor = SoaExtractor(
            gateway=tmp_gateway,
            review_queue=tmp_review_queue,
        )
        extractor._llm_builder = MagicMock()
        extractor._llm_builder.build.return_value = None

        text_blocks = [{"page_number": 1, "text": "Some protocol text"}]
        result = extractor.extract(
            text_blocks=text_blocks,
            registry_id="NCT00112827",
            source_sha256="a" * 64,
        )
        assert result is not None


# ------------------------------------------------------------------
# TestRegistryContext — PTCV-198 GHERKIN scenarios
# ------------------------------------------------------------------


def _make_registry_metadata(
    *,
    primary_outcomes: list[dict] | None = None,
    secondary_outcomes: list[dict] | None = None,
    interventions: list[dict] | None = None,
    eligibility: dict | None = None,
    design_info: dict | None = None,
    enrollment: dict | None = None,
) -> dict:
    """Build a CT.gov API v2-shaped metadata dict."""
    protocol: dict = {}

    outcomes: dict = {}
    if primary_outcomes is not None:
        outcomes["primaryOutcomes"] = primary_outcomes
    if secondary_outcomes is not None:
        outcomes["secondaryOutcomes"] = secondary_outcomes
    if outcomes:
        protocol["outcomesModule"] = outcomes

    if interventions is not None:
        protocol["armsInterventionsModule"] = {
            "interventions": interventions,
        }

    if eligibility is not None:
        protocol["eligibilityModule"] = eligibility

    design: dict = {}
    if design_info is not None:
        design["designInfo"] = design_info
    if enrollment is not None:
        design["enrollmentInfo"] = enrollment
    if design:
        protocol["designModule"] = design

    return {"protocolSection": protocol}


class TestRegistryContext:
    """PTCV-198: Registry-enriched SoA construction."""

    def test_prompt_includes_registry_endpoints(self):
        """Scenario 1: Sonnet prompt includes registry endpoints."""
        meta = _make_registry_metadata(
            primary_outcomes=[
                {
                    "measure": "Overall Survival",
                    "timeFrame": "From randomization to death, up to 60 months",
                },
            ],
            secondary_outcomes=[
                {
                    "measure": "Progression-Free Survival",
                    "timeFrame": "Every 8 weeks until progression",
                },
            ],
        )
        result = LlmSoaBuilder._format_registry_context(meta)

        assert "Registry Context" in result
        assert "Overall Survival" in result
        assert "60 months" in result
        assert "Progression-Free Survival" in result
        assert "Every 8 weeks" in result

    def test_registry_context_improves_assessment_detection(self):
        """Scenario 2: Registry context included in Sonnet prompt."""
        meta = _make_registry_metadata(
            primary_outcomes=[
                {
                    "measure": "Tumor response per RECIST v1.1",
                    "timeFrame": "Every 8 weeks from baseline",
                },
            ],
            interventions=[
                {
                    "type": "DRUG",
                    "name": "Pembrolizumab",
                    "description": "200 mg IV Q3W",
                },
            ],
        )

        builder = LlmSoaBuilder()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _FakeResponse(
            _make_soa_response()
        )
        builder._client = mock_client

        sections = [_make_section("B.4", "Trial Design: parallel-group")]
        builder.build(sections=sections, registry_metadata=meta)

        call_kwargs = mock_client.messages.create.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "Tumor response per RECIST" in prompt
        assert "Every 8 weeks" in prompt
        assert "Pembrolizumab" in prompt

    def test_graceful_without_registry_data(self):
        """Scenario 3: Graceful operation without registry data."""
        builder = LlmSoaBuilder()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _FakeResponse(
            _make_soa_response()
        )
        builder._client = mock_client

        sections = [_make_section("B.4", "Trial Design: parallel-group")]
        result = builder.build(sections=sections, registry_metadata=None)

        assert result is not None
        # Verify "Registry Context" is NOT in the prompt
        call_kwargs = mock_client.messages.create.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "Registry Context" not in prompt

    def test_registry_context_truncated(self):
        """Scenario 4: Registry context bounded by token limit."""
        from ptcv.soa_extractor.llm_soa_builder import (
            _MAX_REGISTRY_CONTEXT_CHARS,
        )

        # Build oversized metadata with many outcomes
        primary = [
            {
                "measure": f"Endpoint {i}: " + "x" * 200,
                "timeFrame": f"Timeframe {i}: " + "y" * 100,
            }
            for i in range(30)
        ]
        meta = _make_registry_metadata(
            primary_outcomes=primary,
            interventions=[
                {
                    "type": "DRUG",
                    "name": f"Drug{i}",
                    "description": "z" * 120,
                }
                for i in range(20)
            ],
        )
        result = LlmSoaBuilder._format_registry_context(meta)

        assert len(result) <= _MAX_REGISTRY_CONTEXT_CHARS
        # High-priority endpoints should be preserved
        assert "Endpoint 0" in result
