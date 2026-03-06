"""Unit tests for NHI param builder (PTCV-55).

Tests natural history data search parameterization from SoA-derived
endpoints and visit windows, covering all 4 GHERKIN acceptance criteria.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.priority_sections import PrioritySectionResult
from ptcv.soa_extractor.models import RawSoaTable
from ptcv.sdtm.soa_mapper import MockSdtmDataset, SoaToSdtmMapper
from ptcv.search.nhi_param_builder import (
    AlignmentScore,
    NaturalHistoryParamBuilder,
    NHSearchParams,
    NHSearchQuery,
    PopulationCriteria,
    _parse_population,
    _parse_endpoints,
)


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------

def _make_mock_dataset(studyid: str = "NCT_TEST") -> MockSdtmDataset:
    soa_table = RawSoaTable(
        visit_headers=[
            "Screening", "Baseline", "Week 4", "Week 8",
            "Week 12", "Week 24", "Follow-up",
        ],
        day_headers=[
            "Day -14 to -1", "Day 1", "Day 29 ± 3", "Day 57 ± 3",
            "Day 85 ± 3", "Day 169 ± 7", "Day 197 ± 7",
        ],
        activities=[
            ("Physical Exam", [True, True, False, True, False, True, True]),
            ("Complete Blood Count", [True, True, True, True, True, True, True]),
            ("Vital Signs", [True, True, True, True, True, True, True]),
            ("12-lead ECG", [True, False, True, False, True, False, True]),
            ("Adverse Events", [False, True, True, True, True, True, True]),
            ("Tumor Assessment", [False, True, False, True, False, True, False]),
            ("ECOG Performance", [True, True, False, False, False, True, True]),
        ],
        section_code="B.4",
    )

    b4_section = IchSection(
        run_id="run-001",
        source_run_id="",
        source_sha256="a" * 64,
        registry_id=studyid,
        section_code="B.4",
        section_name="Trial Design",
        content_json=json.dumps({
            "text_excerpt": (
                "This is a randomized, double-blind, placebo-controlled study.\n"
                "Arm A: Drug X 200mg Q2W\n"
                "Arm B: Placebo Q2W\n"
                "Screening followed by treatment period then follow-up."
            ),
        }),
        confidence_score=0.88,
        review_required=False,
        legacy_format=False,
        extraction_timestamp_utc="2024-01-01T00:00:00+00:00",
    )

    mapper = SoaToSdtmMapper()
    return mapper.map(soa_table, sections=[b4_section], studyid=studyid)


def _make_b5_section(text: str) -> PrioritySectionResult:
    return PrioritySectionResult(
        section_code="B.5",
        section_name="Selection of Subjects",
        full_text=text,
        tables=[],
        soa_candidate_tables=[],
        cross_references=[],
        page_range=(11, 15),
    )


def _make_b10_section(text: str) -> PrioritySectionResult:
    return PrioritySectionResult(
        section_code="B.10",
        section_name="Assessment of Efficacy",
        full_text=text,
        tables=[],
        soa_candidate_tables=[],
        cross_references=[],
        page_range=(19, 22),
    )


def _make_b14_section(text: str) -> PrioritySectionResult:
    return PrioritySectionResult(
        section_code="B.14",
        section_name="Safety Monitoring",
        full_text=text,
        tables=[],
        soa_candidate_tables=[],
        cross_references=[],
        page_range=(30, 33),
    )


@pytest.fixture
def mock_dataset() -> MockSdtmDataset:
    return _make_mock_dataset()


@pytest.fixture
def b5_diabetes() -> PrioritySectionResult:
    return _make_b5_section(
        "Inclusion criteria:\n"
        "1. Adults aged 18-65 with type 2 diabetes\n"
        "2. HbA1c 7.0-10.0%\n"
        "3. BMI 25-40 kg/m2\n"
        "Exclusion criteria:\n"
        "1. Pregnant or lactating women\n"
        "2. Renal impairment (eGFR < 30 mL/min)\n"
        "3. Prior treatment with insulin"
    )


@pytest.fixture
def b10_efficacy() -> PrioritySectionResult:
    return _make_b10_section(
        "Primary endpoint: HbA1c change from baseline at Week 24\n"
        "Secondary endpoints:\n"
        "- Objective response rate per RECIST 1.1\n"
        "- Progression-free survival\n"
        "- Quality of life (EQ-5D)"
    )


@pytest.fixture
def b14_safety() -> PrioritySectionResult:
    return _make_b14_section(
        "Safety monitoring:\n"
        "- Adverse events assessed at every visit\n"
        "- Hepatic function panel every 4 weeks\n"
        "- MACE events adjudicated by independent committee"
    )


@pytest.fixture
def builder() -> NaturalHistoryParamBuilder:
    return NaturalHistoryParamBuilder()


# -----------------------------------------------------------------------
# Scenario 1: Search parameters extracted from SoA data
# -----------------------------------------------------------------------

class TestSearchParamsExtraction:
    """Scenario: Search parameters extracted from SoA data."""

    def test_returns_nh_search_params(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
        b10_efficacy: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(
            mock_dataset, [b5_diabetes, b10_efficacy],
        )
        assert isinstance(params, NHSearchParams)

    def test_condition_extracted(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        assert params.population.condition != ""
        assert "diabetes" in params.population.condition.lower()

    def test_endpoints_extracted(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b10_efficacy: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b10_efficacy])
        assert len(params.endpoints) > 0
        endpoint_names = [ep.name.lower() for ep in params.endpoints]
        # Should find HbA1c, RECIST, PFS, or QoL endpoints
        assert any(
            "hba1c" in n or "recist" in n or "progression" in n or "quality" in n
            for n in endpoint_names
        )

    def test_follow_up_duration_captured(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset)
        assert params.follow_up_weeks > 0

    def test_assessment_frequency_captured(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset)
        assert params.assessment_frequency_weeks > 0

    def test_required_domains_listed(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset)
        assert len(params.required_domains) > 0
        # Should include LB, VS, EG from the SoA assessments
        assert "LB" in params.required_domains
        assert "VS" in params.required_domains

    def test_visit_count_set(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset)
        assert params.visit_count == len(mock_dataset.tv)

    def test_condition_terms_consolidated(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
        b10_efficacy: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(
            mock_dataset, [b5_diabetes, b10_efficacy],
        )
        assert len(params.condition_terms) > 0
        # No duplicates
        assert len(params.condition_terms) == len(set(
            t.lower() for t in params.condition_terms
        ))

    def test_study_duration_days(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset)
        assert params.study_duration_days > 0


# -----------------------------------------------------------------------
# Scenario 2: Search queries generated for data platforms
# -----------------------------------------------------------------------

class TestSearchQueryGeneration:
    """Scenario: Search queries generated for data platforms."""

    def test_queries_generated(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        queries = builder.generate_queries(params)
        assert len(queries) >= 1

    def test_ctgov_query_present(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        queries = builder.generate_queries(params)
        platforms = [q.platform for q in queries]
        assert "ClinicalTrials.gov" in platforms

    def test_ctgov_query_is_observational(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        queries = builder.generate_queries(params)
        ctgov = next(q for q in queries if q.platform == "ClinicalTrials.gov")
        assert ctgov.structured_params.get("filter.studyType") == "OBSERVATIONAL"

    def test_query_includes_condition(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        queries = builder.generate_queries(params)
        ctgov = next(q for q in queries if q.platform == "ClinicalTrials.gov")
        assert "diabetes" in ctgov.structured_params.get("query.cond", "").lower()

    def test_human_readable_format(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        queries = builder.generate_queries(params)
        for q in queries:
            assert isinstance(q.human_readable, str)
            assert len(q.human_readable) > 10

    def test_structured_format(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        queries = builder.generate_queries(params)
        for q in queries:
            assert isinstance(q.structured_params, dict)
            assert len(q.structured_params) > 0

    def test_query_includes_duration(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        queries = builder.generate_queries(params)
        ctgov = next(q for q in queries if q.platform == "ClinicalTrials.gov")
        assert "follow-up" in ctgov.human_readable.lower()


# -----------------------------------------------------------------------
# Scenario 3: Candidate dataset alignment scored
# -----------------------------------------------------------------------

class TestAlignmentScoring:
    """Scenario: Candidate dataset alignment scored."""

    def test_alignment_score_produced(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
        b10_efficacy: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(
            mock_dataset, [b5_diabetes, b10_efficacy],
        )
        candidate = {
            "condition": "type 2 diabetes",
            "age_min": 20,
            "age_max": 70,
            "endpoints": ["HbA1c change", "body weight"],
            "domains": ["LB", "VS", "DM"],
            "follow_up_weeks": 52,
        }
        score = builder.score_alignment(params, candidate)
        assert isinstance(score, AlignmentScore)

    def test_population_overlap_scored(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        candidate = {
            "condition": "type 2 diabetes",
            "age_min": 18,
            "age_max": 65,
        }
        score = builder.score_alignment(params, candidate)
        assert score.population_score > 0.5

    def test_endpoint_coverage_scored(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b10_efficacy: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b10_efficacy])
        # Good match
        good_candidate = {
            "endpoints": ["HbA1c change", "progression-free survival", "QoL"],
        }
        good_score = builder.score_alignment(params, good_candidate)
        # Poor match
        poor_candidate = {
            "endpoints": ["bone mineral density"],
        }
        poor_score = builder.score_alignment(params, poor_candidate)
        assert good_score.endpoint_score > poor_score.endpoint_score

    def test_temporal_alignment_scored(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset)
        # Long follow-up (good)
        long_fu = {"follow_up_weeks": 104}
        long_score = builder.score_alignment(params, long_fu)
        # Short follow-up (poor)
        short_fu = {"follow_up_weeks": 4}
        short_score = builder.score_alignment(params, short_fu)
        assert long_score.temporal_score > short_score.temporal_score

    def test_overall_percentage_produced(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        candidate = {
            "condition": "type 2 diabetes",
            "age_min": 18,
            "age_max": 65,
            "endpoints": ["HbA1c"],
            "domains": ["LB", "VS"],
            "follow_up_weeks": 52,
        }
        score = builder.score_alignment(params, candidate)
        assert 0.0 <= score.overall_percentage <= 100.0

    def test_domain_coverage_scored(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset)
        # Full domain coverage
        full = {"domains": params.required_domains}
        full_score = builder.score_alignment(params, full)
        # No domain coverage
        none = {"domains": []}
        none_score = builder.score_alignment(params, none)
        assert full_score.domain_score > none_score.domain_score

    def test_perfect_match_high_score(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
        b10_efficacy: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(
            mock_dataset, [b5_diabetes, b10_efficacy],
        )
        candidate = {
            "condition": "type 2 diabetes",
            "age_min": 18,
            "age_max": 65,
            "endpoints": [
                "HbA1c change", "progression-free survival",
                "RECIST response", "quality of life",
            ],
            "domains": params.required_domains,
            "follow_up_weeks": 52,
        }
        score = builder.score_alignment(params, candidate)
        assert score.overall_percentage >= 50.0

    def test_score_details_populated(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        candidate = {"condition": "diabetes", "follow_up_weeks": 24}
        score = builder.score_alignment(params, candidate)
        assert "population" in score.details
        assert "endpoint" in score.details
        assert "temporal" in score.details
        assert "domain" in score.details


# -----------------------------------------------------------------------
# Scenario 4: Parameters reflect B5 eligibility criteria
# -----------------------------------------------------------------------

class TestB5EligibilityParsing:
    """Scenario: Parameters reflect B5 eligibility criteria."""

    def test_age_range_extracted(self) -> None:
        text = "Adults aged 18-65 with type 2 diabetes"
        pop = _parse_population(text)
        assert pop.age_min == 18
        assert pop.age_max == 65

    def test_condition_extracted(self) -> None:
        text = "Patients with type 2 diabetes mellitus"
        pop = _parse_population(text)
        assert "diabetes" in pop.condition.lower()

    def test_hba1c_biomarker_extracted(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        biomarkers = params.population.biomarker_criteria
        # Should find HbA1c criteria
        assert any("HbA1c" in bm for bm in biomarkers)

    def test_exclusion_keywords_extracted(
        self,
        builder: NaturalHistoryParamBuilder,
        mock_dataset: MockSdtmDataset,
        b5_diabetes: PrioritySectionResult,
    ) -> None:
        params = builder.build_params(mock_dataset, [b5_diabetes])
        excl = params.population.exclusion_keywords
        assert any("pregnant" in kw.lower() for kw in excl)
        assert any("renal" in kw.lower() for kw in excl)

    def test_age_min_only(self) -> None:
        text = "Age >= 18 years with hypertension"
        pop = _parse_population(text)
        assert pop.age_min == 18
        assert pop.age_max is None

    def test_multiple_conditions(self) -> None:
        text = "Patients with NSCLC and prior breast cancer"
        pop = _parse_population(text)
        assert len(pop.condition_keywords) >= 2

    def test_empty_text_returns_defaults(self) -> None:
        pop = _parse_population("")
        assert pop.age_min is None
        assert pop.age_max is None
        assert pop.condition == ""

    def test_bmi_biomarker(self) -> None:
        text = "BMI 25-40 kg/m2"
        pop = _parse_population(text)
        assert any("BMI" in bm for bm in pop.biomarker_criteria)


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_no_sections_still_works(
        self, builder: NaturalHistoryParamBuilder, mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset, None)
        assert isinstance(params, NHSearchParams)
        assert params.visit_count > 0

    def test_empty_candidate_scores(
        self, builder: NaturalHistoryParamBuilder, mock_dataset: MockSdtmDataset,
    ) -> None:
        params = builder.build_params(mock_dataset)
        score = builder.score_alignment(params, {})
        assert 0.0 <= score.overall_score <= 1.0

    def test_endpoint_parsing_primary(self) -> None:
        text = "Primary: Overall survival. Secondary: PFS"
        endpoints = _parse_endpoints(text, "primary")
        assert len(endpoints) >= 1
        names = [ep.name.lower() for ep in endpoints]
        assert any("survival" in n for n in names)

    def test_alignment_score_percentage_property(self) -> None:
        score = AlignmentScore(
            population_score=0.8,
            endpoint_score=0.6,
            temporal_score=0.9,
            domain_score=0.7,
            overall_score=0.75,
        )
        assert score.overall_percentage == 75.0
