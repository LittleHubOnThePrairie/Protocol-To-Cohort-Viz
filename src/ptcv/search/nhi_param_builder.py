"""Natural history data search parameterization from SoA (PTCV-55).

Extracts structured search parameters from the SoA extraction pipeline
output to find matching natural history datasets for external control
arm construction, sample size estimation, and endpoint validation.

Input sources:
  MockSdtmDataset — visit schedule, assessment domains, arm structure
  PrioritySectionResult (B.5) — eligibility criteria (population)
  PrioritySectionResult (B.10) — efficacy endpoints
  PrioritySectionResult (B.14) — safety endpoints

Output:
  NHSearchParams — structured search criteria
  NHSearchQuery — platform-specific search queries
  AlignmentScore — candidate dataset alignment scoring

Risk tier: LOW — search parameterization only; no data access.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..ich_parser.priority_sections import PrioritySectionResult
    from ..sdtm.soa_mapper import MockSdtmDataset


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PopulationCriteria:
    """Population parameters extracted from B.5 eligibility criteria.

    Attributes:
        age_min: Minimum age in years (None if not specified).
        age_max: Maximum age in years (None if not specified).
        condition: Primary condition/disease being studied.
        condition_keywords: Additional condition-related keywords.
        biomarker_criteria: Biomarker inclusion criteria (e.g. HbA1c range).
        exclusion_keywords: Key exclusion criteria terms.
    """

    age_min: int | None = None
    age_max: int | None = None
    condition: str = ""
    condition_keywords: list[str] = dataclasses.field(default_factory=list)
    biomarker_criteria: list[str] = dataclasses.field(default_factory=list)
    exclusion_keywords: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class EndpointInfo:
    """Endpoint information extracted from B.10/B.14 sections.

    Attributes:
        name: Endpoint name/description.
        endpoint_type: "primary", "secondary", or "safety".
        assessment_method: How the endpoint is measured.
        keywords: Search-relevant keywords.
    """

    name: str
    endpoint_type: str
    assessment_method: str = ""
    keywords: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class NHSearchParams:
    """Structured search parameters for natural history data.

    Attributes:
        population: Population/eligibility criteria.
        endpoints: List of study endpoints.
        follow_up_weeks: Total study follow-up duration in weeks.
        assessment_frequency_weeks: Typical assessment frequency in weeks.
        required_domains: SDTM domains required (e.g. LB, VS, EG).
        visit_count: Number of planned visits.
        condition_terms: Consolidated condition search terms.
        study_duration_days: Total study duration in days.
    """

    population: PopulationCriteria
    endpoints: list[EndpointInfo]
    follow_up_weeks: int
    assessment_frequency_weeks: int
    required_domains: list[str]
    visit_count: int
    condition_terms: list[str]
    study_duration_days: int


@dataclasses.dataclass
class NHSearchQuery:
    """Search query for a specific natural history data platform.

    Attributes:
        platform: Target platform name.
        human_readable: Human-readable search description.
        structured_params: API-ready parameter dictionary.
    """

    platform: str
    human_readable: str
    structured_params: dict[str, Any]


@dataclasses.dataclass
class AlignmentScore:
    """Alignment score between a trial protocol and a candidate dataset.

    Attributes:
        population_score: Population overlap score (0.0-1.0).
        endpoint_score: Endpoint coverage score (0.0-1.0).
        temporal_score: Temporal alignment score (0.0-1.0).
        domain_score: Data domain coverage score (0.0-1.0).
        overall_score: Weighted overall alignment (0.0-1.0).
        details: Component score explanations.
    """

    population_score: float
    endpoint_score: float
    temporal_score: float
    domain_score: float
    overall_score: float
    details: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def overall_percentage(self) -> float:
        """Overall alignment as a percentage."""
        return round(self.overall_score * 100, 1)


# ---------------------------------------------------------------------------
# Eligibility text parsing
# ---------------------------------------------------------------------------

_AGE_RANGE_RE = re.compile(
    r"(?:aged?|age)\s*(?:range\s*)?(\d{1,3})\s*[-–to]+\s*(\d{1,3})",
    re.IGNORECASE,
)

_AGE_MIN_RE = re.compile(
    r"(?:age|aged)\s*(?:>=?|≥|at\s+least|minimum)\s*(\d{1,3})",
    re.IGNORECASE,
)

_AGE_MAX_RE = re.compile(
    r"(?:age|aged)\s*(?:<=?|≤|up\s+to|maximum|no\s+(?:more|older)\s+than)\s*(\d{1,3})",
    re.IGNORECASE,
)

_CONDITION_PATTERNS: list[tuple[str, str]] = [
    (r"type\s*[12]\s*diabet\w*", "diabetes"),
    (r"non.small\s*cell\s*lung\s*cancer|nsclc", "NSCLC"),
    (r"breast\s*cancer", "breast cancer"),
    (r"hypertension", "hypertension"),
    (r"heart\s*failure", "heart failure"),
    (r"chronic\s*kidney\s*disease|ckd", "chronic kidney disease"),
    (r"rheumatoid\s*arthritis", "rheumatoid arthritis"),
    (r"multiple\s*sclerosis", "multiple sclerosis"),
    (r"alzheimer", "Alzheimer's disease"),
    (r"parkinson", "Parkinson's disease"),
    (r"asthma", "asthma"),
    (r"copd|chronic\s*obstructive", "COPD"),
    (r"hepatitis", "hepatitis"),
    (r"melanoma", "melanoma"),
    (r"lymphoma", "lymphoma"),
    (r"leukemia|leukaemia", "leukemia"),
    (r"prostate\s*cancer", "prostate cancer"),
    (r"colorectal\s*cancer", "colorectal cancer"),
    (r"pancreatic\s*cancer", "pancreatic cancer"),
    (r"ovarian\s*cancer", "ovarian cancer"),
    (r"depression|major\s*depressive", "major depressive disorder"),
    (r"schizophrenia", "schizophrenia"),
    (r"epilepsy", "epilepsy"),
    (r"atopic\s*dermatitis|eczema", "atopic dermatitis"),
    (r"psoriasis", "psoriasis"),
    (r"crohn", "Crohn's disease"),
    (r"ulcerative\s*colitis", "ulcerative colitis"),
]

_BIOMARKER_RE = re.compile(
    r"(HbA1c|eGFR|BMI|creatinine|albumin|bilirubin|INR|CRP|ESR|PSA|CEA"
    r"|troponin|BNP|NT-proBNP|FEV1|DLCO|LVEF)"
    r"\s*(?:>=?|<=?|≥|≤|between|of|range)?\s*"
    r"([\d.]+(?:\s*[-–to]+\s*[\d.]+)?(?:\s*%|\s*\w+/\w+)?)",
    re.IGNORECASE,
)

_ENDPOINT_PATTERNS: list[tuple[str, list[str]]] = [
    (r"overall\s*survival|OS\b", ["overall survival", "mortality"]),
    (r"progression.free\s*survival|PFS\b", ["progression-free survival"]),
    (r"objective\s*response\s*rate|ORR\b", ["objective response rate", "tumor response"]),
    (r"complete\s*response|CR\b", ["complete response"]),
    (r"disease.free\s*survival|DFS\b", ["disease-free survival"]),
    (r"event.free\s*survival|EFS\b", ["event-free survival"]),
    (r"time\s*to\s*progression|TTP\b", ["time to progression"]),
    (r"duration\s*of\s*response|DOR\b", ["duration of response"]),
    (r"RECIST", ["RECIST", "tumor assessment"]),
    (r"MACE|major\s*adverse\s*cardiovascular", ["MACE", "cardiovascular events"]),
    (r"HbA1c\s*(?:change|reduction|lowering)", ["HbA1c change", "glycemic control"]),
    (r"blood\s*pressure\s*(?:change|reduction)", ["blood pressure", "hypertension"]),
    (r"FEV1\s*(?:change|improvement)", ["FEV1", "pulmonary function"]),
    (r"ACR\s*20|ACR\s*50|ACR\s*70", ["ACR response", "rheumatoid arthritis"]),
    (r"EDSS\s*(?:change|progression)", ["EDSS", "disability progression"]),
    (r"quality\s*of\s*life|QoL|EQ-5D|SF-36", ["quality of life", "PRO"]),
]


def _parse_population(text: str) -> PopulationCriteria:
    """Extract population criteria from B.5 eligibility text."""
    criteria = PopulationCriteria()

    if not text:
        return criteria

    # Age range
    m = _AGE_RANGE_RE.search(text)
    if m:
        criteria.age_min = int(m.group(1))
        criteria.age_max = int(m.group(2))
    else:
        m_min = _AGE_MIN_RE.search(text)
        m_max = _AGE_MAX_RE.search(text)
        if m_min:
            criteria.age_min = int(m_min.group(1))
        if m_max:
            criteria.age_max = int(m_max.group(1))

    # Condition detection
    conditions: list[str] = []
    keywords: list[str] = []
    for pattern, condition_name in _CONDITION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            conditions.append(condition_name)
            keywords.append(condition_name)
    if conditions:
        criteria.condition = conditions[0]
        criteria.condition_keywords = keywords

    # Biomarker criteria
    for m in _BIOMARKER_RE.finditer(text):
        marker = m.group(1)
        value = m.group(2).strip()
        criteria.biomarker_criteria.append(f"{marker} {value}")

    # Exclusion keywords
    excl_section = ""
    excl_match = re.search(
        r"exclu?sion\s*(?:criteria)?[:\s]*(.*?)(?:$|\n\n)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if excl_match:
        excl_section = excl_match.group(1)
    for keyword in ["pregnant", "renal", "hepatic", "cardiac", "malignancy",
                     "prior treatment", "surgery", "allergy", "HIV", "HBV",
                     "HCV", "transplant"]:
        if keyword.lower() in excl_section.lower():
            criteria.exclusion_keywords.append(keyword)

    return criteria


def _parse_endpoints(
    text: str, endpoint_type: str = "primary"
) -> list[EndpointInfo]:
    """Extract endpoint information from B.10 or B.14 text."""
    endpoints: list[EndpointInfo] = []

    if not text:
        return endpoints

    for pattern, keywords in _ENDPOINT_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            endpoints.append(EndpointInfo(
                name=m.group(0).strip(),
                endpoint_type=endpoint_type,
                keywords=keywords,
            ))

    return endpoints


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class NaturalHistoryParamBuilder:
    """Builds natural history search parameters from SoA extraction data.

    Extracts structured search criteria from MockSdtmDataset and
    PrioritySectionResult objects to parameterize natural history
    dataset searches across multiple data platforms.
    """

    def build_params(
        self,
        dataset: "MockSdtmDataset",
        priority_sections: list["PrioritySectionResult"] | None = None,
    ) -> NHSearchParams:
        """Build NHSearchParams from a MockSdtmDataset and optional sections.

        Args:
            dataset: MockSdtmDataset with TV, TA, domain_mapping.
            priority_sections: Optional list of PrioritySectionResult
                from PTCV-52 extraction.

        Returns:
            NHSearchParams with structured search criteria.
        """
        sections = priority_sections or []
        section_map = {s.section_code: s for s in sections}

        # Population from B.5
        b5_text = section_map["B.5"].full_text if "B.5" in section_map else ""
        population = _parse_population(b5_text)

        # Endpoints from B.10 (efficacy) and B.14 (safety)
        b10_text = section_map["B.10"].full_text if "B.10" in section_map else ""
        b14_text = section_map["B.14"].full_text if "B.14" in section_map else ""
        endpoints = (
            _parse_endpoints(b10_text, "primary")
            + _parse_endpoints(b14_text, "safety")
        )

        # Visit schedule from TV
        follow_up_weeks, assessment_freq, study_days = self._extract_timing(
            dataset,
        )

        # Required SDTM domains from domain_mapping
        required_domains = self._extract_domains(dataset)

        # Visit count
        visit_count = len(dataset.tv) if not dataset.tv.empty else 0

        # Consolidate condition terms
        condition_terms = list(population.condition_keywords)
        for ep in endpoints:
            condition_terms.extend(ep.keywords)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_terms: list[str] = []
        for term in condition_terms:
            lower = term.lower()
            if lower not in seen:
                seen.add(lower)
                unique_terms.append(term)

        return NHSearchParams(
            population=population,
            endpoints=endpoints,
            follow_up_weeks=follow_up_weeks,
            assessment_frequency_weeks=assessment_freq,
            required_domains=required_domains,
            visit_count=visit_count,
            condition_terms=unique_terms,
            study_duration_days=study_days,
        )

    def generate_queries(
        self, params: NHSearchParams
    ) -> list[NHSearchQuery]:
        """Generate platform-specific search queries from NHSearchParams.

        Args:
            params: Structured search parameters.

        Returns:
            List of NHSearchQuery for each target platform.
        """
        queries: list[NHSearchQuery] = []

        queries.append(self._build_ctgov_query(params))
        queries.append(self._build_vivli_query(params))

        return queries

    def score_alignment(
        self,
        params: NHSearchParams,
        candidate: dict[str, Any],
    ) -> AlignmentScore:
        """Score alignment between trial parameters and a candidate dataset.

        Args:
            params: NHSearchParams from the trial protocol.
            candidate: Metadata dict for a candidate natural history dataset.
                Expected keys: "condition", "age_min", "age_max",
                "endpoints", "domains", "follow_up_weeks",
                "n_subjects".

        Returns:
            AlignmentScore with component and overall scores.
        """
        pop_score, pop_detail = self._score_population(params, candidate)
        ep_score, ep_detail = self._score_endpoints(params, candidate)
        temp_score, temp_detail = self._score_temporal(params, candidate)
        dom_score, dom_detail = self._score_domains(params, candidate)

        # Weighted average: population and endpoints are most important
        overall = (
            pop_score * 0.30
            + ep_score * 0.30
            + temp_score * 0.20
            + dom_score * 0.20
        )

        return AlignmentScore(
            population_score=round(pop_score, 3),
            endpoint_score=round(ep_score, 3),
            temporal_score=round(temp_score, 3),
            domain_score=round(dom_score, 3),
            overall_score=round(overall, 3),
            details={
                "population": pop_detail,
                "endpoint": ep_detail,
                "temporal": temp_detail,
                "domain": dom_detail,
            },
        )

    # ------------------------------------------------------------------
    # Private: timing extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_timing(
        dataset: "MockSdtmDataset",
    ) -> tuple[int, int, int]:
        """Extract follow-up duration and assessment frequency from TV.

        Returns:
            Tuple of (follow_up_weeks, assessment_freq_weeks, study_days).
        """
        if dataset.tv.empty:
            return 0, 0, 0

        visit_days = sorted(dataset.tv["VISITDY"].tolist())
        if len(visit_days) < 2:
            return 0, 0, 0

        min_day = int(min(visit_days))
        max_day = int(max(visit_days))
        study_days = max_day - min_day
        follow_up_weeks = max(1, study_days // 7)

        # Assessment frequency: median gap between consecutive visits
        gaps = [
            int(visit_days[i + 1] - visit_days[i])
            for i in range(len(visit_days) - 1)
            if visit_days[i + 1] - visit_days[i] > 0
        ]
        if gaps:
            median_gap = sorted(gaps)[len(gaps) // 2]
            assessment_freq = max(1, median_gap // 7)
        else:
            assessment_freq = follow_up_weeks

        return follow_up_weeks, assessment_freq, study_days

    @staticmethod
    def _extract_domains(dataset: "MockSdtmDataset") -> list[str]:
        """Extract required SDTM domains from domain_mapping."""
        if dataset.domain_mapping.empty:
            return []

        domains = sorted(dataset.domain_mapping["SDTM_DOMAIN"].unique().tolist())
        return domains

    # ------------------------------------------------------------------
    # Private: query builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ctgov_query(params: NHSearchParams) -> NHSearchQuery:
        """Build ClinicalTrials.gov observational study query."""
        condition = params.population.condition or "disease"
        terms = params.condition_terms[:5]
        endpoint_kws = []
        for ep in params.endpoints[:3]:
            endpoint_kws.extend(ep.keywords[:2])

        # Human-readable
        parts = [f"Observational studies in {condition}"]
        if params.population.age_min or params.population.age_max:
            age_str = ""
            if params.population.age_min:
                age_str += f"age >= {params.population.age_min}"
            if params.population.age_max:
                if age_str:
                    age_str += f" and <= {params.population.age_max}"
                else:
                    age_str += f"age <= {params.population.age_max}"
            parts.append(age_str)
        if endpoint_kws:
            parts.append(f"measuring {', '.join(endpoint_kws[:3])}")
        if params.follow_up_weeks > 0:
            parts.append(f"with >= {params.follow_up_weeks} weeks follow-up")

        human_readable = "; ".join(parts)

        # Structured params (ClinicalTrials.gov API v2 format)
        structured: dict[str, Any] = {
            "query.cond": condition,
            "filter.overallStatus": "COMPLETED",
            "filter.studyType": "OBSERVATIONAL",
        }
        if terms:
            structured["query.term"] = " OR ".join(terms)
        if endpoint_kws:
            structured["query.outc"] = " OR ".join(endpoint_kws[:3])
        if params.population.age_min:
            structured["filter.ageRange"] = (
                f"{params.population.age_min}+"
            )

        return NHSearchQuery(
            platform="ClinicalTrials.gov",
            human_readable=human_readable,
            structured_params=structured,
        )

    @staticmethod
    def _build_vivli_query(params: NHSearchParams) -> NHSearchQuery:
        """Build Vivli data sharing platform query."""
        condition = params.population.condition or "disease"
        terms = params.condition_terms[:5]

        parts = [f"Natural history / observational data for {condition}"]
        if params.required_domains:
            parts.append(f"with {', '.join(params.required_domains)} data")
        if params.follow_up_weeks > 0:
            parts.append(f">= {params.follow_up_weeks} weeks observation")

        human_readable = "; ".join(parts)

        structured: dict[str, Any] = {
            "condition": condition,
            "study_type": "observational",
            "data_available": True,
        }
        if terms:
            structured["keywords"] = terms
        if params.required_domains:
            structured["required_domains"] = params.required_domains

        return NHSearchQuery(
            platform="Vivli",
            human_readable=human_readable,
            structured_params=structured,
        )

    # ------------------------------------------------------------------
    # Private: alignment scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_population(
        params: NHSearchParams, candidate: dict[str, Any]
    ) -> tuple[float, str]:
        """Score population overlap."""
        score = 0.0
        components = 0

        # Condition match
        trial_cond = params.population.condition.lower()
        cand_cond = str(candidate.get("condition", "")).lower()
        if trial_cond and cand_cond:
            components += 1
            if trial_cond in cand_cond or cand_cond in trial_cond:
                score += 1.0
            elif any(
                kw.lower() in cand_cond
                for kw in params.population.condition_keywords
            ):
                score += 0.5

        # Age range overlap
        cand_age_min = candidate.get("age_min")
        cand_age_max = candidate.get("age_max")
        if params.population.age_min is not None and cand_age_min is not None:
            components += 1
            trial_min = params.population.age_min
            trial_max = params.population.age_max or 100
            c_min = int(cand_age_min)
            c_max = int(cand_age_max) if cand_age_max else 100

            overlap_low = max(trial_min, c_min)
            overlap_high = min(trial_max, c_max)
            trial_range = trial_max - trial_min
            if trial_range > 0 and overlap_high > overlap_low:
                score += min(1.0, (overlap_high - overlap_low) / trial_range)

        if components == 0:
            return 0.5, "No population data to compare"

        final = score / components
        detail = f"Condition: {trial_cond or 'N/A'} vs {cand_cond or 'N/A'}"
        return round(final, 3), detail

    @staticmethod
    def _score_endpoints(
        params: NHSearchParams, candidate: dict[str, Any]
    ) -> tuple[float, str]:
        """Score endpoint coverage."""
        if not params.endpoints:
            return 0.5, "No endpoints specified in protocol"

        cand_endpoints = [
            str(e).lower() for e in candidate.get("endpoints", [])
        ]
        if not cand_endpoints:
            return 0.0, "Candidate has no endpoint data"

        matched = 0
        for ep in params.endpoints:
            ep_keywords = [kw.lower() for kw in ep.keywords]
            if any(kw in " ".join(cand_endpoints) for kw in ep_keywords):
                matched += 1

        score = matched / len(params.endpoints)
        detail = f"{matched}/{len(params.endpoints)} endpoints covered"
        return round(score, 3), detail

    @staticmethod
    def _score_temporal(
        params: NHSearchParams, candidate: dict[str, Any]
    ) -> tuple[float, str]:
        """Score temporal alignment."""
        if params.follow_up_weeks <= 0:
            return 0.5, "No follow-up duration specified"

        cand_weeks = candidate.get("follow_up_weeks", 0)
        if not cand_weeks or cand_weeks <= 0:
            return 0.0, "Candidate has no follow-up data"

        cand_weeks = int(cand_weeks)
        # Score based on whether candidate observation covers trial duration
        if cand_weeks >= params.follow_up_weeks:
            score = 1.0
        else:
            score = cand_weeks / params.follow_up_weeks

        detail = f"Trial: {params.follow_up_weeks}wk vs Candidate: {cand_weeks}wk"
        return round(score, 3), detail

    @staticmethod
    def _score_domains(
        params: NHSearchParams, candidate: dict[str, Any]
    ) -> tuple[float, str]:
        """Score data domain coverage."""
        if not params.required_domains:
            return 0.5, "No domains specified"

        cand_domains = [
            str(d).upper() for d in candidate.get("domains", [])
        ]
        if not cand_domains:
            return 0.0, "Candidate has no domain data"

        matched = sum(
            1 for d in params.required_domains if d in cand_domains
        )
        score = matched / len(params.required_domains)
        detail = f"{matched}/{len(params.required_domains)} domains available"
        return round(score, 3), detail
