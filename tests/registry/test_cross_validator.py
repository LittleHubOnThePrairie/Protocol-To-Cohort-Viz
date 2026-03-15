"""Tests for CrossValidator (PTCV-197).

Covers all GHERKIN scenarios:
- Detect missing PDF sections using registry data
- Validate PDF-extracted endpoints against registry
- Confirm correct extraction via registry match
- Handle registry-only sections gracefully

Qualification phase: OQ (operational qualification)
Risk tier: LOW
"""

import json

import pytest

from ptcv.ich_parser.models import IchSection
from ptcv.registry.cross_validator import (
    CrossValidationReport,
    CrossValidator,
    SectionStatus,
    SectionValidation,
    _jaccard_similarity,
    _tokenize,
)
from ptcv.registry.ich_mapper import MappedRegistrySection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ich_section(
    section_code: str,
    section_name: str = "Test Section",
    content_text: str = "",
    content_json: str = "{}",
    confidence_score: float = 0.8,
) -> IchSection:
    """Build a minimal IchSection for testing."""
    return IchSection(
        run_id="test-run",
        source_run_id="src-run",
        source_sha256="abc123",
        registry_id="NCT01512251",
        section_code=section_code,
        section_name=section_name,
        content_json=content_json,
        confidence_score=confidence_score,
        review_required=confidence_score < 0.70,
        legacy_format=False,
        content_text=content_text,
    )


def _make_registry_section(
    section_code: str,
    section_name: str = "Test Section",
    content_text: str = "",
    content_json: str = "{}",
    quality_rating: float = 0.9,
) -> MappedRegistrySection:
    """Build a minimal MappedRegistrySection for testing."""
    return MappedRegistrySection(
        section_code=section_code,
        section_name=section_name,
        content_text=content_text,
        content_json=content_json,
        quality_rating=quality_rating,
    )


@pytest.fixture
def validator() -> CrossValidator:
    return CrossValidator()


# ---------------------------------------------------------------------------
# Scenario 1: Detect missing PDF sections using registry data
# ---------------------------------------------------------------------------


class TestDetectMissingSections:
    """Scenario: Detect missing PDF sections using registry data."""

    def test_missing_section_flagged(
        self, validator: CrossValidator
    ) -> None:
        """Registry B.7 present, PDF B.7 absent → missing_from_pdf."""
        reg = [
            _make_registry_section(
                "B.7",
                "Treatment",
                content_text="Primary endpoint: ORR",
                quality_rating=0.9,
            ),
        ]
        report = validator.validate("NCT01512251", [], reg)

        assert len(report.sections) == 1
        sv = report.sections[0]
        assert sv.section_code == "B.7"
        assert sv.status == SectionStatus.MISSING_FROM_PDF
        assert sv.registry_content == "Primary endpoint: ORR"
        assert report.missing_from_pdf_count == 1

    def test_missing_section_suggests_fallback(
        self, validator: CrossValidator
    ) -> None:
        """Fallback suggestion includes the registry quality rating."""
        reg = [
            _make_registry_section(
                "B.7",
                "Treatment",
                content_text="Primary endpoint: ORR",
                quality_rating=0.9,
            ),
        ]
        report = validator.validate("NCT01512251", [], reg)

        sv = report.sections[0]
        assert "quality=0.9" in sv.detail
        assert "fallback" in sv.detail.lower()

    def test_missing_section_creates_synthetic(
        self, validator: CrossValidator
    ) -> None:
        """Missing PDF section generates a synthetic IchSection."""
        reg = [
            _make_registry_section(
                "B.7",
                "Treatment",
                content_text="Primary endpoint: ORR",
                content_json='{"endpoints": ["ORR"]}',
                quality_rating=0.9,
            ),
        ]
        report = validator.validate("NCT01512251", [], reg)

        assert len(report.synthetic_sections) == 1
        synth = report.synthetic_sections[0]
        assert synth.section_code == "B.7"
        assert synth.registry_id == "NCT01512251"
        assert synth.confidence_score == 0.9

    def test_multiple_missing_sections(
        self, validator: CrossValidator
    ) -> None:
        """Multiple registry sections missing from PDF all flagged."""
        reg = [
            _make_registry_section("B.7", "Treatment"),
            _make_registry_section("B.8", "Efficacy"),
        ]
        report = validator.validate("NCT01512251", [], reg)

        assert report.missing_from_pdf_count == 2
        assert len(report.synthetic_sections) == 2


# ---------------------------------------------------------------------------
# Scenario 2: Validate PDF-extracted endpoints against registry
# ---------------------------------------------------------------------------


class TestValidateEndpoints:
    """Scenario: Validate PDF-extracted endpoints against registry."""

    def test_endpoint_count_mismatch_reported(
        self, validator: CrossValidator
    ) -> None:
        """PDF has 2 primary endpoints, registry has 3 → structural note."""
        pdf_json = json.dumps({
            "primary_outcomes": [
                {"measure": "ORR"},
                {"measure": "DLT"},
            ]
        })
        reg_json = json.dumps({
            "primary_outcomes": [
                {"measure": "ORR"},
                {"measure": "DLT"},
                {"measure": "MTD"},
            ]
        })
        pdf = [
            _make_ich_section(
                "B.8",
                "Assessment of Efficacy",
                content_text="Primary: ORR, DLT",
                content_json=pdf_json,
            ),
        ]
        reg = [
            _make_registry_section(
                "B.8",
                "Assessment of Efficacy",
                content_text="Primary: ORR, DLT, MTD",
                content_json=reg_json,
            ),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        sv = report.sections[0]
        assert "2 primary endpoints" in sv.structural_note
        assert "3 primary endpoints" in sv.structural_note
        assert "extraction gap" in sv.structural_note

    def test_matching_endpoints_no_structural_note(
        self, validator: CrossValidator
    ) -> None:
        """Same endpoint count → no structural note."""
        outcomes_json = json.dumps({
            "primary_outcomes": [
                {"measure": "ORR"},
                {"measure": "DLT"},
            ]
        })
        pdf = [
            _make_ich_section(
                "B.8",
                "Assessment of Efficacy",
                content_text="Primary: ORR, DLT",
                content_json=outcomes_json,
            ),
        ]
        reg = [
            _make_registry_section(
                "B.8",
                "Assessment of Efficacy",
                content_text="Primary: ORR, DLT",
                content_json=outcomes_json,
            ),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        sv = report.sections[0]
        assert sv.structural_note == ""

    def test_includes_missing_endpoint_from_registry(
        self, validator: CrossValidator
    ) -> None:
        """Divergent section includes registry content for review."""
        pdf = [
            _make_ich_section(
                "B.8",
                "Assessment of Efficacy",
                content_text="completely different content here",
            ),
        ]
        reg = [
            _make_registry_section(
                "B.8",
                "Assessment of Efficacy",
                content_text="Primary: ORR, DLT, MTD",
            ),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        sv = report.sections[0]
        assert sv.registry_content == "Primary: ORR, DLT, MTD"
        assert sv.pdf_content == "completely different content here"


# ---------------------------------------------------------------------------
# Scenario 3: Confirm correct extraction via registry match
# ---------------------------------------------------------------------------


class TestConfirmExtraction:
    """Scenario: Confirm correct extraction via registry match."""

    def test_matching_content_confirmed(
        self, validator: CrossValidator
    ) -> None:
        """Matching B.6 text → status CONFIRMED."""
        text = (
            "Eligibility Criteria: BRAF V600E/K mutation, "
            "age 18+, ECOG 0-1"
        )
        pdf = [
            _make_ich_section(
                "B.5",
                "Selection of Subjects",
                content_text=text,
                confidence_score=0.75,
            ),
        ]
        reg = [
            _make_registry_section(
                "B.5",
                "Selection of Subjects",
                content_text=text,
            ),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        sv = report.sections[0]
        assert sv.status == SectionStatus.CONFIRMED
        assert report.confirmed_count == 1

    def test_confidence_boosted_by_0_2(
        self, validator: CrossValidator
    ) -> None:
        """Confirmed section gets confidence boosted by 0.2."""
        text = "Eligibility: BRAF V600E/K, age 18+"
        pdf = [
            _make_ich_section(
                "B.5",
                "Selection of Subjects",
                content_text=text,
                confidence_score=0.75,
            ),
        ]
        reg = [
            _make_registry_section(
                "B.5",
                "Selection of Subjects",
                content_text=text,
            ),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        sv = report.sections[0]
        assert sv.boosted_confidence == pytest.approx(0.95)

    def test_confidence_boost_capped_at_1(
        self, validator: CrossValidator
    ) -> None:
        """Boosted confidence does not exceed 1.0."""
        text = "Same content in both sources"
        pdf = [
            _make_ich_section(
                "B.3",
                "Trial Objectives",
                content_text=text,
                confidence_score=0.95,
            ),
        ]
        reg = [
            _make_registry_section(
                "B.3",
                "Trial Objectives",
                content_text=text,
            ),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        sv = report.sections[0]
        assert sv.boosted_confidence == 1.0

    def test_high_confidence_detail_message(
        self, validator: CrossValidator
    ) -> None:
        """Confirmed section detail mentions similarity and boost."""
        text = "Study type: INTERVENTIONAL, Phase 1/2"
        pdf = [
            _make_ich_section(
                "B.4",
                "Trial Design",
                content_text=text,
                confidence_score=0.80,
            ),
        ]
        reg = [
            _make_registry_section(
                "B.4",
                "Trial Design",
                content_text=text,
            ),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        sv = report.sections[0]
        assert "confirms" in sv.detail.lower()
        assert "boosted" in sv.detail.lower()


# ---------------------------------------------------------------------------
# Scenario 4: Handle registry-only sections gracefully
# ---------------------------------------------------------------------------


class TestRegistryOnlySections:
    """Scenario: Handle registry-only sections gracefully."""

    def test_registry_only_creates_synthetic_section(
        self, validator: CrossValidator
    ) -> None:
        """Registry B.10 with no PDF equivalent → synthetic IchSection."""
        reg = [
            _make_registry_section(
                "B.10",
                "Statistics",
                content_text="Enrollment: 30 (ACTUAL)",
                content_json='{"enrollment_count": 30}',
                quality_rating=0.6,
            ),
        ]
        report = validator.validate("NCT01512251", [], reg)

        assert len(report.synthetic_sections) == 1
        synth = report.synthetic_sections[0]
        assert synth.section_code == "B.10"
        assert synth.content_text == "Enrollment: 30 (ACTUAL)"
        assert synth.confidence_score == 0.6
        assert synth.review_required is True  # 0.6 < 0.70

    def test_synthetic_section_marked_as_registry_source(
        self, validator: CrossValidator
    ) -> None:
        """Synthetic section has source traceability to registry."""
        reg = [
            _make_registry_section(
                "B.16",
                "Publication Policy",
                content_text="References: Smith 2015",
                quality_rating=0.9,
            ),
        ]
        report = validator.validate("NCT01512251", [], reg)

        synth = report.synthetic_sections[0]
        assert synth.registry_id == "NCT01512251"
        assert synth.review_required is False  # 0.9 >= 0.70

    def test_high_quality_registry_not_review_required(
        self, validator: CrossValidator
    ) -> None:
        """Direct-mapped registry section (0.9) not flagged for review."""
        reg = [
            _make_registry_section(
                "B.5",
                "Selection of Subjects",
                content_text="Eligibility: age 18+",
                quality_rating=0.9,
            ),
        ]
        report = validator.validate("NCT01512251", [], reg)

        synth = report.synthetic_sections[0]
        assert synth.review_required is False

    def test_low_quality_registry_flagged_for_review(
        self, validator: CrossValidator
    ) -> None:
        """Contextual-mapped registry section (0.3) flagged for review."""
        reg = [
            _make_registry_section(
                "B.2",
                "Background Information",
                content_text="Conditions: Melanoma",
                quality_rating=0.3,
            ),
        ]
        report = validator.validate("NCT01512251", [], reg)

        synth = report.synthetic_sections[0]
        assert synth.review_required is True


# ---------------------------------------------------------------------------
# PDF-only sections (no registry equivalent)
# ---------------------------------------------------------------------------


class TestPdfOnlySections:
    """Sections in PDF but not in registry."""

    def test_pdf_only_section_reported(
        self, validator: CrossValidator
    ) -> None:
        """PDF section with no registry match → PDF_ONLY."""
        pdf = [
            _make_ich_section(
                "B.9",
                "Assessment of Safety",
                content_text="AE monitoring per CTCAE v5",
            ),
        ]
        report = validator.validate("NCT01512251", pdf, [])

        assert len(report.sections) == 1
        sv = report.sections[0]
        assert sv.status == SectionStatus.PDF_ONLY
        assert sv.pdf_content == "AE monitoring per CTCAE v5"
        assert report.pdf_only_count == 1

    def test_pdf_only_no_synthetic_created(
        self, validator: CrossValidator
    ) -> None:
        """PDF-only sections do not create synthetic sections."""
        pdf = [
            _make_ich_section("B.9", "Assessment of Safety"),
        ]
        report = validator.validate("NCT01512251", pdf, [])

        assert len(report.synthetic_sections) == 0


# ---------------------------------------------------------------------------
# Divergent sections
# ---------------------------------------------------------------------------


class TestDivergentSections:
    """Sections present in both but with low similarity."""

    def test_divergent_status_on_low_similarity(
        self, validator: CrossValidator
    ) -> None:
        """Completely different content → DIVERGENT."""
        pdf = [
            _make_ich_section(
                "B.4",
                "Trial Design",
                content_text="Single-arm open-label dose escalation",
            ),
        ]
        reg = [
            _make_registry_section(
                "B.4",
                "Trial Design",
                content_text="Randomized double-blind placebo controlled",
            ),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        sv = report.sections[0]
        assert sv.status == SectionStatus.DIVERGENT
        assert report.divergent_count == 1
        assert sv.similarity_score is not None
        assert "review" in sv.detail.lower()


# ---------------------------------------------------------------------------
# Report summary
# ---------------------------------------------------------------------------


class TestReportSummary:
    """CrossValidationReport.summary() formatting."""

    def test_summary_includes_counts(
        self, validator: CrossValidator
    ) -> None:
        """Summary string includes all non-zero counts."""
        pdf = [
            _make_ich_section(
                "B.3",
                "Trial Objectives",
                content_text="Study of BKM120",
            ),
            _make_ich_section("B.9", "Assessment of Safety"),
        ]
        reg = [
            _make_registry_section(
                "B.3",
                "Trial Objectives",
                content_text="Study of BKM120",
            ),
            _make_registry_section("B.10", "Statistics"),
        ]
        report = validator.validate("NCT01512251", pdf, reg)

        summary = report.summary()
        assert "NCT01512251" in summary
        assert "confirmed" in summary or "divergent" in summary

    def test_empty_report_summary(
        self, validator: CrossValidator
    ) -> None:
        """Empty inputs produce a valid summary."""
        report = validator.validate("NCT01512251", [], [])
        assert "NCT01512251" in report.summary()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestTokenize:
    """_tokenize helper."""

    def test_basic_tokenization(self) -> None:
        tokens = _tokenize("Hello, World! 123")
        assert tokens == {"hello", "world", "123"}

    def test_empty_string(self) -> None:
        assert _tokenize("") == set()


class TestJaccardSimilarity:
    """_jaccard_similarity helper."""

    def test_identical_sets(self) -> None:
        assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert _jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self) -> None:
        # {a, b} ∩ {b, c} = {b}, |union| = 3 → 1/3
        result = _jaccard_similarity({"a", "b"}, {"b", "c"})
        assert result == pytest.approx(1 / 3)

    def test_both_empty(self) -> None:
        assert _jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self) -> None:
        assert _jaccard_similarity({"a"}, set()) == 0.0
