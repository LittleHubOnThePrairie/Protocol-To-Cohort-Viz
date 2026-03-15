"""Registry-PDF Cross-Validation for ICH sections.

PTCV-197: Compares PDF-extracted ICH sections against registry metadata
to detect extraction errors, misclassifications, and missing sections.
When registry data contradicts PDF extraction, flags discrepancies for
review or supplies registry content as fallback.

Comparison methods:
- Text similarity (Jaccard token overlap)
- Keyword overlap (named-entity style matching)
- Structural matching (list count comparison for outcomes/criteria)

Risk tier: LOW — read-only comparison (no mutations to source data).
"""

import dataclasses
import json
import logging
import re
from enum import Enum
from typing import Any, Optional

from ..ich_parser.models import IchSection
from .ich_mapper import MappedRegistrySection

logger = logging.getLogger(__name__)

# Confidence boost when registry confirms PDF extraction
_CONFIRMATION_BOOST = 0.2

# Similarity threshold for "confirmed" status
_CONFIRMATION_THRESHOLD = 0.3


class SectionStatus(Enum):
    """Cross-validation status for a single ICH section."""

    CONFIRMED = "confirmed"
    DIVERGENT = "divergent"
    MISSING_FROM_PDF = "missing_from_pdf"
    REGISTRY_ONLY = "registry_only"
    PDF_ONLY = "pdf_only"


@dataclasses.dataclass
class SectionValidation:
    """Cross-validation result for one ICH section.

    Attributes:
        section_code: ICH section code (e.g. "B.4").
        section_name: Human-readable section name.
        status: Validation outcome.
        pdf_confidence: Confidence score from PDF extraction (None if
            missing from PDF).
        registry_quality: Quality rating from registry mapping (None if
            missing from registry).
        similarity_score: Token-level Jaccard similarity between the
            two content sources (0.0–1.0). None when only one source
            is available.
        boosted_confidence: PDF confidence after applying registry
            confirmation boost (only set when status is CONFIRMED).
        detail: Human-readable explanation of the finding.
        registry_content: Registry content text, provided as fallback
            when section is missing from PDF or divergent.
        pdf_content: PDF-extracted content text (when available).
        structural_note: Additional structural comparison note
            (e.g. endpoint count mismatch).
    """

    section_code: str
    section_name: str
    status: SectionStatus
    pdf_confidence: Optional[float] = None
    registry_quality: Optional[float] = None
    similarity_score: Optional[float] = None
    boosted_confidence: Optional[float] = None
    detail: str = ""
    registry_content: str = ""
    pdf_content: str = ""
    structural_note: str = ""


@dataclasses.dataclass
class CrossValidationReport:
    """Aggregate cross-validation report for one protocol.

    Attributes:
        registry_id: Trial identifier.
        sections: Per-section validation results.
        confirmed_count: Sections confirmed by registry match.
        divergent_count: Sections where PDF and registry disagree.
        missing_from_pdf_count: Sections in registry but not in PDF.
        registry_only_count: Sections synthesised from registry only.
        pdf_only_count: Sections in PDF but not in registry.
        synthetic_sections: IchSection objects created from registry
            data for missing PDF sections.
    """

    registry_id: str
    sections: list[SectionValidation] = dataclasses.field(
        default_factory=list
    )
    confirmed_count: int = 0
    divergent_count: int = 0
    missing_from_pdf_count: int = 0
    registry_only_count: int = 0
    pdf_only_count: int = 0
    synthetic_sections: list[IchSection] = dataclasses.field(
        default_factory=list
    )

    def summary(self) -> str:
        """Return a one-line summary of the report."""
        parts = []
        if self.confirmed_count:
            parts.append(f"{self.confirmed_count} confirmed")
        if self.divergent_count:
            parts.append(f"{self.divergent_count} divergent")
        if self.missing_from_pdf_count:
            parts.append(f"{self.missing_from_pdf_count} missing from PDF")
        if self.registry_only_count:
            parts.append(f"{self.registry_only_count} registry-only")
        if self.pdf_only_count:
            parts.append(f"{self.pdf_only_count} PDF-only")
        return f"{self.registry_id}: " + ", ".join(parts)


def _tokenize(text: str) -> set[str]:
    """Lowercase token set from text, stripping punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _count_list_items(content_json: str, key: str) -> Optional[int]:
    """Count items in a JSON list field, or None if absent."""
    try:
        data = json.loads(content_json)
        val = data.get(key)
        if isinstance(val, list):
            return len(val)
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


class CrossValidator:
    """Compare PDF-extracted ICH sections against registry metadata.

    Usage::

        validator = CrossValidator()
        report = validator.validate(
            registry_id="NCT01512251",
            pdf_sections=ich_sections,
            registry_sections=mapped_sections,
        )
    """

    def validate(
        self,
        registry_id: str,
        pdf_sections: list[IchSection],
        registry_sections: list[MappedRegistrySection],
    ) -> CrossValidationReport:
        """Run cross-validation and return a report.

        Args:
            registry_id: Trial registry identifier.
            pdf_sections: ICH sections extracted from the PDF.
            registry_sections: ICH sections mapped from registry
                metadata.

        Returns:
            CrossValidationReport with per-section results.
        """
        pdf_by_code: dict[str, IchSection] = {
            s.section_code: s for s in pdf_sections
        }
        reg_by_code: dict[str, MappedRegistrySection] = {
            s.section_code: s for s in registry_sections
        }

        all_codes = sorted(set(pdf_by_code) | set(reg_by_code))
        report = CrossValidationReport(registry_id=registry_id)

        for code in all_codes:
            pdf_sec = pdf_by_code.get(code)
            reg_sec = reg_by_code.get(code)

            if pdf_sec and reg_sec:
                validation = self._compare(pdf_sec, reg_sec)
            elif reg_sec and not pdf_sec:
                validation = self._missing_from_pdf(reg_sec)
                synthetic = self._make_synthetic(
                    reg_sec, registry_id
                )
                report.synthetic_sections.append(synthetic)
            else:
                assert pdf_sec is not None
                validation = self._pdf_only(pdf_sec)

            report.sections.append(validation)

        # Update counts
        for sv in report.sections:
            if sv.status == SectionStatus.CONFIRMED:
                report.confirmed_count += 1
            elif sv.status == SectionStatus.DIVERGENT:
                report.divergent_count += 1
            elif sv.status == SectionStatus.MISSING_FROM_PDF:
                report.missing_from_pdf_count += 1
            elif sv.status == SectionStatus.REGISTRY_ONLY:
                report.registry_only_count += 1
            elif sv.status == SectionStatus.PDF_ONLY:
                report.pdf_only_count += 1

        logger.info(
            "Cross-validation for %s: %s",
            registry_id,
            report.summary(),
        )
        return report

    def _compare(
        self,
        pdf_sec: IchSection,
        reg_sec: MappedRegistrySection,
    ) -> SectionValidation:
        """Compare a section present in both sources."""
        pdf_tokens = _tokenize(pdf_sec.content_text)
        reg_tokens = _tokenize(reg_sec.content_text)
        similarity = _jaccard_similarity(pdf_tokens, reg_tokens)

        structural_note = self._structural_compare(
            pdf_sec, reg_sec
        )

        if similarity >= _CONFIRMATION_THRESHOLD:
            boosted = min(
                pdf_sec.confidence_score + _CONFIRMATION_BOOST, 1.0
            )
            return SectionValidation(
                section_code=reg_sec.section_code,
                section_name=reg_sec.section_name,
                status=SectionStatus.CONFIRMED,
                pdf_confidence=pdf_sec.confidence_score,
                registry_quality=reg_sec.quality_rating,
                similarity_score=similarity,
                boosted_confidence=boosted,
                detail=(
                    f"Registry confirms PDF extraction "
                    f"(similarity={similarity:.2f}). "
                    f"Confidence boosted from "
                    f"{pdf_sec.confidence_score:.2f} to "
                    f"{boosted:.2f}."
                ),
                registry_content=reg_sec.content_text,
                pdf_content=pdf_sec.content_text,
                structural_note=structural_note,
            )

        return SectionValidation(
            section_code=reg_sec.section_code,
            section_name=reg_sec.section_name,
            status=SectionStatus.DIVERGENT,
            pdf_confidence=pdf_sec.confidence_score,
            registry_quality=reg_sec.quality_rating,
            similarity_score=similarity,
            detail=(
                f"PDF and registry content diverge "
                f"(similarity={similarity:.2f}). "
                f"Review recommended."
            ),
            registry_content=reg_sec.content_text,
            pdf_content=pdf_sec.content_text,
            structural_note=structural_note,
        )

    def _structural_compare(
        self,
        pdf_sec: IchSection,
        reg_sec: MappedRegistrySection,
    ) -> str:
        """Compare structural elements (list counts) between sources."""
        notes: list[str] = []

        # B.8: Compare endpoint counts
        if reg_sec.section_code == "B.8":
            reg_primary = _count_list_items(
                reg_sec.content_json, "primary_outcomes"
            )
            pdf_primary = _count_list_items(
                pdf_sec.content_json, "primary_outcomes"
            )
            if (
                reg_primary is not None
                and pdf_primary is not None
                and reg_primary != pdf_primary
            ):
                notes.append(
                    f"PDF: {pdf_primary} primary endpoints, "
                    f"Registry: {reg_primary} primary endpoints "
                    f"— possible extraction gap"
                )

            reg_secondary = _count_list_items(
                reg_sec.content_json, "secondary_outcomes"
            )
            pdf_secondary = _count_list_items(
                pdf_sec.content_json, "secondary_outcomes"
            )
            if (
                reg_secondary is not None
                and pdf_secondary is not None
                and reg_secondary != pdf_secondary
            ):
                notes.append(
                    f"PDF: {pdf_secondary} secondary endpoints, "
                    f"Registry: {reg_secondary} secondary endpoints"
                )

        # B.4: Compare intervention counts
        if reg_sec.section_code == "B.4":
            reg_intv = _count_list_items(
                reg_sec.content_json, "interventions"
            )
            pdf_intv = _count_list_items(
                pdf_sec.content_json, "interventions"
            )
            if (
                reg_intv is not None
                and pdf_intv is not None
                and reg_intv != pdf_intv
            ):
                notes.append(
                    f"PDF: {pdf_intv} interventions, "
                    f"Registry: {reg_intv} interventions"
                )

        return "; ".join(notes)

    def _missing_from_pdf(
        self,
        reg_sec: MappedRegistrySection,
    ) -> SectionValidation:
        """Handle a section present in registry but not in PDF."""
        return SectionValidation(
            section_code=reg_sec.section_code,
            section_name=reg_sec.section_name,
            status=SectionStatus.MISSING_FROM_PDF,
            registry_quality=reg_sec.quality_rating,
            detail=(
                f"{reg_sec.section_code} ({reg_sec.section_name}) "
                f"missing from PDF, present in registry. "
                f"Registry fallback available "
                f"(quality={reg_sec.quality_rating:.1f})."
            ),
            registry_content=reg_sec.content_text,
        )

    def _pdf_only(
        self,
        pdf_sec: IchSection,
    ) -> SectionValidation:
        """Handle a section present in PDF but not in registry."""
        return SectionValidation(
            section_code=pdf_sec.section_code,
            section_name=pdf_sec.section_name,
            status=SectionStatus.PDF_ONLY,
            pdf_confidence=pdf_sec.confidence_score,
            detail=(
                f"{pdf_sec.section_code} ({pdf_sec.section_name}) "
                f"present in PDF only — no registry equivalent."
            ),
            pdf_content=pdf_sec.content_text,
        )

    def _make_synthetic(
        self,
        reg_sec: MappedRegistrySection,
        registry_id: str,
    ) -> IchSection:
        """Create a synthetic IchSection from registry data."""
        return IchSection(
            run_id="",
            source_run_id="",
            source_sha256="",
            registry_id=registry_id,
            section_code=reg_sec.section_code,
            section_name=reg_sec.section_name,
            content_json=reg_sec.content_json,
            confidence_score=reg_sec.quality_rating,
            review_required=reg_sec.quality_rating < 0.70,
            legacy_format=False,
            content_text=reg_sec.content_text,
        )
