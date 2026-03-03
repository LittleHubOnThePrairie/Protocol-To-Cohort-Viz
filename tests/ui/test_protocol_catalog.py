"""Unit tests for protocol_catalog (PTCV-42).

Tests the pure-Python therapeutic area classifier, quality score
sorting, display label formatting, and catalog loading logic
without Streamlit.

Qualification: Installation Qualification (IQ)
Regulatory: N/A — UI display logic, no regulated data mutation
Risk Tier: LOW — metadata classification only
Module: ptcv.ui.components.protocol_catalog
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ptcv.ui.components.protocol_catalog import (
    AREA_DISPLAY_ORDER,
    ProtocolEntry,
    QualityScores,
    TherapeuticArea,
    classify_therapeutic_area,
    load_protocol_catalog,
)


# ---------------------------------------------------------------------------
# classify_therapeutic_area
# ---------------------------------------------------------------------------

class TestClassifyTherapeuticArea:
    """Tests for classify_therapeutic_area()."""

    def test_oncology_cancer(self) -> None:
        assert (
            classify_therapeutic_area("Breast Cancer")
            == TherapeuticArea.ONCOLOGY
        )

    def test_oncology_lymphoma(self) -> None:
        assert (
            classify_therapeutic_area("Non-Hodgkin Lymphoma")
            == TherapeuticArea.ONCOLOGY
        )

    def test_oncology_myeloma(self) -> None:
        assert (
            classify_therapeutic_area(
                "Multiple Myeloma, Plasma Cell Neoplasm"
            )
            == TherapeuticArea.ONCOLOGY
        )

    def test_oncology_cml(self) -> None:
        assert (
            classify_therapeutic_area("CML")
            == TherapeuticArea.ONCOLOGY
        )

    def test_oncology_waldenstrom(self) -> None:
        assert (
            classify_therapeutic_area(
                "Waldenstrom's Macroglobulinemia"
            )
            == TherapeuticArea.ONCOLOGY
        )

    def test_oncology_neuroblastoma(self) -> None:
        """Neuroblastoma is oncology, not nervous system."""
        assert (
            classify_therapeutic_area("Neuroblastoma")
            == TherapeuticArea.ONCOLOGY
        )

    def test_oncology_glioblastoma(self) -> None:
        """Glioblastoma is oncology, not nervous system."""
        assert (
            classify_therapeutic_area("Glioblastoma")
            == TherapeuticArea.ONCOLOGY
        )

    def test_oncology_myelodysplastic(self) -> None:
        assert (
            classify_therapeutic_area(
                "Myelodysplastic Syndrome"
            )
            == TherapeuticArea.ONCOLOGY
        )

    def test_cardiovascular_heart_failure(self) -> None:
        assert (
            classify_therapeutic_area("Heart Failure")
            == TherapeuticArea.CARDIOVASCULAR
        )

    def test_cardiovascular_stroke(self) -> None:
        assert (
            classify_therapeutic_area("Acute Ischemic Stroke")
            == TherapeuticArea.CARDIOVASCULAR
        )

    def test_cardiovascular_cardiorenal(self) -> None:
        assert (
            classify_therapeutic_area("Cardiorenal Syndrome")
            == TherapeuticArea.CARDIOVASCULAR
        )

    def test_cardiovascular_hypertension(self) -> None:
        assert (
            classify_therapeutic_area(
                "Pulmonary Arterial Hypertension"
            )
            == TherapeuticArea.CARDIOVASCULAR
        )

    def test_nervous_system_alzheimer(self) -> None:
        assert (
            classify_therapeutic_area("Alzheimer Disease")
            == TherapeuticArea.NERVOUS_SYSTEM
        )

    def test_nervous_system_epilepsy(self) -> None:
        assert (
            classify_therapeutic_area("Drug-resistant Epilepsy")
            == TherapeuticArea.NERVOUS_SYSTEM
        )

    def test_nervous_system_neurofibroma(self) -> None:
        assert (
            classify_therapeutic_area(
                "Neurofibromatosis 1, Cutaneous Neurofibroma"
            )
            == TherapeuticArea.NERVOUS_SYSTEM
        )

    def test_nervous_system_anosmia(self) -> None:
        assert (
            classify_therapeutic_area("Anosmia, Hyposmia")
            == TherapeuticArea.NERVOUS_SYSTEM
        )

    def test_diabetes_type2(self) -> None:
        assert (
            classify_therapeutic_area("Type 2 Diabetes Mellitus")
            == TherapeuticArea.DIABETES_METABOLIC
        )

    def test_diabetes_metabolic(self) -> None:
        assert (
            classify_therapeutic_area("Metabolic Syndrome")
            == TherapeuticArea.DIABETES_METABOLIC
        )

    def test_diabetes_acromegaly(self) -> None:
        assert (
            classify_therapeutic_area("Acromegaly")
            == TherapeuticArea.DIABETES_METABOLIC
        )

    def test_other_unmatched(self) -> None:
        assert (
            classify_therapeutic_area("Hereditary Angioedema")
            == TherapeuticArea.OTHER
        )

    def test_other_empty_string(self) -> None:
        assert (
            classify_therapeutic_area("")
            == TherapeuticArea.OTHER
        )

    def test_case_insensitive(self) -> None:
        assert (
            classify_therapeutic_area("BREAST CANCER")
            == TherapeuticArea.ONCOLOGY
        )

    def test_comma_separated_first_match(self) -> None:
        """Multi-condition string matches first keyword found."""
        assert (
            classify_therapeutic_area(
                "Leukemia, Myelodysplastic Syndrome"
            )
            == TherapeuticArea.ONCOLOGY
        )


# ---------------------------------------------------------------------------
# QualityScores
# ---------------------------------------------------------------------------

class TestQualityScores:
    """Tests for QualityScores defaults."""

    def test_default_values(self) -> None:
        qs = QualityScores()
        assert qs.format_verdict == ""
        assert qs.format_confidence == 0.0
        assert qs.section_count == 0

    def test_frozen(self) -> None:
        qs = QualityScores(format_verdict="ICH_E6R3")
        with pytest.raises(AttributeError):
            qs.format_verdict = "NON_ICH"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ProtocolEntry.sort_key
# ---------------------------------------------------------------------------

def _make_entry(
    verdict: str = "",
    confidence: float = 0.0,
    registry_id: str = "NCT001",
) -> ProtocolEntry:
    """Helper to create a ProtocolEntry for sort testing."""
    return ProtocolEntry(
        registry_id=registry_id,
        title="Test Protocol",
        condition="Test",
        therapeutic_area=TherapeuticArea.OTHER,
        registry_source="clinicaltrials",
        filename=f"{registry_id}_1.0.pdf",
        file_path=Path(f"/fake/{registry_id}_1.0.pdf"),
        quality=QualityScores(
            format_verdict=verdict,
            format_confidence=confidence,
        ),
    )


class TestSortKey:
    """Tests for ProtocolEntry.sort_key ordering."""

    def test_verdict_tier_ordering(self) -> None:
        """ICH_E6R3 < PARTIAL_ICH < NON_ICH < unscored."""
        ich = _make_entry("ICH_E6R3", 0.9)
        partial = _make_entry("PARTIAL_ICH", 0.9)
        non_ich = _make_entry("NON_ICH", 0.9)
        unscored = _make_entry("", 0.0)

        entries = [unscored, non_ich, partial, ich]
        entries.sort(key=lambda e: e.sort_key)
        verdicts = [e.quality.format_verdict for e in entries]
        assert verdicts == [
            "ICH_E6R3",
            "PARTIAL_ICH",
            "NON_ICH",
            "",
        ]

    def test_confidence_tiebreak(self) -> None:
        """Higher confidence sorts first within same tier."""
        high = _make_entry("PARTIAL_ICH", 0.85, "NCT001")
        low = _make_entry("PARTIAL_ICH", 0.55, "NCT002")

        entries = [low, high]
        entries.sort(key=lambda e: e.sort_key)
        assert entries[0].registry_id == "NCT001"
        assert entries[1].registry_id == "NCT002"

    def test_registry_id_tiebreak(self) -> None:
        """Alphabetical ID breaks ties on same verdict+confidence."""
        a = _make_entry("ICH_E6R3", 0.9, "NCT001")
        b = _make_entry("ICH_E6R3", 0.9, "NCT002")

        entries = [b, a]
        entries.sort(key=lambda e: e.sort_key)
        assert entries[0].registry_id == "NCT001"


# ---------------------------------------------------------------------------
# ProtocolEntry.display_label
# ---------------------------------------------------------------------------

class TestDisplayLabel:
    """Tests for ProtocolEntry.display_label."""

    def test_short_title_no_truncation(self) -> None:
        entry = _make_entry()
        entry.title = "Short Title"
        assert entry.display_label == "Short Title (NCT001)"

    def test_long_title_truncated(self) -> None:
        entry = _make_entry()
        entry.title = "A" * 60
        label = entry.display_label
        assert label.endswith("... (NCT001)")
        assert len(label) < 70


# ---------------------------------------------------------------------------
# AREA_DISPLAY_ORDER
# ---------------------------------------------------------------------------

class TestAreaDisplayOrder:
    """Tests for AREA_DISPLAY_ORDER constant."""

    def test_has_five_entries(self) -> None:
        assert len(AREA_DISPLAY_ORDER) == 5

    def test_covers_all_areas(self) -> None:
        assert set(AREA_DISPLAY_ORDER) == set(TherapeuticArea)

    def test_oncology_first(self) -> None:
        assert AREA_DISPLAY_ORDER[0] == TherapeuticArea.ONCOLOGY

    def test_other_last(self) -> None:
        assert AREA_DISPLAY_ORDER[-1] == TherapeuticArea.OTHER


# ---------------------------------------------------------------------------
# load_protocol_catalog
# ---------------------------------------------------------------------------

def _write_metadata(
    metadata_dir: Path,
    registry_id: str,
    condition: str,
    title: str = "Test Protocol",
    quality_scores: dict | None = None,
    therapeutic_area: str = "",
) -> None:
    """Write a minimal metadata JSON file."""
    meta: dict = {
        "source": "ClinicalTrials.gov",
        "registry_id": registry_id,
        "version": "1.0",
        "amendment_number": "1.0",
        "title": title,
        "condition": condition,
    }
    if quality_scores:
        meta["quality_scores"] = quality_scores
    if therapeutic_area:
        meta["therapeutic_area"] = therapeutic_area
    path = metadata_dir / f"{registry_id}_1.0.json"
    path.write_text(json.dumps(meta), encoding="utf-8")


def _write_pdf(ct_dir: Path, registry_id: str) -> None:
    """Write a stub PDF file."""
    (ct_dir / f"{registry_id}_1.0.pdf").write_bytes(b"%PDF-fake")


class TestLoadProtocolCatalog:
    """Integration tests for load_protocol_catalog()."""

    def test_groups_by_therapeutic_area(self, tmp_path: Path) -> None:
        """Protocols are grouped into correct area buckets."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        _write_metadata(meta_dir, "NCT001", "Breast Cancer")
        _write_pdf(ct_dir, "NCT001")
        _write_metadata(meta_dir, "NCT002", "Heart Failure")
        _write_pdf(ct_dir, "NCT002")

        catalog = load_protocol_catalog(tmp_path)

        assert TherapeuticArea.ONCOLOGY in catalog
        assert TherapeuticArea.CARDIOVASCULAR in catalog
        assert len(catalog[TherapeuticArea.ONCOLOGY]) == 1
        assert len(catalog[TherapeuticArea.CARDIOVASCULAR]) == 1

    def test_sorted_by_quality_within_group(
        self, tmp_path: Path
    ) -> None:
        """ICH_E6R3 sorts before PARTIAL_ICH."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        _write_metadata(
            meta_dir,
            "NCT001",
            "Breast Cancer",
            quality_scores={
                "format_verdict": "PARTIAL_ICH",
                "format_confidence": 0.6,
                "section_count": 5,
            },
        )
        _write_pdf(ct_dir, "NCT001")
        _write_metadata(
            meta_dir,
            "NCT002",
            "Colon Cancer",
            quality_scores={
                "format_verdict": "ICH_E6R3",
                "format_confidence": 0.9,
                "section_count": 11,
            },
        )
        _write_pdf(ct_dir, "NCT002")

        catalog = load_protocol_catalog(tmp_path)
        onc = catalog[TherapeuticArea.ONCOLOGY]
        assert onc[0].registry_id == "NCT002"  # ICH_E6R3 first
        assert onc[1].registry_id == "NCT001"

    def test_unscored_protocols_sort_last(
        self, tmp_path: Path
    ) -> None:
        """Protocols without quality_scores sort after scored."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        _write_metadata(meta_dir, "NCT001", "Breast Cancer")
        _write_pdf(ct_dir, "NCT001")
        _write_metadata(
            meta_dir,
            "NCT002",
            "Colon Cancer",
            quality_scores={
                "format_verdict": "NON_ICH",
                "format_confidence": 0.2,
                "section_count": 1,
            },
        )
        _write_pdf(ct_dir, "NCT002")

        catalog = load_protocol_catalog(tmp_path)
        onc = catalog[TherapeuticArea.ONCOLOGY]
        assert onc[0].registry_id == "NCT002"  # scored first
        assert onc[1].registry_id == "NCT001"  # unscored last

    def test_empty_metadata_dir_returns_empty(
        self, tmp_path: Path
    ) -> None:
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        catalog = load_protocol_catalog(tmp_path)
        assert catalog == {}

    def test_no_metadata_dir_returns_empty(
        self, tmp_path: Path
    ) -> None:
        catalog = load_protocol_catalog(tmp_path)
        assert catalog == {}

    def test_metadata_without_pdf_skipped(
        self, tmp_path: Path
    ) -> None:
        """Metadata JSON with no matching PDF is silently skipped."""
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        _write_metadata(meta_dir, "NCT001", "Breast Cancer")
        # No PDF written

        catalog = load_protocol_catalog(tmp_path)
        assert catalog == {}

    def test_quality_scores_loaded(self, tmp_path: Path) -> None:
        """quality_scores dict populates QualityScores fields."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        _write_metadata(
            meta_dir,
            "NCT001",
            "Breast Cancer",
            quality_scores={
                "format_verdict": "ICH_E6R3",
                "format_confidence": 0.92,
                "section_count": 11,
            },
        )
        _write_pdf(ct_dir, "NCT001")

        catalog = load_protocol_catalog(tmp_path)
        entry = catalog[TherapeuticArea.ONCOLOGY][0]
        assert entry.quality.format_verdict == "ICH_E6R3"
        assert entry.quality.format_confidence == 0.92
        assert entry.quality.section_count == 11

    def test_preclassified_area_used(self, tmp_path: Path) -> None:
        """Pre-classified therapeutic_area is used from metadata."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        _write_metadata(
            meta_dir,
            "NCT001",
            "Some Rare Condition",
            therapeutic_area="Oncology",
        )
        _write_pdf(ct_dir, "NCT001")

        catalog = load_protocol_catalog(tmp_path)
        # Should use pre-classified area, not classify from condition
        assert TherapeuticArea.ONCOLOGY in catalog

    def test_title_used_as_display(self, tmp_path: Path) -> None:
        """Protocol title appears in display_label."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        _write_metadata(
            meta_dir,
            "NCT001",
            "Breast Cancer",
            title="A Phase II Study of Drug X",
        )
        _write_pdf(ct_dir, "NCT001")

        catalog = load_protocol_catalog(tmp_path)
        entry = catalog[TherapeuticArea.ONCOLOGY][0]
        assert "A Phase II Study of Drug X" in entry.display_label
        assert "(NCT001)" in entry.display_label

    def test_missing_title_uses_registry_id(
        self, tmp_path: Path
    ) -> None:
        """When title is empty, registry_id is used."""
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        _write_metadata(
            meta_dir, "NCT001", "Breast Cancer", title=""
        )
        _write_pdf(ct_dir, "NCT001")

        catalog = load_protocol_catalog(tmp_path)
        entry = catalog[TherapeuticArea.ONCOLOGY][0]
        assert entry.title == "NCT001"

    def test_eu_ctr_source_detected(self, tmp_path: Path) -> None:
        """PDFs in eu-ctr/ are detected with correct source."""
        eu_dir = tmp_path / "eu-ctr"
        eu_dir.mkdir()
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        meta: dict = {
            "source": "EU-CTR",
            "registry_id": "2024-001234-22",
            "version": "1.0",
            "amendment_number": "1.0",
            "title": "EU Trial",
            "condition": "Breast Cancer",
        }
        path = meta_dir / "2024-001234-22_1.0.json"
        path.write_text(json.dumps(meta), encoding="utf-8")
        (eu_dir / "2024-001234-22_1.0.pdf").write_bytes(
            b"%PDF-fake"
        )

        catalog = load_protocol_catalog(tmp_path)
        entry = catalog[TherapeuticArea.ONCOLOGY][0]
        assert entry.registry_source == "eu-ctr"
