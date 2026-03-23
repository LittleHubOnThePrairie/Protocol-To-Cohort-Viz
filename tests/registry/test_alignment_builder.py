"""Tests for PTCV-232: Ground-Truth Alignment Corpus Builder.

Tests verify fuzzy matching, chunk splitting, corpus building,
and Parquet persistence.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ptcv.registry.alignment_builder import (
    AlignmentBuilder,
    AlignmentRecord,
)


class TestAlignmentRecord:
    """Tests for AlignmentRecord dataclass."""

    def test_creation(self):
        """Test record creation with all fields."""
        rec = AlignmentRecord(
            nct_id="NCT01512251",
            section_code="B.3",
            section_name="Trial Objectives",
            registry_field="IdentificationModule",
            registry_text="A Phase 1/2 Study of BKM120",
            pdf_text_span="A Phase 1/2 Study of BKM120 plus vemurafenib",
            char_offset=150,
            span_length=45,
            quality_rating=0.9,
            similarity_score=0.92,
        )
        assert rec.nct_id == "NCT01512251"
        assert rec.section_code == "B.3"
        assert rec.quality_rating == 0.9
        assert rec.similarity_score == 0.92


class TestChunkSplitting:
    """Tests for AlignmentBuilder._split_into_chunks."""

    def test_short_text_single_chunk(self):
        """Short text returns as single chunk."""
        chunks = AlignmentBuilder._split_into_chunks("Short text here.")
        assert chunks == ["Short text here."]

    def test_long_text_splits_by_paragraph(self):
        """Long text split by double-newline paragraph boundaries."""
        text = ("First paragraph content. " * 20 + "\n\n" +
                "Second paragraph content. " * 20)
        chunks = AlignmentBuilder._split_into_chunks(text)
        assert len(chunks) >= 2

    def test_very_short_paragraphs_skipped(self):
        """Paragraphs shorter than 20 chars are skipped."""
        text = "Short\n\n" + "A much longer paragraph with content. " * 15
        chunks = AlignmentBuilder._split_into_chunks(text)
        for chunk in chunks:
            assert len(chunk) >= 20

    def test_empty_text(self):
        """Empty text returns single chunk."""
        chunks = AlignmentBuilder._split_into_chunks("")
        assert len(chunks) == 1


class TestFuzzyAlign:
    """Tests for AlignmentBuilder._fuzzy_align."""

    def test_exact_match(self):
        """Exact text match produces high similarity."""
        builder = AlignmentBuilder()
        pdf_text = (
            "Introduction\n"
            "A Randomized Double-Blind Study of Drug X\n"
            "This study evaluates..."
        )
        records = builder._fuzzy_align(
            registry_text="A Randomized Double-Blind Study of Drug X",
            pdf_text=pdf_text,
            nct_id="NCT12345678",
            section_code="B.3",
            section_name="Objectives",
            registry_field="IdentificationModule",
            quality_rating=0.9,
        )
        assert len(records) >= 1
        assert records[0].similarity_score >= 0.9
        assert records[0].section_code == "B.3"

    def test_fuzzy_match_with_formatting_diff(self):
        """Match succeeds despite formatting differences."""
        builder = AlignmentBuilder()
        # Registry has no comma, PDF has comma
        registry = "A Randomized Double-Blind Study"
        pdf_text = "Protocol: A Randomized, Double-Blind Study of X"

        records = builder._fuzzy_align(
            registry_text=registry,
            pdf_text=pdf_text,
            nct_id="NCT12345678",
            section_code="B.3",
            section_name="Objectives",
            registry_field="IdentificationModule",
            quality_rating=0.9,
        )
        assert len(records) >= 1
        assert records[0].similarity_score >= 0.80

    def test_no_match_below_threshold(self):
        """No records when text is completely different."""
        builder = AlignmentBuilder(match_threshold=80)
        records = builder._fuzzy_align(
            registry_text="Evaluate the efficacy of pembrolizumab",
            pdf_text="Weather forecast for next week shows rain",
            nct_id="NCT12345678",
            section_code="B.8",
            section_name="Efficacy",
            registry_field="OutcomesModule",
            quality_rating=0.9,
        )
        assert len(records) == 0

    def test_short_text_skipped(self):
        """Very short registry text (< 10 chars) is skipped."""
        builder = AlignmentBuilder()
        records = builder._fuzzy_align(
            registry_text="Hi",
            pdf_text="Hello world this is a long PDF text " * 10,
            nct_id="NCT12345678",
            section_code="B.1",
            section_name="General",
            registry_field="StatusModule",
            quality_rating=0.6,
        )
        assert len(records) == 0


class TestDiscoverLinkableProtocols:
    """Tests for _discover_linkable_protocols."""

    def test_finds_intersection(self, tmp_path):
        """Returns NCT IDs with both cache and PDF."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create cache files
        (cache_dir / "NCT11111111.json").write_text("{}")
        (cache_dir / "NCT22222222.json").write_text("{}")
        (cache_dir / "NCT33333333.json").write_text("{}")

        # Create PDF files (only 2 overlap)
        (pdf_dir / "NCT11111111_1.0.pdf").write_bytes(b"pdf")
        (pdf_dir / "NCT22222222_1.0.pdf").write_bytes(b"pdf")
        (pdf_dir / "NCT44444444_1.0.pdf").write_bytes(b"pdf")

        builder = AlignmentBuilder(
            registry_cache_dir=cache_dir,
            pdf_dir=pdf_dir,
        )
        linkable = builder._discover_linkable_protocols()

        assert linkable == ["NCT11111111", "NCT22222222"]


class TestBuildCorpus:
    """Tests for AlignmentBuilder.build_corpus."""

    @patch.object(AlignmentBuilder, "_align_protocol")
    def test_processes_all_linkable(self, mock_align):
        """Corpus built from all linkable protocols."""
        mock_align.return_value = [
            AlignmentRecord(
                nct_id="NCT12345678",
                section_code="B.3",
                section_name="Objectives",
                registry_field="IdentificationModule",
                registry_text="Title",
                pdf_text_span="Title in PDF",
                char_offset=100,
                span_length=12,
                quality_rating=0.9,
                similarity_score=0.95,
            ),
        ]

        builder = AlignmentBuilder()
        corpus = builder.build_corpus(
            nct_ids=["NCT12345678", "NCT87654321"]
        )

        assert len(corpus) == 2
        assert mock_align.call_count == 2

    @patch.object(AlignmentBuilder, "_align_protocol")
    def test_empty_corpus_on_no_matches(self, mock_align):
        """Empty corpus when no alignments found."""
        mock_align.return_value = []
        builder = AlignmentBuilder()
        corpus = builder.build_corpus(nct_ids=["NCT12345678"])
        assert corpus == []


class TestSaveParquet:
    """Tests for Parquet persistence."""

    def test_save_and_load(self, tmp_path):
        """Corpus round-trips through Parquet."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        corpus = [
            AlignmentRecord(
                nct_id="NCT12345678",
                section_code="B.3",
                section_name="Objectives",
                registry_field="IdentificationModule",
                registry_text="Study Title",
                pdf_text_span="Study Title in PDF",
                char_offset=50,
                span_length=18,
                quality_rating=0.9,
                similarity_score=0.95,
            ),
            AlignmentRecord(
                nct_id="NCT12345678",
                section_code="B.8",
                section_name="Efficacy",
                registry_field="OutcomesModule",
                registry_text="Primary endpoint: ORR",
                pdf_text_span="Overall response rate (ORR)",
                char_offset=2000,
                span_length=27,
                quality_rating=0.9,
                similarity_score=0.88,
            ),
        ]

        output_path = tmp_path / "alignments.parquet"
        builder = AlignmentBuilder()
        result_path = builder.save_parquet(corpus, output_path)

        assert result_path.exists()
        df = pd.read_parquet(result_path)
        assert len(df) == 2
        assert "nct_id" in df.columns
        assert "section_code" in df.columns
        assert "quality_rating" in df.columns
        assert "similarity_score" in df.columns
        assert df.iloc[0]["nct_id"] == "NCT12345678"
        assert df.iloc[1]["section_code"] == "B.8"
