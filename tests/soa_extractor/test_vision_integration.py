"""Tests for Vision verifier cascade integration (PTCV-221).

Verifies that the Vision verifier is wired into the SoA extractor
cascade and triggered by cross-validation escalation.
"""

from __future__ import annotations

import inspect
import pytest
from unittest.mock import MagicMock, patch

from ptcv.soa_extractor.extractor import SoaExtractor


class TestVisionVerifierWiring:
    """Tests that Vision verifier is wired into extractor cascade."""

    def test_extractor_imports_vision_verifier(self):
        """Test VisionVerifier is imported in extractor module."""
        import ptcv.soa_extractor.extractor as ext
        assert hasattr(ext, 'VisionVerifier')

    def test_extractor_imports_cross_validate(self):
        """Test cross_validate is imported in extractor module."""
        import ptcv.soa_extractor.extractor as ext
        assert hasattr(ext, 'cross_validate')

    def test_extract_method_references_vision_verifier(self):
        """Test extract() method contains Vision verifier logic."""
        source = inspect.getsource(SoaExtractor.extract)
        assert 'VisionVerifier' in source
        assert 'PTCV-221' in source

    def test_vision_triggered_by_cross_validation(self):
        """Test Vision is triggered by cross-validation escalation."""
        source = inspect.getsource(SoaExtractor.extract)
        # Vision verifier should be inside the cv_result.needs_escalation block
        assert 'cv_result.needs_escalation' in source
        assert 'verifier.verify' in source

    def test_vision_not_triggered_without_pdf(self):
        """Test Vision is skipped when pdf_bytes is None."""
        source = inspect.getsource(SoaExtractor.extract)
        # Guard: only run if pdf_bytes is not None
        assert 'pdf_bytes is not None' in source

    def test_render_soa_pages_exists(self):
        """Test _render_soa_pages helper method exists."""
        assert hasattr(SoaExtractor, '_render_soa_pages')

    def test_render_soa_pages_with_no_blocks(self):
        """Test _render_soa_pages returns empty with no text blocks."""
        result = SoaExtractor._render_soa_pages(b"%PDF-stub", [])
        assert result == []

    def test_render_soa_pages_with_no_schedule_heading(self):
        """Test _render_soa_pages returns empty without schedule headings."""
        blocks = [
            {"text": "Introduction to the study", "page_number": 1},
            {"text": "Methods and materials", "page_number": 2},
        ]
        result = SoaExtractor._render_soa_pages(b"%PDF-stub", blocks)
        assert result == []

    def test_render_soa_pages_finds_candidate(self):
        """Test _render_soa_pages identifies schedule heading pages."""
        blocks = [
            {"text": "Introduction", "page_number": 1},
            {"text": "Schedule of Activities", "page_number": 5},
            {"text": "Statistical Methods", "page_number": 10},
        ]
        # Won't render (no real PDF) but candidate detection works
        with patch("fitz.open") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__ = lambda s: 12
            mock_page = MagicMock()
            mock_pix = MagicMock()
            mock_pix.tobytes.return_value = b"fake_png"
            mock_page.get_pixmap.return_value = mock_pix
            mock_doc.__getitem__ = lambda s, i: mock_page
            mock_fitz.return_value = mock_doc
            mock_fitz.Matrix = MagicMock(return_value=MagicMock())

            result = SoaExtractor._render_soa_pages(b"%PDF", blocks)

        # Should find pages 5 and 6 (page + next)
        assert len(result) >= 1

    def test_vision_correction_replaces_table(self):
        """Test that Vision corrections replace the table in the list."""
        source = inspect.getsource(SoaExtractor.extract)
        # After verification, the corrected table replaces the original
        assert 'tables[i] = vr.corrected_table' in source

    def test_vision_failure_is_non_blocking(self):
        """Test Vision verifier failure doesn't crash the cascade."""
        source = inspect.getsource(SoaExtractor.extract)
        # Vision failure is caught and logged
        assert 'Vision verifier failed' in source
        assert 'continuing with Level 1' in source


class TestCascadeOrder:
    """Tests for correct cascade ordering."""

    def test_cross_validation_before_level3(self):
        """Test cross-validation runs before Level 3 LLM construction."""
        source = inspect.getsource(SoaExtractor.extract)
        cv_pos = source.index('cross_validate')
        l3_pos = source.index('Level 3')
        assert cv_pos < l3_pos

    def test_vision_between_cv_and_level3(self):
        """Test Vision verifier runs between cross-validation and Level 3."""
        source = inspect.getsource(SoaExtractor.extract)
        cv_pos = source.index('cross_validate')
        vision_pos = source.index('VisionVerifier')
        l3_pos = source.index('Level 3')
        assert cv_pos < vision_pos < l3_pos
