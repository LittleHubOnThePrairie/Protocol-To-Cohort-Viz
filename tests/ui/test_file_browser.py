"""Unit tests for the file browser component (PTCV-33).

Tests the filesystem scanning logic in isolation (no Streamlit
runtime required).
"""

from pathlib import Path

from ptcv.ui.components.file_browser import _scan_pdfs


class TestScanPdfs:
    """Tests for _scan_pdfs()."""

    def test_finds_pdfs_in_clinicaltrials(self, tmp_path: Path) -> None:
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        (ct_dir / "NCT001_00.pdf").write_bytes(b"%PDF-fake")
        (ct_dir / "NCT002_00.pdf").write_bytes(b"%PDF-fake")

        result = _scan_pdfs(tmp_path)

        assert "clinicaltrials" in result
        assert result["clinicaltrials"] == ["NCT001_00.pdf", "NCT002_00.pdf"]

    def test_finds_pdfs_in_eu_ctr(self, tmp_path: Path) -> None:
        eu_dir = tmp_path / "eu-ctr"
        eu_dir.mkdir()
        (eu_dir / "2024-001234-22_00.pdf").write_bytes(b"%PDF-fake")

        result = _scan_pdfs(tmp_path)

        assert "eu-ctr" in result
        assert result["eu-ctr"] == ["2024-001234-22_00.pdf"]

    def test_groups_by_registry_source(self, tmp_path: Path) -> None:
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        (ct_dir / "NCT001_00.pdf").write_bytes(b"%PDF-fake")

        eu_dir = tmp_path / "eu-ctr"
        eu_dir.mkdir()
        (eu_dir / "EU001_00.pdf").write_bytes(b"%PDF-fake")

        result = _scan_pdfs(tmp_path)

        assert len(result) == 2
        assert "clinicaltrials" in result
        assert "eu-ctr" in result

    def test_ignores_non_pdf_files(self, tmp_path: Path) -> None:
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        (ct_dir / "NCT001_00.pdf").write_bytes(b"%PDF-fake")
        (ct_dir / "NCT001_00.txt").write_text("not a pdf")
        (ct_dir / "metadata.json").write_text("{}")

        result = _scan_pdfs(tmp_path)

        assert result["clinicaltrials"] == ["NCT001_00.pdf"]

    def test_empty_directory_omitted(self, tmp_path: Path) -> None:
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        # No PDFs inside

        result = _scan_pdfs(tmp_path)

        assert "clinicaltrials" not in result

    def test_missing_directory_skipped(self, tmp_path: Path) -> None:
        # Neither clinicaltrials/ nor eu-ctr/ exist
        result = _scan_pdfs(tmp_path)

        assert result == {}

    def test_filenames_sorted_alphabetically(self, tmp_path: Path) -> None:
        ct_dir = tmp_path / "clinicaltrials"
        ct_dir.mkdir()
        (ct_dir / "NCT003_00.pdf").write_bytes(b"%PDF")
        (ct_dir / "NCT001_00.pdf").write_bytes(b"%PDF")
        (ct_dir / "NCT002_00.pdf").write_bytes(b"%PDF")

        result = _scan_pdfs(tmp_path)

        assert result["clinicaltrials"] == [
            "NCT001_00.pdf",
            "NCT002_00.pdf",
            "NCT003_00.pdf",
        ]
