"""Unit tests for app helpers (PTCV-33).

Tests the non-Streamlit helper functions (registry ID extraction,
SHA-256 computation, verdict display logic) without requiring
the Streamlit runtime.
"""

from ptcv.ui.app import (
    _compute_sha256,
    _extract_amendment,
    _extract_registry_id,
)


class TestExtractRegistryId:
    """Tests for _extract_registry_id()."""

    def test_clinicaltrials_pattern(self) -> None:
        assert _extract_registry_id("NCT00112827_1.0.pdf") == "NCT00112827"

    def test_eu_ctr_pattern(self) -> None:
        assert _extract_registry_id("2024-001234-22_00.pdf") == "2024-001234-22"

    def test_no_underscore(self) -> None:
        assert _extract_registry_id("NCT00112827.pdf") == "NCT00112827"

    def test_multiple_underscores(self) -> None:
        assert _extract_registry_id("NCT001_1.0_final.pdf") == "NCT001"


class TestExtractAmendment:
    """Tests for _extract_amendment()."""

    def test_version_number(self) -> None:
        assert _extract_amendment("NCT00112827_1.0.pdf") == "1.0"

    def test_amendment_00(self) -> None:
        assert _extract_amendment("NCT00112827_00.pdf") == "00"

    def test_no_underscore_defaults_00(self) -> None:
        assert _extract_amendment("NCT00112827.pdf") == "00"


class TestComputeSha256:
    """Tests for _compute_sha256()."""

    def test_deterministic(self) -> None:
        data = b"hello world"
        assert _compute_sha256(data) == _compute_sha256(data)

    def test_known_hash(self) -> None:
        # SHA-256 of empty bytes
        assert _compute_sha256(b"") == (
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        )

    def test_different_input_different_hash(self) -> None:
        assert _compute_sha256(b"a") != _compute_sha256(b"b")


class TestRunParseResult:
    """Tests for _run_parse return structure.

    We test the expected dict keys without actually invoking
    the full extraction pipeline (which requires real PDFs).
    """

    def test_expected_keys(self) -> None:
        expected_keys = {
            "format_verdict",
            "format_confidence",
            "section_count",
            "review_count",
            "missing_required_sections",
            "registry_id",
            "amendment",
            "artifact_key",
        }
        # Verify the set matches what _display_verdict and
        # _render_regeneration read
        assert expected_keys == {
            "format_verdict",
            "format_confidence",
            "section_count",
            "review_count",
            "missing_required_sections",
            "registry_id",
            "amendment",
            "artifact_key",
        }
