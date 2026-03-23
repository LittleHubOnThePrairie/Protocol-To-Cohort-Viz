"""Tests for Vision API selective verification (PTCV-217).

Tests correction parsing, skip logic, confidence scoring,
and table merging. API calls are mocked.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from ptcv.soa_extractor.vision_verifier import (
    CellCorrection,
    VerificationResult,
    VisionVerifier,
)
from ptcv.soa_extractor.models import RawSoaTable


def _make_table(
    activities: list[tuple[str, list[bool]]],
    visit_headers: list[str] | None = None,
) -> RawSoaTable:
    if visit_headers is None:
        n = max((len(f) for _, f in activities), default=3)
        visit_headers = [f"V{i}" for i in range(1, n + 1)]
    return RawSoaTable(
        visit_headers=visit_headers,
        day_headers=[],
        activities=activities,
        section_code="B.4",
    )


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_no_corrections(self):
        result = VerificationResult(verified=True)
        assert result.correction_count == 0
        assert not result.has_corrections

    def test_has_corrections(self):
        result = VerificationResult(
            verified=True,
            corrections=[
                CellCorrection("Exam", 0, "V1", False, True),
            ],
            added_activities=["Labs"],
        )
        assert result.correction_count == 2
        assert result.has_corrections

    def test_skip_reason(self):
        result = VerificationResult(
            verified=False, skip_reason="High coverage",
        )
        assert not result.verified
        assert result.skip_reason == "High coverage"


class TestVisionVerifierSkipLogic:
    """Tests for skip conditions."""

    def test_skip_when_high_coverage(self):
        """Test verification skipped when coverage >= threshold."""
        verifier = VisionVerifier(min_coverage_for_skip=0.95)
        table = _make_table([("Exam", [True])])

        result = verifier.verify(table, [b"fake_png"], coverage_ratio=0.97)

        assert not result.verified
        assert "threshold" in result.skip_reason
        assert result.corrected_table is table

    def test_skip_when_no_images(self):
        """Test verification skipped when no page images."""
        verifier = VisionVerifier()
        table = _make_table([("Exam", [True])])

        result = verifier.verify(table, [], coverage_ratio=0.5)

        assert not result.verified
        assert "No page images" in result.skip_reason

    def test_proceeds_when_low_coverage(self):
        """Test verification proceeds when coverage is below threshold."""
        verifier = VisionVerifier(min_coverage_for_skip=0.95)
        table = _make_table([("Exam", [True])], ["V1"])

        # Mock the API call
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text=json.dumps({
            "missing_activities": [],
            "extra_activities": [],
            "cell_corrections": [],
        }))]
        mock_response.usage = MagicMock(input_tokens=1000, output_tokens=200)

        with patch.object(verifier, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get.return_value = mock_client

            result = verifier.verify(
                table, [b"fake_png"], coverage_ratio=0.60,
            )

        assert result.verified
        assert result.token_cost == 1200


class TestApplyCorrections:
    """Tests for _apply_corrections method."""

    def test_cell_correction_applied(self):
        """Test cell value correction is applied."""
        verifier = VisionVerifier()
        table = _make_table(
            [("Exam", [True, False, True])],
            ["V1", "V2", "V3"],
        )

        response = json.dumps({
            "missing_activities": [],
            "extra_activities": [],
            "cell_corrections": [
                {
                    "activity_name": "Exam",
                    "visit_index": 1,
                    "old_value": False,
                    "new_value": True,
                },
            ],
        })

        result = verifier._apply_corrections(table, response, 500)

        assert result.verified
        assert len(result.corrections) == 1
        assert result.corrections[0].visit_index == 1
        assert result.corrections[0].new_value is True
        # Check corrected table
        exam = [a for a in result.corrected_table.activities if a[0] == "Exam"][0]
        assert exam[1][1] is True  # Was False, now True

    def test_missing_activity_added(self):
        """Test missing activity is added to corrected table."""
        verifier = VisionVerifier()
        table = _make_table(
            [("Exam", [True, True])],
            ["V1", "V2"],
        )

        response = json.dumps({
            "missing_activities": [
                {"name": "Labs", "visits_scheduled": [False, True]},
            ],
            "extra_activities": [],
            "cell_corrections": [],
        })

        result = verifier._apply_corrections(table, response, 400)

        assert "Labs" in result.added_activities
        names = [n for n, _ in result.corrected_table.activities]
        assert "Labs" in names
        assert result.confidence_map["Labs"] == 0.85  # Vision-only

    def test_extra_activity_flagged(self):
        """Test extra activity is flagged but not removed."""
        verifier = VisionVerifier()
        table = _make_table(
            [
                ("Exam", [True]),
                ("Phantom", [True]),
            ],
            ["V1"],
        )

        response = json.dumps({
            "missing_activities": [],
            "extra_activities": ["Phantom"],
            "cell_corrections": [],
        })

        result = verifier._apply_corrections(table, response, 300)

        assert "Phantom" in result.removed_activities
        # Not actually removed — flagged for review
        names = [n for n, _ in result.corrected_table.activities]
        assert "Phantom" in names
        assert result.confidence_map["Phantom"] == 0.50

    def test_no_corrections_high_confidence(self):
        """Test all activities get 0.95 confidence when no corrections."""
        verifier = VisionVerifier()
        table = _make_table(
            [("Exam", [True]), ("Labs", [False])],
            ["V1"],
        )

        response = json.dumps({
            "missing_activities": [],
            "extra_activities": [],
            "cell_corrections": [],
        })

        result = verifier._apply_corrections(table, response, 200)

        assert result.confidence_map["Exam"] == 0.95
        assert result.confidence_map["Labs"] == 0.95
        assert not result.has_corrections

    def test_invalid_json_handled(self):
        """Test graceful handling of unparseable Vision response."""
        verifier = VisionVerifier()
        table = _make_table([("Exam", [True])], ["V1"])

        result = verifier._apply_corrections(
            table, "not valid json", 100,
        )

        assert result.verified
        assert result.corrected_table is table
        assert "parse error" in result.skip_reason

    def test_construction_method_set(self):
        """Test corrected table has construction_method='llm_vision'."""
        verifier = VisionVerifier()
        table = _make_table([("Exam", [True])], ["V1"])

        response = json.dumps({
            "missing_activities": [
                {"name": "ECG", "visits_scheduled": [True]},
            ],
            "extra_activities": [],
            "cell_corrections": [],
        })

        result = verifier._apply_corrections(table, response, 200)
        assert result.corrected_table.construction_method == "llm_vision"

    def test_missing_activity_padded_to_header_count(self):
        """Test missing activity flags padded to match header count."""
        verifier = VisionVerifier()
        table = _make_table(
            [("Exam", [True, False, True])],
            ["V1", "V2", "V3"],
        )

        response = json.dumps({
            "missing_activities": [
                {"name": "Labs", "visits_scheduled": [True]},  # Only 1 flag
            ],
            "extra_activities": [],
            "cell_corrections": [],
        })

        result = verifier._apply_corrections(table, response, 200)
        labs = [a for a in result.corrected_table.activities if a[0] == "Labs"][0]
        assert len(labs[1]) == 3  # Padded to 3

    def test_out_of_bounds_correction_ignored(self):
        """Test corrections with invalid visit_index are ignored."""
        verifier = VisionVerifier()
        table = _make_table([("Exam", [True])], ["V1"])

        response = json.dumps({
            "missing_activities": [],
            "extra_activities": [],
            "cell_corrections": [
                {
                    "activity_name": "Exam",
                    "visit_index": 99,
                    "old_value": True,
                    "new_value": False,
                },
            ],
        })

        result = verifier._apply_corrections(table, response, 100)
        assert len(result.corrections) == 0
