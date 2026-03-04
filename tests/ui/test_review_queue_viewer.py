"""Unit tests for review_queue_viewer (PTCV-84).

Tests the helper functions and rendering logic using
a temporary SQLite database.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptcv.ich_parser.models import ReviewQueueEntry
from ptcv.ich_parser.review_queue import ReviewQueue
from ptcv.ui.components.review_queue_viewer import (
    get_pending_count,
    render_review_queue,
)


def _make_entry(
    section_code: str = "soa_synonym",
    confidence: float = 0.65,
    registry_id: str = "EUCT-001",
    content_json: str | None = None,
) -> ReviewQueueEntry:
    if content_json is None:
        if section_code == "soa_synonym":
            content_json = json.dumps({
                "original_text": "BP measurement",
                "canonical_label": "Blood Pressure",
                "method": "embedding",
            })
        else:
            content_json = json.dumps({
                "text_excerpt": "The objectives of this trial...",
                "key_concepts": ["efficacy", "safety"],
            })
    return ReviewQueueEntry(
        run_id="run-1",
        registry_id=registry_id,
        section_code=section_code,
        confidence_score=confidence,
        content_json=content_json,
        queue_timestamp_utc="2026-01-01T00:00:00Z",
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "rq.db"


@pytest.fixture()
def rq(db_path: Path) -> ReviewQueue:
    queue = ReviewQueue(db_path=db_path)
    queue.initialise()
    return queue


class TestGetPendingCount:
    """Tests for get_pending_count()."""

    def test_zero_when_empty(self, db_path: Path) -> None:
        count = get_pending_count(db_path=db_path)
        assert count == 0

    def test_counts_pending_items(
        self, rq: ReviewQueue, db_path: Path,
    ) -> None:
        rq.enqueue(_make_entry("soa_synonym"))
        rq.enqueue(_make_entry("B.3"))
        assert get_pending_count(db_path=db_path) == 2

    def test_excludes_resolved(
        self, rq: ReviewQueue, db_path: Path,
    ) -> None:
        e = rq.enqueue(_make_entry("soa_synonym"))
        rq.enqueue(_make_entry("B.3"))
        rq.resolve(e.id, "approved")
        assert get_pending_count(db_path=db_path) == 1

    def test_filter_by_registry(
        self, rq: ReviewQueue, db_path: Path,
    ) -> None:
        rq.enqueue(_make_entry("soa_synonym", registry_id="EUCT-001"))
        rq.enqueue(_make_entry("B.3", registry_id="NCT-999"))
        assert get_pending_count(
            registry_id="EUCT-001", db_path=db_path,
        ) == 1


class TestRenderReviewQueue:
    """Tests for render_review_queue() with mocked Streamlit."""

    @patch("ptcv.ui.components.review_queue_viewer.st")
    def test_shows_success_when_empty(
        self, mock_st: MagicMock, db_path: Path,
    ) -> None:
        render_review_queue(db_path=db_path)
        mock_st.success.assert_called_once()

    @patch("ptcv.ui.components.review_queue_viewer.st")
    def test_renders_synonym_section(
        self, mock_st: MagicMock,
        rq: ReviewQueue, db_path: Path,
    ) -> None:
        rq.enqueue(_make_entry("soa_synonym"))
        # Mock container and columns to prevent AttributeError
        mock_st.container.return_value.__enter__ = MagicMock(
            return_value=MagicMock(),
        )
        mock_st.container.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [mock_col, mock_col]
        mock_st.expander.return_value.__enter__ = MagicMock(
            return_value=MagicMock(),
        )
        mock_st.expander.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        mock_st.button.return_value = False
        mock_st.text_input.return_value = ""

        render_review_queue(db_path=db_path)
        # Should render a subheader for synonym mappings
        calls = [
            str(c) for c in mock_st.subheader.call_args_list
        ]
        assert any("Synonym" in c for c in calls)

    @patch("ptcv.ui.components.review_queue_viewer.st")
    def test_renders_ich_section(
        self, mock_st: MagicMock,
        rq: ReviewQueue, db_path: Path,
    ) -> None:
        rq.enqueue(_make_entry("B.3"))
        mock_st.container.return_value.__enter__ = MagicMock(
            return_value=MagicMock(),
        )
        mock_st.container.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [mock_col, mock_col]
        mock_st.button.return_value = False

        render_review_queue(db_path=db_path)
        calls = [
            str(c) for c in mock_st.subheader.call_args_list
        ]
        assert any("ICH" in c for c in calls)

    @patch("ptcv.ui.components.review_queue_viewer.st")
    def test_renders_both_sections(
        self, mock_st: MagicMock,
        rq: ReviewQueue, db_path: Path,
    ) -> None:
        rq.enqueue(_make_entry("soa_synonym"))
        rq.enqueue(_make_entry("B.7"))
        mock_st.container.return_value.__enter__ = MagicMock(
            return_value=MagicMock(),
        )
        mock_st.container.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [mock_col, mock_col]
        mock_st.expander.return_value.__enter__ = MagicMock(
            return_value=MagicMock(),
        )
        mock_st.expander.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        mock_st.button.return_value = False
        mock_st.text_input.return_value = ""

        render_review_queue(db_path=db_path)
        assert mock_st.subheader.call_count == 2

    @patch("ptcv.ui.components.review_queue_viewer.st")
    def test_filter_by_registry(
        self, mock_st: MagicMock,
        rq: ReviewQueue, db_path: Path,
    ) -> None:
        rq.enqueue(_make_entry("B.3", registry_id="EUCT-001"))
        rq.enqueue(_make_entry("B.5", registry_id="NCT-999"))
        mock_st.container.return_value.__enter__ = MagicMock(
            return_value=MagicMock(),
        )
        mock_st.container.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [mock_col, mock_col]
        mock_st.button.return_value = False

        render_review_queue(
            registry_id="EUCT-001", db_path=db_path,
        )
        # Only 1 ICH section for EUCT-001
        calls = [
            str(c) for c in mock_st.subheader.call_args_list
        ]
        assert any("1" in c for c in calls)
