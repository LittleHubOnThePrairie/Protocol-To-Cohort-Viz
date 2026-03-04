"""Review Queue viewer component (PTCV-84).

Displays pending review items from the ReviewQueue database,
grouped by type (SoA synonym mappings vs ICH sections).
Supports approve, reject, and edit actions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from ptcv.ich_parser.models import ReviewQueueEntry
from ptcv.ich_parser.review_queue import ReviewQueue

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("C:/Dev/PTCV/data/sqlite/review_queue.db")


def _get_review_queue(db_path: Path = _DEFAULT_DB) -> ReviewQueue:
    """Return an initialised ReviewQueue instance."""
    rq = ReviewQueue(db_path=db_path)
    rq.initialise()
    return rq


def get_pending_count(
    registry_id: str | None = None,
    db_path: Path = _DEFAULT_DB,
) -> int:
    """Get count of pending review items for badge display."""
    rq = _get_review_queue(db_path)
    return rq.pending_count(registry_id=registry_id)


def render_review_queue(
    registry_id: str | None = None,
    db_path: Path = _DEFAULT_DB,
) -> None:
    """Render the review queue viewer in Streamlit.

    Shows two sections: synonym mappings and ICH classifications.
    Each item has approve/reject/edit action buttons.

    Args:
        registry_id: Optional filter by trial. None shows all.
        db_path: Path to the review queue database.
    """
    rq = _get_review_queue(db_path)
    items = rq.pending(registry_id=registry_id)

    if not items:
        st.success("All items reviewed — no pending reviews")
        return

    # Split by type
    synonyms = [i for i in items if i.section_code == "soa_synonym"]
    ich_sections = [i for i in items if i.section_code != "soa_synonym"]

    # Synonym mappings
    if synonyms:
        st.subheader(f"Synonym Mappings ({len(synonyms)})")
        for entry in synonyms:
            _render_synonym_card(rq, entry)

    # ICH section classifications
    if ich_sections:
        st.subheader(f"ICH Classifications ({len(ich_sections)})")
        for entry in ich_sections:
            _render_ich_card(rq, entry)


def _render_synonym_card(
    rq: ReviewQueue,
    entry: ReviewQueueEntry,
) -> None:
    """Render one synonym mapping review card."""
    assert entry.id is not None
    try:
        data = json.loads(entry.content_json)
    except (json.JSONDecodeError, TypeError):
        data = {}

    original = data.get("original_text", "?")
    canonical = data.get("canonical_label", "?")
    method = data.get("method", "?")
    conf = entry.confidence_score

    with st.container(border=True):
        col_info, col_actions = st.columns([3, 1])
        with col_info:
            st.markdown(
                f"**\"{original}\"** → **{canonical}**"
            )
            st.caption(
                f"Method: {method} · "
                f"Confidence: {conf:.0%} · "
                f"Trial: {entry.registry_id}"
            )
        with col_actions:
            key_prefix = f"syn_{entry.id}"
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "Approve", key=f"{key_prefix}_approve",
                    type="primary",
                ):
                    rq.resolve(entry.id, "approved")
                    st.rerun()
            with c2:
                if st.button(
                    "Reject", key=f"{key_prefix}_reject",
                ):
                    rq.resolve(entry.id, "rejected")
                    st.rerun()

        # Edit row (expandable)
        with st.expander("Edit mapping", expanded=False):
            new_label = st.text_input(
                "Corrected label",
                value=canonical,
                key=f"{key_prefix}_edit",
            )
            notes = st.text_input(
                "Notes",
                key=f"{key_prefix}_notes",
                placeholder="Optional reviewer notes",
            )
            if st.button("Save", key=f"{key_prefix}_save"):
                corrected = json.dumps({
                    "original_text": original,
                    "canonical_label": new_label,
                    "method": "human_review",
                    "confidence": 1.0,
                })
                rq.resolve(
                    entry.id, "edited",
                    corrected_value=corrected,
                    reviewer_notes=notes or None,
                )
                st.rerun()


def _render_ich_card(
    rq: ReviewQueue,
    entry: ReviewQueueEntry,
) -> None:
    """Render one ICH section classification review card."""
    assert entry.id is not None
    try:
        data = json.loads(entry.content_json)
    except (json.JSONDecodeError, TypeError):
        data = {}

    excerpt = data.get("text_excerpt", "")[:300]
    concepts = data.get("key_concepts", [])
    conf = entry.confidence_score

    with st.container(border=True):
        col_info, col_actions = st.columns([3, 1])
        with col_info:
            st.markdown(
                f"**{entry.section_code}** · "
                f"Confidence: {conf:.0%}"
            )
            if excerpt:
                st.caption(excerpt)
            if concepts:
                st.caption(
                    f"Key concepts: {', '.join(str(c) for c in concepts[:5])}"
                )
        with col_actions:
            key_prefix = f"ich_{entry.id}"
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "Confirm",
                    key=f"{key_prefix}_confirm",
                    type="primary",
                ):
                    rq.resolve(entry.id, "approved")
                    st.rerun()
            with c2:
                if st.button(
                    "Flag",
                    key=f"{key_prefix}_flag",
                ):
                    rq.resolve(
                        entry.id, "rejected",
                        reviewer_notes="Flagged for re-review",
                    )
                    st.rerun()
