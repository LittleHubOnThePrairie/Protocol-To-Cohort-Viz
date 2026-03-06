"""Human-in-the-loop annotation review component (PTCV-40, PTCV-41, PTCV-43).

Presents classified ICH sections with inline confidence flags and
lets a human reviewer accept, reject, or override classifications.
Supports section card view, document view, and full-protocol view
with span selection for unclassified text.
Annotations are persisted as JSONL for future model refinement.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import streamlit as st

from ptcv.annotations.models import AnnotationRecord, AnnotationSession
from ptcv.annotations.service import AnnotationService
from ptcv.annotations.span_mapper import (
    TextSpan,
    compute_coverage,
    map_sections_to_spans,
)
from ptcv.ui.components.ich_regenerator import ICH_SECTIONS, _extract_text

if TYPE_CHECKING:
    from ptcv.ich_parser.models import IchSection
    from ptcv.storage.gateway import StorageGateway


# All valid ICH section labels for the override dropdown
_SECTION_LABELS = [f"{code} {name}" for code, name in ICH_SECTIONS]


def _confidence_colour(level: str) -> str:
    """Return CSS colour for a confidence level.

    Args:
        level: "high" or "low".

    Returns:
        Hex colour string.
    """
    return "#2e7d32" if level == "high" else "#e65100"


def _confidence_bg(level: str) -> str:
    """Return CSS background colour for a confidence level.

    Args:
        level: "high" or "low".

    Returns:
        Hex colour string with transparency.
    """
    return "#e8f5e9" if level == "high" else "#fff3e0"


def _render_section_card(
    section: "IchSection",
    level: str,
    annotation: AnnotationRecord | None,
    svc: AnnotationService,
    session: AnnotationSession,
    idx: int,
) -> AnnotationRecord | None:
    """Render one section card with annotation controls.

    Args:
        section: The classified ICH section.
        level: "high" or "low" confidence classification.
        annotation: Existing annotation if resuming, else None.
        svc: AnnotationService for threshold access.
        session: Current annotation session.
        idx: Unique index for Streamlit widget keys.

    Returns:
        AnnotationRecord if the reviewer submitted, else None.
    """
    colour = _confidence_colour(level)
    bg = _confidence_bg(level)
    badge = "HIGH" if level == "high" else "LOW"
    text_content = _extract_text(section)

    # Section header with confidence badge
    st.markdown(
        f'<div style="border-left: 4px solid {colour}; '
        f'padding: 8px 12px; margin: 8px 0; '
        f'background: {bg}; border-radius: 4px;">'
        f'<strong>{section.section_code} {section.section_name}</strong>'
        f' &nbsp; '
        f'<span style="background: {colour}; color: white; '
        f'padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">'
        f'{badge} {section.confidence_score:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Show existing annotation status
    if annotation:
        st.caption(
            f"Previously annotated: {annotation.reviewer_action} "
            f"as '{annotation.reviewer_label}' "
            f"at {annotation.timestamp}"
        )

    # Full section text (no truncation)
    with st.expander("Section text", expanded=(level == "low")):
        st.text(text_content)

    # Annotation controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        default_action = 0
        if annotation:
            actions = ["accept", "reject", "override"]
            default_action = (
                actions.index(annotation.reviewer_action)
                if annotation.reviewer_action in actions
                else 0
            )
        action = st.selectbox(
            "Action",
            ["accept", "reject", "override"],
            index=default_action,
            key=f"action_{idx}",
            label_visibility="collapsed",
        )

    with col2:
        current_label = f"{section.section_code} {section.section_name}"
        if action == "override":
            default_label_idx = (
                _SECTION_LABELS.index(current_label)
                if current_label in _SECTION_LABELS
                else 0
            )
            reviewer_label = st.selectbox(
                "Assign label",
                _SECTION_LABELS,
                index=default_label_idx,
                key=f"label_{idx}",
                label_visibility="collapsed",
            )
        else:
            reviewer_label = current_label
            st.text(current_label)

    with col3:
        submitted = st.button("Save", key=f"save_{idx}")

    # Reviewer notes
    default_notes = annotation.reviewer_notes if annotation else ""
    notes = st.text_area(
        "Notes (optional)",
        value=default_notes,
        key=f"notes_{idx}",
        height=68,
        placeholder="Add reviewer notes...",
    )

    if submitted:
        section_id = f"{section.run_id}:{section.section_code}"
        record = AnnotationRecord(
            section_id=section_id,
            section_code=section.section_code,
            original_label=section.section_name,
            confidence=section.confidence_score,
            reviewer_label=reviewer_label,
            reviewer_action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
            text_span=text_content,
            reviewer_notes=notes,
        )
        return record

    return None


def _render_document_view(
    sections: list["IchSection"],
    svc: AnnotationService,
    session: AnnotationSession,
) -> None:
    """Render the full document with highlighted annotated sections.

    Shows all sections in document order with confidence-based colour
    highlighting. Clicking a section expands inline annotation controls.

    Args:
        sections: Deduplicated ICH sections sorted by section code.
        svc: AnnotationService for threshold access.
        session: Current annotation session.
    """
    for i, sec in enumerate(sections):
        level = svc.classify_confidence(sec.confidence_score)
        colour = _confidence_colour(level)
        bg = _confidence_bg(level)
        badge = "HIGH" if level == "high" else "LOW"
        text_content = _extract_text(sec)
        section_id = f"{sec.run_id}:{sec.section_code}"
        annotation = session.annotations.get(section_id)
        annotated_marker = " [annotated]" if annotation else ""

        # Render highlighted section block
        escaped_text = html.escape(text_content)
        st.markdown(
            f'<div style="border-left: 4px solid {colour}; '
            f'padding: 10px 14px; margin: 4px 0; '
            f'background: {bg}; border-radius: 4px;">'
            f'<strong>{sec.section_code} {sec.section_name}</strong>'
            f' &nbsp; '
            f'<span style="background: {colour}; color: white; '
            f'padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">'
            f'{badge} {sec.confidence_score:.2f}</span>'
            f'<span style="color: #666; font-size: 0.85em;">'
            f'{annotated_marker}</span>'
            f'<pre style="white-space: pre-wrap; margin-top: 8px; '
            f'font-size: 0.9em; font-family: inherit; '
            f'background: transparent; border: none; padding: 0;">'
            f'{escaped_text}</pre>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Inline annotation controls in expander
        exp_label = (
            f"Annotate {sec.section_code}"
            if not annotation
            else f"Re-annotate {sec.section_code} "
            f"(was: {annotation.reviewer_action})"
        )
        with st.expander(exp_label, expanded=False):
            doc_idx = i + 1000  # Offset keys to avoid collision with card view
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                default_action = 0
                if annotation:
                    actions = ["accept", "reject", "override"]
                    default_action = (
                        actions.index(annotation.reviewer_action)
                        if annotation.reviewer_action in actions
                        else 0
                    )
                action = st.selectbox(
                    "Action",
                    ["accept", "reject", "override"],
                    index=default_action,
                    key=f"doc_action_{doc_idx}",
                    label_visibility="collapsed",
                )

            with col2:
                current_label = (
                    f"{sec.section_code} {sec.section_name}"
                )
                if action == "override":
                    default_label_idx = (
                        _SECTION_LABELS.index(current_label)
                        if current_label in _SECTION_LABELS
                        else 0
                    )
                    reviewer_label = st.selectbox(
                        "Assign label",
                        _SECTION_LABELS,
                        index=default_label_idx,
                        key=f"doc_label_{doc_idx}",
                        label_visibility="collapsed",
                    )
                else:
                    reviewer_label = current_label
                    st.text(current_label)

            with col3:
                submitted = st.button(
                    "Save", key=f"doc_save_{doc_idx}",
                )

            default_notes = (
                annotation.reviewer_notes if annotation else ""
            )
            notes = st.text_area(
                "Notes (optional)",
                value=default_notes,
                key=f"doc_notes_{doc_idx}",
                height=68,
                placeholder="Add reviewer notes...",
            )

            if submitted:
                record = AnnotationRecord(
                    section_id=section_id,
                    section_code=sec.section_code,
                    original_label=sec.section_name,
                    confidence=sec.confidence_score,
                    reviewer_label=reviewer_label,
                    reviewer_action=action,
                    timestamp=datetime.now(
                        timezone.utc,
                    ).isoformat(),
                    text_span=text_content,
                    reviewer_notes=notes,
                )
                session.add(record)
                st.success(
                    f"Saved: {record.section_code} "
                    f"({record.reviewer_action})"
                )
                st.rerun()


def _render_protocol_view(
    protocol_text: str,
    sections: list["IchSection"],
    svc: AnnotationService,
    session: AnnotationSession,
    run_id: str,
) -> None:
    """Render the full protocol with classified sections highlighted.

    Shows the entire raw protocol text. Classified sections are
    highlighted with confidence-based colours. Unclassified gaps
    are shown in normal formatting with an option to assign a label.

    Args:
        protocol_text: Full raw protocol text.
        sections: Classified ICH sections from the parser.
        svc: AnnotationService for threshold access.
        session: Current annotation session.
        run_id: Parser run_id for annotation keys.
    """
    spans = map_sections_to_spans(protocol_text, sections)
    coverage = compute_coverage(spans, len(protocol_text))

    # Coverage summary
    st.markdown("### Coverage Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Coverage", f"{coverage['coverage_pct']:.1f}%")
    with col2:
        st.metric("Classified sections", coverage["classified_count"])
    with col3:
        st.metric("Unclassified gaps", coverage["gap_count"])

    st.divider()

    # Render spans in a scrollable container
    with st.container(height=600):
        for i, span in enumerate(spans):
            text = protocol_text[span.start:span.end]
            escaped = html.escape(text)

            if span.classified:
                # Classified section — highlighted
                level = svc.classify_confidence(span.confidence)
                colour = _confidence_colour(level)
                bg = _confidence_bg(level)
                badge = "HIGH" if level == "high" else "LOW"
                section_id = f"{run_id}:{span.section_code}"
                annotation = session.annotations.get(section_id)
                marker = " [annotated]" if annotation else ""

                st.markdown(
                    f'<div style="border-left: 4px solid {colour}; '
                    f'padding: 10px 14px; margin: 4px 0; '
                    f'background: {bg}; border-radius: 4px;">'
                    f'<strong>{span.section_code} '
                    f'{span.section_name}</strong> &nbsp; '
                    f'<span style="background: {colour}; color: white;'
                    f' padding: 2px 8px; border-radius: 12px; '
                    f'font-size: 0.8em;">'
                    f'{badge} {span.confidence:.2f}</span>'
                    f'<span style="color: #666; font-size: 0.85em;">'
                    f'{marker}</span>'
                    f'<pre style="white-space: pre-wrap; '
                    f'margin-top: 8px; font-size: 0.9em; '
                    f'font-family: inherit; background: transparent;'
                    f' border: none; padding: 0;">'
                    f'{escaped}</pre></div>',
                    unsafe_allow_html=True,
                )
            else:
                # Unclassified gap — plain text
                st.markdown(
                    f'<pre style="white-space: pre-wrap; '
                    f'margin: 2px 0; padding: 6px 14px; '
                    f'font-size: 0.9em; font-family: inherit; '
                    f'color: #555; background: #fafafa; '
                    f'border: none; border-left: 4px solid #ddd; '
                    f'border-radius: 4px;">{escaped}</pre>',
                    unsafe_allow_html=True,
                )

            # Annotation controls for unclassified gaps
            if not span.classified and span.length > 20:
                gap_key = f"{run_id}:manual:{i}"
                existing = session.annotations.get(gap_key)
                exp_text = (
                    f"Label this text ({span.length} chars)"
                    if not existing
                    else f"Re-label (was: {existing.reviewer_label})"
                )
                with st.expander(exp_text, expanded=False):
                    proto_idx = i + 2000
                    label = st.selectbox(
                        "Assign ICH label",
                        _SECTION_LABELS,
                        key=f"proto_label_{proto_idx}",
                    )
                    notes = st.text_area(
                        "Notes (optional)",
                        value=(
                            existing.reviewer_notes
                            if existing
                            else ""
                        ),
                        key=f"proto_notes_{proto_idx}",
                        height=68,
                        placeholder="Why is this text "
                        "this section?",
                    )
                    if st.button("Save", key=f"proto_save_{proto_idx}"):
                        code = label.split(" ", 1)[0]
                        record = AnnotationRecord(
                            section_id=gap_key,
                            section_code=code,
                            original_label="",
                            confidence=0.0,
                            reviewer_label=label,
                            reviewer_action="manual_label",
                            timestamp=datetime.now(
                                timezone.utc,
                            ).isoformat(),
                            text_span=text,
                            reviewer_notes=notes,
                            source="manual_span",
                        )
                        session.add(record)
                        st.success(f"Saved: {code} (manual)")
                        st.rerun()


def render_annotation_review(
    sections: list["IchSection"],
    registry_id: str,
    gateway: "StorageGateway",
    confidence_threshold: float = 0.80,
    protocol_text: str = "",
) -> None:
    """Render the full annotation review interface.

    Args:
        sections: Classified ICH sections from the parser.
        registry_id: Trial registry identifier.
        gateway: StorageGateway for annotation persistence.
        confidence_threshold: Score threshold for high/low split.
        protocol_text: Full raw protocol text for the protocol view.
            When provided, a "Full Protocol" tab is shown.
    """
    if not sections:
        st.warning("No sections to review. Parse a PDF first.")
        return

    svc = AnnotationService(
        gateway=gateway,
        confidence_threshold=confidence_threshold,
    )

    # Use the run_id from the first section (all share the same run)
    run_id = sections[0].run_id

    # Load or create annotation session
    session_key = f"annotation_session_{run_id}"
    if session_key not in st.session_state:
        existing = svc.load(
            registry_id=registry_id,
            run_id=run_id,
            total_sections=len(sections),
        )
        if existing:
            existing.total_sections = len(sections)
            st.session_state[session_key] = existing
        else:
            st.session_state[session_key] = AnnotationSession(
                registry_id=registry_id,
                run_id=run_id,
                total_sections=len(sections),
            )
    session: AnnotationSession = st.session_state[session_key]

    st.subheader("Annotation Review")

    # Progress bar
    progress = (
        session.annotated_count / session.total_sections
        if session.total_sections > 0
        else 0.0
    )
    st.progress(progress, text=(
        f"{session.annotated_count} / {session.total_sections} "
        f"sections annotated"
    ))

    # Threshold control
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=confidence_threshold,
        step=0.05,
        key="annotation_threshold",
    )
    svc.confidence_threshold = threshold

    # Deduplicate: keep highest-confidence per section code
    by_code: dict[str, "IchSection"] = {}
    for sec in sections:
        code = sec.section_code
        if code not in by_code or sec.confidence_score > by_code[code].confidence_score:
            by_code[code] = sec

    # View mode toggle (PTCV-41, PTCV-43)
    tab_names = ["Section View", "Document View"]
    if protocol_text:
        tab_names.append("Full Protocol")
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Section View
        # Separate low and high confidence sections
        low_sections = [
            s for s in by_code.values()
            if svc.classify_confidence(s.confidence_score) == "low"
        ]
        high_sections = [
            s for s in by_code.values()
            if svc.classify_confidence(s.confidence_score) == "high"
        ]

        # Sort by confidence ascending (worst first for low, best first for high)
        low_sections.sort(key=lambda s: s.confidence_score)
        high_sections.sort(key=lambda s: -s.confidence_score)

        # Low-confidence sections first (need attention)
        if low_sections:
            st.markdown(
                f"### Needs review ({len(low_sections)} sections)"
            )
            for i, sec in enumerate(low_sections):
                section_id = f"{sec.run_id}:{sec.section_code}"
                existing = session.annotations.get(section_id)
                record = _render_section_card(
                    sec, "low", existing, svc, session, idx=i,
                )
                if record:
                    session.add(record)
                    st.success(
                        f"Saved: {record.section_code} "
                        f"({record.reviewer_action})"
                    )
                    st.rerun()

        # High-confidence sections
        if high_sections:
            st.markdown(
                f"### High confidence ({len(high_sections)} sections)"
            )
            for i, sec in enumerate(high_sections):
                section_id = f"{sec.run_id}:{sec.section_code}"
                existing = session.annotations.get(section_id)
                record = _render_section_card(
                    sec,
                    "high",
                    existing,
                    svc,
                    session,
                    idx=i + len(low_sections),
                )
                if record:
                    session.add(record)
                    st.success(
                        f"Saved: {record.section_code} "
                        f"({record.reviewer_action})"
                    )
                    st.rerun()

    with tabs[1]:  # Document View
        # Full document view with highlighted sections (PTCV-41)
        all_sections = sorted(
            by_code.values(), key=lambda s: s.section_code,
        )
        _render_document_view(all_sections, svc, session)

    if protocol_text and len(tabs) > 2:
        with tabs[2]:  # Full Protocol (PTCV-43)
            _render_protocol_view(
                protocol_text=protocol_text,
                sections=list(by_code.values()),
                svc=svc,
                session=session,
                run_id=run_id,
            )

    # Submit all / save session
    st.divider()

    col_save, col_download = st.columns(2)

    with col_save:
        if st.button(
            "Save annotations to storage",
            disabled=(session.annotated_count == 0),
        ):
            key = svc.save(session)
            st.success(f"Annotations saved to `{key}`")

    with col_download:
        if session.annotated_count > 0:
            st.download_button(
                label="Download annotations (JSONL)",
                data=session.to_jsonl(),
                file_name=(
                    f"{registry_id}_{run_id}_annotations.jsonl"
                ),
                mime="application/jsonl",
            )

    # Summary stats
    if session.annotated_count > 0:
        st.caption(
            f"Session: {session.annotated_count} annotated, "
            f"{session.total_sections - session.annotated_count} "
            f"remaining"
        )
