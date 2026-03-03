"""Sidebar file browser component (PTCV-33, PTCV-42).

Groups protocol PDFs by therapeutic area with title display
and quality-based ordering.  Falls back to flat filename list
when metadata is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ptcv.ui.components.protocol_catalog import (
    AREA_DISPLAY_ORDER,
    ProtocolEntry,
    load_protocol_catalog,
)

# Registry source directories under data/protocols/
_REGISTRY_SOURCES = ("clinicaltrials", "eu-ctr")

# Verdict badge prefixes for display labels
_VERDICT_BADGE: dict[str, str] = {
    "ICH_E6R3": "[ICH] ",
    "PARTIAL_ICH": "[PART] ",
    "NON_ICH": "[NON] ",
    "": "",
}


def _scan_pdfs(protocols_dir: Path) -> dict[str, list[str]]:
    """Scan protocol directories for PDF files.

    Retained for backward compatibility with existing tests.
    New code should use ``load_protocol_catalog()`` instead.

    Args:
        protocols_dir: Root protocols directory
            (e.g., ``data/protocols/``).

    Returns:
        Dict mapping registry source name to sorted list of
        PDF filenames found in that subdirectory.
    """
    groups: dict[str, list[str]] = {}
    for source in _REGISTRY_SOURCES:
        source_dir = protocols_dir / source
        if source_dir.is_dir():
            pdfs = sorted(
                f.name
                for f in source_dir.iterdir()
                if f.suffix.lower() == ".pdf" and f.is_file()
            )
            if pdfs:
                groups[source] = pdfs
    return groups


def _format_radio_label(entry: ProtocolEntry) -> str:
    """Build a display label with optional verdict badge.

    Args:
        entry: Protocol entry.

    Returns:
        Formatted label string.
    """
    badge = _VERDICT_BADGE.get(
        entry.quality.format_verdict, ""
    )
    return f"{badge}{entry.display_label}"


def render_file_browser(
    protocols_dir: Path,
) -> tuple[str | None, Path | None]:
    """Render the sidebar file browser with therapeutic area grouping.

    Groups protocols by therapeutic area (Oncology, Cardiovascular,
    Nervous System, Diabetes/Metabolic, Other).  Within each group,
    protocols are ordered by quality score (best first).  Each entry
    shows a truncated title with NCT# parenthetical.

    Args:
        protocols_dir: Root protocols directory.

    Returns:
        Tuple of (registry_source, full_path) for the selected
        file, or ``(None, None)`` if nothing is selected.
    """
    st.sidebar.header("Protocol Files")

    catalog = load_protocol_catalog(protocols_dir)

    if not catalog:
        st.sidebar.info("No protocol PDFs found.")
        return None, None

    # Build a global lookup so we can resolve selection
    entry_map: dict[str, ProtocolEntry] = {}

    for area in AREA_DISPLAY_ORDER:
        entries = catalog.get(area)
        if not entries:
            continue

        with st.sidebar.expander(
            f"{area.value} ({len(entries)})",
            expanded=False,
        ):
            labels: list[str] = []
            for entry in entries:
                key = f"{entry.registry_source}/{entry.filename}"
                entry_map[key] = entry
                labels.append(_format_radio_label(entry))

            selected = st.radio(
                f"Select from {area.value}:",
                options=labels,
                index=None,
                key=f"_fb_radio_{area.value}",
                label_visibility="collapsed",
            )

            if selected is not None:
                idx = labels.index(selected)
                entry = entries[idx]
                sel_key = (
                    f"{entry.registry_source}/{entry.filename}"
                )
                st.session_state["_fb_selected"] = sel_key

    # Resolve selection
    selected_key = st.session_state.get("_fb_selected")
    if selected_key and selected_key in entry_map:
        entry = entry_map[selected_key]
        return entry.registry_source, entry.file_path

    return None, None
