"""Sidebar file browser component (PTCV-33).

Scans the protocol data directory for PDF files grouped by
registry source (clinicaltrials / eu-ctr) and renders a
selectable list in the Streamlit sidebar.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Registry source directories under data/protocols/
_REGISTRY_SOURCES = ("clinicaltrials", "eu-ctr")


def _scan_pdfs(protocols_dir: Path) -> dict[str, list[str]]:
    """Scan protocol directories for PDF files.

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
                f.name for f in source_dir.iterdir()
                if f.suffix.lower() == ".pdf" and f.is_file()
            )
            if pdfs:
                groups[source] = pdfs
    return groups


def render_file_browser(
    protocols_dir: Path,
) -> tuple[str | None, Path | None]:
    """Render the sidebar file browser.

    Args:
        protocols_dir: Root protocols directory.

    Returns:
        Tuple of (registry_source, full_path) for the selected
        file, or (None, None) if nothing is selected.
    """
    st.sidebar.header("Protocol Files")

    groups = _scan_pdfs(protocols_dir)

    if not groups:
        st.sidebar.info("No protocol PDFs found.")
        return None, None

    # Build flat list of (label, source, filename) for the radio
    options: list[tuple[str, str, str]] = []
    for source, filenames in groups.items():
        for fname in filenames:
            options.append((f"{source}/{fname}", source, fname))

    if not options:
        return None, None

    # Group labels by source for visual clarity
    labels = [opt[0] for opt in options]

    selected_label = st.sidebar.radio(
        "Select a protocol file:",
        options=labels,
        index=None,
        label_visibility="collapsed",
    )

    if selected_label is None:
        return None, None

    idx = labels.index(selected_label)
    _, source, fname = options[idx]
    full_path = protocols_dir / source / fname
    return source, full_path
