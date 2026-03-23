"""Source Provenance Matrix for Query Pipeline Results.

PTCV-251: Displays a matrix showing how each ICH E6(R3) Appendix B
section was populated — from text, tables, diagrams, or
ClinicalTrials.gov registry fallback — with source page references.

The matrix is built from pipeline result data (no additional extraction
needed) and rendered as a Streamlit dataframe with visual indicators.

Usage::

    from ptcv.ui.components.provenance_matrix import (
        build_provenance_matrix,
        render_provenance_matrix,
    )

    rows = build_provenance_matrix(pipeline_result, nct_id="NCT05376319")
    render_provenance_matrix(rows)  # Streamlit render
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Optional


# All ICH E6(R3) Appendix B sections for the matrix rows.
_ICH_SECTIONS: dict[str, str] = {
    "B.1": "General Information",
    "B.1.1": "Protocol Title",
    "B.1.2": "Sponsor Name & Address",
    "B.1.3": "Investigator Name & Qualifications",
    "B.2": "Background Information",
    "B.3": "Trial Objectives and Purpose",
    "B.4": "Trial Design",
    "B.4.1": "Primary Endpoints",
    "B.4.4": "Blinding",
    "B.5": "Selection of Subjects",
    "B.5.1": "Inclusion Criteria",
    "B.5.2": "Exclusion Criteria",
    "B.6": "Discontinuation and Withdrawal",
    "B.7": "Treatment of Subjects",
    "B.8": "Assessment of Efficacy",
    "B.9": "Assessment of Safety",
    "B.10": "Statistics",
    "B.11": "Direct Access to Source Data",
    "B.12": "Ethics",
    "B.13": "Data Handling and Record Keeping",
    "B.14": "Financing, Insurance, Publication",
}

_REGISTRY_MARKER = "[REGISTRY"
_CT_GOV_URL = "https://clinicaltrials.gov/study/{nct_id}"


@dataclasses.dataclass
class ProvenanceRow:
    """One row of the provenance matrix.

    Attributes:
        section_code: ICH section code (e.g. "B.4").
        section_name: Human-readable name.
        has_text: Content extracted from PDF prose.
        has_table: Content extracted from PDF tables.
        has_diagram: Content extracted from diagrams.
        has_registry: Content from ClinicalTrials.gov fallback.
        source_pages: PDF page references (e.g. "pp. 12-14").
        source_link: Hyperlink to source (CT.gov URL or page ref).
        confidence: Classification confidence (HIGH/REVIEW/LOW).
        populated: Whether any content exists for this section.
    """

    section_code: str
    section_name: str
    has_text: bool = False
    has_table: bool = False
    has_diagram: bool = False
    has_registry: bool = False
    source_pages: str = ""
    source_link: str = ""
    confidence: str = ""
    populated: bool = False


def build_provenance_matrix(
    pipeline_result: dict[str, Any],
    nct_id: str = "",
) -> list[ProvenanceRow]:
    """Build the provenance matrix from pipeline result data.

    Analyzes the pipeline result to determine the extraction method
    and source for each ICH section.

    Args:
        pipeline_result: Dict returned by ``run_query_pipeline()``.
        nct_id: NCT ID for CT.gov links (extracted from filename if empty).

    Returns:
        Ordered list of ProvenanceRow, one per ICH section.
    """
    assembled = pipeline_result.get("assembled")
    match_result = pipeline_result.get("match_result")
    extraction_result = pipeline_result.get("extraction_result")
    protocol_index = pipeline_result.get("protocol_index")

    # Resolve NCT ID
    if not nct_id and protocol_index:
        source_path = getattr(protocol_index, "source_path", "")
        m = re.search(r"NCT\d{8}", source_path)
        if m:
            nct_id = m.group(0)

    ct_gov_url = _CT_GOV_URL.format(nct_id=nct_id) if nct_id else ""

    # Build section → page mapping from MatchResult
    section_pages: dict[str, list[str]] = {}
    section_confidence: dict[str, str] = {}
    if match_result:
        for mapping in getattr(match_result, "mappings", []):
            if not mapping.matches:
                continue
            top = mapping.matches[0]
            code = top.ich_section_code
            conf = getattr(top, "confidence", None)
            section_pages.setdefault(code, []).append(
                mapping.protocol_section_number
            )
            if conf:
                section_confidence[code] = conf.value if hasattr(conf, "value") else str(conf)

    # Build section → extraction method from ExtractionResult
    section_methods: dict[str, set[str]] = {}
    section_has_registry: dict[str, bool] = {}
    if extraction_result:
        for ext in getattr(extraction_result, "extractions", []):
            # PTCV-288: Correct parent extraction for ICH section IDs.
            _parts = ext.section_id.split(".")
            parent = ext.section_id if len(_parts) <= 2 else ".".join(_parts[:-1])
            code = ext.section_id

            method = getattr(ext, "extraction_method", "")
            section_methods.setdefault(code, set()).add(method)
            section_methods.setdefault(parent, set()).add(method)

            # Check for registry content
            content = getattr(ext, "content", "")
            if _REGISTRY_MARKER in content:
                section_has_registry[code] = True
                section_has_registry[parent] = True

    # Build section → populated from AssembledProtocol
    section_populated: dict[str, bool] = {}
    if assembled:
        for code in _ICH_SECTIONS:
            parent = code.split(".")[0] + "." + code.split(".")[1] if len(code.split(".")) > 2 else code
            sec = assembled.get_section(parent)
            if sec and getattr(sec, "populated", False):
                section_populated[parent] = True
                section_populated[code] = True

    # Build rows
    rows: list[ProvenanceRow] = []
    for code, name in _ICH_SECTIONS.items():
        methods = section_methods.get(code, set())
        pages = section_pages.get(
            code.split(".")[0] + "." + code.split(".")[1] if len(code.split(".")) > 2 else code,
            [],
        )
        is_registry = section_has_registry.get(code, False)
        is_populated = section_populated.get(code, False) or bool(methods) or is_registry

        has_text = bool(methods - {"table", "diagram"}) or bool(pages)
        has_table = "table" in methods
        has_diagram = "diagram" in methods

        # Build source pages string
        if pages:
            unique_pages = sorted(set(pages))
            if len(unique_pages) == 1:
                source_pages = f"p. {unique_pages[0]}"
            else:
                source_pages = f"pp. {', '.join(unique_pages)}"
        else:
            source_pages = ""

        # Build source link
        source_parts: list[str] = []
        if source_pages:
            source_parts.append(source_pages)
        if has_table:
            source_parts.append("(table)")
        if has_diagram:
            source_parts.append("(diagram)")
        if is_registry and ct_gov_url:
            source_parts.append(f"[CT.gov]({ct_gov_url})")
        elif is_registry:
            source_parts.append("CT.gov")

        source_link = " ".join(source_parts) if source_parts else ""
        if not is_populated:
            source_link = "Not found"

        conf = section_confidence.get(
            code.split(".")[0] + "." + code.split(".")[1] if len(code.split(".")) > 2 else code,
            "",
        )

        rows.append(ProvenanceRow(
            section_code=code,
            section_name=name,
            has_text=has_text and is_populated,
            has_table=has_table,
            has_diagram=has_diagram,
            has_registry=is_registry,
            source_pages=source_pages,
            source_link=source_link,
            confidence=conf,
            populated=is_populated,
        ))

    return rows


def matrix_to_dataframe(rows: list[ProvenanceRow]) -> Any:
    """Convert provenance rows to a pandas DataFrame for display.

    Args:
        rows: List of ProvenanceRow from build_provenance_matrix().

    Returns:
        pandas DataFrame with display-friendly columns.
    """
    import pandas as pd

    data = []
    for row in rows:
        data.append({
            "Section": f"{row.section_code} {row.section_name}",
            "Text": "✓" if row.has_text else "",
            "Tables": "✓" if row.has_table else "",
            "Diagrams": "✓" if row.has_diagram else "",
            "CT.gov": "✓" if row.has_registry else "",
            "Source": row.source_link if row.populated else "⚠ Not found",
        })

    return pd.DataFrame(data)


def render_provenance_matrix(rows: list[ProvenanceRow]) -> None:
    """Render the provenance matrix in Streamlit.

    Args:
        rows: List of ProvenanceRow from build_provenance_matrix().
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader("Source Provenance Matrix")
    st.caption(
        "Shows how each ICH E6(R3) section was populated: "
        "from PDF text, tables, diagrams, or ClinicalTrials.gov registry."
    )

    df = matrix_to_dataframe(rows)

    # Style: highlight unfilled sections
    def _highlight_unfilled(row: Any) -> list[str]:
        if row["Source"] == "⚠ Not found":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_unfilled, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary stats
    total = len(rows)
    populated = sum(1 for r in rows if r.populated)
    registry_count = sum(1 for r in rows if r.has_registry)
    table_count = sum(1 for r in rows if r.has_table)
    diagram_count = sum(1 for r in rows if r.has_diagram)

    cols = st.columns(5)
    cols[0].metric("Coverage", f"{populated}/{total}")
    cols[1].metric("Text", f"{sum(1 for r in rows if r.has_text)}")
    cols[2].metric("Tables", f"{table_count}")
    cols[3].metric("Diagrams", f"{diagram_count}")
    cols[4].metric("CT.gov", f"{registry_count}")


__all__ = [
    "ProvenanceRow",
    "build_provenance_matrix",
    "matrix_to_dataframe",
    "render_provenance_matrix",
]
