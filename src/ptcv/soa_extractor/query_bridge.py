"""Bridge from Query Pipeline AssembledProtocol to IchSection consumers.

.. deprecated:: PTCV-242
    This module is deprecated. The Query Pipeline now handles all stages
    natively (PTCV-237 through PTCV-241). The ``assembled_to_sections``
    function and session-state helpers remain available for backward
    compatibility but should not be used in new code.

Converts all assembled ICH E6(R3) Appendix B sections (B.1-B.14) into
synthetic IchSection objects that downstream tabs (SoA, Results, Quality,
Review, SDTM) can consume.

PTCV-112: Feed query pipeline parsed output to SOA graphing functions.
PTCV-136: Extend bridge to all 14 ICH sections for downstream migration.
PTCV-179: Add classified pipeline session state accessors.

Risk tier: LOW — adapter module, no side effects.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ..ich_parser.models import IchSection
from ..ich_parser.template_assembler import AssembledProtocol


# All ICH E6(R3) Appendix B section codes (PTCV-136).
_SECTION_CODES = (
    "B.1", "B.2", "B.3", "B.4", "B.5", "B.6", "B.7",
    "B.8", "B.9", "B.10", "B.11", "B.12", "B.13", "B.14",
)

# Section code → fallback name (used when section_name not available).
_CODE_NAMES: dict[str, str] = {
    "B.1": "General Information",
    "B.2": "Background Information",
    "B.3": "Trial Objectives and Purpose",
    "B.4": "Trial Design",
    "B.5": "Selection of Subjects",
    "B.6": "Discontinuation and Participant Withdrawal",
    "B.7": "Treatment of Participants",
    "B.8": "Assessment of Efficacy",
    "B.9": "Assessment of Safety",
    "B.10": "Statistics",
    "B.11": "Direct Access to Source Data/Documents",
    "B.12": "Ethics",
    "B.13": "Data Handling and Record Keeping",
    "B.14": "Financing, Insurance, and Publication Policy",
}


def assembled_to_sections(
    assembled: AssembledProtocol,
    registry_id: str = "",
    source_sha256: str = "",
) -> list[IchSection]:
    """Convert all populated AssembledProtocol sections to IchSection list.

    Iterates over all 14 ICH E6(R3) Appendix B sections (B.1–B.14) and
    wraps their query-hit content as synthetic IchSection objects.  Only
    sections that have actual content (``populated=True``) are included.

    Args:
        assembled: Completed AssembledProtocol from the query pipeline.
        registry_id: Trial identifier for lineage traceability.
        source_sha256: SHA-256 of the source artifact.

    Returns:
        List of IchSection objects consumable by any downstream tab
        (SoA, Results, Quality, Review, SDTM).  May be empty if no
        sections have content.
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    sections: list[IchSection] = []

    for code in _SECTION_CODES:
        asm_section = assembled.get_section(code)
        if asm_section is None or not asm_section.populated:
            continue

        # Concatenate all query-hit content for this section.
        text_parts: list[str] = []
        for hit in asm_section.hits:
            if hit.extracted_content:
                text_parts.append(hit.extracted_content)

        combined_text = "\n\n".join(text_parts)
        if not combined_text.strip():
            continue

        section = IchSection(
            run_id=run_id,
            source_run_id="query_pipeline",
            source_sha256=source_sha256,
            registry_id=registry_id,
            section_code=code,
            section_name=asm_section.section_name or _CODE_NAMES.get(
                code, code
            ),
            content_json=json.dumps({"text": combined_text}),
            confidence_score=asm_section.average_confidence,
            review_required=asm_section.has_low_confidence,
            legacy_format=False,
            extraction_timestamp_utc=timestamp,
            content_text=combined_text,
        )
        sections.append(section)

    return sections


def has_query_pipeline_results(session_state: dict) -> bool:
    """Check if query pipeline results exist in Streamlit session state.

    Args:
        session_state: ``st.session_state`` dict.

    Returns:
        True if at least one query pipeline result is cached.
    """
    cache = session_state.get("query_cache", {})
    return bool(cache)


def get_assembled_protocol(
    session_state: dict,
    file_sha: str,
) -> AssembledProtocol | None:
    """Retrieve the AssembledProtocol from session state for a file.

    Searches the query_cache for any entry matching the given file_sha
    prefix. Query pipeline cache keys are version-stamped
    (``{sha}_v{ver}_t{flag}``) so we match on the SHA prefix.

    Args:
        session_state: ``st.session_state`` dict.
        file_sha: SHA-256 hex digest of the PDF file.

    Returns:
        The AssembledProtocol if found, None otherwise.
    """
    cache = session_state.get("query_cache", {})
    for key, result in cache.items():
        if key.startswith(file_sha) and isinstance(result, dict):
            assembled = result.get("assembled")
            if assembled is not None:
                return assembled
    return None


# ---------------------------------------------------------------------------
# Classified pipeline session state helpers (PTCV-179)
# ---------------------------------------------------------------------------

_CLASSIFIED_CACHE_KEY = "classified_cache"


def store_classified_result(
    session_state: dict,
    file_sha: str,
    assembled: AssembledProtocol,
    cascade_stats: dict[str, object] | None = None,
    stage_timings: dict[str, float] | None = None,
) -> None:
    """Store classified pipeline AssembledProtocol in session state.

    Args:
        session_state: ``st.session_state`` dict.
        file_sha: SHA-256 hex digest of the source PDF.
        assembled: AssembledProtocol from ``assemble_from_classified()``.
        cascade_stats: Optional serialised ``RoutingStats`` dict for
            routing metrics display.
        stage_timings: Optional timing dict for pipeline stages.
    """
    cache = session_state.setdefault(_CLASSIFIED_CACHE_KEY, {})
    cache[file_sha] = {
        "assembled": assembled,
        "cascade_stats": cascade_stats,
        "stage_timings": stage_timings or {},
    }


def get_classified_assembled_protocol(
    session_state: dict,
    file_sha: str,
) -> AssembledProtocol | None:
    """Retrieve classified pipeline AssembledProtocol from session state.

    Args:
        session_state: ``st.session_state`` dict.
        file_sha: SHA-256 hex digest of the PDF file.

    Returns:
        The AssembledProtocol if found, None otherwise.
    """
    cache = session_state.get(_CLASSIFIED_CACHE_KEY, {})
    entry = cache.get(file_sha)
    if entry is not None and isinstance(entry, dict):
        return entry.get("assembled")
    return None


def has_classified_results(
    session_state: dict,
    file_sha: str,
) -> bool:
    """Check if classified pipeline results exist for a file.

    Args:
        session_state: ``st.session_state`` dict.
        file_sha: SHA-256 hex digest of the PDF file.

    Returns:
        True if classified results are cached for this file.
    """
    cache = session_state.get(_CLASSIFIED_CACHE_KEY, {})
    return file_sha in cache
