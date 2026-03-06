"""Bridge from Query Pipeline AssembledProtocol to SOA extraction.

Converts assembled Appendix B sections (B.4 Trial Design, B.7 Schedule
of Activities) into synthetic IchSection objects that the SoaTableParser
can consume. This allows the SoA extraction pipeline to reuse already-
parsed and LLM-transformed content from the query pipeline rather than
re-parsing from the raw PDF.

PTCV-112: Feed query pipeline parsed output to SOA graphing functions.

Risk tier: LOW — adapter module, no side effects.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ..ich_parser.models import IchSection
from ..ich_parser.template_assembler import AssembledProtocol


# Section codes relevant to Schedule of Activities extraction.
# B.4 = Trial Design, B.7 = Schedule of Activities / Treatment.
_SOA_SECTION_CODES = ("B.4", "B.7", "B.10", "B.11", "B.8")

# Section code → fallback name (used when section_name not available).
_CODE_NAMES: dict[str, str] = {
    "B.4": "Trial Design",
    "B.7": "Schedule of Activities",
    "B.8": "Treatment",
    "B.10": "Assessment of Efficacy",
    "B.11": "Assessment of Safety",
}


def assembled_to_sections(
    assembled: AssembledProtocol,
    registry_id: str = "",
    source_sha256: str = "",
) -> list[IchSection]:
    """Convert relevant AssembledProtocol sections to IchSection list.

    Extracts B.4, B.7, B.8, B.10, and B.11 sections from the assembled
    protocol and wraps their query-hit content as synthetic IchSection
    objects. Only sections that have actual content (``populated=True``)
    are included.

    Args:
        assembled: Completed AssembledProtocol from the query pipeline.
        registry_id: Trial identifier for lineage traceability.
        source_sha256: SHA-256 of the source artifact.

    Returns:
        List of IchSection objects suitable for ``SoaExtractor.extract(
        sections=...)``. May be empty if no relevant sections have
        content.
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    sections: list[IchSection] = []

    for code in _SOA_SECTION_CODES:
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
