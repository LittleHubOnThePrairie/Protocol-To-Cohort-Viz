"""ICH E6(R3) Appendix B template assembler (PTCV-92).

Takes query extraction results and populates an ICH E6(R3) Appendix B
template document, producing a restructured protocol in standard format.

The assembler groups ``QueryExtractionHit`` objects by Appendix B
parent section, detects gaps (sections with no or low-confidence content),
and produces a coverage report.

Output formats:
  - Structured dict (for pipeline/JSON consumption)
  - Markdown (for human review)

Risk tier: LOW — read-only data assembly, no I/O beyond in-memory ops.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Optional, Sequence

from ptcv.ich_parser.query_schema import (
    AppendixBQuery,
    get_parent_sections,
    get_queries_for_section,
    load_query_schema,
)

logger = logging.getLogger(__name__)

# Confidence thresholds.
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.70

# Gap placeholder text.
GAP_PLACEHOLDER = (
    "[Content not found in source protocol — manual review required]"
)

# Section names for the 16 Appendix B parent sections.
APPENDIX_B_SECTION_NAMES: dict[str, str] = {
    "B.1": "General Information",
    "B.2": "Background Information",
    "B.3": "Trial Objectives and Purpose",
    "B.4": "Trial Design",
    "B.5": "Selection and Withdrawal of Participants",
    "B.6": "Treatment of Participants",
    "B.7": "Assessment of Efficacy",
    "B.8": "Assessment of Safety",
    "B.9": "Data Handling and Record Keeping",
    "B.10": "Statistics",
    "B.11": "Quality Management",
    "B.12": "Ethics",
    "B.13": "Data Handling and Record Keeping – Supplements",
    "B.14": "Publication Policy",
    "B.15": "Supplements and Amendments",
    "B.16": "Appendices",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SourceReference:
    """Traceability back to the source protocol location.

    Attributes:
        pdf_page: 1-based PDF page where the content was found.
        section_header: Protocol section header (e.g. ``"5.1 Inclusion
            Criteria"``).
        char_offset_start: Start offset in the full-document text.
        char_offset_end: End offset in the full-document text.
    """

    pdf_page: int = 0
    section_header: str = ""
    char_offset_start: int = -1
    char_offset_end: int = -1


@dataclasses.dataclass
class QueryExtractionHit:
    """Result of running one Appendix B query against a protocol.

    This is the interface contract that PTCV-91 (query-driven extraction
    engine) will produce.  The template assembler consumes a list of
    these objects.

    Attributes:
        query_id: Identifier of the query (e.g. ``"B.1.1.q1"``).
        section_id: Appendix B sub-section (e.g. ``"B.1.1"``).
        parent_section: Top-level Appendix B section (e.g. ``"B.1"``).
        query_text: The original query that was run.
        extracted_content: Content extracted from the protocol.
        confidence: Extraction confidence 0.0–1.0.
        source: Traceability reference to the source location.
    """

    query_id: str
    section_id: str
    parent_section: str
    query_text: str
    extracted_content: str
    confidence: float
    source: SourceReference = dataclasses.field(
        default_factory=SourceReference,
    )


@dataclasses.dataclass
class AssembledSection:
    """One section in the assembled Appendix B template.

    Attributes:
        section_code: Appendix B section code (e.g. ``"B.1"``).
        section_name: Human-readable name.
        populated: Whether any content was extracted.
        hits: Individual query hits assigned to this section.
        average_confidence: Mean confidence across hits, ``0.0`` if
            no hits.
        is_gap: ``True`` if the section has no content at all.
        has_low_confidence: ``True`` if any hit has confidence < 0.70.
        required_query_count: Number of required queries for this section.
        answered_required_count: Number of required queries with a hit.
    """

    section_code: str
    section_name: str
    populated: bool
    hits: list[QueryExtractionHit]
    average_confidence: float
    is_gap: bool
    has_low_confidence: bool
    required_query_count: int
    answered_required_count: int


@dataclasses.dataclass
class CoverageReport:
    """Summary of template assembly completeness.

    Attributes:
        total_sections: Number of Appendix B parent sections.
        populated_count: Sections with at least one hit.
        gap_count: Sections with no content.
        average_confidence: Mean confidence across all hits.
        high_confidence_count: Sections with avg confidence >= 0.85.
        medium_confidence_count: Sections with avg confidence
            0.70–0.85.
        low_confidence_count: Sections with avg confidence < 0.70
            (but not gaps).
        total_queries: Total queries in the schema.
        answered_queries: Queries with at least one hit.
        required_queries: Total required queries.
        answered_required: Required queries with a hit.
        gap_sections: Section codes that are gaps.
        low_confidence_sections: Section codes with low confidence.
    """

    total_sections: int
    populated_count: int
    gap_count: int
    average_confidence: float
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    total_queries: int
    answered_queries: int
    required_queries: int
    answered_required: int
    gap_sections: list[str]
    low_confidence_sections: list[str]


@dataclasses.dataclass
class AssembledProtocol:
    """Complete assembled protocol in ICH E6(R3) Appendix B format.

    Attributes:
        sections: Ordered list of assembled sections
            (B.1, B.2, …, B.16).
        coverage: Summary coverage report.
        source_traceability: Mapping from section code to list of
            :class:`SourceReference` objects.
    """

    sections: list[AssembledSection]
    coverage: CoverageReport
    source_traceability: dict[str, list[SourceReference]]

    def get_section(self, code: str) -> Optional[AssembledSection]:
        """Return the assembled section for *code*, or ``None``."""
        for s in self.sections:
            if s.section_code == code:
                return s
        return None

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return _to_dict(self)

    def to_markdown(self) -> str:
        """Render as a Markdown document for human review."""
        return _to_markdown(self)


# ---------------------------------------------------------------------------
# Template assembler
# ---------------------------------------------------------------------------


def assemble_template(
    hits: Sequence[QueryExtractionHit],
    queries: Sequence[AppendixBQuery] | None = None,
) -> AssembledProtocol:
    """Assemble extraction hits into an ICH E6(R3) Appendix B template.

    Args:
        hits: Query extraction results from the extraction engine.
        queries: Full query schema (loaded from YAML if *None*).

    Returns:
        :class:`AssembledProtocol` with sections, coverage, and
        traceability.
    """
    if queries is None:
        queries = load_query_schema()

    parent_sections = get_parent_sections(queries)

    # Index hits by query_id for fast lookup.
    hits_by_query: dict[str, QueryExtractionHit] = {}
    for h in hits:
        hits_by_query[h.query_id] = h

    # Index hits by parent section.
    hits_by_parent: dict[str, list[QueryExtractionHit]] = {}
    for h in hits:
        hits_by_parent.setdefault(h.parent_section, []).append(h)

    # Build assembled sections in Appendix B order.
    assembled: list[AssembledSection] = []
    traceability: dict[str, list[SourceReference]] = {}
    all_confidences: list[float] = []

    total_queries = len(queries)
    answered_queries = 0
    required_queries = sum(1 for q in queries if q.required)
    answered_required = 0

    for section_code in parent_sections:
        section_name = APPENDIX_B_SECTION_NAMES.get(
            section_code, f"Section {section_code}",
        )
        section_hits = hits_by_parent.get(section_code, [])
        section_queries = get_queries_for_section(section_code, queries)

        # Count query coverage.
        hit_query_ids = {h.query_id for h in section_hits}
        required_in_section = sum(1 for q in section_queries if q.required)
        answered_required_in_section = sum(
            1 for q in section_queries
            if q.required and q.query_id in hit_query_ids
        )
        answered_in_section = sum(
            1 for q in section_queries
            if q.query_id in hit_query_ids
        )
        answered_queries += answered_in_section
        answered_required += answered_required_in_section

        populated = len(section_hits) > 0
        is_gap = not populated

        # Confidence.
        confidences = [h.confidence for h in section_hits]
        avg_conf = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        has_low_confidence = any(c < LOW_CONFIDENCE for c in confidences)
        all_confidences.extend(confidences)

        # Traceability.
        refs = [h.source for h in section_hits if h.source.pdf_page > 0]
        if refs:
            traceability[section_code] = refs

        assembled.append(AssembledSection(
            section_code=section_code,
            section_name=section_name,
            populated=populated,
            hits=list(section_hits),
            average_confidence=round(avg_conf, 3),
            is_gap=is_gap,
            has_low_confidence=has_low_confidence,
            required_query_count=required_in_section,
            answered_required_count=answered_required_in_section,
        ))

    # Build coverage report.
    populated_count = sum(1 for s in assembled if s.populated)
    gap_count = sum(1 for s in assembled if s.is_gap)
    overall_avg = (
        sum(all_confidences) / len(all_confidences)
        if all_confidences else 0.0
    )

    high_count = sum(
        1 for s in assembled
        if s.populated and s.average_confidence >= HIGH_CONFIDENCE
    )
    medium_count = sum(
        1 for s in assembled
        if s.populated
        and LOW_CONFIDENCE <= s.average_confidence < HIGH_CONFIDENCE
    )
    low_count = sum(
        1 for s in assembled
        if s.populated and s.average_confidence < LOW_CONFIDENCE
    )

    gap_sections = [s.section_code for s in assembled if s.is_gap]
    low_sections = [
        s.section_code for s in assembled
        if s.populated and s.average_confidence < LOW_CONFIDENCE
    ]

    coverage = CoverageReport(
        total_sections=len(parent_sections),
        populated_count=populated_count,
        gap_count=gap_count,
        average_confidence=round(overall_avg, 3),
        high_confidence_count=high_count,
        medium_confidence_count=medium_count,
        low_confidence_count=low_count,
        total_queries=total_queries,
        answered_queries=answered_queries,
        required_queries=required_queries,
        answered_required=answered_required,
        gap_sections=gap_sections,
        low_confidence_sections=low_sections,
    )

    logger.info(
        "Assembled protocol: %d/%d sections populated, %d gaps, "
        "avg confidence %.2f",
        populated_count,
        len(parent_sections),
        gap_count,
        overall_avg,
    )

    return AssembledProtocol(
        sections=assembled,
        coverage=coverage,
        source_traceability=traceability,
    )


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _to_dict(protocol: AssembledProtocol) -> dict:
    """Serialise an :class:`AssembledProtocol` to a JSON-safe dict."""
    sections = []
    for s in protocol.sections:
        hits = []
        for h in s.hits:
            hits.append({
                "query_id": h.query_id,
                "section_id": h.section_id,
                "query_text": h.query_text,
                "extracted_content": h.extracted_content,
                "confidence": h.confidence,
                "source": {
                    "pdf_page": h.source.pdf_page,
                    "section_header": h.source.section_header,
                    "char_offset_start": h.source.char_offset_start,
                    "char_offset_end": h.source.char_offset_end,
                },
            })
        sections.append({
            "section_code": s.section_code,
            "section_name": s.section_name,
            "populated": s.populated,
            "is_gap": s.is_gap,
            "has_low_confidence": s.has_low_confidence,
            "average_confidence": s.average_confidence,
            "required_query_count": s.required_query_count,
            "answered_required_count": s.answered_required_count,
            "hits": hits,
        })

    return {
        "format": "ICH E6(R3) Appendix B",
        "sections": sections,
        "coverage": dataclasses.asdict(protocol.coverage),
        "source_traceability": {
            code: [dataclasses.asdict(r) for r in refs]
            for code, refs in protocol.source_traceability.items()
        },
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _to_markdown(protocol: AssembledProtocol) -> str:
    """Render an :class:`AssembledProtocol` as Markdown."""
    lines: list[str] = []
    lines.append("# ICH E6(R3) Appendix B — Assembled Protocol")
    lines.append("")

    # Coverage summary.
    c = protocol.coverage
    lines.append("## Coverage Summary")
    lines.append("")
    lines.append(f"- **Sections populated:** {c.populated_count}/{c.total_sections}")
    lines.append(f"- **Gap sections:** {c.gap_count}")
    lines.append(f"- **Average confidence:** {c.average_confidence:.2f}")
    lines.append(
        f"- **Confidence breakdown:** "
        f"{c.high_confidence_count} high, "
        f"{c.medium_confidence_count} medium, "
        f"{c.low_confidence_count} low"
    )
    lines.append(
        f"- **Queries answered:** {c.answered_queries}/{c.total_queries} "
        f"(required: {c.answered_required}/{c.required_queries})"
    )
    if c.gap_sections:
        lines.append(f"- **Gap sections:** {', '.join(c.gap_sections)}")
    if c.low_confidence_sections:
        lines.append(
            f"- **Low confidence:** {', '.join(c.low_confidence_sections)}"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sections.
    for section in protocol.sections:
        lines.append(
            f"## {section.section_code} {section.section_name}"
        )
        lines.append("")

        if section.is_gap:
            lines.append(f"> {GAP_PLACEHOLDER}")
            lines.append("")
            continue

        for hit in section.hits:
            conf_flag = ""
            if hit.confidence < LOW_CONFIDENCE:
                conf_flag = " [LOW CONFIDENCE — verify]"
            elif hit.confidence < HIGH_CONFIDENCE:
                conf_flag = " [moderate confidence]"

            lines.append(f"### {hit.section_id}: {hit.query_text}")
            lines.append("")
            lines.append(f"{hit.extracted_content}")
            if conf_flag:
                lines.append("")
                lines.append(f"*{conf_flag.strip()}*")
            lines.append("")

            # Source reference.
            src = hit.source
            if src.pdf_page > 0:
                ref_parts = [f"Page {src.pdf_page}"]
                if src.section_header:
                    ref_parts.append(f'"{src.section_header}"')
                lines.append(
                    f"*Source: {', '.join(ref_parts)}*"
                )
                lines.append("")

    return "\n".join(lines)
