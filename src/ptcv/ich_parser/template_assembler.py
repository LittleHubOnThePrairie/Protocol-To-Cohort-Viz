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
import hashlib
import json
import logging
from typing import Any, Optional, Sequence, TYPE_CHECKING

from ptcv.ich_parser.query_schema import (
    AppendixBQuery,
    get_parent_sections,
    get_queries_for_section,
    load_query_schema,
    section_sort_key,
)

if TYPE_CHECKING:
    from ptcv.ich_parser.classification_router import (
        CascadeResult,
        RoutingDecision,
    )
    from ptcv.ich_parser.models import IchSection

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
    "B.6": "Discontinuation of Trial Intervention and Participant Withdrawal",
    "B.7": "Treatment of Participants",
    "B.8": "Assessment of Efficacy",
    "B.9": "Assessment of Safety",
    "B.10": "Statistics",
    "B.11": "Direct Access to Source Data/Documents",
    "B.12": "Quality Control and Quality Assurance",
    "B.13": "Ethics",
    "B.14": "Data Handling and Record Keeping",
    "B.15": "Financing and Insurance",
    "B.16": "Publication Policy",
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

    @classmethod
    def from_dict(cls, d: dict) -> "SourceReference":
        """Reconstruct from a JSON-safe dict."""
        return cls(
            pdf_page=d.get("pdf_page", 0),
            section_header=d.get("section_header", ""),
            char_offset_start=d.get("char_offset_start", -1),
            char_offset_end=d.get("char_offset_end", -1),
        )


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

    @classmethod
    def from_dict(cls, d: dict) -> "QueryExtractionHit":
        """Reconstruct from a JSON-safe dict."""
        return cls(
            query_id=d.get("query_id", ""),
            section_id=d.get("section_id", ""),
            parent_section=d.get("parent_section", ""),
            query_text=d.get("query_text", ""),
            extracted_content=d.get("extracted_content", ""),
            confidence=float(d.get("confidence", 0.0)),
            source=SourceReference.from_dict(d.get("source", {})),
        )


@dataclasses.dataclass
class ClassifiedSection:
    """Unified input model for document assembly from classified output.

    Adapter between classification sources (cascade router, LLM
    retemplater, NeoBERT) and the template assembler.  Markdown
    formatting in ``content_text`` is preserved verbatim.

    PTCV-165: ICH E6(R3) document assembly from classified pipeline.

    Attributes:
        section_code: ICH E6(R3) section code (e.g. ``"B.4"``).
        section_name: Human-readable name.
        content_text: Full section text.  Markdown formatting
            (tables, headings, lists) is preserved verbatim.
        confidence: Classification confidence 0.0-1.0.
        extraction_method: Pipeline extraction method tag
            (e.g. ``"E3:pdfplumber"``).
        classification_method: Classification method tag
            (e.g. ``"C2:neobert_sonnet"``).
        source_page: 1-based PDF page where content starts.
        source_header: Original protocol section header.
        content_hash: SHA-256 of ``content_text`` for dedup.
            Auto-computed if empty.
    """

    section_code: str
    section_name: str
    content_text: str
    confidence: float
    extraction_method: str = ""
    classification_method: str = ""
    source_page: int = 0
    source_header: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash and self.content_text:
            self.content_hash = hashlib.sha256(
                self.content_text.encode("utf-8"),
            ).hexdigest()

    @classmethod
    def from_ich_section(
        cls,
        section: "IchSection",
        extraction_method: str = "",
        classification_method: str = "",
    ) -> "ClassifiedSection":
        """Convert a single :class:`IchSection` to ClassifiedSection.

        Args:
            section: Classified ICH section from any pipeline path.
            extraction_method: E.g. ``"E3:pdfplumber"``.
            classification_method: E.g. ``"C2:neobert_sonnet"``.
        """
        source_page = 0
        try:
            cj = json.loads(section.content_json)
            page_range = cj.get("page_range", [])
            if page_range:
                source_page = page_range[0]
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
        return cls(
            section_code=section.section_code,
            section_name=section.section_name,
            content_text=section.content_text,
            confidence=section.confidence_score,
            extraction_method=extraction_method,
            classification_method=classification_method,
            source_page=source_page,
        )

    @classmethod
    def from_cascade_result(
        cls,
        cascade_result: "CascadeResult",
        extraction_method: str = "",
    ) -> list["ClassifiedSection"]:
        """Bulk convert all sections from a :class:`CascadeResult`.

        Extracts routing metadata (local vs sonnet) from the
        ``RoutingDecision`` list to populate ``classification_method``.

        Args:
            cascade_result: Output from
                :meth:`ClassificationRouter.classify`.
            extraction_method: E.g. ``"E3:pdfplumber"``.
        """
        decisions_by_code: dict[str, Any] = {}
        for d in cascade_result.decisions:
            decisions_by_code[d.final_section_code] = d

        result: list[ClassifiedSection] = []
        for section in cascade_result.sections:
            decision = decisions_by_code.get(section.section_code)
            route = decision.route if decision else "local"
            cls_method = f"cascade:{route}"
            result.append(cls.from_ich_section(
                section,
                extraction_method=extraction_method,
                classification_method=cls_method,
            ))
        return result


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
        extraction_method: Pipeline extraction method tag (PTCV-165).
            Empty for query-pipeline path.
        classification_method: Classification method tag (PTCV-165).
            Empty for query-pipeline path.
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
    extraction_method: str = ""
    classification_method: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "AssembledSection":
        """Reconstruct from a JSON-safe dict."""
        return cls(
            section_code=d.get("section_code", ""),
            section_name=d.get("section_name", ""),
            populated=d.get("populated", False),
            hits=[
                QueryExtractionHit.from_dict(h)
                for h in d.get("hits", [])
            ],
            average_confidence=float(
                d.get("average_confidence", 0.0),
            ),
            is_gap=d.get("is_gap", True),
            has_low_confidence=d.get("has_low_confidence", False),
            required_query_count=int(
                d.get("required_query_count", 0),
            ),
            answered_required_count=int(
                d.get("answered_required_count", 0),
            ),
            extraction_method=d.get("extraction_method", ""),
            classification_method=d.get(
                "classification_method", "",
            ),
        )


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

    @classmethod
    def from_dict(cls, d: dict) -> "CoverageReport":
        """Reconstruct from a JSON-safe dict."""
        return cls(
            total_sections=int(d.get("total_sections", 0)),
            populated_count=int(d.get("populated_count", 0)),
            gap_count=int(d.get("gap_count", 0)),
            average_confidence=float(
                d.get("average_confidence", 0.0),
            ),
            high_confidence_count=int(
                d.get("high_confidence_count", 0),
            ),
            medium_confidence_count=int(
                d.get("medium_confidence_count", 0),
            ),
            low_confidence_count=int(
                d.get("low_confidence_count", 0),
            ),
            total_queries=int(d.get("total_queries", 0)),
            answered_queries=int(d.get("answered_queries", 0)),
            required_queries=int(d.get("required_queries", 0)),
            answered_required=int(d.get("answered_required", 0)),
            gap_sections=list(d.get("gap_sections", [])),
            low_confidence_sections=list(
                d.get("low_confidence_sections", []),
            ),
        )

    review_confidence_count: int = 0


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

    @classmethod
    def from_dict(cls, d: dict) -> "AssembledProtocol":
        """Reconstruct from a JSON-safe dict (e.g. checkpoint)."""
        sections = [
            AssembledSection.from_dict(s)
            for s in d.get("sections", [])
        ]
        coverage = CoverageReport.from_dict(d.get("coverage", {}))
        traceability: dict[str, list[SourceReference]] = {}
        for code, refs in d.get("source_traceability", {}).items():
            traceability[code] = [
                SourceReference.from_dict(r) for r in refs
            ]
        return cls(
            sections=sections,
            coverage=coverage,
            source_traceability=traceability,
        )

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

    # Index hits by parent section, then sort within each group
    # by (section_id, query_id) to ensure deterministic subsection
    # order regardless of concurrent extraction completion order.
    hits_by_parent: dict[str, list[QueryExtractionHit]] = {}
    for h in hits:
        hits_by_parent.setdefault(h.parent_section, []).append(h)
    for parent_hits in hits_by_parent.values():
        parent_hits.sort(key=lambda h: (
            section_sort_key(h.section_id), h.query_id,
        ))

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
# Classified section assembly (PTCV-165)
# ---------------------------------------------------------------------------

# Required sections for gap detection.
_REQUIRED_SECTIONS: frozenset[str] = frozenset({"B.3", "B.4", "B.5"})


def assemble_from_classified(
    sections: Sequence[ClassifiedSection],
) -> AssembledProtocol:
    """Assemble classified sections into an ICH E6(R3) Appendix B template.

    Accepts classified sections from the cascade router (PTCV-161) or
    the legacy LLM retemplater, groups them by section code in canonical
    B.1-B.16 order, detects gaps, and produces the same
    :class:`AssembledProtocol` output as :func:`assemble_template`.

    Markdown formatting (tables, headings, lists) in
    ``ClassifiedSection.content_text`` is preserved verbatim.

    Dedup: sections with the same ``content_hash`` are kept only once.

    Args:
        sections: Classified sections from any pipeline path.

    Returns:
        :class:`AssembledProtocol` with sections, coverage, and
        traceability.
    """
    # Dedup by content_hash, group by section_code
    by_code: dict[str, list[ClassifiedSection]] = {}
    seen_hashes: set[str] = set()
    for cs in sections:
        if cs.content_hash in seen_hashes:
            logger.debug(
                "Dedup: skipping duplicate content for %s "
                "(hash=%s...)",
                cs.section_code,
                cs.content_hash[:12],
            )
            continue
        seen_hashes.add(cs.content_hash)
        by_code.setdefault(cs.section_code, []).append(cs)

    # Canonical B.1-B.16 order
    all_codes = sorted(
        APPENDIX_B_SECTION_NAMES.keys(),
        key=section_sort_key,
    )

    assembled: list[AssembledSection] = []
    traceability: dict[str, list[SourceReference]] = {}
    all_confidences: list[float] = []

    for code in all_codes:
        name = APPENDIX_B_SECTION_NAMES.get(
            code, f"Section {code}",
        )
        classified = by_code.get(code, [])

        if not classified:
            assembled.append(AssembledSection(
                section_code=code,
                section_name=name,
                populated=False,
                hits=[],
                average_confidence=0.0,
                is_gap=True,
                has_low_confidence=False,
                required_query_count=0,
                answered_required_count=0,
            ))
            continue

        # Primary = highest confidence for provenance
        primary = max(classified, key=lambda c: c.confidence)

        # Synthetic QueryExtractionHit for backward compat
        synthetic_hits: list[QueryExtractionHit] = []
        for cs in classified:
            synthetic_hits.append(QueryExtractionHit(
                query_id=f"{cs.section_code}.classified",
                section_id=cs.section_code,
                parent_section=cs.section_code,
                query_text=f"Classified content for "
                f"{cs.section_code}",
                extracted_content=cs.content_text,
                confidence=cs.confidence,
                source=SourceReference(
                    pdf_page=cs.source_page,
                    section_header=cs.source_header,
                ),
            ))

        confidences = [cs.confidence for cs in classified]
        avg_conf = sum(confidences) / len(confidences)
        has_low = any(c < LOW_CONFIDENCE for c in confidences)
        all_confidences.extend(confidences)

        # Traceability
        refs = [
            SourceReference(
                pdf_page=cs.source_page,
                section_header=cs.source_header,
            )
            for cs in classified if cs.source_page > 0
        ]
        if refs:
            traceability[code] = refs

        assembled.append(AssembledSection(
            section_code=code,
            section_name=name,
            populated=True,
            hits=synthetic_hits,
            average_confidence=round(avg_conf, 3),
            is_gap=False,
            has_low_confidence=has_low,
            required_query_count=0,
            answered_required_count=0,
            extraction_method=primary.extraction_method,
            classification_method=primary.classification_method,
        ))

    # Coverage report
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

    # Required section gaps
    populated_codes = {
        s.section_code for s in assembled if s.populated
    }
    missing_required = sorted(
        _REQUIRED_SECTIONS - populated_codes,
        key=section_sort_key,
    )
    if missing_required:
        for code in missing_required:
            name = APPENDIX_B_SECTION_NAMES.get(code, code)
            logger.warning(
                "%s (%s) is missing", code, name,
            )

    coverage = CoverageReport(
        total_sections=len(all_codes),
        populated_count=populated_count,
        gap_count=gap_count,
        average_confidence=round(overall_avg, 3),
        high_confidence_count=high_count,
        medium_confidence_count=medium_count,
        low_confidence_count=low_count,
        total_queries=0,
        answered_queries=0,
        required_queries=len(_REQUIRED_SECTIONS),
        answered_required=len(
            _REQUIRED_SECTIONS & populated_codes,
        ),
        gap_sections=gap_sections,
        low_confidence_sections=low_sections,
    )

    logger.info(
        "Assembled from classified: %d/%d sections populated, "
        "%d gaps, avg confidence %.2f, missing required: %s",
        populated_count,
        len(all_codes),
        gap_count,
        overall_avg,
        missing_required or "none",
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
                "parent_section": h.parent_section,
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
        section_dict: dict[str, Any] = {
            "section_code": s.section_code,
            "section_name": s.section_name,
            "populated": s.populated,
            "is_gap": s.is_gap,
            "has_low_confidence": s.has_low_confidence,
            "average_confidence": s.average_confidence,
            "required_query_count": s.required_query_count,
            "answered_required_count": s.answered_required_count,
            "hits": hits,
        }
        # PTCV-165: provenance metadata
        if s.extraction_method:
            section_dict["extraction_method"] = (
                s.extraction_method
            )
        if s.classification_method:
            section_dict["classification_method"] = (
                s.classification_method
            )
        sections.append(section_dict)

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

        # PTCV-165: provenance metadata
        if (
            section.extraction_method
            or section.classification_method
        ):
            meta_parts: list[str] = []
            if section.extraction_method:
                meta_parts.append(
                    f"Extraction: {section.extraction_method}"
                )
            if section.classification_method:
                meta_parts.append(
                    f"Classification: "
                    f"{section.classification_method}"
                )
            meta_parts.append(
                f"Confidence: "
                f"{section.average_confidence:.2f}"
            )
            lines.append(f"*{' | '.join(meta_parts)}*")
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
