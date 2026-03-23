"""Query-driven pipeline component for Streamlit UI (PTCV-95).

Orchestrates the full query-driven pipeline:
  extract_protocol_index -> SectionMatcher.match -> QueryExtractor.extract
  -> assemble_template

Pure-Python helpers are separated from Streamlit rendering for testability.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from ptcv.ui.components.confidence_badge import (
    confidence_color,
    confidence_label,
    format_confidence,
)

logger = logging.getLogger(__name__)

# Maximum chars to show in content preview columns.
_CONTENT_PREVIEW_LEN = 120

# Bump when pipeline output schema changes to invalidate stale caches.
_PIPELINE_CACHE_VERSION = 3  # v3: enable_transformation toggle (PTCV-103)


# ---------------------------------------------------------------------------
# Pure-Python helpers (testable, no Streamlit imports)
# ---------------------------------------------------------------------------


PIPELINE_STAGES = [
    "Document Assembly",
    "Section Classification",
    "Query Extraction",
    "Result Aggregation",
    "SoA Extraction",
    "SDTM Generation",
    "Validation",
]
"""Ordered stage names for the query pipeline (PTCV-123, PTCV-250)."""


def run_query_pipeline(
    pdf_path: str | Path,
    anthropic_api_key: str | None = None,
    enable_summarization: bool = True,
    enable_transformation: bool = False,
    progress_callback: Optional[
        Callable[[str, float, int, int], None]
    ] = None,
) -> dict[str, Any]:
    """Orchestrate the full query-driven pipeline.

    Args:
        pdf_path: Path to the protocol PDF.
        anthropic_api_key: Optional Anthropic API key for LLM-driven
            sub-section scoring. Falls back to keyword-only if absent.
        enable_summarization: Whether to run the sub-section enrichment
            stage (PTCV-96). Defaults to *True*.
        enable_transformation: Whether to apply LLM-driven content
            transformation for high-confidence extractions
            (PTCV-103). Defaults to *False*.
        progress_callback: Optional ``(stage_name, progress)`` callback
            invoked between pipeline stages. *progress* is ``0.0``
            when a stage starts and ``1.0`` when it completes
            (PTCV-123).

    Returns:
        Dict with keys: ``protocol_index``, ``match_result``,
        ``enriched_match_result``, ``extraction_result``, ``assembled``,
        ``assembled_markdown``, ``coverage``.

    Raises:
        Exception: Re-raised from any pipeline stage.
    """
    from ptcv.ich_parser.toc_extractor import extract_protocol_index
    from ptcv.ich_parser.section_classifier import SectionClassifier
    from ptcv.ich_parser.query_extractor import QueryExtractor
    from ptcv.ich_parser.template_assembler import (
        QueryExtractionHit,
        SourceReference,
        assemble_template,
    )

    def _notify(
        stage: str, progress: float,
        done: int = 0, total: int = 0,
    ) -> None:
        if progress_callback is not None:
            progress_callback(stage, progress, done, total)

    stage_timings: dict[str, float] = {}

    # Stage 1: Extract protocol index
    _notify("Document Assembly", 0.0)
    t0 = time.monotonic()
    protocol_index = extract_protocol_index(pdf_path)
    stage_timings["document_assembly"] = round(
        time.monotonic() - t0, 3
    )
    _notify("Document Assembly", 1.0)

    # Stage 2: Hybrid embedding-first classification (PTCV-279)
    # FAISS centroids as primary signal where available, keyword
    # fallback for sections without centroids, LLM sub-section
    # scoring via SummarizationMatcher.
    _notify("Section Classification", 0.0)
    t0 = time.monotonic()
    api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    classifier = SectionClassifier(anthropic_api_key=api_key)

    def _on_classify(done: int, total: int) -> None:
        if total > 0:
            _notify(
                "Section Classification",
                done / total,
                done, total,
            )

    match_result, enriched_result = classifier.classify(
        protocol_index,
        progress_callback=_on_classify,
    )
    stage_timings["section_classification"] = round(
        time.monotonic() - t0, 3
    )
    _notify("Section Classification", 1.0)

    # Stage 3: Query extraction (PTCV-103: optional LLM transform)
    _notify("Query Extraction", 0.0)
    t0 = time.monotonic()
    api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    extractor = QueryExtractor(
        enable_transformation=enable_transformation,
        anthropic_api_key=api_key,
    )

    def _on_extract(done: int, total: int) -> None:
        if total > 0:
            _notify(
                "Query Extraction",
                done / total,
                done, total,
            )

    extraction_result = extractor.extract(
        protocol_index, match_result,
        progress_callback=_on_extract,
    )
    stage_timings["query_extraction"] = round(
        time.monotonic() - t0, 3
    )
    _notify("Query Extraction", 1.0)

    # Stage 4: Template assembly
    _notify("Result Aggregation", 0.0)
    t0 = time.monotonic()
    hits = []
    for ext in extraction_result.extractions:
        # PTCV-288: Correct parent extraction for ICH section IDs.
        # "B.3" has 2 parts → is already a parent. "B.3.1" has 3 → parent is "B.3".
        # The old rsplit(".",1)[0] broke parent-level IDs: "B.3" → "B".
        _parts = ext.section_id.split(".")
        parent = ext.section_id if len(_parts) <= 2 else ".".join(_parts[:-1])
        hits.append(QueryExtractionHit(
            query_id=ext.query_id,
            section_id=ext.section_id,
            parent_section=parent,
            query_text="",
            extracted_content=ext.content,
            confidence=ext.confidence,
            source=SourceReference(
                section_header=ext.source_section,
            ),
        ))

    assembled = assemble_template(hits)
    stage_timings["result_aggregation"] = round(
        time.monotonic() - t0, 3
    )
    _notify("Result Aggregation", 1.0)

    # ------------------------------------------------------------------
    # Stage 5: SoA Extraction (PTCV-239, PTCV-250 progress)
    # Run SoA extractor using structured tables from ProtocolIndex.
    # Uses filter_soa_tables() on pre-extracted tables — no PDF
    # re-parsing needed.
    # ------------------------------------------------------------------
    _notify("SoA Extraction", 0.0)
    soa_result = None
    try:
        from ptcv.soa_extractor.table_bridge import filter_soa_tables
        from ptcv.soa_extractor.mapper import UsdmMapper
        from ptcv.soa_extractor.resolver import SynonymResolver
        from ptcv.extraction.models import ExtractedTable

        t0 = time.monotonic()

        # Convert ProtocolIndex.tables (dicts) to ExtractedTable objects
        extracted_tables = []
        for tbl in getattr(protocol_index, "tables", []):
            extracted_tables.append(ExtractedTable(
                run_id="query-pipeline",
                source_registry_id="",
                source_sha256="",
                page_number=tbl.get("page_number", 0),
                extractor_used=tbl.get("extractor_used", "pdfplumber_table"),
                table_index=tbl.get("table_index", 0),
                header_row=tbl.get("header_row", "[]"),
                data_rows=tbl.get("data_rows", "[]"),
            ))

        soa_tables = filter_soa_tables(extracted_tables)

        if soa_tables:
            mapper = UsdmMapper(resolver=SynonymResolver())
            timestamp = time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
            )
            epochs, timepoints, activities, instances, synonyms = (
                mapper.map(
                    soa_tables,
                    run_id="query-pipeline",
                    source_run_id="",
                    source_sha256="",
                    registry_id="",
                    timestamp=timestamp,
                )
            )
            soa_result = {
                "tables_found": len(soa_tables),
                "timepoint_count": len(timepoints),
                "activity_count": len(activities),
                "instance_count": len(instances),
                "epochs": epochs,
                "timepoints": timepoints,
                "activities": activities,
                "instances": instances,
                "synonyms": synonyms,
            }
            logger.info(
                "SoA extraction: %d tables, %d timepoints, "
                "%d activities",
                len(soa_tables), len(timepoints), len(activities),
            )

        stage_timings["soa_extraction"] = round(
            time.monotonic() - t0, 3
        )
    except Exception as exc:
        logger.debug(
            "SoA extraction in query pipeline failed: %s",
            exc, exc_info=True,
        )
    _notify("SoA Extraction", 1.0)

    # ------------------------------------------------------------------
    # Stage 6: SDTM Domain Spec Generation (PTCV-240, PTCV-250 progress)
    # Generate domain specs from SoA assessments + registry metadata.
    # ------------------------------------------------------------------
    _notify("SDTM Generation", 0.0)
    sdtm_result = None
    try:
        t0 = time.monotonic()

        from ptcv.sdtm.domain_spec_builder import build_domain_specs
        from ptcv.sdtm.ex_domain_builder import build_ex_domain_spec

        domain_specs = None
        ex_spec = None

        # Build observation domain specs from SoA tables
        tables_found: int = int(soa_result.get("tables_found", 0)) if soa_result else 0  # type: ignore[arg-type]
        if tables_found > 0:
            # Reconstruct RawSoaTable from SoA result
            from ptcv.soa_extractor.table_bridge import filter_soa_tables

            soa_tables = filter_soa_tables(extracted_tables)
            if soa_tables:
                domain_specs = build_domain_specs(soa_tables[0])

        # Build EX domain from registry metadata
        registry_id = Path(pdf_path).stem.split("_")[0]
        registry_cache_path = (
            Path(pdf_path).parent / "registry_cache" / f"{registry_id}.json"
        )
        if not registry_cache_path.exists():
            # Try standard location
            registry_cache_path = Path(
                "C:/Dev/PTCV/data/protocols/clinicaltrials"
                "/registry_cache"
            ) / f"{registry_id}.json"

        # Extract B.4/B.6 text from AssembledProtocol for fallback
        b4_text = ""
        for code in ("B.4", "B.6"):
            section = assembled.get_section(code)
            if section and section.hits:
                b4_text += "\n".join(
                    h.extracted_content for h in section.hits
                    if h.extracted_content
                )

        registry_meta = None
        if registry_cache_path.exists():
            import json as _json
            registry_meta = _json.loads(
                registry_cache_path.read_text(encoding="utf-8")
            )

        ex_spec = build_ex_domain_spec(
            registry_metadata=registry_meta,
            protocol_text=b4_text,
        )

        domain_count = len(domain_specs.specs) if domain_specs else 0  # type: ignore[union-attr]
        treatment_count = ex_spec.treatment_count if ex_spec else 0  # type: ignore[union-attr]

        sdtm_result = {
            "domain_specs": domain_specs,
            "ex_spec": ex_spec,
            "domain_count": domain_count,
            "treatment_count": treatment_count,
        }

        stage_timings["sdtm_generation"] = round(
            time.monotonic() - t0, 3
        )
        logger.info(
            "SDTM generation: %d observation domains, %d treatments",
            sdtm_result["domain_count"],
            sdtm_result["treatment_count"],
        )
    except Exception as exc:
        logger.debug(
            "SDTM generation in query pipeline failed: %s",
            exc, exc_info=True,
        )
    _notify("SDTM Generation", 1.0)

    # ------------------------------------------------------------------
    # Stage 7: Validation (PTCV-240, PTCV-250 progress)
    # Check required domain completeness.
    # ------------------------------------------------------------------
    _notify("Validation", 0.0)
    validation_result = None
    try:
        t0 = time.monotonic()

        from ptcv.sdtm.validation.required_domain_checker import (
            check_required_domains,
        )

        # Collect present domains from specs
        present_domains: list[str] = ["TS", "TV", "TA", "TE", "TI"]
        if sdtm_result and sdtm_result.get("domain_specs"):
            present_domains.extend(
                s.domain_code
                for s in sdtm_result["domain_specs"].specs
            )
        if sdtm_result and sdtm_result.get("ex_spec"):
            if sdtm_result["ex_spec"].treatment_count > 0:
                present_domains.append("EX")

        # Collect SoA assessment names for conditional checks
        soa_assessments: list[str] = []
        if soa_result and soa_result.get("activities"):
            soa_assessments = [
                a.activity_name
                for a in soa_result["activities"]
                if hasattr(a, "activity_name")
            ]

        domain_check = check_required_domains(
            domains_present=present_domains,
            soa_assessments=soa_assessments,
        )

        validation_result = {
            "domain_check": domain_check,
            "passed": domain_check.passed,
            "error_count": domain_check.error_count,
            "warning_count": domain_check.warning_count,
        }

        stage_timings["validation"] = round(
            time.monotonic() - t0, 3
        )
    except Exception as exc:
        logger.debug(
            "Validation in query pipeline failed: %s",
            exc, exc_info=True,
        )
    _notify("Validation", 1.0)

    logger.info("Pipeline stage timings: %s", stage_timings)

    return {
        "protocol_index": protocol_index,
        "match_result": match_result,
        "enriched_match_result": enriched_result,
        "extraction_result": extraction_result,
        "assembled": assembled,
        "assembled_markdown": assembled.to_markdown(),
        "coverage": assembled.coverage,
        "soa_result": soa_result,
        "sdtm_result": sdtm_result,
        "validation_result": validation_result,
        "stage_timings": stage_timings,
    }


def format_toc_tree(toc_entries: list[Any]) -> list[dict[str, Any]]:
    """Flatten TOC entries for tabular display.

    Args:
        toc_entries: List of ``TOCEntry`` dataclass instances.

    Returns:
        List of dicts with keys: ``level``, ``number``, ``title``,
        ``page``.
    """
    rows: list[dict[str, Any]] = []
    for entry in toc_entries:
        indent = "  " * (getattr(entry, "level", 1) - 1)
        rows.append({
            "level": getattr(entry, "level", 1),
            "number": getattr(entry, "number", ""),
            "title": f"{indent}{getattr(entry, 'title', '')}",
            "page": getattr(entry, "page_ref", 0),
        })
    return rows


def format_match_table(match_result: Any) -> list[dict[str, Any]]:
    """Format section mappings for tabular display.

    Args:
        match_result: ``MatchResult`` from ``SectionMatcher.match()``.

    Returns:
        List of dicts with keys: ``protocol_section``, ``ich_section``,
        ``score``, ``confidence``, ``method``.
    """
    rows: list[dict[str, Any]] = []
    for mapping in getattr(match_result, "mappings", []):
        matches = getattr(mapping, "matches", [])
        if matches:
            best = matches[0]
            rows.append({
                "protocol_section": (
                    f"{getattr(mapping, 'protocol_section_number', '')} "
                    f"{getattr(mapping, 'protocol_section_title', '')}"
                ).strip(),
                "ich_section": (
                    f"{getattr(best, 'ich_section_code', '')} "
                    f"{getattr(best, 'ich_section_name', '')}"
                ).strip(),
                "score": round(
                    getattr(best, "boosted_score", 0.0)
                    or getattr(best, "similarity_score", 0.0),
                    3,
                ),
                "confidence": getattr(
                    getattr(best, "confidence", None), "value", "LOW"
                ) if hasattr(getattr(best, "confidence", None), "value") else str(
                    getattr(best, "confidence", "LOW")
                ),
                "method": getattr(best, "match_method", ""),
            })
        else:
            rows.append({
                "protocol_section": (
                    f"{getattr(mapping, 'protocol_section_number', '')} "
                    f"{getattr(mapping, 'protocol_section_title', '')}"
                ).strip(),
                "ich_section": "(unmapped)",
                "score": 0.0,
                "confidence": "LOW",
                "method": "",
            })
    return rows


def format_subsection_match_table(
    enriched_result: Any,
) -> list[dict[str, Any]]:
    """Format sub-section matches for tabular display (PTCV-96).

    Args:
        enriched_result: ``EnrichedMatchResult`` from
            ``SummarizationMatcher.refine()``, or *None*.

    Returns:
        List of dicts with keys: ``protocol_section``,
        ``parent_section``, ``sub_section``, ``composite_score``,
        ``summarization_score``, ``confidence``, ``method``.
    """
    if enriched_result is None:
        return []

    rows: list[dict[str, Any]] = []
    for em in getattr(enriched_result, "enriched_mappings", []):
        proto = (
            f"{getattr(em, 'protocol_section_number', '')} "
            f"{getattr(em, 'protocol_section_title', '')}"
        ).strip()
        for sub in getattr(em, "sub_section_matches", []):
            conf = getattr(
                getattr(sub, "confidence", None), "value", "low"
            ) if hasattr(getattr(sub, "confidence", None), "value") else str(
                getattr(sub, "confidence", "low")
            )
            rows.append({
                "protocol_section": proto,
                "parent_section": getattr(
                    sub, "parent_section_code", ""
                ),
                "sub_section": (
                    f"{getattr(sub, 'sub_section_code', '')} "
                    f"{getattr(sub, 'sub_section_name', '')}"
                ).strip(),
                "composite_score": round(
                    getattr(sub, "composite_score", 0.0), 3
                ),
                "summarization_score": round(
                    getattr(sub, "summarization_score", -1.0), 3
                ),
                "confidence": conf,
                "method": getattr(sub, "match_method", ""),
            })
    return rows


def format_extraction_table(
    extraction_result: Any,
) -> list[dict[str, Any]]:
    """Format query extractions for tabular display.

    Args:
        extraction_result: ``ExtractionResult`` from
            ``QueryExtractor.extract()``.

    Returns:
        List of dicts with keys: ``query_id``, ``section``,
        ``content_preview``, ``confidence``, ``method``.
    """
    rows: list[dict[str, Any]] = []
    for ext in getattr(extraction_result, "extractions", []):
        content = getattr(ext, "content", "")
        preview = (
            content[:_CONTENT_PREVIEW_LEN] + "..."
            if len(content) > _CONTENT_PREVIEW_LEN
            else content
        )
        rows.append({
            "query_id": getattr(ext, "query_id", ""),
            "section": getattr(ext, "section_id", ""),
            "content_preview": preview,
            "confidence": format_confidence(
                getattr(ext, "confidence", 0.0)
            ),
            "method": getattr(ext, "extraction_method", ""),
        })
    return rows


def format_gap_table(extraction_result: Any) -> list[dict[str, Any]]:
    """Format extraction gaps for tabular display.

    Args:
        extraction_result: ``ExtractionResult`` from
            ``QueryExtractor.extract()``.

    Returns:
        List of dicts with keys: ``query_id``, ``section``, ``reason``.
    """
    rows: list[dict[str, Any]] = []
    for gap in getattr(extraction_result, "gaps", []):
        rows.append({
            "query_id": getattr(gap, "query_id", ""),
            "section": getattr(gap, "section_id", ""),
            "reason": getattr(gap, "reason", "unknown"),
        })
    return rows


def count_extraction_methods(
    extraction_result: Any,
) -> dict[str, int]:
    """Count extractions by method type (PTCV-103).

    Args:
        extraction_result: ``ExtractionResult`` from
            ``QueryExtractor.extract()``.

    Returns:
        Dict mapping method name to count, e.g.
        ``{"text_short": 5, "llm_transform": 3}``.
    """
    counts: dict[str, int] = {}
    for ext in getattr(extraction_result, "extractions", []):
        method = getattr(ext, "extraction_method", "unknown")
        counts[method] = counts.get(method, 0) + 1
    return counts


def format_provenance_badge(section: Any) -> dict[str, str]:
    """Format provenance metadata for a classified section (PTCV-179).

    Args:
        section: ``AssembledSection`` with provenance fields.

    Returns:
        Dict with keys: ``extraction_method``, ``classification_method``,
        ``confidence_label``, ``confidence_color``.
    """
    ext = getattr(section, "extraction_method", "") or ""
    cls = getattr(section, "classification_method", "") or ""
    avg = getattr(section, "average_confidence", 0.0)
    return {
        "extraction_method": ext,
        "classification_method": cls,
        "confidence_label": confidence_label(avg),
        "confidence_color": confidence_color(avg),
    }


def format_pipeline_comparison_rows(
    query_assembled: Any,
    classified_assembled: Any,
) -> list[dict[str, Any]]:
    """Build per-section query-vs-classified comparison rows (PTCV-179).

    For each ICH section populated in either pipeline's output, pairs
    the query pipeline extracted content with the classified pipeline
    extracted content for side-by-side comparison.

    Args:
        query_assembled: ``AssembledProtocol`` from query pipeline.
        classified_assembled: ``AssembledProtocol`` from classified
            pipeline.

    Returns:
        List of dicts with keys: ``ich_code``, ``ich_name``,
        ``query_text``, ``query_confidence``, ``classified_text``,
        ``classified_confidence``, ``classified_method``,
        ``confidence_delta``.
    """
    # Inline sort key to avoid ptcv.ich_parser __init__ import chain.
    def _sort_key(code: str) -> list[int]:
        return [int(p) if p.isdigit() else 0 for p in code.split(".")]

    # Collect all section codes present in either assembled output.
    codes: set[str] = set()
    for src in (query_assembled, classified_assembled):
        if src is None:
            continue
        for sec in getattr(src, "sections", []):
            if getattr(sec, "populated", False):
                codes.add(getattr(sec, "section_code", ""))

    rows: list[dict[str, Any]] = []
    for code in sorted(codes, key=_sort_key):
        q_sec = (
            query_assembled.get_section(code)
            if query_assembled is not None else None
        )
        c_sec = (
            classified_assembled.get_section(code)
            if classified_assembled is not None else None
        )

        q_text = ""
        q_conf = 0.0
        if q_sec and getattr(q_sec, "populated", False):
            parts = [
                getattr(h, "extracted_content", "").strip()
                for h in getattr(q_sec, "hits", [])
                if getattr(h, "extracted_content", "")
            ]
            q_text = "\n\n".join(parts)
            q_conf = getattr(q_sec, "average_confidence", 0.0)

        c_text = ""
        c_conf = 0.0
        c_method = ""
        if c_sec and getattr(c_sec, "populated", False):
            parts = [
                getattr(h, "extracted_content", "").strip()
                for h in getattr(c_sec, "hits", [])
                if getattr(h, "extracted_content", "")
            ]
            c_text = "\n\n".join(parts)
            c_conf = getattr(c_sec, "average_confidence", 0.0)
            ext = getattr(c_sec, "extraction_method", "") or ""
            cls = getattr(c_sec, "classification_method", "") or ""
            method_parts = [p for p in (ext, cls) if p]
            c_method = " | ".join(method_parts)

        if not q_text and not c_text:
            continue

        rows.append({
            "ich_code": code,
            "ich_name": getattr(
                q_sec or c_sec, "section_name", code,
            ),
            "query_text": q_text,
            "query_confidence": round(q_conf, 2),
            "classified_text": c_text,
            "classified_confidence": round(c_conf, 2),
            "classified_method": c_method,
            "confidence_delta": round(c_conf - q_conf, 2),
        })

    return rows


def format_coverage_metrics(coverage: Any) -> dict[str, Any]:
    """Extract key coverage metrics from a CoverageReport.

    Args:
        coverage: ``CoverageReport`` from ``AssembledProtocol.coverage``.

    Returns:
        Dict with keys: ``total``, ``populated``, ``gaps``,
        ``avg_confidence``, ``high_pct``, ``review_pct``, ``low_pct``.
    """
    total = getattr(coverage, "total_sections", 0)
    populated = getattr(coverage, "populated_count", 0)
    gaps = getattr(coverage, "gap_count", 0)
    avg_conf = getattr(coverage, "avg_confidence", 0.0)
    high = getattr(coverage, "high_confidence_count", 0)
    review = getattr(coverage, "review_confidence_count", 0)
    low = getattr(coverage, "low_confidence_count", 0)

    return {
        "total": total,
        "populated": populated,
        "gaps": gaps,
        "avg_confidence": round(avg_conf, 3),
        "high_pct": round(high / total * 100, 1) if total else 0.0,
        "review_pct": round(review / total * 100, 1) if total else 0.0,
        "low_pct": round(low / total * 100, 1) if total else 0.0,
    }


def format_comparison_rows(
    assembled: Any,
    protocol_index: Any,
    match_result: Any,
) -> list[dict[str, Any]]:
    """Build per-section original-vs-extracted comparison rows (PTCV-141).

    For each populated section in the assembled template, looks up the
    original protocol text via the section matching mappings and pairs
    it with the extracted/transformed content.

    Args:
        assembled: ``AssembledProtocol`` with populated sections.
        protocol_index: ``ProtocolIndex`` with ``content_spans``.
        match_result: ``MatchResult`` with section mappings.

    Returns:
        List of dicts with keys: ``ich_code``, ``ich_name``,
        ``protocol_section``, ``original_text``, ``extracted_text``,
        ``confidence``, ``query_count``.
    """
    # Build ICH code → protocol section number lookup from mappings.
    ich_to_proto: dict[str, list[str]] = {}
    for mapping in getattr(match_result, "mappings", []):
        matches = getattr(mapping, "matches", [])
        if matches:
            code = getattr(matches[0], "ich_section_code", "")
            proto_num = getattr(
                mapping, "protocol_section_number", "",
            )
            if code and proto_num:
                ich_to_proto.setdefault(code, []).append(proto_num)

    rows: list[dict[str, Any]] = []
    for section in getattr(assembled, "sections", []):
        code = getattr(section, "section_code", "")
        name = getattr(section, "section_name", "")
        if not getattr(section, "populated", False):
            continue

        # Gather original text from mapped protocol sections.
        proto_nums = ich_to_proto.get(code, [])
        original_parts: list[str] = []
        for pn in proto_nums:
            text = protocol_index.get_section_text(pn)
            if text:
                original_parts.append(text.strip())
        original_text = "\n\n".join(original_parts)

        # Gather extracted content from hits.
        extracted_parts: list[str] = []
        for hit in getattr(section, "hits", []):
            content = getattr(hit, "extracted_content", "")
            if content:
                extracted_parts.append(content.strip())
        extracted_text = "\n\n".join(extracted_parts)

        rows.append({
            "ich_code": code,
            "ich_name": name,
            "protocol_section": ", ".join(proto_nums) or "(none)",
            "original_text": original_text,
            "extracted_text": extracted_text,
            "confidence": round(
                getattr(section, "average_confidence", 0.0), 2,
            ),
            "query_count": len(
                getattr(section, "hits", []),
            ),
        })

    return rows


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def render_query_pipeline(file_path: Path, file_sha: str) -> None:
    """Render the Query Pipeline tab.

    Runs the full query-driven pipeline on the selected PDF and
    displays: Protocol Index, Section Matching, Extraction Results,
    and Assembled Template.

    Args:
        file_path: Absolute path to the selected PDF.
        file_sha: SHA-256 hex digest for session caching.
    """
    import streamlit as st

    # --- Options (PTCV-103, PTCV-111) ---
    _has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    transform_on = st.checkbox(
        "Enable LLM Transformation",
        value=_has_api_key,
        help=(
            "Reframe scoped extractions using Claude Sonnet to "
            "directly address each Appendix B query. "
            "Requires ANTHROPIC_API_KEY."
        ),
        key="chk_enable_transform",
    )

    if transform_on and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning(
            "LLM Transformation requires an Anthropic API key. "
            "Set **ANTHROPIC_API_KEY** in your environment or "
            "`.secrets` file. Pipeline will run in verbatim mode."
        )

    # Cache init — version-stamped key invalidates stale entries.
    # Include transform toggle in cache key so toggling re-runs.
    cache_key = (
        f"{file_sha}_v{_PIPELINE_CACHE_VERSION}"
        f"_t{int(transform_on)}"
    )
    if "query_cache" not in st.session_state:
        st.session_state["query_cache"] = {}

    cached = st.session_state["query_cache"].get(cache_key)

    if cached:
        st.success("Query pipeline complete. Results below.")
    else:
        if st.button(
            "Run Query Pipeline",
            type="primary",
            key="btn_query_pipeline",
        ):
            # --- Stage progress bars (PTCV-123) ---
            bars: dict[str, Any] = {}
            for stage_name in PIPELINE_STAGES:
                bars[stage_name] = st.progress(
                    0, text=stage_name,
                )

            stage_start_times: dict[str, float] = {}

            def _on_stage(
                stage: str, progress: float,
                done: int = 0, total: int = 0,
            ) -> None:
                if stage not in bars:
                    return
                # Track stage start time (PTCV-124)
                if progress == 0.0:
                    stage_start_times[stage] = time.monotonic()
                elapsed = time.monotonic() - stage_start_times.get(
                    stage, time.monotonic()
                )
                elapsed_str = f"{elapsed:.1f}s"
                pct = int(progress * 100)
                if progress >= 1.0:
                    bars[stage].progress(
                        100,
                        text=(
                            f":white_check_mark: {stage}"
                            f" \u2014 {elapsed_str}"
                        ),
                    )
                elif 0.0 < progress < 1.0:
                    if total > 0:
                        bars[stage].progress(
                            max(pct, 3),
                            text=(
                                f":hourglass_flowing_sand: "
                                f"{stage} \u2014 "
                                f"{done}/{total} ({pct}%) "
                                f"\u2014 {elapsed_str}"
                            ),
                        )
                    else:
                        bars[stage].progress(
                            max(pct, 3),
                            text=(
                                f":hourglass_flowing_sand: "
                                f"{stage}... {pct}%"
                            ),
                        )
                else:
                    bars[stage].progress(
                        max(pct, 3),
                        text=f":hourglass_flowing_sand: {stage}...",
                    )

            t0 = time.monotonic()
            try:
                result = run_query_pipeline(
                    str(file_path),
                    enable_transformation=transform_on,
                    anthropic_api_key=os.environ.get(
                        "ANTHROPIC_API_KEY"
                    ),
                    progress_callback=_on_stage,
                )
                elapsed = time.monotonic() - t0
                st.session_state["query_cache"][cache_key] = result
                cached = result

                # PTCV-250: Propagate Stage 5-7 results to
                # downstream tab session state so SoA/SDTM tabs
                # auto-populate without re-extraction.
                if result.get("soa_result"):
                    soa_data = result["soa_result"]
                    soa_data["_source"] = "query_pipeline"
                    st.session_state.setdefault("soa_cache", {})[
                        file_sha
                    ] = soa_data
                    logger.info(
                        "PTCV-250: Propagated SoA result to soa_cache"
                    )
                if result.get("sdtm_result"):
                    st.session_state.setdefault("sdtm_cache", {})[
                        file_sha
                    ] = result["sdtm_result"]
                    logger.info(
                        "PTCV-250: Propagated SDTM result to sdtm_cache"
                    )
                if result.get("validation_result"):
                    st.session_state.setdefault(
                        "validation_cache", {},
                    )[file_sha] = result["validation_result"]
                    logger.info(
                        "PTCV-250: Propagated validation result"
                    )

                # Persist artifacts to disk (PTCV-126: ALCOA++)
                try:
                    from ptcv.ui.query_persistence import (
                        save_query_artifacts,
                    )
                    _gw = st.session_state.get("gateway")
                    if _gw is not None:
                        _stem = Path(file_path).stem
                        _reg_id = _stem.split("_", 1)[0]
                        save_query_artifacts(
                            _gw, file_sha, result,
                            registry_id=_reg_id,
                        )
                except Exception:
                    logger.warning(
                        "Failed to persist query artifacts",
                        exc_info=True,
                    )

                # Session checkpoint (PTCV-126)
                try:
                    from ptcv.ui.checkpoint_manager import (
                        save_checkpoint,
                    )
                    _cp_root = getattr(
                        _gw, "root", Path("data"),
                    ) if _gw else Path("data")
                    save_checkpoint(
                        _cp_root, file_sha,
                        "query_cache", result,
                    )
                except Exception:
                    logger.warning(
                        "Failed to save query checkpoint",
                        exc_info=True,
                    )

                # Force rerun so that tabs rendered earlier in
                # the script (e.g. SoA & SDTM) pick up the
                # freshly-cached query results (PTCV-130).
                st.rerun()
            except Exception:
                elapsed = time.monotonic() - t0
                st.error(
                    f"Query Pipeline: Error ({elapsed:.1f}s)"
                )
                st.code(traceback.format_exc(), language="text")
                return

    if not cached:
        st.info(
            "Click **Run Query Pipeline** to extract and classify "
            "this protocol using the query-driven approach."
        )
        return

    # --- Protocol Index ---
    protocol_index = cached["protocol_index"]
    st.subheader("Protocol Index")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages", getattr(protocol_index, "page_count", 0))
    with col2:
        st.metric(
            "TOC entries",
            len(getattr(protocol_index, "toc_entries", [])),
        )
    with col3:
        st.metric(
            "Section headers",
            len(getattr(protocol_index, "section_headers", [])),
        )

    toc_rows = format_toc_tree(
        getattr(protocol_index, "toc_entries", [])
    )
    if toc_rows:
        with st.expander("Table of Contents", expanded=False):
            st.dataframe(toc_rows, use_container_width=True)

    st.divider()

    # --- Section Matching ---
    match_result = cached["match_result"]
    st.subheader("Section Matching")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Auto-mapped",
            getattr(match_result, "auto_mapped_count", 0),
        )
    with col2:
        st.metric("Review", getattr(match_result, "review_count", 0))
    with col3:
        st.metric(
            "Unmapped", getattr(match_result, "unmapped_count", 0),
        )

    match_rows = format_match_table(match_result)
    if match_rows:
        with st.expander("Section Mappings", expanded=False):
            st.dataframe(match_rows, use_container_width=True)

    # --- Source Provenance Matrix (PTCV-269: Stage 1-2 output) ---
    from ptcv.ui.components.provenance_matrix import (
        build_provenance_matrix,
        render_provenance_matrix,
    )

    _prov_rows = build_provenance_matrix(cached)
    render_provenance_matrix(_prov_rows)

    # --- Section Coverage Diagram (PTCV-270: Stage 2 visual) ---
    _assembled = cached.get("assembled")
    if _assembled is not None:
        from ptcv.ui.components.section_coverage_diagram import (
            render_coverage_diagram,
        )
        render_coverage_diagram(_assembled)

    st.divider()

    # --- Sub-Section Matching (PTCV-96) ---
    enriched_result = cached.get("enriched_match_result")
    if enriched_result is not None:
        st.subheader("Sub-Section Matching")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Sub-section auto-map",
                f"{getattr(enriched_result, 'sub_section_auto_map_rate', 0.0):.0%}",
            )
        with col2:
            st.metric(
                "LLM calls",
                getattr(enriched_result, "llm_calls_made", 0),
            )
        with col3:
            mode = (
                "Fallback" if getattr(enriched_result, "llm_fallback", True)
                else "Active"
            )
            st.metric("LLM mode", mode)

        sub_rows = format_subsection_match_table(enriched_result)
        if sub_rows:
            with st.expander(
                "Sub-Section Mappings", expanded=False
            ):
                st.dataframe(sub_rows, use_container_width=True)

        st.divider()

    # --- Extraction Results ---
    extraction_result = cached["extraction_result"]
    st.subheader("Query Extraction")
    method_counts = count_extraction_methods(extraction_result)
    transformed_count = method_counts.get("llm_transform", 0)
    verbatim_count = sum(
        v for k, v in method_counts.items() if k != "llm_transform"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Answered",
            getattr(extraction_result, "answered_queries", 0),
        )
    with col2:
        st.metric(
            "Total queries",
            getattr(extraction_result, "total_queries", 0),
        )
    with col3:
        coverage = getattr(extraction_result, "coverage", 0.0)
        st.metric("Coverage", f"{coverage:.0%}")

    # Transformation metrics (PTCV-103)
    if transformed_count > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LLM Transformed", transformed_count)
        with col2:
            st.metric("Verbatim", verbatim_count)

    ext_rows = format_extraction_table(extraction_result)
    if ext_rows:
        with st.expander("Extractions", expanded=False):
            st.dataframe(ext_rows, use_container_width=True)

    gap_rows = format_gap_table(extraction_result)
    if gap_rows:
        with st.expander(f"Gaps ({len(gap_rows)})", expanded=False):
            st.dataframe(gap_rows, use_container_width=True)

    st.divider()

    # --- Assembled Template ---
    assembled = cached["assembled"]
    st.subheader("Assembled Appendix B Template")

    cov = cached["coverage"]
    cov_metrics = format_coverage_metrics(cov)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sections", cov_metrics["total"])
    with col2:
        st.metric("Populated", cov_metrics["populated"])
    with col3:
        st.metric("Gaps", cov_metrics["gaps"])
    with col4:
        st.metric(
            "Avg confidence",
            f"{cov_metrics['avg_confidence']:.2f}",
        )

    # --- Source Provenance Matrix (PTCV-251) ---
    try:
        from ptcv.ui.components.provenance_matrix import (
            build_provenance_matrix,
            render_provenance_matrix,
        )

        _stem = Path(file_path).stem if "file_path" in dir() else ""
        _nct = ""
        _m = re.search(r"NCT\d{8}", _stem) if _stem else None
        if _m:
            _nct = _m.group(0)

        prov_rows = build_provenance_matrix(cached, nct_id=_nct)
        with st.expander("Source Provenance", expanded=False):
            render_provenance_matrix(prov_rows)
    except Exception:
        logger.debug(
            "Provenance matrix render failed", exc_info=True
        )

    md = cached["assembled_markdown"]
    with st.expander("Template Preview", expanded=True):
        with st.container(height=500):
            st.markdown(md)

    st.download_button(
        label="Download as .md",
        data=md,
        file_name="assembled_appendix_b.md",
        mime="text/markdown",
        key="btn_dl_assembled",
    )

    # --- Content Comparison (PTCV-141, PTCV-179) ---
    st.divider()
    st.subheader("Content Comparison")

    # Pipeline comparison mode toggle (PTCV-179).
    from ptcv.soa_extractor.query_bridge import (
        get_classified_assembled_protocol,
        has_classified_results,
    )
    _has_classified = has_classified_results(
        st.session_state, file_sha,
    )
    comparison_mode = "Original vs Extracted"
    if _has_classified:
        comparison_mode = st.radio(
            "Comparison mode",
            ["Original vs Extracted", "Pipeline Comparison"],
            key="radio_comparison_mode",
            horizontal=True,
        )
    else:
        st.caption(
            "Pipeline Comparison mode available when classified "
            "pipeline results exist."
        )

    if comparison_mode == "Pipeline Comparison":
        classified_assembled = get_classified_assembled_protocol(
            st.session_state, file_sha,
        )
        pipeline_rows = format_pipeline_comparison_rows(
            assembled, classified_assembled,
        )
        if not pipeline_rows:
            st.info("No sections to compare across pipelines.")
        else:
            pl_labels = [
                f"{r['ich_code']} \u2014 {r['ich_name']}"
                for r in pipeline_rows
            ]
            pl_selected = st.selectbox(
                "Select section to compare",
                options=range(len(pl_labels)),
                format_func=lambda i: pl_labels[i],
                key="sb_pipeline_compare_section",
            )
            pl_row = pipeline_rows[pl_selected]

            delta = pl_row["confidence_delta"]
            delta_label = f"{delta:+.2f}" if delta != 0 else "equal"
            st.caption(
                f"Confidence delta: **{delta_label}** "
                f"(Query: {pl_row['query_confidence']:.2f} | "
                f"Classified: "
                f"{pl_row['classified_confidence']:.2f})"
            )
            if pl_row.get("classified_method"):
                st.caption(
                    f"Classified provenance: "
                    f"**{pl_row['classified_method']}**"
                )

            col_q, col_c = st.columns(2)
            with col_q:
                st.markdown("**Query Pipeline**")
                with st.container(height=400):
                    if pl_row["query_text"]:
                        st.markdown(pl_row["query_text"])
                    else:
                        st.info("No query pipeline content.")
            with col_c:
                st.markdown("**Classified Pipeline**")
                with st.container(height=400):
                    if pl_row["classified_text"]:
                        st.markdown(pl_row["classified_text"])
                    else:
                        st.info(
                            "No classified pipeline content."
                        )
    else:
        # Original vs Extracted (existing PTCV-141 logic).
        comparison_rows = format_comparison_rows(
            assembled, protocol_index, match_result,
        )
        if not comparison_rows:
            st.info("No populated sections for comparison.")
        else:
            section_labels = [
                f"{r['ich_code']} \u2014 {r['ich_name']}"
                for r in comparison_rows
            ]
            selected = st.selectbox(
                "Select section to compare",
                options=range(len(section_labels)),
                format_func=lambda i: section_labels[i],
                key="sb_compare_section",
            )
            row = comparison_rows[selected]

            st.caption(
                f"Protocol section(s): "
                f"**{row['protocol_section']}** "
                f"| Confidence: **{row['confidence']:.2f}** "
                f"| Queries: **{row['query_count']}**"
            )

            # Provenance metadata (PTCV-179).
            section_obj = assembled.get_section(
                row["ich_code"],
            )
            if section_obj and (
                getattr(section_obj, "extraction_method", "")
                or getattr(
                    section_obj, "classification_method", "",
                )
            ):
                badge = format_provenance_badge(section_obj)
                prov_parts = []
                if badge["extraction_method"]:
                    prov_parts.append(badge["extraction_method"])
                if badge["classification_method"]:
                    prov_parts.append(
                        badge["classification_method"],
                    )
                st.caption(
                    f"Provenance: {' | '.join(prov_parts)} | "
                    f"Confidence: "
                    f":{badge['confidence_color']}["
                    f"{badge['confidence_label']}]"
                )

            col_orig, col_ext = st.columns(2)
            with col_orig:
                st.markdown("**Original (PDF)**")
                with st.container(height=400):
                    if row["original_text"]:
                        st.text(row["original_text"])
                    else:
                        st.info(
                            "No original text mapped for "
                            "this section."
                        )
            with col_ext:
                st.markdown("**Extracted**")
                with st.container(height=400):
                    if row["extracted_text"]:
                        st.markdown(row["extracted_text"])
                    else:
                        st.info("No extracted content.")

    # --- Unified Pipeline Progress Summary (PTCV-179) ---
    st.divider()
    st.subheader("Pipeline Stage Summary")

    from ptcv.ui.components.progress_tracker import (
        format_stage_display,
        map_classified_stages_to_unified,
        map_query_stages_to_unified,
        merge_pipeline_stages,
    )

    query_stages = map_query_stages_to_unified(
        cached.get("stage_timings", {}),
    )
    classified_stages = None
    if _has_classified:
        _cl_cache = st.session_state.get(
            "classified_cache", {},
        ).get(file_sha, {})
        classified_stages = map_classified_stages_to_unified(
            _cl_cache.get("cascade_stats"),
            _cl_cache.get("stage_timings"),
        )
    merged = merge_pipeline_stages(query_stages, classified_stages)

    # Render 8 stages as 4x2 grid.
    for row_start in (0, 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(merged):
                break
            stage = merged[idx]
            display = format_stage_display(stage)
            with col:
                st.markdown(
                    f"{display['icon']} **{display['label']}**",
                )
                if display["detail_text"]:
                    st.caption(display["detail_text"])
