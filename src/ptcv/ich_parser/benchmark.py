"""Benchmark query-driven vs text-first classification (PTCV-93).

Runs both pipelines on the same protocol corpus and produces a
comparative report with five metrics:

1. **Content coverage** — % of protocol text allocated to an
   Appendix B section (char-based via span mapper).
2. **Section accuracy** — fraction of sections assigned to the
   correct Appendix B code (agreement or ground truth).
3. **Coherence score** — average extraction span length (longer =
   more coherent extractions).
4. **Gap rate** — fraction of 16 Appendix B sections with no
   extracted content.
5. **Confidence distribution** — % of extractions at high / medium /
   low confidence tiers.

Input: same protocol PDF processed by both pipelines.
Output: per-protocol and aggregate comparative report.

Risk tier: LOW — read-only analysis, no I/O beyond PDF reads.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Sequence

from ptcv.annotations.span_mapper import (
    TextSpan,
    compute_coverage,
    map_sections_to_spans,
)
from ptcv.ich_parser.classifier import RuleBasedClassifier
from ptcv.ich_parser.query_schema import section_sort_key
from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.query_extractor import (
    ExtractionResult,
    QueryExtractor,
)
from ptcv.ich_parser.section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMatcher,
)
from ptcv.ich_parser.template_assembler import (
    AssembledProtocol,
    QueryExtractionHit,
    SourceReference,
    assemble_template,
)
from ptcv.ich_parser.toc_extractor import (
    ProtocolIndex,
    extract_protocol_index,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APPENDIX_B_SECTION_COUNT = 16

# Confidence tier thresholds (same as template_assembler).
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.70

# Recommendation thresholds.
COVERAGE_DELTA_THRESHOLD = 30.0  # percentage points
COHERENCE_RATIO_THRESHOLD = 2.0
SECTION_ACCURACY_THRESHOLD = 0.80
HIGH_CONFIDENCE_PCT_THRESHOLD = 50.0
GAP_RATE_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PipelineMetrics:
    """Metrics from running one pipeline on one protocol.

    Attributes:
        coverage_pct: % of protocol text allocated to Appendix B.
        avg_confidence: Mean confidence across extractions.
        avg_span_length: Mean character length per classified span.
        gap_rate: Fraction of 16 sections with no content (0.0–1.0).
        section_count: Number of Appendix B sections populated.
        section_codes: Set of populated Appendix B section codes.
        high_confidence_pct: % of extractions with confidence >= 0.85.
        medium_confidence_pct: % at 0.70–0.85.
        low_confidence_pct: % below 0.70.
        total_chars_classified: Characters allocated to sections.
        total_chars: Total protocol text length.
    """

    coverage_pct: float
    avg_confidence: float
    avg_span_length: float
    gap_rate: float
    section_count: int
    section_codes: set[str]
    high_confidence_pct: float
    medium_confidence_pct: float
    low_confidence_pct: float
    total_chars_classified: int
    total_chars: int


@dataclasses.dataclass
class ProtocolBenchmark:
    """Benchmark result for one protocol.

    Attributes:
        nct_id: Registry identifier.
        pdf_path: Path to the source PDF.
        text_first: Metrics from the text-first pipeline (or *None*).
        query_driven: Metrics from the query-driven pipeline (or *None*).
        coverage_delta: query_driven.coverage_pct minus
            text_first.coverage_pct (percentage points).
        coherence_ratio: query_driven.avg_span_length divided by
            text_first.avg_span_length.
        section_agreement: Jaccard similarity of populated section
            codes between the two pipelines.
        error: Non-empty if either pipeline failed.
    """

    nct_id: str
    pdf_path: str
    text_first: Optional[PipelineMetrics]
    query_driven: Optional[PipelineMetrics]
    coverage_delta: float
    coherence_ratio: float
    section_agreement: float
    error: str


@dataclasses.dataclass
class BenchmarkReport:
    """Aggregate benchmark report across the protocol corpus.

    Attributes:
        protocols: Per-protocol results.
        aggregate_text_first: Averaged text-first metrics.
        aggregate_query_driven: Averaged query-driven metrics.
        avg_coverage_delta: Mean coverage delta across protocols.
        avg_coherence_ratio: Mean coherence ratio.
        avg_section_agreement: Mean section agreement.
        recommendation: ``"replace"``, ``"supplement"``, or
            ``"keep_text_first"``.
        recommendation_reason: Human-readable justification.
    """

    protocols: list[ProtocolBenchmark]
    aggregate_text_first: Optional[PipelineMetrics]
    aggregate_query_driven: Optional[PipelineMetrics]
    avg_coverage_delta: float
    avg_coherence_ratio: float
    avg_section_agreement: float
    recommendation: str
    recommendation_reason: str

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        return _report_to_dict(self)

    def to_markdown(self) -> str:
        """Render as a Markdown document."""
        return _report_to_markdown(self)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------


def run_text_first(
    full_text: str,
    nct_id: str,
) -> PipelineMetrics:
    """Run the text-first (RuleBasedClassifier + span mapper) pipeline.

    Args:
        full_text: Complete protocol text.
        nct_id: Registry identifier for the protocol.

    Returns:
        :class:`PipelineMetrics` from the text-first pipeline.
    """
    run_id = str(uuid.uuid4())
    classifier = RuleBasedClassifier()
    sections = classifier.classify(
        full_text, nct_id, run_id, "", "",
    )

    return _metrics_from_sections(sections, full_text)


def _metrics_from_sections(
    sections: list[IchSection],
    full_text: str,
) -> PipelineMetrics:
    """Compute :class:`PipelineMetrics` from classified sections."""
    spans = map_sections_to_spans(full_text, sections)
    cov = compute_coverage(spans, len(full_text))

    classified_spans = [s for s in spans if s.classified]
    total_classified = len(classified_spans)

    avg_span = (
        sum(s.length for s in classified_spans) / total_classified
        if total_classified
        else 0.0
    )

    section_codes = {s.section_code for s in classified_spans}
    gap_rate = (
        1.0 - len(section_codes) / APPENDIX_B_SECTION_COUNT
    )

    confidences = [s.confidence for s in classified_spans]
    avg_conf = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )
    high = sum(1 for c in confidences if c >= HIGH_CONFIDENCE)
    medium = sum(
        1 for c in confidences
        if LOW_CONFIDENCE <= c < HIGH_CONFIDENCE
    )
    low = sum(1 for c in confidences if c < LOW_CONFIDENCE)
    total_conf = len(confidences) or 1

    return PipelineMetrics(
        coverage_pct=cov["coverage_pct"],
        avg_confidence=round(avg_conf, 4),
        avg_span_length=round(avg_span, 1),
        gap_rate=round(gap_rate, 4),
        section_count=len(section_codes),
        section_codes=section_codes,
        high_confidence_pct=round(high / total_conf * 100, 1),
        medium_confidence_pct=round(medium / total_conf * 100, 1),
        low_confidence_pct=round(low / total_conf * 100, 1),
        total_chars_classified=cov["classified_chars"],
        total_chars=len(full_text),
    )


def run_query_driven(
    protocol_index: ProtocolIndex,
) -> PipelineMetrics:
    """Run the query-driven (matcher + extractor + assembler) pipeline.

    Args:
        protocol_index: Navigable document index from TOC extractor.

    Returns:
        :class:`PipelineMetrics` from the query-driven pipeline.
    """
    matcher = SectionMatcher()
    match_result = matcher.match(protocol_index)
    extractor = QueryExtractor()
    extraction = extractor.extract(protocol_index, match_result)
    assembled = _assemble_from_extraction(extraction)

    return _metrics_from_query_driven(
        match_result, extraction, assembled, protocol_index,
    )


def _assemble_from_extraction(
    extraction: ExtractionResult,
) -> AssembledProtocol:
    """Convert :class:`ExtractionResult` to :class:`AssembledProtocol`."""
    hits: list[QueryExtractionHit] = []
    for e in extraction.extractions:
        parent = _parent_section(e.section_id)
        hits.append(
            QueryExtractionHit(
                query_id=e.query_id,
                section_id=e.section_id,
                parent_section=parent,
                query_text="",
                extracted_content=e.content,
                confidence=e.confidence,
                source=SourceReference(),
            )
        )
    return assemble_template(hits)


def _parent_section(section_id: str) -> str:
    """Derive the parent section code from a sub-section id.

    ``"B.1.1"`` → ``"B.1"``, ``"B.10.2"`` → ``"B.10"``.
    """
    parts = section_id.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return section_id


def _metrics_from_query_driven(
    match_result: MatchResult,
    extraction: ExtractionResult,
    assembled: AssembledProtocol,
    protocol_index: ProtocolIndex,
) -> PipelineMetrics:
    """Compute metrics from the query-driven pipeline outputs."""
    total_text = len(protocol_index.full_text) or 1

    # Character coverage: sum content_spans for matched sections.
    matched_section_numbers: set[str] = set()
    for mapping in match_result.mappings:
        if mapping.matches:
            matched_section_numbers.add(
                mapping.protocol_section_number
            )

    classified_chars = sum(
        len(protocol_index.content_spans.get(s, ""))
        for s in matched_section_numbers
    )
    coverage_pct = classified_chars / total_text * 100

    # Avg span length: mean content_span length per matched section.
    span_lengths = [
        len(protocol_index.content_spans.get(s, ""))
        for s in matched_section_numbers
        if protocol_index.content_spans.get(s, "")
    ]
    avg_span = (
        sum(span_lengths) / len(span_lengths)
        if span_lengths
        else 0.0
    )

    # Section-level coverage from assembled protocol.
    cov = assembled.coverage
    section_codes = {
        s.section_code
        for s in assembled.sections
        if s.populated
    }
    gap_rate = cov.gap_count / APPENDIX_B_SECTION_COUNT

    # Confidence from extractions.
    confidences = [e.confidence for e in extraction.extractions]
    avg_conf = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )
    high = sum(1 for c in confidences if c >= HIGH_CONFIDENCE)
    medium = sum(
        1 for c in confidences
        if LOW_CONFIDENCE <= c < HIGH_CONFIDENCE
    )
    low = sum(1 for c in confidences if c < LOW_CONFIDENCE)
    total_conf = len(confidences) or 1

    return PipelineMetrics(
        coverage_pct=round(coverage_pct, 2),
        avg_confidence=round(avg_conf, 4),
        avg_span_length=round(avg_span, 1),
        gap_rate=round(gap_rate, 4),
        section_count=len(section_codes),
        section_codes=section_codes,
        high_confidence_pct=round(high / total_conf * 100, 1),
        medium_confidence_pct=round(medium / total_conf * 100, 1),
        low_confidence_pct=round(low / total_conf * 100, 1),
        total_chars_classified=classified_chars,
        total_chars=total_text,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compute_section_agreement(
    tf: PipelineMetrics,
    qd: PipelineMetrics,
) -> float:
    """Jaccard similarity of populated section codes."""
    union = tf.section_codes | qd.section_codes
    if not union:
        return 0.0
    intersection = tf.section_codes & qd.section_codes
    return len(intersection) / len(union)


def compute_section_accuracy(
    predicted_codes: set[str],
    ground_truth_codes: set[str],
) -> float:
    """Fraction of predicted codes present in ground truth.

    Uses precision: |predicted ∩ truth| / |predicted|.
    """
    if not predicted_codes:
        return 0.0
    correct = predicted_codes & ground_truth_codes
    return len(correct) / len(predicted_codes)


def compare_protocol(
    nct_id: str,
    pdf_path: str,
    text_first: Optional[PipelineMetrics],
    query_driven: Optional[PipelineMetrics],
) -> ProtocolBenchmark:
    """Build a :class:`ProtocolBenchmark` from two pipeline runs."""
    error = ""
    if text_first is None and query_driven is None:
        error = "Both pipelines failed"
    elif text_first is None:
        error = "Text-first pipeline failed"
    elif query_driven is None:
        error = "Query-driven pipeline failed"

    coverage_delta = 0.0
    coherence_ratio = 0.0
    agreement = 0.0

    if text_first and query_driven:
        coverage_delta = (
            query_driven.coverage_pct - text_first.coverage_pct
        )
        if text_first.avg_span_length > 0:
            coherence_ratio = (
                query_driven.avg_span_length
                / text_first.avg_span_length
            )
        agreement = compute_section_agreement(
            text_first, query_driven
        )

    return ProtocolBenchmark(
        nct_id=nct_id,
        pdf_path=pdf_path,
        text_first=text_first,
        query_driven=query_driven,
        coverage_delta=round(coverage_delta, 2),
        coherence_ratio=round(coherence_ratio, 2),
        section_agreement=round(agreement, 4),
        error=error,
    )


# ---------------------------------------------------------------------------
# Aggregate and recommendation
# ---------------------------------------------------------------------------


def _average_metrics(
    metrics_list: Sequence[PipelineMetrics],
) -> Optional[PipelineMetrics]:
    """Average a list of :class:`PipelineMetrics`."""
    if not metrics_list:
        return None

    n = len(metrics_list)
    all_codes: set[str] = set()
    for m in metrics_list:
        all_codes |= m.section_codes

    return PipelineMetrics(
        coverage_pct=round(
            sum(m.coverage_pct for m in metrics_list) / n, 2
        ),
        avg_confidence=round(
            sum(m.avg_confidence for m in metrics_list) / n, 4
        ),
        avg_span_length=round(
            sum(m.avg_span_length for m in metrics_list) / n, 1
        ),
        gap_rate=round(
            sum(m.gap_rate for m in metrics_list) / n, 4
        ),
        section_count=round(
            sum(m.section_count for m in metrics_list) / n
        ),
        section_codes=all_codes,
        high_confidence_pct=round(
            sum(m.high_confidence_pct for m in metrics_list) / n,
            1,
        ),
        medium_confidence_pct=round(
            sum(m.medium_confidence_pct for m in metrics_list) / n,
            1,
        ),
        low_confidence_pct=round(
            sum(m.low_confidence_pct for m in metrics_list) / n,
            1,
        ),
        total_chars_classified=sum(
            m.total_chars_classified for m in metrics_list
        ),
        total_chars=sum(m.total_chars for m in metrics_list),
    )


def generate_recommendation(
    avg_coverage_delta: float,
    avg_coherence_ratio: float,
    qd_high_confidence_pct: float,
    qd_gap_rate: float,
) -> tuple[str, str]:
    """Derive a pipeline recommendation from aggregate metrics.

    Returns:
        Tuple of ``(recommendation, reason)`` where recommendation
        is ``"replace"``, ``"supplement"``, or ``"keep_text_first"``.
    """
    # Count how many criteria the query-driven pipeline meets.
    criteria_met = 0
    reasons: list[str] = []

    if avg_coverage_delta >= COVERAGE_DELTA_THRESHOLD:
        criteria_met += 1
        reasons.append(
            f"coverage delta {avg_coverage_delta:+.1f}pp "
            f"(>= {COVERAGE_DELTA_THRESHOLD}pp)"
        )

    if avg_coherence_ratio >= COHERENCE_RATIO_THRESHOLD:
        criteria_met += 1
        reasons.append(
            f"coherence ratio {avg_coherence_ratio:.1f}x "
            f"(>= {COHERENCE_RATIO_THRESHOLD}x)"
        )

    if qd_high_confidence_pct >= HIGH_CONFIDENCE_PCT_THRESHOLD:
        criteria_met += 1
        reasons.append(
            f"high-confidence {qd_high_confidence_pct:.0f}% "
            f"(>= {HIGH_CONFIDENCE_PCT_THRESHOLD}%)"
        )

    if qd_gap_rate <= GAP_RATE_THRESHOLD:
        criteria_met += 1
        reasons.append(
            f"gap rate {qd_gap_rate:.0%} "
            f"(<= {GAP_RATE_THRESHOLD:.0%})"
        )

    reason_text = "; ".join(reasons) if reasons else "No criteria met"

    if criteria_met >= 3:
        return (
            "replace",
            f"Query-driven meets {criteria_met}/4 criteria: "
            f"{reason_text}. Recommend replacing text-first.",
        )
    elif criteria_met >= 2:
        return (
            "supplement",
            f"Query-driven meets {criteria_met}/4 criteria: "
            f"{reason_text}. Recommend supplementing text-first "
            f"with query-driven for gap filling.",
        )
    else:
        return (
            "keep_text_first",
            f"Query-driven meets only {criteria_met}/4 criteria: "
            f"{reason_text}. Recommend keeping text-first as "
            f"primary pipeline.",
        )


def generate_report(
    benchmarks: Sequence[ProtocolBenchmark],
) -> BenchmarkReport:
    """Build an aggregate :class:`BenchmarkReport`."""
    tf_metrics = [
        b.text_first for b in benchmarks if b.text_first
    ]
    qd_metrics = [
        b.query_driven for b in benchmarks if b.query_driven
    ]
    valid = [
        b for b in benchmarks
        if b.text_first and b.query_driven
    ]

    avg_delta = (
        sum(b.coverage_delta for b in valid) / len(valid)
        if valid
        else 0.0
    )
    avg_ratio = (
        sum(b.coherence_ratio for b in valid) / len(valid)
        if valid
        else 0.0
    )
    avg_agreement = (
        sum(b.section_agreement for b in valid) / len(valid)
        if valid
        else 0.0
    )

    agg_tf = _average_metrics(tf_metrics)
    agg_qd = _average_metrics(qd_metrics)

    qd_high = agg_qd.high_confidence_pct if agg_qd else 0.0
    qd_gap = agg_qd.gap_rate if agg_qd else 1.0

    rec, reason = generate_recommendation(
        avg_delta, avg_ratio, qd_high, qd_gap,
    )

    return BenchmarkReport(
        protocols=list(benchmarks),
        aggregate_text_first=agg_tf,
        aggregate_query_driven=agg_qd,
        avg_coverage_delta=round(avg_delta, 2),
        avg_coherence_ratio=round(avg_ratio, 2),
        avg_section_agreement=round(avg_agreement, 4),
        recommendation=rec,
        recommendation_reason=reason,
    )


# ---------------------------------------------------------------------------
# Entry point — full corpus benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    manifest_path: str | Path,
    data_dir: str | Path,
    max_protocols: Optional[int] = None,
) -> BenchmarkReport:
    """Run the full benchmark across the protocol corpus.

    Args:
        manifest_path: Path to ``refined_qualifying_manifest.json``.
        data_dir: Directory containing protocol PDFs
            (``data/protocols/clinicaltrials/``).
        max_protocols: Limit the number of protocols (for quick tests).

    Returns:
        :class:`BenchmarkReport` with per-protocol and aggregate
        results.
    """
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    if max_protocols:
        manifest = manifest[:max_protocols]

    benchmarks: list[ProtocolBenchmark] = []

    for entry in manifest:
        nct_id = entry["nct_id"]
        pdf_path = data_dir / f"{nct_id}.pdf"

        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            benchmarks.append(
                ProtocolBenchmark(
                    nct_id=nct_id,
                    pdf_path=str(pdf_path),
                    text_first=None,
                    query_driven=None,
                    coverage_delta=0.0,
                    coherence_ratio=0.0,
                    section_agreement=0.0,
                    error=f"PDF not found: {pdf_path}",
                )
            )
            continue

        logger.info("Processing %s ...", nct_id)
        benchmark = _benchmark_one_protocol(
            nct_id, str(pdf_path),
        )
        benchmarks.append(benchmark)

    return generate_report(benchmarks)


def _benchmark_one_protocol(
    nct_id: str,
    pdf_path: str,
) -> ProtocolBenchmark:
    """Run both pipelines on a single protocol."""
    tf_metrics: Optional[PipelineMetrics] = None
    qd_metrics: Optional[PipelineMetrics] = None
    error = ""

    # Extract protocol index (shared).
    try:
        protocol_index = extract_protocol_index(pdf_path)
    except Exception as exc:
        logger.error(
            "Failed to extract index for %s: %s", nct_id, exc,
        )
        return ProtocolBenchmark(
            nct_id=nct_id,
            pdf_path=pdf_path,
            text_first=None,
            query_driven=None,
            coverage_delta=0.0,
            coherence_ratio=0.0,
            section_agreement=0.0,
            error=f"Index extraction failed: {exc}",
        )

    full_text = protocol_index.full_text

    # Text-first pipeline.
    try:
        tf_metrics = run_text_first(full_text, nct_id)
    except Exception as exc:
        logger.error(
            "Text-first failed for %s: %s", nct_id, exc,
        )
        error = f"Text-first failed: {exc}"

    # Query-driven pipeline.
    try:
        qd_metrics = run_query_driven(protocol_index)
    except Exception as exc:
        logger.error(
            "Query-driven failed for %s: %s", nct_id, exc,
        )
        error = (
            f"{error}; " if error else ""
        ) + f"Query-driven failed: {exc}"

    return compare_protocol(nct_id, pdf_path, tf_metrics, qd_metrics)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _metrics_to_dict(m: PipelineMetrics) -> dict:
    """Serialise :class:`PipelineMetrics` to a JSON-safe dict."""
    return {
        "coverage_pct": m.coverage_pct,
        "avg_confidence": m.avg_confidence,
        "avg_span_length": m.avg_span_length,
        "gap_rate": m.gap_rate,
        "section_count": m.section_count,
        "section_codes": sorted(m.section_codes, key=section_sort_key),
        "high_confidence_pct": m.high_confidence_pct,
        "medium_confidence_pct": m.medium_confidence_pct,
        "low_confidence_pct": m.low_confidence_pct,
        "total_chars_classified": m.total_chars_classified,
        "total_chars": m.total_chars,
    }


def _report_to_dict(report: BenchmarkReport) -> dict:
    """Serialise :class:`BenchmarkReport` to a JSON-safe dict."""
    protocols = []
    for p in report.protocols:
        protocols.append({
            "nct_id": p.nct_id,
            "pdf_path": p.pdf_path,
            "text_first": (
                _metrics_to_dict(p.text_first)
                if p.text_first else None
            ),
            "query_driven": (
                _metrics_to_dict(p.query_driven)
                if p.query_driven else None
            ),
            "coverage_delta": p.coverage_delta,
            "coherence_ratio": p.coherence_ratio,
            "section_agreement": p.section_agreement,
            "error": p.error,
        })

    return {
        "protocols": protocols,
        "aggregate_text_first": (
            _metrics_to_dict(report.aggregate_text_first)
            if report.aggregate_text_first else None
        ),
        "aggregate_query_driven": (
            _metrics_to_dict(report.aggregate_query_driven)
            if report.aggregate_query_driven else None
        ),
        "avg_coverage_delta": report.avg_coverage_delta,
        "avg_coherence_ratio": report.avg_coherence_ratio,
        "avg_section_agreement": report.avg_section_agreement,
        "recommendation": report.recommendation,
        "recommendation_reason": report.recommendation_reason,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _report_to_markdown(report: BenchmarkReport) -> str:
    """Render a :class:`BenchmarkReport` as Markdown."""
    lines: list[str] = []
    lines.append("# Benchmark: Query-Driven vs Text-First")
    lines.append("")

    # Aggregate summary.
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- **Protocols benchmarked:** {len(report.protocols)}"
    )
    valid = sum(
        1 for p in report.protocols
        if p.text_first and p.query_driven
    )
    errors = sum(1 for p in report.protocols if p.error)
    lines.append(
        f"- **Successful comparisons:** {valid}"
    )
    if errors:
        lines.append(f"- **Errors:** {errors}")
    lines.append(
        f"- **Avg coverage delta:** "
        f"{report.avg_coverage_delta:+.1f} pp"
    )
    lines.append(
        f"- **Avg coherence ratio:** "
        f"{report.avg_coherence_ratio:.1f}x"
    )
    lines.append(
        f"- **Avg section agreement:** "
        f"{report.avg_section_agreement:.1%}"
    )
    lines.append("")

    # Recommendation.
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"**{report.recommendation.upper()}**: "
        f"{report.recommendation_reason}"
    )
    lines.append("")

    # Aggregate metrics table.
    if report.aggregate_text_first and report.aggregate_query_driven:
        tf = report.aggregate_text_first
        qd = report.aggregate_query_driven
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append(
            "| Metric | Text-First | Query-Driven | Delta |"
        )
        lines.append(
            "|--------|-----------|-------------|-------|"
        )
        lines.append(
            f"| Coverage | {tf.coverage_pct:.1f}% "
            f"| {qd.coverage_pct:.1f}% "
            f"| {qd.coverage_pct - tf.coverage_pct:+.1f}pp |"
        )
        lines.append(
            f"| Avg confidence | {tf.avg_confidence:.3f} "
            f"| {qd.avg_confidence:.3f} "
            f"| {qd.avg_confidence - tf.avg_confidence:+.3f} |"
        )
        lines.append(
            f"| Avg span length | {tf.avg_span_length:.0f} "
            f"| {qd.avg_span_length:.0f} "
            f"| {qd.avg_span_length / tf.avg_span_length:.1f}x |"
            if tf.avg_span_length > 0
            else "| Avg span length | 0 | "
            f"{qd.avg_span_length:.0f} | N/A |"
        )
        lines.append(
            f"| Gap rate | {tf.gap_rate:.0%} "
            f"| {qd.gap_rate:.0%} "
            f"| {qd.gap_rate - tf.gap_rate:+.0%} |"
        )
        lines.append(
            f"| Sections populated | {tf.section_count} "
            f"| {qd.section_count} "
            f"| {qd.section_count - tf.section_count:+d} |"
        )
        lines.append(
            f"| High confidence | {tf.high_confidence_pct:.0f}% "
            f"| {qd.high_confidence_pct:.0f}% "
            f"| {qd.high_confidence_pct - tf.high_confidence_pct:+.0f}pp |"
        )
        lines.append("")

    # Per-protocol table.
    lines.append("## Per-Protocol Results")
    lines.append("")
    lines.append(
        "| NCT ID | TF Cov | QD Cov | Delta | "
        "Coherence | Agreement | Error |"
    )
    lines.append(
        "|--------|--------|--------|-------|"
        "-----------|-----------|-------|"
    )
    for p in report.protocols:
        tf_cov = (
            f"{p.text_first.coverage_pct:.1f}%"
            if p.text_first else "N/A"
        )
        qd_cov = (
            f"{p.query_driven.coverage_pct:.1f}%"
            if p.query_driven else "N/A"
        )
        delta = (
            f"{p.coverage_delta:+.1f}pp"
            if p.text_first and p.query_driven
            else "N/A"
        )
        coherence = (
            f"{p.coherence_ratio:.1f}x"
            if p.coherence_ratio > 0
            else "N/A"
        )
        agreement = (
            f"{p.section_agreement:.0%}"
            if p.text_first and p.query_driven
            else "N/A"
        )
        err = p.error[:30] if p.error else ""
        lines.append(
            f"| {p.nct_id} | {tf_cov} | {qd_cov} | {delta} "
            f"| {coherence} | {agreement} | {err} |"
        )
    lines.append("")

    return "\n".join(lines)
