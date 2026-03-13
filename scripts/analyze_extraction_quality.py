"""Analyse extraction quality from a batch pipeline run.

Classifies every query extraction as limited / verbose / malformed / normal
and produces a per-section quality report.

Usage:
    cd C:/Dev/PTCV && python scripts/analyze_extraction_quality.py
    python scripts/analyze_extraction_quality.py --db data/analysis/sample40_results.db --examples 5
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB = Path("data/analysis/sample40_results.db")

GAP_PHRASES: list[str] = [
    "does not contain",
    "not explicitly",
    "not specified",
    "not provided",
    "not available",
    "no specific",
    "not mentioned",
    "not addressed",
    "no information",
]

# Regex for TOC-like lines: "6.1  CRITERIA ... 13" with dots/ellipsis/garble
_TOC_LINE = re.compile(
    r"^\d+\.\d+\s+.*?(?:\.{4,}|…|[\ufffd\ufffe])",
    re.MULTILINE,
)

# CID garble from broken PDF character maps
_CID_PATTERN = "(cid:"

# ICH parent sections in canonical order
PARENT_ORDER: list[str] = [
    "B.1", "B.2", "B.3", "B.4", "B.5", "B.6", "B.7", "B.8",
    "B.9", "B.10", "B.11", "B.12", "B.13", "B.14", "B.15", "B.16",
]

VERBOSE_METHODS = frozenset({
    "llm_transform", "relevance_excerpt", "heuristic", "table_detection",
})

# Categories (priority order)
MALFORMED = "malformed"
LIMITED = "limited"
VERBOSE = "verbose"
NORMAL = "normal"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExtractionRecord:
    id: int
    query_id: str
    section_id: str
    result_id: int
    extracted_content: str
    verbatim_content: str
    confidence: float
    extraction_method: str
    source_section: str
    nct_id: str
    content_length: int


@dataclass
class ClassificationResult:
    record: ExtractionRecord
    category: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class SectionSummary:
    parent_section: str
    total: int = 0
    limited_count: int = 0
    verbose_count: int = 0
    malformed_count: int = 0
    normal_count: int = 0
    confidences: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    method_counts: Counter = field(default_factory=Counter)
    examples: dict[str, list[ClassificationResult]] = field(
        default_factory=lambda: defaultdict(list)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parent_section(section_id: str) -> str:
    """Extract parent section: 'B.4.3' -> 'B.4', 'B.11' -> 'B.11'."""
    parts = section_id.split(".")
    if len(parts) >= 2:
        candidate = f"{parts[0]}.{parts[1]}"
        # B.10 through B.16 are two-digit; check if section_id starts
        # with a known parent that has sub-sections
        if candidate in PARENT_ORDER:
            return candidate
        # Might be a deeper sub-section like B.10.3 — parent is B.10
        # Walk up until we find a known parent
        for i in range(min(3, len(parts)), 1, -1):
            attempt = ".".join(parts[:i])
            if attempt in PARENT_ORDER:
                return attempt
    return section_id


def _content_hash(text: str) -> str:
    """Fast content fingerprint for dedup detection."""
    return hashlib.md5(text[:5000].encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_extractions(conn: sqlite3.Connection) -> list[ExtractionRecord]:
    """Load all query extractions joined with protocol NCT IDs."""
    cur = conn.execute("""
        SELECT
            qe.id, qe.query_id, qe.section_id, qe.result_id,
            qe.extracted_content, qe.verbatim_content,
            qe.confidence, qe.extraction_method, qe.source_section,
            pr.nct_id, LENGTH(qe.extracted_content)
        FROM query_extractions qe
        JOIN protocol_results pr ON qe.result_id = pr.id
        ORDER BY qe.section_id, pr.nct_id
    """)
    records: list[ExtractionRecord] = []
    for row in cur.fetchall():
        records.append(ExtractionRecord(
            id=row[0],
            query_id=row[1],
            section_id=row[2],
            result_id=row[3],
            extracted_content=row[4] or "",
            verbatim_content=row[5] or "",
            confidence=row[6],
            extraction_method=row[7] or "",
            source_section=row[8] or "",
            nct_id=row[9] or "",
            content_length=row[10] or 0,
        ))
    return records


def detect_repeated_dumps(
    records: list[ExtractionRecord],
) -> set[int]:
    """Find extraction IDs that are duplicate full-doc dumps.

    A "repeated dump" is when the same large content blob appears for
    multiple queries within the same protocol.
    """
    # Group by (result_id, content_hash) for long unscoped content
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for r in records:
        if (
            r.extraction_method == "unscoped_search"
            and r.content_length > 10_000
        ):
            h = _content_hash(r.extracted_content)
            groups[(r.result_id, h)].append(r.id)

    dup_ids: set[int] = set()
    for key, ids in groups.items():
        if len(ids) > 1:
            dup_ids.update(ids)
    return dup_ids


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_extraction(
    record: ExtractionRecord,
    dup_ids: set[int],
) -> ClassificationResult:
    """Classify a single extraction. Priority: malformed > limited > verbose > normal."""
    reasons: list[str] = []
    content = record.extracted_content
    method = record.extraction_method
    conf = record.confidence
    length = record.content_length

    # --- MALFORMED checks ---
    malformed_reasons: list[str] = []

    if method == "unscoped_search" and length > 10_000:
        malformed_reasons.append(
            f"full-doc dump ({length:,} chars via unscoped_search)"
        )

    if _CID_PATTERN in content:
        malformed_reasons.append("(cid: garble from broken PDF char map")

    if length > 0:
        toc_matches = _TOC_LINE.findall(content[:500])
        if len(toc_matches) >= 3:
            malformed_reasons.append(
                f"TOC spillage ({len(toc_matches)} TOC-like lines)"
            )

    if record.id in dup_ids:
        malformed_reasons.append("repeated dump across multiple queries")

    if malformed_reasons:
        return ClassificationResult(record, MALFORMED, malformed_reasons)

    # --- LIMITED checks ---
    limited_reasons: list[str] = []

    if length < 20:
        limited_reasons.append(f"very short content ({length} chars)")

    if method == "llm_transform":
        head = content[:500].lower()
        for phrase in GAP_PHRASES:
            if phrase in head:
                limited_reasons.append(f'LLM gap response ("{phrase}")')
                break

    if length < 50 and conf < 0.50:
        limited_reasons.append(
            f"short + low confidence ({length} chars, conf={conf:.2f})"
        )

    if limited_reasons:
        return ClassificationResult(record, LIMITED, limited_reasons)

    # --- VERBOSE checks ---
    if (
        length > 200
        and conf >= 0.40
        and method in VERBOSE_METHODS
    ):
        return ClassificationResult(
            record, VERBOSE, ["genuine clinical content"]
        )

    # --- NORMAL (default) ---
    return ClassificationResult(record, NORMAL, ["mid-range extraction"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_report(
    classifications: list[ClassificationResult],
    max_examples: int = 3,
) -> dict[str, SectionSummary]:
    """Group classifications by parent section and compute summaries."""
    summaries: dict[str, SectionSummary] = {}
    for ps in PARENT_ORDER:
        summaries[ps] = SectionSummary(parent_section=ps)

    for cl in classifications:
        ps = parent_section(cl.record.section_id)
        if ps not in summaries:
            summaries[ps] = SectionSummary(parent_section=ps)
        s = summaries[ps]
        s.total += 1
        s.confidences.append(cl.record.confidence)
        s.lengths.append(cl.record.content_length)
        s.method_counts[cl.record.extraction_method] += 1

        if cl.category == LIMITED:
            s.limited_count += 1
        elif cl.category == VERBOSE:
            s.verbose_count += 1
        elif cl.category == MALFORMED:
            s.malformed_count += 1
        else:
            s.normal_count += 1

        if len(s.examples[cl.category]) < max_examples:
            s.examples[cl.category].append(cl)

    return summaries


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  -  "
    return f"{100 * n / total:5.1f}%"


def _avg(vals: list[float | int]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _median(vals: list[int]) -> int:
    return int(statistics.median(vals)) if vals else 0


def print_report(
    summaries: dict[str, SectionSummary],
    classifications: list[ClassificationResult],
    max_examples: int = 3,
) -> None:
    """Print the full quality report to stdout."""
    total = len(classifications)
    cat_counts = Counter(cl.category for cl in classifications)

    # ---------------------------------------------------------------
    # Section 1: Global Summary
    # ---------------------------------------------------------------
    print("=" * 72)
    print("EXTRACTION QUALITY ANALYSIS")
    print("=" * 72)
    print(f"Total extractions: {total:,}")
    print()

    for cat in [MALFORMED, LIMITED, VERBOSE, NORMAL]:
        c = cat_counts[cat]
        print(f"  {cat.upper():12s}  {c:5d}  ({_pct(c, total)})")
    print()

    # ---------------------------------------------------------------
    # Section 2: Per-Section Breakdown
    # ---------------------------------------------------------------
    print("-" * 72)
    print("PER-SECTION BREAKDOWN")
    print("-" * 72)
    hdr = (
        f"{'Section':<7s} {'Total':>5s} "
        f"{'Malformed':>10s} {'Limited':>10s} "
        f"{'Verbose':>10s} {'Normal':>10s} "
        f"{'AvgConf':>8s} {'MedLen':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for ps in PARENT_ORDER:
        s = summaries.get(ps)
        if not s or s.total == 0:
            continue
        t = s.total
        print(
            f"{ps:<7s} {t:>5d} "
            f"{s.malformed_count:>4d} {_pct(s.malformed_count, t):>5s} "
            f"{s.limited_count:>4d} {_pct(s.limited_count, t):>5s} "
            f"{s.verbose_count:>4d} {_pct(s.verbose_count, t):>5s} "
            f"{s.normal_count:>4d} {_pct(s.normal_count, t):>5s} "
            f"{_avg(s.confidences):>8.3f} "
            f"{_median(s.lengths):>8,d}"
        )
    print()

    # ---------------------------------------------------------------
    # Section 3: Problem Hotspots
    # ---------------------------------------------------------------
    print("-" * 72)
    print("PROBLEM HOTSPOTS")
    print("-" * 72)

    # Top malformed sections
    ranked_malf = sorted(
        [(ps, s) for ps, s in summaries.items() if s.total > 0],
        key=lambda x: x[1].malformed_count / x[1].total,
        reverse=True,
    )
    print("\nHighest malformed %:")
    for ps, s in ranked_malf[:5]:
        if s.malformed_count == 0:
            break
        print(
            f"  {ps:<7s}  {s.malformed_count:>3d}/{s.total:<3d} "
            f"({_pct(s.malformed_count, s.total)})  "
            f"methods: {dict(s.method_counts)}"
        )

    # Top limited sections
    ranked_lim = sorted(
        [(ps, s) for ps, s in summaries.items() if s.total > 0],
        key=lambda x: x[1].limited_count / x[1].total,
        reverse=True,
    )
    print("\nHighest limited %:")
    for ps, s in ranked_lim[:5]:
        if s.limited_count == 0:
            break
        print(
            f"  {ps:<7s}  {s.limited_count:>3d}/{s.total:<3d} "
            f"({_pct(s.limited_count, s.total)})"
        )

    # Sections dominated by unscoped_search
    print("\nSections dominated by unscoped_search (>40% of extractions):")
    for ps in PARENT_ORDER:
        s = summaries.get(ps)
        if not s or s.total == 0:
            continue
        unscoped = s.method_counts.get("unscoped_search", 0)
        if unscoped / s.total > 0.40:
            print(
                f"  {ps:<7s}  {unscoped:>3d}/{s.total:<3d} "
                f"({_pct(unscoped, s.total)}) unscoped_search"
            )

    # Sections with highest verbose % (the good stuff)
    ranked_verb = sorted(
        [(ps, s) for ps, s in summaries.items() if s.total > 0],
        key=lambda x: x[1].verbose_count / x[1].total,
        reverse=True,
    )
    print("\nHighest verbose % (best quality):")
    for ps, s in ranked_verb[:5]:
        if s.verbose_count == 0:
            break
        print(
            f"  {ps:<7s}  {s.verbose_count:>3d}/{s.total:<3d} "
            f"({_pct(s.verbose_count, s.total)})"
        )
    print()

    # ---------------------------------------------------------------
    # Section 4: Flagged Examples
    # ---------------------------------------------------------------
    print("-" * 72)
    print("FLAGGED EXAMPLES")
    print("-" * 72)

    for cat in [MALFORMED, LIMITED, VERBOSE]:
        examples: list[ClassificationResult] = []
        for ps in PARENT_ORDER:
            s = summaries.get(ps)
            if s:
                examples.extend(s.examples.get(cat, []))
        if not examples:
            continue

        print(f"\n--- {cat.upper()} (sample of {min(max_examples, len(examples))}) ---")
        shown = 0
        seen_sections: set[str] = set()
        for ex in examples:
            ps = parent_section(ex.record.section_id)
            if ps in seen_sections:
                continue
            seen_sections.add(ps)

            snippet = ex.record.extracted_content[:200].replace("\n", " ")
            print(
                f"\n  [{ex.record.query_id}] "
                f"NCT={ex.record.nct_id}  "
                f"conf={ex.record.confidence:.2f}  "
                f"method={ex.record.extraction_method}  "
                f"len={ex.record.content_length:,}"
            )
            print(f"  Reason: {'; '.join(ex.reasons)}")
            print(f"  Content: \"{snippet}...\"")
            shown += 1
            if shown >= max_examples:
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse extraction quality from a batch run."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to the results SQLite database.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of flagged examples per category (default: 3).",
    )
    args = parser.parse_args()

    db_path = args.db
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        records = load_extractions(conn)
        print(f"Loaded {len(records):,} extractions.\n")

        dup_ids = detect_repeated_dumps(records)
        classifications = [
            classify_extraction(r, dup_ids) for r in records
        ]
        summaries = build_report(classifications, max_examples=args.examples)
        print_report(summaries, classifications, max_examples=args.examples)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
