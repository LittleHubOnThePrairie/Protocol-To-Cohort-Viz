"""Analysis CLI for query pipeline debugging (PTCV-149).

Composable subcommands that output JSON for Claude Code to
iteratively reason over batch pipeline outputs.

Usage::

    python -m ptcv.analysis.analysis_cli <subcommand> [OPTIONS]

Subcommands::

    overview          High-level batch run summary
    section-report    Deep-dive into one ICH section
    protocol-report   Full analysis of one protocol
    low-confidence    List low-confidence matches
    gaps              Sections with no extraction
    patterns          Header vocabulary and confusion matrix
    compare           Diff two batch runs
    export            Export results as CSV
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ptcv.analysis.data_store import AnalysisStore


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_overview(
    store: AnalysisStore, run_id: str, top: int,
) -> dict[str, Any]:
    """High-level batch run summary with per-section stats."""
    run = store.get_run(run_id)
    if run is None:
        return {"error": f"Run {run_id} not found"}

    stats = store.get_section_stats(run_id)
    cov = store.get_coverage_distribution(run_id)

    total = run["protocol_count"] or 1
    pass_rate = round((run["pass_count"] or 0) / total, 4)

    # Sort sections by avg_boosted ascending for worst/best
    sorted_stats = sorted(stats, key=lambda s: s["avg_boosted"])

    worst = [
        {
            "code": s["ich_section_code"],
            "name": s["ich_section_name"],
            "avg_confidence": round(s["avg_boosted"], 4),
            "protocol_count": s["protocol_count"],
            "auto_map_rate": round(s["auto_map_rate"], 4),
        }
        for s in sorted_stats[:top]
    ]
    best = [
        {
            "code": s["ich_section_code"],
            "name": s["ich_section_name"],
            "avg_confidence": round(s["avg_boosted"], 4),
            "protocol_count": s["protocol_count"],
            "auto_map_rate": round(s["auto_map_rate"], 4),
        }
        for s in reversed(sorted_stats[-top:])
    ]

    return {
        "run_id": run_id,
        "pipeline_version": run.get("pipeline_version", ""),
        "total_protocols": run["protocol_count"],
        "pass_count": run["pass_count"],
        "error_count": run["error_count"],
        "pass_rate": pass_rate,
        "avg_coverage_pct": cov.get("avg_coverage_pct", 0.0),
        "avg_confidence": cov.get("avg_confidence", 0.0),
        "elapsed_seconds": run.get("elapsed_seconds", 0.0),
        "worst_sections": worst,
        "best_sections": best,
    }


def cmd_section_report(
    store: AnalysisStore, run_id: str, section: str, top: int,
) -> dict[str, Any]:
    """Deep-dive into one ICH section across all protocols."""
    stats = store.get_section_stats(run_id)
    section_stat = next(
        (s for s in stats if s["ich_section_code"] == section),
        None,
    )

    if section_stat is None:
        return {"error": f"Section {section} not found in {run_id}"}

    # Get header vocabulary for this section
    cur = store._conn.execute(
        """SELECT protocol_section_title, COUNT(*) AS cnt
           FROM section_matches sm
           JOIN protocol_results pr ON sm.result_id = pr.id
           WHERE pr.run_id = ? AND sm.ich_section_code = ?
           GROUP BY protocol_section_title
           ORDER BY cnt DESC""",
        (run_id, section),
    )
    headers = [
        {"header": r["protocol_section_title"], "count": r["cnt"]}
        for r in cur.fetchall()
    ]

    # Lowest confidence protocols for this section
    cur = store._conn.execute(
        """SELECT pr.nct_id, sm.boosted_score, sm.similarity_score,
                  sm.match_method, sm.protocol_section_title
           FROM section_matches sm
           JOIN protocol_results pr ON sm.result_id = pr.id
           WHERE pr.run_id = ? AND sm.ich_section_code = ?
           ORDER BY sm.boosted_score ASC
           LIMIT ?""",
        (run_id, section, top),
    )
    lowest = [dict(r) for r in cur.fetchall()]

    # What other ICH sections does this section's content get matched to?
    cur = store._conn.execute(
        """SELECT sm2.ich_section_code AS other_code,
                  COUNT(*) AS cnt,
                  AVG(sm2.boosted_score) AS avg_score
           FROM section_matches sm
           JOIN protocol_results pr ON sm.result_id = pr.id
           JOIN section_matches sm2 ON sm2.result_id = pr.id
             AND sm2.protocol_section_title = sm.protocol_section_title
             AND sm2.ich_section_code != ?
           WHERE pr.run_id = ? AND sm.ich_section_code = ?
           GROUP BY sm2.ich_section_code
           ORDER BY cnt DESC""",
        (section, run_id, section),
    )
    misclass = [
        {
            "matched_to": r["other_code"],
            "count": r["cnt"],
            "avg_score": round(r["avg_score"], 4),
        }
        for r in cur.fetchall()
    ]

    total_protocols = section_stat.get("total_protocols", 0)

    return {
        "section_code": section,
        "section_name": section_stat["ich_section_name"],
        "corpus_stats": {
            "protocol_count": section_stat["protocol_count"],
            "total_protocols": total_protocols,
            "hit_rate": round(section_stat["hit_rate"], 4),
            "avg_confidence": round(
                section_stat["avg_boosted"], 4,
            ),
            "min_confidence": round(
                section_stat["min_boosted"], 4,
            ),
            "max_confidence": round(
                section_stat["max_boosted"], 4,
            ),
            "auto_map_rate": round(
                section_stat["auto_map_rate"], 4,
            ),
        },
        "common_protocol_headers": headers[:top],
        "lowest_confidence_protocols": lowest,
        "misclassification_targets": misclass,
    }


def cmd_protocol_report(
    store: AnalysisStore, run_id: str, nct_id: str,
) -> dict[str, Any]:
    """Full analysis of one protocol's results."""
    summary = store.get_protocol_summary(run_id, nct_id)
    if not summary:
        return {"error": f"Protocol {nct_id} not found in {run_id}"}

    # Trim large text fields for CLI output
    for cp in summary.get("comparison_pairs", []):
        for field in ("original_text", "extracted_text"):
            if cp.get(field) and len(cp[field]) > 500:
                cp[field] = cp[field][:500] + "..."

    return {
        "nct_id": nct_id,
        "run_id": run_id,
        "status": summary.get("status", ""),
        "elapsed_seconds": summary.get("elapsed_seconds", 0.0),
        "toc_section_count": summary.get("toc_section_count", 0),
        "coverage": summary.get("coverage", {}),
        "section_matches": summary.get("section_matches", []),
        "comparison_pairs": summary.get("comparison_pairs", []),
    }


def cmd_low_confidence(
    store: AnalysisStore,
    run_id: str,
    threshold: float,
    top: int,
) -> dict[str, Any]:
    """List ICH sections with low average confidence."""
    low = store.get_low_confidence_sections(run_id, threshold)
    return {
        "run_id": run_id,
        "threshold": threshold,
        "sections": low[:top],
    }


def cmd_gaps(
    store: AnalysisStore, run_id: str, top: int,
) -> dict[str, Any]:
    """Protocols/sections with no extraction despite content."""
    cur = store._conn.execute(
        """SELECT cp.ich_section_code, pr.nct_id,
                  cp.original_text, cp.match_quality
           FROM comparison_pairs cp
           JOIN protocol_results pr ON cp.result_id = pr.id
           WHERE pr.run_id = ?
             AND cp.match_quality IN ('gap', 'missing', 'poor')
           ORDER BY cp.ich_section_code, pr.nct_id
           LIMIT ?""",
        (run_id, top),
    )
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        if d.get("original_text") and len(d["original_text"]) > 200:
            d["original_text"] = d["original_text"][:200] + "..."
        rows.append(d)

    # Summary by section
    cur = store._conn.execute(
        """SELECT cp.ich_section_code,
                  COUNT(*) AS gap_count,
                  cp.match_quality
           FROM comparison_pairs cp
           JOIN protocol_results pr ON cp.result_id = pr.id
           WHERE pr.run_id = ?
             AND cp.match_quality IN ('gap', 'missing', 'poor')
           GROUP BY cp.ich_section_code, cp.match_quality
           ORDER BY gap_count DESC""",
        (run_id,),
    )
    by_section = [dict(r) for r in cur.fetchall()]

    return {
        "run_id": run_id,
        "gap_summary": by_section,
        "details": rows,
    }


def cmd_patterns(
    store: AnalysisStore, run_id: str, top: int,
) -> dict[str, Any]:
    """Header vocabulary and confusion matrix."""
    # Header vocabulary per ICH section
    cur = store._conn.execute(
        """SELECT sm.ich_section_code,
                  sm.protocol_section_title,
                  COUNT(*) AS cnt
           FROM section_matches sm
           JOIN protocol_results pr ON sm.result_id = pr.id
           WHERE pr.run_id = ?
           GROUP BY sm.ich_section_code, sm.protocol_section_title
           ORDER BY sm.ich_section_code, cnt DESC""",
        (run_id,),
    )
    vocab: dict[str, list[dict[str, Any]]] = {}
    for r in cur.fetchall():
        code = r["ich_section_code"]
        vocab.setdefault(code, []).append({
            "header": r["protocol_section_title"],
            "count": r["cnt"],
        })

    # Trim to top N per section
    header_vocabulary = {
        code: {
            "common": entries[:3],
            "rare": entries[-2:] if len(entries) > 3 else [],
        }
        for code, entries in vocab.items()
    }

    # Confusion matrix: protocol titles that map to multiple ICH codes
    cur = store._conn.execute(
        """SELECT
             sm.protocol_section_title,
             sm.ich_section_code,
             COUNT(*) AS cnt
           FROM section_matches sm
           JOIN protocol_results pr ON sm.result_id = pr.id
           WHERE pr.run_id = ?
           GROUP BY sm.protocol_section_title, sm.ich_section_code
           ORDER BY sm.protocol_section_title, cnt DESC""",
        (run_id,),
    )
    # Build title → {code: count} mapping
    title_codes: dict[str, dict[str, int]] = {}
    for r in cur.fetchall():
        title = r["protocol_section_title"]
        code = r["ich_section_code"]
        title_codes.setdefault(title, {})[code] = r["cnt"]

    # Only include titles that map to multiple codes
    confusion: dict[str, int] = {}
    for title, codes in title_codes.items():
        if len(codes) > 1:
            for code, cnt in codes.items():
                key = f"{title} → {code}"
                confusion[key] = cnt

    return {
        "run_id": run_id,
        "header_vocabulary": header_vocabulary,
        "confusion_matrix": confusion,
    }


def cmd_compare(
    store: AnalysisStore, run_a: str, run_b: str,
) -> dict[str, Any]:
    """Diff two batch runs."""
    return store.compare_runs(run_a, run_b)


def cmd_export(
    store: AnalysisStore, run_id: str, table: str,
) -> str:
    """Export a table as CSV."""
    valid_tables = {
        "section_matches", "query_extractions",
        "coverage_reports", "comparison_pairs",
        "protocol_results",
    }
    if table not in valid_tables:
        return json.dumps({
            "error": f"Invalid table: {table}. "
            f"Valid: {', '.join(sorted(valid_tables))}",
        })

    cur = store._conn.execute(
        f"""SELECT t.* FROM {table} t
            JOIN protocol_results pr ON t.result_id = pr.id
            WHERE pr.run_id = ?"""
        if table != "protocol_results"
        else f"SELECT * FROM {table} WHERE run_id = ?",
        (run_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(rows[0].keys())
    for row in rows:
        writer.writerow(tuple(row))
    return output.getvalue()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_output(
    data: Any, fmt: str,
) -> str:
    """Format output according to requested format."""
    if fmt == "json":
        return json.dumps(data, indent=2, default=str)

    if fmt == "markdown":
        return _to_markdown(data)

    # table format — simple aligned text
    return _to_table(data)


def _to_markdown(data: Any) -> str:
    """Convert dict/list to markdown."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        lines: list[str] = []
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                lines.append(f"\n### {k}\n")
                lines.append(_to_markdown(v))
            else:
                lines.append(f"**{k}:** {v}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return "_empty_"
        if isinstance(data[0], dict):
            keys = list(data[0].keys())
            header = "| " + " | ".join(keys) + " |"
            sep = "| " + " | ".join("---" for _ in keys) + " |"
            rows = [
                "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |"
                for r in data
            ]
            return "\n".join([header, sep, *rows])
        return "\n".join(f"- {item}" for item in data)
    return str(data)


def _to_table(data: Any) -> str:
    """Convert dict/list to aligned text table."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if "error" in data:
            return f"ERROR: {data['error']}"
        lines = []
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                lines.append(f"\n{k}:")
                lines.append(_to_table(v))
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return "  (empty)"
        if isinstance(data[0], dict):
            keys = list(data[0].keys())
            widths = {
                k: max(len(k), max(
                    len(str(r.get(k, ""))) for r in data
                ))
                for k in keys
            }
            header = "  ".join(
                str(k).ljust(widths[k]) for k in keys
            )
            sep = "  ".join("-" * widths[k] for k in keys)
            rows = [
                "  ".join(
                    str(r.get(k, "")).ljust(widths[k])
                    for k in keys
                )
                for r in data
            ]
            return "\n".join([header, sep, *rows])
        return "\n".join(f"  {item}" for item in data)
    return str(data)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analysis CLI for query pipeline debugging.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/analysis/results.db"),
        help="SQLite database path",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Batch run ID (default: latest)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "table", "markdown"],
        default="json",
        dest="output_format",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Limit output rows (default: 20)",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("overview", help="Batch run summary")

    sec = sub.add_parser(
        "section-report", help="ICH section deep-dive",
    )
    sec.add_argument(
        "--section", required=True, help="ICH section code",
    )

    proto = sub.add_parser(
        "protocol-report", help="Single protocol analysis",
    )
    proto.add_argument(
        "--nct", required=True, help="NCT ID",
    )

    lc = sub.add_parser(
        "low-confidence", help="Low-confidence sections",
    )
    lc.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence threshold (default: 0.60)",
    )

    sub.add_parser("gaps", help="Missing/poor extractions")
    sub.add_parser("patterns", help="Header vocabulary")

    cmp = sub.add_parser("compare", help="Diff two runs")
    cmp.add_argument("--run-a", required=True, help="First run ID")
    cmp.add_argument("--run-b", required=True, help="Second run ID")

    exp = sub.add_parser("export", help="Export table as CSV")
    exp.add_argument(
        "--table",
        required=True,
        choices=[
            "section_matches", "query_extractions",
            "coverage_reports", "comparison_pairs",
            "protocol_results",
        ],
        help="Table to export",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if not args.db.exists():
        print(
            json.dumps({"error": f"Database not found: {args.db}"}),
        )
        return 1

    store = AnalysisStore(args.db)

    run_id = args.run_id or store.get_latest_run_id()
    if run_id is None:
        print(json.dumps({"error": "No batch runs found"}))
        store.close()
        return 1

    try:
        if args.command == "overview":
            result = cmd_overview(store, run_id, args.top)
        elif args.command == "section-report":
            result = cmd_section_report(
                store, run_id, args.section, args.top,
            )
        elif args.command == "protocol-report":
            result = cmd_protocol_report(store, run_id, args.nct)
        elif args.command == "low-confidence":
            result = cmd_low_confidence(
                store, run_id, args.threshold, args.top,
            )
        elif args.command == "gaps":
            result = cmd_gaps(store, run_id, args.top)
        elif args.command == "patterns":
            result = cmd_patterns(store, run_id, args.top)
        elif args.command == "compare":
            result = cmd_compare(store, args.run_a, args.run_b)
        elif args.command == "export":
            csv_output = cmd_export(store, run_id, args.table)
            print(csv_output)
            store.close()
            return 0
        else:
            parser.print_help()
            store.close()
            return 1

        print(_format_output(result, args.output_format))
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
