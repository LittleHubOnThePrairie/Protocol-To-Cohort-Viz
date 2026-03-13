"""SQLite data store for batch pipeline analysis (PTCV-147).

Provides the persistence layer for the batch runner (PTCV-146),
comparison tool (PTCV-148), and analysis CLI (PTCV-149).

Usage::

    store = AnalysisStore("data/analysis/results.db")
    store.create_run("run_001", {"pipeline_version": "abc123"})
    store.store_protocol_result("run_001", result)
    stats = store.get_section_stats("run_001")
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ptcv.analysis.batch_runner import ProtocolResult, ResultStore

_SCHEMA_VERSION = 2

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    pipeline_version TEXT,
    config_hash     TEXT,
    protocol_count  INTEGER DEFAULT 0,
    pass_count      INTEGER DEFAULT 0,
    fail_count      INTEGER DEFAULT 0,
    error_count     INTEGER DEFAULT 0,
    elapsed_seconds REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS protocol_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES batch_runs(run_id),
    nct_id          TEXT NOT NULL,
    file_sha        TEXT,
    status          TEXT NOT NULL,
    error_message   TEXT,
    elapsed_seconds REAL DEFAULT 0.0,
    toc_section_count INTEGER DEFAULT 0,
    UNIQUE(run_id, nct_id)
);

CREATE TABLE IF NOT EXISTS section_matches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES protocol_results(id),
    protocol_section_number TEXT,
    protocol_section_title  TEXT,
    ich_section_code TEXT,
    ich_section_name TEXT,
    similarity_score REAL,
    boosted_score   REAL,
    confidence      TEXT,
    match_method    TEXT,
    auto_mapped     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS query_extractions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES protocol_results(id),
    query_id        TEXT,
    section_id      TEXT,
    extracted_content TEXT,
    verbatim_content TEXT,
    confidence      REAL,
    extraction_method TEXT,
    source_section  TEXT
);

CREATE TABLE IF NOT EXISTS coverage_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES protocol_results(id),
    total_sections  INTEGER,
    populated_count INTEGER,
    gap_count       INTEGER,
    average_confidence REAL,
    total_queries   INTEGER,
    answered_queries INTEGER,
    gap_sections    TEXT
);

CREATE TABLE IF NOT EXISTS comparison_pairs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES protocol_results(id),
    ich_section_code TEXT,
    original_text   TEXT,
    extracted_text  TEXT,
    confidence      REAL,
    match_quality   TEXT
);

CREATE INDEX IF NOT EXISTS idx_protocol_results_run
    ON protocol_results(run_id);
CREATE INDEX IF NOT EXISTS idx_section_matches_result
    ON section_matches(result_id);
CREATE INDEX IF NOT EXISTS idx_section_matches_ich
    ON section_matches(ich_section_code);
CREATE INDEX IF NOT EXISTS idx_query_extractions_result
    ON query_extractions(result_id);
CREATE INDEX IF NOT EXISTS idx_comparison_pairs_result
    ON comparison_pairs(result_id);
CREATE INDEX IF NOT EXISTS idx_comparison_pairs_ich
    ON comparison_pairs(ich_section_code);
"""


class AnalysisStore(ResultStore):
    """SQLite-backed store for batch pipeline analysis results.

    Implements the :class:`ResultStore` interface from
    ``batch_runner.py`` and adds query methods for analysis.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if missing, apply migrations."""
        self._conn.executescript(_CREATE_TABLES)
        # Check schema version
        cur = self._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # ResultStore interface (writer API)
    # ------------------------------------------------------------------

    def create_run(
        self, run_id: str, metadata: dict[str, Any],
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO batch_runs
               (run_id, timestamp, pipeline_version, config_hash)
               VALUES (?, ?, ?, ?)""",
            (
                run_id,
                metadata.get("timestamp", ""),
                metadata.get("pipeline_version", ""),
                metadata.get("config_hash", ""),
            ),
        )
        self._conn.commit()

    def store_protocol_result(
        self, run_id: str, result: ProtocolResult,
    ) -> None:
        cur = self._conn.execute(
            """INSERT OR REPLACE INTO protocol_results
               (run_id, nct_id, file_sha, status, error_message,
                elapsed_seconds, toc_section_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                result.nct_id,
                result.file_sha,
                result.status,
                result.error_message or None,
                result.elapsed_seconds,
                result.toc_section_count,
            ),
        )
        result_id = cur.lastrowid

        # Section matches
        for sm in result.section_matches:
            self._conn.execute(
                """INSERT INTO section_matches
                   (result_id, protocol_section_number,
                    protocol_section_title, ich_section_code,
                    ich_section_name, similarity_score,
                    boosted_score, confidence, match_method,
                    auto_mapped)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    result_id,
                    sm.get("protocol_section_number", ""),
                    sm.get("protocol_section_title", ""),
                    sm.get("ich_section_code", ""),
                    sm.get("ich_section_name", ""),
                    sm.get("similarity_score", 0.0),
                    sm.get("boosted_score", 0.0),
                    sm.get("confidence", ""),
                    sm.get("match_method", ""),
                    1 if sm.get("auto_mapped") else 0,
                ),
            )

        # Query extractions
        for qe in result.query_extractions:
            self._conn.execute(
                """INSERT INTO query_extractions
                   (result_id, query_id, section_id,
                    extracted_content, verbatim_content,
                    confidence, extraction_method, source_section)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    result_id,
                    qe.get("query_id", ""),
                    qe.get("section_id", ""),
                    qe.get("content", ""),
                    qe.get("verbatim_content", ""),
                    qe.get("confidence", 0.0),
                    qe.get("extraction_method", ""),
                    qe.get("source_section", ""),
                ),
            )

        # Coverage report
        if result.coverage:
            gap_sections = result.coverage.get("gap_sections", [])
            self._conn.execute(
                """INSERT INTO coverage_reports
                   (result_id, total_sections, populated_count,
                    gap_count, average_confidence,
                    total_queries, answered_queries, gap_sections)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    result_id,
                    result.coverage.get("total_sections", 0),
                    result.coverage.get("populated_count", 0),
                    result.coverage.get("gap_count", 0),
                    result.coverage.get("average_confidence", 0.0),
                    result.coverage.get("total_queries", 0),
                    result.coverage.get("answered_queries", 0),
                    json.dumps(gap_sections),
                ),
            )

        # Comparison pairs
        for cp in result.comparison_pairs:
            self._conn.execute(
                """INSERT INTO comparison_pairs
                   (result_id, ich_section_code, original_text,
                    extracted_text, confidence, match_quality)
                   VALUES (?,?,?,?,?,?)""",
                (
                    result_id,
                    cp.get("ich_section_code", ""),
                    cp.get("original_text", ""),
                    cp.get("extracted_text", ""),
                    cp.get("confidence", 0.0),
                    cp.get("match_quality", ""),
                ),
            )

        self._conn.commit()

    def has_result(self, run_id: str, nct_id: str) -> bool:
        cur = self._conn.execute(
            """SELECT 1 FROM protocol_results
               WHERE run_id = ? AND nct_id = ? LIMIT 1""",
            (run_id, nct_id),
        )
        return cur.fetchone() is not None

    def finalize_run(
        self, run_id: str, summary: dict[str, Any],
    ) -> None:
        self._conn.execute(
            """UPDATE batch_runs SET
               protocol_count = ?,
               pass_count = ?,
               fail_count = ?,
               error_count = ?,
               elapsed_seconds = ?
               WHERE run_id = ?""",
            (
                summary.get("protocol_count", 0),
                summary.get("pass_count", 0),
                summary.get("fail_count", 0),
                summary.get("error_count", 0),
                summary.get("elapsed_seconds", 0.0),
                run_id,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get batch run metadata."""
        cur = self._conn.execute(
            "SELECT * FROM batch_runs WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_run_id(self) -> str | None:
        """Get the most recent run_id by timestamp."""
        cur = self._conn.execute(
            "SELECT run_id FROM batch_runs ORDER BY timestamp DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row["run_id"] if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        """List all batch runs, most recent first."""
        cur = self._conn.execute(
            "SELECT * FROM batch_runs ORDER BY timestamp DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def list_ich_sections(
        self, run_id: str,
    ) -> list[dict[str, str]]:
        """List distinct ICH section codes and names in a run."""
        cur = self._conn.execute(
            """SELECT DISTINCT sm.ich_section_code,
                      sm.ich_section_name
               FROM section_matches sm
               JOIN protocol_results pr ON sm.result_id = pr.id
               WHERE pr.run_id = ?
               ORDER BY sm.ich_section_code""",
            (run_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_protocol_summary(
        self, run_id: str, nct_id: str,
    ) -> dict[str, Any]:
        """Get all results for one protocol in a run."""
        cur = self._conn.execute(
            """SELECT * FROM protocol_results
               WHERE run_id = ? AND nct_id = ?""",
            (run_id, nct_id),
        )
        row = cur.fetchone()
        if row is None:
            return {}

        result_id = row["id"]
        result: dict[str, Any] = dict(row)

        # Section matches
        cur = self._conn.execute(
            "SELECT * FROM section_matches WHERE result_id = ?",
            (result_id,),
        )
        result["section_matches"] = [dict(r) for r in cur.fetchall()]

        # Coverage
        cur = self._conn.execute(
            "SELECT * FROM coverage_reports WHERE result_id = ?",
            (result_id,),
        )
        cov_row = cur.fetchone()
        result["coverage"] = dict(cov_row) if cov_row else {}

        # Comparison pairs
        cur = self._conn.execute(
            "SELECT * FROM comparison_pairs WHERE result_id = ?",
            (result_id,),
        )
        result["comparison_pairs"] = [dict(r) for r in cur.fetchall()]

        return result

    def get_low_confidence_sections(
        self, run_id: str, threshold: float = 0.60,
    ) -> list[dict[str, Any]]:
        """Get ICH sections with avg confidence below threshold.

        Returns list sorted by ascending average confidence.
        """
        cur = self._conn.execute(
            """SELECT
                 sm.ich_section_code,
                 sm.ich_section_name,
                 AVG(sm.similarity_score) AS avg_similarity,
                 AVG(sm.boosted_score) AS avg_boosted,
                 COUNT(*) AS protocol_count,
                 SUM(CASE WHEN sm.auto_mapped = 1 THEN 1 ELSE 0 END)
                     AS auto_mapped_count
               FROM section_matches sm
               JOIN protocol_results pr ON sm.result_id = pr.id
               WHERE pr.run_id = ?
               GROUP BY sm.ich_section_code, sm.ich_section_name
               HAVING AVG(sm.boosted_score) < ?
               ORDER BY avg_boosted ASC""",
            (run_id, threshold),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_section_stats(
        self, run_id: str,
    ) -> list[dict[str, Any]]:
        """Get per-ICH-section statistics across the corpus.

        Returns one row per ICH section with: avg confidence,
        hit rate, miss rate, protocol count.
        """
        # Total protocols in this run
        total_cur = self._conn.execute(
            """SELECT COUNT(*) AS total FROM protocol_results
               WHERE run_id = ? AND status = 'pass'""",
            (run_id,),
        )
        total_protocols = total_cur.fetchone()["total"]

        cur = self._conn.execute(
            """SELECT
                 sm.ich_section_code,
                 sm.ich_section_name,
                 COUNT(DISTINCT pr.nct_id) AS protocol_count,
                 AVG(sm.similarity_score) AS avg_similarity,
                 AVG(sm.boosted_score) AS avg_boosted,
                 MIN(sm.boosted_score) AS min_boosted,
                 MAX(sm.boosted_score) AS max_boosted,
                 SUM(CASE WHEN sm.auto_mapped = 1 THEN 1 ELSE 0 END)
                     AS auto_mapped_count,
                 SUM(CASE WHEN sm.auto_mapped = 0 THEN 1 ELSE 0 END)
                     AS review_count
               FROM section_matches sm
               JOIN protocol_results pr ON sm.result_id = pr.id
               WHERE pr.run_id = ?
               GROUP BY sm.ich_section_code, sm.ich_section_name
               ORDER BY sm.ich_section_code""",
            (run_id,),
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["total_protocols"] = total_protocols
            d["hit_rate"] = (
                d["protocol_count"] / total_protocols
                if total_protocols > 0 else 0.0
            )
            d["auto_map_rate"] = (
                d["auto_mapped_count"] / d["protocol_count"]
                if d["protocol_count"] > 0 else 0.0
            )
            rows.append(d)
        return rows

    def get_misclassification_patterns(
        self, run_id: str,
    ) -> list[dict[str, Any]]:
        """Find sections where multiple ICH codes are matched.

        A protocol section that matches multiple ICH codes may
        indicate ambiguity or misclassification.
        """
        cur = self._conn.execute(
            """SELECT
                 sm.protocol_section_title,
                 sm.ich_section_code,
                 COUNT(*) AS occurrence_count,
                 AVG(sm.boosted_score) AS avg_score
               FROM section_matches sm
               JOIN protocol_results pr ON sm.result_id = pr.id
               WHERE pr.run_id = ?
               GROUP BY sm.protocol_section_title, sm.ich_section_code
               HAVING COUNT(*) >= 2
               ORDER BY occurrence_count DESC, avg_score ASC""",
            (run_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_coverage_distribution(
        self, run_id: str,
    ) -> dict[str, Any]:
        """Get histogram of coverage percentages across protocols."""
        cur = self._conn.execute(
            """SELECT
                 cr.total_sections,
                 cr.populated_count,
                 cr.average_confidence,
                 cr.gap_count,
                 pr.nct_id
               FROM coverage_reports cr
               JOIN protocol_results pr ON cr.result_id = pr.id
               WHERE pr.run_id = ?
               ORDER BY cr.average_confidence ASC""",
            (run_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return {
                "protocol_count": 0,
                "avg_coverage_pct": 0.0,
                "avg_confidence": 0.0,
                "buckets": {},
                "protocols": [],
            }

        # Build coverage percentage buckets
        buckets: dict[str, int] = {
            "0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0,
        }
        coverage_pcts: list[float] = []
        confidences: list[float] = []

        for r in rows:
            total = r["total_sections"] or 1
            pct = r["populated_count"] / total * 100
            coverage_pcts.append(pct)
            confidences.append(r["average_confidence"])
            if pct < 25:
                buckets["0-25%"] += 1
            elif pct < 50:
                buckets["25-50%"] += 1
            elif pct < 75:
                buckets["50-75%"] += 1
            else:
                buckets["75-100%"] += 1

        return {
            "protocol_count": len(rows),
            "avg_coverage_pct": round(
                sum(coverage_pcts) / len(coverage_pcts), 2,
            ),
            "avg_confidence": round(
                sum(confidences) / len(confidences), 4,
            ),
            "buckets": buckets,
            "protocols": rows,
        }

    def compare_runs(
        self, run_id_a: str, run_id_b: str,
    ) -> dict[str, Any]:
        """Compare two batch runs, highlighting improvements.

        Returns per-section deltas in confidence and coverage.
        """
        stats_a = {
            s["ich_section_code"]: s
            for s in self.get_section_stats(run_id_a)
        }
        stats_b = {
            s["ich_section_code"]: s
            for s in self.get_section_stats(run_id_b)
        }

        all_codes = sorted(
            set(stats_a.keys()) | set(stats_b.keys())
        )

        sections: list[dict[str, Any]] = []
        for code in all_codes:
            a = stats_a.get(code, {})
            b = stats_b.get(code, {})
            a_score = a.get("avg_boosted", 0.0)
            b_score = b.get("avg_boosted", 0.0)
            delta = round(b_score - a_score, 4)
            if delta > 0.01:
                status = "improved"
            elif delta < -0.01:
                status = "regressed"
            else:
                status = "unchanged"
            sections.append({
                "ich_section_code": code,
                "run_a_avg_score": round(a_score, 4),
                "run_b_avg_score": round(b_score, 4),
                "delta": delta,
                "status": status,
                "run_a_count": a.get("protocol_count", 0),
                "run_b_count": b.get("protocol_count", 0),
            })

        # Aggregate coverage comparison
        cov_a = self.get_coverage_distribution(run_id_a)
        cov_b = self.get_coverage_distribution(run_id_b)

        return {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "coverage_delta": round(
                cov_b.get("avg_coverage_pct", 0.0)
                - cov_a.get("avg_coverage_pct", 0.0),
                2,
            ),
            "confidence_delta": round(
                cov_b.get("avg_confidence", 0.0)
                - cov_a.get("avg_confidence", 0.0),
                4,
            ),
            "sections": sections,
        }

    def get_comparison_pairs(
        self,
        run_id: str,
        nct_id: str,
        ich_section_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get original vs extracted text pairs for debugging."""
        if ich_section_code:
            cur = self._conn.execute(
                """SELECT cp.* FROM comparison_pairs cp
                   JOIN protocol_results pr ON cp.result_id = pr.id
                   WHERE pr.run_id = ? AND pr.nct_id = ?
                     AND cp.ich_section_code = ?""",
                (run_id, nct_id, ich_section_code),
            )
        else:
            cur = self._conn.execute(
                """SELECT cp.* FROM comparison_pairs cp
                   JOIN protocol_results pr ON cp.result_id = pr.id
                   WHERE pr.run_id = ? AND pr.nct_id = ?""",
                (run_id, nct_id),
            )
        return [dict(r) for r in cur.fetchall()]

    def list_protocols(
        self, run_id: str, status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all protocols in a run, optionally filtered by status."""
        if status:
            cur = self._conn.execute(
                """SELECT nct_id, file_sha, status, elapsed_seconds,
                          toc_section_count
                   FROM protocol_results
                   WHERE run_id = ? AND status = ?
                   ORDER BY nct_id""",
                (run_id, status),
            )
        else:
            cur = self._conn.execute(
                """SELECT nct_id, file_sha, status, elapsed_seconds,
                          toc_section_count
                   FROM protocol_results
                   WHERE run_id = ?
                   ORDER BY nct_id""",
                (run_id,),
            )
        return [dict(r) for r in cur.fetchall()]
