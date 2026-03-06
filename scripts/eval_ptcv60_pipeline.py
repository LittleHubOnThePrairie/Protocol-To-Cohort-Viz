"""Evaluate PTCV-60 document-first pipeline on sample protocol PDFs.

New pipeline order:
  1. Extraction (PDF → text blocks + tables)
  2. SoA Extraction (document-first: tables → PDF discovery → text)
  3. LLM Retemplating (Claude Opus 4.6 → ICH sections)  [optional]
  4. Coverage Review (deterministic text overlap)            [optional]

Stages 3-4 require ANTHROPIC_API_KEY. If unset, script reports
SoA results only.

Usage: cd C:/Dev/PTCV && python scripts/eval_ptcv60_pipeline.py
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pyarrow.parquet as pq

from ptcv.extraction.extraction_service import ExtractionService
from ptcv.extraction.models import ExtractedTable
from ptcv.ich_parser.review_queue import ReviewQueue
from ptcv.soa_extractor.extractor import SoaExtractor
from ptcv.storage.filesystem_adapter import FilesystemAdapter

SAMPLE_SIZE = 5
SEED = 42

HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


def main() -> None:
    pdf_dir = Path("C:/Dev/PTCV/data/protocols/clinicaltrials")
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    random.seed(SEED)
    sample = random.sample(all_pdfs, SAMPLE_SIZE)

    tmp = Path(tempfile.mkdtemp())
    gateway = FilesystemAdapter(root=tmp / "eval")
    gateway.initialise()
    review_queue = ReviewQueue(db_path=tmp / "review_queue.db")
    review_queue.initialise()

    extraction_svc = ExtractionService(gateway=gateway)
    soa_extractor = SoaExtractor(
        gateway=gateway, review_queue=review_queue,
    )

    # Optional: LLM retemplater + coverage reviewer
    retemplater = None
    if HAS_ANTHROPIC_KEY:
        from ptcv.ich_parser.llm_retemplater import LlmRetemplater
        retemplater = LlmRetemplater(
            gateway=gateway, review_queue=review_queue,
        )
        print("LLM retemplating ENABLED (ANTHROPIC_API_KEY found)\n")
    else:
        print(
            "LLM retemplating DISABLED "
            "(set ANTHROPIC_API_KEY to enable stages 3-4)\n"
        )

    from ptcv.ich_parser.coverage_reviewer import CoverageReviewer
    coverage_reviewer = CoverageReviewer()

    results: list[dict] = []

    for pdf_path in sample:
        reg_id = pdf_path.stem.rsplit("_", 1)[0]
        print(f"\n{'=' * 70}")
        print(f"Processing: {pdf_path.name} ({reg_id})")
        print("=" * 70)

        t0 = time.time()
        pdf_bytes = pdf_path.read_bytes()
        sha = hashlib.sha256(pdf_bytes).hexdigest()

        try:
            # ── Stage 1: Extraction ──────────────────────────────
            ext_result = extraction_svc.extract(
                protocol_data=pdf_bytes,
                registry_id=reg_id,
                amendment_number="00",
                source_sha256=sha,
                filename=pdf_path.name,
                source="ClinicalTrials.gov",
            )
            print(
                f"  [1] Extraction: {ext_result.format_detected}, "
                f"{ext_result.text_block_count} text blocks, "
                f"{ext_result.table_count} tables"
            )

            # Read text blocks from parquet
            tb_bytes = gateway.get_artifact(ext_result.text_artifact_key)
            tb_df = pq.read_table(io.BytesIO(tb_bytes)).to_pandas()
            text_block_dicts = tb_df.to_dict("records")

            # Read extracted tables for bridge
            extracted_tables = None
            if ext_result.tables_artifact_key:
                tbl_bytes = gateway.get_artifact(
                    ext_result.tables_artifact_key,
                )
                tbl_df = pq.read_table(
                    io.BytesIO(tbl_bytes),
                ).to_pandas()
                if not tbl_df.empty:
                    extracted_tables = []
                    for _, row in tbl_df.iterrows():
                        extracted_tables.append(
                            ExtractedTable(
                                run_id=str(row.get("run_id", "")),
                                source_registry_id=str(
                                    row.get("source_registry_id", ""),
                                ),
                                source_sha256=str(
                                    row.get("source_sha256", ""),
                                ),
                                page_number=int(
                                    row.get("page_number", 0),
                                ),
                                extractor_used=str(
                                    row.get("extractor_used", ""),
                                ),
                                table_index=int(
                                    row.get("table_index", 0),
                                ),
                                header_row=str(
                                    row.get("header_row", "[]"),
                                ),
                                data_rows=str(
                                    row.get("data_rows", "[]"),
                                ),
                            )
                        )

            # Get page count
            meta_bytes = gateway.get_artifact(
                ext_result.metadata_artifact_key,
            )
            meta_df = pq.read_table(io.BytesIO(meta_bytes)).to_pandas()
            page_count = (
                int(meta_df["page_count"].iloc[0])
                if "page_count" in meta_df.columns
                else 0
            )

            # ── Stage 2: SoA Extraction (document-first) ────────
            soa_result = soa_extractor.extract(
                registry_id=reg_id,
                source_run_id=ext_result.run_id,
                source_sha256=ext_result.text_artifact_sha256,
                pdf_bytes=pdf_bytes,
                text_blocks=text_block_dicts,
                page_count=page_count,
                extracted_tables=extracted_tables,
                # NOTE: no sections= arg — document-first!
            )
            print(
                f"  [2] SoA Extract: "
                f"{soa_result.timepoint_count} timepoints, "
                f"{soa_result.activity_count} activities, "
                f"{soa_result.epoch_count} epochs, "
                f"{soa_result.instance_count} instances"
            )
            print(
                f"      Review queue: {soa_result.review_count} uncertain"
            )

            # Read timepoints for detail
            tp_key = soa_result.artifact_keys.get("timepoints", "")
            tp_details: list[dict] = []
            if tp_key:
                tp_bytes = gateway.get_artifact(tp_key)
                tp_df = pq.read_table(
                    io.BytesIO(tp_bytes),
                ).to_pandas()
                for _, row in tp_df.iterrows():
                    tp_details.append({
                        "visit_name": str(row.get("visit_name", "")),
                        "visit_type": str(row.get("visit_type", "")),
                        "day_offset": row.get("day_offset"),
                        "mandatory": row.get("mandatory"),
                    })
                if tp_details:
                    print("      Visits:")
                    for tp in tp_details[:10]:
                        vn = tp["visit_name"][:30]
                        vt = tp["visit_type"][:20]
                        dy = tp["day_offset"]
                        print(
                            f"        {vn:30s} type={vt:20s} "
                            f"day={dy}"
                        )
                    if len(tp_details) > 10:
                        print(
                            f"        ... and "
                            f"{len(tp_details) - 10} more"
                        )

            # Build result dict (stages 1-2)
            row_data: dict = {
                "file": pdf_path.name,
                "registry_id": reg_id,
                "format": ext_result.format_detected,
                "text_blocks": ext_result.text_block_count,
                "tables": ext_result.table_count,
                "page_count": page_count,
                "timepoints": soa_result.timepoint_count,
                "activities": soa_result.activity_count,
                "epochs": soa_result.epoch_count,
                "instances": soa_result.instance_count,
                "review_count": soa_result.review_count,
                "tp_details": tp_details,
                "error": None,
            }

            # ── Stage 3: LLM Retemplating (optional) ────────────
            if retemplater is not None and text_block_dicts:
                soa_summary = None
                if soa_result.timepoint_count > 0:
                    visit_names = [
                        tp["visit_name"] for tp in tp_details
                    ]
                    soa_summary = {
                        "visit_count": soa_result.timepoint_count,
                        "activity_count": soa_result.activity_count,
                        "visit_names": visit_names[:20],
                    }

                retemplate_result = retemplater.retemplate(
                    text_blocks=text_block_dicts,
                    registry_id=reg_id,
                    source_run_id=ext_result.run_id,
                    source_sha256=ext_result.text_artifact_sha256,
                    soa_summary=soa_summary,
                )
                print(
                    f"  [3] Retemplating: "
                    f"{retemplate_result.format_verdict} "
                    f"(conf={retemplate_result.format_confidence:.2f}), "
                    f"{retemplate_result.section_count} sections, "
                    f"{retemplate_result.input_tokens} in / "
                    f"{retemplate_result.output_tokens} out tokens"
                )
                if retemplate_result.missing_required_sections:
                    print(
                        f"      Missing: "
                        f"{retemplate_result.missing_required_sections}"
                    )

                row_data["retemplating_verdict"] = (
                    retemplate_result.format_verdict
                )
                row_data["retemplating_confidence"] = (
                    retemplate_result.format_confidence
                )
                row_data["retemplating_sections"] = (
                    retemplate_result.section_count
                )
                row_data["retemplating_input_tokens"] = (
                    retemplate_result.input_tokens
                )
                row_data["retemplating_output_tokens"] = (
                    retemplate_result.output_tokens
                )

                # ── Stage 4: Coverage Review ─────────────────────
                # Read retemplated sections back from storage
                from ptcv.ich_parser.models import IchSection
                sec_bytes = gateway.get_artifact(
                    retemplate_result.artifact_key,
                )
                sec_df = pq.read_table(
                    io.BytesIO(sec_bytes),
                ).to_pandas()
                retemplated_sections = []
                for _, srow in sec_df.iterrows():
                    retemplated_sections.append(
                        IchSection(
                            run_id=str(srow.get("run_id", "")),
                            source_run_id=str(
                                srow.get("source_run_id", ""),
                            ),
                            source_sha256=str(
                                srow.get("source_sha256", ""),
                            ),
                            registry_id=str(
                                srow.get("registry_id", ""),
                            ),
                            section_code=str(
                                srow.get("section_code", ""),
                            ),
                            section_name=str(
                                srow.get("section_name", ""),
                            ),
                            content_json=str(
                                srow.get("content_json", "{}"),
                            ),
                            confidence_score=float(
                                srow.get("confidence_score", 0.0),
                            ),
                            review_required=bool(
                                srow.get("review_required", False),
                            ),
                            legacy_format=bool(
                                srow.get("legacy_format", False),
                            ),
                            extraction_timestamp_utc=str(
                                srow.get(
                                    "extraction_timestamp_utc", "",
                                ),
                            ),
                        )
                    )

                cov_result = coverage_reviewer.review(
                    original_text_blocks=text_block_dicts,
                    retemplated_sections=retemplated_sections,
                )
                print(
                    f"  [4] Coverage: "
                    f"{cov_result.coverage_score:.1%} "
                    f"({'PASS' if cov_result.passed else 'FAIL'}), "
                    f"{cov_result.covered_chars}/{cov_result.total_original_chars} chars"
                )
                if cov_result.uncovered_blocks:
                    print(
                        f"      Uncovered blocks: "
                        f"{len(cov_result.uncovered_blocks)}"
                    )
                if cov_result.section_coverage:
                    top_sections = sorted(
                        cov_result.section_coverage.items(),
                        key=lambda x: -x[1],
                    )[:5]
                    for code, chars in top_sections:
                        print(f"        {code}: {chars:,} chars")

                row_data["coverage_score"] = cov_result.coverage_score
                row_data["coverage_passed"] = cov_result.passed
                row_data["uncovered_blocks"] = len(
                    cov_result.uncovered_blocks,
                )

            elapsed = time.time() - t0
            row_data["elapsed"] = elapsed
            print(f"  Elapsed: {elapsed:.1f}s")
            results.append(row_data)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "file": pdf_path.name,
                "registry_id": reg_id,
                "error": str(e),
                "elapsed": elapsed,
            })

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("PTCV-60 DOCUMENT-FIRST PIPELINE EVALUATION")
    print("=" * 70)
    successful = [r for r in results if r.get("error") is None]
    failed = [r for r in results if r.get("error") is not None]
    print(
        f"Total: {len(results)} | "
        f"Success: {len(successful)} | "
        f"Failed: {len(failed)}"
    )

    if successful:
        tp_counts = [r["timepoints"] for r in successful]
        act_counts = [r["activities"] for r in successful]
        inst_counts = [r["instances"] for r in successful]
        print(
            f"\nSoA Extraction (document-first, no ICH sections):"
        )
        print(
            f"  Timepoints: min={min(tp_counts)}, "
            f"max={max(tp_counts)}, "
            f"avg={sum(tp_counts) / len(tp_counts):.1f}"
        )
        print(
            f"  Activities: min={min(act_counts)}, "
            f"max={max(act_counts)}, "
            f"avg={sum(act_counts) / len(act_counts):.1f}"
        )
        print(
            f"  Instances:  min={min(inst_counts)}, "
            f"max={max(inst_counts)}, "
            f"avg={sum(inst_counts) / len(inst_counts):.1f}"
        )

        has_soa = [r for r in successful if r["timepoints"] > 0]
        no_soa = [r for r in successful if r["timepoints"] == 0]
        print(f"\n  SoA found: {len(has_soa)}/{len(successful)}")
        if no_soa:
            names = [r["file"] for r in no_soa]
            print(f"  No SoA: {names}")

        # Retemplating summary (if available)
        has_retemplate = [
            r for r in successful
            if "retemplating_verdict" in r
        ]
        if has_retemplate:
            print(f"\nLLM Retemplating:")
            total_in = sum(
                r.get("retemplating_input_tokens", 0)
                for r in has_retemplate
            )
            total_out = sum(
                r.get("retemplating_output_tokens", 0)
                for r in has_retemplate
            )
            verdicts = [r["retemplating_verdict"] for r in has_retemplate]
            print(
                f"  Verdicts: "
                f"{', '.join(f'{v}' for v in verdicts)}"
            )
            print(
                f"  Total tokens: {total_in:,} in / "
                f"{total_out:,} out"
            )

        # Coverage summary (if available)
        has_coverage = [
            r for r in successful if "coverage_score" in r
        ]
        if has_coverage:
            scores = [r["coverage_score"] for r in has_coverage]
            passed = sum(1 for r in has_coverage if r["coverage_passed"])
            print(f"\nCoverage Review:")
            print(
                f"  Scores: min={min(scores):.1%}, "
                f"max={max(scores):.1%}, "
                f"avg={sum(scores) / len(scores):.1%}"
            )
            print(
                f"  Passed: {passed}/{len(has_coverage)}"
            )

        # Per-protocol table
        print(f"\nPer-protocol details:")
        header = (
            f"  {'File':40s} {'Pages':>5s} {'TxtBlk':>6s} "
            f"{'Tbls':>4s} {'TP':>3s} {'Act':>3s} "
            f"{'Inst':>4s}"
        )
        if has_retemplate:
            header += (
                f" {'Verdict':>12s} {'Conf':>5s} "
                f"{'Secs':>4s} {'Cov':>5s}"
            )
        header += f" {'Time':>6s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for r in successful:
            fn = r["file"][:40]
            pg = r.get("page_count", 0)
            tb = r["text_blocks"]
            tbl = r["tables"]
            tp = r["timepoints"]
            ac = r["activities"]
            ins = r["instances"]
            line = (
                f"  {fn:40s} {pg:5d} {tb:6d} "
                f"{tbl:4d} {tp:3d} {ac:3d} {ins:4d}"
            )
            if "retemplating_verdict" in r:
                rv = r["retemplating_verdict"][:12]
                rc = r.get("retemplating_confidence", 0)
                rs = r.get("retemplating_sections", 0)
                cv = r.get("coverage_score", 0)
                line += (
                    f" {rv:>12s} {rc:5.2f} "
                    f"{rs:4d} {cv:5.1%}"
                )
            el = r["elapsed"]
            line += f" {el:6.1f}s"
            print(line)

        # Visit type distribution
        all_types: dict[str, int] = {}
        for r in successful:
            for tp in r.get("tp_details", []):
                vt = tp["visit_type"]
                all_types[vt] = all_types.get(vt, 0) + 1
        if all_types:
            print("\nVisit type distribution:")
            for vt, count in sorted(
                all_types.items(), key=lambda x: -x[1],
            ):
                print(f"  {vt:25s} {count:4d}")

    if failed:
        print("\nFailed protocols:")
        for r in failed:
            err = str(r["error"])[:100]
            print(f"  {r['file']}: {err}")

    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
