"""Evaluate SoA extraction on a random sample of protocol PDFs.

Usage: cd C:/Dev/PTCV && python scripts/eval_soa_sample.py
"""

from __future__ import annotations

import hashlib
import io
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
from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.parser import IchParser
from ptcv.soa_extractor.extractor import SoaExtractor
from ptcv.storage.filesystem_adapter import FilesystemAdapter

SAMPLE_SIZE = 5
SEED = 42


def main() -> None:
    pdf_dir = Path("C:/Dev/PTCV/data/protocols/clinicaltrials")
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    random.seed(SEED)
    sample = random.sample(all_pdfs, SAMPLE_SIZE)

    tmp = Path(tempfile.mkdtemp())
    gateway = FilesystemAdapter(root=tmp / "eval")
    gateway.initialise()

    extraction_svc = ExtractionService(gateway=gateway)
    parser = IchParser(gateway=gateway)
    soa_extractor = SoaExtractor(gateway=gateway)

    results: list[dict] = []

    for pdf_path in sample:
        reg_id = pdf_path.stem.rsplit("_", 1)[0]
        print(f"\n{'=' * 60}")
        print(f"Processing: {pdf_path.name} ({reg_id})")
        print("=" * 60)

        t0 = time.time()
        pdf_bytes = pdf_path.read_bytes()
        sha = hashlib.sha256(pdf_bytes).hexdigest()

        try:
            # Stage 1: Extraction
            ext_result = extraction_svc.extract(
                protocol_data=pdf_bytes,
                registry_id=reg_id,
                amendment_number="00",
                source_sha256=sha,
                filename=pdf_path.name,
                source="ClinicalTrials.gov",
            )
            print(
                f"  Extraction: {ext_result.format_detected}, "
                f"{ext_result.text_block_count} text blocks, "
                f"{ext_result.table_count} tables"
            )

            # Read text blocks from parquet
            tb_bytes = gateway.get_artifact(ext_result.text_artifact_key)
            tb_df = pq.read_table(io.BytesIO(tb_bytes)).to_pandas()
            full_text = "\n".join(tb_df["text"].tolist())

            # Stage 2: ICH Parse
            parse_result = parser.parse(
                text=full_text,
                registry_id=reg_id,
                source_sha256=ext_result.text_artifact_sha256,
            )
            print(
                f"  ICH Parse: {parse_result.format_verdict} "
                f"(confidence={parse_result.format_confidence:.2f}), "
                f"{parse_result.section_count} sections"
            )
            if parse_result.missing_required_sections:
                print(
                    f"    Missing: {parse_result.missing_required_sections}"
                )

            # Read sections back
            sec_bytes = gateway.get_artifact(parse_result.artifact_key)
            sec_df = pq.read_table(io.BytesIO(sec_bytes)).to_pandas()

            sections = []
            for _, row in sec_df.iterrows():
                sections.append(
                    IchSection(
                        run_id=str(row.get("run_id", "")),
                        source_run_id=str(
                            row.get("source_run_id", "")
                        ),
                        source_sha256=str(
                            row.get("source_sha256", "")
                        ),
                        registry_id=str(
                            row.get("registry_id", "")
                        ),
                        section_code=str(
                            row.get("section_code", "")
                        ),
                        section_name=str(
                            row.get("section_name", "")
                        ),
                        content_json=str(
                            row.get("content_json", "{}")
                        ),
                        confidence_score=float(
                            row.get("confidence_score", 0.0)
                        ),
                        review_required=bool(
                            row.get("review_required", False)
                        ),
                        legacy_format=bool(
                            row.get("legacy_format", False)
                        ),
                        extraction_timestamp_utc=str(
                            row.get(
                                "extraction_timestamp_utc", ""
                            )
                        ),
                    )
                )

            # Read extracted tables for bridge
            extracted_tables = None
            if ext_result.tables_artifact_key:
                tbl_bytes = gateway.get_artifact(
                    ext_result.tables_artifact_key
                )
                tbl_df = pq.read_table(io.BytesIO(tbl_bytes)).to_pandas()
                if not tbl_df.empty:
                    extracted_tables = []
                    for _, row in tbl_df.iterrows():
                        extracted_tables.append(
                            ExtractedTable(
                                run_id=str(row.get("run_id", "")),
                                source_registry_id=str(
                                    row.get("source_registry_id", "")
                                ),
                                source_sha256=str(
                                    row.get("source_sha256", "")
                                ),
                                page_number=int(
                                    row.get("page_number", 0)
                                ),
                                extractor_used=str(
                                    row.get("extractor_used", "")
                                ),
                                table_index=int(
                                    row.get("table_index", 0)
                                ),
                                header_row=str(
                                    row.get("header_row", "[]")
                                ),
                                data_rows=str(
                                    row.get("data_rows", "[]")
                                ),
                            )
                        )

            # Get page count from metadata
            meta_bytes = gateway.get_artifact(
                ext_result.metadata_artifact_key
            )
            meta_df = pq.read_table(io.BytesIO(meta_bytes)).to_pandas()
            page_count = (
                int(meta_df["page_count"].iloc[0])
                if "page_count" in meta_df.columns
                else 0
            )

            # Stage 3: SoA Extraction
            soa_result = soa_extractor.extract(
                sections=sections,
                registry_id=reg_id,
                source_run_id=parse_result.run_id,
                source_sha256=parse_result.artifact_sha256,
                pdf_bytes=pdf_bytes,
                text_blocks=tb_df.to_dict("records"),
                page_count=page_count,
                extracted_tables=extracted_tables,
            )
            elapsed = time.time() - t0
            print(
                f"  SoA Extract: {soa_result.timepoint_count} timepoints, "
                f"{soa_result.activity_count} activities, "
                f"{soa_result.epoch_count} epochs, "
                f"{soa_result.instance_count} instances"
            )
            print(
                f"  Review queue: {soa_result.review_count} uncertain"
            )
            print(f"  Elapsed: {elapsed:.1f}s")

            # Read timepoints for deeper inspection
            tp_key = soa_result.artifact_keys.get("timepoints", "")
            tp_details: list[dict] = []
            if tp_key:
                tp_bytes = gateway.get_artifact(tp_key)
                tp_df = pq.read_table(
                    io.BytesIO(tp_bytes)
                ).to_pandas()
                for _, row in tp_df.iterrows():
                    tp_details.append(
                        {
                            "visit_name": str(
                                row.get("visit_name", "")
                            ),
                            "visit_type": str(
                                row.get("visit_type", "")
                            ),
                            "day_offset": row.get("day_offset"),
                            "mandatory": row.get("mandatory"),
                        }
                    )
                if tp_details:
                    print("  Visits:")
                    for tp in tp_details[:12]:
                        vn = tp["visit_name"][:30]
                        vt = tp["visit_type"][:20]
                        dy = tp["day_offset"]
                        mn = tp["mandatory"]
                        print(
                            f"    {vn:30s} type={vt:20s} "
                            f"day={dy:>6} mand={mn}"
                        )
                    if len(tp_details) > 12:
                        extra = len(tp_details) - 12
                        print(f"    ... and {extra} more")

            results.append(
                {
                    "file": pdf_path.name,
                    "registry_id": reg_id,
                    "format": ext_result.format_detected,
                    "text_blocks": ext_result.text_block_count,
                    "tables": ext_result.table_count,
                    "ich_verdict": parse_result.format_verdict,
                    "ich_confidence": parse_result.format_confidence,
                    "sections": parse_result.section_count,
                    "timepoints": soa_result.timepoint_count,
                    "activities": soa_result.activity_count,
                    "epochs": soa_result.epoch_count,
                    "instances": soa_result.instance_count,
                    "review_count": soa_result.review_count,
                    "elapsed": elapsed,
                    "tp_details": tp_details,
                    "error": None,
                }
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append(
                {
                    "file": pdf_path.name,
                    "registry_id": reg_id,
                    "error": str(e),
                    "elapsed": elapsed,
                }
            )

    # Summary
    print(f"\n\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print("=" * 60)
    successful = [r for r in results if r.get("error") is None]
    failed = [r for r in results if r.get("error") is not None]
    print(
        f"Total: {len(results)} | Success: {len(successful)} "
        f"| Failed: {len(failed)}"
    )

    if successful:
        tp_counts = [r["timepoints"] for r in successful]
        act_counts = [r["activities"] for r in successful]
        inst_counts = [r["instances"] for r in successful]
        print(
            f"\nTimepoints: min={min(tp_counts)}, "
            f"max={max(tp_counts)}, "
            f"avg={sum(tp_counts) / len(tp_counts):.1f}"
        )
        print(
            f"Activities: min={min(act_counts)}, "
            f"max={max(act_counts)}, "
            f"avg={sum(act_counts) / len(act_counts):.1f}"
        )
        print(
            f"Instances:  min={min(inst_counts)}, "
            f"max={max(inst_counts)}, "
            f"avg={sum(inst_counts) / len(inst_counts):.1f}"
        )

        has_soa = [r for r in successful if r["timepoints"] > 0]
        no_soa = [r for r in successful if r["timepoints"] == 0]
        print(f"\nSoA found: {len(has_soa)}/{len(successful)}")
        if no_soa:
            names = [r["file"] for r in no_soa]
            print(f"  No SoA: {names}")

        print("\nPer-protocol details:")
        for r in successful:
            fn = r["file"][:40]
            iv = r["ich_verdict"][:12]
            ic = r["ich_confidence"]
            tp = r["timepoints"]
            ac = r["activities"]
            ins = r["instances"]
            rv = r["review_count"]
            el = r["elapsed"]
            print(
                f"  {fn:40s} ICH={iv:12s} conf={ic:.2f} "
                f"tp={tp:3d} act={ac:3d} inst={ins:4d} "
                f"rev={rv:2d} {el:.1f}s"
            )

        # Visit type distribution
        all_types: dict[str, int] = {}
        for r in successful:
            for tp in r.get("tp_details", []):
                vt = tp["visit_type"]
                all_types[vt] = all_types.get(vt, 0) + 1
        if all_types:
            print("\nVisit type distribution:")
            for vt, count in sorted(
                all_types.items(), key=lambda x: -x[1]
            ):
                print(f"  {vt:25s} {count:4d}")

    if failed:
        print("\nFailed protocols:")
        for r in failed:
            err = r["error"][:100]
            print(f"  {r['file']}: {err}")

    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
