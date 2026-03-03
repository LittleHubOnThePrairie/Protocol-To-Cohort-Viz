"""Batch pre-parse script for protocol quality scoring (PTCV-42).

Runs ExtractionService + IchParser on all protocol PDFs that do not
yet have ``quality_scores`` in their metadata JSON, then writes the
enriched metadata back.

Usage::

    cd C:/Dev/PTCV
    python -m scripts.batch_parse [--dry-run] [--limit N]

Estimated runtime: ~2-5 seconds per PDF.  For 182 unscored PDFs,
expect 10-15 minutes.

Regulatory:
- Audit trail: pipeline runs recorded via StorageGateway lineage.
- Metadata updates are non-destructive (additive fields only).
- Original ``file_hash_sha256`` is verified before processing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Reconfigure stdout to UTF-8 for non-ASCII trial titles on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running as ``python -m scripts.batch_parse`` from project root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ptcv.extraction import ExtractionService, parquet_to_text_blocks
from ptcv.ich_parser import IchParser
from ptcv.storage import FilesystemAdapter
from ptcv.ui.components.protocol_catalog import classify_therapeutic_area

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("C:/Dev/PTCV/data")
_PROTOCOLS_DIR = _DATA_ROOT / "protocols"
_METADATA_DIR = _PROTOCOLS_DIR / "metadata"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _needs_scoring(meta: dict) -> bool:
    """Return True if metadata lacks quality_scores."""
    qs = meta.get("quality_scores")
    if not qs:
        return True
    return not qs.get("format_verdict")


def _resolve_pdf_path(meta: dict) -> Optional[Path]:
    """Find the PDF file for a metadata record.

    Args:
        meta: Parsed metadata dict.

    Returns:
        Path to PDF or None if not found.
    """
    registry_id = meta.get("registry_id", "")
    amendment = meta.get("amendment_number", meta.get("version", "1.0"))
    expected = f"{registry_id}_{amendment}.pdf"

    for source in ("clinicaltrials", "eu-ctr"):
        candidate = _PROTOCOLS_DIR / source / expected
        if candidate.is_file():
            return candidate

    fp = meta.get("file_path", "")
    if fp:
        p = Path(fp)
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Single-protocol parse
# ---------------------------------------------------------------------------

def parse_single(
    meta: dict,
    pdf_path: Path,
    gateway: FilesystemAdapter,
) -> dict:
    """Run extraction + ICH parse on a single protocol.

    Args:
        meta: Parsed metadata dict.
        pdf_path: Path to the protocol PDF.
        gateway: Initialised FilesystemAdapter.

    Returns:
        Dict with ``quality_scores`` and ``therapeutic_area``.

    Raises:
        Exception: If extraction or parsing fails.
    """
    registry_id = meta.get("registry_id", pdf_path.stem.split("_")[0])
    amendment = meta.get("amendment_number", meta.get("version", "1.0"))

    pdf_bytes = pdf_path.read_bytes()
    source_sha = hashlib.sha256(pdf_bytes).hexdigest()

    # Stage 1: Extraction
    extraction_svc = ExtractionService(gateway=gateway)
    extraction_result = extraction_svc.extract(
        protocol_data=pdf_bytes,
        registry_id=registry_id,
        amendment_number=amendment,
        source_sha256=source_sha,
        filename=pdf_path.name,
        source=meta.get("source", "batch-parse"),
    )

    # Read text blocks back from storage
    text_bytes = gateway.get_artifact(
        extraction_result.text_artifact_key
    )
    text_blocks = parquet_to_text_blocks(text_bytes)
    text_blocks_sorted = sorted(
        text_blocks, key=lambda b: (b.page_number, b.block_index)
    )
    protocol_text = "\n".join(
        b.text for b in text_blocks_sorted if b.text.strip()
    )

    if not protocol_text.strip():
        protocol_text = (
            f"Protocol {registry_id} "
            "(text extraction produced no content)"
        )

    # Stage 2: ICH Parse
    parser = IchParser(gateway=gateway)
    parse_result = parser.parse(
        text=protocol_text,
        registry_id=registry_id,
        source_run_id=extraction_result.run_id,
        source_sha256=extraction_result.text_artifact_sha256,
    )

    condition = meta.get("condition", "")
    area = classify_therapeutic_area(condition)

    return {
        "quality_scores": {
            "format_verdict": parse_result.format_verdict,
            "format_confidence": parse_result.format_confidence,
            "section_count": parse_result.section_count,
            "review_count": parse_result.review_count,
            "missing_required_sections": (
                parse_result.missing_required_sections
            ),
        },
        "therapeutic_area": area.value,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """Run batch parse on all unscored protocols.

    Args:
        dry_run: Print plan without executing.
        limit: Maximum number of protocols to process.

    Returns:
        Summary dict with counts.
    """
    if not _METADATA_DIR.is_dir():
        print(f"Metadata directory not found: {_METADATA_DIR}")
        return {"scored": 0, "skipped": 0, "failed": 0, "total": 0}

    meta_files = sorted(_METADATA_DIR.glob("*.json"))
    total = len(meta_files)
    scored = 0
    skipped = 0
    failed = 0

    # Initialise gateway once
    gateway: Optional[FilesystemAdapter] = None
    if not dry_run:
        gateway = FilesystemAdapter(root=_DATA_ROOT)
        gateway.initialise()

    for i, meta_path in enumerate(meta_files):
        if limit is not None and scored + failed >= limit:
            break

        try:
            meta = json.loads(
                meta_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  SKIP {meta_path.name}: {exc}")
            failed += 1
            continue

        if not _needs_scoring(meta):
            skipped += 1
            continue

        registry_id = meta.get("registry_id", meta_path.stem)
        pdf_path = _resolve_pdf_path(meta)

        if pdf_path is None:
            print(
                f"  SKIP {registry_id}: PDF not found"
            )
            skipped += 1
            continue

        if dry_run:
            print(
                f"  [{i + 1}/{total}] {registry_id}: "
                f"would parse {pdf_path.name}"
            )
            scored += 1
            continue

        t0 = time.monotonic()
        try:
            assert gateway is not None
            result = parse_single(meta, pdf_path, gateway)

            # Write enriched metadata back
            meta["quality_scores"] = result["quality_scores"]
            meta["therapeutic_area"] = result["therapeutic_area"]
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            elapsed = time.monotonic() - t0
            qs = result["quality_scores"]
            print(
                f"  [{i + 1}/{total}] {registry_id}: "
                f"{qs['format_verdict']} "
                f"({qs['format_confidence']:.2f}) "
                f"- {elapsed:.1f}s"
            )
            scored += 1

        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(
                f"  [{i + 1}/{total}] {registry_id}: "
                f"FAILED ({exc}) - {elapsed:.1f}s"
            )
            failed += 1

    return {
        "scored": scored,
        "skipped": skipped,
        "failed": failed,
        "total": total,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch pre-parse protocols for quality scoring."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of protocols to process.",
    )
    return parser


def main() -> int:
    """CLI entry point.  Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args()

    print("PTCV-42: Batch protocol quality scoring")
    print(f"  Metadata dir : {_METADATA_DIR}")
    print(f"  Dry run      : {args.dry_run}")
    if args.limit is not None:
        print(f"  Limit        : {args.limit}")

    result = run_batch(
        dry_run=args.dry_run,
        limit=args.limit,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total metadata files : {result['total']}")
    print(f"  Scored               : {result['scored']}")
    print(f"  Skipped (scored/no PDF): {result['skipped']}")
    print(f"  Failed               : {result['failed']}")

    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
