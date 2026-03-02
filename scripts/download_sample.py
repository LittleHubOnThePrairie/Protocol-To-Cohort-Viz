"""Download a diverse convenience sample of ~100 clinical trial protocols.

PTCV-27: Uses CTISService and ClinicalTrialsService (PTCV-18) to populate
the PTCV filestore with seed protocols spanning four therapeutic areas.
This is a pragmatic validation sample — not a statistically representative
cross-section of all disease areas.

Therapeutic area split (~25 protocols each):
  - Oncology          : ~12 EU-CTR + ~13 ClinicalTrials.gov
  - Cardiovascular    : ~12 EU-CTR + ~13 ClinicalTrials.gov
  - Neurological      : ~12 EU-CTR + ~13 ClinicalTrials.gov
  - Metabolic/Endocrine: ~12 EU-CTR + ~12 ClinicalTrials.gov

Usage:
    cd C:/Dev/PTCV
    python -m scripts.download_sample [--dry-run] [--target N]

Options:
    --dry-run   Print what would be downloaded without writing to filestore.
    --target N  Override TARGET_PER_BUCKET (default 12).

Regulatory requirements:
- Audit trail: every download logged via AuditLogger (21 CFR 11.10(e))
- SHA-256: computed at download boundary (ALCOA+ Consistent)
- No overwrite: FilestoreManager raises FileExistsError on duplicate (ALCOA+ Original)
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Allow running as `python -m scripts.download_sample` from project root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ptcv.protocol_search.clinicaltrials_service import ClinicalTrialsService
from ptcv.protocol_search.eu_ctr_service import CTISService


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THERAPEUTIC_AREAS: list[tuple[str, str, str]] = [
    # (condition_name, ctis_phase, ct_phase_enum)
    ("oncology", "Phase 2", "PHASE2"),
    ("cardiovascular", "Phase 2", "PHASE2"),
    ("neurology", "Phase 2", "PHASE2"),
    ("diabetes", "Phase 2", "PHASE2"),
]

DEFAULT_TARGET_PER_BUCKET = 12  # EU-CTR per condition; CT.gov gets +1 to reach ~100 total
_AUDIT_REPORT_PATH = Path("data/audit/sample_download_report.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BucketResult:
    """Outcome for a single (condition, registry) search-and-download bucket."""

    condition: str
    registry: str
    search_found: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class SampleReport:
    """Aggregate report written to data/audit/sample_download_report.json."""

    run_timestamp: str
    downloaded: int
    skipped_already_stored: int
    failed: int
    total_attempted: int
    therapeutic_areas: list[str]
    bucket_results: list[dict]
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "run_timestamp": self.run_timestamp,
            "downloaded": self.downloaded,
            "skipped_already_stored": self.skipped_already_stored,
            "failed": self.failed,
            "total_attempted": self.total_attempted,
            "therapeutic_areas": self.therapeutic_areas,
            "bucket_results": self.bucket_results,
            "dry_run": self.dry_run,
        }


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def _download_eu_ctr_bucket(
    ctis: CTISService,
    condition: str,
    ctis_phase: str,
    target: int,
    dry_run: bool,
) -> BucketResult:
    """Search EU-CTR for `condition`, randomly sample `target`, download each."""
    result = BucketResult(condition=condition, registry="EU-CTR")

    search_results = ctis.search(
        condition=condition,
        phase=ctis_phase,
        page_size=min(target * 4, 100),  # fetch more than needed to enable sampling
    )
    result.search_found = len(search_results)

    if not search_results:
        return result

    sample = random.sample(search_results, min(target, len(search_results)))

    for trial in sample:
        if dry_run:
            print(f"  [dry-run] EU-CTR {trial.registry_id}: {trial.title[:60]}")
            result.downloaded += 1
            continue

        dl = ctis.download(
            euct_code=trial.registry_id,
            fmt="PDF",
            why=f"sample_download_100:{condition}",
        )
        if dl.success:
            result.downloaded += 1
            print(f"  ✓ EU-CTR {trial.registry_id}")
        elif "already stored" in (dl.error or "").lower():
            result.skipped += 1
            print(f"  ~ EU-CTR {trial.registry_id} (already stored)")
        else:
            result.failed += 1
            result.errors.append(f"{trial.registry_id}: {dl.error}")
            print(f"  ✗ EU-CTR {trial.registry_id}: {dl.error}")

    return result


def _download_ct_gov_bucket(
    ct: ClinicalTrialsService,
    condition: str,
    ct_phase: str,
    target: int,
    dry_run: bool,
) -> BucketResult:
    """Search ClinicalTrials.gov for `condition`, randomly sample `target`, download each."""
    result = BucketResult(condition=condition, registry="ClinicalTrials.gov")

    search_results = ct.search(
        condition=condition,
        phase=ct_phase,
        max_results=min(target * 4, 100),
    )
    result.search_found = len(search_results)

    if not search_results:
        return result

    sample = random.sample(search_results, min(target, len(search_results)))

    for trial in sample:
        if dry_run:
            print(f"  [dry-run] CT.gov {trial.registry_id}: {trial.title[:60]}")
            result.downloaded += 1
            continue

        dl = ct.download(
            nct_id=trial.registry_id,
            fmt="PDF",
            why=f"sample_download_100:{condition}",
        )
        if dl.success:
            result.downloaded += 1
            print(f"  ✓ CT.gov {trial.registry_id}")
        elif "already stored" in (dl.error or "").lower():
            result.skipped += 1
            print(f"  ~ CT.gov {trial.registry_id} (already stored)")
        else:
            result.failed += 1
            result.errors.append(f"{trial.registry_id}: {dl.error}")
            print(f"  ✗ CT.gov {trial.registry_id}: {dl.error}")

    return result


def run_sample_download(
    target_per_bucket: int = DEFAULT_TARGET_PER_BUCKET,
    dry_run: bool = False,
    seed: Optional[int] = None,
) -> SampleReport:
    """Run the full diverse convenience sample download.

    Args:
        target_per_bucket: Maximum protocols to download per (condition, registry)
            bucket. Default 12 EU-CTR + 13 CT.gov per condition ≈ 100 total.
        dry_run: If True, print planned downloads without writing to filestore.
        seed: Optional random seed for reproducibility in testing.

    Returns:
        SampleReport with aggregate counts and per-bucket breakdown.
    """
    if seed is not None:
        random.seed(seed)

    ctis = CTISService()
    ct = ClinicalTrialsService()

    all_buckets: list[BucketResult] = []
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for condition, ctis_phase, ct_phase in THERAPEUTIC_AREAS:
        print(f"\n[{condition.upper()}]")

        # EU-CTR bucket (target_per_bucket)
        eu_bucket = _download_eu_ctr_bucket(
            ctis, condition, ctis_phase, target_per_bucket, dry_run
        )
        all_buckets.append(eu_bucket)

        # CT.gov bucket (target_per_bucket + 1 to hit ~100 overall)
        ct_bucket = _download_ct_gov_bucket(
            ct, condition, ct_phase, target_per_bucket + 1, dry_run
        )
        all_buckets.append(ct_bucket)

        total_downloaded += eu_bucket.downloaded + ct_bucket.downloaded
        total_skipped += eu_bucket.skipped + ct_bucket.skipped
        total_failed += eu_bucket.failed + ct_bucket.failed

    report = SampleReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        downloaded=total_downloaded,
        skipped_already_stored=total_skipped,
        failed=total_failed,
        total_attempted=total_downloaded + total_skipped + total_failed,
        therapeutic_areas=[c for c, _, _ in THERAPEUTIC_AREAS],
        bucket_results=[
            {
                "condition": b.condition,
                "registry": b.registry,
                "search_found": b.search_found,
                "downloaded": b.downloaded,
                "skipped": b.skipped,
                "failed": b.failed,
            }
            for b in all_buckets
        ],
        dry_run=dry_run,
    )

    return report


def write_report(report: SampleReport) -> None:
    """Write the sample report JSON to data/audit/ and print summary."""
    _AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _AUDIT_REPORT_PATH.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    print(f"\nReport written to {_AUDIT_REPORT_PATH}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a diverse convenience sample of ~100 clinical trial protocols."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing to filestore.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_PER_BUCKET,
        metavar="N",
        help=f"Protocols per (condition, registry) bucket (default {DEFAULT_TARGET_PER_BUCKET}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducible sampling.",
    )
    return parser


def main() -> int:
    """CLI entry point. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args()

    print("PTCV-27: Diverse convenience sample download")
    print(f"  Target per bucket : {args.target} EU-CTR + {args.target + 1} CT.gov")
    print(f"  Therapeutic areas : {[c for c, _, _ in THERAPEUTIC_AREAS]}")
    print(f"  Dry run           : {args.dry_run}")
    if args.seed is not None:
        print(f"  Random seed       : {args.seed}")

    report = run_sample_download(
        target_per_bucket=args.target,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(report.to_dict(), indent=2))

    if not args.dry_run:
        write_report(report)

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
