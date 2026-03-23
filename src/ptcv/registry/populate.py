"""Registry population module (PTCV-201).

Batch-fetches ClinicalTrials.gov metadata for all locally downloaded
protocol PDFs, maps to ICH E6(R3) sections, and optionally seeds the
FAISS RAG index.

Invokable as::

    python -m ptcv.registry.populate [--dry-run] [--seed-rag] [--verbose]

Risk tier: LOW — read-only registry queries and local caching.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_NCT_PATTERN = re.compile(r"(NCT\d{8})")
_DEFAULT_DATA_DIR = Path("data/protocols/clinicaltrials")


@dataclasses.dataclass
class PopulateResult:
    """Summary of a registry population run.

    Attributes:
        total_nct_ids: Number of unique NCT IDs discovered.
        fetched: Number of fresh API fetches made.
        cached: Number loaded from local cache.
        failed: Number of fetch failures.
        total_sections_mapped: Total ICH sections produced.
        sections_seeded: Total vectors added to RAG index (0 if
            RAG seeding was not requested).
        enriched: Number of trials enriched with literature/newswire.
        total_enrichment_articles: Total articles found across all
            enriched trials.
    """

    total_nct_ids: int = 0
    fetched: int = 0
    cached: int = 0
    failed: int = 0
    total_sections_mapped: int = 0
    sections_seeded: int = 0
    enriched: int = 0
    total_enrichment_articles: int = 0


def discover_nct_ids(base_dir: str | Path) -> list[str]:
    """Extract unique NCT IDs from PDF filenames recursively.

    Scans ``base_dir`` and all subdirectories for PDF files whose
    names contain an NCT ID (``NCT`` followed by 8 digits).

    Args:
        base_dir: Root directory containing protocol PDFs.

    Returns:
        Sorted list of unique NCT IDs.
    """
    nct_ids: set[str] = set()
    for pdf_path in glob.glob(
        os.path.join(str(base_dir), "**", "*.pdf"), recursive=True
    ):
        match = _NCT_PATTERN.search(os.path.basename(pdf_path))
        if match:
            nct_ids.add(match.group(1))
    return sorted(nct_ids)


def populate(
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    cache_dir: Optional[str | Path] = None,
    seed_rag: bool = False,
    delay: float = 0.5,
    dry_run: bool = False,
    enrich: bool = False,
    sources: Optional[str] = None,
    newsdata_api_key: Optional[str] = None,
) -> PopulateResult:
    """Batch-fetch CT.gov metadata and map to ICH sections.

    Args:
        data_dir: Directory containing protocol PDFs.
        cache_dir: Cache directory for JSON responses.
            Defaults to ``{data_dir}/registry_cache``.
        seed_rag: If True, seed the FAISS RAG index after mapping.
        delay: Seconds between fresh API requests.
        dry_run: If True, only discover and list NCT IDs.
        enrich: If True, enrich metadata with literature and
            newswire data from GDELT, NewsData.io, and PubMed
            (PTCV-206).
        sources: Comma-separated source list for enrichment
            (e.g. ``"gdelt,newsdata"``). Defaults to all.
        newsdata_api_key: NewsData.io API key. Required if
            newsdata is in sources and enrich is True.

    Returns:
        PopulateResult with counts and statistics.
    """
    # Import directly to avoid __init__.py pulling in fitz
    from ptcv.registry.metadata_fetcher import RegistryMetadataFetcher
    from ptcv.registry.ich_mapper import MetadataToIchMapper

    data_dir = Path(data_dir)
    if cache_dir is None:
        cache_dir = data_dir / "registry_cache"
    else:
        cache_dir = Path(cache_dir)

    nct_ids = discover_nct_ids(data_dir)
    result = PopulateResult(total_nct_ids=len(nct_ids))

    print(f"Discovered {len(nct_ids)} unique NCT IDs in {data_dir}")

    if dry_run:
        for nct_id in nct_ids:
            print(f"  {nct_id}")
        return result

    fetcher = RegistryMetadataFetcher(cache_dir=cache_dir)
    mapper = MetadataToIchMapper()

    # Optional RAG seeding setup
    seeder = None
    rag_index = None
    if seed_rag:
        from ptcv.registry.rag_seeder import RegistryRagSeeder
        seeder = RegistryRagSeeder()
        # Lazy import to avoid sentence-transformers when not needed
        from ptcv.ich_parser.rag_index import RagIndex
        rag_index = RagIndex()

    all_mapped: dict[str, list[Any]] = {}

    for i, nct_id in enumerate(nct_ids, 1):
        cache_path = cache_dir / f"{nct_id}.json"
        was_cached = cache_path.exists()

        metadata = fetcher.fetch(nct_id)

        if metadata is None:
            print(f"  [{i}/{len(nct_ids)}] {nct_id} - FAILED")
            result.failed += 1
            continue

        if was_cached:
            result.cached += 1
            label = "cached"
        else:
            result.fetched += 1
            label = "fetched"

        sections = mapper.map(metadata)
        result.total_sections_mapped += len(sections)
        codes = ", ".join(s.section_code for s in sections)

        print(
            f"  [{i}/{len(nct_ids)}] {nct_id} - "
            f"{label} -> {len(sections)} sections ({codes})"
        )

        all_mapped[nct_id] = sections

        # Rate-limit fresh API calls
        if not was_cached:
            time.sleep(delay)

    # Optional literature/newswire enrichment (PTCV-206)
    if enrich and all_mapped:
        from ptcv.registry.literature_enricher import (
            LiteratureEnricher,
            parse_sources,
        )
        from ptcv.registry.gdelt_adapter import GdeltAdapter
        from ptcv.registry.newsdata_adapter import NewsdataAdapter
        from ptcv.registry.pubmed_adapter import PubmedAdapter

        source_set = parse_sources(sources or "all")
        newswire_cache = cache_dir / "newswire"
        pubmed_cache = cache_dir / "pubmed"

        gdelt = GdeltAdapter(cache_dir=newswire_cache)
        newsdata = None
        if "newsdata" in source_set and newsdata_api_key:
            newsdata = NewsdataAdapter(
                api_key=newsdata_api_key,
                cache_dir=newswire_cache,
            )
        elif "newsdata" in source_set:
            logger.warning(
                "NewsData.io API key not provided, skipping "
                "newsdata source"
            )
            source_set.discard("newsdata")

        pubmed = PubmedAdapter(cache_dir=pubmed_cache)

        enricher = LiteratureEnricher(
            gdelt_adapter=gdelt,
            newsdata_adapter=newsdata,
            pubmed_adapter=pubmed,
            cache_dir=cache_dir,
        )

        print(
            f"\nEnriching {len(all_mapped)} trials from: "
            f"{', '.join(sorted(source_set))}..."
        )
        for i, nct_id in enumerate(all_mapped, 1):
            enrichment = enricher.enrich_trial(
                nct_id, sources=source_set
            )
            if enrichment.total_articles > 0:
                result.enriched += 1
                result.total_enrichment_articles += (
                    enrichment.total_articles
                )
                print(
                    f"  [{i}/{len(all_mapped)}] "
                    f"{enrichment.summary()}"
                )
            else:
                print(
                    f"  [{i}/{len(all_mapped)}] "
                    f"{nct_id}: no articles"
                )

    # Optional RAG seeding pass
    if seeder and rag_index and all_mapped:
        print(f"\nSeeding RAG index with {len(all_mapped)} trials...")
        for nct_id, sections in all_mapped.items():
            if not sections:
                continue
            seed_result = seeder.seed(rag_index, sections, nct_id)
            result.sections_seeded += seed_result.sections_seeded

    # Summary
    print(f"\n{'='*60}")
    print("Registry Population Results:")
    print(f"  Total NCT IDs:          {result.total_nct_ids}")
    print(f"  Freshly fetched:        {result.fetched}")
    print(f"  From cache:             {result.cached}")
    print(f"  Failed:                 {result.failed}")
    print(f"  Total ICH sections:     {result.total_sections_mapped}")
    if seed_rag:
        print(f"  RAG vectors seeded:     {result.sections_seeded}")
    if enrich:
        print(f"  Trials enriched:        {result.enriched}")
        print(f"  Enrichment articles:    {result.total_enrichment_articles}")
    print(f"  Cache dir:              {cache_dir}")

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m ptcv.registry.populate",
        description=(
            "Batch-fetch ClinicalTrials.gov metadata for locally "
            "downloaded protocol PDFs and map to ICH E6(R3) sections."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_DEFAULT_DATA_DIR),
        help=(
            "Base directory containing protocol PDFs "
            "(default: data/protocols/clinicaltrials)"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for CT.gov JSON (default: {data-dir}/registry_cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered NCT IDs without fetching",
    )
    parser.add_argument(
        "--seed-rag",
        action="store_true",
        help="Seed FAISS RAG index after mapping",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help=(
            "Enrich metadata with literature and newswire data "
            "from GDELT, NewsData.io, and PubMed (PTCV-206)"
        ),
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated enrichment sources: "
            "pubmed,gdelt,newsdata or 'all' (default: all)"
        ),
    )
    parser.add_argument(
        "--newsdata-api-key",
        type=str,
        default=None,
        help="NewsData.io API key (required for newsdata source)",
    )
    return parser


def main(argv: list[str] | None = None) -> PopulateResult:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    return populate(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        seed_rag=args.seed_rag,
        delay=args.delay,
        dry_run=args.dry_run,
        enrich=args.enrich,
        sources=args.sources,
        newsdata_api_key=args.newsdata_api_key,
    )


if __name__ == "__main__":
    main()
