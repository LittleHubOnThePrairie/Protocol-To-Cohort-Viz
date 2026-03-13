"""Benchmark PDF-to-Markdown conversion across multiple tools.

Converts 5 representative protocols through all tools and compares
output quality (table preservation, heading hierarchy, list items, images).

Usage:
    cd C:/Dev/PTCV && python scripts/benchmark_pdf_to_markdown.py
    python scripts/benchmark_pdf_to_markdown.py --protocols NCT05855967 NCT01866111
    python scripts/benchmark_pdf_to_markdown.py --output-dir data/analysis/benchmark_md
    python scripts/benchmark_pdf_to_markdown.py --tools pymupdf4llm camelot_hybrid
    python scripts/benchmark_pdf_to_markdown.py --tools pymupdf4llm --append

Regulatory: N/A (benchmarking script, no pipeline changes)
PTCV-168, PTCV-174, PTCV-175
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import traceback
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_DIR = Path("C:/Dev/PTCV/data/protocols/clinicaltrials")

# 5 representative protocols chosen for diversity of layout challenges
BENCHMARK_PROTOCOLS: list[str] = [
    "NCT05855967",  # Synopsis + Protocol Summary headers, complex efficacy tables
    "NCT05478525",  # CID garble artifacts — tests OCR fallback
    "NCT01866111",  # Dense Schedule of Assessments, multi-column lab tests
    "NCT02250651",  # Study design flowchart, nested eligibility criteria
    "NCT03826472",  # No B.1 section match — tests cover page extraction
]

DEFAULT_OUTPUT_DIR = Path("data/analysis/benchmark_md")

TOOLS = [
    "pdfplumber", "docling", "marker", "nougat",
    "pymupdf4llm", "camelot_hybrid", "unstructured",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    nct_id: str
    tool_name: str
    elapsed_seconds: float = 0.0
    markdown_length: int = 0
    table_count: int = 0
    heading_count: int = 0
    list_item_count: int = 0
    image_count: int = 0
    token_estimate: int = 0
    peak_memory_mb: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Markdown element counting
# ---------------------------------------------------------------------------

# Markdown table separator row: | --- | --- | or |:---|:---|
_TABLE_SEP = re.compile(r"^\|[\s:]*-{3,}[\s:]*\|", re.MULTILINE)

# ATX headings: # through ######
_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)

# Unordered list items: - or * at start of line (with optional indent)
_UL_ITEM = re.compile(r"^\s*[-*]\s", re.MULTILINE)

# Ordered list items: 1. 2. etc.
_OL_ITEM = re.compile(r"^\s*\d+\.\s", re.MULTILINE)

# Markdown images: ![alt](url)
_IMAGE = re.compile(r"!\[")


def count_markdown_elements(text: str) -> dict[str, int]:
    """Count structural Markdown elements in text."""
    return {
        "table_count": len(_TABLE_SEP.findall(text)),
        "heading_count": len(_HEADING.findall(text)),
        "list_item_count": (
            len(_UL_ITEM.findall(text)) + len(_OL_ITEM.findall(text))
        ),
        "image_count": len(_IMAGE.findall(text)),
    }


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def extract_with_pdfplumber(pdf_path: Path) -> tuple[BenchmarkResult, str]:
    """Baseline: use existing toc_extractor pipeline."""
    from ptcv.ich_parser.toc_extractor import extract_protocol_index

    nct_id = pdf_path.stem.split("_")[0]
    t0 = time.perf_counter()
    try:
        index = extract_protocol_index(str(pdf_path))
        elapsed = time.perf_counter() - t0
        text = index.full_text
        counts = count_markdown_elements(text)
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="pdfplumber",
            elapsed_seconds=round(elapsed, 2),
            markdown_length=len(text),
            token_estimate=len(text) // 4,
            **counts,
        ), text
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="pdfplumber",
            elapsed_seconds=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        ), ""


def extract_with_docling(pdf_path: Path) -> tuple[BenchmarkResult, str]:
    """Docling (IBM): DocumentConverter -> export_to_markdown()."""
    from docling.document_converter import DocumentConverter

    nct_id = pdf_path.stem.split("_")[0]
    t0 = time.perf_counter()
    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        md = result.document.export_to_markdown()
        elapsed = time.perf_counter() - t0
        counts = count_markdown_elements(md)
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="docling",
            elapsed_seconds=round(elapsed, 2),
            markdown_length=len(md),
            token_estimate=len(md) // 4,
            **counts,
        ), md
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="docling",
            elapsed_seconds=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        ), ""


def extract_with_marker(pdf_path: Path) -> tuple[BenchmarkResult, str]:
    """Marker (DataLab): PdfConverter -> text_from_rendered()."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    nct_id = pdf_path.stem.split("_")[0]
    t0 = time.perf_counter()
    try:
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        text, _metadata, _images = text_from_rendered(rendered)
        elapsed = time.perf_counter() - t0
        counts = count_markdown_elements(text)
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="marker",
            elapsed_seconds=round(elapsed, 2),
            markdown_length=len(text),
            token_estimate=len(text) // 4,
            **counts,
        ), text
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="marker",
            elapsed_seconds=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        ), ""


_nougat_model = None
_nougat_processor = None
_nougat_device = None


def _get_nougat_model():
    """Lazy-load and cache Nougat model/processor across protocols."""
    global _nougat_model, _nougat_processor, _nougat_device
    if _nougat_model is None:
        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel

        model_name = "facebook/nougat-base"
        # use_fast=False avoids StrictDataclassFieldValidationError
        # with do_crop_margin=None in transformers >=5.x fast processor
        _nougat_processor = NougatProcessor.from_pretrained(
            model_name, use_fast=False
        )
        _nougat_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        _nougat_device = "cuda" if torch.cuda.is_available() else "cpu"
        _nougat_model.to(_nougat_device)
        _nougat_model.eval()
    return _nougat_model, _nougat_processor, _nougat_device


def extract_with_nougat(pdf_path: Path) -> tuple[BenchmarkResult, str]:
    """Nougat (Meta): vision transformer PDF->Markdown via HuggingFace.

    Uses facebook/nougat-base model. Converts each PDF page to an image
    via pypdfium2, then runs through the Nougat encoder-decoder to produce
    Markdown output. PTCV-174.
    """
    import torch
    import pypdfium2 as pdfium

    nct_id = pdf_path.stem.split("_")[0]
    t0 = time.perf_counter()
    doc = None
    try:
        model, processor, device = _get_nougat_model()

        doc = pdfium.PdfDocument(str(pdf_path))
        total_pages = len(doc)
        page_texts: list[str] = []

        for page_idx in range(total_pages):
            page = doc[page_idx]
            bitmap = page.render(scale=96 / 72)
            pil_image = bitmap.to_pil()

            pixel_values = processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values.to(device)

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    min_length=1,
                    max_new_tokens=4096,
                    bad_words_ids=[
                        [processor.tokenizer.unk_token_id],
                    ],
                )

            page_md = processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            page_md = processor.post_process_generation(
                page_md, fix_markdown=False
            )
            page_texts.append(page_md)

            if (page_idx + 1) % 10 == 0 or page_idx == total_pages - 1:
                elapsed_so_far = time.perf_counter() - t0
                print(
                    f"\r    page {page_idx + 1}/{total_pages} "
                    f"({elapsed_so_far:.0f}s)",
                    end="",
                    flush=True,
                )

        print()  # newline after progress
        doc.close()
        doc = None
        md = "\n\n".join(page_texts)
        elapsed = time.perf_counter() - t0
        counts = count_markdown_elements(md)
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="nougat",
            elapsed_seconds=round(elapsed, 2),
            markdown_length=len(md),
            token_estimate=len(md) // 4,
            **counts,
        ), md
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="nougat",
            elapsed_seconds=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        ), ""
    finally:
        if doc is not None:
            doc.close()


def extract_with_pymupdf4llm(pdf_path: Path) -> tuple[BenchmarkResult, str]:
    """PyMuPDF4LLM: rule-based PDF -> Markdown (PTCV-175)."""
    import pymupdf4llm

    nct_id = pdf_path.stem.split("_")[0]
    t0 = time.perf_counter()
    try:
        md = pymupdf4llm.to_markdown(str(pdf_path))
        elapsed = time.perf_counter() - t0
        counts = count_markdown_elements(md)
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="pymupdf4llm",
            elapsed_seconds=round(elapsed, 2),
            markdown_length=len(md),
            token_estimate=len(md) // 4,
            **counts,
        ), md
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="pymupdf4llm",
            elapsed_seconds=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        ), ""


def extract_with_camelot_hybrid(pdf_path: Path) -> tuple[BenchmarkResult, str]:
    """Hybrid: pdfplumber text + Camelot table detection (PTCV-175).

    Extracts text per page via pdfplumber, then detects tables via Camelot
    (lattice mode for bordered tables) and appends them as Markdown tables.
    """
    import camelot
    import pdfplumber

    nct_id = pdf_path.stem.split("_")[0]
    t0 = time.perf_counter()
    try:
        pages_md: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages_md.append(page_text)

        # Camelot table detection (lattice mode for bordered tables)
        try:
            tables = camelot.read_pdf(
                str(pdf_path), pages="all", flavor="lattice",
            )
            for table in tables:
                md_table = table.df.to_markdown(index=False)
                pages_md.append(f"\n{md_table}\n")
        except Exception:
            pass  # Some PDFs have no extractable lattice tables

        md = "\n\n".join(pages_md)
        elapsed = time.perf_counter() - t0
        counts = count_markdown_elements(md)
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="camelot_hybrid",
            elapsed_seconds=round(elapsed, 2),
            markdown_length=len(md),
            token_estimate=len(md) // 4,
            **counts,
        ), md
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="camelot_hybrid",
            elapsed_seconds=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        ), ""


def _html_table_to_markdown(html: str) -> str:
    """Convert HTML <table> to Markdown table via pandas."""
    import pandas as pd

    try:
        dfs = pd.read_html(html)
        if dfs:
            return dfs[0].to_markdown(index=False)
    except Exception:
        pass
    return html  # fallback: return raw HTML


def extract_with_unstructured(pdf_path: Path) -> tuple[BenchmarkResult, str]:
    """Unstructured.io: Detectron2 layout + hi_res extraction (PTCV-175).

    Uses partition_pdf with hi_res strategy and infer_table_structure=True.
    Tables arrive as HTML and are converted to Markdown via pandas.
    """
    nct_id = pdf_path.stem.split("_")[0]
    t0 = time.perf_counter()
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(
            str(pdf_path),
            strategy="hi_res",
            infer_table_structure=True,
        )
        # Assemble Markdown from typed elements
        parts: list[str] = []
        for el in elements:
            if el.category == "Title":
                parts.append(f"## {el.text}")
            elif el.category == "Table":
                html = getattr(el.metadata, "text_as_html", None)
                if html:
                    parts.append(_html_table_to_markdown(html))
                else:
                    parts.append(el.text)
            elif el.category == "ListItem":
                parts.append(f"- {el.text}")
            else:
                parts.append(el.text)

        md = "\n\n".join(parts)
        elapsed = time.perf_counter() - t0
        counts = count_markdown_elements(md)
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="unstructured",
            elapsed_seconds=round(elapsed, 2),
            markdown_length=len(md),
            token_estimate=len(md) // 4,
            **counts,
        ), md
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            nct_id=nct_id,
            tool_name="unstructured",
            elapsed_seconds=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        ), ""


EXTRACTORS = {
    "pdfplumber": extract_with_pdfplumber,
    "docling": extract_with_docling,
    "marker": extract_with_marker,
    "nougat": extract_with_nougat,
    "pymupdf4llm": extract_with_pymupdf4llm,
    "camelot_hybrid": extract_with_camelot_hybrid,
    "unstructured": extract_with_unstructured,
}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_output(
    nct_id: str,
    tool_name: str,
    content: str,
    output_dir: Path,
) -> Path:
    """Write raw Markdown to output_dir/{nct_id}/{tool_name}.md."""
    dest = output_dir / nct_id / f"{tool_name}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return dest


def save_results_csv(
    results: list[BenchmarkResult],
    output_dir: Path,
    append: bool = False,
) -> Path:
    """Write benchmark results to CSV.

    When append=True and the CSV already exists, reads existing rows,
    removes any rows matching the same (nct_id, tool_name) pairs from
    the new results (to allow re-runs), then writes all rows together.
    """
    csv_path = output_dir / "benchmark_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "nct_id", "tool_name", "elapsed_seconds", "markdown_length",
        "table_count", "heading_count", "list_item_count", "image_count",
        "token_estimate", "peak_memory_mb", "error",
    ]

    existing_rows: list[dict[str, str]] = []
    if append and csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        # Remove rows that will be replaced by new results
        new_keys = {(r.nct_id, r.tool_name) for r in results}
        existing_rows = [
            row for row in existing_rows
            if (row["nct_id"], row["tool_name"]) not in new_keys
        ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for r in results:
            writer.writerow(asdict(r))
    return csv_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_comparison_report(results: list[BenchmarkResult]) -> None:
    """Print per-protocol comparison matrix and aggregate summary."""
    # Group by nct_id
    by_protocol: dict[str, dict[str, BenchmarkResult]] = {}
    for r in results:
        by_protocol.setdefault(r.nct_id, {})[r.tool_name] = r

    # Discover tools present in results (preserve TOOLS order, add extras)
    seen_tools: list[str] = []
    for t in TOOLS:
        if any(r.tool_name == t for r in results):
            seen_tools.append(t)
    for r in results:
        if r.tool_name not in seen_tools:
            seen_tools.append(r.tool_name)

    print()
    print("=" * 72)
    print("PDF-TO-MARKDOWN BENCHMARK RESULTS")
    print("=" * 72)

    for nct_id in BENCHMARK_PROTOCOLS:
        if nct_id not in by_protocol:
            continue
        tools = by_protocol[nct_id]
        print(f"\nProtocol: {nct_id}")
        print("-" * 72)
        header = (
            f"  {'Tool':<16} {'Time(s)':>8} {'Length':>10} "
            f"{'Tables':>8} {'Headings':>9} {'Lists':>7} "
            f"{'Images':>8} {'Mem(MB)':>8}"
        )
        print(header)
        for tool_name in seen_tools:
            r = tools.get(tool_name)
            if r is None:
                continue
            if r.error:
                print(f"  {tool_name:<16} ERROR: {r.error[:50]}")
            else:
                print(
                    f"  {r.tool_name:<16} {r.elapsed_seconds:>8.1f} "
                    f"{r.markdown_length:>10,} {r.table_count:>8} "
                    f"{r.heading_count:>9} {r.list_item_count:>7} "
                    f"{r.image_count:>8} {r.peak_memory_mb:>8.1f}"
                )

    # Aggregate summary
    print()
    print("=" * 72)
    print(f"AGGREGATE SUMMARY ({len(by_protocol)} protocols)")
    print("=" * 72)

    for tool_name in seen_tools:
        tool_results = [r for r in results if r.tool_name == tool_name and not r.error]
        if not tool_results:
            print(f"  {tool_name:<16} No successful results")
            continue
        n = len(tool_results)
        avg_time = sum(r.elapsed_seconds for r in tool_results) / n
        avg_tables = sum(r.table_count for r in tool_results) / n
        avg_headings = sum(r.heading_count for r in tool_results) / n
        avg_lists = sum(r.list_item_count for r in tool_results) / n
        total_images = sum(r.image_count for r in tool_results)
        avg_length = sum(r.markdown_length for r in tool_results) / n
        max_mem = max(r.peak_memory_mb for r in tool_results)
        errors = sum(1 for r in results if r.tool_name == tool_name and r.error)
        print(
            f"  {tool_name:<16} "
            f"avg_time={avg_time:.1f}s  "
            f"avg_len={avg_length:,.0f}  "
            f"avg_tables={avg_tables:.1f}  "
            f"avg_headings={avg_headings:.1f}  "
            f"avg_lists={avg_lists:.1f}  "
            f"total_images={total_images}  "
            f"max_mem={max_mem:.0f}MB  "
            f"errors={errors}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_pdf(nct_id: str) -> Optional[Path]:
    """Find PDF for an NCT ID (handles _1.0.pdf naming)."""
    matches = list(PDF_DIR.glob(f"{nct_id}_*.pdf"))
    if matches:
        return matches[0]
    # Exact match fallback
    exact = PDF_DIR / f"{nct_id}.pdf"
    if exact.exists():
        return exact
    return None


_GPU_TOOLS = {"nougat", "marker"}


def run_extraction(
    pdf_path: Path,
    tool_name: str,
) -> tuple[BenchmarkResult, str]:
    """Run a single extraction with memory profiling.

    Skips tracemalloc for GPU-based tools (nougat, marker) because
    intercepting every tensor allocation causes 100x+ slowdown.
    """
    extractor = EXTRACTORS[tool_name]
    if tool_name in _GPU_TOOLS:
        result, text = extractor(pdf_path)
    else:
        tracemalloc.start()
        try:
            result, text = extractor(pdf_path)
            _, peak = tracemalloc.get_traced_memory()
            result.peak_memory_mb = round(peak / 1024 / 1024, 1)
        finally:
            tracemalloc.stop()
    return result, text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PDF-to-Markdown conversion across multiple tools",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=BENCHMARK_PROTOCOLS,
        help="NCT IDs to benchmark (default: all 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output Markdown files and CSV",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=TOOLS,
        default=TOOLS,
        help="Tools to benchmark (default: all)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to existing CSV instead of overwriting",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[BenchmarkResult] = []

    for nct_id in args.protocols:
        pdf_path = find_pdf(nct_id)
        if pdf_path is None:
            print(f"WARNING: PDF not found for {nct_id}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'=' * 60}")

        for tool_name in args.tools:
            print(f"  [{tool_name}] extracting...", end=" ", flush=True)
            result, raw_text = run_extraction(pdf_path, tool_name)
            all_results.append(result)

            if result.error:
                print(f"ERROR: {result.error[:60]}")
            else:
                print(
                    f"OK ({result.elapsed_seconds:.1f}s, "
                    f"{result.markdown_length:,} chars, "
                    f"{result.table_count} tables)"
                )
                # Save raw output for manual review
                if raw_text:
                    saved = save_output(nct_id, tool_name, raw_text, output_dir)
                    print(f"         saved -> {saved}")

    # Write CSV
    csv_path = save_results_csv(all_results, output_dir, append=args.append)
    print(f"\nResults CSV -> {csv_path}")

    # Print comparison report
    print_comparison_report(all_results)


if __name__ == "__main__":
    main()
