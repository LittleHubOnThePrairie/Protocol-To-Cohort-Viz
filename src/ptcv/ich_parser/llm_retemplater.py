"""LLM-based ICH section retemplating (PTCV-60, PTCV-64).

Two-pass architecture for reorganizing clinical trial protocols into
ICH E6(R3) Appendix B structure:

Pass 1 (LLM): Classify each page to an ICH section code via Claude
    Opus 4.6. Returns page-level assignments (cheap output tokens).
Pass 2 (Deterministic): Assemble full original text per section by
    grouping text blocks by their assigned section code.

For ICH-native protocols, this is a near-identity transform with
high confidence scores. For non-ICH protocols, the LLM maps
arbitrary document structure into the standard ICH template.

Risk tier: MEDIUM -- data pipeline ML component (no patient data).

Regulatory references:
- ALCOA+ Accurate: confidence_score from LLM self-assessment
- ALCOA+ Traceable: source_sha256 links to extraction artifact
- ALCOA+ Contemporaneous: extraction_timestamp_utc at write boundary
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import uuid
from collections import defaultdict

from ptcv.ich_parser.query_schema import section_sort_key
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..storage import FilesystemAdapter, StorageGateway
from .models import IchSection, ReviewQueueEntry
from .parquet_writer import sections_to_parquet
from .review_queue import ReviewQueue
from .corpus_loader import CorpusExemplar, get_exemplars, load_corpus
from .schema_loader import (
    get_review_threshold,
    get_section_defs,
    get_section_order,
    get_stage_prompt,
)
from .toc_extractor import _is_toc_page, _count_toc_lines

logger = logging.getLogger(__name__)

_USER = "ptcv-llm-retemplater"
_DEFAULT_REVIEW_DB = Path("C:/Dev/PTCV/data/sqlite/review_queue.db")

# Maximum text characters per LLM chunk (~30K tokens with prompt)
_MAX_CHUNK_CHARS = 100_000

# Few-shot exemplar budget (PTCV-47)
_MAX_EXEMPLAR_CHARS = 2_000  # Per exemplar
_MAX_EXEMPLARS = 2  # Per prompt

# ICH E6(R3) Appendix B section definitions — loaded from YAML (PTCV-67)
_ICH_SECTION_DEFS: dict[str, str] = get_section_defs()

# Canonical section order for markdown rendering — loaded from YAML (PTCV-67)
_ICH_SECTION_ORDER: list[tuple[str, str]] = get_section_order()


@dataclasses.dataclass
class _PageAssignment:
    """Page-level section assignment from LLM classification (Pass 1)."""

    section_code: str
    confidence: float
    key_concepts: list[str]
    pages: list[int]


@dataclasses.dataclass
class RetemplatingResult:
    """Result returned by LlmRetemplater.retemplate().

    Attributes:
        run_id: UUID4 for this retemplating run.
        registry_id: Trial identifier.
        artifact_key: Storage key for sections.parquet.
        artifact_sha256: SHA-256 of the stored Parquet bytes.
        section_count: Number of sections produced.
        review_count: Low-confidence sections routed to review queue.
        source_sha256: SHA-256 of upstream extraction artifact.
        format_verdict: "ICH_E6R3", "PARTIAL_ICH", or "NON_ICH".
        format_confidence: Float 0.0-1.0.
        missing_required_sections: Required section codes not found.
        input_tokens: Total input tokens across all LLM calls.
        output_tokens: Total output tokens across all LLM calls.
        chunk_count: Number of page-range chunks processed.
        retemplated_artifact_key: Storage key for retemplated_protocol.md.
        retemplated_artifact_sha256: SHA-256 of the markdown artifact.
    """

    run_id: str
    registry_id: str
    artifact_key: str
    artifact_sha256: str
    section_count: int
    review_count: int
    source_sha256: str
    format_verdict: str = "NON_ICH"
    format_confidence: float = 0.0
    missing_required_sections: list[str] = dataclasses.field(
        default_factory=list,
    )
    input_tokens: int = 0
    output_tokens: int = 0
    chunk_count: int = 0
    retemplated_artifact_key: str = ""
    retemplated_artifact_sha256: str = ""


class LlmRetemplater:
    """Claude-based ICH E6(R3) protocol retemplater (PTCV-64).

    Two-pass architecture:
    1. Page-level classification via Claude Opus 4.6 (which pages
       belong to which ICH section).
    2. Deterministic full-text assembly (group original text blocks
       by assigned section, concatenate in page order).

    Produces both sections.parquet (with full content_text per
    section) and retemplated_protocol.md (downloadable artifact).

    Args:
        gateway: StorageGateway instance.
        review_queue: ReviewQueue for low-confidence sections.
        claude_model: Anthropic model ID (default claude-opus-4-6).
    """

    _DEFAULT_MODEL = "claude-opus-4-6"

    def __init__(
        self,
        gateway: Optional[StorageGateway] = None,
        review_queue: Optional[ReviewQueue] = None,
        claude_model: str = _DEFAULT_MODEL,
    ) -> None:
        if gateway is None:
            gateway = FilesystemAdapter(
                root=Path("C:/Dev/PTCV/data"),
            )
        if review_queue is None:
            review_queue = ReviewQueue(db_path=_DEFAULT_REVIEW_DB)

        self._gateway = gateway
        self._review_queue = review_queue
        self._claude_model = claude_model
        self._claude: Any = None  # Lazy init
        self._corpus = load_corpus()  # PTCV-47: few-shot

        self._gateway.initialise()
        self._review_queue.initialise()

    def _get_client(self) -> Any:
        """Lazy-initialize Anthropic client."""
        if self._claude is None:
            try:
                import anthropic
            except ImportError:
                logger.warning(
                    "LLMRetemplater: 'anthropic' package not installed "
                    "— LLM retemplating unavailable. "
                    "Install with: pip install anthropic"
                )
                raise
            self._claude = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
        return self._claude

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retemplate(
        self,
        text_blocks: list[dict],
        registry_id: str,
        source_run_id: str = "",
        source_sha256: str = "",
        soa_summary: Optional[dict] = None,
        who: str = _USER,
    ) -> RetemplatingResult:
        """Retemplating protocol text into ICH E6(R3) structure.

        Two-pass approach:
        1. LLM classifies pages into ICH section codes.
        2. Original text blocks are assembled per section code.

        Args:
            text_blocks: List of dicts with 'page_number' and
                'text' keys, from the extraction stage.
            registry_id: Trial identifier (NCT-ID or EUCT-Code).
            source_run_id: run_id from the extraction stage.
            source_sha256: SHA-256 of the text_blocks.parquet.
            soa_summary: Optional dict with SoA extraction summary
                (visit_count, activity_count, visit_names) to
                enrich B.4 section. None if SoA found nothing.
            who: Actor identifier for audit trail.

        Returns:
            RetemplatingResult with sections and markdown artifacts.

        Raises:
            ValueError: If text_blocks is empty.
        """
        if not text_blocks:
            raise ValueError("text_blocks must not be empty")

        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # 0. Detect and exclude TOC pages (PTCV-106)
        toc_pages = self._detect_toc_pages(text_blocks)

        # 1. Chunk text blocks by page ranges
        chunks = self._chunk_by_pages(text_blocks, toc_pages=toc_pages)

        # 2. PASS 1: LLM page-level classification
        all_assignments: list[_PageAssignment] = []
        total_input = 0
        total_output = 0

        for ci, (chunk_text, page_range) in enumerate(chunks):
            assignments, in_tok, out_tok = self._classify_chunk(
                chunk_text=chunk_text,
                page_range=page_range,
                registry_id=registry_id,
                soa_summary=soa_summary,
                chunk_index=ci,
            )
            all_assignments.extend(assignments)
            total_input += in_tok
            total_output += out_tok

        # 3. PASS 2: Deterministic full-text assembly
        sections = self._assemble_sections(
            assignments=all_assignments,
            text_blocks=text_blocks,
            run_id=run_id,
            source_run_id=source_run_id,
            source_sha256=source_sha256,
            registry_id=registry_id,
            toc_pages=toc_pages,
        )

        # 4. Enrich B.4 with SoA summary if available
        if soa_summary:
            sections = self._enrich_b4_with_soa(
                sections, soa_summary, run_id,
                source_run_id, source_sha256, registry_id,
            )

        # 5. Fallback: if no sections produced, legacy block
        if not sections:
            sections = self._legacy_fallback(
                text_blocks, registry_id, run_id,
                source_run_id, source_sha256,
            )

        # 6. Compute format verdict
        from .parser import _compute_format_verdict

        fmt_verdict, fmt_conf, missing = _compute_format_verdict(
            sections,
        )

        # 7. Stamp timestamps
        for sec in sections:
            sec.extraction_timestamp_utc = timestamp

        # 8. Write sections.parquet
        parquet_bytes = sections_to_parquet(sections)
        artifact_key = f"ich-json/{run_id}/sections.parquet"
        artifact = self._gateway.put_artifact(
            key=artifact_key,
            data=parquet_bytes,
            content_type="application/vnd.apache.parquet",
            run_id=run_id,
            source_hash=source_sha256,
            user=who,
            immutable=False,
            stage="retemplating",
            registry_id=registry_id,
        )

        # 9. Generate and write retemplated_protocol.md
        md = self._generate_retemplated_markdown(
            sections, registry_id, fmt_verdict, fmt_conf,
        )
        md_key = f"ich-json/{run_id}/retemplated_protocol.md"
        md_artifact = self._gateway.put_artifact(
            key=md_key,
            data=md.encode("utf-8"),
            content_type="text/markdown",
            run_id=run_id,
            source_hash=source_sha256,
            user=who,
            immutable=False,
            stage="retemplating",
            registry_id=registry_id,
        )

        # 10. Review queue for low-confidence sections
        review_count = 0
        for sec in sections:
            if sec.review_required:
                self._review_queue.enqueue(
                    ReviewQueueEntry(
                        run_id=run_id,
                        registry_id=registry_id,
                        section_code=sec.section_code,
                        confidence_score=sec.confidence_score,
                        content_json=sec.content_json,
                        queue_timestamp_utc=timestamp,
                    )
                )
                review_count += 1

        logger.info(
            "Retemplating complete: run_id=%s sections=%d "
            "tokens_in=%d tokens_out=%d chunks=%d",
            run_id, len(sections), total_input,
            total_output, len(chunks),
        )

        return RetemplatingResult(
            run_id=run_id,
            registry_id=registry_id,
            artifact_key=artifact_key,
            artifact_sha256=artifact.sha256,
            section_count=len(sections),
            review_count=review_count,
            source_sha256=source_sha256,
            format_verdict=fmt_verdict,
            format_confidence=fmt_conf,
            missing_required_sections=missing,
            input_tokens=total_input,
            output_tokens=total_output,
            chunk_count=len(chunks),
            retemplated_artifact_key=md_key,
            retemplated_artifact_sha256=md_artifact.sha256,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_toc_pages(text_blocks: list[dict]) -> set[int]:
        """Detect TOC page numbers from text blocks (PTCV-106).

        Aggregates text per page, then uses toc_extractor heuristics
        to identify TOC pages.  Also checks adjacent pages for
        multi-page TOCs (continuation pages without a header).

        Returns:
            Set of 1-based page numbers that are TOC pages.
        """
        page_texts: dict[int, list[str]] = defaultdict(list)
        for block in text_blocks:
            page = block.get("page_number", 0)
            text = block.get("text", "")
            if text.strip():
                page_texts[page].append(text)

        toc_pages: set[int] = set()
        for page_num in sorted(page_texts.keys()):
            if page_num > 15:
                break
            full = "\n".join(page_texts[page_num])
            if _is_toc_page(full) and _count_toc_lines(full) >= 3:
                toc_pages.add(page_num)

        # Check continuation pages adjacent to detected TOC
        if toc_pages:
            last_toc = max(toc_pages)
            for check in range(last_toc + 1, last_toc + 4):
                if check not in page_texts:
                    break
                full = "\n".join(page_texts[check])
                if _count_toc_lines(full) >= 3:
                    toc_pages.add(check)
                else:
                    break

        if toc_pages:
            logger.info(
                "PTCV-106: Detected TOC pages %s — excluding "
                "from retemplating",
                sorted(toc_pages),
            )
        return toc_pages

    def _chunk_by_pages(
        self, text_blocks: list[dict],
        toc_pages: set[int] | None = None,
    ) -> list[tuple[str, tuple[int, int]]]:
        """Split text blocks into chunks fitting the context window.

        Groups consecutive pages until cumulative character count
        exceeds _MAX_CHUNK_CHARS, then starts a new chunk.

        Returns:
            List of (chunk_text, (start_page, end_page)) tuples.
        """
        sorted_blocks = sorted(
            text_blocks,
            key=lambda b: (b.get("page_number", 0), 0),
        )

        chunks: list[tuple[str, tuple[int, int]]] = []
        current_texts: list[str] = []
        current_chars = 0
        start_page = (
            sorted_blocks[0].get("page_number", 1)
            if sorted_blocks
            else 1
        )
        current_page = start_page

        _toc = toc_pages or set()

        for block in sorted_blocks:
            page = block.get("page_number", 0)
            text = block.get("text", "")
            if not text.strip():
                continue
            if page in _toc:
                continue  # PTCV-106: skip TOC pages

            if (
                current_chars + len(text) > _MAX_CHUNK_CHARS
                and current_texts
            ):
                chunk_text = "\n".join(current_texts)
                chunks.append(
                    (chunk_text, (start_page, current_page)),
                )
                current_texts = []
                current_chars = 0
                start_page = page

            current_texts.append(text)
            current_chars += len(text)
            current_page = page

        if current_texts:
            chunk_text = "\n".join(current_texts)
            chunks.append(
                (chunk_text, (start_page, current_page)),
            )

        return chunks

    def _classify_chunk(
        self,
        chunk_text: str,
        page_range: tuple[int, int],
        registry_id: str,
        soa_summary: Optional[dict],
        chunk_index: int = 0,
    ) -> tuple[list[_PageAssignment], int, int]:
        """Call Claude to classify pages into ICH sections (Pass 1).

        Returns:
            Tuple of (assignments, input_tokens, output_tokens).
        """
        prompt = self._build_classification_prompt(
            chunk_text, page_range, registry_id, soa_summary,
            chunk_index=chunk_index,
        )

        client = self._get_client()
        resp = client.messages.create(
            model=self._claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        input_tokens = getattr(resp.usage, "input_tokens", 0)
        output_tokens = getattr(resp.usage, "output_tokens", 0)

        first_block = resp.content[0]
        if not hasattr(first_block, "text"):
            return [], input_tokens, output_tokens

        raw: str = first_block.text.strip()  # type: ignore[union-attr]
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "LLM returned unparseable JSON for pages %d-%d",
                page_range[0], page_range[1],
            )
            return [], input_tokens, output_tokens

        if not isinstance(parsed, list):
            parsed = parsed.get("sections", [parsed])

        assignments: list[_PageAssignment] = []
        for item in parsed:
            code = item.get("section_code", "")
            if code not in _ICH_SECTION_DEFS:
                continue
            confidence = float(item.get("confidence", 0.5))

            # Accept both 'pages' (new format) and 'text_excerpt'
            # (legacy format) for backward compatibility
            pages = item.get("pages", [])
            if not pages:
                # Legacy response: assign all pages in chunk range
                pages = list(
                    range(page_range[0], page_range[1] + 1)
                )

            assignments.append(
                _PageAssignment(
                    section_code=code,
                    confidence=round(confidence, 4),
                    key_concepts=item.get("key_concepts", []),
                    pages=pages,
                )
            )

        return assignments, input_tokens, output_tokens

    # ------------------------------------------------------------------
    # Few-shot exemplar helpers (PTCV-47)
    # ------------------------------------------------------------------

    def _select_exemplars(
        self,
        chunk_index: int = 0,
    ) -> list[CorpusExemplar]:
        """Select diverse few-shot exemplars for a classification prompt.

        Rotates through available section codes based on *chunk_index*
        so different chunks see different examples.

        Returns up to ``_MAX_EXEMPLARS`` exemplars, each truncated to
        ``_MAX_EXEMPLAR_CHARS`` in their ``.text`` field.
        """
        if not self._corpus:
            return []

        codes = sorted(self._corpus.keys())
        if not codes:
            return []

        selected: list[CorpusExemplar] = []
        n_codes = len(codes)

        for i in range(_MAX_EXEMPLARS):
            idx = (chunk_index * _MAX_EXEMPLARS + i) % n_codes
            code = codes[idx]
            exemplars = get_exemplars(self._corpus, code, max_exemplars=1)
            if exemplars:
                selected.append(exemplars[0])

        return selected

    @staticmethod
    def _format_exemplar_block(
        exemplars: list[CorpusExemplar],
    ) -> str:
        """Format exemplars as a prompt block with <example> tags."""
        if not exemplars:
            return ""

        lines = [
            "\nHere are examples of correctly classified "
            "protocol sections:\n",
        ]
        for i, ex in enumerate(exemplars, 1):
            text = ex.text[:_MAX_EXEMPLAR_CHARS]
            lines.append(
                f"Example {i} ({ex.section_code} "
                f"\u2014 {ex.section_name}):\n"
                f"<example>\n{text}\n</example>\n"
            )
        return "\n".join(lines)

    def _build_classification_prompt(
        self,
        chunk_text: str,
        page_range: tuple[int, int],
        registry_id: str,
        soa_summary: Optional[dict],
        stage: str = "retemplater",
        chunk_index: int = 0,
    ) -> str:
        """Build the page-level classification prompt for Claude."""
        section_defs = get_stage_prompt(stage)

        # Few-shot exemplars (PTCV-47)
        exemplars = self._select_exemplars(chunk_index)
        exemplar_block = self._format_exemplar_block(exemplars)

        soa_context = ""
        if soa_summary:
            soa_context = (
                "\n\nNote: A Schedule of Activities table was "
                "already extracted from this protocol with "
                f"{soa_summary.get('visit_count', 0)} visits and "
                f"{soa_summary.get('activity_count', 0)} "
                "activities. Visit names: "
                f"{soa_summary.get('visit_names', [])}. "
                "Content related to the schedule of activities "
                "should be classified under B.4 (Trial Design)."
            )

        return (
            "You are a GCP regulatory expert. Classify the "
            "following clinical trial protocol text (pages "
            f"{page_range[0]}-{page_range[1]}, registry: "
            f"{registry_id}) into ICH E6(R3) Appendix B "
            "sections.\n\n"
            f"ICH E6(R3) Appendix B sections:\n{section_defs}\n"
            f"{exemplar_block}"
            f"{soa_context}\n\n"
            f"Protocol text:\n<text>\n{chunk_text[:80000]}\n"
            "</text>\n\n"
            "Respond with a JSON array. Each element must have:\n"
            '  section_code: ICH section code (e.g. "B.3")\n'
            "  confidence: float 0.0-1.0\n"
            "  key_concepts: list of up to 5 key concepts\n"
            "  pages: list of integer page numbers from "
            f"{page_range[0]}-{page_range[1]} assigned to this "
            "section\n\n"
            "Rules:\n"
            "- Assign every page in the range to exactly one "
            "section\n"
            "- If a page contains content for multiple sections, "
            "assign it to the most prominent one\n"
            "- Pages with no clear match should be assigned to "
            "B.1 (General Information)\n"
            "- Set confidence=0.9+ for pages with clear ICH "
            "headings\n"
            "- Set confidence=0.5-0.8 for pages that match "
            "thematically but lack formal headings\n"
            "- Return only valid JSON (no markdown, no "
            "commentary)"
        )

    @staticmethod
    def _assemble_sections(
        assignments: list[_PageAssignment],
        text_blocks: list[dict],
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        registry_id: str,
        toc_pages: set[int] | None = None,
    ) -> list[IchSection]:
        """Deterministic full-text assembly from page assignments (Pass 2).

        Maps each text block to its assigned section code based on
        page number, then concatenates full text per section.

        Args:
            assignments: Page-level section assignments from LLM.
            text_blocks: Original text blocks from extraction.
            run_id: UUID4 for this retemplating run.
            source_run_id: Extraction run_id.
            source_sha256: SHA-256 of extraction artifact.
            registry_id: Trial identifier.
            toc_pages: Set of page numbers to exclude (PTCV-106).

        Returns:
            List of IchSection with content_text populated.
        """
        if not assignments:
            return []

        # Build page → (section_code, confidence) mapping
        # When a page appears in multiple assignments, highest
        # confidence wins
        page_map: dict[int, tuple[str, float]] = {}
        for assign in assignments:
            for page in assign.pages:
                existing = page_map.get(page)
                if (
                    existing is None
                    or assign.confidence > existing[1]
                ):
                    page_map[page] = (
                        assign.section_code,
                        assign.confidence,
                    )

        # Collect key_concepts per section code
        section_concepts: dict[str, list[str]] = defaultdict(list)
        for assign in assignments:
            section_concepts[assign.section_code].extend(
                assign.key_concepts,
            )

        # Sort text blocks by page + block position
        sorted_blocks = sorted(
            text_blocks,
            key=lambda b: (
                b.get("page_number", 0),
                b.get("block_index", 0),
            ),
        )

        # Group text blocks by assigned section
        section_blocks: dict[str, list[dict]] = defaultdict(list)
        section_confidences: dict[str, list[float]] = defaultdict(
            list,
        )
        section_pages: dict[str, list[int]] = defaultdict(list)

        _toc = toc_pages or set()

        for block in sorted_blocks:
            text = block.get("text", "")
            if not text.strip():
                continue
            page = block.get("page_number", 0)
            if page in _toc:
                continue  # PTCV-106: skip TOC pages
            code, conf = page_map.get(page, ("B.1", 0.30))
            section_blocks[code].append(block)
            section_confidences[code].append(conf)
            if page not in section_pages[code]:
                section_pages[code].append(page)

        # Build IchSection objects with full content_text
        sections: list[IchSection] = []
        for code in sorted(section_blocks.keys(), key=section_sort_key):
            blocks = section_blocks[code]
            full_text = "\n".join(
                b.get("text", "") for b in blocks
            )
            confidences = section_confidences[code]
            avg_confidence = (
                sum(confidences) / len(confidences)
                if confidences
                else 0.5
            )
            pages = section_pages[code]

            # Merge and deduplicate key_concepts
            concepts = list(
                dict.fromkeys(section_concepts.get(code, [])),
            )[:10]

            # Build backward-compatible content_json
            content: dict[str, Any] = {
                "text_excerpt": full_text[:2000],
                "key_concepts": concepts,
                "word_count": len(full_text.split()),
                "page_range": (
                    [min(pages), max(pages)] if pages else []
                ),
            }

            name = _ICH_SECTION_DEFS.get(code, "").split("(")[0]
            name = name.strip() if name else code

            sections.append(
                IchSection(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    section_code=code,
                    section_name=name,
                    content_json=json.dumps(
                        content, ensure_ascii=False,
                    ),
                    confidence_score=round(avg_confidence, 4),
                    review_required=(
                        avg_confidence < get_review_threshold(code)
                    ),
                    legacy_format=False,
                    content_text=full_text,
                )
            )

        return sections

    @staticmethod
    def _generate_retemplated_markdown(
        sections: list[IchSection],
        registry_id: str,
        format_verdict: str,
        format_confidence: float,
    ) -> str:
        """Generate a full ICH E6(R3) retemplated protocol document.

        Iterates B.1 through B.14 in canonical order, rendering
        content_text (full text) for each section.

        Returns:
            Complete markdown string.
        """
        by_code: dict[str, IchSection] = {}
        for sec in sections:
            code = sec.section_code
            if (
                code not in by_code
                or sec.confidence_score
                > by_code[code].confidence_score
            ):
                by_code[code] = sec

        lines: list[str] = [
            f"# {registry_id}: ICH E6(R3) Retemplated Protocol",
            "",
            f"**Format verdict:** {format_verdict} | "
            f"**Confidence:** {format_confidence:.2f} | "
            f"**Sections:** {len(by_code)} / "
            f"{len(_ICH_SECTION_ORDER)}",
            "",
        ]

        if format_verdict == "NON_ICH":
            lines.append(
                "> **Note:** This protocol does not follow ICH "
                "E6(R3) format. Content has been reorganized into "
                "the standard ICH structure by the retemplater."
            )
            lines.append("")

        for code, name in _ICH_SECTION_ORDER:
            matched = by_code.get(code)
            if matched is None:
                lines.append(f"## {code} {name}")
                lines.append("")
                lines.append(
                    "> *Section not detected in source protocol*",
                )
                lines.append("")
                continue

            if matched.confidence_score < get_review_threshold(code):
                lines.append(
                    f"## {code} {name} "
                    f"(confidence: {matched.confidence_score:.2f})"
                )
                lines.append("")
                lines.append(
                    "> *Low confidence -- content may be "
                    "misclassified*"
                )
            else:
                lines.append(f"## {code} {name}")

            lines.append("")

            # Use content_text (full text) if available
            text = matched.content_text
            if not text:
                # Fallback to content_json extraction
                try:
                    data = json.loads(matched.content_json)
                    text = data.get(
                        "text_excerpt",
                        data.get("text", ""),
                    )
                except (json.JSONDecodeError, TypeError):
                    text = matched.content_json

            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _enrich_b4_with_soa(
        sections: list[IchSection],
        soa_summary: dict,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        registry_id: str,
    ) -> list[IchSection]:
        """Inject SoA summary data into the B.4 section content."""
        soa_text = _format_soa_as_text(soa_summary)

        result: list[IchSection] = []
        b4_found = False
        for sec in sections:
            if sec.section_code == "B.4":
                b4_found = True
                try:
                    content = json.loads(sec.content_json)
                except (json.JSONDecodeError, TypeError):
                    content = {}
                content["soa_summary"] = soa_summary
                enriched_text = sec.content_text
                if enriched_text and soa_text:
                    enriched_text = (
                        enriched_text + "\n\n" + soa_text
                    )
                elif soa_text:
                    enriched_text = soa_text
                sec = dataclasses.replace(
                    sec,
                    content_json=json.dumps(
                        content, ensure_ascii=False,
                    ),
                    content_text=enriched_text,
                )
            result.append(sec)

        if (
            not b4_found
            and soa_summary.get("visit_count", 0) > 0
        ):
            content = {
                "text_excerpt": (
                    "Schedule of Activities (auto-generated "
                    "from extracted SoA table)"
                ),
                "key_concepts": [
                    "schedule of activities",
                    "visits",
                    "assessments",
                ],
                "soa_summary": soa_summary,
                "word_count": 0,
            }
            result.append(
                IchSection(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    section_code="B.4",
                    section_name="Trial Design",
                    content_json=json.dumps(
                        content, ensure_ascii=False,
                    ),
                    confidence_score=0.85,
                    review_required=False,
                    legacy_format=False,
                    content_text=soa_text,
                )
            )
            result.sort(key=lambda s: section_sort_key(s.section_code))

        return result

    @staticmethod
    def _legacy_fallback(
        text_blocks: list[dict],
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        """Fallback when LLM produces no sections."""
        full_text = "\n".join(
            b.get("text", "") for b in text_blocks
        )
        return [
            IchSection(
                run_id=run_id,
                source_run_id=source_run_id,
                source_sha256=source_sha256,
                registry_id=registry_id,
                section_code="B.1",
                section_name="General Information",
                content_json=json.dumps(
                    {
                        "text_excerpt": full_text[:2000].strip(),
                        "word_count": len(full_text.split()),
                        "note": (
                            "Legacy fallback -- LLM retemplating "
                            "produced no sections"
                        ),
                    },
                    ensure_ascii=False,
                ),
                confidence_score=0.10,
                review_required=True,
                legacy_format=True,
                content_text=full_text,
            )
        ]


def _format_soa_as_text(soa_summary: dict) -> str:
    """Render SoA summary as readable text for content_text."""
    parts: list[str] = ["### Schedule of Activities Summary"]
    visit_count = soa_summary.get("visit_count", 0)
    activity_count = soa_summary.get("activity_count", 0)
    visit_names = soa_summary.get("visit_names", [])

    if visit_count:
        parts.append(f"- **Visits:** {visit_count}")
    if activity_count:
        parts.append(f"- **Activities:** {activity_count}")
    if visit_names:
        names = ", ".join(str(v) for v in visit_names)
        parts.append(f"- **Visit names:** {names}")

    return "\n".join(parts)
