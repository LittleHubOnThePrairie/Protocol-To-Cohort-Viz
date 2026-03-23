"""PubMed abstract vector builder with section-anchored embedding (PTCV-286).

Reads cached PubMed article abstracts, chunks them by structured
section headers (BACKGROUND/METHODS/RESULTS/CONCLUSIONS), maps chunks
to ICH Appendix B sections, and produces section-anchored text ready
for embedding into the FAISS RAG index.

Anchoring strategy: each chunk is prefixed with its ICH section code
and name (e.g. ``"B.4 Trial Design: This was a randomized..."``).
This bridges the domain gap between journal abstract prose and
protocol text by anchoring vectors in the same region of the
fine-tuned embedding space that the model learned during SetFit
training on registry↔protocol alignments.

See: docs/analysis/pubmed_abstract_analysis.md (PTCV-287)

Risk tier: LOW — read-only text processing of cached public data.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ICH section names for anchor prefixes
_ICH_SECTION_NAMES: dict[str, str] = {
    "B.2": "Background Information",
    "B.3": "Trial Objectives and Purpose",
    "B.4": "Trial Design",
    "B.5": "Selection of Subjects",
    "B.7": "Treatment of Participants",
    "B.8": "Assessment of Efficacy",
    "B.9": "Assessment of Safety",
    "B.10": "Statistics",
}

# Structured abstract section header pattern
_SECTION_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(BACKGROUND|METHODS?|RESULTS?|CONCLUSIONS?|"
    r"OBJECTIVE|PURPOSE|AIM|DESIGN|SETTING|PATIENTS?|"
    r"INTERVENTIONS?|MAIN\s+OUTCOME|MEASUREMENTS?|"
    r"FINDINGS|INTERPRETATION|SIGNIFICANCE|CONTEXT)\s*[:\.]?\s*",
    re.IGNORECASE,
)

# Header → ICH section mapping (unambiguous headers)
_HEADER_TO_ICH: dict[str, str] = {
    "BACKGROUND": "B.2",
    "CONTEXT": "B.2",
    "OBJECTIVE": "B.3",
    "PURPOSE": "B.3",
    "AIM": "B.3",
    "DESIGN": "B.4",
    "SETTING": "B.4",
    "PATIENTS": "B.5",
    "INTERVENTIONS": "B.7",
    "INTERVENTION": "B.7",
    "MAIN OUTCOME": "B.8",
    "MEASUREMENTS": "B.8",
    "CONCLUSIONS": "B.3",
    "CONCLUSION": "B.3",
    "INTERPRETATION": "B.3",
    "SIGNIFICANCE": "B.2",
}

# Sub-classification keywords for mixed sections (METHODS, RESULTS)
_DESIGN_KW = re.compile(
    r"randomiz|double.blind|placebo.control|open.label|"
    r"crossover|parallel.group|phase\s+[I1-4]|"
    r"non.inferiority|superiority|single.arm",
    re.IGNORECASE,
)
_DOSING_KW = re.compile(
    r"mg/kg|mg\s+(?:daily|twice|once|every)|"
    r"dose\s+(?:escalation|reduction|modification)|"
    r"administered\s+(?:orally|intravenously|subcutaneously)|"
    r"treatment\s+cycle",
    re.IGNORECASE,
)
_SAFETY_KW = re.compile(
    r"adverse\s+event|serious\s+adverse|"
    r"dose.limiting\s+toxicit|maximum\s+tolerated|"
    r"grade\s+[3-5]|safety\s+(?:profile|assessment)|"
    r"neutropenia|thrombocytopenia|discontinu.*due\s+to",
    re.IGNORECASE,
)
_EFFICACY_KW = re.compile(
    r"overall\s+response|complete\s+response|"
    r"progression.free\s+survival|overall\s+survival|"
    r"hazard\s+ratio|RECIST|objective\s+response|"
    r"median\s+(?:PFS|OS|DOR)|duration\s+of\s+response",
    re.IGNORECASE,
)

# Max tokens for a single chunk (model window)
_MAX_CHUNK_TOKENS = 256
_MAX_CHUNK_CHARS = 1200  # ~256 tokens * ~4.7 chars/token


@dataclasses.dataclass
class AnchoredChunk:
    """A section-anchored text chunk ready for embedding.

    Attributes:
        anchored_text: Text with ICH prefix (for embedding).
        raw_text: Original chunk text (without prefix).
        ich_code: ICH section code (e.g. ``"B.4"``).
        ich_name: ICH section name (e.g. ``"Trial Design"``).
        abstract_section: Original abstract header (e.g. ``"METHODS"``).
        nct_id: Trial registry ID.
        pmid: PubMed article ID.
        source: Always ``"pubmed"``.
    """

    anchored_text: str
    raw_text: str
    ich_code: str
    ich_name: str
    abstract_section: str
    nct_id: str
    pmid: str
    source: str = "pubmed"


def build_chunks_from_cache(
    cache_dir: str | Path,
) -> list[AnchoredChunk]:
    """Build section-anchored chunks from all cached PubMed abstracts.

    Reads ``{NCT*}_pubmed.json`` files from *cache_dir*, splits
    structured abstracts by section headers, maps to ICH codes,
    and anchors with ICH prefix.

    Args:
        cache_dir: Directory containing PubMed cache JSON files.

    Returns:
        List of AnchoredChunk ready for embedding and indexing.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.warning(
            "PubMed cache dir not found: %s", cache_path,
        )
        return []

    all_chunks: list[AnchoredChunk] = []
    files_processed = 0
    articles_processed = 0

    for f in sorted(cache_path.glob("*_pubmed.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping %s: %s", f.name, exc)
            continue

        nct_id = data.get("nct_id", f.stem.replace("_pubmed", ""))
        files_processed += 1

        for article in data.get("articles", []):
            abstract = article.get("abstract", "")
            if not abstract or len(abstract) < 50:
                continue

            pmid = str(article.get("pmid", ""))
            articles_processed += 1

            chunks = _chunk_abstract(abstract, nct_id, pmid)
            all_chunks.extend(chunks)

    logger.info(
        "PubMed vector builder: %d files, %d articles, "
        "%d anchored chunks",
        files_processed, articles_processed, len(all_chunks),
    )
    return all_chunks


def seed_pubmed_vectors(
    rag_index: Any,
    chunks: list[AnchoredChunk],
) -> int:
    """Add anchored PubMed chunks to the RAG index.

    Embeds the ``anchored_text`` (with ICH prefix) and adds to
    the FAISS index with PubMed-specific metadata.

    Args:
        rag_index: A ``RagIndex`` instance (must be initialised).
        chunks: Output from ``build_chunks_from_cache()``.

    Returns:
        Number of vectors added.
    """
    if not chunks:
        return 0

    from ptcv.ich_parser.rag_index import _ensure_rag_deps
    _ensure_rag_deps()

    if rag_index._index is None:
        raise RuntimeError(
            "RagIndex not initialised. Call load() or "
            "build_from_sections() first.",
        )

    # Deduplicate: skip chunks already in index
    existing = {
        (m.get("nct_id", ""), m.get("pmid", ""), m.get("section_code", ""))
        for m in rag_index._metadata
        if m.get("source") == "pubmed"
    }

    to_add = [
        c for c in chunks
        if (c.nct_id, c.pmid, c.ich_code) not in existing
    ]
    if not to_add:
        logger.info("PubMed seeder: all chunks already indexed")
        return 0

    # Embed anchored text (with ICH prefix)
    texts = [c.anchored_text for c in to_add]
    embeddings = rag_index._encode(texts)
    rag_index._index.add(embeddings)

    # Append metadata
    for chunk in to_add:
        rag_index._metadata.append({
            "section_code": chunk.ich_code,
            "section_name": chunk.ich_name,
            "registry_id": chunk.nct_id,
            "confidence_score": 0.80,
            "content_text": chunk.raw_text[:500],
            "source": "pubmed",
            "nct_id": chunk.nct_id,
            "pmid": chunk.pmid,
            "abstract_section": chunk.abstract_section,
            "content_preview": chunk.raw_text[:200],
        })

    logger.info(
        "PubMed seeder: added %d vectors (%d skipped as duplicates)",
        len(to_add), len(chunks) - len(to_add),
    )
    return len(to_add)


# ------------------------------------------------------------------
# Internal: chunking and mapping
# ------------------------------------------------------------------


def _chunk_abstract(
    abstract: str,
    nct_id: str,
    pmid: str,
) -> list[AnchoredChunk]:
    """Split an abstract into section-anchored chunks."""
    parts = _SECTION_HEADER_RE.split(abstract)

    if len(parts) <= 2:
        # Unstructured — treat as single chunk, classify by content
        return _chunk_unstructured(abstract, nct_id, pmid)

    chunks: list[AnchoredChunk] = []
    # parts alternates: [preamble, header1, content1, header2, ...]
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].upper().strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if not content or len(content) < 30:
            continue

        # Map header to ICH code(s)
        ich_mappings = _map_header_to_ich(header, content)

        for ich_code, sub_content in ich_mappings:
            # Sub-split if too long
            for sub_chunk in _split_if_needed(sub_content):
                ich_name = _ICH_SECTION_NAMES.get(ich_code, "")
                anchored = f"{ich_code} {ich_name}: {sub_chunk}"
                chunks.append(AnchoredChunk(
                    anchored_text=anchored,
                    raw_text=sub_chunk,
                    ich_code=ich_code,
                    ich_name=ich_name,
                    abstract_section=header,
                    nct_id=nct_id,
                    pmid=pmid,
                ))

    return chunks


def _chunk_unstructured(
    abstract: str,
    nct_id: str,
    pmid: str,
) -> list[AnchoredChunk]:
    """Chunk an unstructured abstract by sentences, classify by content."""
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    if not sentences:
        return []

    # Group into ~400-char chunks
    groups: list[str] = []
    current: list[str] = []
    current_len = 0
    for s in sentences:
        if current_len + len(s) > 400 and current:
            groups.append(" ".join(current))
            current = []
            current_len = 0
        current.append(s)
        current_len += len(s)
    if current:
        groups.append(" ".join(current))

    chunks: list[AnchoredChunk] = []
    for group in groups:
        ich_code = _classify_by_content(group)
        if ich_code:
            ich_name = _ICH_SECTION_NAMES.get(ich_code, "")
            anchored = f"{ich_code} {ich_name}: {group}"
            chunks.append(AnchoredChunk(
                anchored_text=anchored,
                raw_text=group,
                ich_code=ich_code,
                ich_name=ich_name,
                abstract_section="UNSTRUCTURED",
                nct_id=nct_id,
                pmid=pmid,
            ))

    return chunks


def _map_header_to_ich(
    header: str,
    content: str,
) -> list[tuple[str, str]]:
    """Map an abstract section header to ICH code(s).

    Unambiguous headers (BACKGROUND, OBJECTIVE, etc.) map directly.
    Ambiguous headers (METHODS, RESULTS) are sub-classified by content.

    Returns:
        List of (ich_code, content_text) tuples. May return multiple
        entries for mixed sections.
    """
    # Normalize header
    h = header.upper().strip()
    # Remove trailing 'S' for singular forms
    h_singular = h.rstrip("S") if h.endswith("S") and h not in (
        "RESULTS", "CONCLUSIONS", "PATIENTS",
    ) else h

    # Direct mapping for unambiguous headers
    if h in _HEADER_TO_ICH:
        return [(_HEADER_TO_ICH[h], content)]
    if h_singular in _HEADER_TO_ICH:
        return [(_HEADER_TO_ICH[h_singular], content)]

    # METHODS → sub-classify B.4 vs B.7
    if h in ("METHODS", "METHOD"):
        return _subclassify_methods(content)

    # RESULTS, FINDINGS → sub-classify B.8 vs B.9
    if h in ("RESULTS", "RESULT", "FINDINGS"):
        return _subclassify_results(content)

    # Unknown header — classify by content
    ich_code = _classify_by_content(content)
    if ich_code:
        return [(ich_code, content)]
    return [("B.4", content)]  # Default fallback


def _subclassify_methods(content: str) -> list[tuple[str, str]]:
    """Split METHODS content into B.4 (design) and B.7 (dosing)."""
    has_design = bool(_DESIGN_KW.search(content))
    has_dosing = bool(_DOSING_KW.search(content))

    if has_design and has_dosing:
        # Both present — return as B.4 (design is dominant in METHODS)
        return [("B.4", content)]
    if has_dosing and not has_design:
        return [("B.7", content)]
    return [("B.4", content)]  # Default: design


def _subclassify_results(content: str) -> list[tuple[str, str]]:
    """Split RESULTS content into B.8 (efficacy) and B.9 (safety)."""
    has_safety = bool(_SAFETY_KW.search(content))
    has_efficacy = bool(_EFFICACY_KW.search(content))

    if has_safety and has_efficacy:
        # Both — return as B.8 (primary signal in RESULTS)
        return [("B.8", content)]
    if has_safety and not has_efficacy:
        return [("B.9", content)]
    return [("B.8", content)]  # Default: efficacy


def _classify_by_content(text: str) -> str:
    """Classify unstructured text by keyword content."""
    scores: dict[str, int] = {}
    for code, pattern in [
        ("B.4", _DESIGN_KW),
        ("B.7", _DOSING_KW),
        ("B.8", _EFFICACY_KW),
        ("B.9", _SAFETY_KW),
    ]:
        hits = len(pattern.findall(text))
        if hits:
            scores[code] = hits

    if not scores:
        return "B.4"  # Default
    return max(scores, key=lambda k: scores[k])


def _split_if_needed(text: str) -> list[str]:
    """Split text into model-window-sized chunks if needed."""
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    # Split by sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for s in sentences:
        if current_len + len(s) > _MAX_CHUNK_CHARS and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(s)
        current_len += len(s)

    if current:
        chunks.append(" ".join(current))

    return chunks
