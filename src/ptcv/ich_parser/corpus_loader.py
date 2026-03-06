"""Load the gold-standard ICH section corpus for few-shot prompting (PTCV-46).

Provides a simple interface to load exemplars from the JSONL corpus
file and retrieve them by section code. Used by the RAGClassifier
to supply dynamic few-shot examples in the classification prompt.

Risk tier: LOW — static data loading only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CORPUS_DIR = Path(__file__).parent / "corpus"
_DEFAULT_CORPUS = _CORPUS_DIR / "gold_standard.jsonl"


@dataclass
class CorpusExemplar:
    """A single gold-standard ICH section exemplar.

    Attributes:
        section_code: ICH section code (e.g. "B.3").
        section_name: Human-readable section name.
        registry_id: Source protocol registry ID.
        text: Full section text content.
        confidence: Classification confidence (1.0 for verified).
        verified_by: Verification method.
    """

    section_code: str
    section_name: str
    registry_id: str
    text: str
    confidence: float = 1.0
    verified_by: str = "pattern_extraction"


def load_corpus(
    path: Path | str | None = None,
) -> dict[str, list[CorpusExemplar]]:
    """Load the gold-standard corpus from JSONL.

    Args:
        path: Path to the JSONL corpus file. Defaults to the
            built-in ``gold_standard.jsonl``.

    Returns:
        Dict mapping section codes to lists of CorpusExemplar.
        Empty dict if the corpus file is missing.
    """
    corpus_path = Path(path) if path else _DEFAULT_CORPUS

    if not corpus_path.exists():
        logger.warning(
            "Gold-standard corpus not found at %s. "
            "Few-shot prompting will be unavailable.",
            corpus_path,
        )
        return {}

    corpus: dict[str, list[CorpusExemplar]] = {}

    with open(corpus_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSON on line %d of %s",
                    line_num,
                    corpus_path,
                )
                continue

            exemplar = CorpusExemplar(
                section_code=data["section_code"],
                section_name=data["section_name"],
                registry_id=data["registry_id"],
                text=data["text"],
                confidence=data.get("confidence", 1.0),
                verified_by=data.get("verified_by", "unknown"),
            )

            if exemplar.section_code not in corpus:
                corpus[exemplar.section_code] = []
            corpus[exemplar.section_code].append(exemplar)

    logger.info(
        "Loaded %d exemplars across %d section codes from %s",
        sum(len(v) for v in corpus.values()),
        len(corpus),
        corpus_path,
    )
    return corpus


def get_exemplars(
    corpus: dict[str, list[CorpusExemplar]],
    section_code: str,
    max_exemplars: int = 2,
) -> list[CorpusExemplar]:
    """Retrieve exemplars for a given section code.

    Args:
        corpus: Loaded corpus from :func:`load_corpus`.
        section_code: ICH section code (e.g. "B.3").
        max_exemplars: Maximum number of exemplars to return.

    Returns:
        List of up to ``max_exemplars`` exemplars, sorted by
        confidence descending.
    """
    exemplars = corpus.get(section_code, [])
    sorted_exemplars = sorted(
        exemplars, key=lambda e: e.confidence, reverse=True
    )
    return sorted_exemplars[:max_exemplars]


def corpus_stats(
    corpus: dict[str, list[CorpusExemplar]],
) -> dict[str, Any]:
    """Compute coverage statistics for the corpus.

    Args:
        corpus: Loaded corpus from :func:`load_corpus`.

    Returns:
        Dict with total_exemplars, section_count,
        exemplars_per_section, and registry_diversity.
    """
    total = sum(len(v) for v in corpus.values())
    per_section = {
        code: len(exemplars)
        for code, exemplars in sorted(corpus.items())
    }
    diversity = {
        code: len({e.registry_id for e in exemplars})
        for code, exemplars in sorted(corpus.items())
    }

    return {
        "total_exemplars": total,
        "section_count": len(corpus),
        "exemplars_per_section": per_section,
        "registry_diversity": diversity,
    }
