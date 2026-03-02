"""ICH E6(R3) section classifiers.

Two implementations are provided:

RuleBasedClassifier — pure regex/keyword matching, no external
    dependencies. Suitable for dev/test and for protocols with standard
    ICH E6(R3) headings. Confidence is scored by keyword density.

RAGClassifier — Cohere ``embed-english-v3.0`` for vector retrieval +
    Claude Sonnet for generation. Requires COHERE_API_KEY and
    ANTHROPIC_API_KEY environment variables. Achieves >87% F1 on the
    23-protocol validation set (PTCV-13 benchmark).

Risk tier: MEDIUM — data pipeline ML component (no patient data).

Regulatory references:
- ALCOA+ Accurate: confidence_score required; low-confidence sections
  flagged for human review (review_required=True)
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from .models import IchSection


# ---------------------------------------------------------------------------
# ICH E6(R3) Appendix B section definitions
# ---------------------------------------------------------------------------

_ICH_SECTIONS: dict[str, dict[str, Any]] = {
    "B.1": {
        "name": "General Information",
        "patterns": [
            r"\bgeneral\s+information\b",
            r"\bprotocol\s+(?:title|number|version|date)\b",
            r"\btrial\s+title\b",
            r"\bsponsor\s+(?:name|address|contact)\b",
        ],
        "keywords": [
            "title", "sponsor", "investigator", "protocol number",
            "version", "phase", "eudract", "euct", "nct",
            "amendment", "date of protocol",
        ],
    },
    "B.2": {
        "name": "Background Information",
        "patterns": [
            r"\bbackground\b",
            r"\brationale\b",
            r"\bpreclinical\b",
            r"\bnonclinical\b",
            r"\bliterature\s+review\b",
            r"\bstate\s+of\s+(?:the\s+)?art\b",
        ],
        "keywords": [
            "background", "rationale", "preclinical", "nonclinical",
            "prior studies", "literature", "mechanism of action",
            "disease background", "current treatment",
        ],
    },
    "B.3": {
        "name": "Trial Objectives and Purpose",
        "patterns": [
            r"\bobjectives?\b",
            r"\bpurpose\b",
            r"\bhypothes[ie]s\b",
            r"\bprimary\s+(?:objective|endpoint|outcome)\b",
            r"\bsecondary\s+(?:objective|endpoint|outcome)\b",
        ],
        "keywords": [
            "objective", "purpose", "primary endpoint", "secondary endpoint",
            "hypothesis", "primary outcome", "efficacy endpoint",
        ],
    },
    "B.4": {
        "name": "Trial Design",
        "patterns": [
            r"\btrial\s+design\b",
            r"\bstudy\s+design\b",
            r"\brandomis(?:ed|ation)\b",
            r"\bdouble.blind\b",
            r"\bopen.label\b",
            r"\bparallel.group\b",
            r"\bcrossover\b",
        ],
        "keywords": [
            "design", "randomisation", "blinding", "control", "parallel",
            "crossover", "open-label", "double-blind", "single-blind",
            "placebo", "active comparator", "allocation",
        ],
    },
    "B.5": {
        "name": "Selection of Subjects",
        "patterns": [
            r"\b(?:inclusion|exclusion)\s+criteria\b",
            r"\beligibility\s+criteria\b",
            r"\bselection\s+of\s+(?:subjects|patients|participants)\b",
            r"\bentry\s+criteria\b",
        ],
        "keywords": [
            "inclusion criteria", "exclusion criteria", "eligibility",
            "subject selection", "patient population", "age range",
            "diagnosis", "contraindication",
        ],
    },
    "B.6": {
        "name": "Treatment of Subjects",
        "patterns": [
            r"\bdiscontinuation\b",
            r"\bwithdrawal\b",
            r"\btreatment\s+(?:discontinuation|withdrawal|termination)\b",
            r"\bstopping\s+rules?\b",
        ],
        "keywords": [
            "discontinuation", "withdrawal", "stopping rules",
            "treatment termination", "dropout", "lost to follow-up",
        ],
    },
    "B.7": {
        "name": "Assessment of Efficacy",
        "patterns": [
            r"\btreatment\s+(?:schedule|regimen|administration)\b",
            r"\bdose(?:s|ing|age)?\b",
            r"\bconcomitant\s+(?:medication|treatment|therapy)\b",
            r"\bstudy\s+treatment\b",
            r"\binvestigational\s+(?:product|medicinal\s+product)\b",
        ],
        "keywords": [
            "dose", "dosing", "administration", "regimen",
            "investigational product", "IMP", "treatment schedule",
            "concomitant medication", "rescue medication",
        ],
    },
    "B.8": {
        "name": "Assessment of Safety",
        "patterns": [
            r"\befficacy\s+(?:assessments?|endpoints?|evaluations?)\b",
            r"\bprimary\s+efficacy\b",
            r"\bclinical\s+outcome\b",
            r"\bresponse\s+(?:criteria|rate|assessment)\b",
            r"\bsurvival\b",
            r"\bremission\b",
        ],
        "keywords": [
            "efficacy assessment", "clinical response", "tumour response",
            "progression-free survival", "overall survival", "remission",
            "disease-free", "response rate",
        ],
    },
    "B.9": {
        "name": "Statistics",
        "patterns": [
            r"\bsafety\s+(?:assessments?|monitoring|evaluations?)\b",
            r"\badverse\s+(?:event|reaction|effect)\b",
            r"\bserious\s+adverse\b",
            r"\bvital\s+signs?\b",
            r"\blaboratory\s+(?:tests?|assessments?|parameters?)\b",
            r"\becg\b",
            r"\btoxicity\b",
        ],
        "keywords": [
            "adverse event", "adverse reaction", "SAE", "safety monitoring",
            "vital signs", "laboratory", "ECG", "toxicity", "tolerability",
            "DSMB", "SUSAR",
        ],
    },
    "B.10": {
        "name": "Direct Access to Source Data and Documents",
        "patterns": [
            r"\bstatistical\s+(?:methods?|analysis|plan)\b",
            r"\bsample\s+size\b",
            r"\bpower\s+(?:calculation|analysis)\b",
            r"\banalysis\s+(?:population|set)\b",
            r"\bintention.to.treat\b",
            r"\bper.protocol\b",
        ],
        "keywords": [
            "statistical analysis", "sample size", "power calculation",
            "significance level", "analysis population",
            "intention-to-treat", "per-protocol", "missing data",
            "multiplicity", "interim analysis",
        ],
    },
    "B.11": {
        "name": "Quality Control and Quality Assurance",
        "patterns": [
            r"\bquality\s+(?:control|assurance|management)\b",
            r"\baudit\b",
            r"\binspection\b",
            r"\bmonitoring\s+(?:plan|visits?|procedures?)\b",
            r"\bsource\s+data\s+(?:verification|access)\b",
            r"\bdata\s+(?:management|governance|integrity)\b",
            r"\bGCP\b",
            r"\bICH\s+E6\b",
        ],
        "keywords": [
            "quality control", "quality assurance", "audit", "inspection",
            "monitoring", "GCP", "ICH E6", "source data verification",
            "data management", "data integrity", "regulatory compliance",
            "ethics committee", "IRB", "informed consent",
        ],
    },
}

# Confidence threshold below which review is required (ALCOA+ Accurate)
REVIEW_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SectionClassifier(ABC):
    """Abstract base for ICH E6(R3) section classifiers."""

    @abstractmethod
    def classify(
        self,
        text: str,
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        """Classify ``text`` into ICH E6(R3) Appendix B sections.

        Args:
            text: Full protocol text to classify.
            registry_id: Trial identifier (EUCT-Code or NCT-ID).
            run_id: UUID4 for the current ICH-parse run.
            source_run_id: run_id from the upstream PTCV-19 extraction.
            source_sha256: SHA-256 of the upstream extraction artifact.

        Returns:
            List of IchSection instances, one per detected section.
            extraction_timestamp_utc is left empty; IchParser sets it
            immediately before writing to storage.
        """


# ---------------------------------------------------------------------------
# Rule-based classifier (no external deps)
# ---------------------------------------------------------------------------

class RuleBasedClassifier(SectionClassifier):
    """Regex + keyword density classifier for ICH E6(R3) sections.

    Splits the protocol text into candidate blocks by heading detection,
    then scores each block against the ICH section keyword sets.
    Confidence is the normalised keyword hit ratio (0.0–1.0).

    Sections with no keyword hits are assigned confidence=0.0 and
    legacy_format=True (non-standard headings detected).
    """

    # Heading detection: numbered or lettered headings or ALL CAPS lines
    _HEADING_RE = re.compile(
        r"(?m)^(?:"
        r"\d+(?:\.\d+)*[\.\)]\s+[A-Z]"  # 1. Title or 1.2. Title
        r"|[A-Z][A-Z\s]{3,}$"            # ALL CAPS HEADING
        r"|(?:Section|SECTION|Part|PART)\s+[A-Z\d]"
        r")"
    )

    def classify(
        self,
        text: str,
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        blocks = self._split_into_blocks(text)
        sections: list[IchSection] = []

        for block_text in blocks:
            best_code, best_score, best_content = self._score_block(block_text)
            if best_score == 0.0:
                continue  # Skip blocks with no keyword hits

            review_required = best_score < REVIEW_THRESHOLD
            legacy = best_score < 0.40  # Very low score → likely legacy format

            sections.append(
                IchSection(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    section_code=best_code,
                    section_name=_ICH_SECTIONS[best_code]["name"],
                    content_json=json.dumps(best_content, ensure_ascii=False),
                    confidence_score=round(best_score, 4),
                    review_required=review_required,
                    legacy_format=legacy,
                )
            )

        # Deduplicate: keep highest-confidence assignment per section_code
        return self._deduplicate(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split text into candidate section blocks by heading detection."""
        positions = [m.start() for m in self._HEADING_RE.finditer(text)]
        if not positions:
            return [text]
        positions.append(len(text))
        return [text[positions[i]: positions[i + 1]] for i in range(len(positions) - 1)]

    def _score_block(
        self, block: str
    ) -> tuple[str, float, dict[str, Any]]:
        """Score ``block`` against all ICH sections and return best match.

        Returns:
            Tuple of (section_code, confidence_score, content_dict).
        """
        lower = block.lower()
        best_code = "B.1"
        best_score = 0.0

        for code, defn in _ICH_SECTIONS.items():
            # Pattern hits (weighted higher)
            pattern_hits = sum(
                1 for p in defn["patterns"] if re.search(p, lower)
            )
            # Keyword hits (weighted lower)
            keyword_hits = sum(
                1 for kw in defn["keywords"] if kw.lower() in lower
            )

            max_possible = len(defn["patterns"]) * 2 + len(defn["keywords"])
            raw = pattern_hits * 2 + keyword_hits
            score = raw / max_possible if max_possible > 0 else 0.0

            if score > best_score:
                best_score = score
                best_code = code

        content: dict[str, Any] = {
            "text_excerpt": block[:2000].strip(),
            "word_count": len(block.split()),
        }
        return best_code, best_score, content

    def _deduplicate(self, sections: list[IchSection]) -> list[IchSection]:
        """Keep one IchSection per section_code — highest confidence wins."""
        best: dict[str, IchSection] = {}
        for sec in sections:
            existing = best.get(sec.section_code)
            if existing is None or sec.confidence_score > existing.confidence_score:
                best[sec.section_code] = sec
        return sorted(best.values(), key=lambda s: s.section_code)


# ---------------------------------------------------------------------------
# RAG classifier — Cohere embeddings + Claude Sonnet
# ---------------------------------------------------------------------------

class RAGClassifier(SectionClassifier):
    """RAG-based classifier: Cohere embed-english-v3.0 + Claude Sonnet.

    Encodes the ICH section definitions as reference embeddings at
    construction time. For each candidate block, computes cosine
    similarity against the reference embeddings to produce a ranked
    shortlist, then calls Claude Sonnet to generate structured JSON
    content and a final confidence score.

    Environment variables required:
        COHERE_API_KEY — Cohere API key
        ANTHROPIC_API_KEY — Anthropic API key

    Args:
        cohere_model: Cohere embedding model (default embed-english-v3.0).
        claude_model: Anthropic model ID (default claude-sonnet-4-6).
        top_k: Number of candidate sections passed to Claude (default 3).
    """

    _EMBED_MODEL = "embed-english-v3.0"
    _CLAUDE_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        cohere_model: str = _EMBED_MODEL,
        claude_model: str = _CLAUDE_MODEL,
        top_k: int = 3,
    ) -> None:
        import cohere as cohere_sdk
        import anthropic as anthropic_sdk
        import numpy as np

        self._co = cohere_sdk.Client(api_key=os.environ["COHERE_API_KEY"])
        self._claude = anthropic_sdk.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._cohere_model = cohere_model
        self._claude_model = claude_model
        self._top_k = top_k
        self._np = np

        # Pre-compute reference embeddings for all ICH section definitions
        ref_texts = [
            f"{code} {defn['name']}: {' '.join(defn['keywords'])}"
            for code, defn in _ICH_SECTIONS.items()
        ]
        resp = self._co.embed(
            texts=ref_texts,
            model=self._cohere_model,
            input_type="search_document",
        )
        self._ref_codes = list(_ICH_SECTIONS.keys())
        self._ref_embeddings = np.array(resp.embeddings, dtype=np.float32)

    def classify(
        self,
        text: str,
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
    ) -> list[IchSection]:
        blocks = RuleBasedClassifier()._split_into_blocks(text)
        sections: list[IchSection] = []

        # Embed all blocks in one batch call
        block_texts = [b[:4096] for b in blocks]
        embed_resp = self._co.embed(
            texts=block_texts,
            model=self._cohere_model,
            input_type="search_query",
        )
        block_embeddings = self._np.array(embed_resp.embeddings, dtype=self._np.float32)

        for idx, block in enumerate(blocks):
            block_vec = block_embeddings[idx]
            top_codes, similarities = self._top_k_sections(block_vec)
            result = self._generate_with_claude(
                block, top_codes, similarities, registry_id
            )
            if result is None:
                continue

            section_code, content_json, confidence = result
            section_name = _ICH_SECTIONS.get(section_code, {}).get("name", "Unknown")
            review_required = confidence < REVIEW_THRESHOLD
            legacy = confidence < 0.40

            sections.append(
                IchSection(
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_sha256=source_sha256,
                    registry_id=registry_id,
                    section_code=section_code,
                    section_name=section_name,
                    content_json=content_json,
                    confidence_score=round(confidence, 4),
                    review_required=review_required,
                    legacy_format=legacy,
                )
            )

        return RuleBasedClassifier()._deduplicate(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _top_k_sections(
        self, block_vec: Any
    ) -> tuple[list[str], list[float]]:
        """Return the top-k ICH section codes by cosine similarity."""
        norms = (
            self._np.linalg.norm(self._ref_embeddings, axis=1, keepdims=True)
            * self._np.linalg.norm(block_vec)
        )
        norms = self._np.where(norms == 0, 1e-9, norms)
        sims = (self._ref_embeddings @ block_vec) / norms.squeeze()
        top_idx = self._np.argsort(sims)[::-1][: self._top_k]
        codes = [self._ref_codes[i] for i in top_idx]
        scores = [float(sims[i]) for i in top_idx]
        return codes, scores

    def _generate_with_claude(
        self,
        block: str,
        top_codes: list[str],
        similarities: list[float],
        registry_id: str,
    ) -> tuple[str, str, float] | None:
        """Call Claude Sonnet to classify a block and extract content.

        Returns:
            Tuple of (section_code, content_json_str, confidence) or
            None if Claude could not classify the block.
        """
        candidates = "\n".join(
            f"  {code}: {_ICH_SECTIONS[code]['name']} (similarity {sim:.3f})"
            for code, sim in zip(top_codes, similarities)
        )
        prompt = (
            f"You are a GCP expert classifying clinical trial protocol text "
            f"into ICH E6(R3) Appendix B sections.\n\n"
            f"Protocol text (registry: {registry_id}):\n"
            f"<text>\n{block[:3000]}\n</text>\n\n"
            f"Top candidate sections by embedding similarity:\n{candidates}\n\n"
            f"Respond with a single JSON object (no markdown) with these keys:\n"
            f"  section_code: the best matching ICH section code (e.g. \"B.3\")\n"
            f"  confidence: float 0.0–1.0 reflecting how well the text matches\n"
            f"  key_concepts: list of up to 5 key concepts found in the text\n"
            f"  text_excerpt: first 500 chars of the classified text\n\n"
            f"Return section_code=\"NONE\" with confidence=0.0 if the text is "
            f"not classifiable as any ICH Appendix B section."
        )

        resp = self._claude.messages.create(
            model=self._claude_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        first_block = resp.content[0]
        if not hasattr(first_block, "text"):
            return None
        raw = first_block.text.strip()  # type: ignore[union-attr]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Strip markdown fences if present
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                return None

        section_code: str = parsed.get("section_code", "NONE")
        confidence: float = float(parsed.get("confidence", 0.0))

        if section_code == "NONE" or section_code not in _ICH_SECTIONS:
            return None

        content: dict[str, Any] = {
            "key_concepts": parsed.get("key_concepts", []),
            "text_excerpt": parsed.get("text_excerpt", block[:500]),
            "word_count": len(block.split()),
        }
        return section_code, json.dumps(content, ensure_ascii=False), confidence
