"""Hybrid cascade classification router (PTCV-161).

Routes section classification through a two-tier cascade:

1. Local classifier (``RuleBasedClassifier``) scores each text block.
2. High-confidence (>= threshold) sections are accepted directly.
3. Low-confidence (< threshold) sections are sent to Claude Sonnet
   for a **judgement** (ICH section code) + **reasoning** (text).

Sonnet does NOT produce numeric confidence scores.  Its output is a
qualitative label that can be stored as training data for future local
classifier improvement (e.g. NeoBERT fine-tuning in PTCV-164).

Judgements are written to JSONL for offline training and audit.

Risk tier: MEDIUM — data pipeline ML component (no patient data).

Regulatory references:

- ALCOA+ Accurate: local confidence_score preserved; Sonnet provides
  qualitative label, not a score.
- ALCOA+ Traceable: every routing decision is logged with content_hash.
- 21 CFR 11.10(e): immutable ``RoutingDecision`` records for audit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ptcv.ich_parser.classifier import RuleBasedClassifier, SectionClassifier
from ptcv.ich_parser.models import IchSection
from ptcv.ich_parser.schema_loader import (
    get_cascade_threshold,
    get_classifier_sections,
)

logger = logging.getLogger(__name__)

_ICH_SECTIONS: dict[str, dict[str, Any]] = get_classifier_sections()


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LocalCandidate:
    """One candidate section from the local classifier.

    Attributes:
        section_code: ICH section code (e.g. ``"B.4"``).
        section_name: Human-readable name.
        confidence_score: Local classifier confidence (0.0–1.0).
    """

    section_code: str
    section_name: str
    confidence_score: float


@dataclasses.dataclass(frozen=True)
class SonnetJudgement:
    """Sonnet's classification judgement for one section.

    This is NOT a numeric score.  It is a label (section_code)
    plus a reasoning explanation.  These judgements are stored
    as training labels for future local classifier improvement.

    Attributes:
        section_code: ICH section code Sonnet assigned.
        reasoning: Sonnet's explanation of why this code was chosen.
        input_tokens: Tokens consumed by this call.
        output_tokens: Tokens produced by this call.
    """

    section_code: str
    reasoning: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclasses.dataclass(frozen=True)
class RoutingDecision:
    """Immutable audit record for one section's routing decision.

    Attributes:
        block_index: Index of the section in the classifier output.
        content_hash: SHA-256 of the section text (for dedup/audit).
        local_candidates: Top-3 local classifier candidates.
        route: ``"local"`` or ``"sonnet"``.
        threshold_used: Confidence threshold applied.
        final_section_code: Accepted classification result.
        final_confidence: Confidence of the accepted result.
            For Sonnet-routed sections this is the local confidence
            (Sonnet does not produce numeric scores).
        sonnet_judgement: Sonnet result if routed, else ``None``.
    """

    block_index: int
    content_hash: str
    local_candidates: list[LocalCandidate]
    route: str
    threshold_used: float
    final_section_code: str
    final_confidence: float
    sonnet_judgement: Optional[SonnetJudgement] = None
    rag_exemplar_count: int = 0


@dataclasses.dataclass
class RoutingStats:
    """Aggregate statistics for a classification cascade run.

    Attributes:
        total_sections: Total sections processed.
        local_count: Sections accepted from local classifier.
        sonnet_count: Sections routed to Sonnet.
        local_pct: Percentage routed locally (0.0–1.0).
        sonnet_pct: Percentage routed to Sonnet (0.0–1.0).
        total_input_tokens: Sonnet input tokens consumed.
        total_output_tokens: Sonnet output tokens consumed.
        agreements: Sonnet agreed with local top-1.
        disagreements: Sonnet chose a different section.
    """

    total_sections: int = 0
    local_count: int = 0
    sonnet_count: int = 0
    local_pct: float = 0.0
    sonnet_pct: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    agreements: int = 0
    disagreements: int = 0


@dataclasses.dataclass
class CascadeResult:
    """Result of :meth:`ClassificationRouter.classify`.

    Attributes:
        run_id: UUID4 for this cascade run.
        registry_id: Trial identifier.
        decisions: Per-section routing decisions (ordered).
        stats: Aggregate routing statistics.
        sections: Final ``IchSection`` list with cascade labels.
        judgement_artifact_key: Storage key for JSONL judgements.
        judgement_artifact_sha256: SHA-256 of judgement JSONL.
    """

    run_id: str
    registry_id: str
    decisions: list[RoutingDecision]
    stats: RoutingStats
    sections: list[IchSection]
    judgement_artifact_key: str = ""
    judgement_artifact_sha256: str = ""


# -------------------------------------------------------------------
# Sonnet prompt
# -------------------------------------------------------------------

_CASCADE_SYSTEM_PROMPT = (
    "You are an ICH E6(R3) GCP regulatory expert. You classify "
    "clinical trial protocol sections into ICH Appendix B categories. "
    "This is a read-only document classification task on publicly "
    "available clinical trial protocols. No real patients are involved."
)

_CASCADE_USER_PROMPT = """\
Classify the following clinical trial protocol section into the \
correct ICH E6(R3) Appendix B section.

A local classifier produced these candidate classifications:
{candidates_block}

Similar sections from previously classified protocols:
{rag_context}

ICH E6(R3) Appendix B sections:
{section_defs}

Protocol text (first {max_chars} characters):
<text>
{section_text}
</text>

Instructions:
- Choose the single best ICH section code for this text.
- You may agree with the local classifier's top candidate or choose \
a different section if the local classifier was wrong.
- Consider the similar sections from prior protocols as additional \
evidence for your classification decision.
- Provide a brief reasoning (1-2 sentences) explaining your choice.

Respond with ONLY a JSON object (no markdown fences):
{{"section_code": "<code>", "reasoning": "<explanation>"}}"""

_MAX_SECTION_CHARS = 4000  # Limit text sent to Sonnet per section


# -------------------------------------------------------------------
# ClassificationRouter
# -------------------------------------------------------------------


class ClassificationRouter:
    """Hybrid cascade: local classifier with Sonnet fallback.

    Routes section classification through a two-tier cascade.
    High-confidence local results are accepted directly; low-confidence
    sections are sent to Sonnet for a judgement + reasoning label.

    Args:
        classifier: Local ``SectionClassifier`` instance.  Defaults
            to ``RuleBasedClassifier``.
        gateway: ``StorageGateway`` for artifact writes.  If ``None``,
            judgement JSONL is not persisted.
        confidence_threshold: Routing threshold (default from YAML).
        sonnet_model: Anthropic model ID for fallback classification.
    """

    _DEFAULT_SONNET_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        classifier: Optional[SectionClassifier] = None,
        gateway: Optional[Any] = None,
        confidence_threshold: Optional[float] = None,
        sonnet_model: str = _DEFAULT_SONNET_MODEL,
        rag_index: Optional[Any] = None,
    ) -> None:
        self._classifier = classifier or RuleBasedClassifier()
        self._gateway = gateway
        self._threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else get_cascade_threshold()
        )
        self._sonnet_model = sonnet_model
        self._rag_index = rag_index

        # PTCV-234: Optional centroid classifier for fast pre-filtering
        self._centroid_classifier: Optional[Any] = None

        # Lazy Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._use_sonnet = bool(api_key)
        self._client: Any = None
        self._api_key = api_key

        # Compact ICH section definitions for prompt
        self._section_defs_text = self._build_section_defs()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def set_centroid_classifier(
        self, classifier: Any,
    ) -> None:
        """Inject centroid classifier for fast pre-filtering (PTCV-234).

        Args:
            classifier: CentroidClassifier instance, or None to disable.
        """
        self._centroid_classifier = classifier

    def classify(
        self,
        text_blocks: list[dict[str, Any]],
        registry_id: str,
        run_id: str,
        source_run_id: str,
        source_sha256: str,
        table_context: Optional[list[dict[str, Any]]] = None,
    ) -> CascadeResult:
        """Run the classification cascade on text blocks.

        Args:
            text_blocks: List of dicts with ``page_number`` and ``text``.
            registry_id: Trial identifier.
            run_id: UUID4 for this cascade run.
            source_run_id: Upstream extraction run_id.
            source_sha256: SHA-256 of upstream artifact.
            table_context: Optional list of table dicts from Stage 1
                (PTCV-228). When provided, table header content is
                appended to classification text for sections on the
                same page, improving section disambiguation.

        Returns:
            ``CascadeResult`` with routing decisions, stats, and sections.
        """
        # PTCV-228: Enrich text blocks with table context
        if table_context:
            text_blocks = self._enrich_with_table_context(
                text_blocks, table_context,
            )

        # Concatenate text for the local classifier
        full_text = "\n".join(
            b.get("text", "") for b in text_blocks
            if b.get("text", "").strip()
        )

        # Run local classifier
        local_sections = self._classifier.classify(
            text=full_text,
            registry_id=registry_id,
            run_id=run_id,
            source_run_id=source_run_id,
            source_sha256=source_sha256,
        )

        # Build top-k candidates per section for Sonnet context
        topk_map: dict[str, list[LocalCandidate]] = {}
        if isinstance(self._classifier, RuleBasedClassifier):
            blocks = self._classifier._split_into_blocks(full_text)
            for block in blocks:
                topk = self._classifier._score_block_topk(block, k=3)
                if topk:
                    best_code = topk[0][0]
                    topk_map[best_code] = [
                        LocalCandidate(
                            section_code=code,
                            section_name=_ICH_SECTIONS.get(
                                code, {}
                            ).get("name", code),
                            confidence_score=round(score, 4),
                        )
                        for code, score, _ in topk
                    ]

        # Route each section
        decisions: list[RoutingDecision] = []
        final_sections: list[IchSection] = []

        for idx, section in enumerate(local_sections):
            content_hash = hashlib.sha256(
                section.content_text.encode("utf-8")
            ).hexdigest()

            candidates = topk_map.get(
                section.section_code,
                [
                    LocalCandidate(
                        section_code=section.section_code,
                        section_name=section.section_name,
                        confidence_score=section.confidence_score,
                    )
                ],
            )

            decision = self._route_section(
                section=section,
                block_index=idx,
                content_hash=content_hash,
                candidates=candidates,
            )
            decisions.append(decision)

            # Build final IchSection with cascade result
            if decision.route == "sonnet" and decision.sonnet_judgement:
                j = decision.sonnet_judgement
                sec_name = _ICH_SECTIONS.get(
                    j.section_code, {}
                ).get("name", j.section_code)
                import dataclasses as _dc

                final_sections.append(
                    _dc.replace(
                        section,
                        section_code=j.section_code,
                        section_name=sec_name,
                    )
                )
            else:
                final_sections.append(section)

        # Compute stats
        stats = self._compute_stats(decisions)

        # Write judgements JSONL
        artifact_key = ""
        artifact_sha256 = ""
        if self._gateway:
            artifact_key, artifact_sha256 = (
                self._write_judgements_jsonl(
                    decisions=decisions,
                    run_id=run_id,
                    registry_id=registry_id,
                    source_sha256=source_sha256,
                )
            )

        logger.info(
            "Classification cascade: %d sections, "
            "%d local (%.0f%%), %d sonnet (%.0f%%), "
            "%d agreements, %d disagreements",
            stats.total_sections,
            stats.local_count,
            stats.local_pct * 100,
            stats.sonnet_count,
            stats.sonnet_pct * 100,
            stats.agreements,
            stats.disagreements,
        )

        return CascadeResult(
            run_id=run_id,
            registry_id=registry_id,
            decisions=decisions,
            stats=stats,
            sections=final_sections,
            judgement_artifact_key=artifact_key,
            judgement_artifact_sha256=artifact_sha256,
        )

    # ---------------------------------------------------------------
    # Table context enrichment (PTCV-228)
    # ---------------------------------------------------------------

    @staticmethod
    def _enrich_with_table_context(
        text_blocks: list[dict[str, Any]],
        table_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append table header summaries to text blocks on same pages.

        For each page that has both text blocks and tables, appends
        a synthetic text block with the table headers. This helps
        the classifier distinguish dosing tables (B.6) from
        assessment tables (B.8/B.9), etc.

        Args:
            text_blocks: Original text blocks.
            table_context: Table dicts with ``page_number`` and
                ``header_row`` (JSON string or list).

        Returns:
            Enriched text block list (original + synthetic blocks).
        """
        import json as _json

        # Build page → table headers map
        table_headers_by_page: dict[int, list[str]] = {}
        for tbl in table_context:
            page = tbl.get("page_number", 0)
            header = tbl.get("header_row", "")
            if isinstance(header, str):
                try:
                    header = _json.loads(header)
                except (ValueError, TypeError):
                    header = [header]
            if header and page > 0:
                summary = " | ".join(str(h) for h in header)
                table_headers_by_page.setdefault(page, []).append(summary)

        if not table_headers_by_page:
            return text_blocks

        enriched = list(text_blocks)
        for page, headers in table_headers_by_page.items():
            for header_text in headers:
                enriched.append({
                    "page_number": page,
                    "text": f"[Table on this page: {header_text}]",
                    "block_type": "table_context",
                    "in_table": True,
                })

        return enriched

    # ---------------------------------------------------------------
    # Routing
    # ---------------------------------------------------------------

    def _route_section(
        self,
        section: IchSection,
        block_index: int,
        content_hash: str,
        candidates: list[LocalCandidate],
    ) -> RoutingDecision:
        """Decide and execute routing for one section."""
        threshold = get_cascade_threshold(section.section_code)

        if section.confidence_score >= threshold:
            return RoutingDecision(
                block_index=block_index,
                content_hash=content_hash,
                local_candidates=candidates,
                route="local",
                threshold_used=threshold,
                final_section_code=section.section_code,
                final_confidence=section.confidence_score,
            )

        # PTCV-234: Centroid classifier pre-filter before Sonnet
        if self._centroid_classifier is not None:
            try:
                from .centroid_classifier import CentroidClassifier

                matches = self._centroid_classifier.classify(
                    section.content_text, top_k=3,
                )
                if CentroidClassifier.is_high_confidence(matches):
                    top = matches[0]
                    logger.info(
                        "Centroid pre-filter: %s -> %s (%.2f), "
                        "skipping Sonnet",
                        section.section_code,
                        top.section_code,
                        top.confidence,
                    )
                    return RoutingDecision(
                        block_index=block_index,
                        content_hash=content_hash,
                        local_candidates=candidates,
                        route="centroid",
                        threshold_used=threshold,
                        final_section_code=top.section_code,
                        final_confidence=top.confidence,
                    )
            except Exception:
                logger.debug(
                    "Centroid classifier failed; falling through",
                    exc_info=True,
                )

        # Low confidence — route to Sonnet if available
        if not self._use_sonnet:
            logger.debug(
                "No API key — keeping local classification for "
                "section %s (confidence %.2f)",
                section.section_code,
                section.confidence_score,
            )
            return RoutingDecision(
                block_index=block_index,
                content_hash=content_hash,
                local_candidates=candidates,
                route="local",
                threshold_used=threshold,
                final_section_code=section.section_code,
                final_confidence=section.confidence_score,
            )

        judgement, rag_count = self._call_sonnet(
            section_text=section.content_text,
            local_candidates=candidates,
            registry_id=section.registry_id,
        )

        return RoutingDecision(
            block_index=block_index,
            content_hash=content_hash,
            local_candidates=candidates,
            route="sonnet",
            threshold_used=threshold,
            final_section_code=judgement.section_code,
            final_confidence=section.confidence_score,
            sonnet_judgement=judgement,
            rag_exemplar_count=rag_count,
        )

    # ---------------------------------------------------------------
    # Sonnet integration
    # ---------------------------------------------------------------

    def _get_sonnet_client(self) -> Any:
        """Lazy-initialise Anthropic client."""
        if self._client is not None:
            return self._client
        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._api_key,
            )
            logger.info(
                "ClassificationRouter: Anthropic client initialised "
                "(model=%s)",
                self._sonnet_model,
            )
        except Exception:
            logger.warning(
                "ClassificationRouter: Failed to init Anthropic "
                "client — falling back to local-only",
                exc_info=True,
            )
            self._use_sonnet = False
            raise
        return self._client

    def _call_sonnet(
        self,
        section_text: str,
        local_candidates: list[LocalCandidate],
        registry_id: str,
    ) -> tuple[SonnetJudgement, int]:
        """Call Sonnet for classification judgement on one section.

        Returns:
            Tuple of (SonnetJudgement, rag_exemplar_count).
        """
        # PTCV-162: Retrieve RAG exemplars if index available
        rag_exemplars: list = []
        if self._rag_index is not None:
            try:
                rag_exemplars = self._rag_index.query(
                    section_text, top_k=3,
                )
            except Exception:
                logger.debug(
                    "RAG query failed; proceeding without context",
                    exc_info=True,
                )

        prompt = self._build_sonnet_prompt(
            section_text, local_candidates, registry_id,
            rag_exemplars=rag_exemplars,
        )

        try:
            client = self._get_sonnet_client()
            response = client.messages.create(
                model=self._sonnet_model,
                max_tokens=200,
                system=_CASCADE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text.strip()
            input_tokens = getattr(
                response.usage, "input_tokens", 0
            )
            output_tokens = getattr(
                response.usage, "output_tokens", 0
            )

            return (
                self._parse_sonnet_response(
                    raw_text, input_tokens, output_tokens,
                    local_candidates,
                ),
                len(rag_exemplars),
            )
        except Exception:
            logger.warning(
                "Sonnet call failed — falling back to local",
                exc_info=True,
            )
            return (
                SonnetJudgement(
                    section_code=local_candidates[0].section_code
                    if local_candidates
                    else "B.1",
                    reasoning="Sonnet call failed; using local fallback",
                ),
                len(rag_exemplars),
            )

    def _build_sonnet_prompt(
        self,
        section_text: str,
        local_candidates: list[LocalCandidate],
        registry_id: str,
        rag_exemplars: Optional[list] = None,
    ) -> str:
        """Build the Sonnet classification prompt."""
        candidates_lines = []
        for i, c in enumerate(local_candidates, 1):
            candidates_lines.append(
                f"  {i}. {c.section_code} "
                f"(confidence: {c.confidence_score:.2f}) "
                f"— {c.section_name}"
            )
        candidates_block = "\n".join(
            candidates_lines
        ) or "  (no candidates)"

        # PTCV-162: Build RAG context block
        rag_context = "  (no prior examples available)"
        if rag_exemplars:
            rag_lines = []
            for i, ex in enumerate(rag_exemplars, 1):
                rag_lines.append(
                    f"  {i}. Section {ex.section_code} "
                    f"({ex.section_name}) from "
                    f"{ex.registry_id} "
                    f"(confidence: {ex.confidence_score:.2f}, "
                    f"similarity: {ex.similarity_score:.2f}):\n"
                    f'     "{ex.content_text[:300]}..."'
                )
            rag_context = "\n".join(rag_lines)

        trimmed_text = section_text[:_MAX_SECTION_CHARS]

        return _CASCADE_USER_PROMPT.format(
            candidates_block=candidates_block,
            rag_context=rag_context,
            section_defs=self._section_defs_text,
            max_chars=_MAX_SECTION_CHARS,
            section_text=trimmed_text,
        )

    def _parse_sonnet_response(
        self,
        raw_text: str,
        input_tokens: int,
        output_tokens: int,
        local_candidates: list[LocalCandidate],
    ) -> SonnetJudgement:
        """Parse Sonnet's JSON response into a SonnetJudgement."""
        try:
            data = json.loads(raw_text)
            section_code = data.get("section_code", "")
            reasoning = data.get("reasoning", "")

            # Validate section code
            if section_code not in _ICH_SECTIONS:
                logger.warning(
                    "Sonnet returned invalid section code: %s",
                    section_code,
                )
                section_code = (
                    local_candidates[0].section_code
                    if local_candidates
                    else "B.1"
                )
                reasoning = (
                    f"Invalid Sonnet response ({section_code}); "
                    f"using local fallback. "
                    f"Original: {reasoning}"
                )

            return SonnetJudgement(
                section_code=section_code,
                reasoning=reasoning,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Failed to parse Sonnet response: %s",
                raw_text[:200],
            )
            return SonnetJudgement(
                section_code=local_candidates[0].section_code
                if local_candidates
                else "B.1",
                reasoning=f"Parse error; local fallback. Raw: "
                f"{raw_text[:100]}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    # ---------------------------------------------------------------
    # JSONL output
    # ---------------------------------------------------------------

    def _write_judgements_jsonl(
        self,
        decisions: list[RoutingDecision],
        run_id: str,
        registry_id: str,
        source_sha256: str,
    ) -> tuple[str, str]:
        """Write Sonnet judgements to JSONL for future training.

        Returns:
            Tuple of ``(artifact_key, artifact_sha256)``.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        lines: list[str] = []

        for d in decisions:
            if d.sonnet_judgement is None:
                continue
            entry = {
                "run_id": run_id,
                "registry_id": registry_id,
                "timestamp_utc": timestamp,
                "section_code_local": (
                    d.local_candidates[0].section_code
                    if d.local_candidates
                    else ""
                ),
                "confidence_local": (
                    d.local_candidates[0].confidence_score
                    if d.local_candidates
                    else 0.0
                ),
                "section_code_sonnet": (
                    d.sonnet_judgement.section_code
                ),
                "reasoning": d.sonnet_judgement.reasoning,
                "content_hash": d.content_hash,
                "local_top3": [
                    {
                        "code": c.section_code,
                        "score": c.confidence_score,
                    }
                    for c in d.local_candidates
                ],
            }
            lines.append(json.dumps(entry, ensure_ascii=False))

        if not lines:
            return "", ""

        content = "\n".join(lines) + "\n"
        content_bytes = content.encode("utf-8")
        sha = hashlib.sha256(content_bytes).hexdigest()
        artifact_key = f"cascade/{run_id}/sonnet_judgements.jsonl"

        if self._gateway:
            self._gateway.put_artifact(
                key=artifact_key,
                data=content_bytes,
                content_type="application/jsonl",
                run_id=run_id,
                source_hash=source_sha256,
                user="ptcv-classification-router",
                stage="classification_cascade",
            )

        return artifact_key, sha

    # ---------------------------------------------------------------
    # Stats
    # ---------------------------------------------------------------

    def _compute_stats(
        self,
        decisions: list[RoutingDecision],
    ) -> RoutingStats:
        """Compute aggregate routing statistics."""
        total = len(decisions)
        local_count = sum(
            1 for d in decisions if d.route == "local"
        )
        sonnet_count = total - local_count

        agreements = 0
        disagreements = 0
        total_in = 0
        total_out = 0

        for d in decisions:
            if d.sonnet_judgement:
                total_in += d.sonnet_judgement.input_tokens
                total_out += d.sonnet_judgement.output_tokens
                local_top1 = (
                    d.local_candidates[0].section_code
                    if d.local_candidates
                    else ""
                )
                if d.sonnet_judgement.section_code == local_top1:
                    agreements += 1
                else:
                    disagreements += 1

        return RoutingStats(
            total_sections=total,
            local_count=local_count,
            sonnet_count=sonnet_count,
            local_pct=local_count / total if total else 0.0,
            sonnet_pct=sonnet_count / total if total else 0.0,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            agreements=agreements,
            disagreements=disagreements,
        )

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _build_section_defs() -> str:
        """Build compact ICH section definitions for prompts."""
        lines = []
        for code in sorted(
            _ICH_SECTIONS.keys(),
            key=lambda c: int(c.split(".")[1]),
        ):
            name = _ICH_SECTIONS[code].get("name", code)
            lines.append(f"  {code}: {name}")
        return "\n".join(lines)
