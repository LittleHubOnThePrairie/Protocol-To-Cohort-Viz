"""Retemplating fidelity checker — two-tier validation (PTCV-65).

Tier 1 (Deterministic): Sentence-level text overlap per section.
    Checks how much retemplated content exists in the original text.
    Fast, free, always runs.
Tier 2 (LLM Semantic): Section-by-section Claude Opus 4.6 comparison
    detecting hallucinations, omissions, and meaning drift. Optional,
    requires ANTHROPIC_API_KEY, costs ~$0.90-1.20 per protocol.

Risk tier: MEDIUM — data pipeline QA component (no patient data).

Regulatory references:
- ALCOA+ Accurate: fidelity_score quantifies retemplating quality
- ALCOA+ Traceable: source_sha256 links to retemplating artifact
- ALCOA+ Contemporaneous: timestamp_utc at check boundary
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .models import IchSection

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-opus-4-6"
_FIDELITY_THRESHOLD = 0.85

# Cost model: Claude Opus 4.6 pricing per 1M tokens
_INPUT_COST_PER_M = 15.00
_OUTPUT_COST_PER_M = 75.00


@dataclasses.dataclass
class FidelityResult:
    """Aggregate fidelity check result for a complete protocol.

    Attributes:
        run_id: UUID4 for this fidelity check run.
        registry_id: Trial registry identifier.
        section_results: Per-section fidelity results.
        fidelity_passed: True if ALL sections >= pass_threshold.
        pass_threshold: Threshold used (default 0.85).
        overall_score: Mean fidelity score across all sections.
        total_hallucinations: Count of all hallucination flags.
        total_omissions: Count of all omission flags.
        total_drift_flags: Count of all drift flags.
        input_tokens: Total input tokens across all LLM calls.
        output_tokens: Total output tokens across all LLM calls.
        estimated_cost_usd: Estimated API cost in USD.
        method: "llm", "deterministic", or "hybrid".
        source_sha256: SHA-256 of retemplating artifact.
        timestamp_utc: ISO 8601 UTC timestamp of the check.
    """

    run_id: str
    registry_id: str
    section_results: list["SectionFidelity"]
    fidelity_passed: bool
    pass_threshold: float
    overall_score: float
    total_hallucinations: int
    total_omissions: int
    total_drift_flags: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    method: str
    source_sha256: str
    timestamp_utc: str


@dataclasses.dataclass
class SectionFidelity:
    """Fidelity check result for one ICH section.

    Attributes:
        section_code: ICH section code (e.g. "B.3").
        fidelity_score: 0.0-1.0 semantic fidelity score.
        hallucinations: Content in retemplated but absent from
            original (fabricated content).
        omissions: Content in original but missing from retemplated.
        drift_flags: Meaning changes between original and retemplated.
        original_char_count: Character count of original text.
        retemplated_char_count: Character count of retemplated text.
        input_tokens: Input tokens for this section's LLM call.
        output_tokens: Output tokens for this section's LLM call.
        method: "llm" or "deterministic".
    """

    section_code: str
    fidelity_score: float
    hallucinations: list[str]
    omissions: list[str]
    drift_flags: list[str]
    original_char_count: int
    retemplated_char_count: int
    input_tokens: int = 0
    output_tokens: int = 0
    method: str = "deterministic"


class FidelityChecker:
    """Two-tier retemplating fidelity validator (PTCV-65).

    Tier 1 runs deterministic sentence-level overlap per section,
    measuring how much retemplated content exists in the original.

    Tier 2 (optional) calls Claude Opus 4.6 per section to detect
    hallucinations, omissions, and meaning drift.

    Args:
        pass_threshold: Minimum per-section fidelity score
            (default 0.85).
        claude_model: Anthropic model ID for Tier 2.
        enable_llm: If True, run Tier 2 LLM check. If False
            or ANTHROPIC_API_KEY not set, deterministic only.
    [PTCV-65 Scenario: Faithful retemplating passes fidelity check]
    [PTCV-65 Scenario: Checker runs without API key using deterministic mode only]
    """

    def __init__(
        self,
        pass_threshold: float = _FIDELITY_THRESHOLD,
        claude_model: str = _DEFAULT_MODEL,
        enable_llm: bool = True,
    ) -> None:
        self._threshold = pass_threshold
        self._claude_model = claude_model
        self._enable_llm = enable_llm
        self._claude: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize Anthropic client."""
        if self._claude is None:
            import anthropic

            self._claude = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
        return self._claude

    @property
    def _has_api_key(self) -> bool:
        """Check if ANTHROPIC_API_KEY is available."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_cost(
        sections: list["IchSection"],
        original_text_blocks: list[dict],
    ) -> tuple[float, int, int]:
        """Estimate LLM fidelity check cost before running.

        Args:
            sections: Retemplated IchSection list.
            original_text_blocks: Original text block dicts.

        Returns:
            Tuple of (estimated_cost_usd, est_input_tokens,
            est_output_tokens).
        [PTCV-65 Scenario: Cost warning displayed before running checker]
        """
        total_original_chars = sum(
            len(b.get("text", "")) for b in original_text_blocks
        )
        total_retemplated_chars = sum(
            len(s.content_text or "") for s in sections
        )
        n_sections = max(len(sections), 1)
        # ~4 chars per token, plus system prompt overhead per section
        est_input_tokens = (
            (total_original_chars + total_retemplated_chars) // 4
            + 500 * n_sections
        )
        # ~500 output tokens per section (JSON response)
        est_output_tokens = 500 * n_sections

        est_cost = (
            est_input_tokens * _INPUT_COST_PER_M / 1_000_000
            + est_output_tokens * _OUTPUT_COST_PER_M / 1_000_000
        )
        return round(est_cost, 2), est_input_tokens, est_output_tokens

    # ------------------------------------------------------------------
    # Main check method
    # ------------------------------------------------------------------

    def check(
        self,
        original_text_blocks: list[dict],
        retemplated_sections: list["IchSection"],
        registry_id: str,
        source_sha256: str = "",
    ) -> FidelityResult:
        """Run fidelity check on retemplated protocol.

        Tier 1 always runs (deterministic). Tier 2 runs only
        when enable_llm=True AND ANTHROPIC_API_KEY is set.

        Args:
            original_text_blocks: List of dicts with 'page_number'
                and 'text' keys from extraction stage.
            retemplated_sections: IchSection list from retemplating.
            registry_id: Trial registry identifier.
            source_sha256: SHA-256 of retemplating artifact.

        Returns:
            FidelityResult with per-section and aggregate scores.
        [PTCV-65 Scenario: Large protocol checked iteratively]
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        use_llm = self._enable_llm and self._has_api_key
        if self._enable_llm and not self._has_api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — running "
                "deterministic-only fidelity check"
            )

        # Build original text corpus
        original_corpus = "\n".join(
            b.get("text", "") for b in original_text_blocks
        )

        section_results: list[SectionFidelity] = []
        total_in = 0
        total_out = 0

        for section in retemplated_sections:
            # Tier 1: Deterministic overlap
            det_score = self._deterministic_section_score(
                section, original_text_blocks,
            )

            if use_llm:
                # Tier 2: LLM semantic check
                llm_result, in_tok, out_tok = (
                    self._llm_section_check(
                        section=section,
                        original_corpus=original_corpus,
                    )
                )
                total_in += in_tok
                total_out += out_tok

                # Combined: 0.3 deterministic + 0.7 LLM
                combined_score = round(
                    0.3 * det_score + 0.7 * llm_result["score"],
                    4,
                )

                section_results.append(SectionFidelity(
                    section_code=section.section_code,
                    fidelity_score=combined_score,
                    hallucinations=llm_result["hallucinations"],
                    omissions=llm_result["omissions"],
                    drift_flags=llm_result["drift_flags"],
                    original_char_count=len(original_corpus),
                    retemplated_char_count=len(
                        section.content_text or "",
                    ),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    method="llm",
                ))
            else:
                section_results.append(SectionFidelity(
                    section_code=section.section_code,
                    fidelity_score=det_score,
                    hallucinations=[],
                    omissions=[],
                    drift_flags=[],
                    original_char_count=len(original_corpus),
                    retemplated_char_count=len(
                        section.content_text or "",
                    ),
                    method="deterministic",
                ))

        # Aggregate
        scores = [r.fidelity_score for r in section_results]
        overall = (
            round(sum(scores) / len(scores), 4) if scores else 1.0
        )
        all_passed = all(
            s.fidelity_score >= self._threshold
            for s in section_results
        ) if section_results else True

        est_cost, _, _ = self.estimate_cost(
            retemplated_sections, original_text_blocks,
        )

        method = "llm" if use_llm else "deterministic"

        return FidelityResult(
            run_id=run_id,
            registry_id=registry_id,
            section_results=section_results,
            fidelity_passed=all_passed,
            pass_threshold=self._threshold,
            overall_score=overall,
            total_hallucinations=sum(
                len(r.hallucinations) for r in section_results
            ),
            total_omissions=sum(
                len(r.omissions) for r in section_results
            ),
            total_drift_flags=sum(
                len(r.drift_flags) for r in section_results
            ),
            input_tokens=total_in,
            output_tokens=total_out,
            estimated_cost_usd=est_cost,
            method=method,
            source_sha256=source_sha256,
            timestamp_utc=timestamp,
        )

    # ------------------------------------------------------------------
    # Tier 1: Deterministic overlap
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse whitespace and lowercase for comparison."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _deterministic_section_score(
        self,
        section: "IchSection",
        original_text_blocks: list[dict],
    ) -> float:
        """Compute deterministic fidelity score for one section.

        Checks what fraction of the retemplated section text can
        be found in the original text. A score of 1.0 means all
        retemplated content comes from the original (no hallucination).

        Returns:
            Float 0.0-1.0 fidelity score.
        [PTCV-65 Scenario: Hallucination detected in retemplated section]
        """
        retemplated_text = section.content_text or ""
        if not retemplated_text.strip():
            return 1.0  # Empty section = nothing to hallucinate

        original_corpus = " ".join(
            b.get("text", "") for b in original_text_blocks
        )
        normalized_original = self._normalize(original_corpus)

        # Check retemplated sentences against original
        sentences = re.split(r"(?<=[.!?])\s+|\n+", retemplated_text)
        found_chars = 0
        total_chars = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            total_chars += len(sent)
            check = self._normalize(sent)[:60]
            if check in normalized_original:
                found_chars += len(sent)

        if total_chars == 0:
            return 1.0

        return round(found_chars / total_chars, 4)

    # ------------------------------------------------------------------
    # Tier 2: LLM semantic check
    # ------------------------------------------------------------------

    def _llm_section_check(
        self,
        section: "IchSection",
        original_corpus: str,
    ) -> tuple[dict, int, int]:
        """LLM semantic fidelity check for one section.

        Calls Claude Opus 4.6 to compare the retemplated section
        against the original text, identifying hallucinations,
        omissions, and meaning drift.

        Args:
            section: Retemplated IchSection.
            original_corpus: Full original text for context.

        Returns:
            Tuple of (result_dict, input_tokens, output_tokens).
            result_dict keys: score, hallucinations, omissions,
            drift_flags.
        [PTCV-65 Scenario: Content omission detected]
        """
        retemplated = section.content_text or ""
        if not retemplated.strip():
            return {
                "score": 1.0,
                "hallucinations": [],
                "omissions": [],
                "drift_flags": [],
            }, 0, 0

        prompt = self._build_fidelity_prompt(
            section_code=section.section_code,
            section_name=section.section_name,
            retemplated_text=retemplated,
            original_text=original_corpus,
        )

        client = self._get_client()
        resp = client.messages.create(
            model=self._claude_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        in_tok = getattr(resp.usage, "input_tokens", 0)
        out_tok = getattr(resp.usage, "output_tokens", 0)

        first_block = resp.content[0]
        if not hasattr(first_block, "text"):
            return {
                "score": 0.5,
                "hallucinations": [],
                "omissions": [],
                "drift_flags": [],
            }, in_tok, out_tok

        raw: str = first_block.text.strip()
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "Fidelity LLM returned unparseable JSON for %s",
                section.section_code,
            )
            return {
                "score": 0.5,
                "hallucinations": [],
                "omissions": [],
                "drift_flags": [],
            }, in_tok, out_tok

        return {
            "score": float(parsed.get("fidelity_score", 0.5)),
            "hallucinations": parsed.get("hallucinations", []),
            "omissions": parsed.get("omissions", []),
            "drift_flags": parsed.get("drift_flags", []),
        }, in_tok, out_tok

    @staticmethod
    def _build_fidelity_prompt(
        section_code: str,
        section_name: str,
        retemplated_text: str,
        original_text: str,
    ) -> str:
        """Build the section-level fidelity check prompt.

        Args:
            section_code: ICH section code (e.g. "B.5").
            section_name: Human-readable section name.
            retemplated_text: Full text of the retemplated section.
            original_text: Full original protocol text.

        Returns:
            Prompt string for Claude.
        """
        max_original = 30_000
        original_truncated = original_text[:max_original]
        if len(original_text) > max_original:
            original_truncated += "\n[... truncated ...]"

        max_retemplated = 15_000
        retemplated_truncated = retemplated_text[:max_retemplated]
        if len(retemplated_text) > max_retemplated:
            retemplated_truncated += "\n[... truncated ...]"

        return (
            "You are a GCP regulatory expert performing a fidelity "
            "audit of a retemplated clinical trial protocol.\n\n"
            f"ICH section: {section_code} ({section_name})\n\n"
            "Compare the RETEMPLATED section text against the "
            "ORIGINAL protocol text below. Verify that the "
            "retemplated section faithfully represents the "
            "original content.\n\n"
            "Check for:\n"
            "1. HALLUCINATIONS: Content in the retemplated "
            "section that is NOT present in the original "
            "protocol (fabricated details, invented endpoints, "
            "made-up criteria)\n"
            "2. OMISSIONS: Substantive content from the original "
            "protocol that should appear in this section but is "
            "missing from the retemplated version (dropped "
            "criteria, missing endpoints, lost procedures)\n"
            "3. DRIFT: Content where the meaning has changed "
            "between original and retemplated (dosage changed, "
            "criteria altered, different study population)\n\n"
            f"<original_text>\n{original_truncated}\n"
            "</original_text>\n\n"
            f"<retemplated_section>\n{retemplated_truncated}\n"
            "</retemplated_section>\n\n"
            "Respond with a JSON object (no markdown fences, "
            "no commentary outside JSON):\n"
            "{\n"
            '  "fidelity_score": float 0.0-1.0 (1.0 = perfect '
            "fidelity, 0.0 = completely unfaithful),\n"
            '  "hallucinations": [list of strings describing '
            "each fabricated item, or empty list],\n"
            '  "omissions": [list of strings describing each '
            "missing item, or empty list],\n"
            '  "drift_flags": [list of strings describing each '
            "meaning change, or empty list]\n"
            "}\n\n"
            "Guidelines:\n"
            "- Score 0.95-1.0: Faithful, no issues\n"
            "- Score 0.85-0.94: Minor formatting differences "
            "but content preserved\n"
            "- Score 0.70-0.84: Some content gaps or additions\n"
            "- Score below 0.70: Significant fidelity problems\n"
            "- If the retemplated section is empty, score 0.0 "
            "and list all original content as omissions\n"
            "- Ignore differences in formatting, whitespace, "
            "and boilerplate headers\n"
            "- Focus on clinical/scientific content accuracy"
        )
