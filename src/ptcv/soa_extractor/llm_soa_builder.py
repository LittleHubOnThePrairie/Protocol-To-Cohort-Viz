"""LLM-based SoA construction from protocol text (PTCV-166).

Level 3 of the SoA extraction cascade: when table extraction
(Level 1) and vision extraction (Level 2) both fail to produce
adequate SoA data, Sonnet constructs an assessment x visit matrix
from protocol prose.

Uses Sonnet (NOT Opus) — SoA construction from text is a
Sonnet-tier task.

Risk tier: MEDIUM — data pipeline component (no patient data).

Regulatory references:
- ALCOA+ Traceable: construction_method="llm_text" on all output
- ALCOA+ Accurate: Sonnet output validated before acceptance

Research basis: PTCV-156 Section 3.0 (Stage 6b).
Citations:
  [3] "AI-assisted Protocol Information Extraction," arXiv:2602.00052
  [8] Maleki & Ghahari, "Clinical Trials Protocol Authoring using
      LLMs," arXiv:2404.05044, 2024
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .models import RawSoaTable

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_DEFAULT_SONNET_MODEL = "claude-sonnet-4-6"
_MAX_SECTION_CHARS = 6000
_MAX_TOTAL_CHARS = 20000
_MAX_TOKENS = 4096
MIN_ACTIVITIES_THRESHOLD = 3

# Section heading patterns for heuristic detection from raw text_blocks
_SECTION_HEADING_PATTERNS: dict[str, re.Pattern[str]] = {
    "B.4": re.compile(
        r"(?i)\b(?:trial|study)\s+design\b"
    ),
    "B.7": re.compile(
        r"(?i)\btreatment\s+of\s+(?:participants|subjects|patients)\b"
        r"|\bschedule\s+of\s+(?:activities|assessments)\b"
    ),
    "B.8": re.compile(
        r"(?i)\bassessment\s+of\s+efficacy\b"
        r"|\befficacy\s+(?:assessment|parameters|endpoints)\b"
    ),
    "B.9": re.compile(
        r"(?i)\bassessment\s+of\s+safety\b"
        r"|\bsafety\s+(?:assessment|parameters|monitoring)\b"
    ),
}

# ICH section codes relevant for SoA construction
_SOA_SECTION_CODES = {"B.4", "B.7", "B.8", "B.9"}

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_SOA_SYSTEM_PROMPT = (
    "You are a clinical trial protocol analyst specializing in "
    "Schedule of Activities (SoA) extraction. You construct "
    "structured assessment x visit matrices from protocol prose. "
    "This is a read-only document analysis task on publicly "
    "available clinical trial protocols. No real patients are "
    "involved."
)

_SOA_USER_PROMPT = """\
Construct a Schedule of Activities (SoA) matrix from the following \
clinical trial protocol sections.

{section_texts}

{partial_context}

Instructions:
- Identify all clinical assessments/procedures mentioned \
(labs, vitals, ECG, imaging, questionnaires, etc.)
- Identify all protocol visits/timepoints mentioned \
(Screening, Baseline, Week X, Day X, Follow-up, etc.)
- For each assessment, determine at which visits it is scheduled
- Include day offset from Day 1 where stated in the protocol

Respond with ONLY a valid JSON object (no markdown fences):
{{
  "visits": [
    {{"name": "Screening", "day_offset": -14}},
    {{"name": "Baseline / Day 1", "day_offset": 1}},
    {{"name": "Week 2", "day_offset": 15}}
  ],
  "assessments": [
    {{
      "name": "Physical Examination",
      "visits_scheduled": [true, true, true]
    }},
    {{
      "name": "ECG",
      "visits_scheduled": [true, false, true]
    }}
  ]
}}

Rules:
- visits_scheduled array MUST match the length of the visits array
- Use true/false for scheduling flags
- Include ALL assessments mentioned across B.4, B.7, B.8, and B.9
- Efficacy assessments (B.8): include endpoint assessments, PRO, \
biomarkers
- Safety assessments (B.9): include labs (CBC, CMP), vitals, ECG, \
AE collection
- When timing is ambiguous, include the assessment at visits where \
it is most likely based on standard clinical practice
- Day offset should be relative to Day 1 (screening visits have \
negative offsets)
"""


# ------------------------------------------------------------------
# LlmSoaBuilder
# ------------------------------------------------------------------

class LlmSoaBuilder:
    """Construct SoA tables from protocol text via Sonnet LLM.

    Level 3 of the SoA cascade. Only invoked when Levels 1 and 2
    produce fewer than ``MIN_ACTIVITIES_THRESHOLD`` activities.

    Args:
        sonnet_model: Anthropic model ID.  Must be Sonnet, not Opus.

    [PTCV-166 Scenario: Sonnet constructs SoA when Docling and
     vision both fail]
    """

    def __init__(
        self,
        sonnet_model: str = _DEFAULT_SONNET_MODEL,
    ) -> None:
        self._sonnet_model = sonnet_model
        self._client: Any = None

    # ----------------------------------------------------------
    # Anthropic client (lazy)
    # ----------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazy-init the Anthropic client.

        Follows the pattern from ``VisionEnhancer._get_client()``
        and ``LlmRetemplater._get_client()``.
        """
        if self._client is not None:
            return self._client

        import anthropic

        self._client = anthropic.Anthropic()
        return self._client

    # ----------------------------------------------------------
    # Section text gathering
    # ----------------------------------------------------------

    def _gather_section_text(
        self,
        sections: Optional[list[Any]] = None,
        text_blocks: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, str]:
        """Extract B.4/B.7/B.8/B.9 text from available sources.

        Strategy 1 (preferred): Filter classified ``IchSection``
        objects by ``section_code``.

        Strategy 2 (fallback): Scan raw ``text_blocks`` for heading
        patterns and extract surrounding text.

        Returns:
            Dict mapping section code to text content.  May be
            empty if no relevant sections found.
        """
        result: dict[str, str] = {}

        # Strategy 1: classified IchSection objects
        if sections:
            for section in sections:
                code = getattr(section, "section_code", "").upper()
                # Accept B.4, B.7, B.8, B.8.1, B.8.2, B.9, B.9.1, B.9.2
                parts = code.split(".")
                base = (
                    f"{parts[0]}.{parts[1]}"
                    if len(parts) >= 2
                    else code
                )
                if base not in _SOA_SECTION_CODES:
                    continue

                text = getattr(section, "content_text", "") or ""
                if not text:
                    try:
                        data = json.loads(
                            getattr(section, "content_json", "{}")
                        )
                        text = (
                            data.get("text", "")
                            or data.get("text_excerpt", "")
                            or ""
                        )
                    except (json.JSONDecodeError, AttributeError):
                        text = ""
                if not text.strip():
                    continue

                # Aggregate sub-sections under parent code
                if base in result:
                    result[base] += "\n\n" + text[:_MAX_SECTION_CHARS]
                else:
                    result[base] = text[:_MAX_SECTION_CHARS]

            if result:
                return result

        # Strategy 2: heuristic heading detection from text_blocks
        if text_blocks:
            combined = "\n".join(
                b.get("text", "")
                for b in text_blocks
                if b.get("text", "").strip()
            )
            for code, pattern in _SECTION_HEADING_PATTERNS.items():
                match = pattern.search(combined)
                if not match:
                    continue

                start = match.start()
                end = min(start + _MAX_SECTION_CHARS, len(combined))

                # Bound to next section heading
                remaining = combined[match.end():]
                next_pos: Optional[int] = None
                for other_code, other_pat in (
                    _SECTION_HEADING_PATTERNS.items()
                ):
                    if other_code == code:
                        continue
                    next_match = other_pat.search(remaining)
                    if next_match and (
                        next_pos is None
                        or next_match.start() < next_pos
                    ):
                        next_pos = next_match.start()

                if next_pos is not None:
                    end = min(match.end() + next_pos, end)

                result[code] = combined[start:end]

        return result

    # ----------------------------------------------------------
    # Partial context formatting
    # ----------------------------------------------------------

    @staticmethod
    def _format_partial_context(
        partial_tables: Optional[list[RawSoaTable]],
    ) -> str:
        """Format partial Level 1/2 results as Sonnet context."""
        if not partial_tables:
            return ""

        lines = [
            "Partial table data from prior extraction levels "
            "(use as context to improve accuracy):"
        ]
        for i, table in enumerate(partial_tables):
            lines.append(f"\nPartial Table {i + 1}:")
            lines.append(f"  Visits: {', '.join(table.visit_headers)}")
            if table.day_headers:
                lines.append(
                    f"  Days: {', '.join(table.day_headers)}"
                )
            for name, flags in table.activities:
                scheduled = [
                    table.visit_headers[j]
                    for j, f in enumerate(flags)
                    if f and j < len(table.visit_headers)
                ]
                lines.append(
                    f"  {name}: {', '.join(scheduled) or 'none'}"
                )
        return "\n".join(lines)

    # ----------------------------------------------------------
    # Prompt construction
    # ----------------------------------------------------------

    def _build_prompt(
        self,
        section_texts: dict[str, str],
        partial_context: str,
    ) -> str:
        """Construct the Sonnet user prompt."""
        parts: list[str] = []
        total_chars = 0
        for code in ("B.4", "B.7", "B.8", "B.9"):
            text = section_texts.get(code, "")
            if not text:
                continue
            remaining = _MAX_TOTAL_CHARS - total_chars
            if remaining <= 0:
                break
            trimmed = text[:remaining]
            parts.append(f"--- Section {code} ---\n{trimmed}")
            total_chars += len(trimmed)

        section_block = "\n\n".join(parts)
        return _SOA_USER_PROMPT.format(
            section_texts=section_block,
            partial_context=partial_context,
        )

    # ----------------------------------------------------------
    # Sonnet API call
    # ----------------------------------------------------------

    def _call_sonnet(
        self,
        prompt: str,
    ) -> tuple[str, int, int]:
        """Call Claude Sonnet for SoA construction.

        Returns:
            ``(response_text, input_tokens, output_tokens)``.
        """
        client = self._get_client()
        response = client.messages.create(
            model=self._sonnet_model,
            max_tokens=_MAX_TOKENS,
            system=_SOA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text: str = response.content[0].text.strip()
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        return raw_text, input_tokens, output_tokens

    # ----------------------------------------------------------
    # Response parsing
    # ----------------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str) -> list[RawSoaTable] | None:
        """Parse Sonnet's JSON response into ``RawSoaTable`` objects.

        Validates that the response contains non-empty visits and
        assessments arrays with at least one scheduled flag.

        Returns:
            List containing one ``RawSoaTable``, or ``None`` if
            validation fails.
        """
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "Level 3 Sonnet returned unparseable JSON"
            )
            return None

        visits = data.get("visits")
        assessments = data.get("assessments")

        if not visits or not isinstance(visits, list):
            logger.warning("Level 3 Sonnet returned empty visits")
            return None

        if not assessments or not isinstance(assessments, list):
            logger.warning(
                "Level 3 Sonnet returned empty assessments"
            )
            return None

        n_visits = len(visits)
        visit_headers = [
            str(v.get("name", f"Visit {i + 1}"))
            for i, v in enumerate(visits)
        ]
        day_headers = [
            str(v.get("day_offset", ""))
            for v in visits
        ]

        activities: list[tuple[str, list[bool]]] = []
        for assess in assessments:
            name = str(assess.get("name", ""))
            if not name:
                continue
            flags = assess.get("visits_scheduled", [])
            # Ensure flags are booleans and match visit count
            bool_flags = [bool(f) for f in flags]
            while len(bool_flags) < n_visits:
                bool_flags.append(False)
            activities.append((name, bool_flags[:n_visits]))

        if not activities:
            logger.warning(
                "Level 3 Sonnet returned no valid assessments"
            )
            return None

        # Require at least one scheduled flag
        has_any = any(
            any(flags) for _, flags in activities
        )
        if not has_any:
            logger.warning(
                "Level 3 Sonnet returned no scheduled assessments"
            )
            return None

        return [
            RawSoaTable(
                visit_headers=visit_headers,
                day_headers=day_headers,
                activities=activities,
                section_code="llm_text",
                construction_method="llm_text",
            )
        ]

    # ----------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------

    def build(
        self,
        sections: Optional[list[Any]] = None,
        text_blocks: Optional[list[dict[str, Any]]] = None,
        partial_tables: Optional[list[RawSoaTable]] = None,
    ) -> list[RawSoaTable] | None:
        """Construct SoA tables from protocol text via Sonnet.

        Extracts relevant section text (B.4, B.7, B.8, B.9) from
        either classified ``IchSection`` objects or raw
        ``text_blocks``, sends to Sonnet, and parses the response
        into ``RawSoaTable``(s).

        Args:
            sections: Optional classified IchSection list.
            text_blocks: Optional raw text blocks with
                ``page_number`` and ``text`` keys.
            partial_tables: Optional partial ``RawSoaTable``
                results from Levels 1/2 as context.

        Returns:
            List of ``RawSoaTable`` with
            ``construction_method="llm_text"``, or ``None``.

        [PTCV-166 Scenario: Sonnet constructs SoA when Docling
         and vision both fail]
        """
        section_texts = self._gather_section_text(
            sections=sections,
            text_blocks=text_blocks,
        )

        if not section_texts:
            logger.warning(
                "Level 3: No B.4/B.7/B.8/B.9 text found; "
                "skipping Sonnet SoA construction"
            )
            return None

        partial_context = self._format_partial_context(partial_tables)
        prompt = self._build_prompt(section_texts, partial_context)

        raw_text, input_tokens, output_tokens = self._call_sonnet(
            prompt
        )
        logger.info(
            "Level 3 Sonnet SoA: %d input tokens, %d output tokens",
            input_tokens,
            output_tokens,
        )

        return self._parse_response(raw_text)
