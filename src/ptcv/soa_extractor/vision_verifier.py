"""Selective Vision API verification for SoA tables (PTCV-217).

Instead of using Vision as a full-fallback extraction, sends SoA
page images alongside the Level 1 extracted table and asks for
diff-based corrections. Reduces token cost by ~70% vs full extraction.

Only triggered when cross-validation (PTCV-216) detects issues —
not on every extraction.

Risk tier: MEDIUM — LLM API call, no patient data.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import RawSoaTable

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CellCorrection:
    """A single cell correction from Vision verification.

    Attributes:
        activity_name: Assessment/activity name for the row.
        visit_index: 0-based index of the visit column.
        visit_name: Name of the visit column (for clarity).
        old_value: Value from Level 1 extraction (True/False).
        new_value: Corrected value from Vision (True/False).
    """

    activity_name: str
    visit_index: int
    visit_name: str = ""
    old_value: bool = False
    new_value: bool = False


@dataclass
class VerificationResult:
    """Result of Vision API verification.

    Attributes:
        verified: Whether verification was performed.
        corrections: List of cell-level corrections.
        added_activities: Activities found by Vision but missing from Level 1.
        removed_activities: Activities in Level 1 that Vision couldn't confirm.
        corrected_table: The merged/corrected RawSoaTable.
        confidence_map: Per-activity confidence (0.0–1.0).
        token_cost: Estimated token cost of the verification call.
        skip_reason: Why verification was skipped (if not performed).
    """

    verified: bool = False
    corrections: list[CellCorrection] = field(default_factory=list)
    added_activities: list[str] = field(default_factory=list)
    removed_activities: list[str] = field(default_factory=list)
    corrected_table: Optional[RawSoaTable] = None
    confidence_map: dict[str, float] = field(default_factory=dict)
    token_cost: int = 0
    skip_reason: str = ""

    @property
    def correction_count(self) -> int:
        """Total corrections applied."""
        return len(self.corrections) + len(self.added_activities)

    @property
    def has_corrections(self) -> bool:
        """Whether any corrections were made."""
        return bool(
            self.corrections
            or self.added_activities
            or self.removed_activities
        )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_VERIFICATION_PROMPT = """\
You are verifying an extracted Schedule of Activities (SoA) table \
against the source PDF page image.

The Level 1 extraction produced this table:

VISIT COLUMNS: {visit_headers}

EXTRACTED ASSESSMENTS:
{activities_text}

Compare this against what you see in the page image. Report:
1. MISSING assessments: rows visible in the image but not in the \
extracted table
2. EXTRA assessments: rows in the extracted table that are NOT visible \
in the image (false positives)
3. CELL CORRECTIONS: specific cells where the X/blank value is wrong

Respond with ONLY a JSON object (no markdown fences):
{{
  "missing_activities": [
    {{"name": "Assessment Name", "visits_scheduled": [true, false, ...]}}
  ],
  "extra_activities": ["Name1", "Name2"],
  "cell_corrections": [
    {{
      "activity_name": "Assessment Name",
      "visit_index": 2,
      "old_value": false,
      "new_value": true
    }}
  ]
}}

Rules:
- visits_scheduled arrays MUST have {n_visits} elements (matching visit \
column count)
- Only report genuine differences — do not restate correct data
- If the extracted table is accurate, return empty arrays for all fields
"""


# ---------------------------------------------------------------------------
# VisionVerifier
# ---------------------------------------------------------------------------


class VisionVerifier:
    """Selective Vision-based verification for SoA tables.

    Sends page images alongside Level 1 results and asks for
    diff-based corrections rather than full re-extraction.

    Args:
        model: Anthropic model ID (Sonnet recommended).
        min_coverage_for_skip: If cross-validation coverage is above
            this threshold, skip verification entirely.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        min_coverage_for_skip: float = 0.95,
    ) -> None:
        self._model = model
        self._min_coverage_for_skip = min_coverage_for_skip
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the Anthropic client."""
        if self._client is not None:
            return self._client
        import anthropic
        self._client = anthropic.Anthropic()
        return self._client

    def verify(
        self,
        table: RawSoaTable,
        page_images: list[bytes],
        coverage_ratio: float = 0.0,
    ) -> VerificationResult:
        """Verify a Level 1 table against page images.

        If ``coverage_ratio`` >= ``min_coverage_for_skip``, the
        table is considered good enough and verification is skipped.

        Args:
            table: Level 1 extracted SoA table.
            page_images: PNG/JPEG bytes of the SoA pages.
            coverage_ratio: Cross-validation coverage ratio from
                PTCV-216 (0.0–1.0).

        Returns:
            VerificationResult with corrections or skip reason.
        """
        if coverage_ratio >= self._min_coverage_for_skip:
            return VerificationResult(
                verified=False,
                corrected_table=table,
                skip_reason=(
                    f"Coverage {coverage_ratio:.0%} >= "
                    f"{self._min_coverage_for_skip:.0%} threshold"
                ),
            )

        if not page_images:
            return VerificationResult(
                verified=False,
                corrected_table=table,
                skip_reason="No page images provided",
            )

        # Build the verification prompt
        activities_text = "\n".join(
            f"  {name}: [{', '.join('X' if s else '.' for s in flags)}]"
            for name, flags in table.activities
        )

        prompt = _VERIFICATION_PROMPT.format(
            visit_headers=", ".join(table.visit_headers),
            activities_text=activities_text,
            n_visits=len(table.visit_headers),
        )

        # Build message content with images
        content: list[dict[str, Any]] = []
        for img_bytes in page_images:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": content}],
            )

            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text = block.text

            token_cost = (
                response.usage.input_tokens + response.usage.output_tokens
            )

            return self._apply_corrections(
                table, response_text, token_cost,
            )

        except Exception as exc:
            logger.warning(
                "Vision verification failed: %s", exc,
            )
            return VerificationResult(
                verified=False,
                corrected_table=table,
                skip_reason=f"API error: {exc}",
            )

    def _apply_corrections(
        self,
        original: RawSoaTable,
        response_text: str,
        token_cost: int,
    ) -> VerificationResult:
        """Parse Vision response and apply corrections to the table.

        Args:
            original: Level 1 extracted table.
            response_text: JSON response from Vision API.
            token_cost: Token cost of the API call.

        Returns:
            VerificationResult with corrected table.
        """
        try:
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            data = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Could not parse Vision verification response")
            return VerificationResult(
                verified=True,
                corrected_table=original,
                token_cost=token_cost,
                skip_reason="Response parse error",
            )

        n_visits = len(original.visit_headers)
        corrections: list[CellCorrection] = []
        added: list[str] = []
        removed: list[str] = []

        # Build mutable copy of activities
        activity_map: dict[str, list[bool]] = {
            name: list(flags) for name, flags in original.activities
        }

        # Apply cell corrections
        for corr in data.get("cell_corrections", []):
            name = corr.get("activity_name", "")
            idx = corr.get("visit_index", -1)
            new_val = corr.get("new_value", False)
            old_val = corr.get("old_value", True)

            if name in activity_map and 0 <= idx < n_visits:
                activity_map[name][idx] = bool(new_val)
                visit_name = (
                    original.visit_headers[idx] if idx < n_visits else ""
                )
                corrections.append(CellCorrection(
                    activity_name=name,
                    visit_index=idx,
                    visit_name=visit_name,
                    old_value=bool(old_val),
                    new_value=bool(new_val),
                ))

        # Add missing activities
        for missing in data.get("missing_activities", []):
            name = missing.get("name", "")
            if not name or name in activity_map:
                continue
            flags = missing.get("visits_scheduled", [])
            padded = [bool(f) for f in flags[:n_visits]]
            while len(padded) < n_visits:
                padded.append(False)
            activity_map[name] = padded
            added.append(name)

        # Mark extra activities
        for extra_name in data.get("extra_activities", []):
            if extra_name in activity_map:
                removed.append(extra_name)
                # Don't actually remove — flag for review instead

        # Rebuild table
        corrected_activities = [
            (name, flags) for name, flags in activity_map.items()
        ]

        corrected_table = RawSoaTable(
            visit_headers=original.visit_headers,
            day_headers=original.day_headers,
            activities=corrected_activities,
            section_code=original.section_code,
            construction_method="llm_vision",
        )

        # Compute per-activity confidence
        confidence_map: dict[str, float] = {}
        for name, _ in corrected_activities:
            if name in added:
                confidence_map[name] = 0.85  # Vision-only
            elif name in removed:
                confidence_map[name] = 0.50  # Disputed
            elif any(c.activity_name == name for c in corrections):
                confidence_map[name] = 0.85  # Corrected
            else:
                confidence_map[name] = 0.95  # Agreed

        return VerificationResult(
            verified=True,
            corrections=corrections,
            added_activities=added,
            removed_activities=removed,
            corrected_table=corrected_table,
            confidence_map=confidence_map,
            token_cost=token_cost,
        )


__all__ = [
    "CellCorrection",
    "VerificationResult",
    "VisionVerifier",
]
