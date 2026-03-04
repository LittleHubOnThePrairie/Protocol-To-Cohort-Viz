"""Pre-parser protocol format detection gate (PTCV-31).

Runs on raw protocol text **before** IchParser's classifier to detect
whether the document follows ICH E6(R3), FDA CTD, FDA IND, or an
unknown/proprietary structure. Non-ICH protocols are routed to the
review queue immediately — skipping full classification.

Risk tier: LOW — regex-only, no LLM/embedding calls, no patient data.

Regulatory references:
- ALCOA+ Accurate: format detection informs downstream reliability
- ALCOA+ Contemporaneous: detection runs before classification
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ProtocolFormat(str, Enum):
    """Detected protocol format family."""

    ICH_E6R3 = "ICH_E6R3"
    ICH_M11 = "ICH_M11"
    CTD = "CTD"
    FDA_IND = "FDA_IND"
    UNKNOWN = "UNKNOWN"


@dataclass
class FormatDetectionResult:
    """Result of pre-parser format detection.

    Attributes:
        format: Detected protocol format family.
        confidence: Detection confidence in [0.0, 1.0].
        positive_markers: Which markers triggered the detection.
        recommendation: Human-readable routing decision.
    """

    format: ProtocolFormat
    confidence: float
    positive_markers: list[str] = field(default_factory=list)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Marker definitions
# ---------------------------------------------------------------------------

_ICH_HEADING_RE = re.compile(
    r"\bB\.\d{1,2}\b",
    re.IGNORECASE,
)

_ICH_KEYWORD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bICH\s+E6\s*\(?\s*R\s*[23]\s*\)?\b", re.IGNORECASE),
     "ICH E6(R3) reference"),
    (re.compile(r"\bAppendix\s+B\b", re.IGNORECASE),
     "Appendix B reference"),
    (re.compile(r"\bEudraCT\b|\bEUCT\b", re.IGNORECASE),
     "EudraCT/EUCT identifier"),
    (re.compile(r"\bNCT\d{5,}\b"),
     "ClinicalTrials.gov NCT ID"),
    (re.compile(r"\bGCP\s+(?:compliance|guideline)\b", re.IGNORECASE),
     "GCP compliance reference"),
    (re.compile(r"\bE6\s*\(\s*R\s*[23]\s*\)\b"),
     "E6(R2/R3) shorthand"),
]

_CTD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bModule\s+[1-5](?:\.[1-9]\d*)*\b", re.IGNORECASE),
     "CTD Module reference"),
    (re.compile(r"\b5\.3\.\d+\b"),
     "CTD 5.3.x section"),
    (re.compile(r"\bCommon\s+Technical\s+Document\b", re.IGNORECASE),
     "CTD full name"),
    (re.compile(r"\beCTD\b"),
     "eCTD reference"),
]

_FDA_IND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bIND\s+(?:Application|submission)\b", re.IGNORECASE),
     "IND Application reference"),
    (re.compile(r"\b21\s*CFR\s*(?:Part\s*)?312\b", re.IGNORECASE),
     "21 CFR 312 reference"),
    (re.compile(r"\bInvestigational\s+New\s+Drug\b", re.IGNORECASE),
     "Investigational New Drug"),
]

# ICH M11 CeSHarP markers (PTCV-56)
_M11_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bCeSHarP\b"),
     "CeSHarP reference"),
    (re.compile(r"\bICH\s+M11\b", re.IGNORECASE),
     "ICH M11 reference"),
    (re.compile(r"\bm11_version\b"),
     "M11 version field"),
    (re.compile(r"\bmachine.readable\s+protocol\b", re.IGNORECASE),
     "Machine-readable protocol"),
]


class FormatDetector:
    """Lightweight pre-parser format gate.

    Uses only regex matching — no embeddings, no LLM calls, no external
    dependencies. Completes in microseconds on any input size.

    [PTCV-31 Scenario: FormatDetector completes without API calls]
    """

    def detect(self, text: str) -> FormatDetectionResult:
        """Detect protocol format from raw text.

        Args:
            text: Raw protocol text (UTF-8 string).

        Returns:
            FormatDetectionResult with format, confidence, and markers.

        [PTCV-31 Scenario: ICH E6(R3) protocol passes through]
        [PTCV-31 Scenario: Non-ICH protocol gated before classification]
        [PTCV-31 Scenario: CTD-formatted protocol detected]
        """
        markers: list[str] = []

        # --- ICH M11 CeSHarP markers (check first — PTCV-56) ---
        m11_markers: list[str] = []
        for pattern, label in _M11_PATTERNS:
            if pattern.search(text):
                m11_markers.append(label)
        if len(m11_markers) >= 2:
            return FormatDetectionResult(
                format=ProtocolFormat.ICH_M11,
                confidence=min(0.80 + 0.05 * len(m11_markers), 1.0),
                positive_markers=m11_markers,
                recommendation=(
                    "Protocol follows ICH M11 CeSHarP structure. "
                    "Route to M11ProtocolParser for direct extraction."
                ),
            )

        # --- ICH markers ---
        heading_hits = _ICH_HEADING_RE.findall(text)
        unique_headings = set(h.upper() for h in heading_hits)
        if unique_headings:
            markers.append(
                f"B.x headings found: {', '.join(sorted(unique_headings))}"
            )

        for pattern, label in _ICH_KEYWORD_PATTERNS:
            if pattern.search(text):
                markers.append(label)

        ich_score = self._ich_confidence(len(unique_headings), len(markers))

        # --- CTD markers ---
        ctd_markers: list[str] = []
        for pattern, label in _CTD_PATTERNS:
            if pattern.search(text):
                ctd_markers.append(label)

        # --- FDA IND markers ---
        ind_markers: list[str] = []
        for pattern, label in _FDA_IND_PATTERNS:
            if pattern.search(text):
                ind_markers.append(label)

        # --- Decision ---
        # Prioritise ICH (most common in the pipeline's target population)
        if ich_score >= 0.70:
            return FormatDetectionResult(
                format=ProtocolFormat.ICH_E6R3,
                confidence=ich_score,
                positive_markers=markers,
                recommendation=(
                    "Protocol follows ICH E6(R3) structure. "
                    "Proceed with full section classification."
                ),
            )

        if len(ctd_markers) >= 2:
            all_markers = markers + ctd_markers
            return FormatDetectionResult(
                format=ProtocolFormat.CTD,
                confidence=min(0.50 + 0.15 * len(ctd_markers), 0.90),
                positive_markers=all_markers,
                recommendation=(
                    "Protocol appears to follow FDA CTD format. "
                    "ICH section classification may be incomplete."
                ),
            )

        if len(ind_markers) >= 2:
            all_markers = markers + ind_markers
            return FormatDetectionResult(
                format=ProtocolFormat.FDA_IND,
                confidence=min(0.50 + 0.15 * len(ind_markers), 0.90),
                positive_markers=all_markers,
                recommendation=(
                    "Protocol appears to follow FDA IND format. "
                    "ICH section classification may be incomplete."
                ),
            )

        # Ambiguous: some ICH markers but below threshold
        if ich_score >= 0.20:
            return FormatDetectionResult(
                format=ProtocolFormat.ICH_E6R3,
                confidence=ich_score,
                positive_markers=markers,
                recommendation=(
                    "Protocol has some ICH markers but structure is "
                    "incomplete. Classification will proceed with "
                    "PARTIAL_ICH expectation."
                ),
            )

        # UNKNOWN — no meaningful markers found
        return FormatDetectionResult(
            format=ProtocolFormat.UNKNOWN,
            confidence=max(ich_score, 0.0),
            positive_markers=markers,
            recommendation=(
                "Protocol does not match ICH E6(R3), CTD, or FDA IND "
                "format. Routing to review queue. Manual review required."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ich_confidence(
        heading_count: int, total_marker_count: int
    ) -> float:
        """Compute ICH confidence from marker counts.

        Heuristic: each unique B.x heading contributes 0.15 (up to 0.60),
        each additional keyword marker contributes 0.10 (up to 0.40).
        """
        heading_score = min(heading_count * 0.15, 0.60)
        # Keyword markers beyond headings
        keyword_markers = max(total_marker_count - (1 if heading_count else 0), 0)
        keyword_score = min(keyword_markers * 0.10, 0.40)
        return round(min(heading_score + keyword_score, 1.0), 4)
