"""Narrative SoA extractor — Stream B (PTCV-260).

Extracts assessment-visit pairs from prose text (B.7/B.9 query hits).
Handles patterns like:
  - "Vital signs at Screening, Day 1, Day 15, and Day 29"
  - "ECG at Screening and End of Treatment"
  - "CBC every 3 cycles"
  - "Physical exam at each visit"

Risk tier: LOW — regex-based text analysis, no API calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AssessmentVisitPair:
    """A single assessment-visit relationship from any stream.

    Attributes:
        assessment_name: Name of the assessment/procedure.
        visit_label: Visit name or timing label.
        source: Origin stream ("table", "narrative", "diagram").
        confidence: Confidence in the pair (0.0–1.0).
    """

    assessment_name: str
    visit_label: str
    source: str = "narrative"
    confidence: float = 0.70


# ---------------------------------------------------------------------------
# Regex patterns for assessment-visit extraction
# ---------------------------------------------------------------------------

# Visit label alternatives for use in regex groups
_VISIT_ALTS = (
    r"Screening|Baseline|Day\s*\d+|Week\s*\d+"
    r"|Month\s*\d+|Cycle\s*\d+|Visit\s*\d+"
    r"|End\s+of\s+(?:Treatment|Study)"
    r"|Early\s+Termination|Follow[\s-]*up|Unscheduled"
)

# "X will be performed/collected/assessed at ..."
_PERFORMED_AT = re.compile(
    r"(?P<assessment>[A-Z][A-Za-z /\-()]+?)"
    r"\s+(?:will\s+be\s+)?(?:performed|collected|assessed|obtained|measured|done)"
    r"\s+(?:at|during|on)\s+"
    r"(?P<visits>(?:(?:" + _VISIT_ALTS + r"|each\s+visit)"
    r"(?:\s*(?:,\s*(?:and\s+)?|\s+and\s+)(?=" + _VISIT_ALTS + r"))?)+)",
    re.IGNORECASE,
)

# "X at/during Screening, Day 1, Day 15, and Day 29"
_AT_PATTERN = re.compile(
    r"(?P<assessment>[A-Z][A-Za-z /\-()]+?)"
    r"\s+(?:at|during|on)\s+"
    r"(?P<visits>(?:(?:" + _VISIT_ALTS + r")"
    r"(?:\s*(?:,\s*(?:and\s+)?|\s+and\s+)(?=" + _VISIT_ALTS + r"))?)+)",
    re.IGNORECASE,
)

# "X every N cycles/weeks/months"
_EVERY_PATTERN = re.compile(
    r"(?P<assessment>[A-Z][A-Za-z /\-()]+?)"
    r"\s+every\s+(?P<interval>\d+)\s*(?P<unit>cycles?|weeks?|months?|visits?)",
    re.IGNORECASE,
)

# "X at each visit" / "X at every visit"
_EACH_VISIT = re.compile(
    r"(?P<assessment>[A-Z][A-Za-z /\-()]+?)"
    r"\s+(?:at\s+)?(?:each|every)\s+(?:scheduled\s+)?visit",
    re.IGNORECASE,
)

# Split visit list: "Screening, Day 1, Day 15, and Day 29"
_VISIT_SPLIT = re.compile(
    r"\s*(?:,\s*(?:and\s+)?|(?:\s+and\s+))\s*",
    re.IGNORECASE,
)

# Valid visit label pattern
_VISIT_LABEL = re.compile(
    r"(?:Screening|Baseline|Day\s*\d+|Week\s*\d+|Month\s*\d+"
    r"|Cycle\s*\d+|Visit\s*\d+|End\s+of\s+(?:Treatment|Study)"
    r"|Early\s+Termination|Follow[\s-]*up|Unscheduled"
    r"|each\s+visit|every\s+visit)",
    re.IGNORECASE,
)


def _split_visits(visit_text: str) -> list[str]:
    """Split a comma/and-separated visit list into individual labels."""
    parts = _VISIT_SPLIT.split(visit_text.strip())
    result = []
    for part in parts:
        part = part.strip()
        if part and _VISIT_LABEL.match(part):
            result.append(part)
    return result


def _clean_assessment(name: str) -> str:
    """Normalise assessment name."""
    name = name.strip()
    # Remove trailing verbs/auxiliary that leaked from the pattern
    name = re.sub(
        r"\s+(?:will\s+be\s+\w+|will|shall|should|is|are)\s*$",
        "", name, flags=re.IGNORECASE,
    )
    return name.strip()


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_from_narrative(
    texts: list[str],
    source_label: str = "narrative",
) -> list[AssessmentVisitPair]:
    """Extract assessment-visit pairs from narrative prose.

    Scans text for patterns describing which assessments are
    performed at which visits. Returns deduplicated pairs.

    Args:
        texts: List of prose text strings (e.g., B.7/B.9 query hits).
        source_label: Source tag for provenance.

    Returns:
        List of AssessmentVisitPair objects.
    """
    pairs: list[AssessmentVisitPair] = []
    seen: set[tuple[str, str]] = set()

    for text in texts:
        if not text or not text.strip():
            continue

        # Pattern 1: "X performed/collected at Visit list"
        for pattern in (_PERFORMED_AT, _AT_PATTERN):
            for m in pattern.finditer(text):
                assessment = _clean_assessment(m.group("assessment"))
                if not assessment or len(assessment) < 3:
                    continue
                visits = _split_visits(m.group("visits"))
                for visit in visits:
                    key = (assessment.lower(), visit.lower())
                    if key not in seen:
                        seen.add(key)
                        pairs.append(AssessmentVisitPair(
                            assessment_name=assessment,
                            visit_label=visit,
                            source=source_label,
                            confidence=0.70,
                        ))

        # Pattern 2: "X every N cycles"
        for m in _EVERY_PATTERN.finditer(text):
            assessment = _clean_assessment(m.group("assessment"))
            if not assessment or len(assessment) < 3:
                continue
            interval = m.group("interval")
            unit = m.group("unit").rstrip("s")
            visit = f"Every {interval} {unit}s"
            key = (assessment.lower(), visit.lower())
            if key not in seen:
                seen.add(key)
                pairs.append(AssessmentVisitPair(
                    assessment_name=assessment,
                    visit_label=visit,
                    source=source_label,
                    confidence=0.65,
                ))

        # Pattern 3: "X at each visit"
        for m in _EACH_VISIT.finditer(text):
            assessment = _clean_assessment(m.group("assessment"))
            if not assessment or len(assessment) < 3:
                continue
            key = (assessment.lower(), "each visit")
            if key not in seen:
                seen.add(key)
                pairs.append(AssessmentVisitPair(
                    assessment_name=assessment,
                    visit_label="Each Visit",
                    source=source_label,
                    confidence=0.60,
                ))

    if pairs:
        logger.info(
            "PTCV-260: Narrative extraction: %d assessment-visit "
            "pairs from %d text blocks",
            len(pairs), len(texts),
        )

    return pairs


__all__ = [
    "AssessmentVisitPair",
    "extract_from_narrative",
]
