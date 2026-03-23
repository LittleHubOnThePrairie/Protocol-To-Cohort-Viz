"""Visit synonym resolver for SoA column header canonicalisation.

Maps raw visit header text (e.g. "Visit 0", "Pre-treatment", "Day 15 ± 3")
to canonical USDM visit types and temporal attributes.

Primary strategy: rule-based lookup dictionary + regex for temporal
expressions. SpaCy EntityRuler is used as an optional enhancement when
SpaCy is installed and compatible with the current Python version (lazy
import — falls back to regex when import fails, e.g. Python 3.14+).

Risk tier: MEDIUM — NLP component (no patient data).

Regulatory references:
- ALCOA+ Accurate: synonym resolution method + confidence recorded in
  synonym_mappings.parquet for audit
- ALCOA+ Traceable: review_required=True when confidence < 0.80
"""

from __future__ import annotations

import re
from typing import NamedTuple

from .models import SynonymMapping


# ---------------------------------------------------------------------------
# Visit type taxonomy (9 types per CDISC USDM v4.0 and PTCV-21 spec)
# ---------------------------------------------------------------------------

VISIT_TYPES = [
    "Screening",
    "Baseline",
    "Treatment",
    "Follow-up",
    "Unscheduled",
    "Early Termination",
    "Remote",
    "Long-term Follow-up",
    "End of Study",
]

# conditional_rule values for special visit types
_CONDITIONAL_RULES: dict[str, str] = {
    "Unscheduled": "PRN",
    "Early Termination": "EARLY_TERM",
}

# ---------------------------------------------------------------------------
# Synonym lookup table  (lower-cased key → canonical type)
# ---------------------------------------------------------------------------

_LOOKUP: dict[str, str] = {
    # Screening
    "screening": "Screening",
    "pre-treatment": "Screening",
    "pretreatment": "Screening",
    "pre-dose": "Screening",
    "predose": "Screening",
    "visit 0": "Screening",
    "v0": "Screening",
    "screen": "Screening",
    "pre-study": "Screening",
    "prestudy": "Screening",
    "eligibility": "Screening",
    # Baseline
    "baseline": "Baseline",
    "day 1": "Baseline",
    "day1": "Baseline",
    "visit 1": "Baseline",
    "v1": "Baseline",
    "randomisation": "Baseline",
    "randomization": "Baseline",
    "enrolment": "Baseline",
    "enrollment": "Baseline",
    "c1d1": "Baseline",
    # Treatment (generic — specific week/day patterns handled via regex)
    "treatment": "Treatment",
    "on treatment": "Treatment",
    "on-treatment": "Treatment",
    "daily": "Treatment",
    "weekly": "Treatment",
    "weekly thereafter": "Treatment",
    # Follow-up
    "follow-up": "Follow-up",
    "follow up": "Follow-up",
    "post-treatment": "Follow-up",
    "post treatment": "Follow-up",
    "safety follow-up": "Follow-up",
    "safety follow up": "Follow-up",
    # Unscheduled
    "unscheduled": "Unscheduled",
    "unscheduled visit": "Unscheduled",
    "prn visit": "Unscheduled",
    "prn": "Unscheduled",
    "as needed": "Unscheduled",
    "unplanned": "Unscheduled",
    # Early Termination
    "early termination": "Early Termination",
    "early withdrawal": "Early Termination",
    "withdrawal": "Early Termination",
    "discontinued": "Early Termination",
    "eotr": "Early Termination",
    "early term": "Early Termination",
    "discontinuation": "Early Termination",
    "premature discontinuation": "Early Termination",
    # Remote
    "remote": "Remote",
    "telephone": "Remote",
    "phone call": "Remote",
    "phone visit": "Remote",
    "telemedicine": "Remote",
    "tele-visit": "Remote",
    "telehealth": "Remote",
    # Long-term Follow-up
    "long-term follow-up": "Long-term Follow-up",
    "long term follow-up": "Long-term Follow-up",
    "long-term follow up": "Long-term Follow-up",
    "ltfu": "Long-term Follow-up",
    "extended follow-up": "Long-term Follow-up",
    "extended follow up": "Long-term Follow-up",
    "late follow-up": "Long-term Follow-up",
    # End of Study
    "end of study": "End of Study",
    "eos": "End of Study",
    "end of treatment": "End of Study",
    "eot": "End of Study",
    "final visit": "End of Study",
    "study completion": "End of Study",
    "completion": "End of Study",
}

# ---------------------------------------------------------------------------
# Temporal regex patterns
# ---------------------------------------------------------------------------

# "Day 15", "Day -14", "Day -1"
_DAY_RE = re.compile(r"[Dd]ay\s*(-?\d+)")
# "Day 15 ± 3" / "Day 15 +/- 3" / "Day 15 ±3"
_DAY_WINDOW_RE = re.compile(
    r"[Dd]ay\s*(-?\d+)\s*(?:[±+]\s*(\d+)|[+]\s*/\s*[-]\s*(\d+))"
)
# "Days -14 to -1" / "Days 8-14"
_DAY_RANGE_RE = re.compile(r"[Dd]ays?\s*(-?\d+)\s*(?:to|-)\s*(-?\d+)")
# "Week 2" / "Week 12"
_WEEK_RE = re.compile(r"[Ww]eek\s*(\d+)")
# "Week 2 ± 3" / "Week 4 + 3"
_WEEK_WINDOW_RE = re.compile(
    r"[Ww]eek\s*(\d+)\s*(?:[±+]\s*(\d+)|[+]\s*/\s*[-]\s*(\d+))"
)
# "Cycle 2 Day 1" / "C2D1"
_CYCLE_RE = re.compile(r"(?:[Cc]ycle|C)\s*(\d+)[,\s]+[Dd]ay\s*(\d+)")
_CYCLE_SHORT_RE = re.compile(r"C(\d+)D(\d+)", re.IGNORECASE)
# "Year 1 / Year 2" (long-term FU)
_YEAR_RE = re.compile(r"[Yy]ear\s*(\d+)")
# "Month 3" / "Month 6"
_MONTH_RE = re.compile(r"[Mm]onth\s*(\d+)")
# "V2", "Visit 3", "Visit 4/ET" (PTCV-131)
_VISIT_NUM_RE = re.compile(r"(?:[Vv]isit\s*|[Vv])(\d+)")


# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------

class ResolvedVisit(NamedTuple):
    """Output of a single SynonymResolver.resolve() call.

    Attributes:
        visit_type: Canonical type from VISIT_TYPES.
        day_offset: Relative day from Day 1.
        window_early: Days before day_offset (negative convention stored as
            negative; stored as non-negative in timepoints.window_early to
            match TVSTRL semantics — the caller negates if needed).
        window_late: Days after day_offset.
        conditional_rule: "PRN", "EARLY_TERM", or "".
        repeat_cycle: Cycle label e.g. "C2", or "".
        mandatory: True by default; False for Unscheduled/Early Termination.
        method: Resolution method string.
        confidence: 0.0–1.0.
    """

    visit_type: str
    day_offset: int
    window_early: int
    window_late: int
    conditional_rule: str
    repeat_cycle: str
    mandatory: bool
    method: str
    confidence: float


# ---------------------------------------------------------------------------
# SynonymResolver
# ---------------------------------------------------------------------------

class SynonymResolver:
    """Map raw SoA column headers to canonical USDM visit attributes.

    Uses lookup dictionary + regex for temporal expressions.
    SpaCy EntityRuler is activated lazily when available and compatible
    with the current Python runtime; falls back to regex otherwise.

    Args:
        use_spacy: Attempt to load SpaCy. Defaults to True. Pass False
            to force pure rule-based mode (useful in tests).
    """

    _SYNONYM_CONFIDENCE = 0.92
    _REGEX_CONFIDENCE = 0.85
    _SPACY_CONFIDENCE = 0.88
    _DEFAULT_CONFIDENCE = 0.60

    def __init__(self, use_spacy: bool = True) -> None:
        self._nlp = None
        if use_spacy:
            self._nlp = self._load_spacy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve(self, header: str) -> ResolvedVisit:
        """Resolve a raw visit header to canonical USDM attributes.

        Args:
            header: Raw column header text from the SoA table,
                e.g. "Visit 0", "Week 2 (Day 8 ± 3)", "Early Termination".

        Returns:
            ResolvedVisit with visit type, temporal attributes, and
            resolution metadata.
        [PTCV-21 Scenario: Synonym resolution logged in synonym_mappings]
        """
        text = header.strip()
        lower = text.lower()

        # 1. Direct lookup (highest confidence)
        if lower in _LOOKUP:
            visit_type = _LOOKUP[lower]
            day, we, wl, cycle = self._default_temporal(visit_type)
            return ResolvedVisit(
                visit_type=visit_type,
                day_offset=day,
                window_early=we,
                window_late=wl,
                conditional_rule=_CONDITIONAL_RULES.get(visit_type, ""),
                repeat_cycle=cycle,
                mandatory=visit_type not in ("Unscheduled", "Early Termination"),
                method="lookup",
                confidence=self._SYNONYM_CONFIDENCE,
            )

        # 2. SpaCy NER (when available)
        if self._nlp is not None:
            result = self._resolve_with_spacy(text)
            if result is not None:
                return result

        # 3. Regex temporal extraction
        result = self._resolve_with_regex(text)
        if result is not None:
            return result

        # 4. Partial match in lookup keys
        for key, visit_type in _LOOKUP.items():
            if key in lower or lower in key:
                day, we, wl, cycle = self._default_temporal(visit_type)
                return ResolvedVisit(
                    visit_type=visit_type,
                    day_offset=day,
                    window_early=we,
                    window_late=wl,
                    conditional_rule=_CONDITIONAL_RULES.get(visit_type, ""),
                    repeat_cycle=cycle,
                    mandatory=visit_type not in (
                        "Unscheduled", "Early Termination"
                    ),
                    method="lookup",
                    confidence=0.75,
                )

        # 5. Default fallback
        return ResolvedVisit(
            visit_type="Treatment",
            day_offset=0,
            window_early=0,
            window_late=0,
            conditional_rule="",
            repeat_cycle="",
            mandatory=True,
            method="default",
            confidence=self._DEFAULT_CONFIDENCE,
        )

    def resolve_to_mapping(
        self, header: str, run_id: str, timestamp: str
    ) -> tuple[ResolvedVisit, SynonymMapping]:
        """Resolve a header and return both result and audit mapping.

        Args:
            header: Raw visit header text.
            run_id: Run UUID4 for this extraction.
            timestamp: ISO 8601 UTC timestamp for the mapping record.

        Returns:
            Tuple of (ResolvedVisit, SynonymMapping).
        [PTCV-21 Scenario: Synonym resolution logged in synonym_mappings]
        """
        resolved = self.resolve(header)
        mapping = SynonymMapping(
            run_id=run_id,
            original_text=header,
            canonical_label=resolved.visit_type,
            method=resolved.method,
            confidence=resolved.confidence,
            review_required=resolved.confidence < 0.80,
            extraction_timestamp_utc=timestamp,
        )
        return resolved, mapping

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_spacy(self):  # type: ignore[return]
        """Attempt to load a SpaCy blank pipeline with EntityRuler.

        Returns None on any import error (e.g. Python 3.14+ incompatibility).
        """
        try:
            import spacy  # type: ignore[import]
            nlp = spacy.blank("en")
            ruler = nlp.add_pipe("entity_ruler")
            ruler.add_patterns(self._spacy_patterns())
            return nlp
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _spacy_patterns() -> list[dict]:
        """EntityRuler patterns for temporal clinical visit expressions."""
        return [
            # "Day 1", "Day -14"
            {"label": "VISIT_DAY", "pattern": [
                {"LOWER": "day"}, {"IS_DIGIT": True}]},
            # "Week 2", "Week 12"
            {"label": "VISIT_WEEK", "pattern": [
                {"LOWER": "week"}, {"IS_DIGIT": True}]},
            # "Cycle 2 Day 1"
            {"label": "VISIT_CYCLE", "pattern": [
                {"LOWER": "cycle"}, {"IS_DIGIT": True},
                {"LOWER": "day"}, {"IS_DIGIT": True}]},
            # "Year 1"
            {"label": "VISIT_YEAR", "pattern": [
                {"LOWER": "year"}, {"IS_DIGIT": True}]},
        ]

    def _resolve_with_spacy(self, text: str) -> ResolvedVisit | None:
        """Use SpaCy EntityRuler to classify temporal visit headers."""
        if self._nlp is None:
            return None
        doc = self._nlp(text)
        for ent in doc.ents:
            digits = [t.text for t in ent if t.is_digit or t.text.lstrip("-").isdigit()]
            if not digits:
                continue
            if ent.label_ == "VISIT_DAY":
                day = int(digits[0])
                vtype = "Baseline" if day == 1 else "Treatment"
                return ResolvedVisit(
                    visit_type=vtype,
                    day_offset=day,
                    window_early=0,
                    window_late=0,
                    conditional_rule="",
                    repeat_cycle="",
                    mandatory=True,
                    method="spacy_ner",
                    confidence=self._SPACY_CONFIDENCE,
                )
            if ent.label_ == "VISIT_WEEK":
                week = int(digits[0])
                day = (week - 1) * 7 + 1
                return ResolvedVisit(
                    visit_type="Treatment",
                    day_offset=day,
                    window_early=0,
                    window_late=0,
                    conditional_rule="",
                    repeat_cycle="",
                    mandatory=True,
                    method="spacy_ner",
                    confidence=self._SPACY_CONFIDENCE,
                )
            if ent.label_ == "VISIT_YEAR":
                year = int(digits[0])
                return ResolvedVisit(
                    visit_type="Long-term Follow-up",
                    day_offset=year * 365,
                    window_early=0,
                    window_late=30,
                    conditional_rule="",
                    repeat_cycle="",
                    mandatory=True,
                    method="spacy_ner",
                    confidence=self._SPACY_CONFIDENCE,
                )
        return None

    def _resolve_with_regex(self, text: str) -> ResolvedVisit | None:
        """Extract temporal attributes via regex patterns."""
        # Cycle pattern — highest specificity
        m = _CYCLE_RE.search(text) or _CYCLE_SHORT_RE.search(text)
        if m:
            cycle_num, day_num = int(m.group(1)), int(m.group(2))
            cycle_label = f"C{cycle_num}"
            vtype = "Baseline" if cycle_num == 1 and day_num == 1 else "Treatment"
            return ResolvedVisit(
                visit_type=vtype,
                day_offset=day_num,
                window_early=0,
                window_late=0,
                conditional_rule="",
                repeat_cycle=cycle_label,
                mandatory=True,
                method="regex",
                confidence=self._REGEX_CONFIDENCE,
            )

        # Year (long-term FU)
        m = _YEAR_RE.search(text)
        if m:
            year = int(m.group(1))
            return ResolvedVisit(
                visit_type="Long-term Follow-up",
                day_offset=year * 365,
                window_early=0,
                window_late=30,
                conditional_rule="",
                repeat_cycle="",
                mandatory=True,
                method="regex",
                confidence=self._REGEX_CONFIDENCE,
            )

        # Month (follow-up or treatment depending on context)
        m = _MONTH_RE.search(text)
        if m:
            month = int(m.group(1))
            return ResolvedVisit(
                visit_type="Follow-up" if month > 6 else "Treatment",
                day_offset=month * 30,
                window_early=0,
                window_late=7,
                conditional_rule="",
                repeat_cycle="",
                mandatory=True,
                method="regex",
                confidence=self._REGEX_CONFIDENCE,
            )

        # Week with window
        m = _WEEK_WINDOW_RE.search(text)
        if m:
            week = int(m.group(1))
            window = int(m.group(2) or m.group(3) or 0)
            day = (week - 1) * 7 + 1
            return ResolvedVisit(
                visit_type="Treatment",
                day_offset=day,
                window_early=window,
                window_late=window,
                conditional_rule="",
                repeat_cycle="",
                mandatory=True,
                method="regex",
                confidence=self._REGEX_CONFIDENCE,
            )

        # Week (no window)
        m = _WEEK_RE.search(text)
        if m:
            week = int(m.group(1))
            day = (week - 1) * 7 + 1
            return ResolvedVisit(
                visit_type="Treatment",
                day_offset=day,
                window_early=0,
                window_late=0,
                conditional_rule="",
                repeat_cycle="",
                mandatory=True,
                method="regex",
                confidence=self._REGEX_CONFIDENCE,
            )

        # Day range: "Days -14 to -1"
        m = _DAY_RANGE_RE.search(text)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            mid = (start + end) // 2
            vtype = "Screening" if start < 0 else "Treatment"
            return ResolvedVisit(
                visit_type=vtype,
                day_offset=start,
                window_early=0,
                window_late=end - start,
                conditional_rule="",
                repeat_cycle="",
                mandatory=True,
                method="regex",
                confidence=self._REGEX_CONFIDENCE,
            )

        # Day with window: "Day 15 ± 3"
        m = _DAY_WINDOW_RE.search(text)
        if m:
            day = int(m.group(1))
            window = int(m.group(2) or m.group(3) or 0)
            vtype = "Baseline" if day == 1 else "Screening" if day < 0 else "Treatment"
            return ResolvedVisit(
                visit_type=vtype,
                day_offset=day,
                window_early=window,
                window_late=window,
                conditional_rule="",
                repeat_cycle="",
                mandatory=True,
                method="regex",
                confidence=self._REGEX_CONFIDENCE,
            )

        # Day (no window): "Day 8"
        m = _DAY_RE.search(text)
        if m:
            day = int(m.group(1))
            vtype = "Baseline" if day == 1 else "Screening" if day < 0 else "Treatment"
            return ResolvedVisit(
                visit_type=vtype,
                day_offset=day,
                window_early=0,
                window_late=0,
                conditional_rule="",
                repeat_cycle="",
                mandatory=True,
                method="regex",
                confidence=self._REGEX_CONFIDENCE,
            )

        # Visit number: "V2", "Visit 3", "V4/ET" (PTCV-131)
        m = _VISIT_NUM_RE.search(text)
        if m:
            num = int(m.group(1))
            if num == 0:
                vtype = "Screening"
            elif num == 1:
                vtype = "Baseline"
            else:
                vtype = "Treatment"
            return ResolvedVisit(
                visit_type=vtype,
                day_offset=num,
                window_early=0,
                window_late=0,
                conditional_rule="",
                repeat_cycle="",
                mandatory=True,
                method="regex",
                confidence=0.70,
            )

        return None

    @staticmethod
    def _default_temporal(
        visit_type: str,
    ) -> tuple[int, int, int, str]:
        """Return (day_offset, window_early, window_late, repeat_cycle) defaults."""
        defaults: dict[str, tuple[int, int, int, str]] = {
            "Screening": (-7, 0, 7, ""),
            "Baseline": (1, 0, 0, ""),
            "Treatment": (0, 0, 0, ""),
            "Follow-up": (0, 0, 14, ""),
            "Unscheduled": (0, 0, 0, ""),
            "Early Termination": (0, 0, 0, ""),
            "Remote": (0, 0, 7, ""),
            "Long-term Follow-up": (0, 0, 30, ""),
            "End of Study": (0, 0, 0, ""),
        }
        return defaults.get(visit_type, (0, 0, 0, ""))
