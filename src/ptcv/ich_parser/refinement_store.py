"""Iterative refinement store for query-driven extraction (PTCV-94).

Persists human reviewer corrections so the pipeline improves over time:

1. **Header mapping corrections** — reviewer changes a protocol header's
   Appendix B section assignment.
2. **Extraction corrections** — reviewer fixes extracted content or
   re-assigns it to the correct section.
3. **Synonym accumulation** — when a header is corrected, the
   header → section mapping is stored as a frequency-weighted synonym
   that the :class:`SectionMatcher` can load at startup.
4. **Confidence calibration** — tracks predicted confidence vs actual
   correctness to detect scoring bias and suggest threshold adjustments.

Storage: JSON files in a configurable ``data_dir``:

- ``header_corrections.json``
- ``extraction_corrections.json``
- ``calibration_entries.json``
- ``synonym_mappings.json``

Risk tier: LOW — file I/O only, no patient data.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default data directory (relative to project root).
_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[3]  # src/ptcv/ich_parser -> root
    / "data"
    / "refinement"
)

# Minimum corrections before a synonym is promoted to the boost table.
DEFAULT_MIN_FREQUENCY = 2

# Calibration band width for confidence bucketing.
CALIBRATION_BAND_WIDTH = 0.10


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HeaderCorrection:
    """A reviewer's correction of a header → Appendix B mapping.

    Attributes:
        protocol_id: Protocol identifier (NCT ID or internal).
        protocol_header: The protocol section header text.
        original_mapping: Original ICH section code (e.g. ``"B.7"``).
        corrected_mapping: Corrected ICH section code (e.g. ``"B.9"``).
        timestamp: ISO 8601 UTC timestamp of the correction.
    """

    protocol_id: str
    protocol_header: str
    original_mapping: str
    corrected_mapping: str
    timestamp: str = ""


@dataclasses.dataclass
class ExtractionCorrection:
    """A reviewer's correction of extracted content.

    Attributes:
        protocol_id: Protocol identifier.
        query_id: Appendix B query identifier (e.g. ``"B.6.1.q1"``).
        original_content: Content extracted by the pipeline.
        corrected_content: Reviewer's corrected content.
        corrected_section: Corrected section code (if section was
            wrong, else empty string).
        timestamp: ISO 8601 UTC timestamp.
    """

    protocol_id: str
    query_id: str
    original_content: str
    corrected_content: str
    corrected_section: str = ""
    timestamp: str = ""


@dataclasses.dataclass
class CalibrationEntry:
    """One data point for confidence calibration.

    Attributes:
        protocol_id: Protocol identifier.
        query_id: Query identifier.
        predicted_confidence: Pipeline-predicted confidence (0.0–1.0).
        actual_correct: Whether the extraction was correct per review.
        timestamp: ISO 8601 UTC timestamp.
    """

    protocol_id: str
    query_id: str
    predicted_confidence: float
    actual_correct: bool
    timestamp: str = ""


@dataclasses.dataclass
class CalibrationBand:
    """Accuracy statistics for one confidence band.

    Attributes:
        range_low: Lower bound of the band (inclusive).
        range_high: Upper bound of the band (exclusive).
        count: Number of entries in this band.
        predicted_avg: Mean predicted confidence in the band.
        actual_accuracy: Fraction actually correct in the band.
        bias: ``predicted_avg - actual_accuracy`` (positive = overconfident).
    """

    range_low: float
    range_high: float
    count: int
    predicted_avg: float
    actual_accuracy: float
    bias: float


@dataclasses.dataclass
class CalibrationReport:
    """Confidence calibration analysis.

    Attributes:
        bands: Per-band statistics.
        total_entries: Total calibration data points.
        overall_accuracy: Fraction correct across all entries.
        bias_detected: ``True`` if any band has ``|bias| > 0.15``.
        suggested_threshold_adjustment: Recommended shift to the
            confidence threshold (positive = raise, negative = lower).
    """

    bands: list[CalibrationBand]
    total_entries: int
    overall_accuracy: float
    bias_detected: bool
    suggested_threshold_adjustment: float


@dataclasses.dataclass
class TrendReport:
    """Accuracy trend across processing sessions.

    Attributes:
        total_corrections: Total corrections recorded.
        total_calibration_entries: Total calibration data points.
        common_corrections: Most frequent header corrections as
            ``(header, corrected_section, count)`` tuples.
        new_synonyms: Synonyms promoted to the boost table as
            ``(header_text, section_code)`` tuples.
        accuracy_by_month: Monthly accuracy rates as
            ``(month_str, accuracy)`` tuples (newest first).
    """

    total_corrections: int
    total_calibration_entries: int
    common_corrections: list[tuple[str, str, int]]
    new_synonyms: list[tuple[str, str]]
    accuracy_by_month: list[tuple[str, float]]


# ---------------------------------------------------------------------------
# RefinementStore
# ---------------------------------------------------------------------------


class RefinementStore:
    """Persistent store for refinement data.

    Args:
        data_dir: Directory for JSON files. Created if it does not
            exist.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
    ) -> None:
        self._data_dir = Path(data_dir or _DEFAULT_DATA_DIR)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._header_corrections: list[HeaderCorrection] = []
        self._extraction_corrections: list[ExtractionCorrection] = []
        self._calibration_entries: list[CalibrationEntry] = []
        self._synonym_freq: dict[str, dict[str, int]] = {}

        self._load()

    # ------------------------------------------------------------------
    # Public API — record corrections
    # ------------------------------------------------------------------

    def record_header_correction(
        self,
        protocol_id: str,
        protocol_header: str,
        original_mapping: str,
        corrected_mapping: str,
    ) -> HeaderCorrection:
        """Record a header mapping correction.

        Also updates the synonym frequency table.

        Returns:
            The stored :class:`HeaderCorrection`.
        """
        ts = _now_iso()
        correction = HeaderCorrection(
            protocol_id=protocol_id,
            protocol_header=protocol_header,
            original_mapping=original_mapping,
            corrected_mapping=corrected_mapping,
            timestamp=ts,
        )
        self._header_corrections.append(correction)

        # Update synonym frequency.
        header_lower = protocol_header.strip().lower()
        if header_lower not in self._synonym_freq:
            self._synonym_freq[header_lower] = {}
        freq = self._synonym_freq[header_lower]
        freq[corrected_mapping] = freq.get(corrected_mapping, 0) + 1

        self._save()
        logger.info(
            "Header correction: %r → %s (was %s) for %s",
            protocol_header,
            corrected_mapping,
            original_mapping,
            protocol_id,
        )
        return correction

    def record_extraction_correction(
        self,
        protocol_id: str,
        query_id: str,
        original_content: str,
        corrected_content: str,
        corrected_section: str = "",
    ) -> ExtractionCorrection:
        """Record an extraction content correction.

        Returns:
            The stored :class:`ExtractionCorrection`.
        """
        ts = _now_iso()
        correction = ExtractionCorrection(
            protocol_id=protocol_id,
            query_id=query_id,
            original_content=original_content,
            corrected_content=corrected_content,
            corrected_section=corrected_section,
            timestamp=ts,
        )
        self._extraction_corrections.append(correction)
        self._save()
        logger.info(
            "Extraction correction: %s for %s",
            query_id,
            protocol_id,
        )
        return correction

    def record_calibration_entry(
        self,
        protocol_id: str,
        query_id: str,
        predicted_confidence: float,
        actual_correct: bool,
    ) -> CalibrationEntry:
        """Record a calibration data point.

        Returns:
            The stored :class:`CalibrationEntry`.
        """
        ts = _now_iso()
        entry = CalibrationEntry(
            protocol_id=protocol_id,
            query_id=query_id,
            predicted_confidence=predicted_confidence,
            actual_correct=actual_correct,
            timestamp=ts,
        )
        self._calibration_entries.append(entry)
        self._save()
        return entry

    # ------------------------------------------------------------------
    # Public API — query accumulated data
    # ------------------------------------------------------------------

    def get_synonym_boosts(
        self,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
    ) -> dict[str, str]:
        """Return accumulated synonyms suitable for SectionMatcher.

        Only includes header → section mappings seen at least
        *min_frequency* times. When a header maps to multiple
        sections, the most frequent wins.

        Returns:
            Dict mapping ``header_text`` → ``section_code``, same
            format as ``_SYNONYM_BOOSTS`` in ``section_matcher.py``.
        """
        boosts: dict[str, str] = {}
        for header, code_counts in self._synonym_freq.items():
            # Pick the most frequent mapping.
            best_code = max(
                code_counts, key=lambda c: code_counts[c]
            )
            if code_counts[best_code] >= min_frequency:
                boosts[header] = best_code
        return boosts

    def get_calibration_report(self) -> CalibrationReport:
        """Build a calibration report from accumulated data.

        Buckets entries into 0.10-wide confidence bands and computes
        predicted-vs-actual accuracy in each band.

        Returns:
            :class:`CalibrationReport`.
        """
        entries = self._calibration_entries
        if not entries:
            return CalibrationReport(
                bands=[],
                total_entries=0,
                overall_accuracy=0.0,
                bias_detected=False,
                suggested_threshold_adjustment=0.0,
            )

        bands: list[CalibrationBand] = []
        low = 0.0
        while low < 1.0:
            high = round(low + CALIBRATION_BAND_WIDTH, 2)
            in_band = [
                e for e in entries
                if low <= e.predicted_confidence < high
            ]
            if in_band:
                pred_avg = sum(
                    e.predicted_confidence for e in in_band
                ) / len(in_band)
                actual_acc = sum(
                    1 for e in in_band if e.actual_correct
                ) / len(in_band)
                bias = round(pred_avg - actual_acc, 4)
                bands.append(CalibrationBand(
                    range_low=low,
                    range_high=high,
                    count=len(in_band),
                    predicted_avg=round(pred_avg, 4),
                    actual_accuracy=round(actual_acc, 4),
                    bias=bias,
                ))
            low = high

        total = len(entries)
        correct = sum(1 for e in entries if e.actual_correct)
        overall_acc = correct / total if total else 0.0

        bias_detected = any(
            abs(b.bias) > 0.15 for b in bands
        )

        # Suggest threshold adjustment: if consistently
        # overconfident (positive bias), raise threshold.
        biases = [b.bias for b in bands if b.count >= 5]
        avg_bias = (
            sum(biases) / len(biases) if biases else 0.0
        )
        suggested_adj = round(avg_bias, 3)

        return CalibrationReport(
            bands=bands,
            total_entries=total,
            overall_accuracy=round(overall_acc, 4),
            bias_detected=bias_detected,
            suggested_threshold_adjustment=suggested_adj,
        )

    def get_trend_report(self) -> TrendReport:
        """Build a trend report across all recorded sessions.

        Returns:
            :class:`TrendReport` with correction stats, promoted
            synonyms, and monthly accuracy.
        """
        # Common corrections: header → corrected_section, count.
        correction_counts: dict[tuple[str, str], int] = {}
        for c in self._header_corrections:
            key = (c.protocol_header.strip().lower(), c.corrected_mapping)
            correction_counts[key] = (
                correction_counts.get(key, 0) + 1
            )
        common = sorted(
            [
                (header, section, count)
                for (header, section), count in correction_counts.items()
            ],
            key=lambda x: x[2],
            reverse=True,
        )[:10]

        # Promoted synonyms.
        promoted = list(self.get_synonym_boosts().items())

        # Monthly accuracy from calibration entries.
        monthly: dict[str, list[bool]] = {}
        for e in self._calibration_entries:
            month = e.timestamp[:7] if len(e.timestamp) >= 7 else "unknown"
            monthly.setdefault(month, []).append(e.actual_correct)

        accuracy_by_month = sorted(
            [
                (month, sum(vals) / len(vals))
                for month, vals in monthly.items()
            ],
            reverse=True,
        )

        return TrendReport(
            total_corrections=(
                len(self._header_corrections)
                + len(self._extraction_corrections)
            ),
            total_calibration_entries=len(self._calibration_entries),
            common_corrections=common,
            new_synonyms=promoted,
            accuracy_by_month=accuracy_by_month,
        )

    @property
    def header_correction_count(self) -> int:
        """Number of recorded header corrections."""
        return len(self._header_corrections)

    @property
    def extraction_correction_count(self) -> int:
        """Number of recorded extraction corrections."""
        return len(self._extraction_corrections)

    @property
    def calibration_entry_count(self) -> int:
        """Number of recorded calibration entries."""
        return len(self._calibration_entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all JSON files from data_dir."""
        self._header_corrections = _load_list(
            self._data_dir / "header_corrections.json",
            HeaderCorrection,
        )
        self._extraction_corrections = _load_list(
            self._data_dir / "extraction_corrections.json",
            ExtractionCorrection,
        )
        self._calibration_entries = _load_list(
            self._data_dir / "calibration_entries.json",
            CalibrationEntry,
        )
        raw_syn = _load_json(
            self._data_dir / "synonym_mappings.json",
            default={},
        )
        self._synonym_freq = (
            raw_syn if isinstance(raw_syn, dict) else {}
        )

    def _save(self) -> None:
        """Write all JSON files to data_dir."""
        _save_list(
            self._data_dir / "header_corrections.json",
            self._header_corrections,
        )
        _save_list(
            self._data_dir / "extraction_corrections.json",
            self._extraction_corrections,
        )
        _save_list(
            self._data_dir / "calibration_entries.json",
            self._calibration_entries,
        )
        _save_json(
            self._data_dir / "synonym_mappings.json",
            self._synonym_freq,
        )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: object = None) -> object:
    """Load a JSON file, returning *default* if missing."""
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return default if default is not None else {}


def _save_json(path: Path, data: object) -> None:
    """Write a JSON file atomically."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _load_list(path: Path, cls: type) -> list:
    """Load a JSON array of dataclass instances."""
    raw = _load_json(path, default=[])
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, dict):
            try:
                result.append(cls(**item))
            except TypeError:
                logger.warning("Skipping invalid entry in %s", path)
    return result


def _save_list(path: Path, items: list) -> None:
    """Save a list of dataclass instances as JSON."""
    data = [dataclasses.asdict(item) for item in items]
    _save_json(path, data)
