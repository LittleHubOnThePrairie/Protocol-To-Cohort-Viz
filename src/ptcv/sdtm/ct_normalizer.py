"""CDISC Controlled Terminology (CT) normalizer for SDTM trial design domains.

Maps free-text protocol values to CDISC NCI Thesaurus codes using a static
lookup table that covers common TS parameter values. Terms absent from the
table are written to the CT review queue for human review.

In production, the CDISC Library API (library.cdisc.org) should be consulted
for exhaustive coverage. This implementation is intentionally offline-capable.

Risk tier: MEDIUM — data pipeline component; affects regulatory submission.

Regulatory references:
- SDTM v1.7 Section 4.1.2.4: Controlled Terminology requirements
- CDISC NCI EVS subset for SDTM Trial Design domains
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Static CDISC CT lookup table
# ---------------------------------------------------------------------------
# Key: (TSPARMCD_upper, value_upper)  →  (NCI_code, TSVALCD)
# Sourced from CDISC SDTM Controlled Terminology 2023-12-15

_TS_PHASE_CT: dict[str, tuple[str, str]] = {
    "PHASE 0": ("C54723", "PHASE0"),
    "PHASE I": ("C15720", "PHASE1"),
    "PHASE 1": ("C15720", "PHASE1"),
    "PHASE I/II": ("C15960", "PHASE1-2"),
    "PHASE I/IIA": ("C15960", "PHASE1-2"),
    "PHASE II": ("C15961", "PHASE2"),
    "PHASE 2": ("C15961", "PHASE2"),
    "PHASE IIA": ("C15961", "PHASE2A"),
    "PHASE IIB": ("C15961", "PHASE2B"),
    "PHASE II/III": ("C49685", "PHASE2-3"),
    "PHASE III": ("C15962", "PHASE3"),
    "PHASE 3": ("C15962", "PHASE3"),
    "PHASE IIIA": ("C15962", "PHASE3A"),
    "PHASE IIIB": ("C15962", "PHASE3B"),
    "PHASE IV": ("C15963", "PHASE4"),
    "PHASE 4": ("C15963", "PHASE4"),
    "NOT APPLICABLE": ("C48660", "NA"),
}

_TS_DESIGN_CT: dict[str, tuple[str, str]] = {
    "PARALLEL": ("C46110", "PARALLEL"),
    "PARALLEL GROUP": ("C46110", "PARALLEL"),
    "CROSSOVER": ("C16066", "CROSSOVER"),
    "FACTORIAL": ("C82635", "FACTORIAL"),
    "SINGLE GROUP": ("C82539", "SINGLE GROUP"),
    "SINGLE ARM": ("C82539", "SINGLE GROUP"),
    "SEQUENTIAL": ("C3161", "SEQUENTIAL"),
}

_TS_TYPE_CT: dict[str, tuple[str, str]] = {
    "INTERVENTIONAL": ("C98388", "INTERVENTIONAL"),
    "OBSERVATIONAL": ("C16084", "OBSERVATIONAL"),
    "EXPANDED ACCESS": ("C156729", "EXPANDED ACCESS"),
}

_TS_BLINDING_CT: dict[str, tuple[str, str]] = {
    "DOUBLE-BLIND": ("C15228", "DOUBLE-BLIND"),
    "DOUBLE BLIND": ("C15228", "DOUBLE-BLIND"),
    "SINGLE-BLIND": ("C15229", "SINGLE-BLIND"),
    "SINGLE BLIND": ("C15229", "SINGLE-BLIND"),
    "OPEN-LABEL": ("C49659", "OPEN-LABEL"),
    "OPEN LABEL": ("C49659", "OPEN-LABEL"),
    "BLINDED": ("C15228", "DOUBLE-BLIND"),
    "UNBLINDED": ("C49659", "OPEN-LABEL"),
    "TRIPLE-BLIND": ("C49656", "TRIPLE-BLIND"),
}

# Combined lookup keyed by (TSPARMCD, value_upper) → (code, tsvalcd)
_CT_BY_PARM: dict[str, dict[str, tuple[str, str]]] = {
    "PHASE": _TS_PHASE_CT,
    "STYPE": _TS_DESIGN_CT,    # study type (design)
    "TTYPE": _TS_TYPE_CT,      # trial type
    "BLIND": _TS_BLINDING_CT,  # blinding
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CtLookupResult:
    """Result of a single CT normalisation call.

    Attributes:
        original_value: Raw value before normalisation.
        tsval: Normalised display value for TSVAL (may equal original if mapped).
        tsvalcd: CT term code (e.g. "PHASE3"), empty when unmapped.
        nci_code: NCI thesaurus C-code (e.g. "C15962"), empty when unmapped.
        mapped: True when an exact CT match was found.
    """

    original_value: str
    tsval: str
    tsvalcd: str
    nci_code: str
    mapped: bool


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class CtNormalizer:
    """Maps raw protocol text values to CDISC Controlled Terminology codes.

    Uses a static lookup table for offline operation. Unmapped values are
    returned with ``mapped=False`` so the caller can route them to the CT
    review queue.

    Args:
        placeholder_prefix: Prefix inserted before unmapped values in TSVAL.
            Default "" — unmapped values are returned as-is in TSVAL.
    """

    def __init__(self, placeholder_prefix: str = "") -> None:
        self._placeholder_prefix = placeholder_prefix

    def normalize(
        self,
        parmcd: str,
        value: str,
    ) -> CtLookupResult:
        """Normalise one TSVAL against the CT table for the given TSPARMCD.

        Args:
            parmcd: SDTM TSPARMCD (e.g. "PHASE", "TTYPE").
            value: Raw string value to normalise.

        Returns:
            CtLookupResult. When not mapped, tsvalcd and nci_code are "".
        """
        domain_table = _CT_BY_PARM.get(parmcd.upper())
        value_upper = value.strip().upper()

        if domain_table is not None:
            result = domain_table.get(value_upper)
            if result:
                return CtLookupResult(
                    original_value=value,
                    tsval=value.strip(),
                    tsvalcd=result[1],
                    nci_code=result[0],
                    mapped=True,
                )
            # Try prefix matching (e.g. "Phase III randomised" → "PHASE III")
            for key, (code, tsvalcd) in domain_table.items():
                if value_upper.startswith(key):
                    return CtLookupResult(
                        original_value=value,
                        tsval=value.strip(),
                        tsvalcd=tsvalcd,
                        nci_code=code,
                        mapped=True,
                    )

        # Not found — return as-is with empty codes
        tsval = (
            f"{self._placeholder_prefix}{value.strip()}"
            if self._placeholder_prefix
            else value.strip()
        )
        return CtLookupResult(
            original_value=value,
            tsval=tsval,
            tsvalcd="",
            nci_code="",
            mapped=False,
        )

    def normalize_phase(self, text: str) -> CtLookupResult:
        """Convenience wrapper: extract and normalise trial phase from text.

        Scans ``text`` for phase patterns (Phase I, Phase II/III, etc.) and
        returns the first match normalised against the PHASE CT table.

        Args:
            text: Free-text section content to scan.

        Returns:
            CtLookupResult. Returns unmapped result if no phase found.
        """
        pattern = re.compile(
            r"phase\s+(?:I{1,3}V?|IV|[1-4](?:/(?:I{1,3}V?|[1-4]))?|"
            r"I{1,3}[AB]?|[1-4][AB]?)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            raw = match.group(0).strip()
            return self.normalize("PHASE", raw)
        return CtLookupResult(
            original_value=text,
            tsval="",
            tsvalcd="",
            nci_code="",
            mapped=False,
        )
