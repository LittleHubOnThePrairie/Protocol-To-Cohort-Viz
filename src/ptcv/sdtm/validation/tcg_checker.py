"""FDA TCG v5.9 Appendix B completeness checker (PTCV-23).

Checks whether all required TS parameters from the FDA Technical
Conformance Guide v5.9 (October 2024), Appendix B are present in the
TS dataset. Produces a TcgParameter record for each required parameter
and a summary passed boolean.

Risk tier: MEDIUM — regulatory submission conformance check; no patient data.

Regulatory references:
- FDA Study Data Technical Conformance Guide v5.9 (October 2024),
  Appendix B: Required Trial Summary Parameters for NDA/BLA
- SDTMIG v3.3 §7.4.1: Trial Summary (TS) domain
"""

from __future__ import annotations

import pandas as pd

from .models import TcgParameter


# ---------------------------------------------------------------------------
# Required TS parameters per FDA TCG v5.9 Appendix B
# (subset of the full Appendix B list relevant to trial design domains)
# ---------------------------------------------------------------------------

_TCG_REQUIRED: list[tuple[str, str]] = [
    ("ACTSUB",   "Actual Subject Count"),
    ("ADDON",    "Add-on to Existing Treatments"),
    ("AGEMAX",   "Planned Maximum Age of Subjects"),
    ("AGEMIN",   "Planned Minimum Age of Subjects"),
    ("BLIND",    "Blinding Schema"),
    ("DCUTDESC", "Description of Data Cutoff"),
    ("DCUTDT",   "Data Cutoff Date"),
    ("INDIC",    "Trial Disease/Condition Indication"),
    ("INTMODEL", "Intervention Model"),
    ("INTTYPE",  "Intervention Type"),
    ("NARMS",    "Planned Number of Arms"),
    ("OBJPRIM",  "Primary Objective"),
    ("PCLAS",    "Pharmacological Class"),
    ("PHASE",    "Trial Phase"),
    ("PLANSUB",  "Planned Number of Subjects"),
    ("RANDOM",   "Randomization Indicator"),
    ("REGID",    "Registry Identifier"),
    ("SDTMVR",   "SDTM Version"),
    ("SPONSOR",  "Clinical Study Sponsor"),
    ("STOPRULE", "Study Stop Rules"),
    ("STYPE",    "Study Type"),
    ("TITLE",    "Trial Title"),
]

_REQUIRED_BY = "FDA TCG v5.9 Appendix B"


class TcgChecker:
    """Checks FDA TCG v5.9 Appendix B required TS parameters.

    Args:
        ts_df: The TS domain DataFrame (must contain TSPARMCD column).
    """

    def __init__(self, ts_df: pd.DataFrame) -> None:
        self._ts_df = ts_df

    def check(self) -> tuple[list[TcgParameter], bool, list[str]]:
        """Run the Appendix B completeness check.

        Returns:
            Tuple of:
              - List of TcgParameter for each required parameter.
              - Passed boolean (True iff all required parameters present).
              - List of missing TSPARMCD values.
        [PTCV-23 Scenario: FDA TCG Appendix B completeness check]
        """
        present_codes: set[str] = set()
        if "TSPARMCD" in self._ts_df.columns and len(self._ts_df) > 0:
            present_codes = set(
                self._ts_df["TSPARMCD"].astype(str).str.strip().str.upper()
            )

        parameters: list[TcgParameter] = []
        missing: list[str] = []

        for tsparmcd, tsparm in _TCG_REQUIRED:
            is_present = tsparmcd in present_codes
            if not is_present:
                missing.append(tsparmcd)
            parameters.append(
                TcgParameter(
                    tsparmcd=tsparmcd,
                    tsparm=tsparm,
                    required_by=_REQUIRED_BY,
                    present=is_present,
                    missing=not is_present,
                )
            )

        passed = len(missing) == 0
        return parameters, passed, missing
