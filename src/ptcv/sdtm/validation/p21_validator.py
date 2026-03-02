"""Pinnacle 21 Community validator — Python re-implementation (PTCV-23).

Applies a subset of the 115 Pinnacle 21 FDA validation rules to SDTM
trial design domains (TS, TA, TE, TV, TI). Each rule is implemented as
a method returning zero or more P21Issue records.

Severity levels:
  Error   — Must fix before FDA submission.
  Warning — Should fix; reviewer will query.
  Notice  — Informational; best-practice suggestion.

Risk tier: MEDIUM — regulatory validation; no patient data.

Regulatory references:
- Pinnacle 21 Community (P21) rule set (115 rules for FDA NDA/BLA)
- SDTMIG v3.3: variable and domain specification
- CDISC SDTM v1.7: variable naming conventions
- FDA Study Data Technical Conformance Guide v5.0
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

from .models import P21Issue

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Required variable sets per domain (SDTMIG v3.3 §7.4)
# ---------------------------------------------------------------------------

_REQUIRED_VARS: dict[str, list[str]] = {
    "TS": ["STUDYID", "DOMAIN", "TSSEQ", "TSPARMCD", "TSPARM", "TSVAL"],
    "TA": ["STUDYID", "DOMAIN", "ARMCD", "ARM", "TAETORD", "ETCD", "ELEMENT"],
    "TE": ["STUDYID", "DOMAIN", "ETCD", "ELEMENT"],
    "TV": ["STUDYID", "DOMAIN", "VISITNUM", "VISIT"],
    "TI": ["STUDYID", "DOMAIN", "IETESTCD", "IETEST", "IECAT"],
}

# Variable length limits per SDTMIG v3.3
_VAR_MAX_LEN: dict[str, dict[str, int]] = {
    "STUDYID": {"TS": 20, "TA": 20, "TE": 20, "TV": 20, "TI": 20},
    "DOMAIN": {"TS": 2, "TA": 2, "TE": 2, "TV": 2, "TI": 2},
    "TSPARMCD": {"TS": 8},
    "TSPARM": {"TS": 40},
    "TSVAL": {"TS": 200},
    "TSVALCD": {"TS": 20},
    "ARMCD": {"TA": 20},
    "ARM": {"TA": 40},
    "ETCD": {"TA": 8, "TE": 8},
    "ELEMENT": {"TA": 40, "TE": 40},
    "TABRANCH": {"TA": 200},
    "TATRANS": {"TA": 200},
    "EPOCH": {"TA": 40},
    "TESTRL": {"TE": 200},
    "TEENRL": {"TE": 200},
    "TEDUR": {"TE": 20},
    "VISIT": {"TV": 40},
    "TVSTRL": {"TV": 8},
    "TVENRL": {"TV": 8},
    "TVTIMY": {"TV": 8},
    "TVTIM": {"TV": 20},
    "TVENDY": {"TV": 20},
    "IETESTCD": {"TI": 8},
    "IETEST": {"TI": 200},
    "IECAT": {"TI": 20},
    "IESCAT": {"TI": 40},
    "TIRL": {"TI": 200},
    "TIVERS": {"TI": 20},
}

# SDTMIG section references for remediation guidance
_SDTMIG_REFS: dict[str, str] = {
    "missing_required": "SDTMIG v3.3 §7.4 — Required variables must be present",
    "studyid_length": (
        "SDTMIG v3.3 §4.1 — STUDYID must not exceed 20 characters"
    ),
    "domain_value": "SDTMIG v3.3 §4.1 — DOMAIN must match dataset name (2 chars)",
    "domain_length": (
        "SDTMIG v3.3 §4.1 — DOMAIN must be exactly 2 characters"
    ),
    "var_length": "SDTMIG v3.3 §4.1 — Variable length must not exceed the SDTMIG maximum",
    "seq_positive": (
        "SDTMIG v3.3 §7.4 — Sequence numbers must be positive integers"
    ),
    "seq_consecutive": (
        "SDTMIG v3.3 §7.4 — Sequence numbers must be consecutive and unique"
    ),
    "tsparmcd_alpha": (
        "SDTMIG v3.3 §7.4.1 — TSPARMCD must contain only alphanumeric characters"
    ),
    "tsval_present": (
        "SDTMIG v3.3 §7.4.1 — TSVAL must not be blank unless TSVALNF is populated"
    ),
    "taetord_positive": (
        "SDTMIG v3.3 §7.4.2 — TAETORD must be a positive integer"
    ),
    "taetord_unique_per_arm": (
        "SDTMIG v3.3 §7.4.2 — TAETORD must be unique within each ARMCD"
    ),
    "visitnum_positive": (
        "SDTMIG v3.3 §7.4.4 — VISITNUM must be a positive number"
    ),
    "visitnum_unique": (
        "SDTMIG v3.3 §7.4.4 — VISITNUM must be unique within the domain"
    ),
    "iecat_value": (
        "SDTMIG v3.3 §7.4.5 — IECAT must be INCLUSION or EXCLUSION"
    ),
    "ietestcd_format": (
        "SDTMIG v3.3 §7.4.5 — IETESTCD must be alphanumeric, max 8 chars"
    ),
    "empty_dataset": "SDTMIG v3.3 §4.1 — Dataset must contain at least one record",
    "etcd_unique": "SDTMIG v3.3 §7.4.3 — ETCD must be unique within TE domain",
}


def _rule_id(domain: str, num: int) -> str:
    return f"P21-{domain}-{num:03d}"


class P21Validator:
    """Applies Pinnacle 21 Community rules to SDTM trial design domains.

    Args:
        domains: Mapping from domain name (e.g. "TS") to DataFrame.
    """

    def __init__(self, domains: dict[str, pd.DataFrame]) -> None:
        self._domains = {k.upper(): v for k, v in domains.items()}

    def validate(self) -> list[P21Issue]:
        """Run all validation rules and return collected findings.

        Returns:
            List of P21Issue sorted by severity (Error first) then domain.
        """
        issues: list[P21Issue] = []
        for domain in ["TS", "TA", "TE", "TV", "TI"]:
            if domain not in self._domains:
                continue
            df = self._domains[domain]
            issues.extend(self._check_empty(domain, df))
            issues.extend(self._check_required_vars(domain, df))
            issues.extend(self._check_domain_field(domain, df))
            issues.extend(self._check_studyid(domain, df))
            issues.extend(self._check_var_lengths(domain, df))

        if "TS" in self._domains:
            issues.extend(self._check_ts(self._domains["TS"]))
        if "TA" in self._domains:
            issues.extend(self._check_ta(self._domains["TA"]))
        if "TE" in self._domains:
            issues.extend(self._check_te(self._domains["TE"]))
        if "TV" in self._domains:
            issues.extend(self._check_tv(self._domains["TV"]))
        if "TI" in self._domains:
            issues.extend(self._check_ti(self._domains["TI"]))

        # Sort: Errors first, then Warning, then Notice
        severity_order = {"Error": 0, "Warning": 1, "Notice": 2}
        issues.sort(key=lambda i: (severity_order.get(i.severity, 9), i.domain))
        return issues

    # ------------------------------------------------------------------
    # Cross-domain rules
    # ------------------------------------------------------------------

    def _check_empty(self, domain: str, df: pd.DataFrame) -> list[P21Issue]:
        if len(df) == 0:
            return [
                P21Issue(
                    rule_id=_rule_id(domain, 1),
                    severity="Error",
                    domain=domain,
                    variable="",
                    description=f"Dataset {domain} contains no records.",
                    remediation_guidance=_SDTMIG_REFS["empty_dataset"],
                )
            ]
        return []

    def _check_required_vars(
        self, domain: str, df: pd.DataFrame
    ) -> list[P21Issue]:
        issues: list[P21Issue] = []
        for var in _REQUIRED_VARS.get(domain, []):
            if var not in df.columns:
                issues.append(
                    P21Issue(
                        rule_id=_rule_id(domain, 2),
                        severity="Error",
                        domain=domain,
                        variable=var,
                        description=(
                            f"Required variable {var} is missing from {domain}."
                        ),
                        remediation_guidance=_SDTMIG_REFS["missing_required"],
                    )
                )
        return issues

    def _check_domain_field(
        self, domain: str, df: pd.DataFrame
    ) -> list[P21Issue]:
        issues: list[P21Issue] = []
        if "DOMAIN" not in df.columns:
            return issues
        wrong = df[df["DOMAIN"].astype(str).str.strip() != domain]
        if not wrong.empty:
            issues.append(
                P21Issue(
                    rule_id=_rule_id(domain, 3),
                    severity="Error",
                    domain=domain,
                    variable="DOMAIN",
                    description=(
                        f"DOMAIN field value does not match dataset name '{domain}'. "
                        f"Found: {wrong['DOMAIN'].iloc[0]!r}"
                    ),
                    remediation_guidance=_SDTMIG_REFS["domain_value"],
                )
            )
        too_long = df[df["DOMAIN"].astype(str).str.len() > 2]
        if not too_long.empty:
            issues.append(
                P21Issue(
                    rule_id=_rule_id(domain, 4),
                    severity="Error",
                    domain=domain,
                    variable="DOMAIN",
                    description="DOMAIN value exceeds 2 characters.",
                    remediation_guidance=_SDTMIG_REFS["domain_length"],
                )
            )
        return issues

    def _check_studyid(
        self, domain: str, df: pd.DataFrame
    ) -> list[P21Issue]:
        issues: list[P21Issue] = []
        if "STUDYID" not in df.columns:
            return issues
        too_long = df[df["STUDYID"].astype(str).str.len() > 20]
        if not too_long.empty:
            issues.append(
                P21Issue(
                    rule_id=_rule_id(domain, 5),
                    severity="Error",
                    domain=domain,
                    variable="STUDYID",
                    description=(
                        "STUDYID exceeds 20 characters: "
                        f"{too_long['STUDYID'].iloc[0]!r}"
                    ),
                    remediation_guidance=_SDTMIG_REFS["studyid_length"],
                )
            )
        return issues

    def _check_var_lengths(
        self, domain: str, df: pd.DataFrame
    ) -> list[P21Issue]:
        issues: list[P21Issue] = []
        for var, domain_limits in _VAR_MAX_LEN.items():
            if domain not in domain_limits:
                continue
            max_len = domain_limits[domain]
            if var not in df.columns:
                continue
            col = df[var].astype(str)
            violators = df[col.str.len() > max_len]
            if not violators.empty:
                issues.append(
                    P21Issue(
                        rule_id=_rule_id(domain, 6),
                        severity="Warning",
                        domain=domain,
                        variable=var,
                        description=(
                            f"{var} exceeds SDTMIG maximum length of {max_len}. "
                            f"Found length: {col.str.len().max()}."
                        ),
                        remediation_guidance=_SDTMIG_REFS["var_length"],
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # TS-specific rules
    # ------------------------------------------------------------------

    def _check_ts(self, df: pd.DataFrame) -> list[P21Issue]:
        issues: list[P21Issue] = []
        if len(df) == 0:
            return issues

        # P21-TS-010: TSSEQ must be positive
        if "TSSEQ" in df.columns:
            neg_seq = df[df["TSSEQ"].apply(
                lambda v: pd.to_numeric(v, errors="coerce")
            ) <= 0]
            if not neg_seq.empty:
                issues.append(P21Issue(
                    rule_id=_rule_id("TS", 10),
                    severity="Error",
                    domain="TS",
                    variable="TSSEQ",
                    description="TSSEQ must be a positive number.",
                    remediation_guidance=_SDTMIG_REFS["seq_positive"],
                ))

        # P21-TS-011: TSPARMCD must be alphanumeric
        if "TSPARMCD" in df.columns:
            bad_parmcd = df[
                ~df["TSPARMCD"].astype(str).str.match(r"^[A-Za-z0-9]+$")
            ]
            if not bad_parmcd.empty:
                issues.append(P21Issue(
                    rule_id=_rule_id("TS", 11),
                    severity="Error",
                    domain="TS",
                    variable="TSPARMCD",
                    description=(
                        "TSPARMCD contains non-alphanumeric characters: "
                        f"{bad_parmcd['TSPARMCD'].iloc[0]!r}"
                    ),
                    remediation_guidance=_SDTMIG_REFS["tsparmcd_alpha"],
                ))

        # P21-TS-012: TSVAL must not be blank unless TSVALNF is populated
        if "TSVAL" in df.columns:
            if "TSVALNF" in df.columns:
                blank_val = df[
                    (df["TSVAL"].astype(str).str.strip() == "")
                    & (df["TSVALNF"].astype(str).str.strip() == "")
                ]
            else:
                blank_val = df[df["TSVAL"].astype(str).str.strip() == ""]
            if not blank_val.empty:
                issues.append(P21Issue(
                    rule_id=_rule_id("TS", 12),
                    severity="Warning",
                    domain="TS",
                    variable="TSVAL",
                    description=(
                        f"TSVAL is blank for {len(blank_val)} row(s) "
                        "without a TSVALNF value."
                    ),
                    remediation_guidance=_SDTMIG_REFS["tsval_present"],
                ))

        # P21-TS-013: TSSEQ must be consecutive
        if "TSSEQ" in df.columns:
            seqs = sorted(
                df["TSSEQ"].dropna().apply(
                    lambda v: pd.to_numeric(v, errors="coerce")
                ).dropna().astype(int).tolist()
            )
            expected = list(range(1, len(seqs) + 1))
            if seqs != expected:
                issues.append(P21Issue(
                    rule_id=_rule_id("TS", 13),
                    severity="Warning",
                    domain="TS",
                    variable="TSSEQ",
                    description=(
                        "TSSEQ values are not consecutive from 1."
                    ),
                    remediation_guidance=_SDTMIG_REFS["seq_consecutive"],
                ))

        return issues

    # ------------------------------------------------------------------
    # TA-specific rules
    # ------------------------------------------------------------------

    def _check_ta(self, df: pd.DataFrame) -> list[P21Issue]:
        issues: list[P21Issue] = []
        if len(df) == 0:
            return issues

        # P21-TA-010: TAETORD must be positive
        if "TAETORD" in df.columns:
            neg = df[
                df["TAETORD"].apply(
                    lambda v: pd.to_numeric(v, errors="coerce")
                ) <= 0
            ]
            if not neg.empty:
                issues.append(P21Issue(
                    rule_id=_rule_id("TA", 10),
                    severity="Error",
                    domain="TA",
                    variable="TAETORD",
                    description="TAETORD must be a positive integer.",
                    remediation_guidance=_SDTMIG_REFS["taetord_positive"],
                ))

        # P21-TA-011: TAETORD must be unique within ARMCD
        if "TAETORD" in df.columns and "ARMCD" in df.columns:
            dups = df.duplicated(subset=["ARMCD", "TAETORD"], keep=False)
            if dups.any():
                issues.append(P21Issue(
                    rule_id=_rule_id("TA", 11),
                    severity="Error",
                    domain="TA",
                    variable="TAETORD",
                    description=(
                        "Duplicate TAETORD values within the same ARMCD."
                    ),
                    remediation_guidance=_SDTMIG_REFS["taetord_unique_per_arm"],
                ))

        return issues

    # ------------------------------------------------------------------
    # TE-specific rules
    # ------------------------------------------------------------------

    def _check_te(self, df: pd.DataFrame) -> list[P21Issue]:
        issues: list[P21Issue] = []
        if len(df) == 0:
            return issues

        # P21-TE-010: ETCD must be unique
        if "ETCD" in df.columns:
            dups = df.duplicated(subset=["ETCD"], keep=False)
            if dups.any():
                issues.append(P21Issue(
                    rule_id=_rule_id("TE", 10),
                    severity="Error",
                    domain="TE",
                    variable="ETCD",
                    description="ETCD values must be unique within TE.",
                    remediation_guidance=_SDTMIG_REFS["etcd_unique"],
                ))

        # P21-TE-011: ETCD must be alphanumeric
        if "ETCD" in df.columns:
            bad = df[
                ~df["ETCD"].astype(str).str.match(r"^[A-Za-z0-9]+$")
            ]
            if not bad.empty:
                issues.append(P21Issue(
                    rule_id=_rule_id("TE", 11),
                    severity="Error",
                    domain="TE",
                    variable="ETCD",
                    description=(
                        "ETCD must be alphanumeric: "
                        f"{bad['ETCD'].iloc[0]!r}"
                    ),
                    remediation_guidance=(
                        "SDTMIG v3.3 §7.4.3 — ETCD must be alphanumeric, max 8 chars"
                    ),
                ))

        return issues

    # ------------------------------------------------------------------
    # TV-specific rules
    # ------------------------------------------------------------------

    def _check_tv(self, df: pd.DataFrame) -> list[P21Issue]:
        issues: list[P21Issue] = []
        if len(df) == 0:
            return issues

        # P21-TV-010: VISITNUM must be positive
        if "VISITNUM" in df.columns:
            neg = df[
                df["VISITNUM"].apply(
                    lambda v: pd.to_numeric(v, errors="coerce")
                ) <= 0
            ]
            if not neg.empty:
                issues.append(P21Issue(
                    rule_id=_rule_id("TV", 10),
                    severity="Error",
                    domain="TV",
                    variable="VISITNUM",
                    description="VISITNUM must be a positive number.",
                    remediation_guidance=_SDTMIG_REFS["visitnum_positive"],
                ))

        # P21-TV-011: VISITNUM must be unique
        if "VISITNUM" in df.columns:
            dups = df.duplicated(subset=["VISITNUM"], keep=False)
            if dups.any():
                issues.append(P21Issue(
                    rule_id=_rule_id("TV", 11),
                    severity="Error",
                    domain="TV",
                    variable="VISITNUM",
                    description="Duplicate VISITNUM values found in TV.",
                    remediation_guidance=_SDTMIG_REFS["visitnum_unique"],
                ))

        return issues

    # ------------------------------------------------------------------
    # TI-specific rules
    # ------------------------------------------------------------------

    def _check_ti(self, df: pd.DataFrame) -> list[P21Issue]:
        issues: list[P21Issue] = []
        if len(df) == 0:
            return issues

        # P21-TI-010: IECAT must be INCLUSION or EXCLUSION
        if "IECAT" in df.columns:
            valid_cats = {"INCLUSION", "EXCLUSION"}
            invalid = df[~df["IECAT"].astype(str).str.upper().isin(valid_cats)]
            if not invalid.empty:
                issues.append(P21Issue(
                    rule_id=_rule_id("TI", 10),
                    severity="Error",
                    domain="TI",
                    variable="IECAT",
                    description=(
                        "IECAT must be INCLUSION or EXCLUSION. "
                        f"Found: {invalid['IECAT'].iloc[0]!r}"
                    ),
                    remediation_guidance=_SDTMIG_REFS["iecat_value"],
                ))

        # P21-TI-011: IETESTCD must be alphanumeric and ≤ 8 chars
        if "IETESTCD" in df.columns:
            bad = df[
                ~df["IETESTCD"].astype(str).str.match(r"^[A-Za-z0-9]{1,8}$")
            ]
            if not bad.empty:
                issues.append(P21Issue(
                    rule_id=_rule_id("TI", 11),
                    severity="Error",
                    domain="TI",
                    variable="IETESTCD",
                    description=(
                        "IETESTCD must be alphanumeric and ≤ 8 chars. "
                        f"Found: {bad['IETESTCD'].iloc[0]!r}"
                    ),
                    remediation_guidance=_SDTMIG_REFS["ietestcd_format"],
                ))

        return issues
