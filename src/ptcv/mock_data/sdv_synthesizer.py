"""SDV synthesizer service for mock SDTM data generation (PTCV-119).

Wraps SDV synthesizer models to generate realistic mock SDTM
observation-class data (DM, VS, LB, AE, EG, CM) from the SDTM
variable metadata registry.

Generation workflow per domain:
  1. Build SDV metadata via ``sdv_adapter.build_sdv_metadata``
  2. Create a seed DataFrame with physiologically plausible values
  3. Fit ``GaussianCopulaSynthesizer`` (or ``CTGANSynthesizer``)
  4. Sample ``num_subjects * records_per_subject`` rows
  5. Post-process: enforce SDTM column ordering, apply CT codelists

SDV is an **optional** dependency (requires PyTorch).

Risk tier: LOW — synthetic data only; no real patient data.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .sdtm_metadata import SdtmDomainSpec, SdtmVariableSpec, get_domain_spec
from .sdv_adapter import SdvAdapterConfig, _check_sdv, build_sdv_metadata


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SdvConfig:
    """Configuration for SDV-based mock data generation.

    Attributes:
        synthesizer_type: ``"gaussian_copula"`` (fast) or
            ``"ctgan"`` (slower, better categorical fidelity).
        random_seed: Seed for reproducibility.
        num_subjects: Number of synthetic subjects.
        records_per_subject_range: ``(min, max)`` records per
            subject for observation-class domains (VS, LB, etc.).
            DM always produces exactly 1 row per subject.
    """

    synthesizer_type: str = "gaussian_copula"
    random_seed: int = 42
    num_subjects: int = 10
    records_per_subject_range: tuple[int, int] = (3, 8)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MockGenerationResult:
    """Result of mock SDTM data generation.

    Attributes:
        domain_dataframes: Mapping of domain code to DataFrame.
        generation_timestamp: ISO 8601 UTC timestamp.
        config_used: Config that produced this result.
        run_id: UUID for traceability.
    """

    domain_dataframes: dict[str, pd.DataFrame]
    generation_timestamp: str
    config_used: SdvConfig
    run_id: str


# ---------------------------------------------------------------------------
# Seed data builders — clinically plausible distributions
# ---------------------------------------------------------------------------

# STUDYID placeholder used in all seed data.
_STUDYID = "MOCK-001"


def _build_seed_dm(
    rng: np.random.Generator,
    n: int,
    studyid: str,
) -> pd.DataFrame:
    """Build a seed DM DataFrame with plausible demographics."""
    rows: list[dict[str, Any]] = []
    for i in range(n):
        subjid = f"{i + 1:04d}"
        age = int(np.clip(rng.normal(55, 12), 18, 100))
        sex = rng.choice(["M", "F"])
        race = rng.choice([
            "WHITE", "BLACK OR AFRICAN AMERICAN", "ASIAN",
        ], p=[0.6, 0.25, 0.15])
        rows.append({
            "STUDYID": studyid,
            "DOMAIN": "DM",
            "USUBJID": f"{studyid}-{subjid}",
            "SUBJID": subjid,
            "RFSTDTC": "2024-01-15T08:00:00",
            "RFENDTC": "2024-06-15T17:00:00",
            "SITEID": f"SITE-{rng.integers(1, 10):02d}",
            "AGE": float(age),
            "AGEU": "YEARS",
            "SEX": str(sex),
            "RACE": str(race),
            "ETHNIC": rng.choice([
                "HISPANIC OR LATINO", "NOT HISPANIC OR LATINO",
            ], p=[0.15, 0.85]),
            "ARMCD": "ARM01",
            "ARM": "Treatment",
            "COUNTRY": str(rng.choice(["USA", "CAN", "GBR"])),
            "DMDTC": "2024-01-15",
            "DMDY": 1.0,
        })
    return pd.DataFrame(rows)


def _build_seed_observations(
    rng: np.random.Generator,
    domain_spec: SdtmDomainSpec,
    usubjids: list[str],
    records_per_subject: int,
    studyid: str,
) -> pd.DataFrame:
    """Build seed data for an observation-class domain."""
    rows: list[dict[str, Any]] = []
    seq = 0

    for usubjid in usubjids:
        for rec in range(records_per_subject):
            seq += 1
            row: dict[str, Any] = {}
            for var in domain_spec.variables:
                row[var.name] = _generate_value(
                    var, rng, studyid, usubjid, seq, rec,
                )
            rows.append(row)

    return pd.DataFrame(rows)


def _generate_value(
    var: SdtmVariableSpec,
    rng: np.random.Generator,
    studyid: str,
    usubjid: str,
    seq: int,
    record_idx: int,
) -> Any:
    """Generate a plausible value for a single variable."""
    name = var.name

    # Fixed values.
    if name == "STUDYID":
        return studyid
    if name == "DOMAIN":
        return var.codelist and next(iter(var.codelist)) or ""
    if name == "USUBJID":
        return usubjid

    # Sequence numbers.
    if name.endswith("SEQ"):
        return float(seq)

    # Date columns.
    if name.endswith("DTC") or name.endswith("DT"):
        day_offset = record_idx * 14
        return f"2024-{1 + day_offset // 30:02d}-{1 + day_offset % 28:02d}"

    # Study day columns.
    if name.endswith("DY") or name.endswith("STDY") or name.endswith("ENDY"):
        return float(1 + record_idx * 14)

    # Visit-related.
    if name == "VISITNUM":
        return float(record_idx + 1)
    if name == "VISIT":
        return f"Visit {record_idx + 1}"

    # Numeric with range.
    if var.type == "Num":
        lo = var.range_min if var.range_min is not None else 0.0
        hi = var.range_max if var.range_max is not None else 100.0
        mean = (lo + hi) / 2.0
        std = (hi - lo) / 6.0
        val = float(np.clip(rng.normal(mean, max(std, 0.1)), lo, hi))
        return round(val, 2)

    # Categorical with codelist.
    if var.codelist:
        values = sorted(var.codelist)
        return str(rng.choice(values))

    # Generic char.
    return f"{name}_{seq}"


# ---------------------------------------------------------------------------
# Synthesizer service
# ---------------------------------------------------------------------------


class SdvSynthesizerService:
    """Generate mock SDTM data using SDV synthesizers.

    Args:
        config: Generation configuration.
    """

    def __init__(self, config: SdvConfig | None = None) -> None:
        self._config = config or SdvConfig()

    @property
    def config(self) -> SdvConfig:
        return self._config

    def generate_domain(
        self,
        domain_spec: SdtmDomainSpec,
        num_subjects: int | None = None,
        num_records_per_subject: int | None = None,
    ) -> pd.DataFrame:
        """Generate mock data for a single SDTM domain.

        Args:
            domain_spec: Variable-level domain specification.
            num_subjects: Override ``config.num_subjects``.
            num_records_per_subject: Fixed records per subject.
                If *None*, uses a random value within
                ``config.records_per_subject_range``.

        Returns:
            DataFrame with SDTM-ordered columns and synthetic data.

        Raises:
            ImportError: If SDV is not installed.
        """
        _check_sdv()
        from sdv.single_table import GaussianCopulaSynthesizer

        cfg = self._config
        rng = np.random.default_rng(cfg.random_seed)
        n_subj = num_subjects or cfg.num_subjects
        studyid = _STUDYID

        # Build SDV metadata.
        sdv_meta = build_sdv_metadata(
            domain_spec,
            SdvAdapterConfig(
                synthesizer_type=cfg.synthesizer_type,
                enforce_codelists=True,
            ),
        )

        # Build seed data.
        usubjids = [
            f"{studyid}-{i + 1:04d}" for i in range(n_subj)
        ]

        if domain_spec.domain_code == "DM":
            seed_df = _build_seed_dm(rng, n_subj, studyid)
            n_rows = n_subj
        else:
            recs = num_records_per_subject
            if recs is None:
                lo, hi = cfg.records_per_subject_range
                recs = int(rng.integers(lo, hi + 1))
            seed_df = _build_seed_observations(
                rng, domain_spec, usubjids, recs, studyid,
            )
            n_rows = n_subj * recs

        # Fit synthesizer on seed data.
        synth = GaussianCopulaSynthesizer(
            sdv_meta,
            enforce_min_max_values=True,
            enforce_rounding=True,
        )
        synth.fit(seed_df)

        # Sample.
        sampled = synth.sample(
            num_rows=n_rows,
            output_file_path=None,
        )

        # Post-process: enforce SDTM column ordering.
        ordered_cols = [v.name for v in domain_spec.variables]
        present_cols = [
            c for c in ordered_cols if c in sampled.columns
        ]
        sampled = sampled[present_cols]

        # Enforce codelist values for categorical columns.
        for var in domain_spec.variables:
            if (
                var.codelist
                and var.name in sampled.columns
                and var.name != "USUBJID"
            ):
                valid = sorted(var.codelist)
                sampled[var.name] = sampled[var.name].apply(
                    lambda v, vl=valid: (  # type: ignore[misc]
                        v if v in vl else rng.choice(vl)
                    )
                )

        return sampled

    def generate_all(
        self,
        specs: dict[str, SdtmDomainSpec],
        num_subjects: int | None = None,
    ) -> MockGenerationResult:
        """Generate mock data for all specified domains.

        Creates a shared USUBJID pool so that all domains reference
        the same set of subjects.

        Args:
            specs: Mapping of domain code to ``SdtmDomainSpec``.
            num_subjects: Override ``config.num_subjects``.

        Returns:
            ``MockGenerationResult`` with all domain DataFrames.

        Raises:
            ImportError: If SDV is not installed.
        """
        _check_sdv()
        from sdv.single_table import GaussianCopulaSynthesizer

        cfg = self._config
        rng = np.random.default_rng(cfg.random_seed)
        n_subj = num_subjects or cfg.num_subjects
        studyid = _STUDYID

        # Shared USUBJID pool.
        usubjids = [
            f"{studyid}-{i + 1:04d}" for i in range(n_subj)
        ]

        result_dfs: dict[str, pd.DataFrame] = {}

        for domain_code, domain_spec in specs.items():
            sdv_meta = build_sdv_metadata(
                domain_spec,
                SdvAdapterConfig(
                    synthesizer_type=cfg.synthesizer_type,
                    enforce_codelists=True,
                ),
            )

            if domain_code == "DM":
                seed_df = _build_seed_dm(rng, n_subj, studyid)
                n_rows = n_subj
            else:
                lo, hi = cfg.records_per_subject_range
                recs = int(rng.integers(lo, hi + 1))
                seed_df = _build_seed_observations(
                    rng, domain_spec, usubjids, recs, studyid,
                )
                n_rows = n_subj * recs

            synth = GaussianCopulaSynthesizer(
                sdv_meta,
                enforce_min_max_values=True,
                enforce_rounding=True,
            )
            synth.fit(seed_df)

            sampled = synth.sample(
                num_rows=n_rows,
                output_file_path=None,
            )

            # Enforce column ordering.
            ordered_cols = [v.name for v in domain_spec.variables]
            present_cols = [
                c for c in ordered_cols if c in sampled.columns
            ]
            sampled = sampled[present_cols]

            # Enforce codelist values.
            for var in domain_spec.variables:
                if (
                    var.codelist
                    and var.name in sampled.columns
                    and var.name != "USUBJID"
                ):
                    valid = sorted(var.codelist)
                    sampled[var.name] = sampled[var.name].apply(
                        lambda v, vl=valid: (  # type: ignore[misc]
                            v if v in vl else rng.choice(vl)
                        )
                    )

            # Force shared USUBJIDs.
            if "USUBJID" in sampled.columns:
                sampled["USUBJID"] = [
                    usubjids[i % n_subj]
                    for i in range(len(sampled))
                ]

            result_dfs[domain_code] = sampled

        return MockGenerationResult(
            domain_dataframes=result_dfs,
            generation_timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            config_used=cfg,
            run_id=str(uuid.uuid4()),
        )
