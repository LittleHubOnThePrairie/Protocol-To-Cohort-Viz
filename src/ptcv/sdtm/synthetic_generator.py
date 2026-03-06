"""Synthetic SDTM data generator from SoA-derived schemas (PTCV-54).

Populates MockSdtmDataset schemas with synthetic patient-level records
for testing statistical models, validating SDTM structures, and
parameterising natural history cohort searches.

Generated domains:
  DM — Demographics (USUBJID, AGE, SEX, RACE, ARM assignment)
  SV — Subject Visits (per-subject visit instances with timing jitter)
  LB — Laboratory Test Results (clinically plausible synthetic values)

Risk tier: LOW — synthetic data only; no real patient data.

Regulatory references:
- SDTMIG v3.4 for domain variable naming
- CDISC Controlled Terminology for codelist values
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .soa_mapper import MockSdtmDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation.

    Attributes:
        n_subjects: Number of synthetic subjects to generate.
        randomization_ratio: Arm allocation ratios (e.g. [2, 1] for 2:1).
        dropout_rate: Fraction of subjects who drop out before study end.
        seed: Random seed for reproducibility.
        study_start_date: First subject's screening date.
    """

    n_subjects: int = 100
    randomization_ratio: list[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.0]
    )
    dropout_rate: float = 0.15
    seed: int = 42
    study_start_date: date = dataclasses.field(
        default_factory=lambda: date(2024, 6, 1)
    )


# ---------------------------------------------------------------------------
# Synthetic result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SyntheticSdtmResult:
    """Result of synthetic SDTM data generation.

    Attributes:
        dm: Demographics domain DataFrame.
        sv: Subject Visits domain DataFrame.
        lb: Laboratory Test Results domain DataFrame.
        config: Configuration used for generation.
        studyid: STUDYID value.
    """

    dm: pd.DataFrame
    sv: pd.DataFrame
    lb: pd.DataFrame
    config: SyntheticConfig
    studyid: str

    @property
    def domains(self) -> dict[str, pd.DataFrame]:
        """Return all generated domain DataFrames."""
        return {"DM": self.dm, "SV": self.sv, "LB": self.lb}


# ---------------------------------------------------------------------------
# Lab test reference ranges
# ---------------------------------------------------------------------------

_LAB_TESTS: list[dict[str, Any]] = [
    {"LBTESTCD": "WBC", "LBTEST": "White Blood Cell Count",
     "LBORRESU": "10^9/L", "mean": 7.0, "std": 2.0, "low": 4.0, "high": 11.0},
    {"LBTESTCD": "RBC", "LBTEST": "Red Blood Cell Count",
     "LBORRESU": "10^12/L", "mean": 4.8, "std": 0.5, "low": 4.0, "high": 5.5},
    {"LBTESTCD": "HGB", "LBTEST": "Hemoglobin",
     "LBORRESU": "g/dL", "mean": 14.0, "std": 1.5, "low": 12.0, "high": 17.0},
    {"LBTESTCD": "PLT", "LBTEST": "Platelet Count",
     "LBORRESU": "10^9/L", "mean": 250.0, "std": 60.0, "low": 150.0, "high": 400.0},
    {"LBTESTCD": "ALT", "LBTEST": "Alanine Aminotransferase",
     "LBORRESU": "U/L", "mean": 25.0, "std": 10.0, "low": 7.0, "high": 56.0},
    {"LBTESTCD": "AST", "LBTEST": "Aspartate Aminotransferase",
     "LBORRESU": "U/L", "mean": 22.0, "std": 8.0, "low": 10.0, "high": 40.0},
    {"LBTESTCD": "CREAT", "LBTEST": "Creatinine",
     "LBORRESU": "mg/dL", "mean": 0.9, "std": 0.2, "low": 0.6, "high": 1.2},
    {"LBTESTCD": "GLUC", "LBTEST": "Glucose",
     "LBORRESU": "mg/dL", "mean": 95.0, "std": 15.0, "low": 70.0, "high": 100.0},
]

# Demographics reference data
_SEX_OPTIONS = ["M", "F"]
_RACE_OPTIONS = [
    "WHITE",
    "BLACK OR AFRICAN AMERICAN",
    "ASIAN",
    "AMERICAN INDIAN OR ALASKA NATIVE",
    "MULTIPLE",
]
_RACE_WEIGHTS = [0.60, 0.15, 0.15, 0.05, 0.05]
_COUNTRY_OPTIONS = ["USA", "CAN", "GBR", "DEU", "FRA", "JPN", "AUS"]
_COUNTRY_WEIGHTS = [0.40, 0.10, 0.15, 0.10, 0.10, 0.10, 0.05]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SyntheticSdtmGenerator:
    """Generates synthetic patient-level SDTM data from SoA-derived schemas.

    Takes a MockSdtmDataset (from PTCV-53 SoaToSdtmMapper) and populates
    it with synthetic subject records for DM, SV, and LB domains.

    Args:
        config: Generation configuration. Uses defaults if None.
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self._config = config or SyntheticConfig()

    @property
    def config(self) -> SyntheticConfig:
        """Return the generation configuration."""
        return self._config

    def generate(
        self,
        dataset: "MockSdtmDataset",
        config: SyntheticConfig | None = None,
    ) -> SyntheticSdtmResult:
        """Generate synthetic SDTM data from a MockSdtmDataset.

        Args:
            dataset: MockSdtmDataset with TV, TA, TE, SE, domain_mapping.
            config: Optional override config. Uses instance config if None.

        Returns:
            SyntheticSdtmResult with DM, SV, LB DataFrames.
        """
        cfg = config or self._config
        rng = np.random.default_rng(cfg.seed)

        studyid = dataset.studyid

        # Extract arm info from TA domain
        arms = self._get_arms(dataset)

        # Generate subjects
        dm_df = self._generate_dm(studyid, arms, cfg, rng)

        # Generate visit instances
        sv_df = self._generate_sv(studyid, dm_df, dataset.tv, cfg, rng)

        # Generate lab data
        lb_df = self._generate_lb(
            studyid, dm_df, sv_df, dataset.domain_mapping, cfg, rng,
        )

        return SyntheticSdtmResult(
            dm=dm_df,
            sv=sv_df,
            lb=lb_df,
            config=cfg,
            studyid=studyid,
        )

    # ------------------------------------------------------------------
    # DM generation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_arms(dataset: "MockSdtmDataset") -> list[tuple[str, str]]:
        """Extract (ARMCD, ARM) pairs from TA domain."""
        if dataset.ta.empty:
            return [("ARM01", "Treatment")]
        unique = dataset.ta[["ARMCD", "ARM"]].drop_duplicates()
        return list(unique.itertuples(index=False, name=None))

    def _generate_dm(
        self,
        studyid: str,
        arms: list[tuple[str, str]],
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate DM (Demographics) domain DataFrame."""
        n = cfg.n_subjects

        # Allocate subjects to arms based on randomization ratio
        ratio = cfg.randomization_ratio
        # Pad ratio to match number of arms
        while len(ratio) < len(arms):
            ratio.append(1.0)
        ratio = ratio[: len(arms)]
        total_ratio = sum(ratio)
        arm_probs = [r / total_ratio for r in ratio]
        arm_indices = rng.choice(len(arms), size=n, p=arm_probs)

        rows: list[dict[str, Any]] = []
        for i in range(n):
            arm_idx = int(arm_indices[i])
            armcd, arm_name = arms[arm_idx]

            age = int(rng.normal(55, 12))
            age = max(18, min(85, age))

            sex = _SEX_OPTIONS[int(rng.choice(len(_SEX_OPTIONS)))]

            race_idx = int(rng.choice(
                len(_RACE_OPTIONS), p=_RACE_WEIGHTS,
            ))
            race = _RACE_OPTIONS[race_idx]

            country_idx = int(rng.choice(
                len(_COUNTRY_OPTIONS), p=_COUNTRY_WEIGHTS,
            ))
            country = _COUNTRY_OPTIONS[country_idx]

            subjid = f"{i + 1:04d}"
            usubjid = f"{studyid}-{subjid}"

            rows.append({
                "STUDYID": studyid,
                "DOMAIN": "DM",
                "USUBJID": usubjid,
                "SUBJID": subjid,
                "AGE": float(age),
                "AGEU": "YEARS",
                "SEX": sex,
                "RACE": race,
                "COUNTRY": country,
                "ARMCD": armcd[:20],
                "ARM": arm_name[:40],
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # SV generation
    # ------------------------------------------------------------------

    def _generate_sv(
        self,
        studyid: str,
        dm_df: pd.DataFrame,
        tv_df: pd.DataFrame,
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate SV (Subject Visits) domain DataFrame."""
        if tv_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "USUBJID", "VISITNUM", "VISIT",
                "VISITDY", "SVSTDTC", "SVENDTC",
            ])

        rows: list[dict[str, Any]] = []
        visits = list(tv_df.itertuples(index=False))

        for _, subj in dm_df.iterrows():
            usubjid = subj["USUBJID"]

            # Determine dropout point for this subject
            dropout_visit = len(visits)  # default: completes all
            if rng.random() < cfg.dropout_rate:
                # Drop out after at least 2 visits (screening + baseline)
                dropout_visit = int(rng.integers(2, max(3, len(visits))))

            # Subject-specific screening date offset (staggered enrollment)
            enroll_offset = int(rng.integers(0, 90))
            base_date = cfg.study_start_date + timedelta(days=enroll_offset)

            for visit_idx, visit in enumerate(visits):
                if visit_idx >= dropout_visit:
                    break

                visitnum = float(visit.VISITNUM)
                visit_name = visit.VISIT
                planned_day = int(visit.VISITDY)

                # Apply visit window jitter
                window_low = int(getattr(visit, "TVSTRL", 0))
                window_high = int(getattr(visit, "TVENRL", 0))
                jitter = int(rng.integers(
                    window_low, max(window_low + 1, window_high + 1),
                ))
                actual_day = planned_day + jitter

                visit_date = base_date + timedelta(days=actual_day)
                visit_date_str = visit_date.isoformat()

                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "SV",
                    "USUBJID": usubjid,
                    "VISITNUM": visitnum,
                    "VISIT": visit_name[:40],
                    "VISITDY": float(actual_day),
                    "SVSTDTC": visit_date_str,
                    "SVENDTC": visit_date_str,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # LB generation
    # ------------------------------------------------------------------

    def _generate_lb(
        self,
        studyid: str,
        dm_df: pd.DataFrame,
        sv_df: pd.DataFrame,
        domain_mapping: pd.DataFrame,
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate LB (Laboratory Test Results) domain DataFrame."""
        # Find which visits have lab assessments
        lab_visits = self._get_lab_visits(domain_mapping, sv_df)

        if not lab_visits or sv_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "USUBJID", "LBTESTCD", "LBTEST",
                "LBORRES", "LBORRESU", "LBORNRLO", "LBORNRHI",
                "VISITNUM", "VISIT", "LBDTC",
            ])

        rows: list[dict[str, Any]] = []
        lbseq_counter: dict[str, int] = {}

        for _, sv_row in sv_df.iterrows():
            visitnum = sv_row["VISITNUM"]
            if visitnum not in lab_visits:
                continue

            usubjid = sv_row["USUBJID"]
            visit_name = sv_row["VISIT"]
            visit_date = sv_row.get("SVSTDTC", "")

            if usubjid not in lbseq_counter:
                lbseq_counter[usubjid] = 0

            for lab in _LAB_TESTS:
                lbseq_counter[usubjid] += 1
                value = float(rng.normal(lab["mean"], lab["std"]))
                value = max(0.0, round(value, 1))

                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "LB",
                    "USUBJID": usubjid,
                    "LBSEQ": float(lbseq_counter[usubjid]),
                    "LBTESTCD": lab["LBTESTCD"],
                    "LBTEST": lab["LBTEST"],
                    "LBORRES": str(value),
                    "LBORRESU": lab["LBORRESU"],
                    "LBORNRLO": str(lab["low"]),
                    "LBORNRHI": str(lab["high"]),
                    "VISITNUM": visitnum,
                    "VISIT": visit_name[:40],
                    "LBDTC": str(visit_date),
                })

        return pd.DataFrame(rows)

    @staticmethod
    def _get_lab_visits(
        domain_mapping: pd.DataFrame,
        sv_df: pd.DataFrame,
    ) -> set[float]:
        """Determine which VISITNUM values have lab assessments scheduled.

        If domain_mapping has LB-mapped assessments, use all visits from SV.
        Otherwise, return all visits (conservative: labs at every visit).
        """
        if domain_mapping.empty:
            # No mapping info — generate labs at all visits
            if sv_df.empty:
                return set()
            return set(sv_df["VISITNUM"].unique())

        lb_rows = domain_mapping[domain_mapping["SDTM_DOMAIN"] == "LB"]
        if lb_rows.empty:
            return set()

        # Lab assessments exist — generate at all attended visits
        if sv_df.empty:
            return set()
        return set(sv_df["VISITNUM"].unique())
