"""Synthetic SDTM data generator from SoA-derived schemas (PTCV-54).

Populates MockSdtmDataset schemas with synthetic patient-level records
for testing statistical models, validating SDTM structures, and
parameterising natural history cohort searches.

Generated domains:
  DM — Demographics (USUBJID, AGE, SEX, RACE, ARM assignment)
  SV — Subject Visits (per-subject visit instances with timing jitter)
  LB — Laboratory Test Results (clinically plausible synthetic values)
  AE — Adverse Events (synthetic AE terms with severity/causality)
  VS — Vital Signs (HR, BP, temperature, weight)
  CM — Concomitant Medications (prior/concomitant meds)
  MH — Medical History (pre-existing conditions)
  DS — Disposition (screening, randomization, completion/discontinuation)
  EX — Exposure (dosing records per arm assignment)

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
        ae: Adverse Events domain DataFrame.
        vs: Vital Signs domain DataFrame.
        cm: Concomitant Medications domain DataFrame.
        mh: Medical History domain DataFrame.
        ds: Disposition domain DataFrame.
        ex: Exposure domain DataFrame.
        config: Configuration used for generation.
        studyid: STUDYID value.
    """

    dm: pd.DataFrame
    sv: pd.DataFrame
    lb: pd.DataFrame
    ae: pd.DataFrame
    vs: pd.DataFrame
    cm: pd.DataFrame
    mh: pd.DataFrame
    ds: pd.DataFrame
    ex: pd.DataFrame
    config: SyntheticConfig
    studyid: str

    @property
    def domains(self) -> dict[str, pd.DataFrame]:
        """Return all generated domain DataFrames."""
        return {
            "DM": self.dm, "SV": self.sv, "LB": self.lb,
            "AE": self.ae, "VS": self.vs, "CM": self.cm,
            "MH": self.mh, "DS": self.ds, "EX": self.ex,
        }


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

# Adverse event reference data
_AE_TERMS: list[dict[str, str]] = [
    {"AETERM": "Headache", "AEDECOD": "HEADACHE",
     "AEBODSYS": "Nervous system disorders"},
    {"AETERM": "Nausea", "AEDECOD": "NAUSEA",
     "AEBODSYS": "Gastrointestinal disorders"},
    {"AETERM": "Fatigue", "AEDECOD": "FATIGUE",
     "AEBODSYS": "General disorders and administration site conditions"},
    {"AETERM": "Dizziness", "AEDECOD": "DIZZINESS",
     "AEBODSYS": "Nervous system disorders"},
    {"AETERM": "Diarrhoea", "AEDECOD": "DIARRHOEA",
     "AEBODSYS": "Gastrointestinal disorders"},
    {"AETERM": "Back pain", "AEDECOD": "BACK PAIN",
     "AEBODSYS": "Musculoskeletal and connective tissue disorders"},
    {"AETERM": "Upper respiratory tract infection",
     "AEDECOD": "UPPER RESPIRATORY TRACT INFECTION",
     "AEBODSYS": "Infections and infestations"},
    {"AETERM": "Insomnia", "AEDECOD": "INSOMNIA",
     "AEBODSYS": "Psychiatric disorders"},
    {"AETERM": "Rash", "AEDECOD": "RASH",
     "AEBODSYS": "Skin and subcutaneous tissue disorders"},
    {"AETERM": "Arthralgia", "AEDECOD": "ARTHRALGIA",
     "AEBODSYS": "Musculoskeletal and connective tissue disorders"},
]
_AE_SEVERITY = ["MILD", "MODERATE", "SEVERE"]
_AE_SEVERITY_WEIGHTS = [0.55, 0.35, 0.10]
_AE_CAUSALITY = [
    "NOT RELATED", "UNLIKELY RELATED", "POSSIBLY RELATED", "RELATED",
]
_AE_CAUSALITY_WEIGHTS = [0.30, 0.25, 0.30, 0.15]
_AE_OUTCOME = [
    "RECOVERED/RESOLVED", "RECOVERING/RESOLVING", "NOT RECOVERED/NOT RESOLVED",
]
_AE_OUTCOME_WEIGHTS = [0.70, 0.20, 0.10]

# Vital signs reference data
_VS_TESTS: list[dict[str, Any]] = [
    {"VSTESTCD": "SYSBP", "VSTEST": "Systolic Blood Pressure",
     "VSORRESU": "mmHg", "mean": 125.0, "std": 15.0},
    {"VSTESTCD": "DIABP", "VSTEST": "Diastolic Blood Pressure",
     "VSORRESU": "mmHg", "mean": 78.0, "std": 10.0},
    {"VSTESTCD": "PULSE", "VSTEST": "Pulse Rate",
     "VSORRESU": "beats/min", "mean": 72.0, "std": 10.0},
    {"VSTESTCD": "TEMP", "VSTEST": "Temperature",
     "VSORRESU": "C", "mean": 36.8, "std": 0.3},
    {"VSTESTCD": "WEIGHT", "VSTEST": "Weight",
     "VSORRESU": "kg", "mean": 78.0, "std": 14.0},
]

# Concomitant medications reference data
_CM_MEDS: list[dict[str, str]] = [
    {"CMTRT": "Paracetamol", "CMDECOD": "PARACETAMOL",
     "CMCAT": "CONCOMITANT", "CMROUTE": "ORAL", "CMINDC": "Pain"},
    {"CMTRT": "Ibuprofen", "CMDECOD": "IBUPROFEN",
     "CMCAT": "CONCOMITANT", "CMROUTE": "ORAL", "CMINDC": "Pain"},
    {"CMTRT": "Omeprazole", "CMDECOD": "OMEPRAZOLE",
     "CMCAT": "CONCOMITANT", "CMROUTE": "ORAL", "CMINDC": "GERD"},
    {"CMTRT": "Metformin", "CMDECOD": "METFORMIN",
     "CMCAT": "PRE-STUDY", "CMROUTE": "ORAL", "CMINDC": "Type 2 diabetes"},
    {"CMTRT": "Lisinopril", "CMDECOD": "LISINOPRIL",
     "CMCAT": "PRE-STUDY", "CMROUTE": "ORAL", "CMINDC": "Hypertension"},
    {"CMTRT": "Atorvastatin", "CMDECOD": "ATORVASTATIN",
     "CMCAT": "PRE-STUDY", "CMROUTE": "ORAL", "CMINDC": "Hyperlipidemia"},
    {"CMTRT": "Multivitamin", "CMDECOD": "MULTIVITAMIN",
     "CMCAT": "CONCOMITANT", "CMROUTE": "ORAL", "CMINDC": "Supplementation"},
]

# Medical history reference data
_MH_CONDITIONS: list[dict[str, str]] = [
    {"MHTERM": "Hypertension", "MHDECOD": "HYPERTENSION",
     "MHBODSYS": "Vascular disorders", "MHCAT": "Cardiovascular"},
    {"MHTERM": "Type 2 diabetes mellitus",
     "MHDECOD": "TYPE 2 DIABETES MELLITUS",
     "MHBODSYS": "Metabolism and nutrition disorders",
     "MHCAT": "Endocrine/Metabolic"},
    {"MHTERM": "Seasonal allergies", "MHDECOD": "SEASONAL ALLERGY",
     "MHBODSYS": "Immune system disorders", "MHCAT": "Allergy"},
    {"MHTERM": "Gastroesophageal reflux disease",
     "MHDECOD": "GASTRO-OESOPHAGEAL REFLUX DISEASE",
     "MHBODSYS": "Gastrointestinal disorders",
     "MHCAT": "Gastrointestinal"},
    {"MHTERM": "Osteoarthritis", "MHDECOD": "OSTEOARTHRITIS",
     "MHBODSYS": "Musculoskeletal and connective tissue disorders",
     "MHCAT": "Musculoskeletal"},
    {"MHTERM": "Anxiety disorder", "MHDECOD": "ANXIETY DISORDER",
     "MHBODSYS": "Psychiatric disorders", "MHCAT": "Psychiatric"},
    {"MHTERM": "Hyperlipidemia", "MHDECOD": "HYPERLIPIDAEMIA",
     "MHBODSYS": "Metabolism and nutrition disorders",
     "MHCAT": "Endocrine/Metabolic"},
    {"MHTERM": "Appendectomy", "MHDECOD": "APPENDICECTOMY",
     "MHBODSYS": "Surgical and medical procedures", "MHCAT": "Surgical"},
]

# Disposition reference data
_DS_DISCONTINUATION_REASONS = [
    "ADVERSE EVENT", "WITHDREW CONSENT", "LOST TO FOLLOW-UP",
    "PHYSICIAN DECISION", "PROTOCOL VIOLATION", "LACK OF EFFICACY",
]
_DS_DISCONTINUATION_WEIGHTS = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SyntheticSdtmGenerator:
    """Generates synthetic patient-level SDTM data from SoA-derived schemas.

    Takes a MockSdtmDataset (from PTCV-53 SoaToSdtmMapper) and populates
    it with synthetic subject records for all clinical domains.

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
            SyntheticSdtmResult with all clinical domain DataFrames.
        """
        return self.generate_from_dataframes(
            studyid=dataset.studyid,
            ta_df=dataset.ta,
            tv_df=dataset.tv,
            domain_mapping=dataset.domain_mapping,
            config=config,
        )

    def generate_from_dataframes(
        self,
        studyid: str,
        ta_df: pd.DataFrame,
        tv_df: pd.DataFrame,
        domain_mapping: pd.DataFrame | None = None,
        config: SyntheticConfig | None = None,
    ) -> SyntheticSdtmResult:
        """Generate synthetic SDTM data from raw DataFrames.

        Standalone entry point that does not require a MockSdtmDataset,
        allowing SdtmService to call this directly (PTCV-284).

        Args:
            studyid: STUDYID value.
            ta_df: Trial Arms DataFrame (ARMCD, ARM columns).
            tv_df: Trial Visits DataFrame.
            domain_mapping: Assessment-to-domain mapping (optional).
            config: Optional override config.

        Returns:
            SyntheticSdtmResult with all clinical domain DataFrames.
        """
        cfg = config or self._config
        rng = np.random.default_rng(cfg.seed)

        if domain_mapping is None:
            domain_mapping = pd.DataFrame()

        # Extract arm info from TA domain
        arms = self._get_arms_from_df(ta_df)

        # Generate subjects
        dm_df = self._generate_dm(studyid, arms, cfg, rng)

        # Generate visit instances
        sv_df = self._generate_sv(studyid, dm_df, tv_df, cfg, rng)

        # Generate lab data
        lb_df = self._generate_lb(
            studyid, dm_df, sv_df, domain_mapping, cfg, rng,
        )

        # Generate clinical domains (PTCV-284)
        ae_df = self._generate_ae(studyid, dm_df, sv_df, cfg, rng)
        vs_df = self._generate_vs(studyid, dm_df, sv_df, cfg, rng)
        cm_df = self._generate_cm(studyid, dm_df, cfg, rng)
        mh_df = self._generate_mh(studyid, dm_df, cfg, rng)
        ds_df = self._generate_ds(studyid, dm_df, sv_df, cfg, rng)
        ex_df = self._generate_ex(studyid, dm_df, sv_df, cfg, rng)

        return SyntheticSdtmResult(
            dm=dm_df,
            sv=sv_df,
            lb=lb_df,
            ae=ae_df,
            vs=vs_df,
            cm=cm_df,
            mh=mh_df,
            ds=ds_df,
            ex=ex_df,
            config=cfg,
            studyid=studyid,
        )

    # ------------------------------------------------------------------
    # DM generation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_arms(dataset: "MockSdtmDataset") -> list[tuple[str, str]]:
        """Extract (ARMCD, ARM) pairs from MockSdtmDataset."""
        return SyntheticSdtmGenerator._get_arms_from_df(dataset.ta)

    @staticmethod
    def _get_arms_from_df(ta_df: pd.DataFrame) -> list[tuple[str, str]]:
        """Extract (ARMCD, ARM) pairs from TA domain DataFrame."""
        if ta_df.empty:
            return [("ARM01", "Treatment")]
        unique = ta_df[["ARMCD", "ARM"]].drop_duplicates()
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

    # ------------------------------------------------------------------
    # AE generation (PTCV-284)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_ae(
        studyid: str,
        dm_df: pd.DataFrame,
        sv_df: pd.DataFrame,
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate AE (Adverse Events) domain DataFrame."""
        if dm_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "USUBJID", "AESEQ",
                "AETERM", "AEDECOD", "AEBODSYS",
                "AESEV", "AESER", "AEREL", "AEOUT",
                "AESTDTC", "AEENDTC",
            ])

        rows: list[dict[str, Any]] = []

        for _, subj in dm_df.iterrows():
            usubjid = subj["USUBJID"]
            n_aes = int(rng.poisson(1.5))  # mean 1.5 AEs per subject
            n_aes = min(n_aes, 5)

            # Get this subject's visit dates for onset timing
            subj_visits = sv_df[sv_df["USUBJID"] == usubjid]
            visit_dates: list[str] = []
            if not subj_visits.empty:
                visit_dates = list(subj_visits["SVSTDTC"].dropna())

            for seq in range(1, n_aes + 1):
                ae = _AE_TERMS[int(rng.choice(len(_AE_TERMS)))]
                sev_idx = int(rng.choice(
                    len(_AE_SEVERITY), p=_AE_SEVERITY_WEIGHTS,
                ))
                rel_idx = int(rng.choice(
                    len(_AE_CAUSALITY), p=_AE_CAUSALITY_WEIGHTS,
                ))
                out_idx = int(rng.choice(
                    len(_AE_OUTCOME), p=_AE_OUTCOME_WEIGHTS,
                ))

                # AE onset: pick a random visit date or offset from study start
                ae_start = ""
                ae_end = ""
                if visit_dates:
                    onset_visit = visit_dates[
                        int(rng.choice(len(visit_dates)))
                    ]
                    onset_date = date.fromisoformat(onset_visit)
                    onset_offset = int(rng.integers(0, 7))
                    onset_date = onset_date + timedelta(days=onset_offset)
                    ae_start = onset_date.isoformat()
                    duration = int(rng.integers(3, 21))
                    ae_end = (onset_date + timedelta(days=duration)).isoformat()

                is_serious = (
                    "Y" if _AE_SEVERITY[sev_idx] == "SEVERE"
                    and rng.random() < 0.5 else "N"
                )

                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "AE",
                    "USUBJID": usubjid,
                    "AESEQ": float(seq),
                    "AETERM": ae["AETERM"],
                    "AEDECOD": ae["AEDECOD"],
                    "AEBODSYS": ae["AEBODSYS"],
                    "AESEV": _AE_SEVERITY[sev_idx],
                    "AESER": is_serious,
                    "AEREL": _AE_CAUSALITY[rel_idx],
                    "AEOUT": _AE_OUTCOME[out_idx],
                    "AESTDTC": ae_start,
                    "AEENDTC": ae_end,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # VS generation (PTCV-284)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_vs(
        studyid: str,
        dm_df: pd.DataFrame,
        sv_df: pd.DataFrame,
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate VS (Vital Signs) domain DataFrame."""
        if sv_df.empty or dm_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "USUBJID", "VSSEQ",
                "VSTESTCD", "VSTEST", "VSORRES", "VSORRESU",
                "VISITNUM", "VISIT", "VSDTC",
            ])

        rows: list[dict[str, Any]] = []
        vsseq_counter: dict[str, int] = {}

        for _, sv_row in sv_df.iterrows():
            usubjid = sv_row["USUBJID"]
            visitnum = sv_row["VISITNUM"]
            visit_name = sv_row["VISIT"]
            visit_date = sv_row.get("SVSTDTC", "")

            if usubjid not in vsseq_counter:
                vsseq_counter[usubjid] = 0

            for vs in _VS_TESTS:
                vsseq_counter[usubjid] += 1
                value = float(rng.normal(vs["mean"], vs["std"]))
                value = max(0.0, round(value, 1))

                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "VS",
                    "USUBJID": usubjid,
                    "VSSEQ": float(vsseq_counter[usubjid]),
                    "VSTESTCD": vs["VSTESTCD"],
                    "VSTEST": vs["VSTEST"],
                    "VSORRES": str(value),
                    "VSORRESU": vs["VSORRESU"],
                    "VISITNUM": visitnum,
                    "VISIT": visit_name[:40],
                    "VSDTC": str(visit_date),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CM generation (PTCV-284)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_cm(
        studyid: str,
        dm_df: pd.DataFrame,
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate CM (Concomitant Medications) domain DataFrame."""
        if dm_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "USUBJID", "CMSEQ",
                "CMTRT", "CMDECOD", "CMCAT", "CMROUTE", "CMINDC",
                "CMSTDTC", "CMENDTC",
            ])

        rows: list[dict[str, Any]] = []

        for _, subj in dm_df.iterrows():
            usubjid = subj["USUBJID"]
            n_meds = int(rng.poisson(1.2))
            n_meds = min(n_meds, 4)

            chosen = rng.choice(
                len(_CM_MEDS), size=min(n_meds, len(_CM_MEDS)),
                replace=False,
            )

            for seq, med_idx in enumerate(chosen, start=1):
                med = _CM_MEDS[int(med_idx)]

                # Pre-study meds start before study
                offset = int(rng.integers(-180, -1))
                start = cfg.study_start_date + timedelta(days=offset)
                if med["CMCAT"] == "CONCOMITANT":
                    offset = int(rng.integers(0, 60))
                    start = cfg.study_start_date + timedelta(days=offset)

                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "CM",
                    "USUBJID": usubjid,
                    "CMSEQ": float(seq),
                    "CMTRT": med["CMTRT"],
                    "CMDECOD": med["CMDECOD"],
                    "CMCAT": med["CMCAT"],
                    "CMROUTE": med["CMROUTE"],
                    "CMINDC": med["CMINDC"],
                    "CMSTDTC": start.isoformat(),
                    "CMENDTC": "",
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # MH generation (PTCV-284)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_mh(
        studyid: str,
        dm_df: pd.DataFrame,
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate MH (Medical History) domain DataFrame."""
        if dm_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "USUBJID", "MHSEQ",
                "MHTERM", "MHDECOD", "MHBODSYS", "MHCAT",
                "MHSTDTC",
            ])

        rows: list[dict[str, Any]] = []

        for _, subj in dm_df.iterrows():
            usubjid = subj["USUBJID"]
            n_conds = int(rng.poisson(1.5))
            n_conds = min(n_conds, 5)

            chosen = rng.choice(
                len(_MH_CONDITIONS),
                size=min(n_conds, len(_MH_CONDITIONS)),
                replace=False,
            )

            for seq, cond_idx in enumerate(chosen, start=1):
                cond = _MH_CONDITIONS[int(cond_idx)]

                # MH onset: years before study start
                years_ago = int(rng.integers(1, 20))
                onset = cfg.study_start_date - timedelta(days=years_ago * 365)

                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "MH",
                    "USUBJID": usubjid,
                    "MHSEQ": float(seq),
                    "MHTERM": cond["MHTERM"],
                    "MHDECOD": cond["MHDECOD"],
                    "MHBODSYS": cond["MHBODSYS"],
                    "MHCAT": cond["MHCAT"],
                    "MHSTDTC": onset.isoformat(),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # DS generation (PTCV-284)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_ds(
        studyid: str,
        dm_df: pd.DataFrame,
        sv_df: pd.DataFrame,
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate DS (Disposition) domain DataFrame."""
        if dm_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "USUBJID", "DSSEQ",
                "DSTERM", "DSDECOD", "DSCAT", "DSSTDTC",
            ])

        rows: list[dict[str, Any]] = []

        for _, subj in dm_df.iterrows():
            usubjid = subj["USUBJID"]
            seq = 0

            # Get subject's visits for date reference
            subj_visits = sv_df[sv_df["USUBJID"] == usubjid]
            first_date = cfg.study_start_date
            last_date = cfg.study_start_date + timedelta(days=180)

            if not subj_visits.empty:
                dates = subj_visits["SVSTDTC"].dropna()
                if len(dates) > 0:
                    first_date = date.fromisoformat(str(dates.iloc[0]))
                    last_date = date.fromisoformat(str(dates.iloc[-1]))

            # Informed consent
            seq += 1
            rows.append({
                "STUDYID": studyid,
                "DOMAIN": "DS",
                "USUBJID": usubjid,
                "DSSEQ": float(seq),
                "DSTERM": "INFORMED CONSENT OBTAINED",
                "DSDECOD": "INFORMED CONSENT OBTAINED",
                "DSCAT": "PROTOCOL MILESTONE",
                "DSSTDTC": first_date.isoformat(),
            })

            # Randomized
            seq += 1
            rand_date = first_date + timedelta(days=int(rng.integers(1, 7)))
            rows.append({
                "STUDYID": studyid,
                "DOMAIN": "DS",
                "USUBJID": usubjid,
                "DSSEQ": float(seq),
                "DSTERM": "RANDOMIZED",
                "DSDECOD": "RANDOMIZED",
                "DSCAT": "PROTOCOL MILESTONE",
                "DSSTDTC": rand_date.isoformat(),
            })

            # Completion or discontinuation
            seq += 1
            max_visits = len(
                sv_df[sv_df["USUBJID"] == usubjid]
            ) if not sv_df.empty else 0
            total_visits = len(
                sv_df["VISITNUM"].unique()
            ) if not sv_df.empty else 0
            completed = max_visits >= total_visits or rng.random() > cfg.dropout_rate

            if completed:
                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "DS",
                    "USUBJID": usubjid,
                    "DSSEQ": float(seq),
                    "DSTERM": "COMPLETED",
                    "DSDECOD": "COMPLETED",
                    "DSCAT": "DISPOSITION EVENT",
                    "DSSTDTC": last_date.isoformat(),
                })
            else:
                reason_idx = int(rng.choice(
                    len(_DS_DISCONTINUATION_REASONS),
                    p=_DS_DISCONTINUATION_WEIGHTS,
                ))
                reason = _DS_DISCONTINUATION_REASONS[reason_idx]
                disc_date = first_date + timedelta(
                    days=int(rng.integers(
                        14,
                        max(15, (last_date - first_date).days),
                    ))
                )
                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "DS",
                    "USUBJID": usubjid,
                    "DSSEQ": float(seq),
                    "DSTERM": reason,
                    "DSDECOD": reason,
                    "DSCAT": "DISPOSITION EVENT",
                    "DSSTDTC": disc_date.isoformat(),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # EX generation (PTCV-284)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_ex(
        studyid: str,
        dm_df: pd.DataFrame,
        sv_df: pd.DataFrame,
        cfg: SyntheticConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate EX (Exposure) domain DataFrame."""
        if dm_df.empty or sv_df.empty:
            return pd.DataFrame(columns=[
                "STUDYID", "DOMAIN", "USUBJID", "EXSEQ",
                "EXTRT", "EXDOSE", "EXDOSU", "EXROUTE",
                "EXSTDTC", "EXENDTC", "VISITNUM", "VISIT",
            ])

        rows: list[dict[str, Any]] = []

        for _, subj in dm_df.iterrows():
            usubjid = subj["USUBJID"]
            arm_name = subj.get("ARM", "Treatment")

            # Get subject's post-screening visits for dosing
            subj_visits = sv_df[sv_df["USUBJID"] == usubjid]
            if subj_visits.empty:
                continue

            exseq = 0
            for _, sv_row in subj_visits.iterrows():
                visit_name = sv_row["VISIT"]
                # Skip screening visits for dosing
                if "screen" in str(visit_name).lower():
                    continue

                exseq += 1
                visit_date = sv_row.get("SVSTDTC", "")

                rows.append({
                    "STUDYID": studyid,
                    "DOMAIN": "EX",
                    "USUBJID": usubjid,
                    "EXSEQ": float(exseq),
                    "EXTRT": arm_name[:40],
                    "EXDOSE": 1.0,
                    "EXDOSU": "mg",
                    "EXROUTE": "ORAL",
                    "EXSTDTC": str(visit_date),
                    "EXENDTC": str(visit_date),
                    "VISITNUM": sv_row["VISITNUM"],
                    "VISIT": str(visit_name)[:40],
                })

        return pd.DataFrame(rows)
