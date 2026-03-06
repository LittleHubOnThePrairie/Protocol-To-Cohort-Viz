"""Mock data orchestrator: SoA to validated mock SDTM pipeline (PTCV-121).

End-to-end orchestrator that ties together SoA extraction output,
mock data generation, and Great Expectations validation into a single
pipeline call.

Two generation backends:

1. **Seed-data backend** (default) — generates clinically plausible
   mock data using numpy distributions.  No SDV/PyTorch required.
2. **SDV backend** — uses ``SdvSynthesizerService`` for ML-based
   synthetic data.  Requires ``pip install sdv``.

Risk tier: LOW — synthetic data only; no real patient data.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from ptcv.expectations.checkpoint_runner import (
    ValidationReport,
    validate_all,
)
from ptcv.expectations.suite_builder import SuiteConfig

from .sdtm_metadata import (
    SdtmDomainSpec,
    get_all_domain_specs,
    get_domain_spec,
)
from .sdv_synthesizer import (
    _build_seed_dm,
    _build_seed_observations,
)

if TYPE_CHECKING:
    from ptcv.sdtm.soa_mapper import MockSdtmDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MockDataConfig:
    """Configuration for the mock data pipeline.

    Attributes:
        num_subjects: Number of synthetic subjects to generate.
        study_id: STUDYID value for all generated domains.
        records_per_subject_range: ``(min, max)`` records per subject
            for observation-class domains.  DM always produces
            exactly 1 row per subject.
        seed: Random seed for reproducibility.  ``None`` for random.
        suite_config: GX suite configuration for validation.
        use_sdv: Attempt to use SDV synthesizer.  Falls back to
            seed-data generation if SDV is unavailable.
    """

    num_subjects: int = 20
    study_id: str = "MOCK-STUDY-001"
    records_per_subject_range: tuple[int, int] = (3, 8)
    seed: Optional[int] = 42
    suite_config: Optional[SuiteConfig] = None
    use_sdv: bool = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MockPipelineResult:
    """Result of the mock data generation + validation pipeline.

    Attributes:
        dataframes: Generated SDTM DataFrames keyed by domain code.
        validation_report: GX validation results for all domains.
        domains_generated: List of generated domain codes.
        run_id: UUID4 for audit traceability.
        num_subjects: Number of subjects generated.
    """

    dataframes: dict[str, pd.DataFrame]
    validation_report: ValidationReport
    domains_generated: list[str]
    run_id: str
    num_subjects: int


# ---------------------------------------------------------------------------
# Data generation — seed-data backend (no SDV)
# ---------------------------------------------------------------------------


def _generate_seed_dataframes(
    domain_specs: dict[str, SdtmDomainSpec],
    num_subjects: int,
    study_id: str,
    records_per_subject_range: tuple[int, int],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Generate DataFrames using seed-data distributions."""
    usubjids = [
        f"{study_id}-{i + 1:04d}" for i in range(num_subjects)
    ]
    result: dict[str, pd.DataFrame] = {}

    for code, spec in domain_specs.items():
        if code == "DM":
            df = _build_seed_dm(rng, num_subjects, study_id)
        else:
            lo, hi = records_per_subject_range
            recs = int(rng.integers(lo, hi + 1))
            df = _build_seed_observations(
                rng, spec, usubjids, recs, study_id,
            )

        # Enforce column ordering per spec.
        ordered = [v.name for v in spec.variables]
        present = [c for c in ordered if c in df.columns]
        result[code] = df[present]

    return result


# ---------------------------------------------------------------------------
# MockDataService
# ---------------------------------------------------------------------------


class MockDataService:
    """End-to-end mock data generation and validation service.

    Args:
        config: Pipeline configuration.
    """

    def __init__(
        self,
        config: Optional[MockDataConfig] = None,
    ) -> None:
        self._config = config or MockDataConfig()

    @property
    def config(self) -> MockDataConfig:
        return self._config

    def generate_from_soa(
        self,
        soa_dataset: MockSdtmDataset,
        num_subjects: Optional[int] = None,
    ) -> MockPipelineResult:
        """Generate validated mock SDTM data from SoA mapper output.

        Extracts domain codes from the ``domain_mapping`` DataFrame
        (``SDTM_DOMAIN`` column), then delegates to
        ``generate_from_domains()``.

        Args:
            soa_dataset: Output of ``SoaToSdtmMapper.map()``.
            num_subjects: Override ``config.num_subjects``.

        Returns:
            ``MockPipelineResult`` with DataFrames + validation.
        """
        domain_col = "SDTM_DOMAIN"
        mapping = soa_dataset.domain_mapping

        if domain_col not in mapping.columns:
            domain_col = "CDASH_DOMAIN"

        domain_codes = sorted(
            mapping[domain_col].dropna().unique().tolist()
        )

        # Filter to domains in our registry.
        registry = get_all_domain_specs()
        valid_codes = [
            c for c in domain_codes if c in registry
        ]

        return self.generate_from_domains(
            valid_codes, num_subjects=num_subjects,
        )

    def generate_from_domains(
        self,
        domain_codes: list[str],
        num_subjects: Optional[int] = None,
    ) -> MockPipelineResult:
        """Generate validated mock SDTM data from explicit domain list.

        Args:
            domain_codes: List of SDTM domain codes.
            num_subjects: Override ``config.num_subjects``.

        Returns:
            ``MockPipelineResult`` with DataFrames + validation.
        """
        cfg = self._config
        n_subj = num_subjects or cfg.num_subjects
        run_id = str(uuid.uuid4())

        # Look up domain specs.
        specs: dict[str, SdtmDomainSpec] = {}
        for code in domain_codes:
            upper = code.upper()
            try:
                specs[upper] = get_domain_spec(upper)
            except KeyError:
                continue

        # Generate DataFrames.
        dataframes = self._generate(specs, n_subj)

        # Validate all.
        report = validate_all(
            dataframes, config=cfg.suite_config,
        )

        return MockPipelineResult(
            dataframes=dataframes,
            validation_report=report,
            domains_generated=sorted(dataframes.keys()),
            run_id=run_id,
            num_subjects=n_subj,
        )

    def _generate(
        self,
        specs: dict[str, SdtmDomainSpec],
        num_subjects: int,
    ) -> dict[str, pd.DataFrame]:
        """Generate DataFrames via SDV or seed-data fallback."""
        cfg = self._config
        rng = np.random.default_rng(cfg.seed)

        if cfg.use_sdv:
            try:
                from .sdv_synthesizer import (
                    SdvConfig,
                    SdvSynthesizerService,
                )
                sdv_cfg = SdvConfig(
                    num_subjects=num_subjects,
                    random_seed=cfg.seed or 42,
                    records_per_subject_range=(
                        cfg.records_per_subject_range
                    ),
                )
                svc = SdvSynthesizerService(config=sdv_cfg)
                result = svc.generate_all(
                    specs, num_subjects=num_subjects,
                )
                return result.domain_dataframes
            except ImportError:
                pass  # Fall through to seed-data backend.

        return _generate_seed_dataframes(
            specs, num_subjects, cfg.study_id,
            cfg.records_per_subject_range, rng,
        )
