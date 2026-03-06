"""Tests for SDV synthesizer service — PTCV-119.

Verifies mock SDTM data generation using the SDV synthesizer
wrapper. Since SDV is an optional dependency, all tests mock the
``sdv`` package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.mock_data.sdtm_metadata import get_domain_spec


# -----------------------------------------------------------------------
# Mock SDV classes
# -----------------------------------------------------------------------


class FakeSingleTableMetadata:
    """Minimal mock of ``sdv.metadata.SingleTableMetadata``."""

    def __init__(self) -> None:
        self.columns: dict[str, dict] = {}
        self.primary_key: str | None = None

    def add_column(self, name: str, **kwargs: object) -> None:
        self.columns[name] = dict(kwargs)

    def set_primary_key(self, column: str) -> None:
        self.primary_key = column


class FakeGaussianCopulaSynthesizer:
    """Mock SDV GaussianCopulaSynthesizer.

    Stores the fit data and returns it (or resampled rows) on
    ``sample()``, preserving column structure.
    """

    def __init__(
        self,
        metadata: Any,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
    ) -> None:
        self._metadata = metadata
        self._data: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame) -> None:
        self._data = data.copy()

    def sample(
        self,
        num_rows: int,
        output_file_path: Any = None,
    ) -> pd.DataFrame:
        """Return rows from fit data, cycling if needed."""
        assert self._data is not None, "Must call fit() first"
        if len(self._data) == 0:
            return self._data.copy()

        # Cycle the seed data to produce requested row count.
        repeats = (num_rows // len(self._data)) + 1
        result = pd.concat(
            [self._data] * repeats, ignore_index=True,
        ).iloc[:num_rows]
        return result.reset_index(drop=True)


class FakeMultiTableMetadata:
    """Minimal mock of ``sdv.metadata.MultiTableMetadata``."""

    def __init__(self) -> None:
        self.tables: dict[str, FakeSingleTableMetadata] = {}
        self.relationships: list[dict] = []

    def add_table(self, name: str, metadata: Any) -> None:
        self.tables[name] = metadata

    def add_relationship(self, **kwargs: Any) -> None:
        self.relationships.append(kwargs)


@pytest.fixture(autouse=True)
def _mock_sdv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake ``sdv`` package into sys.modules."""
    import ptcv.mock_data.sdv_adapter as adapter_mod

    adapter_mod._SDV_AVAILABLE = None

    # Build fake module hierarchy.
    fake_sdv = ModuleType("sdv")
    fake_metadata = ModuleType("sdv.metadata")
    fake_metadata.SingleTableMetadata = FakeSingleTableMetadata  # type: ignore[attr-defined]
    fake_metadata.MultiTableMetadata = FakeMultiTableMetadata  # type: ignore[attr-defined]

    fake_single_table = ModuleType("sdv.single_table")
    fake_single_table.GaussianCopulaSynthesizer = FakeGaussianCopulaSynthesizer  # type: ignore[attr-defined]

    fake_sdv.metadata = fake_metadata  # type: ignore[attr-defined]
    fake_sdv.single_table = fake_single_table  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sdv", fake_sdv)
    monkeypatch.setitem(sys.modules, "sdv.metadata", fake_metadata)
    monkeypatch.setitem(
        sys.modules, "sdv.single_table", fake_single_table,
    )


# -----------------------------------------------------------------------
# Scenario: Generate VS domain with 10 subjects
# -----------------------------------------------------------------------


class TestGenerateVSDomain:
    """Generate VS domain with 10 subjects."""

    def test_columns_match_sdtm_vs_order(self) -> None:
        """Returned DataFrame columns match SDTM VS variable order."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=5, random_seed=42,
        ))
        vs_spec = get_domain_spec("VS")
        df = svc.generate_domain(vs_spec, num_subjects=5)

        expected_order = [v.name for v in vs_spec.variables]
        assert list(df.columns) == expected_order

    def test_usubjid_has_exact_subject_count(self) -> None:
        """USUBJID has exactly 10 unique values."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=10, random_seed=42,
        ))
        df = svc.generate_domain(
            get_domain_spec("VS"), num_subjects=10,
        )

        unique_subjects = df["USUBJID"].nunique()
        assert unique_subjects == 10

    def test_vstestcd_from_codelist(self) -> None:
        """VSTESTCD values are from the VS codelist."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=5, random_seed=42,
        ))
        vs_spec = get_domain_spec("VS")
        df = svc.generate_domain(vs_spec, num_subjects=5)

        codelist = vs_spec.variables[4].codelist  # VSTESTCD
        assert codelist is not None
        actual = set(df["VSTESTCD"].unique())
        assert actual.issubset(codelist)

    def test_returns_dataframe(self) -> None:
        """generate_domain returns a pandas DataFrame."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(num_subjects=3))
        df = svc.generate_domain(get_domain_spec("VS"))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_dm_produces_one_row_per_subject(self) -> None:
        """DM domain produces exactly 1 row per subject."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=7, random_seed=42,
        ))
        df = svc.generate_domain(
            get_domain_spec("DM"), num_subjects=7,
        )
        assert len(df) == 7
        assert df["USUBJID"].nunique() == 7


# -----------------------------------------------------------------------
# Scenario: Generate all domains with shared USUBJID
# -----------------------------------------------------------------------


class TestGenerateAllDomains:
    """Generate all domains with shared USUBJID."""

    def test_shared_usubjids(self) -> None:
        """DM and VS share the same 5 USUBJIDs."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=5, random_seed=42,
        ))
        result = svc.generate_all(
            {
                "DM": get_domain_spec("DM"),
                "VS": get_domain_spec("VS"),
            },
            num_subjects=5,
        )

        dm_ids = set(result.domain_dataframes["DM"]["USUBJID"])
        vs_ids = set(result.domain_dataframes["VS"]["USUBJID"])
        assert dm_ids == vs_ids
        assert len(dm_ids) == 5

    def test_dm_has_one_row_per_subject(self) -> None:
        """DM has exactly num_subjects rows."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=5, random_seed=42,
        ))
        result = svc.generate_all(
            {
                "DM": get_domain_spec("DM"),
                "VS": get_domain_spec("VS"),
            },
            num_subjects=5,
        )

        assert len(result.domain_dataframes["DM"]) == 5

    def test_vs_has_multiple_rows_per_subject(self) -> None:
        """VS has multiple rows per subject."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=5,
            random_seed=42,
            records_per_subject_range=(3, 5),
        ))
        result = svc.generate_all(
            {
                "DM": get_domain_spec("DM"),
                "VS": get_domain_spec("VS"),
            },
            num_subjects=5,
        )

        vs_df = result.domain_dataframes["VS"]
        assert len(vs_df) > 5  # More rows than subjects

    def test_result_has_metadata(self) -> None:
        """MockGenerationResult has timestamp, run_id, config."""
        from ptcv.mock_data.sdv_synthesizer import (
            MockGenerationResult,
            SdvConfig,
            SdvSynthesizerService,
        )

        cfg = SdvConfig(num_subjects=3, random_seed=99)
        svc = SdvSynthesizerService(cfg)
        result = svc.generate_all(
            {"DM": get_domain_spec("DM")}, num_subjects=3,
        )

        assert isinstance(result, MockGenerationResult)
        assert result.run_id
        assert result.generation_timestamp
        assert result.config_used is cfg

    def test_three_domains(self) -> None:
        """Can generate DM, VS, and LB together."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=3, random_seed=42,
        ))
        result = svc.generate_all({
            "DM": get_domain_spec("DM"),
            "VS": get_domain_spec("VS"),
            "LB": get_domain_spec("LB"),
        })

        assert set(result.domain_dataframes.keys()) == {
            "DM", "VS", "LB",
        }
        # All share the same USUBJIDs.
        dm_ids = set(result.domain_dataframes["DM"]["USUBJID"])
        vs_ids = set(result.domain_dataframes["VS"]["USUBJID"])
        lb_ids = set(result.domain_dataframes["LB"]["USUBJID"])
        assert dm_ids == vs_ids == lb_ids


# -----------------------------------------------------------------------
# Scenario: Random seed ensures reproducibility
# -----------------------------------------------------------------------


class TestReproducibility:
    """Random seed ensures reproducibility."""

    def test_identical_with_same_seed(self) -> None:
        """Two calls with seed=42 produce identical DataFrames."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        vs_spec = get_domain_spec("VS")

        svc1 = SdvSynthesizerService(SdvConfig(
            random_seed=42, num_subjects=5,
        ))
        df1 = svc1.generate_domain(
            vs_spec, num_subjects=5, num_records_per_subject=3,
        )

        svc2 = SdvSynthesizerService(SdvConfig(
            random_seed=42, num_subjects=5,
        ))
        df2 = svc2.generate_domain(
            vs_spec, num_subjects=5, num_records_per_subject=3,
        )

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_differs(self) -> None:
        """Different seeds produce different data."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        vs_spec = get_domain_spec("VS")

        svc1 = SdvSynthesizerService(SdvConfig(
            random_seed=42, num_subjects=5,
        ))
        df1 = svc1.generate_domain(
            vs_spec, num_subjects=5, num_records_per_subject=3,
        )

        svc2 = SdvSynthesizerService(SdvConfig(
            random_seed=99, num_subjects=5,
        ))
        df2 = svc2.generate_domain(
            vs_spec, num_subjects=5, num_records_per_subject=3,
        )

        # At least some values should differ.
        assert not df1.equals(df2)


# -----------------------------------------------------------------------
# Scenario: SDTM column ordering enforced
# -----------------------------------------------------------------------


class TestColumnOrdering:
    """SDTM column ordering enforced."""

    @pytest.mark.parametrize(
        "domain_code",
        ["DM", "VS", "LB", "AE", "EG", "CM"],
    )
    def test_column_order_matches_spec(
        self, domain_code: str,
    ) -> None:
        """Columns follow SDTMIG variable ordering."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        spec = get_domain_spec(domain_code)
        svc = SdvSynthesizerService(SdvConfig(
            num_subjects=3, random_seed=42,
        ))
        df = svc.generate_domain(spec, num_subjects=3)

        expected = [v.name for v in spec.variables]
        assert list(df.columns) == expected

    def test_studyid_is_first_column(self) -> None:
        """STUDYID is always the first column."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(num_subjects=3))
        df = svc.generate_domain(get_domain_spec("VS"))
        assert df.columns[0] == "STUDYID"

    def test_domain_is_second_column(self) -> None:
        """DOMAIN is always the second column."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(num_subjects=3))
        df = svc.generate_domain(get_domain_spec("LB"))
        assert df.columns[1] == "DOMAIN"

    def test_usubjid_is_third_column(self) -> None:
        """USUBJID is always the third column."""
        from ptcv.mock_data.sdv_synthesizer import (
            SdvConfig,
            SdvSynthesizerService,
        )

        svc = SdvSynthesizerService(SdvConfig(num_subjects=3))
        df = svc.generate_domain(get_domain_spec("AE"))
        assert df.columns[2] == "USUBJID"


# -----------------------------------------------------------------------
# SdvConfig
# -----------------------------------------------------------------------


class TestSdvConfig:
    """SdvConfig defaults and construction."""

    def test_defaults(self) -> None:
        from ptcv.mock_data.sdv_synthesizer import SdvConfig

        cfg = SdvConfig()
        assert cfg.synthesizer_type == "gaussian_copula"
        assert cfg.random_seed == 42
        assert cfg.num_subjects == 10
        assert cfg.records_per_subject_range == (3, 8)

    def test_custom_values(self) -> None:
        from ptcv.mock_data.sdv_synthesizer import SdvConfig

        cfg = SdvConfig(
            synthesizer_type="ctgan",
            random_seed=123,
            num_subjects=50,
            records_per_subject_range=(1, 3),
        )
        assert cfg.synthesizer_type == "ctgan"
        assert cfg.random_seed == 123
        assert cfg.num_subjects == 50
        assert cfg.records_per_subject_range == (1, 3)
