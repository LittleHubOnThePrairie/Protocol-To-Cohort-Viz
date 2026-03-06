"""Tests for SDV metadata adapter — PTCV-118.

Verifies that SdtmDomainSpec is correctly mapped to SDV
SingleTableMetadata and MultiTableMetadata objects.

Since SDV is an optional dependency (not installed in test env),
all tests mock the ``sdv.metadata`` module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.mock_data.sdtm_metadata import (
    SdtmDomainSpec,
    SdtmVariableSpec,
    get_domain_spec,
)


# -----------------------------------------------------------------------
# Mock SDV metadata classes
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


class FakeMultiTableMetadata:
    """Minimal mock of ``sdv.metadata.MultiTableMetadata``."""

    def __init__(self) -> None:
        self.tables: dict[str, FakeSingleTableMetadata] = {}
        self.relationships: list[dict] = []

    def add_table(
        self, name: str, metadata: FakeSingleTableMetadata,
    ) -> None:
        self.tables[name] = metadata

    def add_relationship(
        self,
        parent_table_name: str,
        child_table_name: str,
        parent_primary_key: str,
        child_foreign_key: str,
    ) -> None:
        self.relationships.append({
            "parent_table_name": parent_table_name,
            "child_table_name": child_table_name,
            "parent_primary_key": parent_primary_key,
            "child_foreign_key": child_foreign_key,
        })


@pytest.fixture(autouse=True)
def _mock_sdv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``sdv`` package so the adapter can import it."""
    import ptcv.mock_data.sdv_adapter as adapter_mod

    # Reset the cached import check so each test gets a fresh state.
    adapter_mod._SDV_AVAILABLE = None

    # Build a fake sdv.metadata module.
    fake_sdv = ModuleType("sdv")
    fake_metadata = ModuleType("sdv.metadata")
    fake_metadata.SingleTableMetadata = FakeSingleTableMetadata  # type: ignore[attr-defined]
    fake_metadata.MultiTableMetadata = FakeMultiTableMetadata  # type: ignore[attr-defined]
    fake_sdv.metadata = fake_metadata  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sdv", fake_sdv)
    monkeypatch.setitem(sys.modules, "sdv.metadata", fake_metadata)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _var(
    name: str,
    type_: str = "Char",
    codelist: frozenset[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
) -> SdtmVariableSpec:
    return SdtmVariableSpec(
        name=name,
        label=f"Label for {name}",
        type=type_,
        core="Expected",
        codelist=codelist,
        range_min=range_min,
        range_max=range_max,
    )


# -----------------------------------------------------------------------
# Scenario: VS domain mapped to SingleTableMetadata
# -----------------------------------------------------------------------


class TestVSDomainMapping:
    """VS domain mapped to SingleTableMetadata."""

    def test_all_vs_variables_present(self) -> None:
        """Returned metadata has columns for all VS variables."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        vs_spec = get_domain_spec("VS")
        meta = build_sdv_metadata(vs_spec)

        expected_names = {v.name for v in vs_spec.variables}
        assert set(meta.columns.keys()) == expected_names

    def test_vstestcd_is_categorical(self) -> None:
        """VSTESTCD is sdtype categorical."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("VS"))
        assert meta.columns["VSTESTCD"]["sdtype"] == "categorical"

    def test_vsstresn_is_numerical(self) -> None:
        """VSSTRESN is sdtype numerical."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("VS"))
        assert meta.columns["VSSTRESN"]["sdtype"] == "numerical"

    def test_usubjid_is_id(self) -> None:
        """USUBJID is sdtype id."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("VS"))
        assert meta.columns["USUBJID"]["sdtype"] == "id"

    def test_vsdtc_is_datetime(self) -> None:
        """VSDTC (date-suffix) is sdtype datetime."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("VS"))
        assert meta.columns["VSDTC"]["sdtype"] == "datetime"

    def test_vsseq_is_primary_key(self) -> None:
        """VSSEQ is set as primary key for observation domain."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("VS"))
        assert meta.primary_key == "VSSEQ"

    def test_domain_column_is_categorical(self) -> None:
        """DOMAIN column is categorical."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("VS"))
        assert meta.columns["DOMAIN"]["sdtype"] == "categorical"

    def test_visitnum_is_numerical(self) -> None:
        """VISITNUM (Num type) is sdtype numerical."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("VS"))
        assert meta.columns["VISITNUM"]["sdtype"] == "numerical"


# -----------------------------------------------------------------------
# Scenario: Codelist constraints applied
# -----------------------------------------------------------------------


class TestCodelistConstraints:
    """Codelist constraints applied when enforce_codelists=True."""

    def test_vstestcd_has_regex_constraint(self) -> None:
        """VSTESTCD has regex matching codelist values."""
        from ptcv.mock_data.sdv_adapter import (
            SdvAdapterConfig,
            build_sdv_metadata,
        )

        cfg = SdvAdapterConfig(enforce_codelists=True)
        meta = build_sdv_metadata(get_domain_spec("VS"), config=cfg)

        col = meta.columns["VSTESTCD"]
        assert "regex_format" in col
        # Should match each codelist value.
        regex = col["regex_format"]
        for value in ("SYSBP", "DIABP", "PULSE", "TEMP"):
            assert value in regex

    def test_no_regex_when_codelists_off(self) -> None:
        """No regex_format when enforce_codelists=False."""
        from ptcv.mock_data.sdv_adapter import (
            SdvAdapterConfig,
            build_sdv_metadata,
        )

        cfg = SdvAdapterConfig(enforce_codelists=False)
        meta = build_sdv_metadata(get_domain_spec("VS"), config=cfg)

        col = meta.columns["VSTESTCD"]
        assert "regex_format" not in col

    def test_aesev_codelist_constraint(self) -> None:
        """AESEV codelist applied to AE domain."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("AE"))
        col = meta.columns["AESEV"]
        assert "regex_format" in col
        for value in ("MILD", "MODERATE", "SEVERE"):
            assert value in col["regex_format"]

    def test_non_codelist_char_no_regex(self) -> None:
        """Char variable without codelist has no regex_format."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("AE"))
        # AETERM has no codelist.
        col = meta.columns["AETERM"]
        assert col["sdtype"] == "categorical"
        assert "regex_format" not in col

    def test_id_column_not_constrained_by_codelist(self) -> None:
        """USUBJID keeps sdtype=id regardless of codelist setting."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec("DM"))
        assert meta.columns["USUBJID"]["sdtype"] == "id"


# -----------------------------------------------------------------------
# Scenario: Multi-table metadata preserves USUBJID relationship
# -----------------------------------------------------------------------


class TestMultiTableMetadata:
    """Multi-table metadata preserves USUBJID relationships."""

    def test_dm_vs_relationship(self) -> None:
        """VS.USUBJID → DM.USUBJID relationship created."""
        from ptcv.mock_data.sdv_adapter import (
            build_multi_table_metadata,
        )

        specs = {
            "DM": get_domain_spec("DM"),
            "VS": get_domain_spec("VS"),
        }
        multi = build_multi_table_metadata(specs)

        assert len(multi.relationships) == 1
        rel = multi.relationships[0]
        assert rel["parent_table_name"] == "DM"
        assert rel["child_table_name"] == "VS"
        assert rel["parent_primary_key"] == "USUBJID"
        assert rel["child_foreign_key"] == "USUBJID"

    def test_all_child_domains_linked(self) -> None:
        """All non-DM domains get USUBJID relationships."""
        from ptcv.mock_data.sdv_adapter import (
            build_multi_table_metadata,
        )

        specs = {
            "DM": get_domain_spec("DM"),
            "VS": get_domain_spec("VS"),
            "LB": get_domain_spec("LB"),
            "AE": get_domain_spec("AE"),
        }
        multi = build_multi_table_metadata(specs)

        assert len(multi.relationships) == 3
        child_tables = {
            r["child_table_name"] for r in multi.relationships
        }
        assert child_tables == {"VS", "LB", "AE"}

    def test_tables_registered(self) -> None:
        """Each domain spec produces a table in the metadata."""
        from ptcv.mock_data.sdv_adapter import (
            build_multi_table_metadata,
        )

        specs = {
            "DM": get_domain_spec("DM"),
            "VS": get_domain_spec("VS"),
        }
        multi = build_multi_table_metadata(specs)

        assert "DM" in multi.tables
        assert "VS" in multi.tables

    def test_missing_dm_raises(self) -> None:
        """ValueError when DM is not in specs."""
        from ptcv.mock_data.sdv_adapter import (
            build_multi_table_metadata,
        )

        with pytest.raises(ValueError, match="DM"):
            build_multi_table_metadata({
                "VS": get_domain_spec("VS"),
            })


# -----------------------------------------------------------------------
# Scenario: Import guard when SDV not installed
# -----------------------------------------------------------------------


class TestImportGuard:
    """Clear ImportError when SDV is not installed."""

    def test_build_sdv_metadata_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """build_sdv_metadata raises ImportError without SDV."""
        import ptcv.mock_data.sdv_adapter as adapter_mod

        # Remove the fake sdv from sys.modules.
        adapter_mod._SDV_AVAILABLE = None
        monkeypatch.delitem(sys.modules, "sdv", raising=False)
        monkeypatch.delitem(
            sys.modules, "sdv.metadata", raising=False,
        )

        with pytest.raises(ImportError, match="sdv"):
            adapter_mod.build_sdv_metadata(get_domain_spec("VS"))

    def test_build_multi_table_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """build_multi_table_metadata raises ImportError without SDV."""
        import ptcv.mock_data.sdv_adapter as adapter_mod

        adapter_mod._SDV_AVAILABLE = None
        monkeypatch.delitem(sys.modules, "sdv", raising=False)
        monkeypatch.delitem(
            sys.modules, "sdv.metadata", raising=False,
        )

        with pytest.raises(ImportError, match="pip install sdv"):
            adapter_mod.build_multi_table_metadata({
                "DM": get_domain_spec("DM"),
                "VS": get_domain_spec("VS"),
            })

    def test_error_message_has_install_instructions(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ImportError message includes install command."""
        import ptcv.mock_data.sdv_adapter as adapter_mod

        adapter_mod._SDV_AVAILABLE = None
        monkeypatch.delitem(sys.modules, "sdv", raising=False)
        monkeypatch.delitem(
            sys.modules, "sdv.metadata", raising=False,
        )

        with pytest.raises(ImportError) as exc_info:
            adapter_mod.build_sdv_metadata(get_domain_spec("VS"))

        msg = str(exc_info.value)
        assert "pip install sdv" in msg
        assert "PyTorch" in msg


# -----------------------------------------------------------------------
# Additional coverage: all 6 domains
# -----------------------------------------------------------------------


class TestAllDomains:
    """Every registered domain maps without error."""

    @pytest.mark.parametrize(
        "domain_code",
        ["DM", "VS", "LB", "AE", "EG", "CM"],
    )
    def test_domain_builds_metadata(
        self, domain_code: str,
    ) -> None:
        """Each domain produces metadata with correct columns."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        spec = get_domain_spec(domain_code)
        meta = build_sdv_metadata(spec)

        assert len(meta.columns) == len(spec.variables)

    @pytest.mark.parametrize(
        "domain_code",
        ["DM", "VS", "LB", "AE", "EG", "CM"],
    )
    def test_usubjid_always_id(self, domain_code: str) -> None:
        """USUBJID is sdtype=id in every domain."""
        from ptcv.mock_data.sdv_adapter import build_sdv_metadata

        meta = build_sdv_metadata(get_domain_spec(domain_code))
        assert meta.columns["USUBJID"]["sdtype"] == "id"


# -----------------------------------------------------------------------
# SdvAdapterConfig
# -----------------------------------------------------------------------


class TestSdvAdapterConfig:
    """SdvAdapterConfig defaults and usage."""

    def test_defaults(self) -> None:
        from ptcv.mock_data.sdv_adapter import SdvAdapterConfig

        cfg = SdvAdapterConfig()
        assert cfg.synthesizer_type == "gaussian_copula"
        assert cfg.enforce_codelists is True

    def test_ctgan_config(self) -> None:
        from ptcv.mock_data.sdv_adapter import SdvAdapterConfig

        cfg = SdvAdapterConfig(synthesizer_type="ctgan")
        assert cfg.synthesizer_type == "ctgan"

    def test_config_passed_to_builder(self) -> None:
        """Config is respected by build_sdv_metadata."""
        from ptcv.mock_data.sdv_adapter import (
            SdvAdapterConfig,
            build_sdv_metadata,
        )

        cfg = SdvAdapterConfig(enforce_codelists=False)
        meta = build_sdv_metadata(get_domain_spec("VS"), config=cfg)

        # No codelist regex should be present.
        for col_data in meta.columns.values():
            if col_data["sdtype"] == "categorical":
                assert "regex_format" not in col_data
