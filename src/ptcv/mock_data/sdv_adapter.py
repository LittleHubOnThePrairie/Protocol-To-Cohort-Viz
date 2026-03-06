"""SDV metadata adapter for SDTM domain specs (PTCV-118).

Bridges the SDTM variable metadata registry to the Synthetic Data
Vault (SDV) metadata format, enabling SDV synthesizers to generate
structurally correct SDTM data.

Maps ``SdtmVariableSpec`` attributes to SDV column sdtypes:

- ``Char`` with codelist → ``"categorical"``
- ``Char`` without codelist (date-like suffix) → ``"datetime"``
- ``Char`` without codelist (other) → ``"categorical"``
- ``Num`` with range → ``"numerical"``
- ``Num`` without range → ``"numerical"``
- ``USUBJID`` → ``"id"``

SDV is an **optional** dependency (requires PyTorch).  All public
functions raise ``ImportError`` with install guidance when SDV is
not available.

Risk tier: LOW — metadata mapping only; no patient data.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from .sdtm_metadata import SdtmDomainSpec, SdtmVariableSpec

# SDV lazy-import guard — checked at function call time.
_SDV_AVAILABLE: bool | None = None


def _check_sdv() -> None:
    """Raise ``ImportError`` if SDV is not installed."""
    global _SDV_AVAILABLE  # noqa: PLW0603
    if _SDV_AVAILABLE is None:
        try:
            import sdv  # noqa: F401
            _SDV_AVAILABLE = True
        except ImportError:
            _SDV_AVAILABLE = False

    if not _SDV_AVAILABLE:
        raise ImportError(
            "The 'sdv' package is required for synthetic data "
            "generation but is not installed.  Install with:\n\n"
            "    pip install sdv\n\n"
            "Note: SDV requires PyTorch.  See "
            "https://docs.sdv.dev/sdv/getting-started/install "
            "for platform-specific instructions."
        )


# Date-suffix pattern for Char variables that represent dates
# (e.g. RFSTDTC, AESTDTC, VSDTC).
_DATE_SUFFIX_RE = re.compile(r"DTC$|DT$")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SdvAdapterConfig:
    """Configuration for SDV metadata adapter.

    Attributes:
        synthesizer_type: SDV synthesizer backend.
            ``"gaussian_copula"`` (fast, good for numeric) or
            ``"ctgan"`` (slower, better for categorical).
        enforce_codelists: When *True*, categorical columns derived
            from SDTM codelists have their allowed values set as
            SDV regex constraints.  When *False*, SDV learns the
            distribution from training data.
    """

    synthesizer_type: str = "gaussian_copula"
    enforce_codelists: bool = True


# ---------------------------------------------------------------------------
# Single-table metadata builder
# ---------------------------------------------------------------------------


def _classify_variable(
    var: SdtmVariableSpec,
) -> dict[str, Any]:
    """Return SDV column kwargs for a single variable.

    Returns:
        Dict with ``"sdtype"`` and optional constraint keys
        (``"regex_format"``, ``"computer_representation"``, etc.).
    """
    # USUBJID is always the row-level ID.
    if var.name == "USUBJID":
        return {"sdtype": "id", "regex_format": "[A-Z0-9-]{5,40}"}

    # Numeric variables.
    if var.type == "Num":
        meta: dict[str, Any] = {
            "sdtype": "numerical",
            "computer_representation": "Float",
        }
        return meta

    # Char variables with a date-like suffix → datetime.
    if _DATE_SUFFIX_RE.search(var.name):
        return {
            "sdtype": "datetime",
            "datetime_format": "%Y-%m-%dT%H:%M:%S",
        }

    # Char variables — categorical.
    return {"sdtype": "categorical"}


def build_sdv_metadata(
    domain_spec: SdtmDomainSpec,
    config: SdvAdapterConfig | None = None,
) -> Any:
    """Build ``SingleTableMetadata`` from an SDTM domain spec.

    Args:
        domain_spec: Variable-level SDTM domain specification.
        config: Optional adapter configuration.  Defaults to
            ``SdvAdapterConfig()`` (gaussian_copula, codelists on).

    Returns:
        ``sdv.metadata.SingleTableMetadata`` instance with columns
        for every variable in the domain.

    Raises:
        ImportError: If the ``sdv`` package is not installed.
    """
    _check_sdv()
    from sdv.metadata import SingleTableMetadata

    cfg = config or SdvAdapterConfig()
    metadata = SingleTableMetadata()

    for var in domain_spec.variables:
        col_kwargs = _classify_variable(var)

        # Apply codelist constraints when enabled.
        if (
            cfg.enforce_codelists
            and var.codelist
            and col_kwargs["sdtype"] == "categorical"
        ):
            # SDV uses regex_format for categorical constraints.
            # Build a regex that matches any codelist value.
            escaped = [re.escape(v) for v in sorted(var.codelist)]
            col_kwargs["regex_format"] = (
                "^(" + "|".join(escaped) + ")$"
            )

        metadata.add_column(var.name, **col_kwargs)

    # Set primary key to the sequence variable if present,
    # otherwise fall back to USUBJID.
    seq_var = next(
        (
            v.name
            for v in domain_spec.variables
            if v.name.endswith("SEQ")
        ),
        None,
    )
    if seq_var:
        # Sequence numbers are better primary keys for
        # observation-class domains.
        metadata.set_primary_key(seq_var)

    return metadata


# ---------------------------------------------------------------------------
# Multi-table metadata builder
# ---------------------------------------------------------------------------


def build_multi_table_metadata(
    specs: dict[str, SdtmDomainSpec],
    config: SdvAdapterConfig | None = None,
) -> Any:
    """Build ``MultiTableMetadata`` for cross-domain relationships.

    Creates per-domain table metadata and adds USUBJID foreign-key
    relationships from observation domains to DM (Demographics).

    Args:
        specs: Mapping of domain code to ``SdtmDomainSpec``.
            Must include ``"DM"`` as the parent table.
        config: Optional adapter configuration.

    Returns:
        ``sdv.metadata.MultiTableMetadata`` instance.

    Raises:
        ImportError: If the ``sdv`` package is not installed.
        ValueError: If ``"DM"`` is not in *specs*.
    """
    _check_sdv()
    from sdv.metadata import MultiTableMetadata

    if "DM" not in specs:
        raise ValueError(
            "Multi-table metadata requires DM (Demographics) "
            "as the parent table."
        )

    cfg = config or SdvAdapterConfig()
    multi = MultiTableMetadata()

    # Add each domain as a table.
    for domain_code, domain_spec in specs.items():
        table_meta = build_sdv_metadata(domain_spec, cfg)
        multi.add_table(domain_code, metadata=table_meta)

    # Add USUBJID relationships from child domains to DM.
    for domain_code in specs:
        if domain_code == "DM":
            continue

        # Only add relationship if the domain has USUBJID.
        has_usubjid = any(
            v.name == "USUBJID"
            for v in specs[domain_code].variables
        )
        if has_usubjid:
            multi.add_relationship(
                parent_table_name="DM",
                child_table_name=domain_code,
                parent_primary_key="USUBJID",
                child_foreign_key="USUBJID",
            )

    return multi
