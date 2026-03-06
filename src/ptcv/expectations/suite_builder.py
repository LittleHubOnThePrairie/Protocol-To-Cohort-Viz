"""Great Expectations suite builder from SDTM variable metadata (PTCV-117).

Generates expectation descriptors from ``SdtmDomainSpec`` variable
metadata.  Each descriptor is a plain dataclass capturing the GX
expectation type and kwargs, enabling testing and inspection without
a ``great_expectations`` installation.

The optional ``build_domain_suite()`` function converts descriptors
into a GX ``ExpectationSuite`` — this requires ``great_expectations``
to be installed.

Risk tier: LOW — generates validation rules only; no patient data.

Regulatory references:
- SDTMIG v3.4 domain specifications
- CDISC Controlled Terminology 2023-12-15
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from ptcv.mock_data.sdtm_metadata import (
    SdtmDomainSpec,
    get_all_domain_specs,
    get_domain_spec,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExpectationDescriptor:
    """A GX expectation captured as a plain descriptor.

    Attributes:
        expectation_type: GX expectation class name
            (e.g. ``"expect_column_to_exist"``).
        kwargs: Keyword arguments for the expectation constructor.
    """

    expectation_type: str
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class SuiteConfig:
    """Configuration for suite generation.

    Attributes:
        include_column_existence: Generate ``expect_column_to_exist``.
        include_not_null: Generate
            ``expect_column_values_to_not_be_null``
            for Required variables.
        include_value_set: Generate
            ``expect_column_values_to_be_in_set``
            for codelist variables.
        include_value_range: Generate
            ``expect_column_values_to_be_between``
            for numeric-range variables.
        mostly: GX ``mostly`` parameter for nullable expectations
            (0.0-1.0, 1.0 = strict).
    """

    include_column_existence: bool = True
    include_not_null: bool = True
    include_value_set: bool = True
    include_value_range: bool = True
    mostly: float = 1.0


_DEFAULT_CONFIG = SuiteConfig()


# ---------------------------------------------------------------------------
# Core logic — no GX dependency
# ---------------------------------------------------------------------------


def build_domain_expectations(
    domain_code: str,
    config: Optional[SuiteConfig] = None,
    spec: Optional[SdtmDomainSpec] = None,
) -> list[ExpectationDescriptor]:
    """Build expectation descriptors for a single SDTM domain.

    Args:
        domain_code: Two-letter SDTM domain code (e.g. ``"VS"``).
        config: Optional suite configuration; uses defaults if ``None``.
        spec: Optional pre-fetched domain spec; looked up if ``None``.

    Returns:
        Ordered list of ``ExpectationDescriptor`` instances.
    """
    cfg = config or _DEFAULT_CONFIG
    domain = spec or get_domain_spec(domain_code)
    expectations: list[ExpectationDescriptor] = []

    for var in domain.variables:
        # Column existence
        if cfg.include_column_existence:
            expectations.append(ExpectationDescriptor(
                expectation_type="expect_column_to_exist",
                kwargs={"column": var.name},
            ))

        # Not-null for Required variables
        if cfg.include_not_null and var.core == "Required":
            expectations.append(ExpectationDescriptor(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": var.name, "mostly": cfg.mostly},
            ))

        # Value set for codelist variables
        if cfg.include_value_set and var.codelist is not None:
            expectations.append(ExpectationDescriptor(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": var.name,
                    "value_set": sorted(var.codelist),
                    "mostly": cfg.mostly,
                },
            ))

        # Numeric range
        if (cfg.include_value_range
                and var.range_min is not None
                and var.range_max is not None):
            expectations.append(ExpectationDescriptor(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": var.name,
                    "min_value": var.range_min,
                    "max_value": var.range_max,
                    "mostly": cfg.mostly,
                },
            ))

    return expectations


def build_all_expectation_sets(
    config: Optional[SuiteConfig] = None,
) -> dict[str, list[ExpectationDescriptor]]:
    """Build expectation descriptors for all registered SDTM domains.

    Args:
        config: Optional suite configuration; uses defaults if ``None``.

    Returns:
        Dictionary mapping domain code to list of descriptors.
    """
    return {
        code: build_domain_expectations(code, config=config, spec=spec)
        for code, spec in get_all_domain_specs().items()
    }


# ---------------------------------------------------------------------------
# GX conversion layer — requires great_expectations
# ---------------------------------------------------------------------------


def build_domain_suite(
    domain_code: str,
    config: Optional[SuiteConfig] = None,
    suite_name: Optional[str] = None,
) -> Any:
    """Build a GX ``ExpectationSuite`` for a single SDTM domain.

    Requires ``great_expectations`` to be installed.

    Args:
        domain_code: Two-letter SDTM domain code.
        config: Optional suite configuration.
        suite_name: Optional suite name; defaults to
            ``"sdtm_{domain_code}_suite"``.

    Returns:
        ``great_expectations.core.ExpectationSuite`` instance.

    Raises:
        ImportError: If ``great_expectations`` is not installed.
    """
    try:
        import great_expectations as gx  # type: ignore[import-untyped]
        from great_expectations.core import (  # type: ignore[import-untyped]
            ExpectationConfiguration,
            ExpectationSuite,
        )
    except ImportError as exc:
        raise ImportError(
            "great_expectations is required for build_domain_suite(). "
            "Install with: pip install great-expectations"
        ) from exc

    name = suite_name or f"sdtm_{domain_code.lower()}_suite"
    descriptors = build_domain_expectations(domain_code, config=config)

    suite = ExpectationSuite(expectation_suite_name=name)
    for desc in descriptors:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type=desc.expectation_type,
                kwargs=desc.kwargs,
            )
        )

    return suite
