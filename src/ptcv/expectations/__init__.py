"""Great Expectations integration for SDTM validation (PTCV-114).

Generates ExpectationSuites from SDTM domain metadata and validates
SDTM DataFrames against those suites.

.. note::

    ``great_expectations`` is an optional dependency.  The core logic
    (expectation descriptor generation) works without it.  The GX
    conversion layer requires ``pip install great-expectations``.
"""

from __future__ import annotations

from .checkpoint_runner import (
    DomainValidationResult,
    FailedExpectation,
    SdtmValidator,
    ValidationReport,
    validate_all,
    validate_domain,
)
from .suite_builder import (
    ExpectationDescriptor,
    SuiteConfig,
    build_all_expectation_sets,
    build_domain_expectations,
    build_domain_suite,
)

__all__ = [
    "DomainValidationResult",
    "ExpectationDescriptor",
    "FailedExpectation",
    "SdtmValidator",
    "SuiteConfig",
    "ValidationReport",
    "build_all_expectation_sets",
    "build_domain_expectations",
    "build_domain_suite",
    "validate_all",
    "validate_domain",
]
