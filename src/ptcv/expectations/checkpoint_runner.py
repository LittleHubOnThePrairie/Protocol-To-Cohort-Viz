"""Great Expectations checkpoint runner for SDTM validation (PTCV-120).

Validates SDTM DataFrames against ExpectationSuites and produces
structured validation reports.

Two validation paths are provided:

1. **Native path** (default) — evaluates ``ExpectationDescriptor``
   rules using pandas directly.  No ``great_expectations`` install
   required.
2. **GX path** — ``SdtmValidator`` wraps a full GX DataContext /
   Checkpoint workflow.  Requires ``pip install great-expectations``.

Risk tier: LOW — validation logic only; no patient data persisted.

Regulatory references:
- SDTMIG v3.4 domain specifications
- CDISC Controlled Terminology 2023-12-15
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Optional, Sequence

import pandas as pd

from ptcv.expectations.suite_builder import (
    ExpectationDescriptor,
    SuiteConfig,
    build_domain_expectations,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FailedExpectation:
    """Details of a single failed expectation check.

    Attributes:
        column: Column name the check targeted.
        expectation_type: GX expectation class name.
        observed_value: What was actually found.
        details: Human-readable failure description.
    """

    column: str
    expectation_type: str
    observed_value: Any
    details: str


@dataclasses.dataclass(frozen=True)
class DomainValidationResult:
    """Validation outcome for a single SDTM domain.

    Attributes:
        domain_code: Two-letter SDTM domain code.
        success: ``True`` if every expectation passed.
        total_expectations: Number of expectations evaluated.
        passed: Count of passing expectations.
        failed: Count of failing expectations.
        failed_expectations: Details per failed expectation.
    """

    domain_code: str
    success: bool
    total_expectations: int
    passed: int
    failed: int
    failed_expectations: tuple[FailedExpectation, ...]


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    """Aggregate validation report across multiple domains.

    Attributes:
        domain_results: Per-domain validation outcomes.
        overall_success: ``True`` if all domains passed.
        total_passed: Sum of passed expectations across domains.
        total_failed: Sum of failed expectations across domains.
        timestamp: ISO 8601 timestamp of the validation run.
    """

    domain_results: dict[str, DomainValidationResult]
    overall_success: bool
    total_passed: int
    total_failed: int
    timestamp: str


# ---------------------------------------------------------------------------
# Native expectation evaluator (no GX required)
# ---------------------------------------------------------------------------


def _check_expectation(
    desc: ExpectationDescriptor,
    df: pd.DataFrame,
) -> Optional[FailedExpectation]:
    """Evaluate one ``ExpectationDescriptor`` against a DataFrame.

    Returns ``None`` on success, or a ``FailedExpectation`` on failure.
    """
    etype = desc.expectation_type
    kwargs = desc.kwargs
    column: str = kwargs.get("column", "")
    mostly: float = kwargs.get("mostly", 1.0)

    if etype == "expect_column_to_exist":
        if column not in df.columns:
            return FailedExpectation(
                column=column,
                expectation_type=etype,
                observed_value=list(df.columns),
                details=f"Column '{column}' not found in DataFrame",
            )

    elif etype == "expect_column_values_to_not_be_null":
        if column not in df.columns:
            return FailedExpectation(
                column=column,
                expectation_type=etype,
                observed_value=None,
                details=f"Column '{column}' not found",
            )
        null_frac = df[column].isna().mean()
        if (1.0 - null_frac) < mostly:
            return FailedExpectation(
                column=column,
                expectation_type=etype,
                observed_value=float(null_frac),
                details=(
                    f"Column '{column}' has {null_frac:.1%} nulls "
                    f"(mostly={mostly})"
                ),
            )

    elif etype == "expect_column_values_to_be_in_set":
        if column not in df.columns:
            return FailedExpectation(
                column=column,
                expectation_type=etype,
                observed_value=None,
                details=f"Column '{column}' not found",
            )
        value_set = set(kwargs.get("value_set", []))
        non_null = df[column].dropna()
        if len(non_null) > 0:
            invalid = set(non_null.unique()) - value_set
            invalid_frac = non_null.isin(value_set).mean()
            if invalid_frac < mostly:
                return FailedExpectation(
                    column=column,
                    expectation_type=etype,
                    observed_value=sorted(str(v) for v in invalid),
                    details=(
                        f"Column '{column}' has values outside "
                        f"allowed set: {sorted(str(v) for v in invalid)}"
                    ),
                )

    elif etype == "expect_column_values_to_be_between":
        if column not in df.columns:
            return FailedExpectation(
                column=column,
                expectation_type=etype,
                observed_value=None,
                details=f"Column '{column}' not found",
            )
        min_val = kwargs.get("min_value")
        max_val = kwargs.get("max_value")
        non_null = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(non_null) > 0:
            in_range = (non_null >= min_val) & (non_null <= max_val)
            if in_range.mean() < mostly:
                out_vals = non_null[~in_range].tolist()
                return FailedExpectation(
                    column=column,
                    expectation_type=etype,
                    observed_value=out_vals,
                    details=(
                        f"Column '{column}' has values outside "
                        f"[{min_val}, {max_val}]: {out_vals}"
                    ),
                )

    return None


# ---------------------------------------------------------------------------
# Public validation API
# ---------------------------------------------------------------------------


def validate_domain(
    domain_code: str,
    df: pd.DataFrame,
    config: Optional[SuiteConfig] = None,
    expectations: Optional[Sequence[ExpectationDescriptor]] = None,
) -> DomainValidationResult:
    """Validate a DataFrame against SDTM domain expectations.

    Uses native pandas evaluation of ``ExpectationDescriptor`` rules.
    No ``great_expectations`` installation required.

    Args:
        domain_code: Two-letter SDTM domain code (e.g. ``"VS"``).
        df: DataFrame to validate.
        config: Optional suite configuration.
        expectations: Optional pre-built expectation list; built from
            the domain registry if ``None``.

    Returns:
        ``DomainValidationResult`` with pass/fail counts and details.
    """
    descs = (
        list(expectations)
        if expectations is not None
        else build_domain_expectations(domain_code, config=config)
    )

    failures: list[FailedExpectation] = []
    for desc in descs:
        result = _check_expectation(desc, df)
        if result is not None:
            failures.append(result)

    passed = len(descs) - len(failures)
    return DomainValidationResult(
        domain_code=domain_code.upper(),
        success=len(failures) == 0,
        total_expectations=len(descs),
        passed=passed,
        failed=len(failures),
        failed_expectations=tuple(failures),
    )


def validate_all(
    dataframes: dict[str, pd.DataFrame],
    config: Optional[SuiteConfig] = None,
) -> ValidationReport:
    """Validate multiple SDTM domain DataFrames.

    Args:
        dataframes: Dictionary mapping domain code to DataFrame.
        config: Optional suite configuration.

    Returns:
        ``ValidationReport`` with per-domain results and totals.
    """
    results: dict[str, DomainValidationResult] = {}
    for code, df in dataframes.items():
        results[code] = validate_domain(code, df, config=config)

    total_passed = sum(r.passed for r in results.values())
    total_failed = sum(r.failed for r in results.values())
    overall = all(r.success for r in results.values())

    return ValidationReport(
        domain_results=results,
        overall_success=overall,
        total_passed=total_passed,
        total_failed=total_failed,
        timestamp=datetime.datetime.now(
            datetime.timezone.utc,
        ).isoformat(),
    )


# ---------------------------------------------------------------------------
# GX-powered validator (requires great_expectations)
# ---------------------------------------------------------------------------


class SdtmValidator:
    """GX-powered SDTM DataFrame validator.

    Requires ``great_expectations`` to be installed.

    Args:
        suites: Dictionary mapping domain code to GX
            ``ExpectationSuite`` instances.

    Raises:
        ImportError: If ``great_expectations`` is not installed.
    """

    def __init__(self, suites: dict[str, Any]) -> None:
        try:
            import great_expectations  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "great_expectations is required for SdtmValidator. "
                "Install with: pip install great-expectations"
            ) from exc
        self._suites = suites

    def validate_domain(
        self,
        domain_code: str,
        df: pd.DataFrame,
    ) -> DomainValidationResult:
        """Validate one DataFrame via GX checkpoint.

        Args:
            domain_code: Two-letter SDTM domain code.
            df: DataFrame to validate.

        Returns:
            ``DomainValidationResult`` parsed from GX results.
        """
        import great_expectations as gx  # type: ignore[import-untyped]

        code = domain_code.upper()
        suite = self._suites.get(code)
        if suite is None:
            raise KeyError(
                f"No suite registered for domain '{code}'"
            )

        context = gx.get_context()
        ds = context.sources.add_pandas(name=f"{code}_ds")
        asset = ds.add_dataframe_asset(name=f"{code}_asset")
        batch = asset.build_batch_request(dataframe=df)

        result = context.run_checkpoint(
            checkpoint_name=f"{code}_checkpoint",
            validations=[{
                "batch_request": batch,
                "expectation_suite_name": suite.expectation_suite_name,
            }],
        )

        stats = result.statistics
        failures: list[FailedExpectation] = []
        for vr in result.list_validation_results():
            for er in vr.results:
                if not er.success:
                    col = er.expectation_config.kwargs.get(
                        "column", "",
                    )
                    failures.append(FailedExpectation(
                        column=col,
                        expectation_type=(
                            er.expectation_config.expectation_type
                        ),
                        observed_value=er.result.get(
                            "observed_value",
                        ),
                        details=str(er.result),
                    ))

        return DomainValidationResult(
            domain_code=code,
            success=result.success,
            total_expectations=stats["evaluated_expectations"],
            passed=stats["successful_expectations"],
            failed=stats["unsuccessful_expectations"],
            failed_expectations=tuple(failures),
        )

    def validate_all(
        self,
        dataframes: dict[str, pd.DataFrame],
    ) -> ValidationReport:
        """Validate all domain DataFrames via GX.

        Args:
            dataframes: Dictionary mapping domain code to DataFrame.

        Returns:
            ``ValidationReport`` with per-domain results.
        """
        results: dict[str, DomainValidationResult] = {}
        for code, df in dataframes.items():
            results[code] = self.validate_domain(code, df)

        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())

        return ValidationReport(
            domain_results=results,
            overall_success=all(r.success for r in results.values()),
            total_passed=total_passed,
            total_failed=total_failed,
            timestamp=datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat(),
        )
